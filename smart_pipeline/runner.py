from __future__ import annotations

import argparse
import json
import logging
from dataclasses import asdict
from pathlib import Path
from typing import Optional

import duckdb

from .config import PipelineConfig, load_config
from .discovery import run_smart_discovery
from .profiling import run_smart_profiling
from .feature_engineering import generate_time_features
from .stats import basic_stats, prune_low_variance, detect_outliers_iqr, detect_gaps_datetime_index
from .result import PipelineResult
import glob
from .utils import ensure_dir, maybe_sample_df, setup_logging
from .validation import run_smart_validation


def _create_connection(cfg) -> duckdb.DuckDBPyConnection:
    if cfg.engine.lower() != "duckdb":
        raise ValueError(f"Unsupported engine: {cfg.engine}")
    return duckdb.connect(cfg.database, **cfg.extra)


class PipelineRunner:
    """Orchestrates discovery → validation → profiling."""

    def __init__(
        self,
        config: PipelineConfig,
        logger: Optional[logging.Logger] = None,
        connection: Optional[duckdb.DuckDBPyConnection] = None,
    ):
        self.config = config
        self.logger = logger or setup_logging(config.logging.log_dir, config.logging.level)
        self.con = connection or _create_connection(config.connection)

    def _line_tables(self):
        if self.config.sources.line_tables:
            return self.config.sources.line_tables
        return [self.config.sources.line_table]

    def run(self):
        cfg = self.config
        run_id = cfg.run_id
        self.logger.info("Starting pipeline run_id=%s", run_id)

        aggregated_outputs = []
        spark_sessions = []
        validations = []
        feature_catalog = {}
        run_stats = {}

        for lt in self._line_tables():
            self.logger.info("Processing line table: %s", lt)
            df, spark_session = run_smart_discovery(
                self.con, cfg.sources, cfg.overrides, cfg.discovery, current_line_table=lt
            )
            if cfg.validation.enabled:
                validation_results = run_smart_validation(self.con, cfg.sources, spark_session)
            else:
                validation_results = []

            aggregated_outputs.append({"line_table": lt, "df": df})
            spark_sessions.append(spark_session)
            validations.append(validation_results)

            if cfg.profiling.enabled:
                sampled = maybe_sample_df(df, cfg.profiling.sample_size)
                run_smart_profiling(sampled, {"columns": spark_session.strategy_columns}, f"Aggregated Data ({lt})")

        # If multiple line tables, stitch by date with suffixes
        if len(aggregated_outputs) > 1:
            combined = None
            for item in aggregated_outputs:
                df = item["df"].rename(columns={"TotalAmount": f"TotalAmount_{item['line_table']}"})
                combined = df if combined is None else combined.join(df, how="outer")
            df = combined.sort_index()
            spark_session = spark_sessions[0]
            validation_results = [v for sub in validations for v in sub]
        else:
            df = aggregated_outputs[0]["df"]
            spark_session = spark_sessions[0]
            validation_results = validations[0]

        # Optional pruning before feature eng
        dropped_cols = []
        variances = {}
        if cfg.feature_engineering.enabled:
            if cfg.feature_engineering.prune_low_variance is not None:
                thresh = cfg.feature_engineering.prune_low_variance
                df, dropped_cols, variances = prune_low_variance(df, threshold=thresh)

            self.logger.info("Generating time-series features")
            fe_df, feature_catalog = generate_time_features(
                df, cfg.feature_engineering, target_columns=list(df.select_dtypes(include=["number"]).columns)
            )
            df = df.join(fe_df)
            spark_session.feature_catalog = feature_catalog
            run_stats["post_features"] = basic_stats(df)
            run_stats["pruned_low_variance"] = {"dropped": dropped_cols, "variances": variances}
            # outlier/gap diagnostics
            if not df.empty and isinstance(df.index, pd.DatetimeIndex):
                primary_num = df.select_dtypes(include=["number"]).columns[0] if len(df.select_dtypes(include=["number"]).columns) else None
                if primary_num:
                    run_stats["outliers"] = detect_outliers_iqr(df[primary_num])
                run_stats["gaps"] = detect_gaps_datetime_index(df)

        if cfg.metadata.persist_context:
            output_dir = ensure_dir(cfg.metadata.output_dir)
            payload = {
                "run_id": run_id,
                "strategy": spark_session.selected_strategy,
                "confidence": spark_session.selected_confidence,
                "context": asdict(spark_session),
                "validation": [asdict(v) for v in validation_results],
                "row_count": len(df) if df is not None else 0,
                "line_tables": self._line_tables(),
                "feature_engineering": {
                    "enabled": cfg.feature_engineering.enabled,
                    "feature_columns": [c for cols in feature_catalog.values() for c in cols],
                },
                "stats": run_stats,
            }
            out_path = Path(output_dir) / f"{run_id}.json"
            out_path.write_text(json.dumps(payload, indent=2), encoding="utf-8")
            self.logger.info("Wrote run metadata to %s", out_path)

        drift = {}
        # Drift: compare to latest prior run stats if enabled
        if cfg.drift.enabled:
            prior_stats = self._load_latest_stats(cfg.metadata.output_dir, exclude_run_id=run_id)
            drift = self._compute_drift(run_stats.get("post_features", {}), prior_stats, cfg.drift)

        target_col = self._select_target(df, cfg.target)

        return PipelineResult(
            df=df,
            spark_session=spark_session,
            validation=validation_results,
            stats=run_stats,
            drift=drift,
            target_column=target_col,
        )

    def _load_latest_stats(self, output_dir: str, exclude_run_id: str):
        paths = sorted(glob.glob(str(Path(output_dir) / "*.json")), reverse=True)
        for p in paths:
            if exclude_run_id in p:
                continue
            try:
                import json
                with open(p, "r", encoding="utf-8") as f:
                    data = json.load(f)
                stats = data.get("stats", {})
                if stats:
                    return stats.get("post_features", {})
            except Exception:
                continue
        return {}

    def _compute_drift(self, current_stats, prior_stats, drift_cfg):
        if not current_stats or not prior_stats:
            return {}
        drift_report = {}
        for col, cur in current_stats.items():
            if col not in prior_stats:
                continue
            prev = prior_stats[col]
            cur_mean, prev_mean = cur.get("mean"), prev.get("mean")
            cur_std, prev_std = cur.get("std"), prev.get("std")
            mean_pct = abs((cur_mean - prev_mean) / prev_mean * 100) if prev_mean else None
            std_pct = abs((cur_std - prev_std) / prev_std * 100) if prev_std else None
            alerts = []
            if mean_pct is not None and mean_pct > drift_cfg.mean_pct_threshold:
                alerts.append(f"mean drift {mean_pct:.1f}%")
            if std_pct is not None and std_pct > drift_cfg.std_pct_threshold:
                alerts.append(f"std drift {std_pct:.1f}%")
            if alerts:
                drift_report[col] = {"mean_pct": mean_pct, "std_pct": std_pct, "alerts": alerts}
        return drift_report

    def _select_target(self, df, target_cfg):
        if df is None or df.empty:
            return None
        if target_cfg.column and target_cfg.column in df.columns:
            return target_cfg.column
        if target_cfg.auto_detect:
            # Prefer configured candidate list, numeric first
            numeric_cols = list(df.select_dtypes(include=["number"]).columns)
            for cand in target_cfg.candidates:
                for col in df.columns:
                    if col.lower() == cand.lower():
                        return col
            if numeric_cols:
                return numeric_cols[0]
        return None


def main():
    parser = argparse.ArgumentParser(description="Run Smart ML Preprocessing pipeline.")
    parser.add_argument("--config", default="config/base_config.yaml", help="Path to YAML config.")
    args = parser.parse_args()

    cfg = load_config(args.config)
    runner = PipelineRunner(cfg)
    runner.run()


if __name__ == "__main__":
    main()
