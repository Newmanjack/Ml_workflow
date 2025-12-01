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
        return PipelineResult(df=df, spark_session=spark_session, validation=validation_results, stats=run_stats, drift=drift)


def main():
    parser = argparse.ArgumentParser(description="Run Smart ML Preprocessing pipeline.")
    parser.add_argument("--config", default="config/base_config.yaml", help="Path to YAML config.")
    args = parser.parse_args()

    cfg = load_config(args.config)
    runner = PipelineRunner(cfg)
    runner.run()


if __name__ == "__main__":
    main()
