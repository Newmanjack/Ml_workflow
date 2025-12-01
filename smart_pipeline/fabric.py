from __future__ import annotations

from dataclasses import asdict
from pathlib import Path
from typing import Any, Dict, Optional, Tuple

import duckdb
import pandas as pd

from .config import OverridesConfig, PipelineConfig, TableConfig
from .runner import PipelineRunner
from .utils import ensure_dir, setup_logging


def _merge_overrides(base: OverridesConfig, override_dict: Optional[Dict[str, Dict[str, str]]]) -> OverridesConfig:
    if not override_dict:
        return base

    merged = base.model_dump()
    for section, overrides in override_dict.items():
        if section in merged and isinstance(overrides, dict):
            merged[section] = {**merged.get(section, {}), **{k: v for k, v in overrides.items() if v is not None}}
    return OverridesConfig.parse_obj(merged)


def build_config(config_dict: Optional[Dict[str, Any]] = None, overrides: Optional[Dict[str, Dict[str, str]]] = None) -> PipelineConfig:
    """
    Create a PipelineConfig from a dict (or defaults) and optional overrides.
    Useful in notebooks where you want to avoid managing YAML files.
    """
    cfg = PipelineConfig.model_validate(config_dict or {})
    cfg = cfg.model_copy(update={"overrides": _merge_overrides(cfg.overrides, overrides)})
    return cfg


def create_duckdb_with_tables(
    header_df: pd.DataFrame,
    line_df: pd.DataFrame,
    tables: Optional[TableConfig] = None,
    database: str = ":memory:",
) -> duckdb.DuckDBPyConnection:
    """
    Bootstrap an in-memory DuckDB, register pandas DataFrames, and create header/line tables.
    Designed for quick notebook runs in Fabric or local dev.
    """
    tbl_cfg = tables or TableConfig()
    con = duckdb.connect(database)
    con.register("header_df", header_df)
    con.register("line_df", line_df)
    con.execute(f"CREATE TABLE {tbl_cfg.header_table} AS SELECT * FROM header_df")
    con.execute(f"CREATE TABLE {tbl_cfg.line_table} AS SELECT * FROM line_df")
    con.unregister("header_df")
    con.unregister("line_df")
    return con


def run_pipeline_on_dfs(
    header_df: pd.DataFrame,
    line_df: pd.DataFrame,
    config_dict: Optional[Dict[str, Any]] = None,
    overrides: Optional[Dict[str, Dict[str, str]]] = None,
) -> Tuple[pd.DataFrame, Dict[str, Any], list]:
    """
    Convenience entry point for notebooks:
      - Build config from dict + overrides
      - Spin up in-memory DuckDB and create tables
      - Run discovery → validation → profiling
    Returns (aggregated_df, discovery_context_dict, validation_results)
    """
    effective_cfg = config_dict or {"profiling": {"enabled": False}}
    cfg = build_config(effective_cfg, overrides)
    logger = setup_logging(cfg.logging.log_dir, cfg.logging.level)
    con = create_duckdb_with_tables(header_df, line_df, cfg.sources)

    runner = PipelineRunner(cfg, logger=logger, connection=con)
    df, spark_session, validation_results = runner.run()

    context_dict = asdict(spark_session)
    return df, context_dict, validation_results


def save_results(df: pd.DataFrame, context: Dict[str, Any], validation_results: list, output_dir: str = "logs") -> Path:
    """
    Write aggregated data and metadata to disk; handy when Fabric allows file output.
    """
    out_dir = ensure_dir(output_dir)
    data_path = out_dir / "aggregated.parquet"
    meta_path = out_dir / "context.json"

    df.reset_index().to_parquet(data_path, index=False)
    import json

    meta_payload = {
        "context": context,
        "validation": [asdict(v) for v in validation_results],
    }
    meta_path.write_text(json.dumps(meta_payload, indent=2), encoding="utf-8")
    return data_path
