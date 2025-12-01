from __future__ import annotations

import logging
from typing import Optional, Sequence

import pandas as pd

logger = logging.getLogger("smart_pipeline.spark")


def spark_to_pandas(
    spark_df,
    limit: Optional[int] = None,
    sample_fraction: Optional[float] = None,
    columns: Optional[Sequence[str]] = None,
    use_arrow: bool = True,
    parquet_path: Optional[str] = None,
):
    """
    Safely convert a Spark DataFrame to pandas.

    - Reduces size by optional column projection, limit, or sampling.
    - Uses Arrow for speed by default.
    - For larger extracts, if `parquet_path` is provided, writes to Parquet then loads via pandas to avoid driver OOM.
    """
    try:
        import pyspark  # noqa: F401
    except ImportError as exc:  # pragma: no cover - optional dependency
        raise ImportError("pyspark is required for spark_to_pandas; install with extras `[spark]`.") from exc

    sdf = spark_df

    if columns:
        sdf = sdf.select(*columns)

    if sample_fraction is not None:
        sdf = sdf.sample(fraction=sample_fraction, seed=42)

    if limit is not None:
        sdf = sdf.limit(limit)

    if parquet_path:
        logger.info("Writing Spark DF to parquet at %s and loading via pandas.", parquet_path)
        sdf.write.mode("overwrite").parquet(parquet_path)
        return pd.read_parquet(parquet_path)

    if use_arrow:
        # Enable Arrow to speed up conversion
        sdf.sql_ctx.sparkSession.conf.set("spark.sql.execution.arrow.pyspark.enabled", "true")

    return sdf.toPandas()


def reduce_spark_df(spark_df, columns=None, filter_expr=None, sample_fraction=None, limit=None):
    """
    Apply projection/filter/sample/limit to a Spark DataFrame.
    """
    sdf = spark_df
    if columns:
        sdf = sdf.select(*columns)
    if filter_expr is not None:
        sdf = sdf.filter(filter_expr)
    if sample_fraction is not None:
        sdf = sdf.sample(fraction=sample_fraction, seed=42)
    if limit is not None:
        sdf = sdf.limit(limit)
    return sdf


def run_pipeline_on_spark(
    headers_sdf,
    lines_sdfs,
    config_dict: Optional[dict] = None,
    overrides: Optional[dict] = None,
    reduction: Optional[dict] = None,
    parquet_dir: Optional[str] = None,
):
    """
    Spark-first runner:
      1) Reduce Spark DataFrames (projection/filter/sample/limit).
      2) Land to Parquet.
      3) Spin up DuckDB over Parquet and run the pipeline end-to-end.

    Args:
        headers_sdf: Spark DataFrame for headers.
        lines_sdfs: Spark DataFrame or dict{name: Spark DataFrame} for line tables.
        config_dict: Optional config dict to build PipelineConfig.
        overrides: Optional overrides dict.
        reduction: Optional dict for reduction parameters. Supports:
            - global defaults: {"columns": [...], "filter": "...", "sample_fraction": 0.1, "limit": 100000}
            - per-table: reduction["headers"], reduction["lines"], reduction["<table_name>"]
        parquet_dir: Optional directory to write temp parquet files; default: temp dir.
    """
    import os
    import tempfile
    import duckdb

    from .fabric import build_config
    from .runner import PipelineRunner

    reduction = reduction or {}
    line_sdfs_map = lines_sdfs if isinstance(lines_sdfs, dict) else {"line_items": lines_sdfs}

    tmpdir = parquet_dir or tempfile.mkdtemp(prefix="spark_pipeline_")

    def _opts_for(name, is_header=False):
        if is_header and "headers" in reduction:
            return reduction["headers"]
        if name in reduction:
            return reduction[name]
        if not is_header and "lines" in reduction:
            return reduction["lines"]
        return reduction  # global defaults

    def _reduce_and_land(name, sdf, opts):
        cols = opts.get("columns")
        fexpr = opts.get("filter")
        sample = opts.get("sample_fraction")
        lim = opts.get("limit")
        path = os.path.join(tmpdir, f"{name}.parquet")
        reduced = reduce_spark_df(sdf, cols, fexpr, sample, lim)
        reduced.write.mode("overwrite").parquet(path)
        return path

    header_path = _reduce_and_land("headers", headers_sdf, _opts_for("headers", is_header=True))
    line_paths = {
        name: _reduce_and_land(name, sdf, _opts_for(name))
        for name, sdf in line_sdfs_map.items()
    }

    cfg = build_config(config_dict, overrides)
    # Ensure line_tables reflects the landed tables
    line_table_names = list(line_paths.keys())
    sources = cfg.sources.model_copy(update={"line_tables": line_table_names, "line_table": line_table_names[0]})
    cfg = cfg.model_copy(update={"sources": sources})

    con = duckdb.connect()
    con.execute(f"CREATE OR REPLACE VIEW {sources.header_table} AS SELECT * FROM read_parquet('{header_path}')")
    for name, path in line_paths.items():
        con.execute(f"CREATE OR REPLACE VIEW {name} AS SELECT * FROM read_parquet('{path}')")

    runner = PipelineRunner(cfg, connection=con)
    df, ctx, validation = runner.run()

    return df, ctx, validation, {"tmpdir": tmpdir, "header_path": header_path, "line_paths": line_paths}


def _is_spark_df(obj) -> bool:
    return hasattr(obj, "toPandas") and hasattr(obj, "sql_ctx")


def run_pipeline_auto(
    headers_df,
    lines_df,
    config_dict: Optional[dict] = None,
    overrides: Optional[dict] = None,
    limit_headers: int = 200_000,
    limit_lines: int = 1_000_000,
    sample_fraction: Optional[float] = None,
    columns_headers: Optional[Sequence[str]] = None,
    columns_lines: Optional[Sequence[str]] = None,
):
    """
    Convenience entry that:
      - Detects Spark vs pandas input.
      - If Spark: reduces (projection/sample/limit), converts to pandas safely.
      - Runs the pandas-based pipeline via run_pipeline_on_dfs.

    Use this when you just want to hand Spark DFs and get an aggregated pandas result with minimal code.
    """
    from .fabric import run_pipeline_on_dfs

    # Spark → pandas reduction
    if _is_spark_df(headers_df):
        headers_pdf = spark_to_pandas(
            headers_df,
            columns=columns_headers,
            sample_fraction=sample_fraction,
            limit=limit_headers,
        )
    else:
        headers_pdf = headers_df

    if _is_spark_df(lines_df):
        lines_pdf = spark_to_pandas(
            lines_df,
            columns=columns_lines,
            sample_fraction=sample_fraction,
            limit=limit_lines,
        )
    else:
        lines_pdf = lines_df

    return run_pipeline_on_dfs(
        header_df=headers_pdf,
        line_df=lines_pdf,
        config_dict=config_dict,
        overrides=overrides,
    )
