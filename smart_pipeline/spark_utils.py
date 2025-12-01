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
