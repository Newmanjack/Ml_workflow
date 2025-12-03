from __future__ import annotations

"""
Helpers to interact with PySpark DataFrames using pandas/numpy approaches safely.

These utilities keep heavy work in Spark and only collect bounded subsets to the driver.
"""

import warnings
from typing import Optional, Sequence

import pandas as pd


def spark_to_pandas_safe(
    spark_df,
    columns: Optional[Sequence[str]] = None,
    limit: Optional[int] = None,
    sample_fraction: Optional[float] = None,
    use_arrow: bool = True,
    warn_on_truncate: bool = True,
):
    """
    Convert a Spark DataFrame to pandas safely:
      - optional column projection
      - optional sampling or limit to avoid OOM
      - Arrow-enabled conversion for speed
    """
    try:
        import pyspark  # noqa: F401
    except ImportError as exc:  # pragma: no cover - optional dep
        raise ImportError("pyspark is required for spark_to_pandas_safe.") from exc

    sdf = spark_df
    if columns:
        sdf = sdf.select(*columns)
    if sample_fraction is not None:
        sdf = sdf.sample(fraction=sample_fraction, seed=42)
    if limit is not None:
        sdf = sdf.limit(limit)
    if use_arrow:
        sdf.sql_ctx.sparkSession.conf.set("spark.sql.execution.arrow.pyspark.enabled", "true")

    pdf = sdf.toPandas()
    if warn_on_truncate and limit is not None:
        warnings.warn(f"Collected at most {limit} rows to pandas; original Spark DF may be larger.")
    return pdf


def pandas_api_on_spark(spark_df):
    """
    Get a pandas-on-Spark view to use pandas-like syntax without collecting data.
    """
    try:
        return spark_df.pandas_api()
    except AttributeError as exc:  # pragma: no cover
        raise RuntimeError("pandas API on Spark not available; requires PySpark 3.2+") from exc
