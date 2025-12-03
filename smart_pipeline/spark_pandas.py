from __future__ import annotations

"""
Thin wrapper around pandas API on Spark (pyspark.pandas) to make it easy to opt-in
to pandas-like syntax on Spark DataFrames without collecting them.
"""

def _require_ps():
    try:
        import pyspark.pandas as ps  # noqa: F401
        return ps
    except ImportError as exc:  # pragma: no cover - optional dependency
        raise ImportError("pyspark.pandas is required for Spark pandas API; install PySpark 3.2+ with pandas.") from exc


def to_pandas_api(spark_df):
    """
    Return a pandas-on-Spark view of a Spark DataFrame (no collect).
    """
    ps = _require_ps()
    return spark_df.pandas_api()


def shape(spark_df):
    """
    Return (rows, cols) for a Spark DataFrame by counting rows and columns.
    Note: count() triggers a Spark job.
    """
    return spark_df.count(), len(spark_df.columns)


# Convenience alias to the pyspark.pandas module itself
def ps():
    return _require_ps()
