from __future__ import annotations

import pandas as pd
import numpy as np


def basic_stats(df: pd.DataFrame, numeric_cols=None):
    """Compute basic stats for numeric columns."""
    if df is None or df.empty:
        return {}
    numeric_cols = numeric_cols or list(df.select_dtypes(include=[np.number]).columns)
    stats = {}
    for col in numeric_cols:
        s = df[col].dropna()
        if s.empty:
            continue
        stats[col] = {
            "count": int(s.count()),
            "mean": float(s.mean()),
            "std": float(s.std()),
            "min": float(s.min()),
            "max": float(s.max()),
            "null_rate": float(1 - len(s) / len(df)) if len(df) else 0.0,
            "distinct": int(df[col].nunique(dropna=True)),
        }
    return stats


def prune_low_variance(df: pd.DataFrame, threshold: float = 0.0):
    """
    Drop numeric columns with zero variance (or below threshold).
    Returns pruned df and list of dropped columns.
    """
    numeric = df.select_dtypes(include=[np.number])
    variances = numeric.var()
    drop_cols = [col for col, v in variances.items() if v <= threshold]
    return df.drop(columns=drop_cols), drop_cols, variances.to_dict()


def detect_outliers_iqr(series: pd.Series, factor: float = 1.5):
    if series.empty:
        return {"low": None, "high": None, "outlier_fraction": 0}
    q1 = series.quantile(0.25)
    q3 = series.quantile(0.75)
    iqr = q3 - q1
    low = q1 - factor * iqr
    high = q3 + factor * iqr
    mask = (series < low) | (series > high)
    return {
        "low": float(low),
        "high": float(high),
        "outlier_fraction": float(mask.mean()),
    }


def detect_gaps_datetime_index(df: pd.DataFrame, freq: str = "D"):
    if df is None or df.empty or not isinstance(df.index, pd.DatetimeIndex):
        return {"missing_count": 0, "completeness": 100.0}
    full_range = pd.date_range(start=df.index.min(), end=df.index.max(), freq=freq)
    missing = full_range.difference(df.index)
    completeness = 100 * (1 - len(missing) / len(full_range))
    return {
        "missing_count": int(len(missing)),
        "completeness": float(completeness),
        "first_missing": missing[0].isoformat() if len(missing) > 0 else None,
    }
