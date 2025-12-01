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
