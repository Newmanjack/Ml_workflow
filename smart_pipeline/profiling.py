from __future__ import annotations

import logging
from typing import Dict, Optional

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

logger = logging.getLogger("smart_pipeline.profiling")

try:
    from IPython.display import display
except ImportError:  # pragma: no cover - fallback
    def display(obj):
        print(obj)


def run_smart_profiling(df: pd.DataFrame, context: Optional[Dict] = None, df_name: str = "Time Series Data") -> None:
    """Profile aggregated data; avoids loading raw data."""
    context = context or {}

    if df is None:
        logger.warning("Profiling skipped: no DataFrame provided.")
        return

    if df.empty:
        logger.warning("Profiling skipped: DataFrame is empty.")
        return

    if "TotalAmount" in df.columns:
        expected_col = "TotalAmount"
    else:
        expected_col = context.get("columns", {}).get("amount") or context.get("columns", {}).get("line_amount")

    if not expected_col:
        num_cols = df.select_dtypes(include=[np.number]).columns
        expected_col = num_cols[0] if len(num_cols) > 0 else None

    logger.info("Profiling dataset '%s' using column '%s'", df_name, expected_col)

    try:
        from ydata_profiling import ProfileReport  # pragma: no cover - optional dependency

        profile = ProfileReport(df, title=f"Profiling Report: {df_name}", explorative=True, minimal=True)
        try:
            display(profile.to_widgets())
        except Exception:
            display(profile.to_notebook_iframe())
        return
    except ImportError:
        logger.info("ydata_profiling not installed; using lightweight profiling.")

    desc = df.describe(percentiles=[0.01, 0.05, 0.25, 0.5, 0.75, 0.95, 0.99])
    logger.info("Summary statistics:\n%s", desc)

    if isinstance(df.index, pd.DatetimeIndex):
        date_min, date_max = df.index.min(), df.index.max()
        full_range = pd.date_range(start=date_min, end=date_max, freq="D")
        missing_dates = full_range.difference(df.index)
        completeness = 100 * (1 - len(missing_dates) / len(full_range))
        logger.info(
            "Time range %s to %s | completeness %.2f%% (%s observed / %s days)",
            date_min.date(),
            date_max.date(),
            completeness,
            len(df),
            len(full_range),
        )
        if len(missing_dates) > 0:
            logger.info("Missing first 5 days: %s", [d.date() for d in missing_dates[:5]])

    if expected_col and expected_col in df.columns:
        plt.figure(figsize=(12, 4))
        plt.plot(df.index, df[expected_col], label="Observed", alpha=0.7)
        rolling_mean = df[expected_col].rolling(window=7, center=True).mean()
        plt.plot(rolling_mean.index, rolling_mean, label="7-Day Trend", color="orange", linewidth=2)
        plt.title(f"Time Series Overview: {expected_col}")
        plt.legend()
        plt.grid(True, alpha=0.3)
        plt.tight_layout()
        plt.show()

        plt.figure(figsize=(10, 4))
        df[expected_col].hist(bins=50, alpha=0.7, edgecolor="black")
        plt.title(f"Distribution of {expected_col}")
        plt.xlabel("Value")
        plt.ylabel("Frequency")
        plt.tight_layout()
        plt.show()
    else:
        logger.warning("Column '%s' not found for visualization.", expected_col)
