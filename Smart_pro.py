import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import logging

# Try to import display for IPython environments
try:
    from IPython.display import display
except ImportError:
    def display(obj):
        print(obj)

def run_smart_profiling(df: pd.DataFrame, context: dict = None, df_name: str = "Time Series Data"):
    """
    Executes robust profiling on the provided dataframe.
    Tries ydata_profiling first, falls back to custom analysis.
    """
    context = context or {}

    # Determine the column to analyze (Amount/Metric)
    # Priority 1: Standardized 'TotalAmount' (from SmartDiscovery)
    if 'TotalAmount' in df.columns:
        expected_col = 'TotalAmount'
    else:
        # Priority 2: Context-defined original column
        expected_col = context.get("columns", {}).get("amount") or context.get("columns", {}).get("line_amount")

    # If not in context, guess the first numeric column
    if not expected_col and not df.empty:
        num_cols = df.select_dtypes(include=[np.number]).columns
        if len(num_cols) > 0:
            expected_col = num_cols[0]

    print(f"--- Profiling: {df_name} ---")

    # 1. Try ydata_profiling
    try:
        from ydata_profiling import ProfileReport
        print("🚀 ydata_profiling library detected. Generating comprehensive report...")
        # Minimal configuration for speed on larger datasets
        profile = ProfileReport(df, title=f"Profiling Report: {df_name}", explorative=True, minimal=True)
        try:
            display(profile.to_widgets())
        except:
            # Fallback for environments that don't support widgets well
            print("ℹ️ Displaying as IFrame...")
            display(profile.to_notebook_iframe())
        return # Done if successful
    except ImportError:
        print("ℹ️ ydata_profiling not found. Using robust custom profiling...")

    # 2. Custom Profiling
    if df.empty:
        print("⚠️ DataFrame is empty.")
        return

    # Statistical Summary
    desc = df.describe(percentiles=[0.01, 0.05, 0.25, 0.5, 0.75, 0.95, 0.99])

    # Gap Analysis (if DatetimeIndex)
    if isinstance(df.index, pd.DatetimeIndex):
        date_min, date_max = df.index.min(), df.index.max()
        full_range = pd.date_range(start=date_min, end=date_max, freq='D')
        missing_dates = full_range.difference(df.index)
        completeness = 100 * (1 - len(missing_dates) / len(full_range))

        print(f"\n📊 Dataset Health:")
        print(f"- Time Range: {date_min.date()} to {date_max.date()}")
        print(f"- Completeness: {completeness:.2f}% ({len(df)} observed / {len(full_range)} expected days)")
        if len(missing_dates) > 0:
            print(f"- Missing Days: {len(missing_dates)} (First 5: {[d.date() for d in missing_dates[:5]]})")

    print(f"\n📈 Statistical Summary:")
    display(desc)

    # Visual Inspection
    if expected_col and expected_col in df.columns:
        plt.figure(figsize=(15, 5))
        plt.plot(df.index, df[expected_col], label='Observed', alpha=0.7)

        # Trend line (Rolling 7-day)
        rolling_mean = df[expected_col].rolling(window=7, center=True).mean()
        plt.plot(rolling_mean.index, rolling_mean, label='7-Day Trend', color='orange', linewidth=2)

        plt.title(f"Time Series Overview: {expected_col}")
        plt.legend()
        plt.grid(True, alpha=0.3)
        plt.show()

        # Distribution
        plt.figure(figsize=(10, 4))
        df[expected_col].hist(bins=50, alpha=0.7, edgecolor='black')
        plt.title(f"Distribution of {expected_col}")
        plt.xlabel("Value")
        plt.ylabel("Frequency")
        plt.show()
    else:
        print(f"⚠️ Column '{expected_col}' not found for visualization.")