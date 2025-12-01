from __future__ import annotations

import logging
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd

from .config import FeatureEngineeringConfig

logger = logging.getLogger("smart_pipeline.feature_engineering")


def _date_part_features(df: pd.DataFrame) -> pd.DataFrame:
    idx = df.index
    out = pd.DataFrame(index=idx)
    out["dayofweek"] = idx.dayofweek
    out["weekofyear"] = idx.isocalendar().week.astype(int)
    out["month"] = idx.month
    out["quarter"] = idx.quarter
    out["is_month_start"] = idx.is_month_start.astype(int)
    out["is_month_end"] = idx.is_month_end.astype(int)
    return out


def _lag_features(series: pd.Series, lags: List[int]) -> Dict[str, pd.Series]:
    feats = {}
    for lag in lags:
        feats[f"{series.name}_lag{lag}"] = series.shift(lag)
    return feats


def _rolling_features(series: pd.Series, windows: List[int]) -> Dict[str, pd.Series]:
    feats = {}
    for w in windows:
        feats[f"{series.name}_rollmean{w}"] = series.rolling(window=w, min_periods=1).mean()
        feats[f"{series.name}_rollstd{w}"] = series.rolling(window=w, min_periods=1).std()
    return feats


def _pct_change_features(series: pd.Series, windows: List[int]) -> Dict[str, pd.Series]:
    feats = {}
    for w in windows:
        feats[f"{series.name}_pctchg{w}"] = series.pct_change(periods=w)
    return feats


def generate_time_features(
    df: pd.DataFrame,
    cfg: FeatureEngineeringConfig,
    target_columns: Optional[List[str]] = None,
) -> Tuple[pd.DataFrame, Dict[str, List[str]]]:
    """
    Generate lag/rolling/pct-change/date-part features for each numeric column in df.
    Returns the feature dataframe (aligned to original index) and a catalog of created columns.
    """
    if df is None or df.empty:
        logger.warning("Feature engineering skipped: empty dataframe.")
        return df, {}

    if not isinstance(df.index, pd.DatetimeIndex):
        raise ValueError("Feature engineering expects a DatetimeIndex on the aggregated dataframe.")

    feature_df = pd.DataFrame(index=df.index)
    catalog: Dict[str, List[str]] = {}

    numeric_cols = target_columns or list(df.select_dtypes(include=[np.number]).columns)
    if not numeric_cols:
        logger.warning("No numeric columns found for feature generation.")
        return df, {}

    for col in numeric_cols:
        series = df[col]
        created = []

        lag_feats = _lag_features(series, cfg.lag_periods)
        feature_df[list(lag_feats.keys())] = pd.DataFrame(lag_feats)
        created.extend(lag_feats.keys())

        roll_feats = _rolling_features(series, cfg.rolling_windows)
        feature_df[list(roll_feats.keys())] = pd.DataFrame(roll_feats)
        created.extend(roll_feats.keys())

        pct_feats = _pct_change_features(series, cfg.pct_change_windows)
        feature_df[list(pct_feats.keys())] = pd.DataFrame(pct_feats)
        created.extend(pct_feats.keys())

        catalog[col] = created

    if cfg.add_date_parts:
        dp = _date_part_features(df)
        feature_df[dp.columns] = dp

    if cfg.drop_na:
        feature_df = feature_df.dropna()

    return feature_df, catalog
