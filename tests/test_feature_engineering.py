import pandas as pd

from smart_pipeline.feature_engineering import generate_time_features
from smart_pipeline.config import FeatureEngineeringConfig


def test_generate_time_features_adds_lags_and_rolls():
    idx = pd.date_range("2023-01-01", periods=5, freq="D")
    df = pd.DataFrame({"TotalAmount": [10, 20, 30, 40, 50]}, index=idx)

    cfg = FeatureEngineeringConfig(
        enabled=True,
        lag_periods=[1],
        rolling_windows=[2],
        pct_change_windows=[1],
        add_date_parts=False,
        drop_na=False,
    )

    feats, catalog = generate_time_features(df, cfg)

    assert "TotalAmount_lag1" in feats.columns
    assert "TotalAmount_rollmean2" in feats.columns
    assert "TotalAmount_pctchg1" in feats.columns
    assert catalog["TotalAmount"]
    # Check a specific value for correctness
    assert feats.loc["2023-01-02", "TotalAmount_lag1"] == 10
