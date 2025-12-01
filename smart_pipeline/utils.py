from __future__ import annotations

import logging
from pathlib import Path
from typing import Optional
import decimal
import pandas as pd


def ensure_dir(path: str | Path) -> Path:
    target = Path(path)
    target.mkdir(parents=True, exist_ok=True)
    return target


def setup_logging(log_dir: str, level: str = "INFO") -> logging.Logger:
    """Configure root logger with file + console handlers."""
    ensure_dir(log_dir)
    log_path = Path(log_dir) / "smart_pipeline.log"

    logger = logging.getLogger("smart_pipeline")
    logger.setLevel(getattr(logging, level.upper(), logging.INFO))

    # Avoid duplicate handlers if re-running in notebooks
    if not logger.handlers:
        fmt = logging.Formatter("%(asctime)s [%(levelname)s] %(name)s - %(message)s")

        fh = logging.FileHandler(log_path, encoding="utf-8")
        fh.setFormatter(fmt)
        logger.addHandler(fh)

        ch = logging.StreamHandler()
        ch.setFormatter(fmt)
        logger.addHandler(ch)

    return logger


def maybe_sample_df(df, sample_size: Optional[int] = None):
    if sample_size is None or df is None:
        return df
    if len(df) <= sample_size:
        return df
    return df.sample(sample_size, random_state=42)


def sanitize_dataframe(df: pd.DataFrame) -> pd.DataFrame:
    """Coerce decimal.Decimal/object numerics to float to avoid DuckDB cast issues."""
    if df is None or df.empty:
        return df

    def _needs_decimal_to_float(col: pd.Series) -> bool:
        if not pd.api.types.is_object_dtype(col):
            return False
        # Check a small sample for Decimal instances
        return col.dropna().map(lambda x: isinstance(x, decimal.Decimal)).any()

    out = df.copy()
    for col in out.columns:
        series = out[col]
        if _needs_decimal_to_float(series):
            out[col] = series.map(lambda x: float(x) if isinstance(x, decimal.Decimal) else x)

    return out


def apply_column_policies(df: pd.DataFrame, policies: Optional[dict]) -> pd.DataFrame:
    """
    Apply per-column policies:
      - expected_type: "int","float","str","date"
      - null_policy: "drop","fill","keep"
      - fill_value: value used when null_policy="fill"
      - date_format: strptime format when expected_type="date"
      - drop_high_cardinality: bool and cardinality_threshold
    """
    if not policies or df is None or df.empty:
        return df

    out = df.copy()
    for col, policy in policies.items():
        if col not in out.columns:
            continue
        p = policy or {}
        expected_type = p.get("expected_type")
        null_policy = p.get("null_policy", "keep")
        fill_value = p.get("fill_value")
        date_format = p.get("date_format")
        drop_high_card = p.get("drop_high_cardinality", False)
        card_threshold = p.get("cardinality_threshold", 1000)

        if expected_type == "date":
            out[col] = pd.to_datetime(out[col], errors="coerce", format=date_format)
        elif expected_type == "int":
            out[col] = pd.to_numeric(out[col], errors="coerce").astype("Int64")
        elif expected_type == "float":
            out[col] = pd.to_numeric(out[col], errors="coerce")
        elif expected_type == "str":
            out[col] = out[col].astype(str)

        if null_policy == "drop":
            out = out.dropna(subset=[col])
        elif null_policy == "fill":
            out[col] = out[col].fillna(fill_value)

        if drop_high_card:
            if out[col].nunique(dropna=True) > card_threshold:
                out = out.drop(columns=[col])

    return out
