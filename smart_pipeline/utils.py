from __future__ import annotations

import logging
from pathlib import Path
from typing import Optional


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
