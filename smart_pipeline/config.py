from __future__ import annotations

from pathlib import Path
from typing import Any, Dict, Optional

import yaml
from pydantic import BaseModel, Field, field_validator


class LoggingConfig(BaseModel):
    level: str = "INFO"
    log_dir: str = "logs"


class ConnectionConfig(BaseModel):
    engine: str = "duckdb"
    database: str = ":memory:"
    extra: Dict[str, Any] = Field(default_factory=dict)


class TableConfig(BaseModel):
    header_table: str = "headers"
    line_table: str = "line_items"
    line_tables: list[str] | None = None  # optional multi-line support
    reduction: Dict[str, Any] | None = None  # optional Spark reduction rules
    column_policies: Dict[str, Any] | None = None  # optional per-column policies


class ColumnOverride(BaseModel):
    date: Optional[str] = None
    amount: Optional[str] = None


class JoinKeyOverride(BaseModel):
    header: Optional[str] = None
    line: Optional[str] = None


class OverridesConfig(BaseModel):
    header: ColumnOverride = ColumnOverride()
    line: ColumnOverride = ColumnOverride()
    join_key: JoinKeyOverride = JoinKeyOverride()
    per_line: Dict[str, ColumnOverride] = Field(default_factory=dict)  # table -> overrides


class DiscoveryConfig(BaseModel):
    strategy_preference: list[str] = Field(
        default_factory=lambda: ["header_only", "line_join_header", "line_direct"]
    )
    min_confidence: float = 0.0

    @field_validator("min_confidence")
    def _min_confidence_range(cls, v: float) -> float:
        if v < 0.0 or v > 100.0:
            raise ValueError("min_confidence must be between 0 and 100")
        return v


class ProfilingConfig(BaseModel):
    enabled: bool = True
    sample_size: int = 5000


class FeatureEngineeringConfig(BaseModel):
    enabled: bool = False
    lag_periods: list[int] = Field(default_factory=lambda: [1, 7, 28])
    rolling_windows: list[int] = Field(default_factory=lambda: [7, 28])
    pct_change_windows: list[int] = Field(default_factory=lambda: [7])
    add_date_parts: bool = True
    drop_na: bool = False  # keep NaNs from lags by default
    prune_low_variance: float | None = None  # drop numeric columns with variance <= threshold


class ValidationConfig(BaseModel):
    enabled: bool = True


class DriftConfig(BaseModel):
    enabled: bool = False
    mean_pct_threshold: float = 10.0  # percent change
    std_pct_threshold: float = 20.0   # percent change
    null_pct_threshold: float = 5.0   # percent change
    distinct_pct_threshold: float = 10.0


class MetadataConfig(BaseModel):
    output_dir: str = "logs"
    persist_context: bool = True


class PipelineConfig(BaseModel):
    run_id_prefix: str = "dev"
    logging: LoggingConfig = LoggingConfig()
    connection: ConnectionConfig = ConnectionConfig()
    sources: TableConfig = TableConfig()
    overrides: OverridesConfig = OverridesConfig()
    discovery: DiscoveryConfig = DiscoveryConfig()
    profiling: ProfilingConfig = ProfilingConfig()
    feature_engineering: FeatureEngineeringConfig = FeatureEngineeringConfig()
    validation: ValidationConfig = ValidationConfig()
    drift: DriftConfig = DriftConfig()
    metadata: MetadataConfig = MetadataConfig()

    @property
    def run_id(self) -> str:
        import datetime as _dt

        ts = _dt.datetime.now(_dt.timezone.utc).strftime("%Y%m%dT%H%M%SZ")
        return f"{self.run_id_prefix}-{ts}"


def load_config(path: str | Path) -> PipelineConfig:
    """Load pipeline configuration from a YAML file."""
    cfg_path = Path(path)
    if not cfg_path.exists():
        raise FileNotFoundError(f"Config file not found: {cfg_path}")

    with cfg_path.open("r", encoding="utf-8") as f:
        raw = yaml.safe_load(f) or {}

    return PipelineConfig.model_validate(raw)
