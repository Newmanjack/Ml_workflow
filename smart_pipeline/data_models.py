from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional


@dataclass
class ColumnMetadata:
    name: str
    dtype: str
    normalized: str
    original_field: Any


@dataclass
class DiscoveryContext:
    header_date: Optional[str] = None
    header_amount: Optional[str] = None
    line_date: Optional[str] = None
    line_amount: Optional[str] = None
    join_keys: Dict[str, Optional[str]] = field(default_factory=dict)
    overrides_used: bool = False
    selected_strategy: Optional[str] = None
    selected_confidence: float = 0.0
    strategy_columns: Dict[str, str] = field(default_factory=dict)
    feature_catalog: Dict[str, List[str]] = field(default_factory=dict)


@dataclass
class AggregationStrategy:
    name: str
    label: str
    sql_query: str
    confidence_score: float
    required_columns: Dict[str, str]


@dataclass
class ValidationResult:
    check_name: str
    status: str  # PASS|FAIL|WARN|ERROR
    message: str
    details: Dict[str, Any] = field(default_factory=dict)
