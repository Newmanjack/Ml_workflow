from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Dict, List
import pandas as pd


@dataclass
class PipelineResult:
    df: pd.DataFrame
    spark_session: Any
    validation: List[Any]
    stats: Dict[str, Any] = field(default_factory=dict)
    drift: Dict[str, Any] = field(default_factory=dict)

    def summary(self) -> Dict[str, Any]:
        return {
            "rows": len(self.df) if self.df is not None else 0,
            "columns": list(self.df.columns) if self.df is not None else [],
            "validation": [getattr(v, "status", str(v)) for v in self.validation],
            "drift": self.drift,
        }
