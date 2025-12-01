from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Dict, List
import pandas as pd
import json
from pathlib import Path


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


def export_pipeline_result(result: PipelineResult, output_dir: str = "logs", parquet_name: str = "aggregated.parquet", meta_name: str = "run_meta.json") -> Dict[str, Path]:
    """Persist aggregated dataframe and metadata to disk."""
    from .utils import ensure_dir

    out_dir = ensure_dir(output_dir)
    df_path = out_dir / parquet_name
    meta_path = out_dir / meta_name

    if result.df is not None:
        result.df.reset_index().to_parquet(df_path, index=False)

    meta_payload = {
        "spark_session": getattr(result.spark_session, "__dict__", {}),
        "validation": [getattr(v, "__dict__", str(v)) for v in result.validation],
        "stats": result.stats,
        "drift": result.drift,
    }
    meta_path.write_text(json.dumps(meta_payload, indent=2), encoding="utf-8")
    return {"data": df_path, "meta": meta_path}
