"""Smart ML preprocessing pipeline package."""

from .runner import PipelineRunner
from .config import PipelineConfig, load_config
from .discovery import SmartDiscoveryEngine, run_smart_discovery
from .validation import SmartValidator, run_smart_validation
from .profiling import run_smart_profiling
from .feature_engineering import generate_time_features
from .spark_utils import spark_to_pandas, run_pipeline_on_spark, run_pipeline_auto
from .fabric import (
    build_config,
    create_duckdb_with_tables,
    run_pipeline_on_dfs,
    save_results,
)

__all__ = [
    "PipelineRunner",
    "PipelineConfig",
    "load_config",
    "SmartDiscoveryEngine",
    "run_smart_discovery",
    "SmartValidator",
    "run_smart_validation",
    "run_smart_profiling",
    "generate_time_features",
    "spark_to_pandas",
    "run_pipeline_on_spark",
    "run_pipeline_auto",
    "build_config",
    "create_duckdb_with_tables",
    "run_pipeline_on_dfs",
    "save_results",
]
