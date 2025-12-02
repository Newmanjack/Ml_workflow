"""Smart ML preprocessing pipeline package."""

import logging

logger = logging.getLogger("smart_pipeline")

from .config import PipelineConfig, load_config
from .discovery import SmartDiscoveryEngine, run_smart_discovery
from .validation import SmartValidator, run_smart_validation
from .profiling import run_smart_profiling
from .feature_engineering import generate_time_features
from .spark_utils import spark_to_pandas, run_pipeline_on_spark, run_pipeline_auto
from .stats import prune_low_variance, basic_stats
from .result import PipelineResult, export_pipeline_result
from .modeling import train_spark_model, prepare_spark_features
from .pyspark_ml import (
    run_full_spark_ml_pipeline,
    load_pipeline,
    suggest_joins,
    TableSourceConfig,
    ModelConfig,
)

# DuckDB/pandas helpers are optional; guard import so Spark-only envs can still import the package.
try:
    from .runner import PipelineRunner
    from .fabric import (
        build_config,
        create_duckdb_with_tables,
        run_pipeline_on_dfs,
        save_results,
    )
    _DUCKDB_AVAILABLE = True
except Exception as exc:  # pragma: no cover - optional dependency path
    PipelineRunner = None
    build_config = create_duckdb_with_tables = run_pipeline_on_dfs = save_results = None
    _DUCKDB_AVAILABLE = False
    logger.warning(
        "DuckDB-related helpers not available (missing dependency?): %s. Spark APIs remain usable.",
        exc,
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
    "prune_low_variance",
    "basic_stats",
    "PipelineResult",
    "export_pipeline_result",
    "train_spark_model",
    "prepare_spark_features",
    "run_full_spark_ml_pipeline",
    "load_pipeline",
    "suggest_joins",
    "TableSourceConfig",
    "ModelConfig",
    "build_config",
    "create_duckdb_with_tables",
    "run_pipeline_on_dfs",
    "save_results",
]
