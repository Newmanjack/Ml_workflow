"""Smart ML preprocessing pipeline package."""

import logging

logger = logging.getLogger("smart_pipeline")

from .config import PipelineConfig, load_config
from .discovery import SmartDiscoveryEngine, run_smart_discovery
from .feature_engineering import generate_time_features
from .spark_utils import spark_to_pandas, run_pipeline_on_spark, run_pipeline_auto
from .stats import prune_low_variance, basic_stats
from .result import PipelineResult, export_pipeline_result
from .modeling import train_spark_model, prepare_spark_features
from .pyspark_ml import (
    run_full_spark_ml_pipeline,
    load_pipeline,
    suggest_joins,
    plan_joins,
    join_tables_with_plan,
    TableSourceConfig,
    ModelConfig,
    auto_join_and_train,
)
from .pandas_bridge import spark_to_pandas_safe, pandas_api_on_spark
from .spark_pandas import to_pandas_api, shape as spark_shape, ps as spark_pandas_module

# DuckDB/pandas helpers are optional; guard import so Spark-only envs can still import the package.
PipelineRunner = None
SmartValidator = None
run_smart_validation = None
run_smart_profiling = None
build_config = create_duckdb_with_tables = run_pipeline_on_dfs = save_results = None
_DUCKDB_AVAILABLE = False
try:
    import duckdb  # noqa: F401
    from .runner import PipelineRunner
    from .validation import SmartValidator, run_smart_validation
    from .profiling import run_smart_profiling
    from .fabric import (
        build_config,
        create_duckdb_with_tables,
        run_pipeline_on_dfs,
        save_results,
    )
    _DUCKDB_AVAILABLE = True
except Exception as exc:  # pragma: no cover - optional dependency path
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
    "plan_joins",
    "join_tables_with_plan",
    "TableSourceConfig",
    "ModelConfig",
    "auto_join_and_train",
    "spark_to_pandas_safe",
    "pandas_api_on_spark",
    "to_pandas_api",
    "spark_shape",
    "spark_pandas_module",
]

if _DUCKDB_AVAILABLE:
    __all__.extend(
        [
            "build_config",
            "create_duckdb_with_tables",
            "run_pipeline_on_dfs",
            "save_results",
        ]
    )
