### Install (Spark)
```python
%pip install --upgrade --no-cache-dir "smart-pipeline-preprocessing[spark]@ git+https://github.com/Newmanjack/Ml_workflow.git"
# Private repo: %pip install --upgrade --no-cache-dir "smart-pipeline-preprocessing[spark]@ git+https://${GH_TOKEN}@github.com/Newmanjack/Ml_workflow.git"
```

# Smart ML Preprocessing Pipeline

A modular, config-driven preprocessing layer that discovers schemas, aggregates transactional data in-SQL, validates integrity, and profiles aggregated outputs. Built to be lakehouse/DB friendly (DuckDB by default) and easy to extend.

```python
%pip install --upgrade --no-cache-dir "smart-pipeline-preprocessing[spark]@ git+https://github.com/Newmanjack/Ml_workflow.git"
# Private repo: %pip install --upgrade --no-cache-dir "smart-pipeline-preprocessing[spark]@ git+https://${GH_TOKEN}@github.com/Newmanjack/Ml_workflow.git"
```
If you need the legacy DuckDB/pandas helpers, also install `duckdb` (included in the base requirements). Spark-only users can ignore DuckDB imports; the package will still load.

### Install (Spark)
```python
%pip install "smart-pipeline-preprocessing[spark]@ git+https://github.com/Newmanjack/Ml_workflow.git"
# Private repo: %pip install "smart-pipeline-preprocessing[spark]@ git+https://${GH_TOKEN}@github.com/Newmanjack/Ml_workflow.git"
```
> Spark-first: For large datasets, use the Spark APIs (`run_full_spark_ml_pipeline`, `train_spark_model`, `run_pipeline_auto` with Spark DataFrames). The pandas/DuckDB helpers remain available but are secondary for small workloads or legacy use.
## Table of Contents
- [Quick start (target + joins)](#quick-start-target--joins)
- [Spark-first workflow & auto-join](#spark-first-workflow--auto-join)
- [Spark-only multi-table ML pipeline](#spark-only-multi-table-ml-pipeline-scales-to-many-tables)
- [Spark ML models](#spark-ml-models)
- [Project Layout](#project-layout)
- [Setup](#setup)
- [Running the pipeline](#running-the-pipeline)
- [Legacy pandas/DuckDB usage (optional)](#legacy-pandasduckdb-usage-optional)
- [Multiple line tables](#multiple-line-tables-eg-orders--returns--adjustments)
- [Automatic feature engineering](#automatic-feature-engineering-lagsrollingpct-change--date-parts)
- [Column policies](#column-policies-typenullcardinality-to-clean-inputs)
- [Export aggregated data + metadata](#export-aggregated-data--metadata)
- [Installation](#installation-gitpip)



## Quick start (target + joins)
```python
from smart_pipeline import run_pipeline_auto as auto

# df1 = headers (Spark or pandas), df2 = lines (Spark or pandas)
cfg = {
    "target": {"column": "Revenue", "auto_detect": True},
    "feature_engineering": {"enabled": True, "lag_periods": [1, 7], "rolling_windows": [7]},
}
overrides = {
    # join_key supports lists for composite joins
    "join_key": {"header": ["OrderID", "Company"], "line": ["OrderID", "Company"]},
    "header": {"date": "OrderDate", "amount": "TotalAmount"},
    "line": {"date": "LineDate", "amount": "LineAmount"},
}

result = auto(df1, df2, config_dict=cfg, overrides=overrides)
print("Target:", result.target_column)
print(result.df.head())
```

## Spark-first workflow & auto-join
```python
from smart_pipeline import suggest_joins, run_full_spark_ml_pipeline, PipelineConfig, TableSourceConfig, ModelConfig

# 1) Inspect suggested joins across many Spark tables
tables = ["orders", "line_items", "customers", "payments"]
dfs = {t: spark.read.table(t) for t in tables}
print(suggest_joins(dfs))  # {(orders, line_items): ['order_id'], ...}

# 2) Configure join keys (use suggestions or set explicitly)
join_cfg = {
    "orders": {"join_key": "order_id"},
    "line_items": {"join_key": "order_id"},
    "customers": {"join_key": "customer_id"},
    "payments": {"join_key": "order_id"},
}

# 3) Train Spark ML pipeline end-to-end (no pandas)
cfg = PipelineConfig(
    selected_tables=tables,
    table_source=TableSourceConfig(source_type="catalog"),  # or parquet/jdbc
    join_config=join_cfg,
    feature_columns=None,  # auto-detect numerics + encoded categoricals
    label={"table": "orders", "column": "Revenue"},
    model=ModelConfig(
        problem_type="regression",
        model_type="random_forest",
        label_column="Revenue",
        apply_scaling=False,
        train_fraction=0.8,
        metrics=["rmse", "mae", "r2"],
    ),
    output_path="ml_pipeline_spark",
)

model, meta = run_full_spark_ml_pipeline(spark, cfg)
print(meta)
```

## Project Layout
- `config/base_config.yaml` — sample pipeline configuration
- `smart_pipeline/` — core package
  - `config.py` — typed config (Pydantic) + loader
  - `discovery.py` — schema discovery + aggregation strategy selection/execution
  - `validation.py` — DB-side quality checks and reconciliation
  - `profiling.py` — lightweight profiling for aggregated data
  - `runner.py` — orchestration entry point / CLI
  - `data_models.py`, `utils.py` — shared models/helpers
- `tests/` — synthetic integration/unit tests for discovery & validation
- `logs/` — run logs and persisted run metadata (JSON)

## Setup
```
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```
Optional: install `ydata-profiling` if you want rich HTML profiling reports.

## Running the pipeline
```
python -m smart_pipeline.runner --config config/base_config.yaml
```
What happens:
1) Connect to the database (DuckDB by default)
2) Discover dates/amounts/join keys; pick an aggregation strategy
3) Aggregate to a time series and validate joins + sums in-SQL
4) Profile aggregated output (lightweight by default)
5) Persist run metadata under `logs/` as `<run_id>.json`

### Configuration highlights
- `connection`: engine/database target (DuckDB by default)
- `sources`: header/line table names
- `overrides`: optional forced columns for date/amount/join keys
- `discovery`: strategy preference + minimum confidence threshold
- `validation`/`profiling`: enable/disable + profiling sample size
- `metadata`: where to write run logs/metadata

## Tests
```
pytest
```
Synthetic DuckDB tables cover discovery correctness, join analysis, and reconciliation (including a deliberate mismatch).

## Legacy pandas/DuckDB usage (optional)
You can drop the `smart_pipeline` folder into a notebook environment and run end-to-end without YAML:

```python
from smart_pipeline import run_pipeline_on_dfs
import pandas as pd

# Replace with your own dataframes
header_df = pd.DataFrame([...])
line_df = pd.DataFrame([...])

# Optional overrides for tricky schemas
overrides = {
    "header": {"date": "OrderDate", "amount": "TotalAmount"},
    "line": {"date": "LineDate", "amount": "LineAmount"},
    "join_key": {"header": "OrderID", "line": "OrderID"},
}

df, context, validation = run_pipeline_on_dfs(
    header_df,
    line_df,
    config_dict={"profiling": {"enabled": False}},  # skip plots if headless
    overrides=overrides,
)
display(df.head())
print(context)  # discovery context
for check in validation:
    print(check)
```

If Fabric only accepts a single file, you can still upload the whole folder (preferred). Otherwise, create a lightweight wrapper script that imports `run_pipeline_on_dfs` from this package.

### Multiple line tables (e.g., orders + returns + adjustments)
- Add `line_tables` under `sources` in config, or pass via `config_dict`:
```python
cfg = {
    "sources": {
        "header_table": "headers",
        "line_table": "line_items",  # default
        "line_tables": ["line_items", "line_items_returns", "line_items_adjustments"],
    },
    "profiling": {"enabled": False},  # skip plots in headless
}
df, context, validation = run_pipeline_on_dfs(header_df, line_df, config_dict=cfg)
```
- Each line table is processed separately; results are stitched on the Date index with suffixed columns (e.g., `TotalAmount_line_items`, `TotalAmount_line_items_returns`). Validation runs per line table and is merged.

### Automatic feature engineering (lags/rolling/pct-change + date parts)
- Enable via config: `feature_engineering.enabled: true`
- Defaults: lags `[1,7,28]`, rolling windows `[7,28]`, pct-change `[7]`, date parts (dow/week/month/quarter/start/end flags)
- Features are generated on all numeric aggregated columns and appended to the returned dataframe; catalog is stored in the run metadata.
- In notebooks, pass via `config_dict={"feature_engineering": {"enabled": True}}` when calling `run_pipeline_on_dfs`.
- Optional pruning: set `feature_engineering.prune_low_variance` to drop constant/low-variance numeric columns before feature gen.
- After running, you can export results/metadata: `from smart_pipeline import export_pipeline_result; export_pipeline_result(result, output_dir="logs")`.
- Target column: set `target.column` (or rely on auto-detect over candidate names/numeric columns); available on `PipelineResult.target_column` and in exported metadata.

## Quick-start examples

### Minimal (auto) — works with Spark or pandas inputs
```python
from smart_pipeline import run_pipeline_auto as auto

result = auto(headers_df=df1, lines_df=df2)  # df1/df2 can be Spark or pandas
df = result.df                       # aggregated + features
print("Target:", result.target_column)
for v in result.validation:
    print(v)
```

### With target, overrides, and feature flags
```python
cfg = {
    "profiling": {"enabled": False},
    "feature_engineering": {"enabled": True, "lag_periods": [1,7], "rolling_windows": [7]},
    "target": {"column": "Revenue", "auto_detect": True},
}
ovr = {
    # join_key supports lists for composite joins
    "join_key": {"header": ["OrderID", "Company"], "line": ["OrderID", "Company"]},
    "header": {"date": "OrderDate", "amount": "TotalAmount"},
    "line": {"date": "LineDate", "amount": "LineAmount"},
}
result = auto(df1, df2, config_dict=cfg, overrides=ovr)
```

### Multiple line tables (Spark or pandas)
```python
cfg = {
    "sources": {
        "header_table": "headers",
        "line_table": "line_items",
        "line_tables": ["line_items", "line_items_returns"],
    },
    "profiling": {"enabled": False},
    "feature_engineering": {"enabled": True},
}
ovr = {
    "join_key": {"header": "OrderID", "line": "OrderID"},
    "per_line": {"line_items_returns": {"amount": "ReturnAmount"}},
}
result = auto(df1, df2, config_dict=cfg, overrides=ovr)
```

### Column policies (type/null/cardinality) to clean inputs
```python
cfg = {
    "sources": {
        "column_policies": {
            "OrderDate": {"expected_type": "date", "null_policy": "drop"},
            "TotalAmount": {"expected_type": "float", "null_policy": "fill", "fill_value": 0},
            "CustomerCode": {"drop_high_cardinality": True, "cardinality_threshold": 5000},
        }
    },
    "feature_engineering": {"enabled": True, "prune_low_variance": 0.0},
}
result = auto(df1, df2, config_dict=cfg)
```

### Export aggregated data + metadata
```python
from smart_pipeline import export_pipeline_result
paths = export_pipeline_result(result, output_dir="logs")
print(paths)
```

### (Optional) Train a simple Spark ML model on the aggregated output
```python
from smart_pipeline import train_spark_model
# Assume you have the aggregated pandas df from result; to stay in Spark, load it to Spark:
# sdf = spark.createDataFrame(result.df.reset_index())

model, test_df, metrics = train_spark_model(
    sdf,                     # Spark DataFrame
    target_col=result.target_column or "TotalAmount",
    feature_cols=None,       # auto-select numeric features
    model_type="regression", # or "classification"
    test_fraction=0.2,
    handle_missing="zero",
    scale=False,
)
print(metrics)
```

### Join planning (joinable vs unjoinable)
```python
from smart_pipeline import plan_joins
plan = plan_joins(dfs, semantic_relations=None, join_map={("orders","line_items"): {"from_col": "order_id", "to_col": "order_id"}})
print("Joinable:", plan['joinable'])
print("Unjoinable:", plan['unjoinable'])
```
## Spark-only multi-table ML pipeline (scales to many tables)
```python
from smart_pipeline import run_full_spark_ml_pipeline, PipelineConfig, TableSourceConfig, ModelConfig

cfg = PipelineConfig(
    selected_tables=["orders", "line_items", "customers", "payments", "shipments"],  # add as many as needed
    table_source=TableSourceConfig(source_type="catalog"),  # or parquet/jdbc with base_path/jdbc_url
    join_config={
        "orders": {"join_key": "order_id"},
        "line_items": {"join_key": "order_id"},
        "payments": {"join_key": "order_id"},
        "customers": {"join_key": "customer_id"},
        "shipments": {"join_key": "order_id"},
    },
    feature_columns=None,  # auto-detect numerics + encoded categoricals across joined tables
    label={"table": "orders", "column": "Revenue"},
    model=ModelConfig(
        problem_type="regression",
        model_type="random_forest",
        label_column="Revenue",
        apply_scaling=False,
        train_fraction=0.8,
        metrics=["rmse", "mae"],
    ),
    output_path="ml_pipeline_spark",
)

model, meta = run_full_spark_ml_pipeline(spark, cfg)
print(meta)  # metrics + config + features used
```

## Next steps
- Wire in real lakehouse/DB connection details via config
- Extend validation with business rules or additional anomaly checks
- Add CI (GitHub Actions) to run tests + linting on commits

## Installation (Git/pip)
- Public repo: `pip install git+https://github.com/Newmanjack/Ml_workflow.git`
- Private repo: use a PAT with `repo` scope, stored as an env var/secret:
  `pip install git+https://${GH_TOKEN}@github.com/Newmanjack/Ml_workflow.git`
- Optional extras:
  - `pip install "smart-pipeline-preprocessing[fabric]@ git+https://github.com/Newmanjack/Ml_workflow.git"` (adds `deltalake`)
  - `pip install "smart-pipeline-preprocessing[profiling]@ git+https://github.com/Newmanjack/Ml_workflow.git"` (adds `ydata-profiling`)
  - `pip install "smart-pipeline-preprocessing[spark]@ git+https://github.com/Newmanjack/Ml_workflow.git"` (adds `pyspark`)

## Using with Spark
- Keep heavy joins/filters/aggregations in Spark; only move reduced data to pandas.
- Helpers:
  - `spark_to_pandas`: projection/filter/sample/limit + optional Parquet round-trip, then to pandas
  - `run_pipeline_on_spark`: Spark-first entry that lands reduced Spark DataFrames to Parquet, spins up DuckDB on top, and runs the pipeline end-to-end.
   - `run_pipeline_auto`: minimal entry point; detects Spark vs pandas, reduces (limit/sample) if Spark, then runs the pipeline.
  ```python
  # Simplest Spark path (auto-reduce, then pipeline)
  from smart_pipeline import run_pipeline_auto as auto
  df, spark_session, validation = auto(
      headers_df=headers_spark_df,
      lines_df=lines_spark_df,
      config_dict={"profiling": {"enabled": False}, "feature_engineering": {"enabled": True}},
      overrides=None,              # let discovery auto-pick first
      limit_headers=200_000,
      limit_lines=1_000_000,
      sample_fraction=None,        # set e.g. 0.1 to sample
      columns_headers=None,        # or pass a projection list
      columns_lines=None,
  )
  # df is aggregated (and features if enabled); heavy lifting stayed in Spark reduction.
  ```
