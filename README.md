# Smart ML Preprocessing Pipeline

A modular, config-driven preprocessing layer that discovers schemas, aggregates transactional data in-SQL, validates integrity, and profiles aggregated outputs. Built to be lakehouse/DB friendly (DuckDB by default) and easy to extend.

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

## Fabric / notebook-friendly usage
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
