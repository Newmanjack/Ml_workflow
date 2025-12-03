# Smart ML Preprocessing & Spark ML Pipeline (Spark-first)

A Spark-first toolkit for joining many tables, discovering schemas, aggregating, validating, feature engineering, and training Spark ML models. Legacy pandas/DuckDB helpers remain optional for small data.

## Table of Contents
- [Install (Spark)](#install-spark)
- [Quick Start](#quick-start)
- [Join Planning (joinable vs unjoinable)](#join-planning-joinable-vs-unjoinable)
- [Spark ML Pipeline](#spark-ml-pipeline)
- [Spark ML Models](#spark-ml-models)
- [Legacy pandas/DuckDB (optional)](#legacy-pandasduckdb-optional)
- [Project Layout](#project-layout)
- [Tests](#tests)

## Install (Spark)
```python
%pip install --upgrade --no-cache-dir "smart-pipeline-preprocessing[spark]@ git+https://github.com/Newmanjack/Ml_workflow.git"
# Private repo: %pip install --upgrade --no-cache-dir "smart-pipeline-preprocessing[spark]@ git+https://${GH_TOKEN}@github.com/Newmanjack/Ml_workflow.git"
```

## Quick Start
```python
from smart_pipeline import plan_joins, suggest_joins, run_full_spark_ml_pipeline, PipelineConfig, TableSourceConfig, ModelConfig

# 1) Load your Spark tables
tables = ["orders", "line_items", "customers", "payments"]
dfs = {t: spark.read.table(t) for t in tables}

# 2) Plan joins (semantic relations or join_map optional)
join_map = { (rel["from_table"], rel["to_table"]): (rel["from_col"], rel["to_col"]) for _, rel in relationships.iterrows() }
plan = plan_joins(dfs, semantic_relations=None, join_map=join_map)
print("Joinable:", plan["joinable"])
print("Unjoinable:", plan["unjoinable"])

# 3) Configure and train Spark ML end-to-end
cfg = PipelineConfig(
    selected_tables=tables,
    table_source=TableSourceConfig(source_type="catalog"),
    join_config={tbl: {"join_key": "order_id"} for tbl in ["orders", "line_items", "payments"]},
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

## Join Planning (joinable vs unjoinable)
```python
from smart_pipeline import plan_joins
join_map = { (rel["from_table"], rel["to_table"]): (rel["from_col"], rel["to_col"]) for _, rel in relationships.iterrows() }
plan = plan_joins(dfs, semantic_relations=None, join_map=join_map)
print("Joinable:", plan["joinable"])
print("Unjoinable:", plan["unjoinable"])
```

## Inspect joined data before training
```python
from smart_pipeline import plan_joins, join_tables_with_plan
# plan joins
join_map = { (rel['from_table'], rel['to_table']): (rel['from_col'], rel['to_col']) for _, rel in relationships.iterrows() }
plan = plan_joins(dfs, semantic_relations=None, join_map=join_map)
# build a joined Spark DF for EDA
joined = join_tables_with_plan(dfs, plan['joinable'], base_table='orders', how='inner')
joined.show(5)
joined.printSchema()
```
## Spark ML Pipeline
- `run_full_spark_ml_pipeline`: load/join tables, auto-detect features, preprocess (encode/scale), train, evaluate, and save pipeline + metadata.
- `auto_join_and_train`: given Spark DataFrames, label table/column, join_map/semantic hints → join and train in one step.
- `suggest_joins`: heuristic join-key suggestions (semantic relations → join_map → common columns → frequency fallback).

## Spark ML Models
- **Linear/Logistic Regression**: elastic net (`reg_param`, `elastic_net_param`, `max_iter`); scaling recommended.
- **Decision Tree**: `max_depth`.
- **Random Forest**: `num_trees`, `max_depth`.
- **Gradient-Boosted Trees (GBT)**: `max_depth`, `max_iter`, `step_size`.
- **Linear SVM (LinearSVC)**: `svm_reg_param`, `svm_max_iter`; scaling recommended.
- Extras: `apply_scaling`, `class_weight`, `polynomial_degree`, categorical encoding (StringIndexer+OHE), metrics (rmse/mae/mse/r2, auc/auprc/accuracy/f1/logloss).

## Legacy pandas/DuckDB (optional)
Pandas/DuckDB runners remain for small data/notebook use: `run_pipeline_on_dfs`, `run_pipeline_auto` (accepts pandas or Spark via reduction), and `export_pipeline_result`. DuckDB imports are optional/guarded; Spark-only users can ignore.

## Project Layout
- `smart_pipeline/pyspark_ml.py` — Spark ML loading/joining/training APIs
- `smart_pipeline/spark_utils.py` — Spark helpers
- `smart_pipeline/spark_pandas.py` — pandas-on-Spark helpers
- `smart_pipeline/` legacy modules (`runner.py`, etc.) — optional pandas/DuckDB path
- `tests/` — synthetic tests
- `config/base_config.yaml` — sample config

## Tests
```bash
pytest
```
