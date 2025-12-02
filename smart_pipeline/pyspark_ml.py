from __future__ import annotations

"""
PySpark ML pipeline builder with table loading, joining, preprocessing, and model training.

This module stays in Spark for large datasets and builds a feature + model pipeline with minimal
driver-side work. It includes math-aware options (scaling for linear models, encoding for categoricals)
and persists the full pipeline + metadata for reproducibility.
"""

import json
from dataclasses import dataclass, asdict
from typing import Dict, List, Optional, Tuple


def _require_pyspark():
    try:
        import pyspark  # noqa: F401
    except ImportError as exc:  # pragma: no cover
        raise ImportError("pyspark is required for pyspark_ml; install with extras `[spark]`.") from exc


@dataclass
class TableSourceConfig:
    source_type: str = "catalog"  # catalog | parquet | jdbc
    base_path: Optional[str] = None  # for parquet
    jdbc_url: Optional[str] = None
    jdbc_properties: Optional[Dict[str, str]] = None


@dataclass
class ModelConfig:
    feature_columns: Optional[List[str]] = None  # if None, auto-detect numerics + encoded categoricals
    label_column: str = "label"
    problem_type: str = "regression"  # or classification
    model_type: str = "linear_regression"  # linear_regression | logistic_regression | random_forest | gbt
    apply_scaling: bool = True
    train_fraction: float = 0.8
    random_seed: int = 42
    metrics: Optional[List[str]] = None  # e.g., ["rmse","mae"] or ["auc","accuracy"]


@dataclass
class PipelineConfig:
    selected_tables: List[str]
    table_source: TableSourceConfig
    join_config: Optional[Dict[str, Dict[str, str]]] = None  # table -> {"join_key": "..."}
    feature_columns: Optional[Dict[str, List[str]]] = None  # table -> cols
    label: Dict[str, str] = None  # {"table": "...", "column": "..."}
    model: ModelConfig = ModelConfig()
    output_path: str = "ml_pipeline"


def load_table(spark, table_name: str, source_cfg: TableSourceConfig):
    if source_cfg.source_type == "catalog":
        return spark.read.table(table_name)
    if source_cfg.source_type == "parquet":
        if not source_cfg.base_path:
            raise ValueError("base_path required for parquet source.")
        return spark.read.parquet(f"{source_cfg.base_path.rstrip('/')}/{table_name}")
    if source_cfg.source_type == "jdbc":
        if not source_cfg.jdbc_url:
            raise ValueError("jdbc_url required for jdbc source.")
        props = source_cfg.jdbc_properties or {}
        return spark.read.jdbc(url=source_cfg.jdbc_url, table=table_name, properties=props)
    raise ValueError(f"Unsupported source_type: {source_cfg.source_type}")


def _infer_join_key(dfs: Dict[str, object]) -> str:
    # heuristic: look for common columns ending with "id"
    common_cols = set.intersection(*[set(df.columns) for df in dfs.values()])
    id_like = [c for c in common_cols if c.lower().endswith("id")]
    if id_like:
        return id_like[0]
    if common_cols:
        return list(common_cols)[0]
    raise ValueError("Could not infer join key; please provide join_config.")


def join_tables(dfs: Dict[str, object], join_config: Optional[Dict[str, Dict[str, str]]] = None):
    tables = list(dfs.keys())
    if not tables:
        raise ValueError("No tables provided for join.")
    base = tables[0]
    joined = dfs[base]
    for tbl in tables[1:]:
        if join_config and tbl in join_config and base in join_config:
            base_key = join_config[base]["join_key"]
            tbl_key = join_config[tbl]["join_key"]
        elif join_config and tbl in join_config:
            base_key = _infer_join_key({base: joined, tbl: dfs[tbl]})
            tbl_key = join_config[tbl]["join_key"]
        elif join_config and base in join_config:
            base_key = join_config[base]["join_key"]
            tbl_key = _infer_join_key({base: joined, tbl: dfs[tbl]})
        else:
            base_key = tbl_key = _infer_join_key({base: joined, tbl: dfs[tbl]})
        joined = joined.join(dfs[tbl], on=joined[base_key] == dfs[tbl][tbl_key], how="inner")
    return joined


def suggest_joins(dfs: Dict[str, object]) -> Dict[Tuple[str, str], List[str]]:
    """
    Suggest join keys between all pairs of tables based on common column names.
    Returns a mapping: (tableA, tableB) -> [candidate_cols]
    """
    suggestions = {}
    tables = list(dfs.keys())
    for i, a in enumerate(tables):
        for b in tables[i + 1 :]:
            common = set(dfs[a].columns) & set(dfs[b].columns)
            id_like = [c for c in common if c.lower().endswith("id")]
            candidates = id_like if id_like else list(common)
            if candidates:
                suggestions[(a, b)] = candidates
    return suggestions


def detect_feature_types(df, exclude: Optional[List[str]] = None) -> Tuple[List[str], List[str]]:
    exclude = exclude or []
    numeric_types = {"int", "bigint", "double", "float", "long", "decimal", "smallint", "tinyint"}
    numeric_cols, cat_cols = [], []
    for name, dtype in df.dtypes:
        if name in exclude:
            continue
        base = dtype.split("(")[0].lower()
        if base in numeric_types:
            numeric_cols.append(name)
        else:
            cat_cols.append(name)
    return numeric_cols, cat_cols


def build_preprocess_pipeline(df, model_cfg: ModelConfig, feature_cols: Optional[List[str]] = None):
    _require_pyspark()
    from pyspark.ml.feature import StringIndexer, OneHotEncoder, VectorAssembler, StandardScaler
    from pyspark.ml import Pipeline

    label_col = model_cfg.label_column
    exclude = [label_col]
    numeric_cols, cat_cols = detect_feature_types(df, exclude=exclude)
    if feature_cols:
        # respect provided list
        numeric_cols = [c for c in feature_cols if c in numeric_cols]
        cat_cols = [c for c in feature_cols if c in cat_cols]

    stages = []
    output_features = []

    # Categorical: index + one-hot
    if cat_cols:
        indexers = [StringIndexer(inputCol=c, outputCol=f"{c}_idx", handleInvalid="keep") for c in cat_cols]
        encoders = [OneHotEncoder(inputCols=[f"{c}_idx"], outputCols=[f"{c}_oh"], handleInvalid="keep") for c in cat_cols]
        stages.extend(indexers)
        stages.extend(encoders)
        output_features.extend([f"{c}_oh" for c in cat_cols])

    # Numeric
    if numeric_cols:
        if model_cfg.apply_scaling:
            assembler_num = VectorAssembler(inputCols=numeric_cols, outputCol="numeric_vec", handleInvalid="keep")
            scaler = StandardScaler(inputCol="numeric_vec", outputCol="numeric_scaled", withStd=True, withMean=False)
            stages.extend([assembler_num, scaler])
            output_features.append("numeric_scaled")
        else:
            output_features.extend(numeric_cols)

    # Assemble full feature vector
    assembler = VectorAssembler(inputCols=output_features, outputCol="features", handleInvalid="keep")
    stages.append(assembler)

    pipeline = Pipeline(stages=stages)
    return pipeline, output_features


def _get_estimator(model_cfg: ModelConfig, feature_col: str = "features"):
    _require_pyspark()
    from pyspark.ml.regression import LinearRegression, RandomForestRegressor, GBTRegressor
    from pyspark.ml.classification import LogisticRegression, RandomForestClassifier, GBTClassifier

    mt = model_cfg.model_type
    if model_cfg.problem_type == "classification":
        if mt == "logistic_regression":
            return LogisticRegression(featuresCol=feature_col, labelCol=model_cfg.label_column, maxIter=50, regParam=0.01)
        if mt == "random_forest":
            return RandomForestClassifier(featuresCol=feature_col, labelCol=model_cfg.label_column, numTrees=100, maxDepth=10)
        if mt == "gbt":
            return GBTClassifier(featuresCol=feature_col, labelCol=model_cfg.label_column, maxDepth=5, maxIter=50)
        raise ValueError(f"Unsupported classification model_type: {mt}")
    else:
        if mt == "linear_regression":
            return LinearRegression(featuresCol=feature_col, labelCol=model_cfg.label_column, regParam=0.01, elasticNetParam=0.0)
        if mt == "random_forest":
            return RandomForestRegressor(featuresCol=feature_col, labelCol=model_cfg.label_column, numTrees=200, maxDepth=12)
        if mt == "gbt":
            return GBTRegressor(featuresCol=feature_col, labelCol=model_cfg.label_column, maxDepth=5, maxIter=100)
        raise ValueError(f"Unsupported regression model_type: {mt}")


def evaluate_model(pred_df, model_cfg: ModelConfig) -> Dict[str, float]:
    _require_pyspark()
    from pyspark.ml.evaluation import RegressionEvaluator, BinaryClassificationEvaluator, MulticlassClassificationEvaluator

    metrics = model_cfg.metrics or (["rmse", "mae"] if model_cfg.problem_type == "regression" else ["auc", "accuracy"])
    results = {}
    if model_cfg.problem_type == "regression":
        for m in metrics:
            if m.lower() in ("rmse", "mae", "mse", "r2"):
                evaluator = RegressionEvaluator(labelCol=model_cfg.label_column, predictionCol="prediction", metricName=m.lower())
                results[m.lower()] = float(evaluator.evaluate(pred_df))
    else:
        for m in metrics:
            m_low = m.lower()
            if m_low == "auc":
                evaluator = BinaryClassificationEvaluator(labelCol=model_cfg.label_column, rawPredictionCol="rawPrediction", metricName="areaUnderROC")
                results["auc"] = float(evaluator.evaluate(pred_df))
            elif m_low == "accuracy":
                evaluator = MulticlassClassificationEvaluator(labelCol=model_cfg.label_column, predictionCol="prediction", metricName="accuracy")
                results["accuracy"] = float(evaluator.evaluate(pred_df))
            elif m_low == "f1":
                evaluator = MulticlassClassificationEvaluator(labelCol=model_cfg.label_column, predictionCol="prediction", metricName="f1")
                results["f1"] = float(evaluator.evaluate(pred_df))
    return results


def train_pipeline(spark_df, model_cfg: ModelConfig, feature_cols: Optional[List[str]] = None):
    _require_pyspark()
    from pyspark.ml import Pipeline

    preprocess, used_features = build_preprocess_pipeline(spark_df, model_cfg, feature_cols)
    estimator = _get_estimator(model_cfg)
    pipeline = Pipeline(stages=preprocess.getStages() + [estimator])

    train_frac = model_cfg.train_fraction
    train_df, test_df = spark_df.randomSplit([train_frac, 1 - train_frac], seed=model_cfg.random_seed) if 0 < train_frac < 1 else (spark_df, None)

    model = pipeline.fit(train_df)
    metrics = {}
    if test_df:
        preds = model.transform(test_df)
        metrics = evaluate_model(preds, model_cfg)

    meta = {
        "model_config": asdict(model_cfg),
        "features_used": used_features,
        "metrics": metrics,
    }
    return model, meta


def save_pipeline(model, meta: Dict[str, object], path: str):
    # Save Spark pipeline
    model.write().overwrite().save(path)
    with open(f"{path}/metadata.json", "w", encoding="utf-8") as f:
        json.dump(meta, f, indent=2)


def load_pipeline(spark, path: str):
    _require_pyspark()
    from pyspark.ml.pipeline import PipelineModel

    model = PipelineModel.load(path)
    meta_path = f"{path}/metadata.json"
    meta = {}
    try:
        with open(meta_path, "r", encoding="utf-8") as f:
            meta = json.load(f)
    except FileNotFoundError:
        meta = {}
    return model, meta


def run_full_spark_ml_pipeline(spark, cfg: PipelineConfig):
    """
    End-to-end Spark ML pipeline:
      - Load selected tables
      - Join using join_config or heuristic
      - Select features/label
      - Build preprocess + model pipeline
      - Train, evaluate, persist
    """
    _require_pyspark()

    # Load tables
    dfs = {tbl: load_table(spark, tbl, cfg.table_source) for tbl in cfg.selected_tables}

    # Join
    joined = join_tables(dfs, cfg.join_config)

    # Column selection
    feature_cols = []
    if cfg.feature_columns:
        for tbl, cols in cfg.feature_columns.items():
            feature_cols.extend(cols)
    label_table = cfg.label["table"]
    label_col = cfg.label["column"]
    model_cfg = cfg.model
    model_cfg.label_column = label_col

    selected_cols = list(set(feature_cols + [label_col])) if feature_cols else joined.columns
    dataset = joined.select(*selected_cols)

    model, meta = train_pipeline(dataset, model_cfg, feature_cols if feature_cols else None)
    meta.update({"tables_used": cfg.selected_tables, "join_config": cfg.join_config, "feature_columns": feature_cols or "auto"})
    save_pipeline(model, meta, cfg.output_path)
    return model, meta
