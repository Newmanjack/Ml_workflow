from __future__ import annotations

"""
PySpark ML pipeline builder with table loading, joining, preprocessing, and model training.

This module stays in Spark for large datasets and builds a feature + model pipeline with minimal
driver-side work. It includes math-aware options (scaling for linear models, encoding for categoricals)
and persists the full pipeline + metadata for reproducibility.
"""

import json
import logging
from dataclasses import dataclass, asdict
from typing import Dict, List, Optional, Tuple
import copy


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
    model_type: str = "linear_regression"  # linear_regression | logistic_regression | random_forest | gbt | decision_tree | svm
    apply_scaling: bool = True
    train_fraction: float = 0.8
    random_seed: int = 42
    metrics: Optional[List[str]] = None  # e.g., ["rmse","mae"] or ["auc","accuracy"]
    # Regularization / iterations for linear/logistic
    reg_param: float = 0.01
    elastic_net_param: float = 0.0  # 0=L2, 1=L1
    max_iter: int = 100
    # Tree/GBM params
    max_depth: int = 8
    num_trees: int = 200
    step_size: float = 0.1
    # SVM (linear SVC) / soft margin
    svm_reg_param: float = 0.0
    svm_max_iter: int = 100
    # Class imbalance handling
    class_weight: bool = False
    # Feature engineering extras
    polynomial_degree: Optional[int] = None  # only applied to numeric features if set (>1)
    drop_correlated_threshold: Optional[float] = None  # drop one of highly correlated numeric pairs above this


@dataclass
class PipelineConfig:
    selected_tables: List[str]
    table_source: TableSourceConfig = None
    join_config: Optional[Dict[str, Dict[str, str]]] = None  # table -> {"join_key": "..."}
    feature_columns: Optional[Dict[str, List[str]]] = None  # table -> cols
    label: Optional[Dict[str, str]] = None  # {"table": "...", "column": "..."}
    model: ModelConfig = None
    output_path: str = "ml_pipeline"

    def __post_init__(self):
        if self.table_source is None:
            self.table_source = TableSourceConfig()
        if self.model is None:
            self.model = ModelConfig()
        if self.label is None:
            raise ValueError("label must be provided: {'table': '...', 'column': '...'}")
        if not isinstance(self.model, ModelConfig):
            raise ValueError("model must be a ModelConfig instance")


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


def suggest_joins(
    dfs: Dict[str, object],
    semantic_relations: Optional[Dict[str, Dict[str, str]]] = None,
    join_map: Optional[Dict[Tuple[str, str], Tuple[str, str]]] = None,
) -> Dict[Tuple[str, str], List[str]]:
    """
    Suggest join keys between all pairs of tables based on:
      1) Provided semantic_relations (table -> {"join_key": "..."}).
      2) Provided join_map: {(from_table, to_table): (from_col, to_col)}}.
      2) Common column names (prioritizing *id columns).
      3) Most frequent column names across all tables (fallback).
    Returns a mapping: (tableA, tableB) -> [candidate_cols]
    """
    suggestions: Dict[Tuple[str, str], List[str]] = {}
    tables = list(dfs.keys())

    # Global frequency of column names across all tables
    col_freq: Dict[str, int] = {}
    for df in dfs.values():
        for c in df.columns:
            col_freq[c] = col_freq.get(c, 0) + 1

    for i, a in enumerate(tables):
        for b in tables[i + 1 :]:
            # 1b) explicit join_map
            if join_map and (a, b) in join_map:
                fk, tk = join_map[(a, b)]
                if fk and tk:
                    suggestions[(a, b)] = [fk, tk] if fk != tk else [fk]
                    continue
            # 1) semantic relations
            if semantic_relations and a in semantic_relations and b in semantic_relations:
                ak = semantic_relations[a].get("join_key")
                bk = semantic_relations[b].get("join_key")
                if ak and bk:
                    suggestions[(a, b)] = [ak] if ak == bk else [ak, bk]
                    continue

            common = set(dfs[a].columns) & set(dfs[b].columns)
            id_like = [c for c in common if c.lower().endswith("id")]
            if id_like:
                suggestions[(a, b)] = id_like
            elif common:
                suggestions[(a, b)] = list(common)
            else:
                # 3) Fallback: most frequent column names across all tables
                frequent_cols = sorted(col_freq.items(), key=lambda x: x[1], reverse=True)
                if frequent_cols:
                    top = [c for c, freq in frequent_cols if freq > 1][:1]
                    if top:
                        suggestions[(a, b)] = top
    return suggestions


def plan_joins(
    dfs: Dict[str, object],
    semantic_relations: Optional[Dict[str, Dict[str, str]]] = None,
    join_map: Optional[Dict[Tuple[str, str], Tuple[str, str]]] = None,
) -> Dict[str, object]:
    """
    Plan joins across tables and report joinable vs unjoinable pairs.
    Returns {"joinable": {(a,b): [candidates]}, "unjoinable": [(a,b), ...]}
    """
    suggestions = suggest_joins(dfs, semantic_relations=semantic_relations, join_map=join_map)
    tables = list(dfs.keys())
    unjoinable = []
    for i, a in enumerate(tables):
        for b in tables[i + 1 :]:
            if (a, b) not in suggestions and (b, a) not in suggestions:
                unjoinable.append((a, b))
    return {"joinable": suggestions, "unjoinable": unjoinable}


def join_tables_with_plan(
    dfs: Dict[str, object],
    join_plan: Dict[Tuple[str, str], List[str]],
    base_table: Optional[str] = None,
    how: str = "inner",
    fallback_on_common_cols: bool = True,
):
    """
    Join a dict of Spark DataFrames using a provided join_plan (from plan_joins/suggest_joins).
    Returns the joined Spark DataFrame.
    """
    tables = list(dfs.keys())
    base = base_table or tables[0]
    joined = dfs[base]
    for tbl in tables:
        if tbl == base:
            continue
        key = (base, tbl) if (base, tbl) in join_plan else (tbl, base)
        if key not in join_plan:
            continue
        cols = join_plan[key]
        if len(cols) == 1:
            lk = rk = cols[0]
        else:
            lk, rk = cols[0], cols[1]
        joined = joined.join(dfs[tbl], on=joined[lk] == dfs[tbl][rk], how=how)
    return joined


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
    from pyspark.ml.feature import StringIndexer, OneHotEncoder, VectorAssembler, StandardScaler, PolynomialExpansion
    from pyspark.ml import Pipeline
    logger = logging.getLogger("smart_pipeline.pyspark_ml")

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

        if model_cfg.polynomial_degree and model_cfg.polynomial_degree > 1:
            poly_input = "numeric_scaled" if model_cfg.apply_scaling else "numeric_vec_poly"
            if not model_cfg.apply_scaling:
                assembler_num = VectorAssembler(inputCols=numeric_cols, outputCol="numeric_vec_poly", handleInvalid="keep")
                stages.append(assembler_num)
            poly = PolynomialExpansion(inputCol=poly_input, outputCol="numeric_poly", degree=model_cfg.polynomial_degree)
            stages.append(poly)
            output_features.append("numeric_poly")

    # Assemble full feature vector
    assembler = VectorAssembler(inputCols=output_features, outputCol="features", handleInvalid="keep")
    stages.append(assembler)

    pipeline = Pipeline(stages=stages)
    return pipeline, output_features


def _get_estimator(model_cfg: ModelConfig, feature_col: str = "features"):
    _require_pyspark()
    from pyspark.ml.regression import LinearRegression, RandomForestRegressor, GBTRegressor
    from pyspark.ml.classification import LogisticRegression, RandomForestClassifier, GBTClassifier
    from pyspark.ml.classification import DecisionTreeClassifier
    from pyspark.ml.regression import DecisionTreeRegressor
    from pyspark.ml.classification import LinearSVC
    logger = logging.getLogger("smart_pipeline.pyspark_ml")

    mt = model_cfg.model_type
    if model_cfg.problem_type == "classification":
        if mt == "logistic_regression":
            if not model_cfg.apply_scaling:
                logger.warning("Logistic regression without scaling may underperform when features differ in scale.")
            return LogisticRegression(
                featuresCol=feature_col,
                labelCol=model_cfg.label_column,
                maxIter=model_cfg.max_iter,
                regParam=model_cfg.reg_param,
                elasticNetParam=model_cfg.elastic_net_param,
            )
        if mt == "decision_tree":
            return DecisionTreeClassifier(featuresCol=feature_col, labelCol=model_cfg.label_column, maxDepth=model_cfg.max_depth)
        if mt == "random_forest":
            return RandomForestClassifier(
                featuresCol=feature_col,
                labelCol=model_cfg.label_column,
                numTrees=model_cfg.num_trees,
                maxDepth=model_cfg.max_depth,
            )
        if mt == "gbt":
            return GBTClassifier(
                featuresCol=feature_col,
                labelCol=model_cfg.label_column,
                maxDepth=model_cfg.max_depth,
                maxIter=model_cfg.max_iter,
                stepSize=model_cfg.step_size,
            )
        if mt == "svm":
            if not model_cfg.apply_scaling:
                logger.warning("SVM/LinearSVC without scaling may perform poorly when features differ in scale.")
            return LinearSVC(
                featuresCol=feature_col,
                labelCol=model_cfg.label_column,
                maxIter=model_cfg.svm_max_iter,
                regParam=model_cfg.svm_reg_param,
            )
        raise ValueError(f"Unsupported classification model_type: {mt}")
    else:
        if mt == "linear_regression":
            if not model_cfg.apply_scaling:
                logger.warning("Linear regression without scaling may underperform when features differ in scale.")
            return LinearRegression(
                featuresCol=feature_col,
                labelCol=model_cfg.label_column,
                regParam=model_cfg.reg_param,
                elasticNetParam=model_cfg.elastic_net_param,
                maxIter=model_cfg.max_iter,
            )
        if mt == "decision_tree":
            return DecisionTreeRegressor(featuresCol=feature_col, labelCol=model_cfg.label_column, maxDepth=model_cfg.max_depth)
        if mt == "random_forest":
            return RandomForestRegressor(
                featuresCol=feature_col,
                labelCol=model_cfg.label_column,
                numTrees=model_cfg.num_trees,
                maxDepth=model_cfg.max_depth,
            )
        if mt == "gbt":
            return GBTRegressor(
                featuresCol=feature_col,
                labelCol=model_cfg.label_column,
                maxDepth=model_cfg.max_depth,
                maxIter=model_cfg.max_iter,
                stepSize=model_cfg.step_size,
            )
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
            elif m_low == "auprc":
                evaluator = BinaryClassificationEvaluator(labelCol=model_cfg.label_column, rawPredictionCol="rawPrediction", metricName="areaUnderPR")
                results["auprc"] = float(evaluator.evaluate(pred_df))
            elif m_low == "accuracy":
                evaluator = MulticlassClassificationEvaluator(labelCol=model_cfg.label_column, predictionCol="prediction", metricName="accuracy")
                results["accuracy"] = float(evaluator.evaluate(pred_df))
            elif m_low == "f1":
                evaluator = MulticlassClassificationEvaluator(labelCol=model_cfg.label_column, predictionCol="prediction", metricName="f1")
                results["f1"] = float(evaluator.evaluate(pred_df))
            elif m_low == "logloss":
                # custom log loss for binary classification using probability vector
                import pyspark.sql.functions as F
                probs = pred_df.select(
                    F.col(model_cfg.label_column).cast("double").alias("label"),
                    F.col("probability").getItem(1).alias("p1"),
                ).withColumn("p1_clipped", F.when(F.col("p1") < 1e-15, 1e-15).when(F.col("p1") > 1 - 1e-15, 1 - 1e-15).otherwise(F.col("p1")))
                logloss = probs.select(
                    F.avg(
                        -F.col("label") * F.log(F.col("p1_clipped"))
                        - (1 - F.col("label")) * F.log(1 - F.col("p1_clipped"))
                    ).alias("ll")
                ).collect()[0]["ll"]
                results["logloss"] = float(logloss)
    return results


def train_pipeline(spark_df, model_cfg: ModelConfig, feature_cols: Optional[List[str]] = None):
    _require_pyspark()
    from pyspark.ml import Pipeline
    import pyspark.sql.functions as F

    preprocess, used_features = build_preprocess_pipeline(spark_df, model_cfg, feature_cols)
    estimator = _get_estimator(model_cfg)
    pipeline = Pipeline(stages=preprocess.getStages() + [estimator])

    train_frac = model_cfg.train_fraction
    train_df, test_df = spark_df.randomSplit([train_frac, 1 - train_frac], seed=model_cfg.random_seed) if 0 < train_frac < 1 else (spark_df, None)

    # class weights for imbalance (classification only)
    if model_cfg.problem_type == "classification" and model_cfg.class_weight:
        label_col = model_cfg.label_column
        counts = train_df.groupBy(label_col).count().collect()
        total = sum(r["count"] for r in counts)
        weights = {r[label_col]: total / (len(counts) * r["count"]) for r in counts}
        train_df = train_df.withColumn(
            "classWeight",
            F.col(label_col).cast("double")
        )
        # map weights
        train_df = train_df.replace(to_replace=list(weights.keys()), value=list(weights.values()), subset=["classWeight"])
        estimator = estimator.copy({estimator.weightCol: "classWeight"}) if estimator.hasParam("weightCol") else estimator

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


def auto_join_and_train(
    dfs: Dict[str, object],
    label_table: str,
    label_column: str,
    model_cfg: ModelConfig,
    join_map: Optional[Dict[Tuple[str, str], Tuple[str, str]]] = None,
    semantic_relations: Optional[Dict[str, Dict[str, str]]] = None,
    feature_columns: Optional[List[str]] = None,
):
    """
    Auto-join multiple Spark DataFrames using join_map/semantic_relations/common columns,
    then train a Spark ML model.
    """
    from pyspark.sql import functions as F

    if label_table not in dfs:
        raise ValueError(f"Label table '{label_table}' not found in provided dfs.")

    base = dfs[label_table]
    join_suggestions = suggest_joins(dfs, semantic_relations=semantic_relations, join_map=join_map)

    for tbl_name, sdf in dfs.items():
        if tbl_name == label_table:
            continue
        key = (label_table, tbl_name) if (label_table, tbl_name) in join_suggestions else (tbl_name, label_table)
        if key not in join_suggestions:
            # skip if no join path
            continue
        candidates = join_suggestions[key]
        if len(candidates) == 1:
            lk = rk = candidates[0]
        elif len(candidates) >= 2:
            lk, rk = candidates[0], candidates[1]
        else:
            continue
        base = base.join(sdf, on=base[lk] == sdf[rk], how="inner")

    model_cfg = model_cfg
    model_cfg.label_column = label_column
    model, meta = train_pipeline(base, model_cfg, feature_columns)
    meta.update({"joined_tables": list(dfs.keys()), "join_suggestions": {str(k): v for k, v in join_suggestions.items()}})
    return model, meta, base


def train_per_table(
    dfs: Dict[str, object],
    label_column: str,
    model_cfg: Optional[ModelConfig] = None,
    feature_columns: Optional[List[str]] = None,
) -> Dict[str, Tuple[object, Dict]]:
    """
    Train separate models on each table that contains the label_column (no joins).
    Returns a mapping table_name -> (model, meta).
    """
    results = {}
    for name, sdf in dfs.items():
        if label_column not in sdf.columns:
            continue
        cfg = copy.deepcopy(model_cfg) if model_cfg else ModelConfig(label_column=label_column)
        cfg.label_column = label_column
        model, meta = train_pipeline(sdf, cfg, feature_columns)
        meta.update({"table": name, "join_used": False})
        results[name] = (model, meta)
    return results


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
