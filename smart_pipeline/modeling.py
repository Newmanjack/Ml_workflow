from __future__ import annotations

import logging
from typing import List, Optional, Tuple

logger = logging.getLogger("smart_pipeline.modeling")


def _import_spark_ml():
    try:
        from pyspark.ml.feature import VectorAssembler, StandardScaler
        from pyspark.ml.regression import LinearRegression
        from pyspark.ml.classification import LogisticRegression
        from pyspark.ml import Pipeline
        from pyspark.sql import DataFrame
        from pyspark.sql import functions as F

        return {
            "VectorAssembler": VectorAssembler,
            "StandardScaler": StandardScaler,
            "LinearRegression": LinearRegression,
            "LogisticRegression": LogisticRegression,
            "Pipeline": Pipeline,
            "DataFrame": DataFrame,
            "F": F,
        }
    except ImportError as exc:  # pragma: no cover - optional dependency
        raise ImportError("pyspark is required for Spark ML modeling; install with extras `[spark]`.") from exc


def prepare_spark_features(
    df,
    target_col: str,
    feature_cols: Optional[List[str]] = None,
    handle_missing: str = "zero",
    scale: bool = False,
):
    """
    Prepare a Spark DataFrame with assembled features for ML.
    """
    ml = _import_spark_ml()
    F = ml["F"]

    if feature_cols is None:
        feature_cols = [c for (c, t) in df.dtypes if c != target_col and t in ("double", "int", "bigint", "float")]

    sdf = df
    if handle_missing == "zero":
        sdf = sdf.fillna(0, subset=feature_cols + [target_col])
    elif handle_missing == "drop":
        sdf = sdf.na.drop(subset=feature_cols + [target_col])

    assembler = ml["VectorAssembler"](inputCols=feature_cols, outputCol="features", handleInvalid="keep")
    stages = [assembler]
    if scale:
        scaler = ml["StandardScaler"](inputCol="features", outputCol="features_scaled", withStd=True, withMean=False)
        stages.append(scaler)

    return sdf, stages, feature_cols


def train_spark_model(
    df,
    target_col: str,
    feature_cols: Optional[List[str]] = None,
    model_type: str = "regression",
    test_fraction: float = 0.2,
    handle_missing: str = "zero",
    scale: bool = False,
) -> Tuple[object, object, dict]:
    """
    Train a simple Spark ML model (regression/classification) with automatic feature assembly.
    """
    ml = _import_spark_ml()
    F = ml["F"]

    sdf, stages, used_features = prepare_spark_features(df, target_col, feature_cols, handle_missing, scale)

    if test_fraction and 0 < test_fraction < 1:
        train_df, test_df = sdf.randomSplit([1 - test_fraction, test_fraction], seed=42)
    else:
        train_df, test_df = sdf, None

    if model_type == "classification":
        algo = ml["LogisticRegression"](featuresCol=stages[-1].getOutputCol() if scale else "features", labelCol=target_col)
    else:
        algo = ml["LinearRegression"](featuresCol=stages[-1].getOutputCol() if scale else "features", labelCol=target_col)

    pipeline = ml["Pipeline"](stages=stages + [algo])
    model = pipeline.fit(train_df)

    metrics = {"features_used": used_features}
    if test_df is not None and model_type == "regression":
        preds = model.transform(test_df)
        metrics["rmse"] = preds.select(F.sqrt(F.avg((F.col("prediction") - F.col(target_col)) ** 2))).collect()[0][0]
        metrics["mae"] = preds.select(F.avg(F.abs(F.col("prediction") - F.col(target_col)))).collect()[0][0]
    elif test_df is not None and model_type == "classification":
        preds = model.transform(test_df)
        correct = preds.filter(F.col("prediction") == F.col(target_col)).count()
        total = preds.count()
        metrics["accuracy"] = correct / total if total else None

    return model, test_df, metrics
