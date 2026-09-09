"""Session-scoped training for incompatible uploaded datasets."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np
import pandas as pd

from src.data_split import SplitData, stratified_train_val_test_split
from src.dataset_schema import UploadDatasetSpec, normalize_binary_target, validate_retrain_dataset
from src.evaluation import compute_classification_metrics, select_min_cost_threshold, sweep_thresholds
from src.inference import predict_batch
from src.model import ChurnModelBundle, build_calibrated_model, compute_scale_pos_weight
from src.policy import (
    churn_prediction_label,
    classify_risk_level,
    recommended_action,
)
from src.preprocessing import build_preprocessor, fit_transform_train, transform_features


@dataclass
class TrainingResult:
    """Output of a session-scoped upload training run."""

    bundle: ChurnModelBundle
    results: pd.DataFrame
    validation_metrics: dict[str, Any]
    test_metrics: dict[str, Any]
    spec: UploadDatasetSpec


def _prepare_upload_frame(
    df: pd.DataFrame,
    spec: UploadDatasetSpec,
) -> tuple[pd.DataFrame, pd.Series, pd.Series]:
    working = df.copy()
    y, _ = normalize_binary_target(working[spec.target_col])

    if spec.id_col == "__upload_row_id__":
        customer_ids = pd.Series(range(len(working)), name=spec.id_col)
    else:
        customer_ids = working[spec.id_col].astype(str)

    X = working[spec.feature_cols].copy()
    return X, y, customer_ids


def train_and_score_upload(
    df: pd.DataFrame,
    target_col: str,
    id_col: str | None = None,
) -> TrainingResult:
    """
    Train a session-scoped model on uploaded data and score all rows.

    Does not overwrite the frozen production Telco pipeline.
    """
    spec = validate_retrain_dataset(df, target_col, id_col=id_col)
    X, y, customer_ids = _prepare_upload_frame(df, spec)

    split = stratified_train_val_test_split(X, y, customer_ids)
    scale_pos_weight = compute_scale_pos_weight(split.y_train)

    preprocessor = build_preprocessor(spec.numeric_cols, spec.categorical_cols)
    preprocessor, _ = fit_transform_train(preprocessor, split.X_train)

    model = build_calibrated_model(scale_pos_weight)
    X_train_t = transform_features(preprocessor, split.X_train)
    y_train = (split.y_train == "Yes").astype(int).to_numpy()
    model.fit(X_train_t, y_train)

    X_val_t = transform_features(preprocessor, split.X_val)
    val_probs = model.predict_proba(X_val_t)[:, 1]
    threshold_df = sweep_thresholds(split.y_val, val_probs)
    chosen = select_min_cost_threshold(threshold_df)
    threshold = float(chosen["Threshold"])

    X_test_t = transform_features(preprocessor, split.X_test)
    test_probs = model.predict_proba(X_test_t)[:, 1]

    validation_metrics = compute_classification_metrics(split.y_val, val_probs, threshold)
    test_metrics = compute_classification_metrics(split.y_test, test_probs, threshold)

    metadata = {
        "model_family": "XGBoost",
        "training_mode": "upload_session",
        "feature_cols": spec.feature_cols,
        "numeric_cols": spec.numeric_cols,
        "categorical_cols": spec.categorical_cols,
        "positive_class": spec.positive_class,
        "target_col": spec.target_col,
        "scale_pos_weight": scale_pos_weight,
    }

    bundle = ChurnModelBundle(
        preprocessor=preprocessor,
        model=model,
        threshold=threshold,
        metadata=metadata,
    )

    all_features = df[spec.feature_cols].copy()
    batch = predict_batch(bundle, all_features, customer_ids=customer_ids)
    batch = batch.rename(columns={"customerID": "primary_key"})
    batch["prediction"] = batch["churn_probability"].map(
        lambda p: churn_prediction_label(float(p), threshold)
    )
    batch["risk_level"] = batch["churn_probability"].map(
        lambda p: classify_risk_level(float(p), threshold)
    )
    batch["recommended_action"] = batch["retention_recommended"].map(recommended_action)
    batch["actual_churn"] = normalize_binary_target(df[spec.target_col])[0].reset_index(drop=True)

    for col in spec.feature_cols:
        batch[col] = df[col].reset_index(drop=True)

    batch = batch.sort_values("churn_probability", ascending=False).reset_index(drop=True)

    return TrainingResult(
        bundle=bundle,
        results=batch,
        validation_metrics=validation_metrics,
        test_metrics=test_metrics,
        spec=spec,
    )
