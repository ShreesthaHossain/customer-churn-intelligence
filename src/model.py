"""Training and persistence helpers for the final calibrated XGBoost model."""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import joblib
import numpy as np
import pandas as pd
from sklearn.calibration import CalibratedClassifierCV
from sklearn.compose import ColumnTransformer
from xgboost import XGBClassifier

from src.config import (
    CALIBRATION_CV,
    CALIBRATION_METHOD,
    XGB_FIXED_PARAMS,
    churn_pipeline_path,
    get_decision_threshold,
    model_bundle_path,
    model_config_path,
    models_dir,
)
from src.data_separation import (
    CATEGORICAL_FEATURE_COLS,
    FEATURE_COLS,
    NUMERIC_FEATURE_COLS,
)
from src.data_split import SplitData, load_split_from_manifest
from src.evaluation import compute_classification_metrics, compute_simulated_business_cost
from src.preprocessing import build_preprocessor, fit_transform_train, transform_features


@dataclass(frozen=True)
class ChurnModelBundle:
    """Fitted preprocessor, calibrated model, and frozen decision policy."""

    preprocessor: ColumnTransformer
    model: CalibratedClassifierCV
    threshold: float
    metadata: dict[str, Any]


def compute_scale_pos_weight(y_train: pd.Series | np.ndarray) -> float:
    """Class weighting ratio used by the selected XGBoost configuration."""
    if isinstance(y_train, pd.Series):
        labels = (y_train == "Yes").astype(int).to_numpy()
    else:
        labels = np.asarray(y_train)
        if labels.dtype == object:
            labels = (labels == "Yes").astype(int)
        else:
            labels = labels.astype(int)

    negatives = int((labels == 0).sum())
    positives = int((labels == 1).sum())
    if positives == 0:
        raise ValueError("Cannot compute scale_pos_weight without positive training examples.")
    return negatives / positives


def build_xgb_classifier(scale_pos_weight: float) -> XGBClassifier:
    """Build the uncalibrated XGBoost model with frozen hyperparameters."""
    return XGBClassifier(**XGB_FIXED_PARAMS, scale_pos_weight=scale_pos_weight)


def build_calibrated_model(scale_pos_weight: float) -> CalibratedClassifierCV:
    """Build the sigmoid-calibrated final model (Step 17 retained configuration)."""
    base_estimator = build_xgb_classifier(scale_pos_weight)
    return CalibratedClassifierCV(
        base_estimator,
        method=CALIBRATION_METHOD,
        cv=CALIBRATION_CV,
    )


def train_final_model(
    split: SplitData,
    *,
    threshold: float | None = None,
) -> ChurnModelBundle:
    """
    Fit preprocessing and the calibrated model on training data only.

    Validation and test partitions are not used for fitting.
    """
    threshold = float(threshold if threshold is not None else get_decision_threshold())
    scale_pos_weight = compute_scale_pos_weight(split.y_train)

    preprocessor = build_preprocessor()
    preprocessor, _ = fit_transform_train(preprocessor, split.X_train)

    model = build_calibrated_model(scale_pos_weight)
    X_train_t = transform_features(preprocessor, split.X_train)
    y_train = (split.y_train == "Yes").astype(int).to_numpy()
    model.fit(X_train_t, y_train)

    metadata = {
        "model_family": "XGBoost",
        "calibration_method": CALIBRATION_METHOD,
        "calibration_cv": CALIBRATION_CV,
        "scale_pos_weight": scale_pos_weight,
        "xgb_params": {**XGB_FIXED_PARAMS, "scale_pos_weight": scale_pos_weight},
        "threshold_source": "reports/chosen_threshold.json",
        "trained_on": "train_only",
    }
    return ChurnModelBundle(
        preprocessor=preprocessor,
        model=model,
        threshold=threshold,
        metadata=metadata,
    )


def save_model_bundle(bundle: ChurnModelBundle, path: Path | str | None = None) -> Path:
    """Persist a fitted model bundle for reuse by inference and future apps."""
    output_path = Path(path) if path else model_bundle_path()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump(bundle, output_path)
    return output_path


def load_model_bundle(path: Path | str | None = None) -> ChurnModelBundle:
    """Load a previously saved model bundle from disk."""
    input_path = Path(path) if path else model_bundle_path()
    if not input_path.exists():
        raise FileNotFoundError(
            f"Model bundle not found at {input_path.resolve()}. "
            "Train and save the model first."
        )
    bundle = joblib.load(input_path)
    if not isinstance(bundle, ChurnModelBundle):
        raise TypeError(f"Expected ChurnModelBundle, got {type(bundle)!r}.")
    return bundle


def load_churn_pipeline(path: Path | str | None = None) -> ChurnModelBundle:
    """Load the saved inference pipeline artifact used by Streamlit and FastAPI."""
    from src.config import churn_pipeline_path

    return load_model_bundle(path or churn_pipeline_path())


def run_training_pipeline(
    split: SplitData | None = None,
    *,
    threshold: float | None = None,
    save_path: Path | str | None = None,
) -> tuple[ChurnModelBundle, Path]:
    """Train the final model on train data and persist the bundle."""
    split = split or load_split_from_manifest()
    bundle = train_final_model(split, threshold=threshold)
    output_path = save_model_bundle(bundle, save_path)
    return bundle, output_path


def build_model_config(bundle: ChurnModelBundle, split: SplitData) -> dict[str, Any]:
    """Build deployment metadata consumed by Streamlit, FastAPI, and tests."""
    from src.inference import predict_churn_probability

    val_probs = predict_churn_probability(bundle, split.X_val)
    metrics = compute_classification_metrics(split.y_val, val_probs, bundle.threshold)
    return {
        "selected_model": "XGBoost + class weighting",
        "model_family": "XGBoost",
        "calibration_retained": True,
        "calibration_method": "sigmoid (Platt)",
        "decision_threshold": bundle.threshold,
        "threshold_source": "validation_min_simulated_business_cost",
        "probabilities": "calibrated",
        "feature_columns": FEATURE_COLS,
        "numeric_feature_columns": NUMERIC_FEATURE_COLS,
        "categorical_feature_columns": CATEGORICAL_FEATURE_COLS,
        "trained_on": "train_only",
        "scale_pos_weight": bundle.metadata.get("scale_pos_weight"),
        "validation_metrics_at_threshold": {
            "Precision": metrics["precision"],
            "Recall": metrics["recall"],
            "F1": metrics["f1"],
            "False_Positives": metrics["false_positives"],
            "False_Negatives": metrics["false_negatives"],
            "Predicted_Churners": metrics["predicted_churners"],
            "Total_Cost_USD_simulated": compute_simulated_business_cost(
                metrics["false_positives"],
                metrics["false_negatives"],
            ),
        },
        "pipeline_artifact": "models/churn_pipeline.joblib",
    }


def write_model_config(config: dict[str, Any], path: Path | str | None = None) -> Path:
    """Persist deployment metadata for apps and services."""
    output_path = Path(path) if path else model_config_path()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(config, indent=2), encoding="utf-8")
    return output_path


def save_deployment_artifacts(
    bundle: ChurnModelBundle,
    split: SplitData | None = None,
) -> tuple[Path, Path]:
    """Save the inference pipeline and deployment config used in production."""
    split = split or load_split_from_manifest()
    pipeline_path = save_model_bundle(bundle, churn_pipeline_path())
    config_path = write_model_config(build_model_config(bundle, split))
    return pipeline_path, config_path


def ensure_models_dir() -> Path:
    """Create the models directory if it does not exist."""
    path = models_dir()
    path.mkdir(parents=True, exist_ok=True)
    return path
