"""Project paths and frozen model configuration from prior notebook steps."""

from __future__ import annotations

import json
from pathlib import Path

RANDOM_STATE = 42
CALIBRATION_METHOD = "sigmoid"
CALIBRATION_CV = 5

XGB_FIXED_PARAMS = {
    "n_estimators": 200,
    "max_depth": 3,
    "learning_rate": 0.1,
    "subsample": 0.8,
    "colsample_bytree": 1.0,
    "random_state": RANDOM_STATE,
    "eval_metric": "logloss",
    "n_jobs": -1,
}

DEFAULT_RETENTION_OFFER_COST = 50
DEFAULT_LOST_CUSTOMER_COST = 500

CHOSEN_THRESHOLD_FILENAME = "chosen_threshold.json"
MODEL_BUNDLE_FILENAME = "churn_model_bundle.joblib"
CHURN_PIPELINE_FILENAME = "churn_pipeline.joblib"
MODEL_CONFIG_FILENAME = "model_config.json"


def project_root() -> Path:
    """Return the repository root directory."""
    return Path(__file__).resolve().parents[1]


def data_raw_dir() -> Path:
    return project_root() / "data" / "raw"


def data_processed_dir() -> Path:
    return project_root() / "data" / "processed"


def reports_dir() -> Path:
    return project_root() / "reports"


def models_dir() -> Path:
    return project_root() / "models"


def chosen_threshold_path() -> Path:
    return reports_dir() / CHOSEN_THRESHOLD_FILENAME


def model_bundle_path() -> Path:
    return models_dir() / MODEL_BUNDLE_FILENAME


def churn_pipeline_path() -> Path:
    return models_dir() / CHURN_PIPELINE_FILENAME


def model_config_path() -> Path:
    return models_dir() / MODEL_CONFIG_FILENAME


def load_model_config(path: Path | str | None = None) -> dict:
    """Load the saved model configuration for apps and services."""
    config_path = Path(path) if path else model_config_path()
    if not config_path.exists():
        raise FileNotFoundError(
            f"Model config not found at {config_path.resolve()}. "
            "Save the inference pipeline first."
        )
    return json.loads(config_path.read_text(encoding="utf-8"))


def load_chosen_threshold_config(path: Path | str | None = None) -> dict:
    """Load the frozen business threshold selected on calibrated validation data."""
    path = Path(path) if path else chosen_threshold_path()
    if not path.exists():
        raise FileNotFoundError(
            f"Chosen threshold config not found at {path.resolve()}. "
            "Run calibrated threshold optimization first."
        )
    return json.loads(path.read_text(encoding="utf-8"))


def get_decision_threshold(path: Path | str | None = None) -> float:
    """Return the validation-tuned decision threshold for retention outreach."""
    config = load_chosen_threshold_config(path)
    return float(config["threshold"])
