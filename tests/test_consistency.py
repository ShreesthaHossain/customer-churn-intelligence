"""Deployment consistency tests using saved model artifacts."""

from __future__ import annotations

from pathlib import Path

import pytest

from src.config import get_decision_threshold, load_chosen_threshold_config, load_model_config
from src.inference import build_churn_prediction_response, predict_churn_probability, prepare_inference_features
from src.model import ChurnModelBundle, load_churn_pipeline
from src.policy import (
    HIGH_RISK_UI_BAND,
    churn_prediction_label,
    classify_risk_level,
    recommended_action,
)

PROJECT_ROOT = Path(__file__).resolve().parents[1]
PIPELINE_PATH = PROJECT_ROOT / "models" / "churn_pipeline.joblib"
MODEL_CONFIG_PATH = PROJECT_ROOT / "models" / "model_config.json"


@pytest.mark.skipif(not PIPELINE_PATH.exists(), reason="pipeline artifact missing")
def test_saved_pipeline_loads(saved_pipeline: ChurnModelBundle) -> None:
    assert isinstance(saved_pipeline, ChurnModelBundle)
    assert hasattr(saved_pipeline.preprocessor, "transformers_")
    assert hasattr(saved_pipeline.model, "calibrated_classifiers_")


@pytest.mark.skipif(not MODEL_CONFIG_PATH.exists(), reason="model config missing")
def test_saved_model_config_loads(saved_model_config: dict) -> None:
    assert saved_model_config["selected_model"] == "XGBoost + class weighting"
    assert saved_model_config["decision_threshold"] == 0.1
    assert saved_model_config["calibration_retained"] is True


@pytest.mark.skipif(not PIPELINE_PATH.exists(), reason="pipeline artifact missing")
def test_threshold_consistency_across_artifacts(
    saved_pipeline: ChurnModelBundle,
    saved_model_config: dict,
) -> None:
    chosen = load_chosen_threshold_config()
    assert saved_pipeline.threshold == saved_model_config["decision_threshold"]
    assert saved_pipeline.threshold == chosen["threshold"]
    assert saved_pipeline.threshold == get_decision_threshold()
    assert saved_pipeline.threshold == 0.1


@pytest.mark.skipif(not PIPELINE_PATH.exists(), reason="pipeline artifact missing")
def test_validation_customer_probability_bounds(
    saved_pipeline: ChurnModelBundle,
    validation_customer_record: dict,
) -> None:
    features = prepare_inference_features(validation_customer_record)
    probability = float(predict_churn_probability(saved_pipeline, features)[0])
    assert 0.0 <= probability <= 1.0


@pytest.mark.skipif(not PIPELINE_PATH.exists(), reason="pipeline artifact missing")
def test_validation_customer_prediction_policy_consistency(
    saved_pipeline: ChurnModelBundle,
    saved_model_config: dict,
    validation_customer_record: dict,
) -> None:
    response = build_churn_prediction_response(
        saved_pipeline,
        validation_customer_record,
        model_config=saved_model_config,
    )
    probability = response["churn_probability"]
    threshold = response["threshold"]

    assert response["prediction"] == churn_prediction_label(probability, threshold)
    assert response["risk_level"] == classify_risk_level(probability, threshold)
    assert response["recommended_action"] == recommended_action(probability >= threshold)

    if probability < threshold:
        assert response["prediction"] == "No Churn"
        assert response["risk_level"] == "Low"
    elif probability < HIGH_RISK_UI_BAND:
        assert response["prediction"] == "Churn"
        assert response["risk_level"] == "Elevated"
    else:
        assert response["prediction"] == "Churn"
        assert response["risk_level"] == "High"


@pytest.mark.skipif(not PIPELINE_PATH.exists(), reason="pipeline artifact missing")
def test_unseen_categorical_value_does_not_crash_inference(
    saved_pipeline: ChurnModelBundle,
    validation_customer_record: dict,
) -> None:
    record = dict(validation_customer_record)
    record["gender"] = "__UNSEEN_CATEGORY__"

    features = prepare_inference_features(record)
    probability = float(predict_churn_probability(saved_pipeline, features)[0])

    assert 0.0 <= probability <= 1.0


@pytest.mark.skipif(not PIPELINE_PATH.exists(), reason="pipeline artifact missing")
def test_churn_pipeline_path_loader_matches_fixture(saved_pipeline: ChurnModelBundle) -> None:
    loaded = load_churn_pipeline()
    assert loaded.threshold == saved_pipeline.threshold
    assert loaded.metadata == saved_pipeline.metadata
