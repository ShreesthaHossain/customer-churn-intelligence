"""Tests for inference helpers."""

from pathlib import Path

import pytest

from src.data_split import load_split_from_manifest
from src.inference import (
    load_bundle_and_predict,
    predict_batch,
    predict_single_customer,
    prepare_inference_features,
)
from src.model import run_training_pipeline

PROJECT_ROOT = Path(__file__).resolve().parents[1]
SPLIT_MANIFEST = PROJECT_ROOT / "data" / "processed" / "split_manifest.json"
THRESHOLD_PATH = PROJECT_ROOT / "reports" / "chosen_threshold.json"


@pytest.fixture(scope="module")
def trained_bundle(tmp_path_factory):
    if not SPLIT_MANIFEST.exists() or not THRESHOLD_PATH.exists():
        pytest.skip("project artifacts missing")

    split = load_split_from_manifest()
    model_path = tmp_path_factory.mktemp("models") / "bundle.joblib"
    bundle, _ = run_training_pipeline(split, save_path=model_path)
    return bundle, split, model_path


def test_prepare_inference_features_normalizes_blank_total_charges() -> None:
    record = {
        "gender": "Female",
        "SeniorCitizen": 0,
        "Partner": "Yes",
        "Dependents": "No",
        "tenure": 0,
        "PhoneService": "Yes",
        "MultipleLines": "No",
        "InternetService": "DSL",
        "OnlineSecurity": "No",
        "OnlineBackup": "Yes",
        "DeviceProtection": "No",
        "TechSupport": "No",
        "StreamingTV": "No",
        "StreamingMovies": "No",
        "Contract": "Month-to-month",
        "PaperlessBilling": "Yes",
        "PaymentMethod": "Electronic check",
        "MonthlyCharges": 29.85,
        "TotalCharges": "",
    }

    features = prepare_inference_features(record)
    assert features.loc[0, "TotalCharges"] == 0.0


def test_prepare_inference_features_rejects_invalid_blank_total_charges() -> None:
    record = {
        "gender": "Female",
        "SeniorCitizen": 0,
        "Partner": "Yes",
        "Dependents": "No",
        "tenure": 12,
        "PhoneService": "Yes",
        "MultipleLines": "No",
        "InternetService": "DSL",
        "OnlineSecurity": "No",
        "OnlineBackup": "Yes",
        "DeviceProtection": "No",
        "TechSupport": "No",
        "StreamingTV": "No",
        "StreamingMovies": "No",
        "Contract": "Month-to-month",
        "PaperlessBilling": "Yes",
        "PaymentMethod": "Electronic check",
        "MonthlyCharges": 29.85,
        "TotalCharges": "",
    }

    with pytest.raises(ValueError, match="tenure is 0"):
        prepare_inference_features(record)


def test_predict_single_customer_from_validation_row(trained_bundle) -> None:
    bundle, split, _ = trained_bundle
    row = split.X_val.iloc[0].to_dict()
    row["customerID"] = split.id_val.iloc[0]

    result = predict_single_customer(bundle, row)

    assert result["customerID"] == row["customerID"]
    assert 0.0 <= result["churn_probability"] <= 1.0
    assert result["decision_threshold"] == bundle.threshold
    assert isinstance(result["retention_recommended"], bool)


def test_predict_batch_with_customer_ids(trained_bundle) -> None:
    bundle, split, _ = trained_bundle
    batch = split.X_val.iloc[:3]
    ids = split.id_val.iloc[:3]

    output = predict_batch(bundle, batch, customer_ids=ids)

    assert len(output) == 3
    assert list(output.columns[:2]) == ["customerID", "churn_probability"]
    assert (output["churn_probability"] >= 0).all()
    assert (output["churn_probability"] <= 1).all()


def test_load_bundle_and_predict(trained_bundle) -> None:
    bundle, split, model_path = trained_bundle
    row = split.X_val.iloc[1].to_dict()
    row["customerID"] = split.id_val.iloc[1]

    result = load_bundle_and_predict(row, bundle_path=str(model_path))

    assert result["customerID"] == row["customerID"]
    assert "churn_probability" in result
