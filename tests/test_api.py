"""Tests for FastAPI churn prediction service."""

from pathlib import Path

import pytest
from fastapi.testclient import TestClient

PROJECT_ROOT = Path(__file__).resolve().parents[1]
PIPELINE_PATH = PROJECT_ROOT / "models" / "churn_pipeline.joblib"
CONFIG_PATH = PROJECT_ROOT / "models" / "model_config.json"


@pytest.fixture(scope="module")
def client():
    if not PIPELINE_PATH.exists() or not CONFIG_PATH.exists():
        pytest.skip("model artifacts missing")
    from api.main import app

    with TestClient(app) as test_client:
        yield test_client


@pytest.mark.skipif(not PIPELINE_PATH.exists(), reason="pipeline artifact missing")
def test_health_endpoint(client) -> None:
    response = client.get("/health")
    assert response.status_code == 200
    payload = response.json()
    assert payload["status"] == "ok"
    assert payload["model_loaded"] is True
    assert "XGBoost" in payload["model_version"]


@pytest.mark.skipif(not PIPELINE_PATH.exists(), reason="pipeline artifact missing")
def test_predict_churn_endpoint(client) -> None:
    payload = {
        "gender": "Female",
        "SeniorCitizen": 0,
        "Partner": "Yes",
        "Dependents": "No",
        "tenure": 1,
        "PhoneService": "No",
        "MultipleLines": "No phone service",
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
        "TotalCharges": 29.85,
        "customerID": "API-TEST-001",
    }

    response = client.post("/predict_churn", json=payload)
    assert response.status_code == 200
    result = response.json()
    assert 0.0 <= result["churn_probability"] <= 1.0
    assert result["threshold"] == 0.1
    assert result["prediction"] in {"Churn", "No Churn"}
    assert result["risk_level"] in {"Low", "Elevated", "High"}
    assert result["recommended_action"]
    assert result["customerID"] == "API-TEST-001"


@pytest.mark.skipif(not PIPELINE_PATH.exists(), reason="pipeline artifact missing")
def test_predict_churn_invalid_service_combo(client) -> None:
    payload = {
        "gender": "Female",
        "SeniorCitizen": 0,
        "Partner": "Yes",
        "Dependents": "No",
        "tenure": 1,
        "PhoneService": "No",
        "MultipleLines": "Yes",
        "InternetService": "No",
        "OnlineSecurity": "No internet service",
        "OnlineBackup": "No internet service",
        "DeviceProtection": "No internet service",
        "TechSupport": "No internet service",
        "StreamingTV": "No internet service",
        "StreamingMovies": "No internet service",
        "Contract": "Month-to-month",
        "PaperlessBilling": "Yes",
        "PaymentMethod": "Electronic check",
        "MonthlyCharges": 20.0,
        "TotalCharges": 20.0,
    }

    response = client.post("/predict_churn", json=payload)
    assert response.status_code == 422
