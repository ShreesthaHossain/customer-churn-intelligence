"""Tests for batch FastAPI churn prediction routes."""

from __future__ import annotations

import io
from pathlib import Path

import pytest
from fastapi.testclient import TestClient

from src.data_cleaning import IDENTIFIER_COL, TARGET_COL
from src.data_split import load_split_from_manifest
from src.inference import build_batch_api_response, score_uploaded_batch
from src.model import load_churn_pipeline

PROJECT_ROOT = Path(__file__).resolve().parents[1]
PIPELINE_PATH = PROJECT_ROOT / "models" / "churn_pipeline.joblib"


@pytest.fixture(scope="module")
def validation_batch_sample() -> tuple[list[dict], "pd.DataFrame"]:
    import pandas as pd

    split = load_split_from_manifest()
    sample = split.X_val.iloc[:5].copy()
    sample[IDENTIFIER_COL] = split.id_val.iloc[:5].astype(str).tolist()
    sample[TARGET_COL] = split.y_val.iloc[:5].tolist()
    records = sample.drop(columns=[TARGET_COL]).to_dict(orient="records")
    return records, sample


@pytest.mark.skipif(not PIPELINE_PATH.exists(), reason="pipeline artifact missing")
def test_predict_churn_batch_json(client, validation_batch_sample) -> None:
    records, sample_df = validation_batch_sample
    response = client.post(
        "/predict_churn_batch",
        json={"primary_key_column": IDENTIFIER_COL, "customers": records},
    )
    assert response.status_code == 200
    payload = response.json()
    assert payload["customers_scored"] == 5
    assert len(payload["results"]) == 5
    assert payload["results"][0]["recommended_action"]
    assert payload["results"][0]["primary_key"]

    pipeline = load_churn_pipeline()
    expected = build_batch_api_response(score_uploaded_batch(pipeline, sample_df, IDENTIFIER_COL))
    assert payload["customers_scored"] == expected["customers_scored"]
    assert payload["retention_outreach_flagged"] == expected["retention_outreach_flagged"]


@pytest.mark.skipif(not PIPELINE_PATH.exists(), reason="pipeline artifact missing")
def test_predict_churn_batch_csv_upload(client, validation_batch_sample) -> None:
    _, sample_df = validation_batch_sample
    csv_bytes = sample_df.to_csv(index=False).encode("utf-8")
    response = client.post(
        "/predict_churn_batch/file",
        params={"primary_key_column": IDENTIFIER_COL},
        files={"file": ("batch.csv", io.BytesIO(csv_bytes), "text/csv")},
    )
    assert response.status_code == 200
    payload = response.json()
    assert payload["customers_scored"] == 5


@pytest.mark.skipif(not PIPELINE_PATH.exists(), reason="pipeline artifact missing")
def test_monitoring_summary_endpoint(client) -> None:
    response = client.get("/monitoring/summary")
    assert response.status_code == 200
    payload = response.json()
    assert "service_metrics" in payload
    assert payload["batch_max_rows"] >= 1


@pytest.mark.skipif(not PIPELINE_PATH.exists(), reason="pipeline artifact missing")
def test_health_includes_service_metrics(client) -> None:
    response = client.get("/health")
    assert response.status_code == 200
    payload = response.json()
    assert "predictions_total" in payload
    assert "batch_requests_total" in payload
