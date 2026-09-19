"""Tests for monitoring helpers."""

from __future__ import annotations

import pandas as pd

from src.monitoring import ServiceMetrics, compute_drift_report


def test_service_metrics_record_predictions() -> None:
    metrics = ServiceMetrics()
    metrics.record_predictions(3, batch=True)
    metrics.record_predictions(1, batch=False)

    snapshot = metrics.as_dict()
    assert snapshot["predictions_total"] == 4
    assert snapshot["batch_requests_total"] == 1
    assert snapshot["last_prediction_at"] is not None


def test_compute_drift_report_on_identical_sample() -> None:
    sample = pd.DataFrame(
        {
            "tenure": [1, 2, 3, 4, 5],
            "MonthlyCharges": [20.0, 30.0, 40.0, 50.0, 60.0],
            "TotalCharges": [20.0, 60.0, 120.0, 200.0, 300.0],
            "SeniorCitizen": [0, 0, 1, 0, 0],
            "Contract": ["Month-to-month"] * 5,
        }
    )
    report = compute_drift_report(sample)
    assert report["rows_compared"] == 5
    assert report["status"] in {"ok", "alert"}
    assert "numeric_drift" in report
