"""Tests for session-scoped upload training."""

from __future__ import annotations

import pandas as pd
import pytest

from src.dataset_schema import normalize_binary_target, validate_retrain_dataset
from src.training_service import train_and_score_upload


def _synthetic_training_df(rows: int = 120) -> pd.DataFrame:
    records = []
    for i in range(rows):
        churn = "Yes" if i % 4 == 0 else "No"
        records.append(
            {
                "customerID": f"C{i:04d}",
                "tenure": i % 72,
                "MonthlyCharges": 30 + (i % 50),
                "TotalCharges": float(30 + (i % 50)) * max(1, i % 24),
                "Contract": "Month-to-month" if i % 2 == 0 else "One year",
                "Churn": churn,
            }
        )
    return pd.DataFrame(records)


def test_normalize_binary_target_supports_zero_one() -> None:
    series = pd.Series([0, 1, 0, 1])
    normalized, positive = normalize_binary_target(series)
    assert positive == "Yes"
    assert normalized.tolist() == ["No", "Yes", "No", "Yes"]


def test_validate_retrain_dataset_requires_minimum_rows() -> None:
    df = _synthetic_training_df(rows=20)
    with pytest.raises(ValueError, match="at least"):
        validate_retrain_dataset(df, "Churn", id_col="customerID")


def test_train_and_score_upload_runs_on_synthetic_data() -> None:
    df = _synthetic_training_df(rows=120)
    result = train_and_score_upload(df, target_col="Churn", id_col="customerID")

    assert len(result.results) == len(df)
    assert "primary_key" in result.results.columns
    assert "churn_probability" in result.results.columns
    assert result.validation_metrics["precision"] >= 0
    assert result.test_metrics["recall"] >= 0
    assert result.bundle.metadata["training_mode"] == "upload_session"
