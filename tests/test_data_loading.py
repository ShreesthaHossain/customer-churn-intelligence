"""Tests for unified data-loading helpers."""

from pathlib import Path

import pytest

from src.data_loading import (
    load_chosen_threshold,
    load_cleaned_churn_data,
    load_raw_churn_data,
    load_separated_data,
    load_train_val_split,
)

PROJECT_ROOT = Path(__file__).resolve().parents[1]
RAW_PATH = PROJECT_ROOT / "data" / "raw" / "WA_Fn-UseC_-Telco-Customer-Churn.csv"
CLEANED_PATH = PROJECT_ROOT / "data" / "processed" / "cleaned_churn.csv"
THRESHOLD_PATH = PROJECT_ROOT / "reports" / "chosen_threshold.json"


@pytest.mark.skipif(not RAW_PATH.exists(), reason="raw dataset missing")
def test_load_raw_churn_data() -> None:
    df = load_raw_churn_data()
    assert df.shape == (7043, 21)


@pytest.mark.skipif(not CLEANED_PATH.exists(), reason="cleaned dataset missing")
def test_load_cleaned_and_separated_data() -> None:
    cleaned = load_cleaned_churn_data()
    separated = load_separated_data()

    assert cleaned.shape[0] == separated.X.shape[0]
    assert separated.X.shape[1] == 19


@pytest.mark.skipif(not CLEANED_PATH.exists(), reason="cleaned dataset missing")
def test_load_train_val_split_shapes() -> None:
    split = load_train_val_split()

    assert len(split.X_train) == 4930
    assert len(split.X_val) == 1056
    assert len(split.X_test) == 1057


@pytest.mark.skipif(not THRESHOLD_PATH.exists(), reason="threshold config missing")
def test_load_chosen_threshold() -> None:
    config = load_chosen_threshold()
    assert config["threshold"] == 0.1
