"""Tests for data cleaning utilities."""

from pathlib import Path

import pandas as pd
import pytest

from src.data_cleaning import (
    CLEANED_FILENAME,
    RAW_FILENAME,
    clean_churn_data,
    load_raw_data,
    run_cleaning_pipeline,
    validate_cleaned_data,
)

PROJECT_ROOT = Path(__file__).resolve().parents[1]
RAW_PATH = PROJECT_ROOT / "data" / "raw" / RAW_FILENAME


@pytest.fixture
def raw_df() -> pd.DataFrame:
    if not RAW_PATH.exists():
        pytest.skip(f"Raw dataset not found at {RAW_PATH}")
    return load_raw_data(RAW_PATH)


def test_load_raw_data_shape(raw_df: pd.DataFrame) -> None:
    assert raw_df.shape == (7043, 21)


def test_clean_churn_data_preserves_rows_and_target(raw_df: pd.DataFrame) -> None:
    cleaned, audit = clean_churn_data(raw_df)

    assert cleaned.shape == (7043, 21)
    assert audit["total_charges_imputed_rows"] == 11
    assert cleaned["TotalCharges"].dtype == "float64"
    assert cleaned["TotalCharges"].isna().sum() == 0
    assert cleaned["Churn"].value_counts().to_dict() == raw_df["Churn"].value_counts().to_dict()


def test_validate_cleaned_data_passes(raw_df: pd.DataFrame) -> None:
    cleaned, _ = clean_churn_data(raw_df)
    validation = validate_cleaned_data(cleaned)

    assert validation["passed"] is True
    assert validation["missing_values_total"] == 0


def test_run_cleaning_pipeline_writes_output(tmp_path: Path, raw_df: pd.DataFrame) -> None:
    output_path = tmp_path / CLEANED_FILENAME
    cleaned, audit, validation = run_cleaning_pipeline(
        raw_path=RAW_PATH,
        processed_path=output_path,
    )

    assert output_path.exists()
    assert cleaned.shape == (7043, 21)
    assert audit["total_charges_imputed_rows"] == 11
    assert validation["passed"] is True
