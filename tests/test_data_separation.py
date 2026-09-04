"""Tests for feature/target/ID separation."""

from pathlib import Path

import pandas as pd
import pytest

from src.data_cleaning import CLEANED_FILENAME, IDENTIFIER_COL, TARGET_COL, run_cleaning_pipeline
from src.data_separation import (
    CATEGORICAL_FEATURE_COLS,
    FEATURE_COLS,
    NUMERIC_FEATURE_COLS,
    load_cleaned_data,
    run_separation_pipeline,
    separate_features_target_id,
    validate_separation,
)

PROJECT_ROOT = Path(__file__).resolve().parents[1]
CLEANED_PATH = PROJECT_ROOT / "data" / "processed" / CLEANED_FILENAME


@pytest.fixture(scope="module")
def cleaned_df() -> pd.DataFrame:
    if not CLEANED_PATH.exists():
        run_cleaning_pipeline()
    return load_cleaned_data(CLEANED_PATH)


def test_feature_column_counts() -> None:
    assert len(FEATURE_COLS) == 19
    assert len(NUMERIC_FEATURE_COLS) == 4
    assert len(CATEGORICAL_FEATURE_COLS) == 15


def test_separate_features_target_id_excludes_id_and_target(cleaned_df: pd.DataFrame) -> None:
    separated = separate_features_target_id(cleaned_df)

    assert separated.X.shape == (7043, 19)
    assert IDENTIFIER_COL not in separated.X.columns
    assert TARGET_COL not in separated.X.columns
    assert len(separated.y) == 7043
    assert len(separated.customer_ids) == 7043
    assert separated.customer_ids.nunique() == 7043


def test_validate_separation_passes(cleaned_df: pd.DataFrame) -> None:
    separated = separate_features_target_id(cleaned_df)
    validation = validate_separation(cleaned_df, separated)

    assert validation["passed"] is True
    assert validation["churn_counts"] == {"No": 5174, "Yes": 1869}


def test_run_separation_pipeline_writes_manifest(tmp_path: Path, cleaned_df: pd.DataFrame) -> None:
    manifest_path = tmp_path / "column_roles.json"
    separated, manifest, validation = run_separation_pipeline(manifest_path=manifest_path)

    assert manifest_path.exists()
    assert separated.X.shape == (7043, 19)
    assert manifest["feature_count"] == 19
    assert validation["passed"] is True
