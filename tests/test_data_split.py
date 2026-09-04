"""Tests for train/validation/test splitting."""

from pathlib import Path

import pytest

from src.data_cleaning import IDENTIFIER_COL, TARGET_COL, run_cleaning_pipeline
from src.data_separation import load_cleaned_data, separate_features_target_id
from src.data_split import (
    RANDOM_STATE,
    load_split_from_manifest,
    run_split_pipeline,
    stratified_train_val_test_split,
    validate_split,
)

PROJECT_ROOT = Path(__file__).resolve().parents[1]
CLEANED_PATH = PROJECT_ROOT / "data" / "processed" / "cleaned_churn.csv"


@pytest.fixture(scope="module")
def separated_data():
    if not CLEANED_PATH.exists():
        run_cleaning_pipeline()
    df = load_cleaned_data(CLEANED_PATH)
    return separate_features_target_id(df)


def test_stratified_split_sizes(separated_data) -> None:
    split = stratified_train_val_test_split(
        separated_data.X,
        separated_data.y,
        separated_data.customer_ids,
        random_state=RANDOM_STATE,
    )
    validation = validate_split(split, total_rows=len(separated_data.X))

    assert validation["train_rows"] == 4930
    assert validation["val_rows"] == 1056
    assert validation["test_rows"] == 1057
    assert validation["passed"] is True


def test_no_overlap_between_partitions(separated_data) -> None:
    split = stratified_train_val_test_split(
        separated_data.X,
        separated_data.y,
        separated_data.customer_ids,
    )
    validation = validate_split(split, total_rows=len(separated_data.X))

    assert validation["row_index_overlap"] == 0
    assert validation["customer_id_overlap"] == 0


def test_features_exclude_id_and_target(separated_data) -> None:
    split = stratified_train_val_test_split(
        separated_data.X,
        separated_data.y,
        separated_data.customer_ids,
    )

    for frame in (split.X_train, split.X_val, split.X_test):
        assert IDENTIFIER_COL not in frame.columns
        assert TARGET_COL not in frame.columns
        assert frame.shape[1] == 19


def test_run_split_pipeline_and_reload(tmp_path: Path, separated_data) -> None:
    manifest_path = tmp_path / "split_manifest.json"
    split, manifest, validation = run_split_pipeline(manifest_path=manifest_path)

    assert manifest_path.exists()
    assert validation["passed"] is True
    assert split.X_train.shape == (4930, 19)

    reloaded = load_split_from_manifest(manifest_path=manifest_path)
    assert len(reloaded.X_train) == len(split.X_train)
    assert reloaded.id_test.equals(split.id_test)
