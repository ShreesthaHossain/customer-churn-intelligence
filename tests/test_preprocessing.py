"""Tests for leakage-safe preprocessing."""

from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from src.data_separation import CATEGORICAL_FEATURE_COLS, NUMERIC_FEATURE_COLS
from src.data_split import load_split_from_manifest, run_split_pipeline
from src.preprocessing import (
    ORIGINAL_FEATURE_COUNT,
    build_preprocessor,
    fit_preprocessor,
    fit_transform_train,
    run_preprocessing_pipeline,
    transform_features,
    transform_splits,
    validate_transformed_splits,
)

PROJECT_ROOT = Path(__file__).resolve().parents[1]
SPLIT_MANIFEST = PROJECT_ROOT / "data" / "processed" / "split_manifest.json"


@pytest.fixture(scope="module")
def split_data():
    if not SPLIT_MANIFEST.exists():
        run_split_pipeline()
    return load_split_from_manifest()


def test_build_preprocessor_column_groups() -> None:
    preprocessor = build_preprocessor()
    transformers = {name: cols for name, _, cols in preprocessor.transformers}

    assert transformers["num"] == NUMERIC_FEATURE_COLS
    assert transformers["cat"] == CATEGORICAL_FEATURE_COLS


def test_fit_only_on_train(split_data) -> None:
    preprocessor = build_preprocessor()
    fit_preprocessor(preprocessor, split_data.X_train)

    X_val = transform_features(preprocessor, split_data.X_val)
    assert X_val.shape == (len(split_data.X_val), 45)


def test_transformed_shapes_and_feature_count(split_data) -> None:
    transformed, _, validation = run_preprocessing_pipeline(split_data)

    assert ORIGINAL_FEATURE_COUNT == 19
    assert validation["transformed_feature_count"] == 45
    assert transformed.X_train.shape == (4930, 45)
    assert transformed.X_val.shape == (1056, 45)
    assert transformed.X_test.shape == (1057, 45)
    assert validation["passed"] is True


def test_no_nan_or_inf(split_data) -> None:
    transformed, _, validation = run_preprocessing_pipeline(split_data)

    for arr in (transformed.X_train, transformed.X_val, transformed.X_test):
        assert not np.isnan(arr).any()
        assert not np.isinf(arr).any()

    assert validation["train_nan"] is False
    assert validation["test_inf"] is False


def test_unseen_category_does_not_crash(split_data) -> None:
    preprocessor = build_preprocessor()
    _, _ = fit_transform_train(preprocessor, split_data.X_train)

    probe = split_data.X_val.iloc[[0]].copy()
    probe.loc[probe.index[0], CATEGORICAL_FEATURE_COLS[0]] = "__UNSEEN__"
    out = transform_features(preprocessor, probe)
    assert out.shape == (1, 45)


def test_feature_names_available(split_data) -> None:
    transformed, _, _ = run_preprocessing_pipeline(split_data)

    assert len(transformed.feature_names) == 45
    assert any(name.startswith("num__") for name in transformed.feature_names)
    assert any(name.startswith("cat__") for name in transformed.feature_names)
