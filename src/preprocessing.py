"""Leakage-safe preprocessing for churn prediction features."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

from src.data_separation import CATEGORICAL_FEATURE_COLS, NUMERIC_FEATURE_COLS
from src.data_split import SplitData, load_split_from_manifest

ORIGINAL_FEATURE_COUNT = len(NUMERIC_FEATURE_COLS) + len(CATEGORICAL_FEATURE_COLS)


@dataclass(frozen=True)
class TransformedSplits:
    """Container for transformed train/validation/test feature matrices."""

    X_train: np.ndarray
    X_val: np.ndarray
    X_test: np.ndarray
    feature_names: list[str]
    preprocessor: ColumnTransformer


def build_preprocessor(
    numeric_cols: list[str] | None = None,
    categorical_cols: list[str] | None = None,
) -> ColumnTransformer:
    """
    Build a ColumnTransformer for numeric scaling and categorical one-hot encoding.

    Imputers are included defensively; cleaned data should have no missing values.
    StandardScaler supports scale-sensitive models (e.g., Logistic Regression).
    OneHotEncoder(handle_unknown='ignore') supports safe inference on unseen categories.
    """
    numeric_cols = numeric_cols or NUMERIC_FEATURE_COLS
    categorical_cols = categorical_cols or CATEGORICAL_FEATURE_COLS

    numeric_pipeline = Pipeline(
        steps=[
            ("imputer", SimpleImputer(strategy="median")),
            ("scaler", StandardScaler()),
        ]
    )
    categorical_pipeline = Pipeline(
        steps=[
            ("imputer", SimpleImputer(strategy="most_frequent")),
            (
                "encoder",
                OneHotEncoder(handle_unknown="ignore", sparse_output=False),
            ),
        ]
    )
    return ColumnTransformer(
        transformers=[
            ("num", numeric_pipeline, numeric_cols),
            ("cat", categorical_pipeline, categorical_cols),
        ],
        remainder="drop",
    )


def feature_cols_from_preprocessor(preprocessor: ColumnTransformer) -> list[str]:
    """Return the input feature column list configured on a preprocessor."""
    cols: list[str] = []
    for _, _, col_list in preprocessor.transformers:
        cols.extend(col_list)
    return cols


def get_transformed_feature_names(preprocessor: ColumnTransformer) -> list[str]:
    """Return output feature names from a fitted preprocessor."""
    return preprocessor.get_feature_names_out().tolist()


def fit_preprocessor(preprocessor: ColumnTransformer, X_train: pd.DataFrame) -> ColumnTransformer:
    """Fit preprocessing on training features only (AGENTS.md rule 4)."""
    _validate_feature_frame(X_train, expected_cols=feature_cols_from_preprocessor(preprocessor))
    preprocessor.fit(X_train)
    return preprocessor


def transform_features(
    preprocessor: ColumnTransformer,
    X: pd.DataFrame,
    *,
    expected_cols: list[str] | None = None,
) -> np.ndarray:
    """Transform features without refitting."""
    cols = expected_cols or feature_cols_from_preprocessor(preprocessor)
    _validate_feature_frame(X, expected_cols=cols)
    return preprocessor.transform(X)


def fit_transform_train(
    preprocessor: ColumnTransformer,
    X_train: pd.DataFrame,
) -> tuple[ColumnTransformer, np.ndarray]:
    """Fit on training data and return transformed training matrix."""
    _validate_feature_frame(X_train, expected_cols=feature_cols_from_preprocessor(preprocessor))
    X_train_transformed = preprocessor.fit_transform(X_train)
    return preprocessor, X_train_transformed


def transform_splits(
    preprocessor: ColumnTransformer,
    split: SplitData,
) -> TransformedSplits:
    """
    Transform train/validation/test partitions using a fitted preprocessor.

    Validation and test sets use transform() only — never fit() or fit_transform().
    """
    if not hasattr(preprocessor, "transformers_"):
        raise ValueError("Preprocessor must be fitted on X_train before transforming splits.")

    X_train = transform_features(preprocessor, split.X_train)
    X_val = transform_features(preprocessor, split.X_val)
    X_test = transform_features(preprocessor, split.X_test)

    return TransformedSplits(
        X_train=X_train,
        X_val=X_val,
        X_test=X_test,
        feature_names=get_transformed_feature_names(preprocessor),
        preprocessor=preprocessor,
    )


def validate_transformed_splits(
    split: SplitData,
    transformed: TransformedSplits,
) -> dict:
    """Validate transformed matrices for shape consistency and numerical safety."""
    n_features = len(transformed.feature_names)
    feature_counts = {
        "train": transformed.X_train.shape[1],
        "validation": transformed.X_val.shape[1],
        "test": transformed.X_test.shape[1],
    }

    checks: dict[str, Any] = {
        "original_feature_count": ORIGINAL_FEATURE_COUNT,
        "transformed_feature_count": n_features,
        "feature_counts_consistent": len(set(feature_counts.values())) == 1,
        "train_rows": transformed.X_train.shape[0],
        "val_rows": transformed.X_val.shape[0],
        "test_rows": transformed.X_test.shape[0],
        "train_rows_preserved": transformed.X_train.shape[0] == len(split.X_train),
        "val_rows_preserved": transformed.X_val.shape[0] == len(split.X_val),
        "test_rows_preserved": transformed.X_test.shape[0] == len(split.X_test),
        "train_nan": bool(np.isnan(transformed.X_train).any()),
        "val_nan": bool(np.isnan(transformed.X_val).any()),
        "test_nan": bool(np.isnan(transformed.X_test).any()),
        "train_inf": bool(np.isinf(transformed.X_train).any()),
        "val_inf": bool(np.isinf(transformed.X_val).any()),
        "test_inf": bool(np.isinf(transformed.X_test).any()),
    }

    # Unseen category should not crash inference (all-zero/new category ignored by OHE)
    probe = split.X_val.iloc[[0]].copy()
    probe.loc[probe.index[0], "gender"] = "__UNSEEN_CATEGORY__"
    try:
        probe_out = transform_features(transformed.preprocessor, probe)
        checks["unseen_category_transform_ok"] = probe_out.shape == (1, n_features)
    except Exception as exc:  # pragma: no cover - explicit failure path
        checks["unseen_category_transform_ok"] = False
        checks["unseen_category_error"] = str(exc)

    checks["passed"] = (
        checks["feature_counts_consistent"]
        and checks["train_rows_preserved"]
        and checks["val_rows_preserved"]
        and checks["test_rows_preserved"]
        and not checks["train_nan"]
        and not checks["val_nan"]
        and not checks["test_nan"]
        and not checks["train_inf"]
        and not checks["val_inf"]
        and not checks["test_inf"]
        and checks["unseen_category_transform_ok"]
    )
    return checks


def run_preprocessing_pipeline(
    split: SplitData | None = None,
) -> tuple[TransformedSplits, ColumnTransformer, dict]:
    """
    Load splits, fit preprocessor on X_train, transform all partitions, validate.

    Does not persist transformed arrays or fitted preprocessor to disk.
    """
    split = split or load_split_from_manifest()
    preprocessor = build_preprocessor()
    preprocessor, _ = fit_transform_train(preprocessor, split.X_train)
    transformed = transform_splits(preprocessor, split)
    validation = validate_transformed_splits(split, transformed)
    if not validation["passed"]:
        raise ValueError(f"Preprocessing validation failed: {validation}")
    return transformed, preprocessor, validation


def _validate_feature_frame(X: pd.DataFrame, expected_cols: list[str] | None = None) -> None:
    """Ensure input contains only approved predictive columns."""
    expected = set(expected_cols or (NUMERIC_FEATURE_COLS + CATEGORICAL_FEATURE_COLS))
    actual = set(X.columns)
    if actual != expected:
        missing = expected - actual
        extra = actual - expected
        raise ValueError(f"Feature frame mismatch. Missing={missing}, Extra={extra}")
