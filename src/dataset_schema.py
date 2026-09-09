"""Generic dataset schema helpers for the upload retrain fallback."""

from __future__ import annotations

from dataclasses import dataclass

import pandas as pd

MIN_TRAIN_ROWS = 100
MIN_CLASS_COUNT = 5


@dataclass(frozen=True)
class UploadDatasetSpec:
    """Resolved schema for a user-uploaded training dataset."""

    target_col: str
    id_col: str
    feature_cols: list[str]
    numeric_cols: list[str]
    categorical_cols: list[str]
    positive_class: str


def normalize_binary_target(series: pd.Series) -> tuple[pd.Series, str]:
    """
    Normalize a binary target column to Yes/No strings.

    Returns the normalized series and the positive class label used for churn.
    """
    values = series.copy()
    if values.dtype == bool:
        return values.map({True: "Yes", False: "No"}), "Yes"

    if pd.api.types.is_numeric_dtype(values):
        unique = sorted(values.dropna().unique().tolist())
        if unique == [0, 1] or unique == [0.0, 1.0]:
            return values.map({0: "No", 1: "Yes", 0.0: "No", 1.0: "Yes"}), "Yes"
        if len(unique) == 2:
            low, high = unique
            mapping = {low: "No", high: "Yes"}
            return values.map(mapping), "Yes"

    normalized = values.astype(str).str.strip().str.lower()
    mapping_candidates = {
        "yes": "Yes",
        "no": "No",
        "true": "Yes",
        "false": "No",
        "1": "Yes",
        "0": "No",
        "churn": "Yes",
        "no churn": "No",
        "not churn": "No",
    }
    mapped = normalized.map(mapping_candidates)
    if mapped.isna().any():
        unknown = sorted(normalized[mapped.isna()].unique().tolist())
        raise ValueError(f"Unsupported target values: {unknown}")

    return mapped, "Yes"


def infer_column_types(df: pd.DataFrame, feature_cols: list[str]) -> tuple[list[str], list[str]]:
    """Infer numeric vs categorical feature columns from an uploaded dataframe."""
    numeric_cols: list[str] = []
    categorical_cols: list[str] = []

    for col in feature_cols:
        series = df[col]
        if pd.api.types.is_numeric_dtype(series):
            numeric_cols.append(col)
        else:
            categorical_cols.append(col)

    return numeric_cols, categorical_cols


def validate_retrain_dataset(
    df: pd.DataFrame,
    target_col: str,
    id_col: str | None = None,
) -> UploadDatasetSpec:
    """Validate that an uploaded dataset can be used for session-scoped retraining."""
    if target_col not in df.columns:
        raise ValueError(f"Target column '{target_col}' not found.")

    if len(df) < MIN_TRAIN_ROWS:
        raise ValueError(f"Dataset must contain at least {MIN_TRAIN_ROWS} rows for retraining.")

    y, positive_class = normalize_binary_target(df[target_col])
    class_counts = y.value_counts()
    if len(class_counts) != 2:
        raise ValueError("Target column must contain exactly two classes.")

    for label, count in class_counts.items():
        if count < MIN_CLASS_COUNT:
            raise ValueError(
                f"Each target class must have at least {MIN_CLASS_COUNT} rows. "
                f"Class '{label}' has {count}."
            )

    exclude = {target_col}
    resolved_id_col = id_col or "__upload_row_id__"
    if id_col:
        if id_col not in df.columns:
            raise ValueError(f"ID column '{id_col}' not found.")
        exclude.add(id_col)
        if df[id_col].duplicated().any():
            raise ValueError(f"ID column '{id_col}' must be unique.")

    feature_cols = [col for col in df.columns if col not in exclude]
    if not feature_cols:
        raise ValueError("No feature columns found after excluding target and ID.")

    constant_cols = [col for col in feature_cols if df[col].nunique(dropna=False) <= 1]
    if constant_cols:
        raise ValueError(f"Remove constant feature columns: {constant_cols}")

    numeric_cols, categorical_cols = infer_column_types(df, feature_cols)
    if not numeric_cols and not categorical_cols:
        raise ValueError("Could not infer any usable feature columns.")

    return UploadDatasetSpec(
        target_col=target_col,
        id_col=resolved_id_col,
        feature_cols=feature_cols,
        numeric_cols=numeric_cols,
        categorical_cols=categorical_cols,
        positive_class=positive_class,
    )
