"""Unified data-loading entry points for notebooks, training, and inference."""

from __future__ import annotations

from pathlib import Path

import pandas as pd

from src.config import data_processed_dir, data_raw_dir, load_chosen_threshold_config
from src.data_cleaning import RAW_FILENAME, load_raw_data, run_cleaning_pipeline
from src.data_separation import (
    CLEANED_FILENAME,
    FEATURE_COLS,
    SeparatedData,
    load_cleaned_data,
    run_separation_pipeline,
    separate_features_target_id,
)
from src.data_split import SplitData, load_split_from_manifest, run_split_pipeline

__all__ = [
    "FEATURE_COLS",
    "SeparatedData",
    "SplitData",
    "load_chosen_threshold",
    "load_cleaned_churn_data",
    "load_raw_churn_data",
    "load_separated_data",
    "load_train_val_split",
    "run_cleaning_pipeline",
    "run_separation_pipeline",
    "run_split_pipeline",
]


def load_raw_churn_data(raw_path: Path | str | None = None) -> pd.DataFrame:
    """Load the raw Telco churn CSV from data/raw/."""
    path = Path(raw_path) if raw_path else data_raw_dir() / RAW_FILENAME
    return load_raw_data(path)


def load_cleaned_churn_data(processed_path: Path | str | None = None) -> pd.DataFrame:
    """Load the cleaned dataset from data/processed/."""
    path = Path(processed_path) if processed_path else data_processed_dir() / CLEANED_FILENAME
    return load_cleaned_data(path)


def load_separated_data(processed_path: Path | str | None = None) -> SeparatedData:
    """Load cleaned data and separate features, target, and customer IDs."""
    return separate_features_target_id(load_cleaned_churn_data(processed_path))


def load_train_val_split(
    manifest_path: Path | str | None = None,
    cleaned_path: Path | str | None = None,
) -> SplitData:
    """
    Recreate the stratified train/validation/test split from saved indices.

    Callers that must avoid the test set should use only X_train/X_val partitions.
    """
    return load_split_from_manifest(manifest_path, cleaned_path)


def load_chosen_threshold(path: Path | str | None = None) -> dict:
    """Load the frozen retention decision threshold and related metadata."""
    return load_chosen_threshold_config(path)
