"""Stratified train/validation/test splitting for the churn dataset."""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path

import pandas as pd
from sklearn.model_selection import train_test_split

from src.data_cleaning import IDENTIFIER_COL, TARGET_COL
from src.data_separation import (
    SeparatedData,
    load_cleaned_data,
    separate_features_target_id,
)

RANDOM_STATE = 42
TRAIN_SIZE = 0.70
HOLDOUT_SIZE = 0.30  # validation + test
VAL_TEST_RATIO = 0.50  # split holdout evenly into 15% / 15%

SPLIT_MANIFEST_FILENAME = "split_manifest.json"


@dataclass(frozen=True)
class SplitData:
    """Container for stratified train/validation/test partitions."""

    X_train: pd.DataFrame
    X_val: pd.DataFrame
    X_test: pd.DataFrame
    y_train: pd.Series
    y_val: pd.Series
    y_test: pd.Series
    id_train: pd.Series
    id_val: pd.Series
    id_test: pd.Series
    train_indices: list[int]
    val_indices: list[int]
    test_indices: list[int]


def stratified_train_val_test_split(
    X: pd.DataFrame,
    y: pd.Series,
    customer_ids: pd.Series,
    *,
    random_state: int = RANDOM_STATE,
    train_size: float = TRAIN_SIZE,
    holdout_size: float = HOLDOUT_SIZE,
    val_test_ratio: float = VAL_TEST_RATIO,
) -> SplitData:
    """
    Create a stratified 70/15/15 split in two stages.

    Stage 1: train (70%) vs temporary holdout (30%)
    Stage 2: holdout -> validation (15%) and test (15%)
    """
    if not (0 < train_size < 1 and 0 < holdout_size < 1 and 0 < val_test_ratio < 1):
        raise ValueError("Split proportions must be between 0 and 1.")

    expected_holdout = round(len(X) * holdout_size)
    if train_size + holdout_size > 1 + 1e-9:
        raise ValueError("train_size + holdout_size must not exceed 1.")

    X_train, X_temp, y_train, y_temp, id_train, id_temp = train_test_split(
        X,
        y,
        customer_ids,
        test_size=holdout_size,
        random_state=random_state,
        stratify=y,
    )

    X_val, X_test, y_val, y_test, id_val, id_test = train_test_split(
        X_temp,
        y_temp,
        id_temp,
        test_size=val_test_ratio,
        random_state=random_state,
        stratify=y_temp,
    )

    return SplitData(
        X_train=X_train.reset_index(drop=True),
        X_val=X_val.reset_index(drop=True),
        X_test=X_test.reset_index(drop=True),
        y_train=y_train.reset_index(drop=True),
        y_val=y_val.reset_index(drop=True),
        y_test=y_test.reset_index(drop=True),
        id_train=id_train.reset_index(drop=True),
        id_val=id_val.reset_index(drop=True),
        id_test=id_test.reset_index(drop=True),
        train_indices=X_train.index.tolist(),
        val_indices=X_val.index.tolist(),
        test_indices=X_test.index.tolist(),
    )


def _churn_summary(y: pd.Series) -> dict:
    counts = y.value_counts().to_dict()
    yes_pct = round(float((y == "Yes").mean() * 100), 2)
    return {"count": int(len(y)), "churn_no": int(counts.get("No", 0)), "churn_yes": int(counts.get("Yes", 0)), "churn_yes_pct": yes_pct}


def validate_split(split: SplitData, total_rows: int) -> dict:
    """Verify partition sizes, stratification, and non-overlapping IDs/indices."""
    train_ids = set(split.id_train)
    val_ids = set(split.id_val)
    test_ids = set(split.id_test)

    train_idx = set(split.train_indices)
    val_idx = set(split.val_indices)
    test_idx = set(split.test_indices)

    row_overlap = len(train_idx & val_idx) + len(train_idx & test_idx) + len(val_idx & test_idx)
    id_overlap = len(train_ids & val_ids) + len(train_ids & test_ids) + len(val_ids & test_ids)

    partition_rows = len(split.X_train) + len(split.X_val) + len(split.X_test)

    checks = {
        "total_rows": total_rows,
        "partition_rows": partition_rows,
        "train_rows": len(split.X_train),
        "val_rows": len(split.X_val),
        "test_rows": len(split.X_test),
        "train_pct": round(len(split.X_train) / total_rows * 100, 2),
        "val_pct": round(len(split.X_val) / total_rows * 100, 2),
        "test_pct": round(len(split.X_test) / total_rows * 100, 2),
        "row_index_overlap": row_overlap,
        "customer_id_overlap": id_overlap,
        "identifier_in_train_features": IDENTIFIER_COL in split.X_train.columns,
        "identifier_in_val_features": IDENTIFIER_COL in split.X_val.columns,
        "identifier_in_test_features": IDENTIFIER_COL in split.X_test.columns,
        "target_in_train_features": TARGET_COL in split.X_train.columns,
        "churn_summary": {
            "train": _churn_summary(split.y_train),
            "validation": _churn_summary(split.y_val),
            "test": _churn_summary(split.y_test),
        },
    }

    churn_pcts = [
        checks["churn_summary"]["train"]["churn_yes_pct"],
        checks["churn_summary"]["validation"]["churn_yes_pct"],
        checks["churn_summary"]["test"]["churn_yes_pct"],
    ]
    checks["max_churn_pct_spread"] = round(max(churn_pcts) - min(churn_pcts), 2)

    checks["passed"] = (
        partition_rows == total_rows
        and row_overlap == 0
        and id_overlap == 0
        and not checks["identifier_in_train_features"]
        and not checks["identifier_in_val_features"]
        and not checks["identifier_in_test_features"]
        and not checks["target_in_train_features"]
        and checks["max_churn_pct_spread"] <= 0.10
    )
    return checks


def build_split_manifest(split: SplitData, validation: dict) -> dict:
    """
    Build a lightweight reproducibility manifest.

    Full feature CSVs are not saved — splits can be recreated from
    cleaned_churn.csv plus these indices and random_state.
    """
    return {
        "random_state": RANDOM_STATE,
        "split_strategy": "stratified_two_stage",
        "proportions": {
            "train": TRAIN_SIZE,
            "validation": round((1 - TRAIN_SIZE) * VAL_TEST_RATIO, 4),
            "test": round((1 - TRAIN_SIZE) * (1 - VAL_TEST_RATIO), 4),
        },
        "train_indices": split.train_indices,
        "validation_indices": split.val_indices,
        "test_indices": split.test_indices,
        "validation_checks": validation,
        "notes": {
            "features": "Raw cleaned features only — no scaling, encoding, or SMOTE.",
            "test_set_policy": "Final test set remains untouched until model and threshold are frozen.",
            "csv_duplication": "Split CSVs not saved; use this manifest with cleaned_churn.csv.",
        },
    }


def save_split_manifest(manifest: dict, processed_path: Path | str | None = None) -> Path:
    """Persist split indices and metadata to data/processed/."""
    if processed_path is None:
        project_root = Path(__file__).resolve().parents[1]
        processed_path = project_root / "data" / "processed" / SPLIT_MANIFEST_FILENAME

    path = Path(processed_path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    return path


def load_split_from_manifest(
    manifest_path: Path | str | None = None,
    cleaned_path: Path | str | None = None,
) -> SplitData:
    """Recreate split partitions from cleaned data and a saved manifest."""
    if manifest_path is None:
        project_root = Path(__file__).resolve().parents[1]
        manifest_path = project_root / "data" / "processed" / SPLIT_MANIFEST_FILENAME

    manifest = json.loads(Path(manifest_path).read_text(encoding="utf-8"))
    separated = separate_features_target_id(load_cleaned_data(cleaned_path))

    def _subset(indices: list[int], component: str) -> pd.DataFrame | pd.Series:
        subset = getattr(separated, component).loc[indices]
        return subset.reset_index(drop=True)

    return SplitData(
        X_train=_subset(manifest["train_indices"], "X"),
        X_val=_subset(manifest["validation_indices"], "X"),
        X_test=_subset(manifest["test_indices"], "X"),
        y_train=_subset(manifest["train_indices"], "y"),
        y_val=_subset(manifest["validation_indices"], "y"),
        y_test=_subset(manifest["test_indices"], "y"),
        id_train=_subset(manifest["train_indices"], "customer_ids"),
        id_val=_subset(manifest["validation_indices"], "customer_ids"),
        id_test=_subset(manifest["test_indices"], "customer_ids"),
        train_indices=manifest["train_indices"],
        val_indices=manifest["validation_indices"],
        test_indices=manifest["test_indices"],
    )


def run_split_pipeline(
    manifest_path: Path | str | None = None,
) -> tuple[SplitData, dict, dict]:
    """Load cleaned data, create stratified splits, validate, and save manifest."""
    df = load_cleaned_data()
    separated: SeparatedData = separate_features_target_id(df)

    split = stratified_train_val_test_split(
        separated.X,
        separated.y,
        separated.customer_ids,
        random_state=RANDOM_STATE,
    )
    validation = validate_split(split, total_rows=len(df))
    if not validation["passed"]:
        raise ValueError(f"Split validation failed: {validation}")

    manifest = build_split_manifest(split, validation)
    output_path = save_split_manifest(manifest, manifest_path)
    manifest["output_path"] = str(output_path.resolve())
    return split, manifest, validation
