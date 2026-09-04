"""Feature, target, and identifier separation for the churn dataset."""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path

import pandas as pd

from src.data_cleaning import CLEANED_FILENAME, IDENTIFIER_COL, TARGET_COL

# Explicit feature lists derived from cleaned schema (EDA + cleaning steps).
# TotalCharges is numeric after cleaning; SeniorCitizen remains int64 0/1.
NUMERIC_FEATURE_COLS: list[str] = [
    "SeniorCitizen",
    "tenure",
    "MonthlyCharges",
    "TotalCharges",
]

CATEGORICAL_FEATURE_COLS: list[str] = [
    "gender",
    "Partner",
    "Dependents",
    "PhoneService",
    "MultipleLines",
    "InternetService",
    "OnlineSecurity",
    "OnlineBackup",
    "DeviceProtection",
    "TechSupport",
    "StreamingTV",
    "StreamingMovies",
    "Contract",
    "PaperlessBilling",
    "PaymentMethod",
]

FEATURE_COLS: list[str] = NUMERIC_FEATURE_COLS + CATEGORICAL_FEATURE_COLS

COLUMN_ROLES_FILENAME = "column_roles.json"


@dataclass(frozen=True)
class SeparatedData:
    """Container for feature matrix, target, and customer identifiers."""

    X: pd.DataFrame
    y: pd.Series
    customer_ids: pd.Series
    numeric_features: list[str]
    categorical_features: list[str]


def load_cleaned_data(processed_path: Path | str | None = None) -> pd.DataFrame:
    """Load the cleaned dataset from data/processed/."""
    if processed_path is None:
        project_root = Path(__file__).resolve().parents[1]
        processed_path = project_root / "data" / "processed" / CLEANED_FILENAME

    path = Path(processed_path)
    if not path.exists():
        raise FileNotFoundError(
            f"Cleaned dataset not found at {path.resolve()}. "
            "Run the data cleaning step first."
        )
    return pd.read_csv(path)


def _validate_input_columns(df: pd.DataFrame) -> None:
    """Ensure the dataframe matches the expected cleaned schema."""
    expected_cols = [IDENTIFIER_COL] + FEATURE_COLS + [TARGET_COL]
    missing = [col for col in expected_cols if col not in df.columns]
    extra = [col for col in df.columns if col not in expected_cols]

    if missing:
        raise ValueError(f"Missing expected columns: {missing}")
    if extra:
        raise ValueError(f"Unexpected columns not in schema: {extra}")


def separate_features_target_id(df: pd.DataFrame) -> SeparatedData:
    """
    Split cleaned data into features (X), target (y), and customer IDs.

    customerID is excluded from X per AGENTS.md — it is kept only for linking
    predictions back to customers.
    """
    _validate_input_columns(df)

    if df[IDENTIFIER_COL].nunique() != len(df):
        raise ValueError("customerID must be unique before separation.")

    X = df[FEATURE_COLS].copy()
    y = df[TARGET_COL].copy()
    customer_ids = df[IDENTIFIER_COL].copy()

    return SeparatedData(
        X=X,
        y=y,
        customer_ids=customer_ids,
        numeric_features=NUMERIC_FEATURE_COLS.copy(),
        categorical_features=CATEGORICAL_FEATURE_COLS.copy(),
    )


def validate_separation(df: pd.DataFrame, separated: SeparatedData) -> dict:
    """Validate that separation preserved rows and excluded ID/target from X."""
    checks = {
        "row_count": len(df),
        "feature_count": separated.X.shape[1],
        "numeric_feature_count": len(separated.numeric_features),
        "categorical_feature_count": len(separated.categorical_features),
        "identifier_in_features": IDENTIFIER_COL in separated.X.columns,
        "target_in_features": TARGET_COL in separated.X.columns,
        "index_aligned": (
            len(separated.X) == len(separated.y) == len(separated.customer_ids) == len(df)
        ),
        "customer_id_unique": int(separated.customer_ids.nunique()) == len(df),
        "target_values": sorted(separated.y.unique().tolist()),
        "churn_counts": separated.y.value_counts().to_dict(),
    }
    checks["passed"] = (
        checks["feature_count"] == 19
        and checks["numeric_feature_count"] == 4
        and checks["categorical_feature_count"] == 15
        and not checks["identifier_in_features"]
        and not checks["target_in_features"]
        and checks["index_aligned"]
        and checks["customer_id_unique"]
        and checks["target_values"] == ["No", "Yes"]
    )
    return checks


def build_column_roles_manifest(separated: SeparatedData) -> dict:
    """Build a JSON-serializable manifest of column roles for downstream steps."""
    return {
        "identifier_col": IDENTIFIER_COL,
        "target_col": TARGET_COL,
        "feature_cols": FEATURE_COLS,
        "numeric_feature_cols": separated.numeric_features,
        "categorical_feature_cols": separated.categorical_features,
        "feature_count": len(FEATURE_COLS),
        "notes": {
            "customerID": "Linkage key only — never include in model features.",
            "Churn": "Binary target (Yes/No); encoding deferred to preprocessing.",
            "SeniorCitizen": "Stored as int64 0/1 in cleaned data.",
            "sentinel_categories": (
                "Several service columns contain 'No internet service' or "
                "'No phone service'; encoding deferred to preprocessing."
            ),
        },
    }


def save_column_roles_manifest(manifest: dict, processed_path: Path | str | None = None) -> Path:
    """Persist column role definitions to data/processed/."""
    if processed_path is None:
        project_root = Path(__file__).resolve().parents[1]
        processed_path = project_root / "data" / "processed" / COLUMN_ROLES_FILENAME

    path = Path(processed_path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    return path


def run_separation_pipeline(
    processed_path: Path | str | None = None,
    manifest_path: Path | str | None = None,
) -> tuple[SeparatedData, dict, dict]:
    """Load cleaned data, separate columns, validate, and save column roles."""
    df = load_cleaned_data(processed_path)
    separated = separate_features_target_id(df)
    validation = validate_separation(df, separated)
    if not validation["passed"]:
        raise ValueError(f"Feature/target/ID separation failed validation: {validation}")

    manifest = build_column_roles_manifest(separated)
    manifest_output = save_column_roles_manifest(manifest, manifest_path)
    manifest["output_path"] = str(manifest_output.resolve())
    return separated, manifest, validation
