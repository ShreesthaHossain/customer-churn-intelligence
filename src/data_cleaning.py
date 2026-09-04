"""Data cleaning utilities for the Telco Customer Churn dataset."""

from __future__ import annotations

from pathlib import Path

import pandas as pd

RAW_FILENAME = "WA_Fn-UseC_-Telco-Customer-Churn.csv"
CLEANED_FILENAME = "cleaned_churn.csv"

IDENTIFIER_COL = "customerID"
TARGET_COL = "Churn"
TOTAL_CHARGES_COL = "TotalCharges"
TENURE_COL = "tenure"


def load_raw_data(raw_path: Path | str) -> pd.DataFrame:
    """Load the raw CSV without modifying the source file."""
    path = Path(raw_path)
    if not path.exists():
        raise FileNotFoundError(
            f"Raw dataset not found at {path.resolve()}. "
            f"Place {RAW_FILENAME} in data/raw/."
        )
    return pd.read_csv(path)


def _fix_total_charges(df: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    """
    Coerce TotalCharges to numeric and impute blank values for new customers.

    EDA showed 11 blank strings, all with tenure=0. Those customers have not
    accumulated charges yet, so 0.0 is the correct business value (not a
    learned imputation from other rows).
    """
    cleaned = df.copy()
    cleaned[TOTAL_CHARGES_COL] = cleaned[TOTAL_CHARGES_COL].astype(str).str.strip()
    blank_mask = cleaned[TOTAL_CHARGES_COL].eq("")

    coerced = pd.to_numeric(cleaned[TOTAL_CHARGES_COL], errors="coerce")
    non_numeric_mask = coerced.isna()

    if non_numeric_mask.any() and not blank_mask.equals(non_numeric_mask):
        bad_rows = cleaned.loc[non_numeric_mask & ~blank_mask, [IDENTIFIER_COL, TENURE_COL, TOTAL_CHARGES_COL]]
        raise ValueError(
            "Unexpected non-numeric TotalCharges values that are not blank strings:\n"
            f"{bad_rows.to_string(index=False)}"
        )

    imputed_mask = blank_mask & (cleaned[TENURE_COL] == 0)
    unexpected_blank_mask = blank_mask & (cleaned[TENURE_COL] != 0)
    if unexpected_blank_mask.any():
        bad_rows = cleaned.loc[unexpected_blank_mask, [IDENTIFIER_COL, TENURE_COL, TOTAL_CHARGES_COL]]
        raise ValueError(
            "Blank TotalCharges found for customers with tenure > 0:\n"
            f"{bad_rows.to_string(index=False)}"
        )

    cleaned.loc[imputed_mask, TOTAL_CHARGES_COL] = "0"
    cleaned[TOTAL_CHARGES_COL] = pd.to_numeric(cleaned[TOTAL_CHARGES_COL])

    audit = pd.DataFrame(
        {
            "customerID": cleaned.loc[imputed_mask, IDENTIFIER_COL],
            "tenure": cleaned.loc[imputed_mask, TENURE_COL],
            "MonthlyCharges": cleaned.loc[imputed_mask, "MonthlyCharges"],
            "TotalCharges_before": "",
            "TotalCharges_after": cleaned.loc[imputed_mask, TOTAL_CHARGES_COL],
            "action": "imputed_zero_for_new_customer",
        }
    )
    return cleaned, audit.reset_index(drop=True)


def clean_churn_data(df: pd.DataFrame) -> tuple[pd.DataFrame, dict]:
    """
    Apply deterministic data-quality fixes identified during EDA.

    Returns the cleaned dataframe and an audit dictionary describing changes.
    """
    original_shape = df.shape
    original_churn_counts = df[TARGET_COL].value_counts().to_dict()

    if df.duplicated().sum() > 0:
        raise ValueError("Duplicate rows detected; resolve before cleaning.")

    if df[IDENTIFIER_COL].nunique() != len(df):
        raise ValueError("customerID is not unique; resolve before cleaning.")

    cleaned, total_charges_audit = _fix_total_charges(df)

    if cleaned[TOTAL_CHARGES_COL].isna().any():
        raise ValueError("TotalCharges still contains missing values after cleaning.")

    if cleaned.shape != original_shape:
        raise ValueError("Cleaning changed row/column count; only dtype fixes are allowed.")

    if cleaned[TARGET_COL].value_counts().to_dict() != original_churn_counts:
        raise ValueError("Cleaning changed target distribution; no rows should be dropped.")

    audit = {
        "rows_before": original_shape[0],
        "rows_after": cleaned.shape[0],
        "columns_before": original_shape[1],
        "columns_after": cleaned.shape[1],
        "total_charges_imputed_rows": len(total_charges_audit),
        "total_charges_imputation_audit": total_charges_audit,
        "churn_distribution_unchanged": True,
    }
    return cleaned, audit


def validate_cleaned_data(df: pd.DataFrame) -> dict:
    """Run post-cleaning validation checks."""
    checks = {
        "row_count": len(df),
        "column_count": df.shape[1],
        "duplicate_rows": int(df.duplicated().sum()),
        "duplicate_customer_ids": int(df[IDENTIFIER_COL].duplicated().sum()),
        "missing_values_total": int(df.isnull().sum().sum()),
        "total_charges_dtype": str(df[TOTAL_CHARGES_COL].dtype),
        "total_charges_missing": int(df[TOTAL_CHARGES_COL].isna().sum()),
        "churn_counts": df[TARGET_COL].value_counts().to_dict(),
    }
    checks["passed"] = (
        checks["duplicate_rows"] == 0
        and checks["duplicate_customer_ids"] == 0
        and checks["missing_values_total"] == 0
        and checks["total_charges_dtype"] == "float64"
        and checks["total_charges_missing"] == 0
    )
    return checks


def save_cleaned_data(df: pd.DataFrame, processed_path: Path | str) -> Path:
    """Persist cleaned data to data/processed/."""
    path = Path(processed_path)
    path.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(path, index=False)
    return path


def run_cleaning_pipeline(
    raw_path: Path | str | None = None,
    processed_path: Path | str | None = None,
) -> tuple[pd.DataFrame, dict, dict]:
    """
    Load raw data, clean it, validate, and save to processed storage.

    Default paths are relative to the project root.
    """
    project_root = Path(__file__).resolve().parents[1]
    raw_path = Path(raw_path) if raw_path else project_root / "data" / "raw" / RAW_FILENAME
    processed_path = (
        Path(processed_path)
        if processed_path
        else project_root / "data" / "processed" / CLEANED_FILENAME
    )

    raw_df = load_raw_data(raw_path)
    cleaned_df, cleaning_audit = clean_churn_data(raw_df)
    validation = validate_cleaned_data(cleaned_df)
    if not validation["passed"]:
        raise ValueError(f"Cleaned data failed validation: {validation}")

    save_cleaned_data(cleaned_df, processed_path)
    cleaning_audit["output_path"] = str(processed_path.resolve())
    return cleaned_df, cleaning_audit, validation
