"""Validate uploaded CSV files against the frozen Telco churn model schema."""

from __future__ import annotations

import difflib
import re
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import pandas as pd

from src.data_cleaning import IDENTIFIER_COL, TARGET_COL, TENURE_COL, TOTAL_CHARGES_COL
from src.data_separation import CATEGORICAL_FEATURE_COLS, FEATURE_COLS, NUMERIC_FEATURE_COLS
from src.policy import INTERNET_DEPENDENT_FIELDS, NO_INTERNET_SERVICE, NO_PHONE_SERVICE

TEMPLATE_FILENAME = "telco_scoring_template.csv"

ALLOWED_CATEGORICAL_VALUES: dict[str, list[str]] = {
    "gender": ["Female", "Male"],
    "Partner": ["Yes", "No"],
    "Dependents": ["Yes", "No"],
    "PhoneService": ["Yes", "No"],
    "MultipleLines": ["Yes", "No", NO_PHONE_SERVICE],
    "InternetService": ["No", "DSL", "Fiber optic"],
    "OnlineSecurity": ["Yes", "No", NO_INTERNET_SERVICE],
    "OnlineBackup": ["Yes", "No", NO_INTERNET_SERVICE],
    "DeviceProtection": ["Yes", "No", NO_INTERNET_SERVICE],
    "TechSupport": ["Yes", "No", NO_INTERNET_SERVICE],
    "StreamingTV": ["Yes", "No", NO_INTERNET_SERVICE],
    "StreamingMovies": ["Yes", "No", NO_INTERNET_SERVICE],
    "Contract": ["Month-to-month", "One year", "Two year"],
    "PaperlessBilling": ["Yes", "No"],
    "PaymentMethod": [
        "Electronic check",
        "Mailed check",
        "Bank transfer (automatic)",
        "Credit card (automatic)",
    ],
}

_COLUMN_ALIASES: dict[str, str] = {
    "customer_id": IDENTIFIER_COL,
    "customerid": IDENTIFIER_COL,
    "id": IDENTIFIER_COL,
    "senior_citizen": "SeniorCitizen",
    "seniorcitizen": "SeniorCitizen",
    "monthly_charges": "MonthlyCharges",
    "monthlycharges": "MonthlyCharges",
    "total_charges": TOTAL_CHARGES_COL,
    "totalcharges": TOTAL_CHARGES_COL,
    "phone_service": "PhoneService",
    "phoneservice": "PhoneService",
    "multiple_lines": "MultipleLines",
    "multiplelines": "MultipleLines",
    "internet_service": "InternetService",
    "internetservice": "InternetService",
    "online_security": "OnlineSecurity",
    "onlinesecurity": "OnlineSecurity",
    "online_backup": "OnlineBackup",
    "onlinebackup": "OnlineBackup",
    "device_protection": "DeviceProtection",
    "deviceprotection": "DeviceProtection",
    "tech_support": "TechSupport",
    "techsupport": "TechSupport",
    "streaming_tv": "StreamingTV",
    "streamingtv": "StreamingTV",
    "streaming_movies": "StreamingMovies",
    "streamingmovies": "StreamingMovies",
    "paperless_billing": "PaperlessBilling",
    "paperlessbilling": "PaperlessBilling",
    "payment_method": "PaymentMethod",
    "paymentmethod": "PaymentMethod",
    "churn": TARGET_COL,
}


@dataclass
class CompatibilityReport:
    """Structured result of an upload compatibility check."""

    is_compatible: bool
    primary_key_col: str | None = None
    column_mapping: dict[str, str] = field(default_factory=dict)
    missing_features: list[str] = field(default_factory=list)
    extra_columns: list[str] = field(default_factory=list)
    rename_suggestions: dict[str, str] = field(default_factory=dict)
    dtype_issues: list[str] = field(default_factory=list)
    invalid_categorical_values: dict[str, list[str]] = field(default_factory=dict)
    service_field_issues: list[str] = field(default_factory=list)
    duplicate_key_count: int = 0
    row_count: int = 0
    blocking_errors: list[str] = field(default_factory=list)
    warnings: list[str] = field(default_factory=list)
    suggested_actions: list[str] = field(default_factory=list)


def template_path() -> Path:
    """Return path to the Telco scoring schema template CSV."""
    return Path(__file__).resolve().parents[1] / "data" / "templates" / TEMPLATE_FILENAME


def load_schema_template() -> pd.DataFrame:
    """Load the Telco scoring template for download or reference."""
    path = template_path()
    if not path.exists():
        raise FileNotFoundError(f"Schema template not found at {path.resolve()}.")
    return pd.read_csv(path)


def suggest_column_renames(uploaded_columns: list[str], expected_columns: list[str]) -> dict[str, str]:
    """Suggest uploaded column renames to match expected Telco feature names."""
    suggestions: dict[str, str] = {}
    expected_lower = {col.lower(): col for col in expected_columns}
    used_expected: set[str] = set()

    for col in uploaded_columns:
        if col in expected_columns:
            continue

        normalized = re.sub(r"[\s_-]+", "_", col.strip().lower())
        if normalized in _COLUMN_ALIASES:
            target = _COLUMN_ALIASES[normalized]
            if target in expected_columns and target not in used_expected:
                suggestions[col] = target
                used_expected.add(target)
                continue

        if normalized in expected_lower:
            target = expected_lower[normalized]
            if target not in used_expected:
                suggestions[col] = target
                used_expected.add(target)
                continue

        close = difflib.get_close_matches(col, expected_columns, n=1, cutoff=0.85)
        if close and close[0] not in used_expected:
            suggestions[col] = close[0]
            used_expected.add(close[0])

    return suggestions


def apply_column_mapping(df: pd.DataFrame, column_mapping: dict[str, str]) -> pd.DataFrame:
    """Rename uploaded columns according to a confirmed mapping."""
    if not column_mapping:
        return df.copy()
    return df.rename(columns=column_mapping).copy()


def _normalize_key(value: Any) -> str:
    return str(value).strip()


def _check_numeric_columns(df: pd.DataFrame, report: CompatibilityReport) -> None:
    for col in NUMERIC_FEATURE_COLS:
        if col not in df.columns:
            continue

        coerced = pd.to_numeric(df[col], errors="coerce")
        if coerced.isna().any():
            report.blocking_errors.append(
                f"Column '{col}' has {int(coerced.isna().sum())} non-numeric value(s)."
            )

        if col == "SeniorCitizen" and not coerced.dropna().isin([0, 1]).all():
            report.blocking_errors.append(f"Column '{col}' must contain only 0 or 1.")
        elif col == TENURE_COL and (coerced.dropna() < 0).any():
            report.blocking_errors.append(f"Column '{col}' contains negative values.")
        elif col in {"MonthlyCharges", TOTAL_CHARGES_COL} and (coerced.dropna() < 0).any():
            report.blocking_errors.append(f"Column '{col}' contains negative values.")

    if TOTAL_CHARGES_COL in df.columns and TENURE_COL in df.columns:
        charges_text = df[TOTAL_CHARGES_COL].astype(str).str.strip()
        blank_mask = charges_text.eq("") | df[TOTAL_CHARGES_COL].isna()
        tenure = pd.to_numeric(df[TENURE_COL], errors="coerce")
        invalid_blank = blank_mask & (tenure != 0)
        if invalid_blank.any():
            report.blocking_errors.append(
                f"TotalCharges is blank for {int(invalid_blank.sum())} row(s) where tenure is not 0."
            )


def _check_categorical_columns(df: pd.DataFrame, report: CompatibilityReport) -> None:
    for col in CATEGORICAL_FEATURE_COLS:
        if col not in df.columns:
            continue

        allowed = set(ALLOWED_CATEGORICAL_VALUES[col])
        values = df[col].astype(str).str.strip()
        empty_mask = values.eq("") | df[col].isna()
        if empty_mask.any():
            report.blocking_errors.append(
                f"Column '{col}' has {int(empty_mask.sum())} missing or blank value(s)."
            )

        invalid = sorted(set(values[~empty_mask]) - allowed)
        if invalid:
            report.invalid_categorical_values[col] = invalid[:10]
            report.warnings.append(
                f"Column '{col}' has values outside the training schema. "
                "Scoring will proceed but those rows may be less reliable."
            )


def _check_service_combinations(df: pd.DataFrame, report: CompatibilityReport) -> None:
    if "InternetService" not in df.columns:
        return

    for idx, row in df.iterrows():
        internet = str(row.get("InternetService", "")).strip()
        if internet == "No":
            for field in INTERNET_DEPENDENT_FIELDS:
                if field not in df.columns:
                    continue
                value = str(row.get(field, "")).strip()
                if value and value != NO_INTERNET_SERVICE:
                    report.service_field_issues.append(
                        f"Row {idx}: {field} will be auto-fixed for InternetService=No."
                    )

        phone = str(row.get("PhoneService", "")).strip()
        if phone == "No" and "MultipleLines" in df.columns:
            multiple = str(row.get("MultipleLines", "")).strip()
            if multiple and multiple != NO_PHONE_SERVICE:
                report.service_field_issues.append(
                    f"Row {idx}: MultipleLines will be auto-fixed for PhoneService=No."
                )


def _build_suggested_actions(report: CompatibilityReport) -> list[str]:
    actions: list[str] = []

    if report.missing_features:
        actions.append(
            "Add or rename columns to include all 19 required Telco features: "
            + ", ".join(FEATURE_COLS)
        )
        actions.append("Download the schema template CSV and align your file to its headers.")

    if report.rename_suggestions and not report.column_mapping:
        pairs = [f"'{src}' -> '{dst}'" for src, dst in report.rename_suggestions.items()]
        actions.append("Apply suggested column mapping: " + "; ".join(pairs))

    if report.duplicate_key_count > 0 and report.primary_key_col:
        actions.append(
            f"Remove or fix {report.duplicate_key_count} duplicate value(s) in "
            f"'{report.primary_key_col}' so each customer has a unique primary key."
        )

    if report.blocking_errors:
        actions.append("Fix the blocking errors listed above, then re-upload the file.")

    if not report.is_compatible and not actions:
        actions.append(
            "This file does not match the Telco schema. Reshape it using the template, "
            "or use the optional 'Train on your data' fallback if you have a Churn label column."
        )

    return actions


def check_upload_compatibility(
    df: pd.DataFrame,
    id_col: str,
    *,
    column_mapping: dict[str, str] | None = None,
) -> CompatibilityReport:
    """Check whether an uploaded dataframe can be scored by the frozen Telco model."""
    mapping = column_mapping or {}
    working = apply_column_mapping(df, mapping)
    resolved_id_col = mapping.get(id_col, id_col)

    report = CompatibilityReport(
        is_compatible=False,
        primary_key_col=resolved_id_col if resolved_id_col in working.columns else id_col,
        column_mapping=dict(mapping),
        row_count=len(df),
    )

    report.extra_columns = [
        col
        for col in working.columns
        if col not in FEATURE_COLS + [TARGET_COL, IDENTIFIER_COL]
    ]
    report.rename_suggestions = {
        src: dst
        for src, dst in suggest_column_renames(
            list(df.columns),
            FEATURE_COLS + [IDENTIFIER_COL, TARGET_COL],
        ).items()
        if src not in mapping
    }

    if resolved_id_col not in working.columns:
        report.blocking_errors.append(
            f"Primary key column '{id_col}' not found after applying column mapping."
        )
    else:
        keys = working[resolved_id_col].map(_normalize_key)
        empty_keys = keys.eq("") | working[resolved_id_col].isna()
        if empty_keys.any():
            report.blocking_errors.append(
                f"Primary key column '{resolved_id_col}' has {int(empty_keys.sum())} blank value(s)."
            )
        report.duplicate_key_count = int(keys.duplicated(keep=False).sum())

    report.missing_features = [col for col in FEATURE_COLS if col not in working.columns]

    if not report.missing_features:
        _check_numeric_columns(working, report)
        _check_categorical_columns(working, report)
        _check_service_combinations(working, report)

    report.suggested_actions = _build_suggested_actions(report)
    report.is_compatible = (
        not report.missing_features
        and not report.blocking_errors
        and report.duplicate_key_count == 0
        and report.primary_key_col is not None
        and report.primary_key_col in working.columns
    )
    return report
