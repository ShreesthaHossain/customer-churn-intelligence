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
COLUMN_GUIDE_FILENAME = "telco_column_guide.csv"
REQUIRED_FEATURE_COUNT = len(FEATURE_COLS)

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

COLUMN_DESCRIPTIONS: dict[str, str] = {
    IDENTIFIER_COL: "Unique customer identifier (one per row).",
    "gender": "Customer gender: Female or Male.",
    "SeniorCitizen": "1 if senior citizen, otherwise 0.",
    "Partner": "Whether customer has a partner: Yes or No.",
    "Dependents": "Whether customer has dependents: Yes or No.",
    "tenure": "Months as a customer (0-100).",
    "PhoneService": "Phone service: Yes or No.",
    "MultipleLines": "Multiple phone lines: Yes, No, or No phone service.",
    "InternetService": "Internet type: No, DSL, or Fiber optic.",
    "OnlineSecurity": "Online security add-on: Yes, No, or No internet service.",
    "OnlineBackup": "Online backup add-on: Yes, No, or No internet service.",
    "DeviceProtection": "Device protection add-on: Yes, No, or No internet service.",
    "TechSupport": "Tech support add-on: Yes, No, or No internet service.",
    "StreamingTV": "Streaming TV add-on: Yes, No, or No internet service.",
    "StreamingMovies": "Streaming movies add-on: Yes, No, or No internet service.",
    "Contract": "Contract type: Month-to-month, One year, or Two year.",
    "PaperlessBilling": "Paperless billing: Yes or No.",
    "PaymentMethod": "Payment method (Electronic check, Mailed check, etc.).",
    "MonthlyCharges": "Current monthly charge amount (numeric, >= 0).",
    "TotalCharges": "Lifetime charges (numeric; blank only when tenure is 0).",
    TARGET_COL: "Optional churn label for comparison or session retraining.",
}

_ID_ALIASES = [IDENTIFIER_COL, "customer_id", "customerid", "id", "cust_id"]

_COLUMN_ALIASES: dict[str, str] = {
    "customer_id": IDENTIFIER_COL,
    "customerid": IDENTIFIER_COL,
    "id": IDENTIFIER_COL,
    "cust_id": IDENTIFIER_COL,
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
    matched_features: list[str] = field(default_factory=list)
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
    user_friendly_messages: list[str] = field(default_factory=list)
    readiness_summary: dict[str, Any] = field(default_factory=dict)


def template_path() -> Path:
    """Return path to the Telco scoring schema template CSV."""
    return Path(__file__).resolve().parents[1] / "data" / "templates" / TEMPLATE_FILENAME


def column_guide_path() -> Path:
    """Return path to the human-readable Telco column guide CSV."""
    return Path(__file__).resolve().parents[1] / "data" / "templates" / COLUMN_GUIDE_FILENAME


def load_schema_template() -> pd.DataFrame:
    """Load the Telco scoring template for download or reference."""
    path = template_path()
    if not path.exists():
        raise FileNotFoundError(f"Schema template not found at {path.resolve()}.")
    return pd.read_csv(path)


def load_column_guide() -> pd.DataFrame:
    """Load the Telco column guide for download or reference."""
    path = column_guide_path()
    if not path.exists():
        return pd.DataFrame(
            {
                "expected_column": list(COLUMN_DESCRIPTIONS.keys()),
                "description": list(COLUMN_DESCRIPTIONS.values()),
            }
        )
    return pd.read_csv(path)


def _normalize_header_name(column: str) -> str:
    return re.sub(r"[\s_-]+", "_", str(column).strip().lower())


def _resolve_header_name(column: str, expected_columns: list[str]) -> str | None:
    """Map one uploaded header to an expected Telco column name, if possible."""
    stripped = str(column).strip()
    if stripped in expected_columns:
        return stripped

    normalized = _normalize_header_name(stripped)
    if normalized in _COLUMN_ALIASES:
        target = _COLUMN_ALIASES[normalized]
        if target in expected_columns:
            return target

    expected_lower = {col.lower(): col for col in expected_columns}
    if normalized in expected_lower:
        return expected_lower[normalized]
    if stripped.lower() in expected_lower:
        return expected_lower[stripped.lower()]

    return None


def normalize_upload_headers(df: pd.DataFrame) -> pd.DataFrame:
    """
    Normalize uploaded CSV headers to Telco names when unambiguous.

    Strips whitespace, applies case-insensitive matching, and common aliases.
    """
    expected_columns = FEATURE_COLS + [IDENTIFIER_COL, TARGET_COL]
    rename_map: dict[str, str] = {}
    used_targets: set[str] = set()

    for column in df.columns:
        target = _resolve_header_name(column, expected_columns)
        if target is None or target in used_targets:
            continue
        if column != target:
            rename_map[column] = target
        used_targets.add(target)

    normalized = df.rename(columns=rename_map).copy()
    normalized.columns = [str(col).strip() for col in normalized.columns]
    return normalized


def suggest_column_renames(uploaded_columns: list[str], expected_columns: list[str]) -> dict[str, str]:
    """Suggest uploaded column renames to match expected Telco feature names."""
    suggestions: dict[str, str] = {}
    expected_lower = {col.lower(): col for col in expected_columns}
    used_expected: set[str] = set()

    for col in uploaded_columns:
        if col in expected_columns:
            continue

        normalized = _normalize_header_name(col)
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


def build_column_mapping_from_assignments(
    assignments: dict[str, str | None],
) -> dict[str, str]:
    """
    Convert expected->uploaded assignments into uploaded->expected mapping.

    Raises ValueError if the same uploaded column is assigned twice.
    """
    mapping: dict[str, str] = {}
    for expected_col, uploaded_col in assignments.items():
        if not uploaded_col:
            continue
        if uploaded_col in mapping and mapping[uploaded_col] != expected_col:
            raise ValueError(
                f"Uploaded column '{uploaded_col}' cannot map to both "
                f"'{mapping[uploaded_col]}' and '{expected_col}'."
            )
        mapping[uploaded_col] = expected_col
    return mapping


def build_default_assignments(df: pd.DataFrame) -> dict[str, str | None]:
    """Pre-fill expected->uploaded assignments from normalized headers and suggestions."""
    uploaded_columns = list(df.columns)
    suggestions = suggest_column_renames(uploaded_columns, FEATURE_COLS)
    assignments: dict[str, str | None] = {}

    for expected_col in FEATURE_COLS:
        if expected_col in uploaded_columns:
            assignments[expected_col] = expected_col
            continue

        suggested_source = next(
            (src for src, dst in suggestions.items() if dst == expected_col),
            None,
        )
        assignments[expected_col] = suggested_source

    return assignments


def build_readiness_summary(
    df: pd.DataFrame,
    column_mapping: dict[str, str] | None = None,
) -> dict[str, Any]:
    """Summarize how many required Telco feature columns are ready for scoring."""
    working = apply_column_mapping(df, column_mapping or {})
    matched_columns = [col for col in FEATURE_COLS if col in working.columns]
    missing_columns = [col for col in FEATURE_COLS if col not in working.columns]
    matched_count = len(matched_columns)

    return {
        "matched_count": matched_count,
        "required_count": REQUIRED_FEATURE_COUNT,
        "matched_columns": matched_columns,
        "missing_columns": missing_columns,
        "extra_columns": [
            col
            for col in working.columns
            if col not in FEATURE_COLS + [TARGET_COL, IDENTIFIER_COL]
        ],
        "ready_to_score": matched_count == REQUIRED_FEATURE_COUNT,
        "checklist": {col: col in working.columns for col in FEATURE_COLS},
    }


def detect_primary_key_column(df: pd.DataFrame) -> str | None:
    """Detect the best primary key column in an uploaded dataframe."""
    for candidate in _ID_ALIASES:
        if candidate not in df.columns:
            continue
        series = df[candidate]
        if series.isna().any():
            continue
        if series.astype(str).str.strip().eq("").any():
            continue
        if series.nunique() == len(series):
            return candidate

    for column in df.columns:
        series = df[column]
        if series.isna().any() or series.astype(str).str.strip().eq("").any():
            continue
        if series.nunique() == len(series):
            return column

    return None


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
            count = int(coerced.isna().sum())
            report.blocking_errors.append(
                f"Column '{col}' has {count} non-numeric value(s)."
            )
            report.dtype_issues.append(f"{col}: {count} non-numeric values")

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
            count = int(invalid_blank.sum())
            report.blocking_errors.append(
                f"TotalCharges is blank for {count} row(s) where tenure is not 0."
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


def _build_user_friendly_messages(report: CompatibilityReport) -> list[str]:
    messages: list[str] = []

    for col in report.missing_features:
        messages.append(
            f"We couldn't find a **{col}** column. Map one of your columns to **{col}** below."
        )

    if report.duplicate_key_count > 0 and report.primary_key_col:
        messages.append(
            f"**{report.duplicate_key_count}** customer row(s) share the same ID in "
            f"**{report.primary_key_col}**. Each row needs a unique primary key."
        )

    for error in report.blocking_errors:
        if "TotalCharges is blank" in error:
            messages.append(
                "Some rows have blank **Total Charges** but **tenure** is greater than 0. "
                "Fix those rows or set Total Charges to 0 for new customers only."
            )
        elif "Primary key column" in error and "not found" in error:
            messages.append(
                "The selected primary key column was not found after mapping. "
                "Choose a different ID column."
            )
        elif "blank value(s)" in error and report.primary_key_col:
            messages.append(
                f"The primary key column **{report.primary_key_col}** has blank values. "
                "Fill in a unique ID for every customer row."
            )
        elif "non-numeric" in error:
            messages.append(
                "Some numeric columns contain text values. Check **SeniorCitizen**, "
                "**tenure**, **MonthlyCharges**, and **TotalCharges**."
            )
        elif "must contain only 0 or 1" in error:
            messages.append("**SeniorCitizen** must use only 0 or 1.")
        elif "missing or blank value(s)" in error:
            messages.append(
                "Some categorical columns have blank values. Fill in every required field."
            )

    if not messages and not report.is_compatible:
        messages.append(
            "Your file is not ready yet. Use the checklist and column mapper below, "
            "or download the column guide for help."
        )

    return messages


def _build_suggested_actions(report: CompatibilityReport) -> list[str]:
    actions: list[str] = []

    if report.missing_features:
        actions.append(
            "Map your columns to the 19 required Telco fields using the column mapper below."
        )
        actions.append("Download the schema template or column guide if you need a reference.")

    if report.rename_suggestions and not report.column_mapping:
        pairs = [f"'{src}' -> '{dst}'" for src, dst in report.rename_suggestions.items()]
        actions.append("Apply suggested column mapping: " + "; ".join(pairs))

    if report.duplicate_key_count > 0 and report.primary_key_col:
        actions.append(
            f"Remove or fix duplicate IDs in '{report.primary_key_col}' so each customer is unique."
        )

    if report.blocking_errors:
        actions.append("Fix the blocking issues above, then score again.")

    if not report.is_compatible and not actions:
        actions.append(
            "Reshape your file using the template, or use the optional session retrain fallback "
            "if you have a churn label column."
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

    report.readiness_summary = build_readiness_summary(df, column_mapping=mapping)
    report.matched_features = report.readiness_summary["matched_columns"]
    report.extra_columns = report.readiness_summary["extra_columns"]
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

    report.missing_features = report.readiness_summary["missing_columns"]

    if not report.missing_features:
        _check_numeric_columns(working, report)
        _check_categorical_columns(working, report)
        _check_service_combinations(working, report)

    report.suggested_actions = _build_suggested_actions(report)
    report.user_friendly_messages = _build_user_friendly_messages(report)
    report.is_compatible = (
        not report.missing_features
        and not report.blocking_errors
        and report.duplicate_key_count == 0
        and report.primary_key_col is not None
        and report.primary_key_col in working.columns
    )
    return report
