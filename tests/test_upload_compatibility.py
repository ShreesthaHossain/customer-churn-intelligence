"""Tests for upload compatibility checks."""

from __future__ import annotations

import pandas as pd
import pytest

from src.data_cleaning import IDENTIFIER_COL, TARGET_COL
from src.data_separation import FEATURE_COLS
from src.upload_compatibility import (
    apply_column_mapping,
    build_column_mapping_from_assignments,
    build_default_assignments,
    build_readiness_summary,
    check_upload_compatibility,
    detect_primary_key_column,
    normalize_upload_headers,
    suggest_column_renames,
)


def _valid_row(customer_id: str = "C001") -> dict:
    return {
        IDENTIFIER_COL: customer_id,
        "gender": "Female",
        "SeniorCitizen": 0,
        "Partner": "Yes",
        "Dependents": "No",
        "tenure": 12,
        "PhoneService": "Yes",
        "MultipleLines": "No",
        "InternetService": "DSL",
        "OnlineSecurity": "No",
        "OnlineBackup": "Yes",
        "DeviceProtection": "No",
        "TechSupport": "No",
        "StreamingTV": "No",
        "StreamingMovies": "No",
        "Contract": "Month-to-month",
        "PaperlessBilling": "Yes",
        "PaymentMethod": "Electronic check",
        "MonthlyCharges": 50.0,
        "TotalCharges": 600.0,
        TARGET_COL: "No",
    }


def test_compatible_upload_passes() -> None:
    df = pd.DataFrame([_valid_row("A1"), _valid_row("A2")])
    report = check_upload_compatibility(df, IDENTIFIER_COL)
    assert report.is_compatible
    assert report.missing_features == []
    assert report.duplicate_key_count == 0
    assert report.readiness_summary["matched_count"] == 19


def test_missing_features_fail() -> None:
    row = _valid_row()
    del row["gender"]
    df = pd.DataFrame([row])
    report = check_upload_compatibility(df, IDENTIFIER_COL)
    assert not report.is_compatible
    assert "gender" in report.missing_features
    assert any("gender" in msg for msg in report.user_friendly_messages)


def test_duplicate_primary_key_fails() -> None:
    df = pd.DataFrame([_valid_row("DUPE"), _valid_row("DUPE")])
    report = check_upload_compatibility(df, IDENTIFIER_COL)
    assert not report.is_compatible
    assert report.duplicate_key_count == 2
    assert any("share the same ID" in msg for msg in report.user_friendly_messages)


def test_suggest_column_renames() -> None:
    suggestions = suggest_column_renames(
        ["customer_id", "monthly_charges"],
        FEATURE_COLS + [IDENTIFIER_COL],
    )
    assert suggestions["customer_id"] == IDENTIFIER_COL
    assert suggestions["monthly_charges"] == "MonthlyCharges"


def test_apply_column_mapping_enables_compatibility() -> None:
    row = _valid_row("A1")
    df = pd.DataFrame([row]).rename(columns={"MonthlyCharges": "monthly_charges"})
    mapping = {"monthly_charges": "MonthlyCharges"}
    report = check_upload_compatibility(df, IDENTIFIER_COL, column_mapping=mapping)
    assert report.is_compatible
    mapped = apply_column_mapping(df, mapping)
    assert "MonthlyCharges" in mapped.columns


def test_blank_total_charges_with_tenure_blocks() -> None:
    row = _valid_row("A1")
    row["tenure"] = 5
    row["TotalCharges"] = ""
    df = pd.DataFrame([row])
    report = check_upload_compatibility(df, IDENTIFIER_COL)
    assert not report.is_compatible
    assert any("Total Charges" in msg for msg in report.user_friendly_messages)


def test_normalize_upload_headers_case_insensitive() -> None:
    row = _valid_row("A1")
    df = pd.DataFrame([row]).rename(columns={"gender": "Gender", "tenure": "TENURE"})
    normalized = normalize_upload_headers(df)
    assert "gender" in normalized.columns
    assert "tenure" in normalized.columns


def test_detect_primary_key_column_prefers_customer_id() -> None:
    df = pd.DataFrame([_valid_row("A1"), _valid_row("A2")])
    assert detect_primary_key_column(df) == IDENTIFIER_COL


def test_build_readiness_summary() -> None:
    row = _valid_row("A1")
    df = pd.DataFrame([row])
    summary = build_readiness_summary(df)
    assert summary["matched_count"] == 19
    assert summary["ready_to_score"] is True
    assert summary["checklist"]["gender"] is True


def test_build_column_mapping_from_assignments() -> None:
    assignments = {"gender": "cust_gender", "tenure": "months_active"}
    mapping = build_column_mapping_from_assignments(assignments)
    assert mapping == {"cust_gender": "gender", "months_active": "tenure"}


def test_build_column_mapping_rejects_duplicate_sources() -> None:
    assignments = {"gender": "col_a", "Partner": "col_a"}
    with pytest.raises(ValueError, match="cannot map"):
        build_column_mapping_from_assignments(assignments)


def test_build_default_assignments_uses_direct_matches() -> None:
    df = pd.DataFrame([_valid_row("A1")])
    assignments = build_default_assignments(df)
    assert assignments["gender"] == "gender"
    assert assignments["tenure"] == "tenure"
