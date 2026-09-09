"""Streamlit UI helpers for the batch upload workflow."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import pandas as pd
import streamlit as st

from src.data_cleaning import IDENTIFIER_COL, TARGET_COL
from src.data_separation import FEATURE_COLS
from src.upload_compatibility import (
    CompatibilityReport,
    build_column_mapping_from_assignments,
    build_default_assignments,
    build_readiness_summary,
    detect_primary_key_column,
    load_column_guide,
    load_schema_template,
    normalize_upload_headers,
)

PROJECT_ROOT = Path(__file__).resolve().parents[1]
SAMPLE_DATA_PATH = PROJECT_ROOT / "data" / "processed" / "cleaned_churn.csv"
SAMPLE_ROW_COUNT = 10


def render_upload_help_box() -> None:
    st.info(
        "**Easy upload tips:** Column order does not matter — only column names and values. "
        "Extra columns are OK and will be ignored. **Churn** is optional for prediction-only uploads."
    )


def load_sample_upload_data() -> pd.DataFrame | None:
    """Load a small Telco sample for demo scoring."""
    if not SAMPLE_DATA_PATH.exists():
        return None
    sample = pd.read_csv(SAMPLE_DATA_PATH).head(SAMPLE_ROW_COUNT)
    return normalize_upload_headers(sample)


def render_readiness_checklist(readiness: dict[str, Any]) -> None:
    matched = readiness["matched_count"]
    required = readiness["required_count"]
    st.markdown(f"**Readiness:** {matched}/{required} required fields mapped")

    cols = st.columns(3)
    checklist = readiness["checklist"]
    chunk_size = (len(FEATURE_COLS) + 2) // 3
    for col_idx in range(3):
        start = col_idx * chunk_size
        end = min(start + chunk_size, len(FEATURE_COLS))
        with cols[col_idx]:
            for feature in FEATURE_COLS[start:end]:
                icon = "✅" if checklist.get(feature) else "❌"
                st.markdown(f"{icon} `{feature}`")


def render_column_mapper(
    upload_df: pd.DataFrame,
    session_key: str = "column_assignments",
) -> dict[str, str | None]:
    """Render expected->uploaded mapping controls and return assignments."""
    if session_key not in st.session_state:
        st.session_state[session_key] = build_default_assignments(upload_df)

    assignments: dict[str, str | None] = dict(st.session_state[session_key])
    uploaded_options = ["— not mapped —"] + list(upload_df.columns)
    used_uploaded: set[str] = set()

    st.markdown("**Column mapper**")
    st.caption("Match each required Telco field to a column from your file.")

    for expected_col in FEATURE_COLS:
        current = assignments.get(expected_col)
        option_values = uploaded_options.copy()
        selected_index = 0
        if current and current in upload_df.columns:
            selected_index = option_values.index(current)
            used_uploaded.add(current)

        selected = st.selectbox(
            expected_col,
            options=option_values,
            index=selected_index,
            key=f"mapper_{session_key}_{expected_col}",
        )
        assignments[expected_col] = None if selected == "— not mapped —" else selected

    duplicate_sources = [
        src
        for src in assignments.values()
        if src and list(assignments.values()).count(src) > 1
    ]
    if duplicate_sources:
        st.warning(
            "Each uploaded column can only map to one expected field. "
            f"Duplicate mappings: {sorted(set(duplicate_sources))}"
        )

    st.session_state[session_key] = assignments
    return assignments


def build_mapping_from_ui(
    upload_df: pd.DataFrame,
    assignments: dict[str, str | None],
) -> dict[str, str]:
    """Build uploaded->expected mapping from UI assignments, keeping direct name matches."""
    explicit_mapping = build_column_mapping_from_assignments(assignments)
    for expected_col in FEATURE_COLS:
        if expected_col in upload_df.columns and expected_col not in explicit_mapping.values():
            explicit_mapping[expected_col] = expected_col
    return explicit_mapping


def render_compatibility_report(report: CompatibilityReport) -> None:
    readiness = report.readiness_summary
    if readiness:
        render_readiness_checklist(readiness)

    if report.is_compatible:
        st.success("Your file is ready to score with the saved Telco churn model.")
    else:
        st.error("Your file is not ready yet. Use the checklist and mapper below.")

    if report.user_friendly_messages:
        st.markdown("**What to fix**")
        for message in report.user_friendly_messages:
            st.markdown(f"- {message}")

    if report.warnings:
        st.markdown("**Warnings**")
        for warning in report.warnings:
            st.markdown(f"- {warning}")

    if report.service_field_issues:
        with st.expander(f"Service-field auto-fixes ({len(report.service_field_issues)})"):
            for note in report.service_field_issues[:20]:
                st.caption(note)
            if len(report.service_field_issues) > 20:
                st.caption(f"... and {len(report.service_field_issues) - 20} more rows.")

    if report.suggested_actions:
        st.markdown("**Suggested next steps**")
        for action in report.suggested_actions:
            st.markdown(f"- {action}")

    if report.blocking_errors:
        with st.expander("Technical details"):
            for issue in report.blocking_errors:
                st.markdown(f"- {issue}")
            if report.missing_features:
                st.code(", ".join(report.missing_features))


def render_template_downloads() -> None:
    template_df = load_schema_template()
    guide_df = load_column_guide()

    col1, col2 = st.columns(2)
    with col1:
        st.download_button(
            "Download schema template (CSV)",
            data=template_df.to_csv(index=False).encode("utf-8"),
            file_name="telco_scoring_template.csv",
            mime="text/csv",
            use_container_width=True,
        )
    with col2:
        st.download_button(
            "Download column guide (CSV)",
            data=guide_df.to_csv(index=False).encode("utf-8"),
            file_name="telco_column_guide.csv",
            mime="text/csv",
            use_container_width=True,
        )


def get_upload_dataframe() -> pd.DataFrame | None:
    return st.session_state.get("batch_upload_df")


def initialize_upload_dataframe(df: pd.DataFrame) -> pd.DataFrame:
    normalized = normalize_upload_headers(df)
    st.session_state["batch_upload_df"] = normalized
    st.session_state.pop("column_assignments", None)
    st.session_state.pop("batch_results", None)
    st.session_state.pop("batch_results_session_model", None)
    st.session_state.pop("batch_training_metrics", None)
    return normalized


def resolve_upload_settings(upload_df: pd.DataFrame) -> tuple[str, dict[str, str], dict[str, Any]]:
    """Resolve primary key, mapping, and readiness from current UI state."""
    assignments = render_column_mapper(upload_df)
    column_mapping = build_mapping_from_ui(upload_df, assignments)
    readiness = build_readiness_summary(upload_df, column_mapping=column_mapping)

    detected_id = detect_primary_key_column(upload_df)
    id_candidates = list(upload_df.columns)
    default_id = detected_id or (IDENTIFIER_COL if IDENTIFIER_COL in upload_df.columns else id_candidates[0])

    id_col = st.selectbox(
        "Primary key column",
        options=id_candidates,
        index=id_candidates.index(default_id) if default_id in id_candidates else 0,
        help="Must uniquely identify each customer row in the output.",
    )

    if upload_df[id_col].nunique() != len(upload_df):
        st.warning(
            f"The selected primary key **{id_col}** is not unique. "
            "Choose a column with one value per customer."
        )

    return id_col, column_mapping, readiness
