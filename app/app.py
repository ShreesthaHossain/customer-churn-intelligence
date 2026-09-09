"""Customer Churn Intelligence — Streamlit scoring dashboard."""

from __future__ import annotations

import sys
from pathlib import Path
from typing import Any

import pandas as pd
import streamlit as st

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.config import load_model_config, reports_dir
from src.data_cleaning import IDENTIFIER_COL, TARGET_COL
from src.data_loading import load_cleaned_churn_data, load_train_val_split
from src.data_separation import CATEGORICAL_FEATURE_COLS, FEATURE_COLS
from src.inference import predict_single_customer, score_uploaded_batch
from src.model import ChurnModelBundle, load_churn_pipeline
from src.policy import (
    HIGH_RISK_UI_BAND,
    NO_INTERNET_SERVICE,
    NO_PHONE_SERVICE,
    classify_risk_level,
    normalize_service_fields,
)
from src.training_service import train_and_score_upload
from src.upload_compatibility import (
    check_upload_compatibility,
    load_schema_template,
    suggest_column_renames,
)

st.set_page_config(
    page_title="Customer Churn Intelligence",
    page_icon="📡",
    layout="wide",
    initial_sidebar_state="expanded",
)

CUSTOM_CSS = """
<style>
    .main-header { font-size: 2rem; font-weight: 700; margin-bottom: 0.25rem; }
    .sub-header { color: #5f6368; margin-bottom: 1.5rem; }
    .result-panel {
        background: linear-gradient(180deg, #f8f9fb 0%, #ffffff 100%);
        border: 1px solid #d9dee7;
        border-radius: 16px;
        padding: 1.5rem;
        margin-top: 1rem;
        box-shadow: 0 8px 24px rgba(15, 23, 42, 0.06);
    }
    .result-label {
        color: #5f6368;
        font-size: 0.85rem;
        text-transform: uppercase;
        letter-spacing: 0.04em;
        margin-bottom: 0.25rem;
    }
    .result-value { font-size: 1.75rem; font-weight: 700; margin-bottom: 0.75rem; }
    .risk-low { color: #1e8e3e; }
    .risk-elevated { color: #e37400; }
    .risk-high { color: #d93025; }
    .policy-note {
        background: #eef3fb;
        border-left: 4px solid #4c72b0;
        padding: 0.75rem 1rem;
        border-radius: 8px;
        margin-top: 0.75rem;
        font-size: 0.92rem;
    }
</style>
"""
st.markdown(CUSTOM_CSS, unsafe_allow_html=True)


@st.cache_resource
def get_pipeline() -> ChurnModelBundle:
    return load_churn_pipeline()


@st.cache_resource
def get_model_config() -> dict[str, Any]:
    return load_model_config()


@st.cache_data
def get_categorical_options() -> dict[str, list[str]]:
    df = load_cleaned_churn_data()
    return {col: sorted(df[col].dropna().unique().tolist()) for col in CATEGORICAL_FEATURE_COLS}


@st.cache_data
def get_default_feature_values() -> dict[str, Any]:
    """Default form values from one validation customer (not test data)."""
    split = load_train_val_split()
    return split.X_val.iloc[0].to_dict()


@st.cache_data
def get_validation_pr_auc() -> float | None:
    """Load calibrated validation PR-AUC from saved model comparison reports."""
    calibration_path = reports_dir() / "calibration_comparison.csv"
    if calibration_path.exists():
        df = pd.read_csv(calibration_path)
        sigmoid = df[df["Method"].str.contains("Sigmoid", case=False, na=False)]
        if not sigmoid.empty:
            return float(sigmoid.iloc[0]["PR_AUC"])

    comparison_path = reports_dir() / "final_model_comparison.csv"
    if comparison_path.exists():
        df = pd.read_csv(comparison_path)
        leading = df[df["Rank"] == 1]
        if not leading.empty and "PR_AUC" in leading.columns:
            return float(leading.iloc[0]["PR_AUC"])
    return None


def risk_level_class(level: str) -> str:
    return {
        "Low": "risk-low",
        "Elevated": "risk-elevated",
        "High": "risk-high",
    }.get(level, "")


def risk_band_policy_text(decision_threshold: float) -> str:
    return (
        f"**Decision threshold (model policy):** `{decision_threshold:.2f}` — probabilities "
        f"at or above this value trigger retention outreach.\n\n"
        f"**UI risk bands (display only):**\n"
        f"- **Low** — probability `< {decision_threshold:.2f}`\n"
        f"- **Elevated** — `{decision_threshold:.2f}` to `< {HIGH_RISK_UI_BAND:.2f}` "
        f"(above outreach cutoff, not yet clearly high)\n"
        f"- **High** — probability `≥ {HIGH_RISK_UI_BAND:.2f}` "
        f"(UI demo band for priority escalation; not a model threshold)"
    )


def options_for_internet_dependent(internet_service: str, all_options: list[str]) -> list[str]:
    if internet_service == "No":
        return [NO_INTERNET_SERVICE]
    return all_options


def options_for_multiple_lines(phone_service: str, all_options: list[str]) -> list[str]:
    if phone_service == "No":
        return [NO_PHONE_SERVICE]
    return [opt for opt in all_options if opt != NO_PHONE_SERVICE]


def build_customer_record(form_values: dict[str, Any]) -> dict[str, Any]:
    record = dict(form_values)
    record["customerID"] = form_values.get("customerID") or None
    record["tenure"] = int(record["tenure"])
    record["SeniorCitizen"] = int(record["SeniorCitizen"])
    record["MonthlyCharges"] = float(record["MonthlyCharges"])

    total_charges = record["TotalCharges"]
    if total_charges is None or str(total_charges).strip() == "":
        record["TotalCharges"] = ""
    else:
        record["TotalCharges"] = float(total_charges)
    return record


def validate_form_values(form_values: dict[str, Any]) -> str | None:
    if form_values["tenure"] < 0:
        return "Tenure must be zero or greater."
    if form_values["MonthlyCharges"] < 0:
        return "Monthly charges must be zero or greater."

    total_charges = form_values["TotalCharges"]
    if total_charges is not None and str(total_charges).strip() != "":
        try:
            if float(total_charges) < 0:
                return "Total charges must be zero or greater."
        except ValueError:
            return "Total charges must be a valid number."
    elif form_values["tenure"] != 0:
        return "Total charges can be blank only when tenure is 0."

    return None


def render_model_summary(config: dict[str, Any]) -> None:
    st.subheader("Model Summary")
    st.markdown(f"**Selected model:** {config['selected_model']}")
    st.markdown(f"**Final threshold:** `{config['decision_threshold']}`")
    if config.get("calibration_retained"):
        st.markdown(f"**Calibration:** {config.get('calibration_method', 'N/A')}")
    st.caption("Threshold tuned on validation data. Test set not used for deployment policy.")

    st.markdown("---")
    st.markdown("**Validation performance**")

    metrics = config.get("validation_metrics_at_threshold", {})
    pr_auc = get_validation_pr_auc()

    metric_cols = st.columns(2)
    if metrics:
        metric_cols[0].metric("Precision", f"{metrics.get('Precision', 0):.3f}")
        metric_cols[1].metric("Recall", f"{metrics.get('Recall', 0):.3f}")
        metric_cols[0].metric("F1", f"{metrics.get('F1', 0):.3f}")
        if pr_auc is not None:
            metric_cols[1].metric("PR-AUC", f"{pr_auc:.3f}")
        else:
            metric_cols[1].caption("PR-AUC not found in saved reports.")
    st.caption("Precision / Recall / F1 are at the final decision threshold on validation.")

    with st.expander("Risk band policy"):
        st.markdown(risk_band_policy_text(float(config["decision_threshold"])))


def render_prediction_results(
    result: dict[str, Any],
    threshold: float,
) -> None:
    probability = result["churn_probability"]
    risk_level = classify_risk_level(probability, threshold)
    predicted_label = "Churn" if probability >= threshold else "No Churn"
    retention_action = (
        "Recommend retention outreach — offer proactive save campaign."
        if result["retention_recommended"]
        else "Routine monitoring — no immediate retention outreach."
    )

    st.markdown('<div class="result-panel">', unsafe_allow_html=True)

    row1_a, row1_b, row1_c, row1_d = st.columns(4)
    with row1_a:
        st.markdown('<div class="result-label">Churn Probability</div>', unsafe_allow_html=True)
        st.markdown(f'<div class="result-value">{probability:.1%}</div>', unsafe_allow_html=True)
    with row1_b:
        st.markdown('<div class="result-label">Risk Level</div>', unsafe_allow_html=True)
        st.markdown(
            f'<div class="result-value {risk_level_class(risk_level)}">{risk_level}</div>',
            unsafe_allow_html=True,
        )
    with row1_c:
        st.markdown('<div class="result-label">Decision</div>', unsafe_allow_html=True)
        st.markdown(f'<div class="result-value">{predicted_label}</div>', unsafe_allow_html=True)
    with row1_d:
        st.markdown('<div class="result-label">Threshold Applied</div>', unsafe_allow_html=True)
        st.markdown(f'<div class="result-value">{threshold:.2f}</div>', unsafe_allow_html=True)

    st.markdown("---")
    st.markdown('<div class="result-label">Recommended Action</div>', unsafe_allow_html=True)
    st.markdown(f"### {retention_action}")

    st.markdown(
        f'<div class="policy-note">{result["interpretation_note"]}</div>',
        unsafe_allow_html=True,
    )
    st.markdown("</div>", unsafe_allow_html=True)


def _index_for_option(field: str, available: list[str]) -> int:
    preferred = st.session_state.get(f"field_{field}", available[0])
    return available.index(preferred) if preferred in available else 0


def _init_session_defaults(defaults: dict[str, Any]) -> None:
    if st.session_state.get("_form_initialized"):
        return
    for key, value in defaults.items():
        st.session_state[f"field_{key}"] = value
    st.session_state["_form_initialized"] = True


def render_compatibility_report(report) -> None:
    if report.is_compatible:
        st.success("This file is compatible with the saved Telco churn model.")
    else:
        st.error("This file is not compatible with the saved Telco churn model.")

    if report.blocking_errors:
        st.markdown("**Blocking issues**")
        for issue in report.blocking_errors:
            st.markdown(f"- {issue}")

    if report.missing_features:
        st.markdown("**Missing feature columns**")
        st.code(", ".join(report.missing_features))

    if report.rename_suggestions:
        st.markdown("**Suggested column mapping**")
        mapping_df = pd.DataFrame(
            [{"Uploaded column": src, "Expected column": dst} for src, dst in report.rename_suggestions.items()]
        )
        st.dataframe(mapping_df, use_container_width=True, hide_index=True)

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


def render_batch_results(results: pd.DataFrame, *, session_model: bool = False) -> None:
    total = len(results)
    predicted_churners = int(results["retention_recommended"].sum())
    high_risk = int((results["risk_level"] == "High").sum())

    c1, c2, c3 = st.columns(3)
    c1.metric("Customers scored", total)
    c2.metric("Retention outreach flagged", predicted_churners)
    c3.metric("High-risk customers", high_risk)

    if session_model:
        st.warning(
            "These predictions come from a custom session model trained on your upload. "
            "They do not use the frozen production Telco pipeline."
        )

    show_flagged = st.checkbox("Show only retention outreach customers", value=False)
    display = results.copy()
    if show_flagged:
        display = display[display["retention_recommended"] == 1]

    st.dataframe(display, use_container_width=True, hide_index=True)

    csv_bytes = display.to_csv(index=False).encode("utf-8")
    st.download_button(
        "Download prediction results (CSV)",
        data=csv_bytes,
        file_name="churn_predictions.csv",
        mime="text/csv",
        use_container_width=True,
    )


def render_batch_upload_tab(pipeline: ChurnModelBundle, config: dict[str, Any]) -> None:
    st.subheader("Batch Upload")
    st.caption(
        "Upload a CSV to check whether the saved Telco model can score it. "
        "If compatible, you will get ranked churn predictions with customer details."
    )

    uploaded = st.file_uploader("Upload customer CSV", type=["csv"])
    if uploaded is None:
        template_df = load_schema_template()
        st.markdown("**Required schema**")
        st.caption(
            "Your file needs the 19 Telco feature columns used by the saved model. "
            "Download the template below if you need a starting point."
        )
        st.dataframe(template_df.head(1), use_container_width=True, hide_index=True)
        st.download_button(
            "Download schema template (CSV)",
            data=template_df.to_csv(index=False).encode("utf-8"),
            file_name="telco_scoring_template.csv",
            mime="text/csv",
        )
        return

    try:
        upload_df = pd.read_csv(uploaded)
    except Exception as exc:
        st.error(f"Could not read CSV: {exc}")
        return

    st.markdown(f"**Rows:** {len(upload_df):,} | **Columns:** {len(upload_df.columns)}")
    st.dataframe(upload_df.head(), use_container_width=True, hide_index=True)

    id_candidates = list(upload_df.columns)
    default_id = IDENTIFIER_COL if IDENTIFIER_COL in upload_df.columns else id_candidates[0]
    id_col = st.selectbox(
        "Primary key column",
        options=id_candidates,
        index=id_candidates.index(default_id),
        help="Must uniquely identify each customer row in the output.",
    )

    suggested = suggest_column_renames(list(upload_df.columns), FEATURE_COLS + [IDENTIFIER_COL, TARGET_COL])
    use_suggestions = False
    if suggested:
        use_suggestions = st.checkbox(
            "Apply suggested column mapping",
            value=all(dst not in upload_df.columns for dst in suggested.values()),
        )

    column_mapping = suggested if use_suggestions else {}
    report = check_upload_compatibility(upload_df, id_col, column_mapping=column_mapping)
    render_compatibility_report(report)

    if report.is_compatible:
        if st.button("Score customers with saved model", type="primary", use_container_width=True):
            with st.spinner("Scoring uploaded customers..."):
                try:
                    results = score_uploaded_batch(
                        pipeline,
                        upload_df,
                        id_col,
                        column_mapping=column_mapping,
                    )
                    st.session_state["batch_results"] = results
                    st.session_state["batch_results_session_model"] = False
                except Exception as exc:
                    st.error(f"Batch scoring failed: {exc}")
                    return

    if "batch_results" in st.session_state and not st.session_state.get("batch_results_session_model"):
        st.markdown("---")
        st.subheader("Prediction Results")
        render_batch_results(st.session_state["batch_results"])

    if not report.is_compatible:
        with st.expander("Train on your data instead (session-only fallback)"):
            st.caption(
                "Use this only when your file has a binary churn label but does not match "
                "the Telco schema required by the saved model."
            )
            target_candidates = [col for col in upload_df.columns if col != id_col]
            if TARGET_COL in target_candidates:
                default_target_idx = target_candidates.index(TARGET_COL)
            else:
                default_target_idx = 0
            target_col = st.selectbox(
                "Target / churn label column",
                options=target_candidates,
                index=default_target_idx,
            )
            if st.button("Train custom session model and score", use_container_width=True):
                with st.spinner("Training session model on uploaded data..."):
                    try:
                        training_result = train_and_score_upload(
                            upload_df,
                            target_col=target_col,
                            id_col=id_col,
                        )
                        st.session_state["batch_results"] = training_result.results
                        st.session_state["batch_results_session_model"] = True
                        st.session_state["batch_training_metrics"] = {
                            "validation": training_result.validation_metrics,
                            "test": training_result.test_metrics,
                        }
                    except Exception as exc:
                        st.error(f"Session training failed: {exc}")
                        return

            if st.session_state.get("batch_results_session_model") and "batch_training_metrics" in st.session_state:
                metrics = st.session_state["batch_training_metrics"]
                st.markdown("**Session model validation metrics**")
                vcols = st.columns(4)
                vcols[0].metric("Precision", f"{metrics['validation']['precision']:.3f}")
                vcols[1].metric("Recall", f"{metrics['validation']['recall']:.3f}")
                vcols[2].metric("F1", f"{metrics['validation']['f1']:.3f}")
                vcols[3].metric("Threshold", f"{metrics['validation']['threshold']:.2f}")
                st.markdown("---")
                st.subheader("Prediction Results")
                render_batch_results(st.session_state["batch_results"], session_model=True)


def render_single_customer_tab(
    pipeline: ChurnModelBundle,
    config: dict[str, Any],
    threshold: float,
    defaults: dict[str, Any],
    options: dict[str, list[str]],
) -> None:
    st.subheader("Customer Profile")
    with st.form("customer_form"):
        top_col1, top_col2 = st.columns(2)

        with top_col1:
            customer_id = st.text_input("Customer ID (optional)", value="")
            gender = st.selectbox(
                "Gender",
                options["gender"],
                index=options["gender"].index(st.session_state["field_gender"]),
            )
            senior = st.selectbox(
                "Senior Citizen",
                options=[0, 1],
                format_func=lambda x: "Yes" if x == 1 else "No",
                index=int(st.session_state["field_SeniorCitizen"]),
            )
            partner = st.selectbox(
                "Partner",
                options["Partner"],
                index=options["Partner"].index(st.session_state["field_Partner"]),
            )
            dependents = st.selectbox(
                "Dependents",
                options["Dependents"],
                index=options["Dependents"].index(st.session_state["field_Dependents"]),
            )
            tenure = st.number_input(
                "Tenure (months)",
                min_value=0,
                max_value=100,
                value=int(st.session_state["field_tenure"]),
            )
            phone = st.selectbox(
                "Phone Service",
                options["PhoneService"],
                index=options["PhoneService"].index(st.session_state["field_PhoneService"]),
            )
            multiple_line_options = options_for_multiple_lines(phone, options["MultipleLines"])
            multiple_lines = st.selectbox(
                "Multiple Lines",
                options=multiple_line_options,
                index=_index_for_option("MultipleLines", multiple_line_options),
            )
            internet = st.selectbox(
                "Internet Service",
                options["InternetService"],
                index=options["InternetService"].index(st.session_state["field_InternetService"]),
            )
            online_security = st.selectbox(
                "Online Security",
                options=options_for_internet_dependent(internet, options["OnlineSecurity"]),
                index=_index_for_option("OnlineSecurity", options_for_internet_dependent(internet, options["OnlineSecurity"])),
            )
            online_backup = st.selectbox(
                "Online Backup",
                options=options_for_internet_dependent(internet, options["OnlineBackup"]),
                index=_index_for_option("OnlineBackup", options_for_internet_dependent(internet, options["OnlineBackup"])),
            )

        with top_col2:
            device_protection = st.selectbox(
                "Device Protection",
                options=options_for_internet_dependent(internet, options["DeviceProtection"]),
                index=_index_for_option("DeviceProtection", options_for_internet_dependent(internet, options["DeviceProtection"])),
            )
            tech_support = st.selectbox(
                "Tech Support",
                options=options_for_internet_dependent(internet, options["TechSupport"]),
                index=_index_for_option("TechSupport", options_for_internet_dependent(internet, options["TechSupport"])),
            )
            streaming_tv = st.selectbox(
                "Streaming TV",
                options=options_for_internet_dependent(internet, options["StreamingTV"]),
                index=_index_for_option("StreamingTV", options_for_internet_dependent(internet, options["StreamingTV"])),
            )
            streaming_movies = st.selectbox(
                "Streaming Movies",
                options=options_for_internet_dependent(internet, options["StreamingMovies"]),
                index=_index_for_option("StreamingMovies", options_for_internet_dependent(internet, options["StreamingMovies"])),
            )
            contract = st.selectbox(
                "Contract",
                options["Contract"],
                index=options["Contract"].index(st.session_state["field_Contract"]),
            )
            paperless = st.selectbox(
                "Paperless Billing",
                options["PaperlessBilling"],
                index=options["PaperlessBilling"].index(st.session_state["field_PaperlessBilling"]),
            )
            payment = st.selectbox(
                "Payment Method",
                options["PaymentMethod"],
                index=options["PaymentMethod"].index(st.session_state["field_PaymentMethod"]),
            )
            monthly_charges = st.number_input(
                "Monthly Charges ($)",
                min_value=0.0,
                value=float(st.session_state["field_MonthlyCharges"]),
                step=1.0,
            )
            total_charges = st.text_input(
                "Total Charges ($)",
                value=str(st.session_state["field_TotalCharges"]),
                help="Leave blank only when tenure is 0 (new customer).",
            )

        submitted = st.form_submit_button("Predict Churn", type="primary", use_container_width=True)

    if submitted:
        form_values = {
            "customerID": customer_id.strip() or None,
            "gender": gender,
            "SeniorCitizen": senior,
            "Partner": partner,
            "Dependents": dependents,
            "tenure": tenure,
            "PhoneService": phone,
            "MultipleLines": multiple_lines,
            "InternetService": internet,
            "OnlineSecurity": online_security,
            "OnlineBackup": online_backup,
            "DeviceProtection": device_protection,
            "TechSupport": tech_support,
            "StreamingTV": streaming_tv,
            "StreamingMovies": streaming_movies,
            "Contract": contract,
            "PaperlessBilling": paperless,
            "PaymentMethod": payment,
            "MonthlyCharges": monthly_charges,
            "TotalCharges": total_charges,
        }

        validation_error = validate_form_values(form_values)
        if validation_error:
            st.error(validation_error)
            return

        form_values, sync_notes = normalize_service_fields(form_values)
        for note in sync_notes:
            st.info(note)

        try:
            record = build_customer_record(form_values)
            result = predict_single_customer(pipeline, record)
        except ValueError as exc:
            st.error(str(exc))
            return
        except Exception as exc:
            st.error(f"Prediction failed: {exc}")
            return

        st.markdown("---")
        st.subheader("Prediction Results")
        if result.get("customerID"):
            st.caption(f"Customer ID: {result['customerID']}")
        render_prediction_results(result, threshold)


def main() -> None:
    st.markdown('<p class="main-header">Customer Churn Intelligence</p>', unsafe_allow_html=True)
    st.markdown(
        '<p class="sub-header">Score telecom customers with the saved calibrated XGBoost pipeline or upload a CSV for batch scoring.</p>',
        unsafe_allow_html=True,
    )

    try:
        pipeline = get_pipeline()
        config = get_model_config()
    except FileNotFoundError as exc:
        st.error(f"Missing deployment artifact: {exc}")
        st.stop()

    threshold = float(config["decision_threshold"])
    defaults = get_default_feature_values()
    options = get_categorical_options()
    _init_session_defaults(defaults)

    with st.sidebar:
        render_model_summary(config)

    tab_single, tab_batch = st.tabs(["Single customer", "Batch upload"])
    with tab_single:
        render_single_customer_tab(pipeline, config, threshold, defaults, options)
    with tab_batch:
        render_batch_upload_tab(pipeline, config)


if __name__ == "__main__":
    main()
