"""Customer Churn Intelligence — Streamlit scoring dashboard."""

from __future__ import annotations

import sys
from pathlib import Path
from typing import Any

import pandas as pd
import streamlit as st

PROJECT_ROOT = Path(__file__).resolve().parents[1]
APP_DIR = Path(__file__).resolve().parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))
if str(APP_DIR) not in sys.path:
    sys.path.insert(0, str(APP_DIR))

from src.config import load_model_config, reports_dir
from src.data_cleaning import TARGET_COL
from src.data_loading import load_cleaned_churn_data, load_train_val_split
from upload_ui import (
    get_upload_dataframe,
    initialize_upload_dataframe,
    load_sample_upload_data,
    render_compatibility_report,
    render_template_downloads,
    render_upload_help_box,
    resolve_upload_settings,
)
from src.data_separation import CATEGORICAL_FEATURE_COLS, FEATURE_COLS
from src.inference import predict_single_customer, score_uploaded_batch
from src.model import ChurnModelBundle, load_churn_pipeline
from src.deployment import get_settings, load_env_file
from src.policy import (
    HIGH_RISK_UI_BAND,
    NO_INTERNET_SERVICE,
    NO_PHONE_SERVICE,
    classify_risk_level,
    normalize_service_fields,
    recommended_action,
)
from src.training_service import train_and_score_upload
from src.upload_compatibility import check_upload_compatibility

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
    .batch-action-card {
        background: #fff;
        border: 1px solid #d9dee7;
        border-left: 4px solid #e37400;
        border-radius: 12px;
        padding: 1rem 1.25rem;
        margin-bottom: 0.75rem;
    }
    .batch-action-card .action-headline {
        font-size: 1.05rem;
        font-weight: 700;
        margin: 0.35rem 0 0.5rem 0;
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


BATCH_ACTION_COLUMNS = [
    "primary_key",
    "churn_probability",
    "risk_level",
    "prediction",
    "retention_recommended",
    "recommended_action",
]

INTERPRETATION_NOTE = (
    "Probability and recommendation reflect model output; they do not "
    "establish causal reasons for churn."
)


def _batch_action_columns(results: pd.DataFrame) -> list[str]:
    cols = [col for col in BATCH_ACTION_COLUMNS if col in results.columns]
    if "actual_churn" in results.columns:
        cols.append("actual_churn")
    return cols


def _format_batch_action_table(df: pd.DataFrame) -> pd.DataFrame:
    """Human-readable display for the action summary table."""
    formatted = df.copy()
    if "churn_probability" in formatted.columns:
        formatted["churn_probability"] = formatted["churn_probability"].map(lambda p: f"{p:.1%}")
    if "retention_recommended" in formatted.columns:
        formatted["retention_recommended"] = formatted["retention_recommended"].map(
            {1: "Yes — outreach", 0: "No — monitor"}
        )
    return formatted


def _render_batch_priority_actions(flagged: pd.DataFrame, *, max_visible: int = 10) -> None:
    st.markdown("**Priority retention outreach**")
    st.caption("Customers flagged at or above the decision threshold, highest risk first.")

    for _, row in flagged.head(max_visible).iterrows():
        probability = float(row["churn_probability"])
        risk = str(row["risk_level"])
        action = str(row["recommended_action"])
        st.markdown(
            '<div class="batch-action-card">'
            f'<div class="result-label">{row["primary_key"]}</div>'
            f'<span class="result-value" style="font-size:1.15rem;margin-bottom:0">'
            f'{probability:.1%}</span> '
            f'<span class="result-value {risk_level_class(risk)}" '
            f'style="font-size:1rem;margin-bottom:0">{risk}</span>'
            f'<div class="result-label" style="margin-top:0.75rem">Recommended Action</div>'
            f'<div class="action-headline">{action}</div>'
            "</div>",
            unsafe_allow_html=True,
        )

    remaining = len(flagged) - max_visible
    if remaining > 0:
        with st.expander(f"Show {remaining} more flagged customer{'s' if remaining != 1 else ''}"):
            for _, row in flagged.iloc[max_visible:].iterrows():
                probability = float(row["churn_probability"])
                risk = str(row["risk_level"])
                action = str(row["recommended_action"])
                st.markdown(
                    f"**{row['primary_key']}** · {probability:.1%} · "
                    f"**{risk}** risk  \n"
                    f"**Recommended Action:** {action}"
                )


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


def options_for_phone_dependent(phone_service: str, all_options: list[str]) -> list[str]:
    if phone_service == "No":
        return [NO_PHONE_SERVICE]
    return [option for option in all_options if option != NO_PHONE_SERVICE]


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


def render_developer_api_section() -> None:
    """Sidebar links for FastAPI integration and interactive docs."""
    load_env_file()
    get_settings.cache_clear()
    settings = get_settings()

    st.markdown("---")
    st.subheader("Developer / API access")
    st.caption(
        "Integrate the same churn model into CRM, scripts, or other apps via the FastAPI service."
    )

    st.link_button("Open interactive API docs", settings.api_docs_url, use_container_width=True)
    st.markdown(f"**Health check:** [{settings.api_health_url}]({settings.api_health_url})")
    st.markdown("**Score one customer:** `POST /predict_churn`")
    st.markdown(
        "Send the same 19 customer fields as JSON. Response includes "
        "`churn_probability`, `risk_level`, and `recommended_action`."
    )

    if settings.auth_enabled:
        st.info(
            "API key required. Send header `X-API-Key` on `POST /predict_churn`. "
            "Keys are configured by the server admin in `.env` — not generated here."
        )
    else:
        st.caption("API auth is off locally — no key needed until `CHURN_API_KEY` is set.")

    with st.expander("Example request header"):
        if settings.auth_enabled:
            st.code("X-API-Key: <your-api-key>", language="http")
        st.code(f"POST {settings.api_base_url.rstrip('/')}/predict_churn", language="http")


def render_prediction_results(
    result: dict[str, Any],
    threshold: float,
) -> None:
    probability = result["churn_probability"]
    risk_level = classify_risk_level(probability, threshold)
    predicted_label = "Churn" if probability >= threshold else "No Churn"
    retention_action = recommended_action(bool(result["retention_recommended"]))

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


def _select_index(field: str, choices: list[str]) -> int:
    """Safe selectbox index — avoids crashes when stored value is not in choices."""
    if not choices:
        return 0
    preferred = st.session_state.get(f"field_{field}", choices[0])
    return choices.index(preferred) if preferred in choices else 0


def _internet_addon_options(internet_service: str, field: str, options: dict[str, list[str]]) -> list[str]:
    return options_for_internet_dependent(internet_service, options[field])


def _sync_profile_session(form_values: dict[str, Any]) -> None:
    """Persist submitted profile values so dropdowns stay consistent on rerun."""
    for key, value in form_values.items():
        if key == "customerID":
            continue
        st.session_state[f"field_{key}"] = value


def _init_session_defaults(defaults: dict[str, Any]) -> None:
    if st.session_state.get("_form_initialized"):
        return
    for key, value in defaults.items():
        st.session_state[f"field_{key}"] = value
    st.session_state["_form_initialized"] = True


def _session_field(field: str) -> Any:
    return st.session_state[f"field_{field}"]


def render_batch_results(results: pd.DataFrame, *, session_model: bool = False) -> None:
    total = len(results)
    predicted_churners = int(results["retention_recommended"].sum())
    high_risk = int((results["risk_level"] == "High").sum())
    routine_count = total - predicted_churners
    flagged = results[results["retention_recommended"] == 1]

    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Customers scored", total)
    c2.metric("Retention outreach flagged", predicted_churners)
    c3.metric("Routine monitoring", routine_count)
    c4.metric("High-risk customers", high_risk)

    st.markdown('<div class="result-panel">', unsafe_allow_html=True)
    st.markdown('<div class="result-label">Batch Recommended Actions</div>', unsafe_allow_html=True)
    if predicted_churners > 0:
        customer_word = "customer" if predicted_churners == 1 else "customers"
        st.markdown(
            f"### {predicted_churners} {customer_word} need retention outreach · "
            f"{routine_count} in routine monitoring"
        )
    else:
        st.markdown("### Routine monitoring — no immediate retention outreach for this batch.")
    st.markdown(f'<div class="policy-note">{INTERPRETATION_NOTE}</div>', unsafe_allow_html=True)
    st.markdown("</div>", unsafe_allow_html=True)

    if session_model:
        st.warning(
            "These predictions come from a custom session model trained on your upload. "
            "They do not use the frozen production Telco pipeline."
        )

    if predicted_churners > 0:
        _render_batch_priority_actions(flagged)
    else:
        st.success("All scored customers are below the retention outreach threshold.")

    show_flagged = st.checkbox("Show only retention outreach customers", value=False)
    filtered = results[results["retention_recommended"] == 1] if show_flagged else results

    action_tab, details_tab = st.tabs(["Actions & scores", "Full customer details"])
    action_cols = _batch_action_columns(results)

    with action_tab:
        st.caption("Decision columns only — use the second tab for the full customer profile.")
        st.dataframe(
            _format_batch_action_table(filtered[action_cols]),
            use_container_width=True,
            hide_index=True,
        )

    with details_tab:
        st.dataframe(filtered, use_container_width=True, hide_index=True)

    profile_cols = [col for col in FEATURE_COLS if col in results.columns]
    if profile_cols:
        with st.expander("Customer profile columns reference"):
            st.caption("These fields were used as model inputs and appear in the full details tab.")
            st.code(", ".join(profile_cols))

    csv_bytes = filtered.to_csv(index=False).encode("utf-8")
    st.download_button(
        "Download prediction results (CSV)",
        data=csv_bytes,
        file_name="churn_predictions.csv",
        mime="text/csv",
        use_container_width=True,
    )


def render_batch_upload_tab(pipeline: ChurnModelBundle, config: dict[str, Any]) -> None:
    st.subheader("Batch Upload")
    render_upload_help_box()

    uploaded = st.file_uploader("Upload customer CSV", type=["csv"])
    sample_col1, sample_col2 = st.columns([1, 2])
    with sample_col1:
        if st.button("Try with sample Telco data", use_container_width=True):
            sample_df = load_sample_upload_data()
            if sample_df is None:
                st.error("Sample data not found. Run the data cleaning step first.")
            else:
                initialize_upload_dataframe(sample_df)
                st.session_state["batch_upload_signature"] = ("sample_data", len(sample_df))
                st.rerun()

    if uploaded is not None:
        upload_signature = (uploaded.name, uploaded.size)
        if st.session_state.get("batch_upload_signature") != upload_signature:
            try:
                initialize_upload_dataframe(pd.read_csv(uploaded))
                st.session_state["batch_upload_signature"] = upload_signature
            except Exception as exc:
                st.error(f"Could not read CSV: {exc}")
                return

    upload_df = get_upload_dataframe()
    if upload_df is None:
        st.markdown("**Get started**")
        st.caption(
            "Upload your customer CSV or try the sample data button. "
            "Use the templates below if you need a reference."
        )
        render_template_downloads()
        return

    st.markdown(f"**Rows:** {len(upload_df):,} | **Columns:** {len(upload_df.columns)}")
    st.dataframe(upload_df.head(), use_container_width=True, hide_index=True)

    id_col, column_mapping, _readiness = resolve_upload_settings(upload_df)
    report = check_upload_compatibility(upload_df, id_col, column_mapping=column_mapping)
    render_compatibility_report(report)

    if report.is_compatible:
        if st.button("Score all customers", type="primary", use_container_width=True):
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
        render_template_downloads()
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
    st.caption("All 19 model features are editable — every field affects the prediction.")

    service_col1, service_col2 = st.columns(2)
    with service_col1:
        phone = st.selectbox(
            "Phone Service",
            options=options["PhoneService"],
            index=_select_index("PhoneService", options["PhoneService"]),
            key="profile_phone_service",
        )
        st.session_state["field_PhoneService"] = phone
    with service_col2:
        internet = st.selectbox(
            "Internet Service",
            options=options["InternetService"],
            index=_select_index("InternetService", options["InternetService"]),
            key="profile_internet_service",
        )
        st.session_state["field_InternetService"] = internet

    multiple_lines_choices = options_for_phone_dependent(phone, options["MultipleLines"])
    online_security_choices = _internet_addon_options(internet, "OnlineSecurity", options)
    online_backup_choices = _internet_addon_options(internet, "OnlineBackup", options)
    device_protection_choices = _internet_addon_options(internet, "DeviceProtection", options)
    tech_support_choices = _internet_addon_options(internet, "TechSupport", options)
    streaming_tv_choices = _internet_addon_options(internet, "StreamingTV", options)
    streaming_movies_choices = _internet_addon_options(internet, "StreamingMovies", options)

    with st.form("customer_form"):
        col1, col2 = st.columns(2)

        with col1:
            st.markdown("**Account & billing**")
            gender = st.selectbox(
                "Gender",
                options=options["gender"],
                index=_select_index("gender", options["gender"]),
            )
            senior = st.selectbox(
                "Senior Citizen",
                options=[0, 1],
                format_func=lambda value: "Yes" if value == 1 else "No",
                index=_select_index("SeniorCitizen", [0, 1]),
            )
            partner = st.selectbox(
                "Partner",
                options=options["Partner"],
                index=_select_index("Partner", options["Partner"]),
            )
            dependents = st.selectbox(
                "Dependents",
                options=options["Dependents"],
                index=_select_index("Dependents", options["Dependents"]),
            )
            tenure = st.number_input(
                "Tenure (months)",
                min_value=0,
                max_value=100,
                value=int(_session_field("tenure")),
            )
            contract = st.selectbox(
                "Contract",
                options=options["Contract"],
                index=_select_index("Contract", options["Contract"]),
            )
            monthly_charges = st.number_input(
                "Monthly Charges ($)",
                min_value=0.0,
                value=float(_session_field("MonthlyCharges")),
                step=1.0,
            )
            total_charges = st.text_input(
                "Total Charges ($)",
                value=str(_session_field("TotalCharges")),
                help="Leave blank only when tenure is 0 (new customer).",
            )
            paperless = st.selectbox(
                "Paperless Billing",
                options=options["PaperlessBilling"],
                index=_select_index("PaperlessBilling", options["PaperlessBilling"]),
            )
            payment = st.selectbox(
                "Payment Method",
                options=options["PaymentMethod"],
                index=_select_index("PaymentMethod", options["PaymentMethod"]),
            )

        with col2:
            st.markdown("**Phone & internet add-ons**")
            multiple_lines = st.selectbox(
                "Multiple Lines",
                options=multiple_lines_choices,
                index=_select_index("MultipleLines", multiple_lines_choices),
            )
            online_security = st.selectbox(
                "Online Security",
                options=online_security_choices,
                index=_select_index("OnlineSecurity", online_security_choices),
            )
            online_backup = st.selectbox(
                "Online Backup",
                options=online_backup_choices,
                index=_select_index("OnlineBackup", online_backup_choices),
            )
            device_protection = st.selectbox(
                "Device Protection",
                options=device_protection_choices,
                index=_select_index("DeviceProtection", device_protection_choices),
            )
            tech_support = st.selectbox(
                "Tech Support",
                options=tech_support_choices,
                index=_select_index("TechSupport", tech_support_choices),
            )
            streaming_tv = st.selectbox(
                "Streaming TV",
                options=streaming_tv_choices,
                index=_select_index("StreamingTV", streaming_tv_choices),
            )
            streaming_movies = st.selectbox(
                "Streaming Movies",
                options=streaming_movies_choices,
                index=_select_index("StreamingMovies", streaming_movies_choices),
            )

        submitted = st.form_submit_button("Predict Churn", type="primary", use_container_width=True)

    if submitted:
        form_values = {
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
        _sync_profile_session(form_values)
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
        render_developer_api_section()

    tab_single, tab_batch = st.tabs(["Single customer", "Batch upload"])
    with tab_single:
        render_single_customer_tab(pipeline, config, threshold, defaults, options)
    with tab_batch:
        render_batch_upload_tab(pipeline, config)


if __name__ == "__main__":
    main()
