"""Manual functional smoke test for churn intelligence system."""

from __future__ import annotations

import sys
from pathlib import Path

import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))
sys.path.insert(0, str(PROJECT_ROOT / "app"))

checks: list[tuple[str, bool, str]] = []


def ok(name: str, cond: bool, detail: str = "") -> None:
    checks.append((name, bool(cond), detail))


def main() -> int:
    # Imports
    try:
        from upload_ui import load_sample_upload_data, build_mapping_from_ui, build_default_assignments
        from src.inference import score_uploaded_batch, predict_single_customer, prepare_inference_features
        from src.model import load_churn_pipeline
        from src.policy import normalize_service_fields
        from src.upload_compatibility import check_upload_compatibility, normalize_upload_headers
        from src.training_service import train_and_score_upload
        from src.data_split import load_split_from_manifest

        ok("module imports", True)
    except Exception as exc:
        ok("module imports", False, str(exc))
        _report()
        return 1

    bundle = load_churn_pipeline()
    split = load_split_from_manifest()

    # 1) Simplified single-customer path (hidden defaults + visible fields)
    visible = {
        "tenure": 12,
        "Contract": "Month-to-month",
        "MonthlyCharges": 75.0,
        "TotalCharges": 900.0,
        "InternetService": "Fiber optic",
        "OnlineSecurity": "No",
        "TechSupport": "No",
        "StreamingTV": "Yes",
        "StreamingMovies": "No",
        "PaymentMethod": "Electronic check",
    }
    hidden = {
        "gender": "Female",
        "SeniorCitizen": 0,
        "Partner": "Yes",
        "Dependents": "No",
        "PhoneService": "Yes",
        "MultipleLines": "No",
        "OnlineBackup": "No",
        "DeviceProtection": "No",
        "PaperlessBilling": "Yes",
    }
    record, notes = normalize_service_fields({**hidden, **visible})
    ok("single customer normalize", isinstance(notes, list))
    try:
        prepare_inference_features(record)
        result = predict_single_customer(bundle, record)
        ok("single customer predict", 0 <= result["churn_probability"] <= 1)
    except Exception as exc:
        ok("single customer predict", False, str(exc))

    # 2) Internet=No auto-fix path
    no_internet = {**hidden, **visible}
    no_internet["InternetService"] = "No"
    no_internet["OnlineSecurity"] = "Yes"  # invalid combo — should auto-fix
    fixed, fix_notes = normalize_service_fields(no_internet)
    ok("internet=no auto-fix", fixed["OnlineSecurity"] == "No internet service", str(fix_notes))

    # 3) Batch upload compatible file
    batch = split.X_val.iloc[:50].copy()
    batch["customerID"] = split.id_val.iloc[:50].astype(str).tolist()
    batch["Churn"] = split.y_val.iloc[:50].tolist()
    shuffled_cols = ["customerID", "Churn"] + [c for c in batch.columns if c not in {"customerID", "Churn"}]
    batch = batch[shuffled_cols]
    normalized = normalize_upload_headers(batch)
    report = check_upload_compatibility(normalized, "customerID")
    ok("batch compatibility", report.is_compatible, f"missing={report.missing_features}")
    if report.is_compatible:
        results = score_uploaded_batch(bundle, normalized, "customerID")
        ok("batch score rows", len(results) == 50)
        ok("batch sorted", results["churn_probability"].is_monotonic_decreasing)

    # 4) Case-insensitive headers
    renamed = batch.rename(columns={"gender": "Gender", "MonthlyCharges": "monthly_charges"})
    mapping = build_mapping_from_ui(
        renamed,
        build_default_assignments(normalize_upload_headers(renamed)),
    )
    report2 = check_upload_compatibility(renamed, "customerID", column_mapping=mapping)
    ok("column mapper + case headers", report2.is_compatible or report2.readiness_summary["matched_count"] >= 18)

    # 5) Sample data loader
    sample = load_sample_upload_data()
    ok("sample data", sample is not None and len(sample) == 10)

    # 6) Session retrain
    from tests.test_training_service import _synthetic_training_df

    tr = train_and_score_upload(_synthetic_training_df(120), "Churn", "customerID")
    ok("session retrain", len(tr.results) == 120)

    # 7) Invalid upload should fail cleanly
    bad = pd.DataFrame([{"id": 1, "foo": "bar"}])
    bad_report = check_upload_compatibility(bad, "id")
    ok("incompatible upload detected", not bad_report.is_compatible)

    # 8) FastAPI app construct
    try:
        from fastapi.testclient import TestClient
        import api.main as api_main

        with TestClient(api_main.app) as client:
            health = client.get("/health")
            ok("api health", health.status_code == 200 and health.json()["status"] == "ok")
            pred = client.post("/predict_churn", json=_api_payload(split))
            ok("api predict", pred.status_code == 200 and "churn_probability" in pred.json())
    except Exception as exc:
        ok("api health", False, str(exc))
        ok("api predict", False, str(exc))

    return _report()


def _api_payload(split) -> dict:
    record = split.X_val.iloc[3].to_dict()
    record["customerID"] = str(split.id_val.iloc[3])
    return record


def _report() -> int:
    failed = [c for c in checks if not c[1]]
    print("FUNCTIONAL SMOKE TEST")
    for name, passed, detail in checks:
        status = "PASS" if passed else "FAIL"
        line = f"[{status}] {name}"
        if detail:
            line += f" — {detail}"
        print(line)
    print("---")
    print(f"PASSED: {len(checks) - len(failed)} / {len(checks)}")
    return 1 if failed else 0


if __name__ == "__main__":
    raise SystemExit(main())
