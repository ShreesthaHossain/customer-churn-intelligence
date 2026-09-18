"""One-off project audit script (run manually)."""

from __future__ import annotations

import json
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.config import load_chosen_threshold_config, load_model_config
from src.data_split import load_split_from_manifest
from src.inference import build_churn_prediction_response, predict_churn_probability
from src.model import load_churn_pipeline


def main() -> int:
    checks: list[tuple[str, bool, str]] = []

    def ok(name: str, cond: bool, detail: str = "") -> None:
        checks.append((name, bool(cond), detail))

    root = PROJECT_ROOT
    ok("churn_pipeline.joblib", (root / "models/churn_pipeline.joblib").exists())
    ok("model_config.json", (root / "models/model_config.json").exists())
    ok("chosen_threshold.json", (root / "reports/chosen_threshold.json").exists())
    ok("final_test_metrics.json", (root / "reports/final_test_metrics.json").exists())
    ok("cleaned data", (root / "data/processed/cleaned_churn.csv").exists())
    ok("split manifest", (root / "data/processed/split_manifest.json").exists())
    ok("raw data", (root / "data/raw/WA_Fn-UseC_-Telco-Customer-Churn.csv").exists())

    bundle = load_churn_pipeline()
    config = load_model_config()
    chosen = load_chosen_threshold_config()
    split = load_split_from_manifest()

    ok(
        "threshold aligned",
        bundle.threshold == config["decision_threshold"] == chosen["threshold"] == 0.1,
        f"{bundle.threshold}, {config['decision_threshold']}, {chosen['threshold']}",
    )
    ok(
        "split sizes",
        len(split.X_train) == 4930 and len(split.X_val) == 1056 and len(split.X_test) == 1057,
        f"{len(split.X_train)},{len(split.X_val)},{len(split.X_test)}",
    )

    for name, X in [("val", split.X_val), ("test", split.X_test)]:
        proba = predict_churn_probability(bundle, X)
        ok(
            f"{name} proba bounds",
            proba.min() >= 0 and proba.max() <= 1,
            f"min={proba.min():.4f}, max={proba.max():.4f}",
        )

    ftm = json.loads((root / "reports/final_test_metrics.json").read_text(encoding="utf-8"))
    tm = ftm["test_metrics"]
    ok(
        "final test metrics present",
        all(k in tm for k in ["precision", "recall", "f1", "roc_auc", "pr_auc", "brier_score", "predicted_churners"]),
    )

    record = split.X_val.iloc[0].to_dict()
    record["customerID"] = str(split.id_val.iloc[0])
    direct = build_churn_prediction_response(bundle, record, model_config=config)
    ok(
        "direct inference keys",
        all(
            k in direct
            for k in [
                "churn_probability",
                "threshold",
                "prediction",
                "risk_level",
                "recommended_action",
                "model_version",
            ]
        ),
    )

    # App/API import smoke
    try:
        import api.main  # noqa: F401
        ok("api.main import", True)
    except Exception as exc:
        ok("api.main import", False, str(exc))

    failed = [c for c in checks if not c[1]]
    print("AUDIT CHECKS")
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
