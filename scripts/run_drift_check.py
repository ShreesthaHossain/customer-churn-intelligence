"""Compare a scored batch or validation slice against training feature baselines."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.data_cleaning import IDENTIFIER_COL
from src.data_split import load_split_from_manifest
from src.inference import score_uploaded_batch
from src.model import load_churn_pipeline
from src.monitoring import compute_drift_report


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--csv",
        type=Path,
        help="Optional Telco-compatible CSV to score and compare. Defaults to validation slice.",
    )
    parser.add_argument(
        "--rows",
        type=int,
        default=200,
        help="Validation rows to use when --csv is not provided.",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=PROJECT_ROOT / "reports" / "monitoring" / "drift_report.json",
        help="Where to write the drift report JSON.",
    )
    args = parser.parse_args()

    bundle = load_churn_pipeline()

    if args.csv:
        upload_df = pd.read_csv(args.csv)
        id_col = IDENTIFIER_COL if IDENTIFIER_COL in upload_df.columns else upload_df.columns[0]
        scored = score_uploaded_batch(bundle, upload_df, id_col)
    else:
        split = load_split_from_manifest()
        sample = split.X_val.iloc[: args.rows].copy()
        sample[IDENTIFIER_COL] = split.id_val.iloc[: args.rows].astype(str).tolist()
        scored = score_uploaded_batch(bundle, sample, IDENTIFIER_COL)

    from src.data_separation import FEATURE_COLS

    feature_cols = [col for col in FEATURE_COLS if col in scored.columns]
    report = compute_drift_report(scored[feature_cols])
    report["source"] = str(args.csv) if args.csv else f"validation_slice:{args.rows}"

    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2), encoding="utf-8")

    print(f"Drift status: {report['status']}")
    if report["alerts"]:
        print("Alerts:")
        for alert in report["alerts"]:
            print(f"  - {alert}")
    print(f"Saved report to {args.output}")
    return 0 if report["status"] == "ok" else 1


if __name__ == "__main__":
    raise SystemExit(main())
