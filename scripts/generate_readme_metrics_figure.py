"""Generate README metrics summary figure from saved final test results."""

from __future__ import annotations

import json
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parents[1]
METRICS_PATH = PROJECT_ROOT / "reports" / "final_test_metrics.json"
OUTPUT_PATH = PROJECT_ROOT / "docs" / "images" / "final_test_metrics.png"


def main() -> int:
    if not METRICS_PATH.exists():
        print(f"Missing metrics file: {METRICS_PATH}", file=sys.stderr)
        return 1

    payload = json.loads(METRICS_PATH.read_text(encoding="utf-8"))
    test = payload["test_metrics"]
    threshold = payload["final_threshold"]

    metric_labels = ["Precision", "Recall", "F1", "ROC-AUC", "PR-AUC"]
    metric_values = [
        test["precision"],
        test["recall"],
        test["f1"],
        test["roc_auc"],
        test["pr_auc"],
    ]

    fig = plt.figure(figsize=(11, 5.5), facecolor="white", constrained_layout=True)
    gs = fig.add_gridspec(2, 2, width_ratios=[1.35, 1], height_ratios=[1.1, 1], wspace=0.3, hspace=0.35)

    ax_metrics = fig.add_subplot(gs[:, 0])
    colors = ["#4c72b0", "#55a868", "#8172b2", "#c44e52", "#ccb974"]
    bars = ax_metrics.barh(metric_labels, metric_values, color=colors)
    ax_metrics.set_xlim(0, 1)
    ax_metrics.set_xlabel("Score")
    ax_metrics.set_title(f"Final Test Metrics (threshold = {threshold:.2f})")
    ax_metrics.grid(axis="x", alpha=0.25)
    for bar, value in zip(bars, metric_values):
        ax_metrics.text(
            min(value + 0.02, 0.98),
            bar.get_y() + bar.get_height() / 2,
            f"{value:.3f}",
            va="center",
            fontsize=10,
        )

    ax_summary = fig.add_subplot(gs[0, 1])
    ax_summary.axis("off")
    summary_lines = [
        "Frozen holdout evaluation",
        f"n = {test['n_samples']:,} customers",
        f"Accuracy = {test['accuracy']:.3f}",
        f"Brier score = {test['brier_score']:.3f}",
        "",
        "Confusion matrix @ threshold 0.10",
        f"TN = {test['true_negatives']:,}   FP = {test['false_positives']:,}",
        f"FN = {test['false_negatives']:,}    TP = {test['true_positives']:,}",
        "",
        f"Predicted churners = {test['predicted_churners']:,}",
        f"Simulated cost = ${test['total_cost_usd_simulated']:,}",
    ]
    ax_summary.text(
        0.0,
        1.0,
        "\n".join(summary_lines),
        va="top",
        ha="left",
        fontsize=11,
        family="monospace",
        bbox={"boxstyle": "round,pad=0.6", "facecolor": "#f5f7fb", "edgecolor": "#d9dee7"},
    )

    cm = np.array(
        [
            [test["true_negatives"], test["false_positives"]],
            [test["false_negatives"], test["true_positives"]],
        ]
    )
    ax_cm = fig.add_subplot(gs[1, 1])
    ax_cm.imshow(cm, cmap="Blues")
    ax_cm.set_xticks([0, 1], labels=["Pred No", "Pred Yes"])
    ax_cm.set_yticks([0, 1], labels=["Actual No", "Actual Yes"])
    ax_cm.set_title("Confusion Matrix", fontsize=10)
    for (row, col), value in np.ndenumerate(cm):
        ax_cm.text(col, row, f"{value:,}", ha="center", va="center", color="black", fontsize=9)

    fig.suptitle(
        "Customer Churn Intelligence — Final Test Evaluation",
        fontsize=14,
        fontweight="bold",
        y=0.98,
    )
    OUTPUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(OUTPUT_PATH, dpi=160, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved {OUTPUT_PATH}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
