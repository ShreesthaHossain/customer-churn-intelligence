"""Build processed data and model artifacts for CI, Docker, and fresh clones."""

from __future__ import annotations

import argparse
import sys
import urllib.request
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.config import churn_pipeline_path, model_config_path, project_root
from src.data_cleaning import RAW_FILENAME, run_cleaning_pipeline
from src.data_separation import run_separation_pipeline
from src.data_split import run_split_pipeline
from src.model import run_training_pipeline, save_deployment_artifacts

DEFAULT_DATA_URL = (
    "https://raw.githubusercontent.com/IBM/telco-customer-churn-on-icp4d/"
    "master/data/Telco-Customer-Churn.csv"
)


def download_raw_dataset(url: str, destination: Path) -> Path:
    """Download the public Telco churn dataset when raw data is missing."""
    destination.parent.mkdir(parents=True, exist_ok=True)
    print(f"Downloading dataset to {destination} ...")
    urllib.request.urlretrieve(url, destination)
    return destination


def artifacts_present() -> bool:
    return churn_pipeline_path().exists() and model_config_path().exists()


def bootstrap(*, force: bool = False, data_url: str = DEFAULT_DATA_URL) -> None:
    root = project_root()
    raw_path = root / "data" / "raw" / RAW_FILENAME

    if artifacts_present() and not force:
        print("Deployment artifacts already present — skipping bootstrap.")
        return

    if not raw_path.exists():
        download_raw_dataset(data_url, raw_path)

    print("Running data cleaning ...")
    run_cleaning_pipeline(raw_path=raw_path)

    print("Running feature/target separation ...")
    run_separation_pipeline()

    print("Creating train/validation/test split ...")
    split, _, _ = run_split_pipeline()

    print("Training calibrated model on training data ...")
    bundle, _ = run_training_pipeline(split)

    print("Saving deployment artifacts ...")
    pipeline_path, config_path = save_deployment_artifacts(bundle, split)
    print(f"Saved pipeline: {pipeline_path}")
    print(f"Saved config:   {config_path}")


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--force",
        action="store_true",
        help="Rebuild artifacts even when models/ already exists.",
    )
    parser.add_argument(
        "--data-url",
        default=DEFAULT_DATA_URL,
        help="Public CSV URL used when data/raw is missing.",
    )
    args = parser.parse_args()

    try:
        bootstrap(force=args.force, data_url=args.data_url)
    except Exception as exc:
        print(f"Bootstrap failed: {exc}", file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
