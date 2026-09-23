"""Run the complete data-to-dashboard artifact pipeline in one command."""

from __future__ import annotations

import json
import sys
from pathlib import Path

import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

from src.analysis.eda import generate_eda_artifacts
from src.config import (
    CLEAN_DATA_FILE,
    CLEANING_REPORT_FILE,
    DATA_PROFILE_FILE,
    EDA_SUMMARY_FILE,
    FIGURE_FILES,
    RAW_DATA_DIR,
)
from src.data.download import download_uci_dataset
from src.data.prepare import clean_student_data, load_raw_student_data, save_clean_dataset
from src.data.understanding import build_data_profile
from src.modeling.evaluation import (
    plot_feature_importance,
    plot_model_comparison,
    plot_prediction_diagnostics,
)
from src.modeling.train import save_training_result, train_and_evaluate


def _write_json(output_path: Path, payload: object) -> None:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w", encoding="utf-8") as output_file:
        json.dump(payload, output_file, indent=2)


def main() -> None:
    print("1/4 Downloading or validating raw data...")
    download_result = download_uci_dataset(RAW_DATA_DIR)
    print(download_result["message"])

    print("2/4 Cleaning data and generating a data profile...")
    raw_data = load_raw_student_data(RAW_DATA_DIR)
    cleaned_data, cleaning_report = clean_student_data(raw_data)
    save_clean_dataset(cleaned_data, CLEAN_DATA_FILE)
    _write_json(CLEANING_REPORT_FILE, cleaning_report)
    _write_json(DATA_PROFILE_FILE, build_data_profile(cleaned_data))

    print("3/4 Producing exploratory charts...")
    eda_artifacts = generate_eda_artifacts(cleaned_data)
    _write_json(EDA_SUMMARY_FILE, eda_artifacts)

    print("4/4 Training and evaluating regression models...")
    result = train_and_evaluate(pd.read_csv(CLEAN_DATA_FILE))
    save_training_result(result)
    plot_model_comparison(result.comparison, FIGURE_FILES["model_comparison"])
    plot_prediction_diagnostics(
        result.test_predictions, FIGURE_FILES["prediction_diagnostics"]
    )
    plot_feature_importance(result.feature_importance, FIGURE_FILES["feature_importance"])

    print("Pipeline complete.")
    print(f"Selected model: {result.best_model_name}")
    for name, value in result.holdout_metrics.items():
        print(f"- held-out {name.upper()}: {value:.3f}")
    print("Start the dashboard with: streamlit run app.py")


if __name__ == "__main__":
    main()
