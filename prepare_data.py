"""Create the cleaned dataset and data-understanding artifacts."""

from __future__ import annotations

import json
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

from src.config import CLEAN_DATA_FILE, CLEANING_REPORT_FILE, DATA_PROFILE_FILE, RAW_DATA_DIR
from src.data.prepare import clean_student_data, load_raw_student_data, save_clean_dataset
from src.data.understanding import build_data_profile


def main() -> None:
    raw_data = load_raw_student_data(RAW_DATA_DIR)
    cleaned_data, cleaning_report = clean_student_data(raw_data)
    data_profile = build_data_profile(cleaned_data)

    save_clean_dataset(cleaned_data, CLEAN_DATA_FILE)
    for output_path, payload in (
        (CLEANING_REPORT_FILE, cleaning_report),
        (DATA_PROFILE_FILE, data_profile),
    ):
        output_path.parent.mkdir(parents=True, exist_ok=True)
        with output_path.open("w", encoding="utf-8") as output_file:
            json.dump(payload, output_file, indent=2)

    print(f"Created cleaned dataset: {CLEAN_DATA_FILE}")
    print(f"Rows after cleaning: {len(cleaned_data)}")
    print(f"Data profile: {DATA_PROFILE_FILE}")
    print(f"Cleaning report: {CLEANING_REPORT_FILE}")


if __name__ == "__main__":
    main()
