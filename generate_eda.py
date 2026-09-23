"""Generate reproducible EDA charts from the cleaned dataset."""

from __future__ import annotations

import json
import sys
from pathlib import Path

import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

from src.analysis.eda import generate_eda_artifacts
from src.config import CLEAN_DATA_FILE, EDA_SUMMARY_FILE


def main() -> None:
    if not CLEAN_DATA_FILE.exists():
        raise FileNotFoundError(
            "Cleaned data is missing. Run `python scripts/prepare_data.py` first."
        )

    data = pd.read_csv(CLEAN_DATA_FILE)
    artifacts = generate_eda_artifacts(data)
    EDA_SUMMARY_FILE.parent.mkdir(parents=True, exist_ok=True)
    with EDA_SUMMARY_FILE.open("w", encoding="utf-8") as output_file:
        json.dump(artifacts, output_file, indent=2)

    print("Created EDA artifacts:")
    for name, path in artifacts.items():
        print(f"- {name}: {path}")


if __name__ == "__main__":
    main()
