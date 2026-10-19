"""Download the UCI Student Performance raw files."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

from src.config import RAW_DATA_DIR, UCI_DATASET_PAGE
from src.data.download import download_uci_dataset


def main() -> None:
    parser = argparse.ArgumentParser(description="Download the UCI Student Performance dataset.")
    parser.add_argument(
        "--force",
        action="store_true",
        help="Download again even when both expected raw files already exist.",
    )
    arguments = parser.parse_args()

    result = download_uci_dataset(RAW_DATA_DIR, force=arguments.force)
    print(result["message"])
    print(f"Dataset page: {UCI_DATASET_PAGE}")
    for file_path in result["files"]:
        print(f"- {file_path}")


if __name__ == "__main__":
    main()
