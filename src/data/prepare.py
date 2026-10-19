"""Load, clean, validate, and persist the Student Performance dataset."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from src.config import (
    DATASET_FILES,
    GRADE_MAX,
    GRADE_MIN,
    NUMERIC_COLUMNS,
    SUBJECT_COLUMN,
    TARGET_COLUMN,
)


class DataValidationError(ValueError):
    """Raised when the raw files do not match the expected dataset schema."""


def load_raw_student_data(raw_directory: str | Path) -> pd.DataFrame:
    """Read both course files and append a clearly labelled subject field."""

    raw_path = Path(raw_directory)
    frames: list[pd.DataFrame] = []

    for filename, subject in DATASET_FILES.items():
        file_path = raw_path / filename
        try:
            course_data = pd.read_csv(file_path, sep=";", encoding="utf-8")
        except FileNotFoundError as error:
            raise FileNotFoundError(
                f"Missing {filename}. Run `python scripts/download_data.py` first."
            ) from error
        course_data[SUBJECT_COLUMN] = subject
        frames.append(course_data)

    combined = pd.concat(frames, ignore_index=True, sort=False)
    if combined.empty:
        raise DataValidationError("The raw data files were loaded but contain no records.")
    return combined


def clean_student_data(data: pd.DataFrame) -> tuple[pd.DataFrame, dict[str, Any]]:
    """Apply transparent, minimal cleaning appropriate for the UCI source data."""

    cleaned = data.copy()
    missing_required = {TARGET_COLUMN, SUBJECT_COLUMN}.difference(cleaned.columns)
    if missing_required:
        raise DataValidationError(
            f"Required columns are missing from the raw data: {sorted(missing_required)}"
        )

    rows_before_cleaning = len(cleaned)
    text_columns = [column for column in cleaned.columns if column not in NUMERIC_COLUMNS]
    for column in text_columns:
        cleaned[column] = (
            cleaned[column].astype("string").str.strip().replace({"": pd.NA})
        )

    numeric_columns_present = [
        column for column in NUMERIC_COLUMNS if column in cleaned.columns
    ]
    conversion_failures: dict[str, int] = {}
    for column in numeric_columns_present:
        original_non_null = cleaned[column].notna().sum()
        cleaned[column] = pd.to_numeric(cleaned[column], errors="coerce")
        conversion_failures[column] = int(original_non_null - cleaned[column].notna().sum())

    duplicate_rows = int(cleaned.duplicated().sum())
    cleaned = cleaned.drop_duplicates().copy()

    invalid_target = cleaned[TARGET_COLUMN].isna() | ~cleaned[TARGET_COLUMN].between(
        GRADE_MIN, GRADE_MAX
    )
    invalid_target_rows = int(invalid_target.sum())
    cleaned = cleaned.loc[~invalid_target].reset_index(drop=True)

    report = {
        "rows_before_cleaning": rows_before_cleaning,
        "exact_duplicate_rows_removed": duplicate_rows,
        "invalid_or_missing_target_rows_removed": invalid_target_rows,
        "rows_after_cleaning": len(cleaned),
        "numeric_conversion_failures": conversion_failures,
        "remaining_missing_values": {
            column: int(count)
            for column, count in cleaned.isna().sum().items()
            if count > 0
        },
        "cleaning_notes": [
            "Whitespace is normalized in text fields.",
            "Numeric UCI columns are coerced to numeric values.",
            "Only exact duplicate records are removed; math and Portuguese course records remain distinct.",
            "Records with a missing or out-of-range final grade are removed.",
            "Any remaining feature-level missing values are imputed inside each model pipeline.",
        ],
    }
    return cleaned, report


def save_clean_dataset(data: pd.DataFrame, output_file: str | Path) -> None:
    """Save a clean, portable CSV without serializing pandas index values."""

    output_path = Path(output_file)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    data.to_csv(output_path, index=False)


def get_numeric_columns(data: pd.DataFrame) -> list[str]:
    """Return the UCI numeric fields that are present in a given dataset."""

    return [column for column in NUMERIC_COLUMNS if column in data.columns]
