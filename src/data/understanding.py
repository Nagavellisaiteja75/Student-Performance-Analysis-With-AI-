"""Data-understanding summaries used in reports and the dashboard."""

from __future__ import annotations

from typing import Any

import pandas as pd

from src.config import SUBJECT_COLUMN, TARGET_COLUMN
from src.data.prepare import get_numeric_columns


def build_data_profile(data: pd.DataFrame) -> dict[str, Any]:
    """Create a JSON-serializable overview without fabricating interpretations."""

    numeric_columns = get_numeric_columns(data)
    numeric_summary = (
        data[numeric_columns]
        .describe()
        .round(3)
        .transpose()
        .reset_index(names="feature")
        .to_dict(orient="records")
    )

    return {
        "rows": int(len(data)),
        "columns": int(data.shape[1]),
        "course_records": {
            str(subject): int(count)
            for subject, count in data[SUBJECT_COLUMN].value_counts().sort_index().items()
        },
        "column_types": {column: str(dtype) for column, dtype in data.dtypes.items()},
        "missing_values": {
            column: int(count) for column, count in data.isna().sum().items()
        },
        "target": {
            "name": TARGET_COLUMN,
            "minimum": float(data[TARGET_COLUMN].min()),
            "maximum": float(data[TARGET_COLUMN].max()),
            "mean": round(float(data[TARGET_COLUMN].mean()), 3),
            "median": round(float(data[TARGET_COLUMN].median()), 3),
            "zero_grade_records": int((data[TARGET_COLUMN] == 0).sum()),
        },
        "numeric_summary": numeric_summary,
        "data_understanding_notes": [
            "Each row represents a course record, not a verified unique person across both subjects.",
            "The source combines school reports with questionnaire responses.",
            "G3 is the final grade on a 0–20 scale and is the regression target.",
        ],
    }
