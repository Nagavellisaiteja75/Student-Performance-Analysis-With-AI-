"""Leakage-aware feature selection and preprocessing definitions."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

from src.config import MODEL_EXCLUDED_COLUMNS, NUMERIC_COLUMNS


@dataclass(frozen=True)
class FeatureSpec:
    """Column groups required to reproduce the model input transformation."""

    feature_columns: list[str]
    numeric_columns: list[str]
    categorical_columns: list[str]
    excluded_columns: list[str]


def build_feature_spec(data: pd.DataFrame) -> FeatureSpec:
    """Build an early-intervention feature set from the cleaned dataset.

    G1 and G2 are deliberately excluded because they are prior period grades
    strongly related to the final grade. This keeps the default model useful
    before term-end assessment information is available.
    """

    feature_columns = [
        str(column) for column in data.columns if column not in MODEL_EXCLUDED_COLUMNS
    ]
    numeric_columns = [
        column for column in NUMERIC_COLUMNS if column in feature_columns
    ]
    categorical_columns = [
        column for column in feature_columns if column not in numeric_columns
    ]

    if not numeric_columns and not categorical_columns:
        raise ValueError("No model features remain after applying leakage exclusions.")

    return FeatureSpec(
        feature_columns=feature_columns,
        numeric_columns=numeric_columns,
        categorical_columns=categorical_columns,
        excluded_columns=[column for column in MODEL_EXCLUDED_COLUMNS if column in data],
    )


def build_preprocessor(feature_spec: FeatureSpec) -> ColumnTransformer:
    """Create a fitted-inside-pipeline transformer with leakage-safe imputation."""

    numeric_pipeline = Pipeline(
        steps=[
            ("imputer", SimpleImputer(strategy="median")),
            ("scaler", StandardScaler()),
        ]
    )
    categorical_pipeline = Pipeline(
        steps=[
            ("imputer", SimpleImputer(strategy="most_frequent")),
            (
                "one_hot_encoder",
                OneHotEncoder(handle_unknown="ignore", sparse_output=False),
            ),
        ]
    )

    transformers = []
    if feature_spec.numeric_columns:
        transformers.append(("numeric", numeric_pipeline, feature_spec.numeric_columns))
    if feature_spec.categorical_columns:
        transformers.append(
            ("categorical", categorical_pipeline, feature_spec.categorical_columns)
        )

    return ColumnTransformer(transformers=transformers, remainder="drop")


def build_input_metadata(data: pd.DataFrame, feature_spec: FeatureSpec) -> dict[str, Any]:
    """Create native-Python input defaults and ranges for the Streamlit form."""

    numeric_features: dict[str, dict[str, Any]] = {}
    for column in feature_spec.numeric_columns:
        values = pd.to_numeric(data[column], errors="coerce").dropna()
        if values.empty:
            raise ValueError(f"Cannot create dashboard defaults for empty feature: {column}")
        minimum = float(values.min())
        maximum = float(values.max())
        default = float(values.median())
        is_integer = bool(np.allclose(values.to_numpy(), np.round(values.to_numpy())))
        numeric_features[column] = {
            "minimum": int(minimum) if is_integer else minimum,
            "maximum": int(maximum) if is_integer else maximum,
            "default": int(round(default)) if is_integer else default,
            "integer": is_integer,
        }

    categorical_features: dict[str, dict[str, Any]] = {}
    for column in feature_spec.categorical_columns:
        values = sorted(str(value) for value in data[column].dropna().unique())
        if not values:
            raise ValueError(f"Cannot create dashboard options for empty feature: {column}")
        mode = str(data[column].mode(dropna=True).iloc[0])
        categorical_features[column] = {"options": values, "default": mode}

    return {
        "feature_columns": feature_spec.feature_columns,
        "numeric_features": numeric_features,
        "categorical_features": categorical_features,
        "excluded_for_early_intervention": {
            "G3": "The final grade is the prediction target.",
            "G1": "First-period grade is excluded to avoid assessment-grade leakage.",
            "G2": "Second-period grade is excluded to avoid assessment-grade leakage.",
        },
        "availability_note": (
            "Only enter values genuinely available at the time of the intervention. "
            "For example, absences should reflect a defined attendance snapshot."
        ),
    }
