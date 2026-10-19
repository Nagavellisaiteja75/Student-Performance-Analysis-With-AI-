"""Train reproducible regression models and persist transparent artifacts."""

from __future__ import annotations

import json
from dataclasses import dataclass
from datetime import datetime, timezone
from typing import Any

import joblib
import numpy as np
import pandas as pd
from sklearn.ensemble import GradientBoostingRegressor, RandomForestRegressor
from sklearn.inspection import permutation_importance
from sklearn.linear_model import Ridge
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.model_selection import KFold, cross_validate, train_test_split
from sklearn.pipeline import Pipeline

from src.config import (
    CV_FOLDS,
    FEATURE_IMPORTANCE_FILE,
    MODEL_COMPARISON_FILE,
    MODEL_FILE,
    MODEL_METADATA_FILE,
    PERMUTATION_REPEATS,
    RANDOM_STATE,
    TARGET_COLUMN,
    TEST_PREDICTIONS_FILE,
    TEST_SIZE,
)
from src.modeling.evaluation import add_support_bands
from src.modeling.features import FeatureSpec, build_feature_spec, build_input_metadata, build_preprocessor


@dataclass
class TrainingResult:
    """In-memory output from a training run before it is written to disk."""

    best_estimator: Pipeline
    best_model_name: str
    feature_spec: FeatureSpec
    comparison: pd.DataFrame
    holdout_metrics: dict[str, float]
    test_predictions: pd.DataFrame
    feature_importance: pd.DataFrame
    metadata: dict[str, Any]


def _candidate_models(feature_spec: FeatureSpec) -> dict[str, Pipeline]:
    """Build comparable model pipelines with a fresh preprocessor per candidate."""

    regressors = {
        "Ridge Regression": Ridge(alpha=5.0),
        "Random Forest": RandomForestRegressor(
            n_estimators=300,
            min_samples_leaf=2,
            max_features=0.8,
            random_state=RANDOM_STATE,
            n_jobs=-1,
        ),
        "Gradient Boosting": GradientBoostingRegressor(
            n_estimators=250,
            learning_rate=0.04,
            max_depth=2,
            min_samples_leaf=3,
            loss="huber",
            random_state=RANDOM_STATE,
        ),
    }
    return {
        name: Pipeline(
            steps=[("preprocessor", build_preprocessor(feature_spec)), ("model", estimator)]
        )
        for name, estimator in regressors.items()
    }


def train_and_evaluate(data: pd.DataFrame) -> TrainingResult:
    """Compare models via CV, select by CV MAE, then evaluate one test holdout."""

    feature_spec = build_feature_spec(data)
    if len(data) < 20:
        raise ValueError("At least 20 cleaned rows are required for a reliable training run.")

    features = data[feature_spec.feature_columns].copy()
    target = pd.to_numeric(data[TARGET_COLUMN], errors="raise")
    train_features, test_features, train_target, test_target = train_test_split(
        features,
        target,
        test_size=TEST_SIZE,
        random_state=RANDOM_STATE,
    )

    folds = min(CV_FOLDS, len(train_features))
    if folds < 2:
        raise ValueError("Training data is too small for cross-validation.")
    cross_validator = KFold(n_splits=folds, shuffle=True, random_state=RANDOM_STATE)
    scoring = {
        "mae": "neg_mean_absolute_error",
        "rmse": "neg_root_mean_squared_error",
        "r2": "r2",
    }

    candidates = _candidate_models(feature_spec)
    comparison_rows: list[dict[str, float | str]] = []
    for model_name, estimator in candidates.items():
        cross_validation = cross_validate(
            estimator,
            train_features,
            train_target,
            cv=cross_validator,
            scoring=scoring,
            n_jobs=1,
            return_train_score=False,
        )
        comparison_rows.append(
            {
                "model": model_name,
                "cv_mae_mean": float(-np.mean(cross_validation["test_mae"])),
                "cv_mae_std": float(np.std(-cross_validation["test_mae"])),
                "cv_rmse_mean": float(-np.mean(cross_validation["test_rmse"])),
                "cv_rmse_std": float(np.std(-cross_validation["test_rmse"])),
                "cv_r2_mean": float(np.mean(cross_validation["test_r2"])),
                "cv_r2_std": float(np.std(cross_validation["test_r2"])),
            }
        )

    comparison = pd.DataFrame(comparison_rows).sort_values(
        "cv_mae_mean", ascending=True, ignore_index=True
    )
    best_model_name = str(comparison.iloc[0]["model"])
    best_estimator = candidates[best_model_name].fit(train_features, train_target)
    predicted_grades = best_estimator.predict(test_features)

    holdout_metrics = {
        "mae": float(mean_absolute_error(test_target, predicted_grades)),
        "rmse": float(np.sqrt(mean_squared_error(test_target, predicted_grades))),
        "r2": float(r2_score(test_target, predicted_grades)),
    }
    test_predictions = pd.DataFrame(
        {
            "actual_grade": test_target.to_numpy(),
            "predicted_grade": predicted_grades,
            "residual": test_target.to_numpy() - predicted_grades,
        }
    )
    test_predictions = pd.concat(
        [test_predictions, add_support_bands(test_predictions["predicted_grade"])], axis=1
    )

    permutation = permutation_importance(
        best_estimator,
        test_features,
        test_target,
        scoring="neg_mean_absolute_error",
        n_repeats=PERMUTATION_REPEATS,
        random_state=RANDOM_STATE,
        n_jobs=1,
    )
    feature_importance = pd.DataFrame(
        {
            "feature": feature_spec.feature_columns,
            "importance_mean": permutation.importances_mean,
            "importance_std": permutation.importances_std,
        }
    ).sort_values("importance_mean", ascending=False, ignore_index=True)

    metadata = {
        "generated_at_utc": datetime.now(timezone.utc).isoformat(),
        "training_rows": int(len(train_features)),
        "test_rows": int(len(test_features)),
        "random_state": RANDOM_STATE,
        "test_size": TEST_SIZE,
        "cv_folds": folds,
        "selection_rule": "Candidate with the lowest mean cross-validation MAE on training data.",
        "selected_model": best_model_name,
        "holdout_metrics": holdout_metrics,
        "model_input": build_input_metadata(data, feature_spec),
        "explainability_method": (
            "Permutation importance on the held-out test set, scored by MAE increase."
        ),
        "model_use_note": (
            "Predictions are support indicators for human review, not automated academic decisions."
        ),
    }
    return TrainingResult(
        best_estimator=best_estimator,
        best_model_name=best_model_name,
        feature_spec=feature_spec,
        comparison=comparison,
        holdout_metrics=holdout_metrics,
        test_predictions=test_predictions,
        feature_importance=feature_importance,
        metadata=metadata,
    )


def save_training_result(result: TrainingResult) -> None:
    """Write the fitted model and all tabular artifacts used by the dashboard."""

    for output_path in (
        MODEL_FILE,
        MODEL_COMPARISON_FILE,
        TEST_PREDICTIONS_FILE,
        FEATURE_IMPORTANCE_FILE,
        MODEL_METADATA_FILE,
    ):
        output_path.parent.mkdir(parents=True, exist_ok=True)

    joblib.dump(result.best_estimator, MODEL_FILE)
    result.comparison.to_csv(MODEL_COMPARISON_FILE, index=False)
    result.test_predictions.to_csv(TEST_PREDICTIONS_FILE, index=False)
    result.feature_importance.to_csv(FEATURE_IMPORTANCE_FILE, index=False)
    with MODEL_METADATA_FILE.open("w", encoding="utf-8") as metadata_file:
        json.dump(result.metadata, metadata_file, indent=2)
