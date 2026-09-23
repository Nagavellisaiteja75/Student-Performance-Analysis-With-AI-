"""Metrics, explanatory plots, and non-diagnostic support indicators."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from src.analysis.eda import PLOT_COLORS
from src.config import GRADE_MAX, GRADE_MIN


def support_indicator(predicted_grade: float) -> dict[str, str]:
    """Map a prediction to a transparent support-priority suggestion.

    These thresholds are communication aids, not diagnoses, placements, or
    automated decisions. Staff must review context before offering support.
    """

    if predicted_grade < 10:
        return {
            "priority": "High-priority outreach",
            "band": "Predicted below pass threshold",
            "color": "red",
            "recommended_action": (
                "Arrange a timely, supportive check-in; review barriers, attendance, "
                "and available academic resources with the student."
            ),
        }
    if predicted_grade < 12:
        return {
            "priority": "Targeted support",
            "band": "Predicted near pass threshold",
            "color": "orange",
            "recommended_action": (
                "Offer targeted study planning or tutoring and agree on a follow-up date."
            ),
        }
    if predicted_grade < 14:
        return {
            "priority": "Monitor and encourage",
            "band": "Predicted developing performance",
            "color": "blue",
            "recommended_action": (
                "Share progress feedback and check whether additional learning support is useful."
            ),
        }
    return {
        "priority": "Continue strengths-based support",
        "band": "Predicted on-track performance",
        "color": "green",
        "recommended_action": (
            "Recognize effective habits and keep regular, student-led progress conversations."
        ),
    }


def add_support_bands(predictions: pd.Series) -> pd.DataFrame:
    """Return prediction labels used in exported test-prediction artifacts."""

    labels = [support_indicator(float(value))["band"] for value in predictions]
    priorities = [support_indicator(float(value))["priority"] for value in predictions]
    return pd.DataFrame({"support_band": labels, "support_priority": priorities})


def _save_figure(figure: plt.Figure, output_path: Path) -> None:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    figure.tight_layout()
    figure.savefig(output_path, dpi=180, bbox_inches="tight")
    plt.close(figure)


def plot_model_comparison(comparison: pd.DataFrame, output_path: Path) -> None:
    """Plot cross-validation MAE with variability for each candidate model."""

    ordered = comparison.sort_values("cv_mae_mean", ascending=True)
    figure, axis = plt.subplots(figsize=(8, 4.5))
    axis.bar(
        ordered["model"],
        ordered["cv_mae_mean"],
        yerr=ordered["cv_mae_std"],
        capsize=4,
        color=[PLOT_COLORS["teal"], PLOT_COLORS["blue"], PLOT_COLORS["gold"]],
    )
    axis.set_title("Model Comparison: Cross-Validation MAE", weight="bold")
    axis.set_xlabel("Regression model")
    axis.set_ylabel("Mean absolute error (lower is better)")
    axis.tick_params(axis="x", rotation=12)
    _save_figure(figure, output_path)


def plot_prediction_diagnostics(predictions: pd.DataFrame, output_path: Path) -> None:
    """Plot held-out actual versus predicted grades after model selection."""

    figure, axis = plt.subplots(figsize=(6.5, 5.5))
    axis.scatter(
        predictions["actual_grade"],
        predictions["predicted_grade"],
        alpha=0.7,
        color=PLOT_COLORS["blue"],
        edgecolors="white",
        linewidths=0.4,
    )
    axis.plot(
        [GRADE_MIN, GRADE_MAX],
        [GRADE_MIN, GRADE_MAX],
        linestyle="--",
        color=PLOT_COLORS["coral"],
        label="Perfect prediction",
    )
    axis.set_xlim(GRADE_MIN, GRADE_MAX)
    axis.set_ylim(GRADE_MIN, GRADE_MAX)
    axis.set_title("Held-Out Predictions vs. Actual Grades", weight="bold")
    axis.set_xlabel("Actual final grade")
    axis.set_ylabel("Predicted final grade")
    axis.legend(frameon=False)
    _save_figure(figure, output_path)


def plot_feature_importance(importance: pd.DataFrame, output_path: Path) -> None:
    """Plot the largest permutation-importance results for the chosen model."""

    top_features = importance.head(15).sort_values("importance_mean", ascending=True)
    figure, axis = plt.subplots(figsize=(8, 6))
    axis.barh(
        top_features["feature"],
        top_features["importance_mean"],
        xerr=top_features["importance_std"],
        color=PLOT_COLORS["teal"],
        alpha=0.9,
    )
    axis.set_title("Permutation Importance on Held-Out Data", weight="bold")
    axis.set_xlabel("Increase in MAE when feature is shuffled")
    axis.set_ylabel("Feature")
    _save_figure(figure, output_path)


def metric_summary_text(metrics: dict[str, float]) -> dict[str, str]:
    """Format generated metrics consistently for dashboard display."""

    return {
        "MAE": f"{metrics['mae']:.3f}",
        "RMSE": f"{metrics['rmse']:.3f}",
        "R²": f"{metrics['r2']:.3f}",
    }
