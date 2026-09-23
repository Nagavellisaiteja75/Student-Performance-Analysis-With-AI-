"""Create reproducible exploratory-data-analysis visualizations."""

from __future__ import annotations

from pathlib import Path

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from src.config import FIGURE_FILES, SUBJECT_COLUMN, TARGET_COLUMN
from src.data.prepare import get_numeric_columns


PLOT_COLORS = {
    "navy": "#12355B",
    "blue": "#2B7A9A",
    "teal": "#2A9D8F",
    "gold": "#E9C46A",
    "orange": "#F4A261",
    "coral": "#E76F51",
}


def _save_figure(figure: plt.Figure, output_path: Path) -> None:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    figure.tight_layout()
    figure.savefig(output_path, dpi=180, bbox_inches="tight")
    plt.close(figure)


def generate_eda_artifacts(data: pd.DataFrame) -> dict[str, str]:
    """Generate charts from the cleaned data and return their file locations."""

    created: dict[str, str] = {}

    figure, axis = plt.subplots(figsize=(8, 4.5))
    bins = np.arange(-0.5, 21.5, 1)
    axis.hist(data[TARGET_COLUMN], bins=bins, color=PLOT_COLORS["blue"], edgecolor="white")
    axis.set_title("Distribution of Final Grades (G3)", weight="bold")
    axis.set_xlabel("Final grade (0–20)")
    axis.set_ylabel("Course records")
    axis.set_xticks(range(0, 21, 2))
    _save_figure(figure, FIGURE_FILES["target_distribution"])
    created["target_distribution"] = str(FIGURE_FILES["target_distribution"])

    subjects = sorted(data[SUBJECT_COLUMN].dropna().unique())
    grade_groups = [
        data.loc[data[SUBJECT_COLUMN] == subject, TARGET_COLUMN].dropna()
        for subject in subjects
    ]
    figure, axis = plt.subplots(figsize=(7, 4.5))
    try:
        boxplot = axis.boxplot(grade_groups, tick_labels=subjects, patch_artist=True)
    except TypeError:
        boxplot = axis.boxplot(grade_groups, labels=subjects, patch_artist=True)
    for patch, color in zip(boxplot["boxes"], [PLOT_COLORS["teal"], PLOT_COLORS["gold"]]):
        patch.set_facecolor(color)
        patch.set_alpha(0.85)
    axis.set_title("Final Grades by Course Subject", weight="bold")
    axis.set_xlabel("Course subject")
    axis.set_ylabel("Final grade (G3)")
    axis.set_ylim(GRADE_LOWER_BOUND, GRADE_UPPER_BOUND)
    _save_figure(figure, FIGURE_FILES["subject_grade_distribution"])
    created["subject_grade_distribution"] = str(FIGURE_FILES["subject_grade_distribution"])

    numeric_columns = get_numeric_columns(data)
    correlation = data[numeric_columns].corr(numeric_only=True)
    figure, axis = plt.subplots(figsize=(10, 8))
    image = axis.imshow(correlation, cmap="YlGnBu", vmin=-1, vmax=1)
    axis.set_xticks(range(len(correlation.columns)), correlation.columns, rotation=75, ha="right")
    axis.set_yticks(range(len(correlation.index)), correlation.index)
    axis.set_title("Correlation Among Numeric Features", weight="bold")
    figure.colorbar(image, ax=axis, shrink=0.8, label="Pearson correlation")
    _save_figure(figure, FIGURE_FILES["numeric_correlation"])
    created["numeric_correlation"] = str(FIGURE_FILES["numeric_correlation"])

    figure, axis = plt.subplots(figsize=(8, 4.5))
    for subject, color in zip(subjects, [PLOT_COLORS["teal"], PLOT_COLORS["orange"]]):
        subset = data.loc[data[SUBJECT_COLUMN] == subject]
        axis.scatter(
            subset["absences"],
            subset[TARGET_COLUMN],
            label=subject.title(),
            alpha=0.65,
            color=color,
            edgecolors="none",
        )
    axis.set_title("Absences and Final Grade", weight="bold")
    axis.set_xlabel("Recorded absences")
    axis.set_ylabel("Final grade (G3)")
    axis.legend(frameon=False)
    _save_figure(figure, FIGURE_FILES["absences_vs_grade"])
    created["absences_vs_grade"] = str(FIGURE_FILES["absences_vs_grade"])

    return created


GRADE_LOWER_BOUND = -0.5
GRADE_UPPER_BOUND = 20.5
