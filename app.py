"""Interactive Streamlit dashboard for Student Performance Analytics."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import joblib
import pandas as pd
import streamlit as st

from src.config import (
    CLEAN_DATA_FILE,
    DATA_PROFILE_FILE,
    FEATURE_IMPORTANCE_FILE,
    FIGURE_FILES,
    MODEL_COMPARISON_FILE,
    MODEL_FILE,
    MODEL_METADATA_FILE,
)
from src.modeling.evaluation import metric_summary_text, support_indicator


st.set_page_config(
    page_title="Student Performance Analytics",
    page_icon="🎓",
    layout="wide",
    initial_sidebar_state="expanded",
)

st.markdown(
    """
    <style>
    .block-container {max-width: 1220px; padding-top: 2.2rem;}
    [data-testid="stMetric"] {background: #f6f9fc; border-radius: 12px; padding: 0.8rem;}
    .small-note {color: #506176; font-size: 0.92rem;}
    </style>
    """,
    unsafe_allow_html=True,
)


FEATURE_LABELS = {
    "school": "School",
    "sex": "Sex",
    "age": "Age",
    "address": "Home address type",
    "famsize": "Family size",
    "Pstatus": "Parent cohabitation status",
    "Medu": "Mother's education level",
    "Fedu": "Father's education level",
    "Mjob": "Mother's job",
    "Fjob": "Father's job",
    "reason": "School-choice reason",
    "guardian": "Guardian",
    "traveltime": "Travel-time category",
    "studytime": "Weekly study-time category",
    "failures": "Previous class failures",
    "schoolsup": "School educational support",
    "famsup": "Family educational support",
    "paid": "Paid classes for the subject",
    "activities": "Extracurricular activities",
    "nursery": "Attended nursery school",
    "higher": "Wants higher education",
    "internet": "Internet access at home",
    "romantic": "Romantic relationship",
    "famrel": "Family-relationship rating",
    "freetime": "Free-time rating",
    "goout": "Going-out rating",
    "Dalc": "Workday alcohol-consumption rating",
    "Walc": "Weekend alcohol-consumption rating",
    "health": "Current health rating",
    "absences": "Absences at the selected snapshot",
    "subject": "Course subject",
}


def _modified_time(path: Path) -> float:
    return path.stat().st_mtime if path.exists() else 0.0


@st.cache_data(show_spinner=False)
def load_csv(path_as_string: str, modified_time: float) -> pd.DataFrame:
    """Load a generated CSV and invalidate cache when the file changes."""

    del modified_time
    return pd.read_csv(path_as_string)


@st.cache_data(show_spinner=False)
def load_json(path_as_string: str, modified_time: float) -> dict[str, Any]:
    """Load generated JSON and invalidate cache when the file changes."""

    del modified_time
    with Path(path_as_string).open(encoding="utf-8") as input_file:
        return json.load(input_file)


@st.cache_resource(show_spinner=False)
def load_model(path_as_string: str, modified_time: float) -> Any:
    """Load the persisted sklearn pipeline only when a prediction is requested."""

    del modified_time
    return joblib.load(path_as_string)


def format_feature_label(feature: str) -> str:
    """Return a reader-friendly input label without changing the model schema."""

    return FEATURE_LABELS.get(feature, feature.replace("_", " ").title())


def show_setup_guidance() -> None:
    """Show actionable setup help wherever generated artifacts are unavailable."""

    st.warning("Generated data or model artifacts are not available yet.")
    st.code("python -m pip install -r requirements.txt\npython scripts/run_pipeline.py\nstreamlit run app.py")
    st.caption(
        "The pipeline downloads the UCI source data, produces analysis artifacts, "
        "trains the models, and then enables all dashboard pages."
    )


def show_home() -> None:
    st.title("AI-Powered Student Performance Analytics")
    st.subheader("Early intervention support, grounded in transparent data analysis")
    st.write(
        "Explore the UCI Student Performance data, compare reproducible regression "
        "models, and use predicted grade ranges as prompts for supportive human review."
    )

    workflow_columns = st.columns(4)
    for column, title, text in zip(
        workflow_columns,
        ["1. Collect", "2. Understand", "3. Compare", "4. Support"],
        [
            "Download and validate two UCI course files.",
            "Clean data and inspect patterns with EDA.",
            "Select the lowest-CV-error regression model.",
            "Use estimates to start a contextual conversation.",
        ],
    ):
        with column:
            st.markdown(f"**{title}**")
            st.caption(text)

    if not CLEAN_DATA_FILE.exists():
        show_setup_guidance()
        return

    data = load_csv(str(CLEAN_DATA_FILE), _modified_time(CLEAN_DATA_FILE))
    profile = (
        load_json(str(DATA_PROFILE_FILE), _modified_time(DATA_PROFILE_FILE))
        if DATA_PROFILE_FILE.exists()
        else None
    )
    data_columns = st.columns(4)
    data_columns[0].metric("Course records", f"{len(data):,}")
    data_columns[1].metric("Columns", data.shape[1])
    data_columns[2].metric("Subjects", data["subject"].nunique())
    data_columns[3].metric("Target scale", "0–20")

    if profile:
        st.caption(
            "Data profile loaded from the latest pipeline run. "
            f"Final-grade mean: {profile['target']['mean']:.2f}; "
            f"median: {profile['target']['median']:.2f}."
        )
    st.info(
        "This dashboard is an educational portfolio project. It is not a student "
        "ranking, diagnosis, or automated decision system."
    )


def show_explore_data() -> None:
    st.title("Data Understanding & Exploratory Analysis")
    if not CLEAN_DATA_FILE.exists():
        show_setup_guidance()
        return

    data = load_csv(str(CLEAN_DATA_FILE), _modified_time(CLEAN_DATA_FILE))
    selected_subjects = st.multiselect(
        "Filter the preview by course subject",
        options=sorted(data["subject"].unique()),
        default=sorted(data["subject"].unique()),
    )
    filtered_data = data[data["subject"].isin(selected_subjects)]
    st.dataframe(filtered_data.head(25), use_container_width=True, hide_index=True)
    st.caption(
        "The cleaned data keeps the two course files as distinct records and appends "
        "a `subject` field before analysis."
    )

    chart_layout = [
        ("target_distribution", "Final-grade distribution"),
        ("subject_grade_distribution", "Final grades by course"),
        ("numeric_correlation", "Numeric-feature correlation"),
        ("absences_vs_grade", "Absences and final grade"),
    ]
    for left_chart, right_chart in zip(chart_layout[::2], chart_layout[1::2]):
        left_column, right_column = st.columns(2)
        for column, (chart_name, caption) in ((left_column, left_chart), (right_column, right_chart)):
            with column:
                chart_path = FIGURE_FILES[chart_name]
                if chart_path.exists():
                    st.image(str(chart_path), caption=caption, use_column_width=True)
                else:
                    st.info(f"Run the pipeline to create the {caption.lower()} chart.")

    st.warning(
        "EDA reveals associations, not causes. The dataset is a historical sample "
        "from two Portuguese schools and should not be generalized without validation."
    )


def show_model_performance() -> None:
    st.title("Regression Model Comparison & Explainability")
    if not MODEL_COMPARISON_FILE.exists() or not MODEL_METADATA_FILE.exists():
        show_setup_guidance()
        return

    comparison = load_csv(
        str(MODEL_COMPARISON_FILE), _modified_time(MODEL_COMPARISON_FILE)
    )
    metadata = load_json(str(MODEL_METADATA_FILE), _modified_time(MODEL_METADATA_FILE))
    st.write(
        "Candidate models are selected by the lowest mean cross-validation MAE on "
        "training data. The held-out test split is evaluated after selection."
    )
    st.dataframe(comparison.round(3), use_container_width=True, hide_index=True)
    model_chart = FIGURE_FILES["model_comparison"]
    if model_chart.exists():
        st.image(str(model_chart), caption="Cross-validation comparison", use_column_width=True)

    st.subheader(f"Selected model: {metadata['selected_model']}")
    st.caption("The values below are generated by the latest local training run; none are prefilled.")
    metric_values = metric_summary_text(metadata["holdout_metrics"])
    metric_columns = st.columns(3)
    for column, (metric_name, metric_value) in zip(metric_columns, metric_values.items()):
        column.metric(metric_name, metric_value)

    diagnostic_chart = FIGURE_FILES["prediction_diagnostics"]
    if diagnostic_chart.exists():
        st.image(
            str(diagnostic_chart),
            caption="Held-out predictions compared with actual final grades",
            use_column_width=True,
        )

    st.subheader("What influenced the model globally?")
    st.caption(metadata["explainability_method"])
    importance_chart = FIGURE_FILES["feature_importance"]
    if importance_chart.exists():
        st.image(
            str(importance_chart),
            caption="Higher values indicate a larger deterioration in MAE after shuffling a feature.",
            use_column_width=True,
        )
    elif FEATURE_IMPORTANCE_FILE.exists():
        importance = load_csv(
            str(FEATURE_IMPORTANCE_FILE), _modified_time(FEATURE_IMPORTANCE_FILE)
        )
        st.dataframe(importance.head(15).round(4), use_container_width=True, hide_index=True)

    st.info(
        "Feature importance describes model reliance in this dataset; it does not "
        "prove that changing a feature would cause a grade to change."
    )


def _render_number_input(feature: str, settings: dict[str, Any]) -> int | float:
    label = format_feature_label(feature)
    if settings["integer"]:
        return st.number_input(
            label,
            min_value=int(settings["minimum"]),
            max_value=int(settings["maximum"]),
            value=int(settings["default"]),
            step=1,
            key=f"input_{feature}",
        )
    range_width = float(settings["maximum"]) - float(settings["minimum"])
    step = max(round(range_width / 100, 2), 0.01)
    return st.number_input(
        label,
        min_value=float(settings["minimum"]),
        max_value=float(settings["maximum"]),
        value=float(settings["default"]),
        step=step,
        key=f"input_{feature}",
    )


def _render_support_message(support: dict[str, str]) -> None:
    message = f"**{support['priority']}** — {support['recommended_action']}"
    if support["color"] == "red":
        st.error(message)
    elif support["color"] == "orange":
        st.warning(message)
    elif support["color"] == "blue":
        st.info(message)
    else:
        st.success(message)


def show_prediction() -> None:
    st.title("Prediction-Based Academic Support Indicator")
    if not MODEL_FILE.exists() or not MODEL_METADATA_FILE.exists():
        show_setup_guidance()
        return

    metadata = load_json(str(MODEL_METADATA_FILE), _modified_time(MODEL_METADATA_FILE))
    model = load_model(str(MODEL_FILE), _modified_time(MODEL_FILE))
    input_metadata = metadata["model_input"]
    st.write(
        "Enter values that were genuinely known at the intended support point. "
        "The form intentionally excludes G1 and G2 prior grades."
    )
    st.caption(input_metadata["availability_note"])

    input_values: dict[str, Any] = {}
    with st.form("student_prediction_form"):
        st.subheader("Student context")
        categorical_columns = st.columns(2)
        for index, (feature, settings) in enumerate(
            input_metadata["categorical_features"].items()
        ):
            options = settings["options"]
            default_index = options.index(settings["default"])
            with categorical_columns[index % 2]:
                input_values[feature] = st.selectbox(
                    format_feature_label(feature),
                    options=options,
                    index=default_index,
                    key=f"input_{feature}",
                )

        st.subheader("Study, wellbeing, and attendance context")
        numeric_columns = st.columns(2)
        for index, (feature, settings) in enumerate(input_metadata["numeric_features"].items()):
            with numeric_columns[index % 2]:
                input_values[feature] = _render_number_input(feature, settings)

        submitted = st.form_submit_button("Generate support indicator", type="primary")

    if not submitted:
        return

    ordered_input = {
        feature: input_values[feature] for feature in input_metadata["feature_columns"]
    }
    predicted_grade = float(model.predict(pd.DataFrame([ordered_input]))[0])
    predicted_grade = max(0.0, min(20.0, predicted_grade))
    support = support_indicator(predicted_grade)

    prediction_columns = st.columns([1, 2])
    prediction_columns[0].metric("Estimated final grade", f"{predicted_grade:.1f} / 20")
    prediction_columns[1].markdown(f"**Support band:** {support['band']}")
    _render_support_message(support)
    st.caption(
        "Review this estimate with the student and relevant staff. Do not use it as "
        "the sole basis for intervention, placement, discipline, or resource allocation."
    )

    with st.expander("Review entered values"):
        review_data = pd.DataFrame(
            {"Feature": list(ordered_input), "Value": list(ordered_input.values())}
        )
        st.dataframe(review_data, use_container_width=True, hide_index=True)


def show_limitations() -> None:
    st.title("Responsible Use, Assumptions & Limitations")
    st.markdown(
        """
        - **Dataset scope:** This project uses UCI's historical Student Performance dataset from two Portuguese schools. It is a learning project, not a deployment-ready institutional model.
        - **Timing:** The default model excludes G1 and G2 to avoid prior-grade leakage. Other fields, especially absences, must be captured at a clearly defined decision point before they are used.
        - **Fairness and privacy:** Do not use sensitive student attributes to make high-impact decisions. A real deployment needs consent, data minimization, access controls, subgroup evaluation, and ongoing monitoring.
        - **Prediction uncertainty:** A grade estimate is not a causal explanation or a diagnosis. Support decisions need educator judgement and a student-centered conversation.
        - **Validation:** Re-train and externally validate on local, representative data before any operational use. Performance can change across schools, cohorts, policies, and time.
        """
    )
    st.markdown(
        "Source: [UCI Student Performance dataset](https://archive.ics.uci.edu/dataset/320/student%2Bperformance)."
    )


PAGES = {
    "Overview": show_home,
    "Explore data": show_explore_data,
    "Model performance": show_model_performance,
    "Support prediction": show_prediction,
    "Limitations": show_limitations,
}

with st.sidebar:
    st.title("Student Analytics")
    selected_page = st.radio("Navigate", list(PAGES))
    st.divider()
    st.caption("Portfolio project • Python • scikit-learn • Streamlit")

PAGES[selected_page]()
