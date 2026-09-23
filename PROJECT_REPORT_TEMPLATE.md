# AI-Powered Student Performance Analytics & Early Intervention System

## Portfolio Project Report Template

**Author:** [Your name]  
**Program / internship:** IBM Data Analytics with AI — Foundation to Implementation  
**Date:** [Submission date]  
**Repository:** [GitHub repository URL]  

---

## Executive summary

This project develops a reproducible analytics workflow and Streamlit dashboard for exploring student-performance patterns and estimating final course grades using the UCI Student Performance dataset. The system combines data collection, cleaning, exploratory analysis, regression model comparison, evaluation, and permutation-based explainability. Its output is a human-reviewed academic support indicator, not an automated student decision.

After running the pipeline, add your generated selected model and metrics here:

- **Selected model:** [Insert model name from `artifacts/model_metadata.json`]
- **Held-out MAE:** [Insert generated value]
- **Held-out RMSE:** [Insert generated value]
- **Held-out R²:** [Insert generated value]

Do not insert estimates or sample metrics that were not produced by your own run.

## 1. Problem statement

Educational teams often need an early, structured way to identify students who may benefit from a supportive academic conversation. This project asks whether available student-context features can estimate final course grade and help frame a constructive support discussion.

The objective is decision support. The model does not diagnose students, determine eligibility, or replace educator judgement.

## 2. Data collection and understanding

### Dataset

- **Source:** UCI Machine Learning Repository, Student Performance dataset
- **Files:** `student-mat.csv` and `student-por.csv`
- **Target:** `G3`, final grade on a 0–20 scale
- **Unit of analysis:** Course record

The project downloads the source archive programmatically, validates the expected files, and appends a `subject` column before analysis. The generated data profile reports row counts, data types, missingness, target summary, and subject distribution.

### Data understanding findings

Fill this section using `artifacts/data_profile.json` and the dashboard:

- **Records after cleaning:** [Insert generated count]
- **Columns after combining course files:** [Insert generated count]
- **Subject distribution:** [Insert generated values]
- **Target distribution summary:** [Insert generated summary]
- **Missingness observation:** [Insert generated observation]

## 3. Data cleaning and preparation

The cleaning workflow:

1. Normalized text whitespace.
2. Converted documented numeric columns safely.
3. Removed exact duplicate course records only.
4. Removed records with missing or invalid final-grade targets.
5. Kept remaining feature missingness for imputation within model pipelines.

This approach prevents test-set information from influencing preprocessing during training. The detailed generated record is in `artifacts/cleaning_report.json`.

## 4. Exploratory data analysis

Include your actual generated charts here:

1. Final-grade distribution
2. Final grades by course subject
3. Numeric-feature correlation heatmap
4. Absences versus final grade

### Key descriptive insights

- [Insight supported by the final-grade distribution]
- [Insight supported by subject comparison]
- [Insight supported by correlation chart]
- [Insight supported by absences chart]

Use association-based language. The source is observational, so the charts do not establish causal effects.

## 5. Machine learning methodology

### Leakage-aware feature design

`G1` and `G2` are excluded because they are first- and second-period grades that are strongly related to the final target `G3`. Excluding them creates a more difficult but more honest early-intervention task.

### Candidate models

| Model | Role |
| --- | --- |
| Ridge Regression | Regularized linear baseline |
| Random Forest Regressor | Nonlinear ensemble baseline |
| Gradient Boosting Regressor | Nonlinear boosting candidate |

### Validation protocol

The project creates a reproducible 80/20 train/test split with `random_state=42`. Candidate models are compared with five-fold cross-validation on training data. The model with the lowest mean cross-validation MAE is selected and evaluated once on the held-out test data.

## 6. Results

Paste the generated model comparison table here:

| Model | CV MAE | CV RMSE | CV R² |
| --- | --- | --- | --- |
| [Copy from generated file] | [ ] | [ ] | [ ] |

### Selected model and held-out evaluation

- **Selected model:** [Insert generated model]
- **MAE:** [Insert generated value and plain-language interpretation]
- **RMSE:** [Insert generated value and plain-language interpretation]
- **R²:** [Insert generated value and cautious interpretation]

### Explainability

Use the permutation-importance artifact to list the most influential features:

1. [Feature and generated importance]
2. [Feature and generated importance]
3. [Feature and generated importance]

Permutation importance measures how model error changes when a feature is shuffled. It does not prove that modifying a feature would cause grades to change.

## 7. Streamlit dashboard

The dashboard contains:

- Overview of the workflow and generated data status
- EDA charts and cleaned-data preview
- Model-comparison table, held-out metrics, and explainability visual
- Individual grade estimate with transparent support indicators
- Responsible-use and limitations page

Include 2–4 screenshots from your local run and describe how an educator would use each page responsibly.

## 8. Ethical considerations and limitations

- The dataset is a historical sample from two Portuguese schools and may not generalize.
- Variables must be available at a defined decision point; `absences`, for example, changes over time.
- The project is observational and does not make causal claims.
- Predictions should support a conversation, never automatically decide placement, discipline, or resources.
- Real deployment requires privacy safeguards, consent, fairness review, local validation, and monitoring.

## 9. Conclusion and next steps

This project demonstrates an end-to-end data analytics workflow using Python, Pandas, scikit-learn, Matplotlib, Streamlit, Git, and GitHub. The most valuable next step is to validate the approach with local, representative, time-stamped data under appropriate educational governance.

Potential enhancements:

1. Add time-aware features collected at explicit checkpoints.
2. Compare prediction intervals and calibration, not only point estimates.
3. Evaluate performance across relevant subgroups with privacy-preserving governance.
4. Add model monitoring and data-drift checks for any approved operational setting.

## References

Cortez, P. (2008). *Student Performance* [Dataset]. UCI Machine Learning Repository. https://doi.org/10.24432/C5TG7T
