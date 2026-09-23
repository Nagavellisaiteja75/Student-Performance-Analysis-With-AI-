# AI-Powered Student Performance Analytics & Early Intervention System

A complete, portfolio-ready data analytics project that turns the UCI Student Performance dataset into a reproducible analysis workflow and a Streamlit decision-support dashboard.

> **Important:** This project deliberately contains no claimed model metrics before execution. Run the pipeline locally to produce your own reproducible metrics, charts, and model artifacts.

## Project goal

Estimate a student's final course grade (`G3`, 0–20 scale) using demographic, school, study, wellbeing, and attendance context, then translate the estimate into a **human-reviewed academic support indicator**. The dashboard is designed to start supportive conversations—not to rank students or make automated decisions.

## What this project demonstrates

- Data collection from the official [UCI Student Performance dataset](https://archive.ics.uci.edu/dataset/320/student%2Bperformance)
- Data understanding, schema checks, transparent cleaning, and reproducible data profiles
- Exploratory data analysis with Matplotlib
- Regression model comparison using Ridge Regression, Random Forest, and Gradient Boosting
- Cross-validation model selection, final held-out evaluation, and permutation-feature importance
- A clean Streamlit dashboard for EDA, model performance, and individual support indicators
- Modular Python code, tests, Git/GitHub guidance, and report-ready documentation

## Technology stack

Python, Pandas, NumPy, scikit-learn, Matplotlib, Streamlit, Joblib, Git, and GitHub.

## Project structure

```text
ai-powered-student-performance-analytics/
├── app.py                         # Streamlit dashboard
├── requirements.txt               # Reproducible dependencies
├── scripts/
│   ├── download_data.py           # Fetch and validate UCI raw files
│   ├── prepare_data.py            # Clean data and build profile artifacts
│   ├── generate_eda.py            # Create EDA charts
│   ├── train_models.py            # Train, compare, evaluate, explain
│   └── run_pipeline.py            # Run all project stages in one command
├── src/
│   ├── data/                      # Downloading, cleaning, profiling
│   ├── analysis/                  # EDA charts
│   └── modeling/                  # Features, models, evaluation, indicators
├── data/                          # Local raw and processed data (gitignored)
├── artifacts/                     # Generated profiles, charts, CSVs (gitignored)
├── models/                        # Trained model file (gitignored)
├── docs/                          # Workflow, GitHub setup, data dictionary
├── reports/                       # Portfolio report template and checklist
├── notebooks/                     # Notebook-ready documented workflow
└── tests/                         # Focused quality checks
```

## Quick start

### 1. Create and activate a virtual environment

**Windows PowerShell**

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
```

**macOS / Linux**

```bash
python3 -m venv .venv
source .venv/bin/activate
```

### 2. Install dependencies

```bash
python -m pip install --upgrade pip
python -m pip install -r requirements.txt
```

> **Windows note:** If a deeply nested folder causes a path-length error while creating `.venv`, enable Windows long paths or create the virtual environment in a shorter location and use that environment to install this project's `requirements.txt`.

### 3. Run the complete pipeline

```bash
python scripts/run_pipeline.py
```

This command downloads the UCI archive if needed, prepares a cleaned dataset, creates EDA charts, compares regression models, evaluates the selected model on a held-out test set, calculates permutation importance, and saves the dashboard artifacts.

### 4. Start the dashboard

```bash
streamlit run app.py
```

Open the local URL shown by Streamlit in a browser.

## Optional stage-by-stage workflow

Run these commands if you want to demonstrate every internship stage separately:

```bash
python scripts/download_data.py
python scripts/prepare_data.py
python scripts/generate_eda.py
python scripts/train_models.py
streamlit run app.py
```

Use `python scripts/download_data.py --force` to fetch a fresh copy of the source files.

## Data collection and understanding

The project downloads the official UCI archive containing:

- `student-mat.csv` — mathematics course records
- `student-por.csv` — Portuguese language course records

Both datasets are appended with a `subject` column before analysis. A row represents a **course record**, so the project does not assume that all rows across subjects are distinct people. The pipeline creates:

- `data/processed/student_performance_clean.csv`
- `artifacts/cleaning_report.json`
- `artifacts/data_profile.json`

The UCI dataset contains academic, demographic, family, social, and school-related features. `G3` is the final grade on a 0–20 scale. See [docs/data_dictionary.md](docs/data_dictionary.md) for an analysis-focused guide.

## Cleaning and preparation choices

- Normalizes whitespace in text columns.
- Coerces documented numeric columns to numeric values.
- Removes only exact duplicate course records.
- Removes rows with missing or out-of-range final grades.
- Preserves missing feature values for model-pipeline imputation, avoiding whole-dataset imputation leakage.
- Stores cleaning counts and assumptions in `artifacts/cleaning_report.json`.

## Modeling approach

### Leakage-aware early intervention design

The default model excludes `G1`, `G2`, and `G3` from its features:

- `G3` is the final-grade target.
- `G1` and `G2` are first- and second-period grades, which are highly related to `G3`.

Excluding early assessment grades makes prediction harder, but it makes the project more honest and more useful for earlier support conversations. The UCI dataset documentation specifically notes this relationship.

### Candidate models

| Model | Why it is included |
| --- | --- |
| Ridge Regression | Interpretable regularized linear baseline |
| Random Forest Regressor | Nonlinear ensemble baseline |
| Gradient Boosting Regressor | Strong nonlinear tabular-data candidate |

Each candidate uses the same leakage-safe preprocessing pipeline:

- Median imputation and scaling for numeric fields
- Most-frequent imputation and one-hot encoding for categorical fields
- Cross-validation on the training partition

The selected model is the one with the lowest mean cross-validation MAE. The chosen model is then evaluated once on a fixed held-out test split (`random_state=42`). Generated outputs include MAE, RMSE, R², a model comparison table, prediction diagnostics, and permutation importance. No values are hard-coded in this repository.

## Dashboard pages

| Page | Purpose |
| --- | --- |
| Overview | Project workflow and generated data status |
| Explore data | Cleaned-data preview and EDA visuals |
| Model performance | Cross-validation comparison, held-out metrics, and global feature importance |
| Support prediction | Individual grade estimate and contextual support indicator |
| Limitations | Responsible-use guidance, timing caveats, privacy, and validation needs |

Support bands are intentionally simple and transparent:

| Estimated grade | Indicator | Suggested human action |
| --- | --- | --- |
| Below 10 | High-priority outreach | Timely supportive check-in and barrier review |
| 10 to below 12 | Targeted support | Study planning, tutoring, and a follow-up |
| 12 to below 14 | Monitor and encourage | Progress feedback and optional support |
| 14 or above | Continue strengths-based support | Recognize strengths and maintain check-ins |

These bands are communication aids only. They are not diagnoses, student labels, or automated decisions.

## Reproducibility and quality checks

The project fixes the train/test random state, saves the selected model and metadata, and records all generated outputs locally. Run the focused checks with:

```bash
pytest
```

## Portfolio workflow and report deliverables

- [Documented analysis workflow](docs/workflow.md)
- [Notebook-ready walkthrough](notebooks/01_documented_analysis_workflow.md)
- [Git and GitHub publishing guide](docs/github_setup.md)
- [Portfolio project report template](reports/PROJECT_REPORT_TEMPLATE.md)
- [Submission deliverable checklist](reports/DELIVERABLE_CHECKLIST.md)

## Responsible-use limitations

- The source is a historical, limited sample from two Portuguese schools; it is not representative of every institution.
- The dataset may contain variables observed at different points in time. In a real system, define an explicit data snapshot before prediction.
- Model patterns are not causal explanations. Use educator judgment and invite student perspective before offering support.
- Do not use this system for high-impact decisions such as discipline, placement, eligibility, or withholding resources.
- A real deployment requires privacy controls, consent, fairness review, local validation, monitoring, and institutional governance.

## GitHub publication

Do not commit downloaded data, trained models, or generated artifacts; the included `.gitignore` already protects them. Follow [docs/github_setup.md](docs/github_setup.md) to create a repository, make an initial commit, publish to GitHub, and write a strong portfolio README summary.

## Dataset attribution

Cortez, P. (2008). *Student Performance* [Dataset]. UCI Machine Learning Repository. https://doi.org/10.24432/C5TG7T
