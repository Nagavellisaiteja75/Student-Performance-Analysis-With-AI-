# Git and GitHub Publishing Guide

Use this guide after reviewing the project locally and running the pipeline successfully.

## 1. Initialize the repository

From the project folder:

```bash
git init
git add .
git status
git commit -m "Initial commit: student performance analytics portfolio project"
```

Before committing, confirm that `.gitignore` prevents `data/raw/`, `data/processed/`, `artifacts/`, and `models/` generated content from appearing in `git status`.

## 2. Create a GitHub repository

On GitHub, create a new empty repository named:

```text
ai-powered-student-performance-analytics
```

Do not add a README, license, or `.gitignore` through GitHub because this project already includes them.

## 3. Connect and push

Replace `YOUR-USERNAME` with your GitHub username:

```bash
git branch -M main
git remote add origin https://github.com/YOUR-USERNAME/ai-powered-student-performance-analytics.git
git push -u origin main
```

If you use GitHub CLI and are already authenticated, an alternative is:

```bash
gh repo create ai-powered-student-performance-analytics --public --source=. --remote=origin --push
```

## 4. Strong repository description

Use this GitHub description:

> A reproducible Python and Streamlit portfolio project for UCI student-performance analytics, regression model comparison, explainability, and human-reviewed early academic support indicators.

Suggested repository topics:

```text
python pandas scikit-learn streamlit data-analytics machine-learning education analytics portfolio
```

## 5. Suggested screenshots for your README or LinkedIn post

After running the dashboard, capture:

1. The Overview page showing the project workflow.
2. The EDA page showing a chart grid.
3. The Model Performance page showing your actual latest-run metrics and feature importance.
4. The Support Prediction page showing a clearly labeled non-diagnostic indicator.

Never publish raw student-level data, local identifiers, tokens, secrets, or screenshots of any real students.

## 6. Optional improvement commits

Make small, clear commits as you iterate:

```bash
git add README.md docs/ reports/
git commit -m "docs: add portfolio workflow and report template"

git add src/ app.py tests/
git commit -m "feat: add leakage-aware model dashboard"
```

## 7. Portfolio-ready README sentence

> Built a reproducible student-performance analytics pipeline with Python, Pandas, scikit-learn, Matplotlib, and Streamlit; compared regression models, added permutation-based explainability, and designed human-reviewed early academic support indicators with leakage and responsible-AI safeguards.
