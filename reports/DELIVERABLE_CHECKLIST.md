# Internship / Portfolio Deliverable Checklist

Complete this checklist after running the project locally.

## Core project

- [ ] `README.md` explains the problem, setup, approach, limitations, and run commands.
- [ ] `requirements.txt` installs successfully in a clean virtual environment.
- [ ] `python scripts/run_pipeline.py` completes successfully.
- [ ] `streamlit run app.py` opens the dashboard.
- [ ] `pytest` passes.

## Data analytics evidence

- [ ] Include the actual generated data profile and cleaning report in your presentation notes.
- [ ] Include all four EDA visuals and describe only observations shown by your run.
- [ ] Include the actual model comparison table.
- [ ] Include generated held-out MAE, RMSE, and R²—never invented values.
- [ ] Include the permutation-importance chart and explain its limitations.

## Dashboard evidence

- [ ] Capture an Overview screenshot.
- [ ] Capture an EDA screenshot.
- [ ] Capture a Model Performance screenshot with your actual results.
- [ ] Capture a Support Prediction screenshot showing the non-diagnostic disclaimer.
- [ ] Mention the responsible-use / limitations page in your demo.

## GitHub evidence

- [ ] Initialize Git and make meaningful commits.
- [ ] Push the source code and documentation to a public or permitted private repository.
- [ ] Verify that raw data, generated artifacts, models, virtual environments, and secrets are not committed.
- [ ] Add a clear repository description and relevant topics.
- [ ] Add your final GitHub URL to the project report.

## Presentation summary

Prepare a 60–90 second introduction:

> I built an AI-powered student performance analytics project that downloads and cleans UCI education data, performs exploratory analysis, compares regression models with cross-validation, explains the selected model with permutation importance, and presents human-reviewed academic support indicators in a Streamlit dashboard. I excluded prior-grade leakage and documented ethical limitations so the project demonstrates responsible data analytics rather than automated student decision-making.
