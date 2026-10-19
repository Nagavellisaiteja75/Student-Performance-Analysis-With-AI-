"""Central project configuration and filesystem locations."""

from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[1]
DATA_DIR = PROJECT_ROOT / "data"
RAW_DATA_DIR = DATA_DIR / "raw"
PROCESSED_DATA_DIR = DATA_DIR / "processed"
ARTIFACTS_DIR = PROJECT_ROOT / "artifacts"
FIGURES_DIR = ARTIFACTS_DIR / "figures"
MODELS_DIR = PROJECT_ROOT / "models"

UCI_DATASET_URL = "https://archive.ics.uci.edu/static/public/320/student%2Bperformance.zip"
UCI_DATASET_PAGE = "https://archive.ics.uci.edu/dataset/320/student%2Bperformance"
DATASET_FILES = {
    "student-mat.csv": "math",
    "student-por.csv": "portuguese",
}

TARGET_COLUMN = "G3"
SUBJECT_COLUMN = "subject"
PRIOR_GRADE_COLUMNS = ("G1", "G2")
MODEL_EXCLUDED_COLUMNS = (TARGET_COLUMN, *PRIOR_GRADE_COLUMNS)

NUMERIC_COLUMNS = (
    "age",
    "Medu",
    "Fedu",
    "traveltime",
    "studytime",
    "failures",
    "famrel",
    "freetime",
    "goout",
    "Dalc",
    "Walc",
    "health",
    "absences",
    "G1",
    "G2",
    "G3",
)

RANDOM_STATE = 42
TEST_SIZE = 0.20
CV_FOLDS = 5
PERMUTATION_REPEATS = 20
GRADE_MIN = 0
GRADE_MAX = 20

CLEAN_DATA_FILE = PROCESSED_DATA_DIR / "student_performance_clean.csv"
DATA_PROFILE_FILE = ARTIFACTS_DIR / "data_profile.json"
CLEANING_REPORT_FILE = ARTIFACTS_DIR / "cleaning_report.json"
EDA_SUMMARY_FILE = ARTIFACTS_DIR / "eda_summary.json"
MODEL_COMPARISON_FILE = ARTIFACTS_DIR / "model_comparison.csv"
TEST_PREDICTIONS_FILE = ARTIFACTS_DIR / "test_predictions.csv"
FEATURE_IMPORTANCE_FILE = ARTIFACTS_DIR / "feature_importance.csv"
MODEL_METADATA_FILE = ARTIFACTS_DIR / "model_metadata.json"
MODEL_FILE = MODELS_DIR / "best_student_grade_model.joblib"

FIGURE_FILES = {
    "target_distribution": FIGURES_DIR / "target_distribution.png",
    "subject_grade_distribution": FIGURES_DIR / "subject_grade_distribution.png",
    "numeric_correlation": FIGURES_DIR / "numeric_correlation.png",
    "absences_vs_grade": FIGURES_DIR / "absences_vs_final_grade.png",
    "model_comparison": FIGURES_DIR / "model_comparison.png",
    "prediction_diagnostics": FIGURES_DIR / "prediction_diagnostics.png",
    "feature_importance": FIGURES_DIR / "feature_importance.png",
}
