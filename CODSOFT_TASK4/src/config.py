"""Central configuration: file paths and reproducibility settings."""

from pathlib import Path

ROOT_DIR = Path(__file__).resolve().parent.parent

DATA_DIR = ROOT_DIR / "data"
RAW_DATA_PATH = DATA_DIR / "spam.csv"
PROCESSED_DATA_PATH = DATA_DIR / "processed" / "spam_clean.csv"

MODELS_DIR = ROOT_DIR / "models"
MODEL_PATH = MODELS_DIR / "spam_classifier.joblib"

REPORTS_DIR = ROOT_DIR / "reports"
FIGURES_DIR = REPORTS_DIR / "figures"
METRICS_PATH = REPORTS_DIR / "metrics.json"
COMPARISON_PATH = REPORTS_DIR / "model_comparison.csv"
REPORT_MD_PATH = REPORTS_DIR / "evaluation_report.md"

RANDOM_STATE = 42
TEST_SIZE = 0.20
CV_FOLDS = 5

# Label encoding: the positive class (1) is SPAM.
LABEL_MAP = {"ham": 0, "spam": 1}
LABEL_NAMES = {0: "LEGITIMATE", 1: "SPAM"}

MAX_MESSAGE_LENGTH = 5000
