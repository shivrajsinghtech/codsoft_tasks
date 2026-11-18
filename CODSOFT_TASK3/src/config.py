"""Central configuration: paths, column groups and constants."""
from pathlib import Path

ROOT_DIR = Path(__file__).resolve().parent.parent
DATA_PATH = ROOT_DIR / "data" / "Churn_Modelling.csv"
MODELS_DIR = ROOT_DIR / "models"
MODEL_PATH = MODELS_DIR / "churn_pipeline.joblib"
REPORTS_DIR = ROOT_DIR / "reports"
FIGURES_DIR = REPORTS_DIR / "figures"
METRICS_PATH = REPORTS_DIR / "metrics.json"

RANDOM_STATE = 42
TEST_SIZE = 0.20
CV_FOLDS = 5

TARGET = "Exited"
ID_COLUMNS = ["RowNumber", "CustomerId", "Surname"]
NUMERIC_FEATURES = [
    "CreditScore",
    "Age",
    "Tenure",
    "Balance",
    "NumOfProducts",
    "EstimatedSalary",
]
BINARY_FEATURES = ["HasCrCard", "IsActiveMember"]
CATEGORICAL_FEATURES = ["Geography", "Gender"]
FEATURES = NUMERIC_FEATURES + BINARY_FEATURES + CATEGORICAL_FEATURES

EXPECTED_COLUMNS = ID_COLUMNS + FEATURES + [TARGET]
