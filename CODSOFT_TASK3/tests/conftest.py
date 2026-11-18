import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from src.data import clean_data, load_raw_data, split_features_target  # noqa: E402


@pytest.fixture(scope="session")
def raw_df():
    return load_raw_data()


@pytest.fixture(scope="session")
def clean_df(raw_df):
    return clean_data(raw_df)


@pytest.fixture(scope="session")
def xy(clean_df):
    return split_features_target(clean_df)


@pytest.fixture
def valid_customer():
    return {
        "CreditScore": 619, "Geography": "France", "Gender": "Female", "Age": 42,
        "Tenure": 2, "Balance": 0.0, "NumOfProducts": 1, "HasCrCard": 1,
        "IsActiveMember": 1, "EstimatedSalary": 101348.88,
    }


@pytest.fixture(scope="session")
def small_model(xy):
    """A quickly trained pipeline on a stratified subsample (for fast tests)."""
    from sklearn.ensemble import RandomForestClassifier
    from sklearn.model_selection import train_test_split

    from src.models import build_pipeline

    X, y = xy
    X_s, _, y_s, _ = train_test_split(X, y, train_size=2000, stratify=y, random_state=0)
    model = build_pipeline(RandomForestClassifier(n_estimators=30, random_state=0,
                                                  class_weight="balanced"))
    return model.fit(X_s, y_s)
