import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.config import MODEL_PATH, RAW_DATA_PATH  # noqa: E402


@pytest.fixture(scope="session")
def model():
    if not MODEL_PATH.exists():
        pytest.skip("Trained model missing - run `python train.py` first.")
    from src.predict import load_model
    return load_model(MODEL_PATH)


@pytest.fixture(scope="session")
def clean_df():
    if not RAW_DATA_PATH.exists():
        pytest.skip("data/spam.csv missing.")
    from src.data_preprocessing import load_clean_data
    return load_clean_data()
