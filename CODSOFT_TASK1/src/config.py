"""Central configuration: paths and constants shared by every module."""

from __future__ import annotations

import os
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent

# The raw dataset is not committed (≈105 MB). Place the extracted
# "Genre Classification Dataset" folder under data/raw/ (see README).
DATA_DIR = Path(
    os.environ.get(
        "MOVIE_GENRE_DATA_DIR",
        PROJECT_ROOT / "data" / "raw" / "Genre Classification Dataset",
    )
)
TRAIN_FILE = "train_data.txt"
TEST_FILE = "test_data.txt"
TEST_SOLUTION_FILE = "test_data_solution.txt"
DESCRIPTION_FILE = "description.txt"

MODELS_DIR = PROJECT_ROOT / "models"
MODEL_PATH = Path(os.environ.get("MOVIE_GENRE_MODEL_PATH", MODELS_DIR / "genre_classifier.joblib"))
REPORTS_DIR = Path(os.environ.get("MOVIE_GENRE_REPORTS_DIR", PROJECT_ROOT / "reports"))
FIGURES_DIR = REPORTS_DIR / "figures"

SEPARATOR = ":::"
RANDOM_STATE = 42
VALIDATION_SIZE = 0.20
TOP_K = 3
MIN_PLOT_WORDS = 3
