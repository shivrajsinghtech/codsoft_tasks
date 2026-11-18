"""Input validation, model persistence and single-customer prediction."""
from __future__ import annotations

import math
from pathlib import Path
from typing import Any, Mapping

import joblib
import pandas as pd

from .config import FEATURES, MODEL_PATH

GEOGRAPHIES = ("France", "Germany", "Spain")
GENDERS = ("Female", "Male")

# (min, max, must_be_integer) - ranges cover the training data with sensible margins.
NUMERIC_RULES: dict[str, tuple[float, float, bool]] = {
    "CreditScore": (300, 900, True),
    "Age": (18, 100, True),
    "Tenure": (0, 10, True),
    "Balance": (0, 1_000_000, False),
    "NumOfProducts": (1, 4, True),
    "HasCrCard": (0, 1, True),
    "IsActiveMember": (0, 1, True),
    "EstimatedSalary": (0, 1_000_000, False),
}
CATEGORICAL_RULES: dict[str, tuple[str, ...]] = {
    "Geography": GEOGRAPHIES,
    "Gender": GENDERS,
}


class InvalidInputError(ValueError):
    """Raised when customer input fails validation. `errors` lists every problem."""

    def __init__(self, errors: list[str]):
        self.errors = errors
        super().__init__("; ".join(errors))


def validate_customer(data: Mapping[str, Any]) -> dict:
    """Validate and normalise one customer record. Raises InvalidInputError."""
    if not isinstance(data, Mapping):
        raise InvalidInputError(["Customer data must be a dictionary of field values."])
    errors: list[str] = []
    clean: dict[str, Any] = {}

    for field in FEATURES:
        value = data.get(field)
        if value is None or (isinstance(value, str) and not value.strip()):
            errors.append(f"{field} is required.")
            continue

        if field in CATEGORICAL_RULES:
            allowed = CATEGORICAL_RULES[field]
            text = str(value).strip()
            match = next((a for a in allowed if a.lower() == text.lower()), None)
            if match is None:
                errors.append(f"{field} must be one of {', '.join(allowed)} (got {value!r}).")
            else:
                clean[field] = match
            continue

        lo, hi, integer = NUMERIC_RULES[field]
        if isinstance(value, bool):
            value = int(value)
        try:
            number = float(value)
        except (TypeError, ValueError):
            errors.append(f"{field} must be a number (got {value!r}).")
            continue
        if math.isnan(number) or math.isinf(number):
            errors.append(f"{field} must be a finite number.")
            continue
        if integer and not number.is_integer():
            errors.append(f"{field} must be a whole number (got {value!r}).")
            continue
        if not lo <= number <= hi:
            errors.append(f"{field} must be between {lo:g} and {hi:g} (got {number:g}).")
            continue
        clean[field] = int(number) if integer else number

    if errors:
        raise InvalidInputError(errors)
    return clean


def save_model(model, path: str | Path = MODEL_PATH) -> Path:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump(model, path, compress=3)
    return path


def load_model(path: str | Path = MODEL_PATH):
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(
            f"Trained model not found at {path}. Run `python train.py` first."
        )
    return joblib.load(path)


def predict_customer(model, data: Mapping[str, Any], threshold: float = 0.5) -> dict:
    """Validate input and return the churn prediction for one customer."""
    clean = validate_customer(data)
    frame = pd.DataFrame([clean], columns=FEATURES)
    proba = float(model.predict_proba(frame)[0, 1])
    will_churn = proba >= threshold
    return {
        "will_churn": bool(will_churn),
        "label": "Likely to churn" if will_churn else "Likely to stay",
        "churn_probability": proba,
        "stay_probability": 1.0 - proba,
        # Confidence = probability assigned to the predicted class.
        "confidence": proba if will_churn else 1.0 - proba,
        "input": clean,
    }
