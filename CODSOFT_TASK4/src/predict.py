"""Model loading and single-message prediction with input validation."""

import re
from functools import lru_cache
from pathlib import Path

import joblib

from src.config import LABEL_NAMES, MAX_MESSAGE_LENGTH, MODEL_PATH

_HAS_ALNUM = re.compile(r"[A-Za-z0-9]")


class InvalidInputError(ValueError):
    """Raised when an SMS message cannot be classified."""


def load_model(path: Path = MODEL_PATH):
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(
            f"Trained model not found at {path}. Run `python train.py` first."
        )
    model = joblib.load(path)
    if not (hasattr(model, "predict") and hasattr(model, "predict_proba")):
        raise TypeError("Loaded object is not a classifier pipeline with predict_proba.")
    return model


@lru_cache(maxsize=1)
def get_default_model():
    return load_model(MODEL_PATH)


def validate_message(message) -> str:
    if message is None:
        raise InvalidInputError("Message is empty. Please enter an SMS message.")
    if not isinstance(message, str):
        raise InvalidInputError(
            f"Message must be text, got {type(message).__name__}."
        )
    message = message.strip()
    if not message:
        raise InvalidInputError("Message is empty. Please enter an SMS message.")
    if len(message) > MAX_MESSAGE_LENGTH:
        raise InvalidInputError(
            f"Message is too long ({len(message)} characters, max {MAX_MESSAGE_LENGTH})."
        )
    if not _HAS_ALNUM.search(message):
        raise InvalidInputError(
            "Message contains no letters or digits, so it cannot be classified."
        )
    return message


def predict_message(message, model=None) -> dict:
    """Classify one SMS.

    Returns ``{"label", "is_spam", "confidence", "spam_probability"}`` where
    ``confidence`` is the model probability of the predicted class.
    """
    text = validate_message(message)
    model = model if model is not None else get_default_model()
    proba = model.predict_proba([text])[0]
    classes = list(model.classes_)
    spam_prob = float(proba[classes.index(1)])
    pred = int(model.predict([text])[0])
    return {
        "label": LABEL_NAMES[pred],
        "is_spam": pred == 1,
        "confidence": spam_prob if pred == 1 else 1.0 - spam_prob,
        "spam_probability": spam_prob,
    }
