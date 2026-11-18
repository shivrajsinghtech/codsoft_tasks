"""Model persistence and user-facing prediction with input validation."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import joblib
import numpy as np

from src.config import MIN_PLOT_WORDS, MODEL_PATH, TOP_K
from src.preprocessing import build_text, clean_text


class InvalidInputError(ValueError):
    """Raised for empty, non-text or too-short plot summaries."""


def metadata_path(model_path: str | Path) -> Path:
    model_path = Path(model_path)
    return model_path.with_name(model_path.stem + "_metadata.json")


def save_model(pipeline, path: str | Path = MODEL_PATH, metadata: dict | None = None) -> Path:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump(pipeline, path, compress=3)
    if metadata is not None:
        metadata_path(path).write_text(json.dumps(metadata, indent=2, default=str))
    return path


def load_model(path: str | Path = MODEL_PATH):
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(f"Model not found at {path}. Run `python train.py` first.")
    return joblib.load(path)


def validate_plot(plot: object, min_words: int = MIN_PLOT_WORDS) -> str:
    if plot is None or not isinstance(plot, str):
        raise InvalidInputError("Plot summary must be text.")
    plot = plot.strip()
    if not plot:
        raise InvalidInputError("Plot summary is empty. Please describe the movie's story.")
    if not any(ch.isalpha() for ch in plot):
        raise InvalidInputError("Plot summary must contain words, not only numbers or symbols.")
    n_words = len(clean_text(plot).split())
    if n_words < min_words:
        raise InvalidInputError(
            f"Plot summary is too short ({n_words} word(s)); please write at least {min_words} words."
        )
    return plot


def _scores(pipeline, texts: list[str]) -> np.ndarray:
    if hasattr(pipeline, "predict_proba"):
        return pipeline.predict_proba(texts)
    decision = pipeline.decision_function(texts)  # softmax fallback
    decision = decision - decision.max(axis=1, keepdims=True)
    exp = np.exp(decision)
    return exp / exp.sum(axis=1, keepdims=True)


class GenrePredictor:
    """Wraps a fitted text pipeline and returns the top-k genres with scores."""

    def __init__(self, pipeline, metadata: dict | None = None) -> None:
        self.pipeline = pipeline
        self.metadata = metadata or {}
        self.classes = [str(c) for c in pipeline.classes_]

    @classmethod
    def load(cls, path: str | Path = MODEL_PATH) -> GenrePredictor:
        meta_file = metadata_path(path)
        metadata = json.loads(meta_file.read_text()) if meta_file.exists() else {}
        return cls(load_model(path), metadata)

    def predict(self, plot: object, title: object = "", top_k: int = TOP_K) -> dict[str, Any]:
        plot = validate_plot(plot)
        if title is not None and not isinstance(title, str):
            raise InvalidInputError("Title must be text.")
        if not isinstance(top_k, int) or top_k < 1:
            raise InvalidInputError("top_k must be a positive integer.")
        top_k = min(top_k, len(self.classes))
        scores = _scores(self.pipeline, [build_text(title or "", plot)])[0]
        order = np.argsort(scores)[::-1][:top_k]
        top = [{"genre": self.classes[i], "score": float(scores[i])} for i in order]
        return {"genre": top[0]["genre"], "confidence": top[0]["score"], "top_k": top}

    def predict_batch(self, titles, descriptions, top_k: int = TOP_K) -> list[dict[str, Any]]:
        texts = [build_text(t, d) for t, d in zip(titles, descriptions)]
        scores = _scores(self.pipeline, texts)
        order = np.argsort(scores, axis=1)[:, ::-1][:, :top_k]
        results = []
        for row, idx in zip(scores, order):
            top = [{"genre": self.classes[i], "score": float(row[i])} for i in idx]
            results.append({"genre": top[0]["genre"], "confidence": top[0]["score"], "top_k": top})
        return results
