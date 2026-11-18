"""Pipeline factories and hyper-parameter grids for every model."""

from __future__ import annotations

from typing import Any

import numpy as np
from sklearn.calibration import CalibratedClassifierCV
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.naive_bayes import MultinomialNB
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import LinearSVC

from src.config import RANDOM_STATE
from src.embeddings import Word2VecVectorizer
from src.preprocessing import clean_text

MODEL_DISPLAY_NAMES = {
    "naive_bayes": "TF-IDF + Multinomial Naive Bayes",
    "logistic_regression": "TF-IDF + Logistic Regression",
    "linear_svm": "TF-IDF + Linear SVM",
    "word2vec_logreg": "Word2Vec embeddings + Logistic Regression (supplementary)",
}
TFIDF_MODELS = ("naive_bayes", "logistic_regression", "linear_svm")

TFIDF_PARAMS: dict[str, Any] = {
    "preprocessor": clean_text,
    "stop_words": "english",
    "ngram_range": (1, 2),  # unigrams + bigrams
    "min_df": 2,  # drop terms seen in only one plot (typos, names)
    "max_df": 0.9,
    "sublinear_tf": True,  # 1 + log(tf)
    "max_features": 150_000,  # keeps the saved model a manageable size
    "dtype": np.float32,
}

# Grids are evaluated on the stratified validation split only.
PARAM_GRIDS: dict[str, list[dict[str, Any]]] = {
    "naive_bayes": [
        {"alpha": a, "fit_prior": fp} for a in (0.01, 0.03, 0.1, 0.3, 1.0) for fp in (True, False)
    ],
    "logistic_regression": [
        {"C": c, "class_weight": cw} for c in (1.0, 5.0, 20.0) for cw in (None, "balanced")
    ],
    "linear_svm": [
        {"C": c, "class_weight": cw} for c in (0.1, 0.3, 1.0) for cw in (None, "balanced")
    ],
    "word2vec_logreg": [{"C": 1.0, "class_weight": cw} for cw in (None, "balanced")],
}


def build_vectorizer(**overrides: Any) -> TfidfVectorizer:
    return TfidfVectorizer(**{**TFIDF_PARAMS, **overrides})


def build_classifier(name: str, **params: Any):
    """Classifier for a TF-IDF model (or the LR head of the embedding model)."""
    if name == "naive_bayes":
        return MultinomialNB(**params)
    if name in ("logistic_regression", "word2vec_logreg"):
        return LogisticRegression(max_iter=3000, random_state=RANDOM_STATE, **params)
    if name == "linear_svm":
        calibration_cv = params.pop("calibration_cv", 3)
        # LinearSVC has no predict_proba; sigmoid calibration gives scores for top-3 output.
        return CalibratedClassifierCV(
            LinearSVC(random_state=RANDOM_STATE, **params),
            method="sigmoid",
            cv=calibration_cv,
            ensemble=False,
        )
    raise ValueError(f"Unknown model name {name!r}; choose from {sorted(MODEL_DISPLAY_NAMES)}")


def build_pipeline(
    name: str,
    vectorizer_params: dict[str, Any] | None = None,
    embedding_params: dict[str, Any] | None = None,
    **clf_params: Any,
) -> Pipeline:
    """Full text -> genre pipeline (vectoriser and classifier fitted together)."""
    if name == "word2vec_logreg":
        return Pipeline(
            [
                ("embed", Word2VecVectorizer(**(embedding_params or {}))),
                ("scale", StandardScaler()),
                ("clf", build_classifier(name, **clf_params)),
            ]
        )
    return Pipeline(
        [
            ("tfidf", build_vectorizer(**(vectorizer_params or {}))),
            ("clf", build_classifier(name, **clf_params)),
        ]
    )
