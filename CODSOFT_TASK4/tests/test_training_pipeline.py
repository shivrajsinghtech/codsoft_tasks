"""Checks on the model-evaluation pipeline and the saved reports."""

import json

import numpy as np
import pytest
from sklearn.model_selection import train_test_split

from src.config import METRICS_PATH, RANDOM_STATE
from src.evaluation import compute_metrics
from src.models import PARAM_GRIDS, build_pipelines


def test_compute_metrics_known_values():
    y_true = [0, 0, 1, 1, 1]
    y_pred = [0, 1, 1, 1, 0]
    m = compute_metrics(y_true, y_pred)
    assert m["accuracy"] == 0.6
    assert m["recall_spam"] == round(2 / 3, 4)
    assert m["precision_spam"] == round(2 / 3, 4)
    assert m["confusion_matrix"] == [[1, 1], [1, 2]]
    assert (m["tn"], m["fp"], m["fn"], m["tp"]) == (1, 1, 1, 2)


def test_three_models_defined():
    pipes = build_pipelines()
    assert set(pipes) == {"Multinomial Naive Bayes", "Logistic Regression", "Support Vector Machine"}
    assert set(PARAM_GRIDS) == set(pipes)
    for pipe in pipes.values():
        assert pipe.steps[0][0] == "tfidf"


def test_tfidf_fitted_on_training_data_only(clean_df):
    """The vectoriser vocabulary must come from the training split only."""
    X_tr, X_te, y_tr, y_te = train_test_split(
        clean_df["message"], clean_df["target"], test_size=0.2,
        stratify=clean_df["target"], random_state=RANDOM_STATE,
    )
    pipe = build_pipelines()["Multinomial Naive Bayes"]
    pipe.fit(X_tr, y_tr)
    vocab = set(pipe.named_steps["tfidf"].vocabulary_)
    from src.text_cleaning import clean_text
    train_tokens = {t for msg in X_tr for t in clean_text(msg).split()}
    unigrams = {v for v in vocab if " " not in v}
    assert unigrams <= train_tokens


def test_stratified_split_preserves_ratio(clean_df):
    _, _, y_tr, y_te = train_test_split(
        clean_df["message"], clean_df["target"], test_size=0.2,
        stratify=clean_df["target"], random_state=RANDOM_STATE,
    )
    assert abs(y_tr.mean() - y_te.mean()) < 0.01


@pytest.mark.parametrize("name", ["Multinomial Naive Bayes", "Logistic Regression", "Support Vector Machine"])
def test_each_pipeline_trains_and_evaluates(clean_df, name):
    X_tr, X_te, y_tr, y_te = train_test_split(
        clean_df["message"], clean_df["target"], test_size=0.2,
        stratify=clean_df["target"], random_state=RANDOM_STATE,
    )
    pipe = build_pipelines()[name].fit(X_tr, y_tr)
    proba = pipe.predict_proba(X_te)
    assert proba.shape == (len(X_te), 2)
    assert np.allclose(proba.sum(axis=1), 1.0)
    m = compute_metrics(y_te, pipe.predict(X_te), proba[:, 1])
    assert m["accuracy"] > 0.9
    # Untuned defaults: NB with alpha=1.0 reaches only ~0.66 spam recall,
    # which is why train.py tunes alpha (best 0.05 -> ~0.91 recall).
    assert m["recall_spam"] > 0.6


def test_saved_metrics_report_consistent():
    if not METRICS_PATH.exists():
        pytest.skip("Run train.py first.")
    data = json.loads(METRICS_PATH.read_text())
    assert data["best_model"] in data["models"]
    best = data["models"][data["best_model"]]
    assert best["cv_f1"] == max(m["cv_f1"] for m in data["models"].values())
    total = best["tn"] + best["fp"] + best["fn"] + best["tp"]
    assert total == data["split"]["test_size"]
    assert best["tp"] + best["fn"] == data["split"]["test_spam"]
