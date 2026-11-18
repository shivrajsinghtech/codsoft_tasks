import numpy as np
import pytest
from sklearn.pipeline import Pipeline

from src.models import MODEL_DISPLAY_NAMES, PARAM_GRIDS, build_pipeline
from tests.conftest import TINY_GENRES, TINY_VEC


@pytest.mark.parametrize(
    "name,params",
    [
        ("naive_bayes", {"alpha": 0.1}),
        ("logistic_regression", {"C": 5.0, "class_weight": "balanced"}),
        ("linear_svm", {"C": 0.5, "class_weight": "balanced"}),
    ],
)
def test_tfidf_models_train_and_predict(name, params, tiny_texts_labels):
    X, y = tiny_texts_labels
    pipe = build_pipeline(name, vectorizer_params=TINY_VEC, **params)
    assert isinstance(pipe, Pipeline) and pipe.steps[0][0] == "tfidf"
    pipe.fit(X, y)
    assert list(pipe.classes_) == TINY_GENRES
    preds = pipe.predict(["a haunted ghost and a demon scream in the graveyard at night"])
    assert preds[0] == "horror"
    proba = pipe.predict_proba(X[:4])
    assert proba.shape == (4, 3) and np.allclose(proba.sum(axis=1), 1)
    assert (pipe.predict(X) == np.array(y)).mean() >= 0.9


def test_word2vec_model_trains(tiny_texts_labels):
    X, y = tiny_texts_labels
    pipe = build_pipeline("word2vec_logreg", embedding_params={"min_count": 1, "workers": 1, "epochs": 5})
    pipe.fit(X * 3, y * 3)
    assert pipe.predict_proba(X[:2]).shape == (2, 3)
    emb = pipe.named_steps["embed"].transform(["zzzz unknownword"])
    assert emb.shape == (1, 100) and not emb.any()  # OOV text -> zero vector


def test_unknown_model_name():
    with pytest.raises(ValueError):
        build_pipeline("random_forest")


def test_grids_cover_all_models():
    assert set(PARAM_GRIDS) == set(MODEL_DISPLAY_NAMES)
    assert any(p.get("class_weight") == "balanced" for p in PARAM_GRIDS["linear_svm"])
