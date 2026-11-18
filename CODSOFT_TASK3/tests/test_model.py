import json

import numpy as np
from sklearn.model_selection import train_test_split

from src.config import METRICS_PATH, MODEL_PATH
from src.evaluate import evaluate_model
from src.models import build_pipeline, get_model_specs
from src.predict import load_model, predict_customer, save_model


def test_three_required_algorithms():
    names = [s.name for s in get_model_specs()]
    assert names == ["Logistic Regression", "Random Forest", "Gradient Boosting"]


def test_each_model_trains_and_predicts(xy):
    X, y = xy
    X_s, X_t, y_s, y_t = train_test_split(X, y, train_size=1500, test_size=500,
                                          stratify=y, random_state=1)
    for spec in get_model_specs():
        model = build_pipeline(spec.estimator)
        model.set_params(**{k: v[0] for k, v in spec.param_grid.items()})
        model.fit(X_s, y_s)
        m = evaluate_model(model, X_t, y_t)
        assert 0.5 < m["roc_auc"] <= 1.0, spec.name
        assert np.array(m["confusion_matrix"]).sum() == len(X_t)


def test_prediction_output(small_model, valid_customer):
    r = predict_customer(small_model, valid_customer)
    assert 0.0 <= r["churn_probability"] <= 1.0
    assert abs(r["churn_probability"] + r["stay_probability"] - 1) < 1e-9
    assert r["confidence"] >= 0.5
    assert r["will_churn"] == (r["churn_probability"] >= 0.5)


def test_serialization_roundtrip(small_model, xy, tmp_path):
    X, _ = xy
    path = save_model(small_model, tmp_path / "m.joblib")
    loaded = load_model(path)
    np.testing.assert_array_equal(loaded.predict_proba(X.head(50)),
                                  small_model.predict_proba(X.head(50)))


def test_load_missing_model_raises(tmp_path):
    import pytest
    with pytest.raises(FileNotFoundError):
        load_model(tmp_path / "missing.joblib")


def test_saved_production_model(xy, valid_customer):
    """The committed pipeline loads and performs well above chance."""
    model = load_model(MODEL_PATH)
    X, y = xy
    from sklearn.metrics import roc_auc_score
    assert roc_auc_score(y, model.predict_proba(X)[:, 1]) > 0.8
    r = predict_customer(model, valid_customer)
    assert 0.0 <= r["churn_probability"] <= 1.0
    metrics = json.loads(METRICS_PATH.read_text())
    assert metrics["best_model"] in metrics["models"]
