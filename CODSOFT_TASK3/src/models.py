"""Model definitions, hyper-parameter grids and cross-validated tuning."""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd
from sklearn.ensemble import GradientBoostingClassifier, RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import GridSearchCV, StratifiedKFold
from sklearn.pipeline import Pipeline
from sklearn.utils.class_weight import compute_sample_weight

from .config import CV_FOLDS, RANDOM_STATE
from .preprocessing import build_preprocessor

SCORING = {
    "f1": "f1",
    "recall": "recall",
    "precision": "precision",
    "roc_auc": "roc_auc",
    "accuracy": "accuracy",
}
# Model selection metric: F1 on the churn class balances catching churners
# (recall) against wasting retention effort on loyal customers (precision).
REFIT_METRIC = "f1"


@dataclass
class ModelSpec:
    name: str
    estimator: object
    param_grid: dict
    # GradientBoostingClassifier has no class_weight, so balanced sample
    # weights are passed to fit() instead.
    use_sample_weight: bool = False


def get_model_specs() -> list[ModelSpec]:
    return [
        ModelSpec(
            "Logistic Regression",
            LogisticRegression(
                class_weight="balanced", max_iter=2000, random_state=RANDOM_STATE
            ),
            {"clf__C": [0.01, 0.1, 1.0, 10.0]},
        ),
        ModelSpec(
            "Random Forest",
            RandomForestClassifier(
                class_weight="balanced", random_state=RANDOM_STATE, n_jobs=-1
            ),
            {
                "clf__n_estimators": [300],
                "clf__max_depth": [8, 12, None],
                "clf__min_samples_leaf": [1, 5, 10],
            },
        ),
        ModelSpec(
            "Gradient Boosting",
            GradientBoostingClassifier(random_state=RANDOM_STATE),
            {
                "clf__n_estimators": [150, 300],
                "clf__learning_rate": [0.05, 0.1],
                "clf__max_depth": [3, 4],
            },
            use_sample_weight=True,
        ),
    ]


def build_pipeline(estimator) -> Pipeline:
    return Pipeline([("preprocess", build_preprocessor()), ("clf", estimator)])


def tune_model(spec: ModelSpec, X: pd.DataFrame, y: pd.Series, n_jobs: int = -1):
    """Grid-search a model with stratified K-fold CV on the training data only."""
    cv = StratifiedKFold(n_splits=CV_FOLDS, shuffle=True, random_state=RANDOM_STATE)
    search = GridSearchCV(
        build_pipeline(spec.estimator),
        spec.param_grid,
        scoring=SCORING,
        refit=REFIT_METRIC,
        cv=cv,
        n_jobs=n_jobs,
    )
    fit_params = {}
    if spec.use_sample_weight:
        fit_params["clf__sample_weight"] = compute_sample_weight("balanced", y)
    search.fit(X, y, **fit_params)
    return search


def cv_summary(search: GridSearchCV) -> dict:
    """Mean/std CV scores of the selected hyper-parameter combination."""
    i = search.best_index_
    res = search.cv_results_
    out = {}
    for metric in SCORING:
        out[metric] = {
            "mean": float(res[f"mean_test_{metric}"][i]),
            "std": float(res[f"std_test_{metric}"][i]),
        }
    out["best_params"] = {
        k.replace("clf__", ""): (v if not isinstance(v, np.generic) else v.item())
        for k, v in search.best_params_.items()
    }
    return out
