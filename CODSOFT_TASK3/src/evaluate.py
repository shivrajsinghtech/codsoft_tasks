"""Evaluation helpers."""
from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.inspection import permutation_importance
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
)

from .config import FEATURES, RANDOM_STATE


def evaluate_model(model, X_test: pd.DataFrame, y_test: pd.Series) -> dict:
    y_pred = model.predict(X_test)
    y_proba = model.predict_proba(X_test)[:, 1]
    return {
        "accuracy": float(accuracy_score(y_test, y_pred)),
        "precision": float(precision_score(y_test, y_pred, zero_division=0)),
        "recall": float(recall_score(y_test, y_pred, zero_division=0)),
        "f1": float(f1_score(y_test, y_pred, zero_division=0)),
        "roc_auc": float(roc_auc_score(y_test, y_proba)),
        "confusion_matrix": confusion_matrix(y_test, y_pred).tolist(),
        "classification_report": classification_report(
            y_test, y_pred, target_names=["Stayed", "Churned"], output_dict=True,
            zero_division=0,
        ),
        "classification_report_text": classification_report(
            y_test, y_pred, target_names=["Stayed", "Churned"], zero_division=0
        ),
    }


def feature_importance(model, X_test: pd.DataFrame, y_test: pd.Series) -> dict:
    """Permutation importance (drop in ROC-AUC) on the raw input features.

    Model-agnostic and computed on held-out data, so it is comparable across
    algorithms and reflects the original (un-encoded) features.
    """
    result = permutation_importance(
        model, X_test[FEATURES], y_test, scoring="roc_auc", n_repeats=10,
        random_state=RANDOM_STATE, n_jobs=-1,
    )
    order = np.argsort(result.importances_mean)[::-1]
    return {
        FEATURES[i]: {
            "mean": float(result.importances_mean[i]),
            "std": float(result.importances_std[i]),
        }
        for i in order
    }
