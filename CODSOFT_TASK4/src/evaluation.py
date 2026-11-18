"""Metric computation and plotting helpers."""

from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
)

CLASS_LABELS = ["LEGITIMATE", "SPAM"]


def compute_metrics(y_true, y_pred, y_proba=None) -> dict:
    """Metrics where SPAM (1) is the positive class."""
    metrics = {
        "accuracy": accuracy_score(y_true, y_pred),
        "precision_spam": precision_score(y_true, y_pred, pos_label=1, zero_division=0),
        "recall_spam": recall_score(y_true, y_pred, pos_label=1, zero_division=0),
        "f1_spam": f1_score(y_true, y_pred, pos_label=1, zero_division=0),
        "precision_macro": precision_score(y_true, y_pred, average="macro", zero_division=0),
        "recall_macro": recall_score(y_true, y_pred, average="macro", zero_division=0),
        "f1_macro": f1_score(y_true, y_pred, average="macro", zero_division=0),
    }
    if y_proba is not None:
        metrics["roc_auc"] = roc_auc_score(y_true, y_proba)
    cm = confusion_matrix(y_true, y_pred, labels=[0, 1])
    metrics = {k: round(float(v), 4) for k, v in metrics.items()}
    metrics["confusion_matrix"] = cm.tolist()
    tn, fp, fn, tp = cm.ravel()
    metrics.update({"tn": int(tn), "fp": int(fp), "fn": int(fn), "tp": int(tp)})
    return metrics


def text_report(y_true, y_pred) -> str:
    return classification_report(
        y_true, y_pred, labels=[0, 1], target_names=CLASS_LABELS, digits=4, zero_division=0
    )


def plot_confusion_matrix(cm, title: str, path: Path) -> None:
    cm = np.asarray(cm)
    fig, ax = plt.subplots(figsize=(4.6, 3.9))
    sns.heatmap(
        cm, annot=True, fmt="d", cmap="Blues", cbar=False,
        xticklabels=CLASS_LABELS, yticklabels=CLASS_LABELS, ax=ax,
        annot_kws={"size": 13},
    )
    ax.set_xlabel("Predicted")
    ax.set_ylabel("Actual")
    ax.set_title(title, fontsize=11)
    fig.tight_layout()
    fig.savefig(path, dpi=130)
    plt.close(fig)


def plot_model_comparison(results: dict[str, dict], path: Path) -> None:
    keys = ["accuracy", "precision_spam", "recall_spam", "f1_spam"]
    names = list(results)
    x = np.arange(len(keys))
    width = 0.8 / len(names)
    fig, ax = plt.subplots(figsize=(8, 4.2))
    for i, name in enumerate(names):
        vals = [results[name][k] for k in keys]
        bars = ax.bar(x + i * width - 0.4 + width / 2, vals, width, label=name)
        ax.bar_label(bars, fmt="%.3f", fontsize=7, padding=2)
    ax.set_xticks(x, ["Accuracy", "Precision (spam)", "Recall (spam)", "F1 (spam)"])
    lo = min(results[n][k] for n in names for k in keys)
    ax.set_ylim(max(0.0, lo - 0.08), 1.02)
    ax.set_title("Model comparison on held-out test set")
    ax.legend(fontsize=8, loc="lower right")
    fig.tight_layout()
    fig.savefig(path, dpi=130)
    plt.close(fig)
