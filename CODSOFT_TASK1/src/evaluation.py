"""Metrics and evaluation plots."""

from __future__ import annotations

from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
import seaborn as sns  # noqa: E402
from sklearn.metrics import (  # noqa: E402
    accuracy_score,
    confusion_matrix,
    f1_score,
    precision_recall_fscore_support,
    precision_score,
    recall_score,
)


def compute_metrics(y_true, y_pred) -> dict[str, float]:
    return {
        "accuracy": float(accuracy_score(y_true, y_pred)),
        "macro_precision": float(precision_score(y_true, y_pred, average="macro", zero_division=0)),
        "macro_recall": float(recall_score(y_true, y_pred, average="macro", zero_division=0)),
        "macro_f1": float(f1_score(y_true, y_pred, average="macro", zero_division=0)),
        "weighted_f1": float(f1_score(y_true, y_pred, average="weighted", zero_division=0)),
    }


def per_genre_report(y_true, y_pred, labels) -> pd.DataFrame:
    p, r, f, s = precision_recall_fscore_support(y_true, y_pred, labels=labels, zero_division=0)
    return (
        pd.DataFrame({"genre": labels, "precision": p, "recall": r, "f1": f, "support": s})
        .sort_values("support", ascending=False)
        .reset_index(drop=True)
    )


def plot_confusion_matrix(y_true, y_pred, labels, path: str | Path, title: str) -> np.ndarray:
    """Row-normalised confusion matrix (each row = recall distribution of a true genre)."""
    cm = confusion_matrix(y_true, y_pred, labels=labels)
    with np.errstate(invalid="ignore", divide="ignore"):
        norm = np.nan_to_num(cm / cm.sum(axis=1, keepdims=True))
    fig, ax = plt.subplots(figsize=(16, 13))
    sns.heatmap(
        norm,
        annot=True,
        fmt=".2f",
        cmap="Blues",
        xticklabels=labels,
        yticklabels=labels,
        cbar_kws={"label": "Fraction of true genre"},
        annot_kws={"size": 7},
        ax=ax,
        vmin=0,
        vmax=1,
    )
    ax.set_xlabel("Predicted genre")
    ax.set_ylabel("True genre")
    ax.set_title(title)
    fig.tight_layout()
    fig.savefig(path, dpi=110)
    plt.close(fig)
    return cm


def plot_model_comparison(df: pd.DataFrame, path: str | Path, title: str) -> None:
    metrics = ["accuracy", "macro_precision", "macro_recall", "macro_f1", "weighted_f1"]
    long = df.melt(id_vars="model", value_vars=metrics, var_name="metric", value_name="score")
    fig, ax = plt.subplots(figsize=(12, 5.5))
    sns.barplot(data=long, x="metric", y="score", hue="model", ax=ax)
    for container in ax.containers:
        ax.bar_label(container, fmt="%.3f", fontsize=7, padding=2)
    ax.set_ylim(0, 1)
    ax.set_title(title)
    ax.set_xlabel("")
    ax.legend(loc="upper right", fontsize=8)
    fig.tight_layout()
    fig.savefig(path, dpi=110)
    plt.close(fig)


def plot_per_genre(report: pd.DataFrame, path: str | Path, title: str) -> None:
    data = report.sort_values("f1")
    fig, ax = plt.subplots(figsize=(10, 9))
    y = np.arange(len(data))
    ax.barh(y - 0.27, data["precision"], height=0.27, label="Precision")
    ax.barh(y, data["recall"], height=0.27, label="Recall")
    ax.barh(y + 0.27, data["f1"], height=0.27, label="F1")
    ax.set_yticks(y, [f"{g} (n={n})" for g, n in zip(data["genre"], data["support"])])
    ax.set_xlim(0, 1)
    ax.set_title(title)
    ax.legend(loc="lower right")
    fig.tight_layout()
    fig.savefig(path, dpi=110)
    plt.close(fig)
