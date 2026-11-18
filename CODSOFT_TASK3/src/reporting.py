"""Model-evaluation figures and markdown reports."""
from __future__ import annotations

from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
from sklearn.metrics import ConfusionMatrixDisplay, RocCurveDisplay  # noqa: E402

METRIC_NAMES = ["accuracy", "precision", "recall", "f1", "roc_auc"]


def plot_model_comparison(results: dict, out: Path) -> None:
    names = list(results)
    x = np.arange(len(METRIC_NAMES))
    w = 0.8 / len(names)
    fig, ax = plt.subplots(figsize=(11, 5))
    for i, n in enumerate(names):
        vals = [results[n]["test"][m] for m in METRIC_NAMES]
        bars = ax.bar(x + i * w - 0.4 + w / 2, vals, w, label=n)
        ax.bar_label(bars, fmt="%.2f", fontsize=7)
    ax.set_xticks(x, [m.upper() if m == "roc_auc" else m.title() for m in METRIC_NAMES])
    ax.set_ylim(0, 1.05)
    ax.set_title("Test-set metrics by model (churn = positive class)")
    ax.legend(loc="lower right")
    fig.tight_layout()
    fig.savefig(out, dpi=120)
    plt.close(fig)


def plot_confusion_matrices(models: dict, X_test, y_test, out: Path) -> None:
    fig, axes = plt.subplots(1, len(models), figsize=(5 * len(models), 4.2))
    for ax, (name, model) in zip(np.atleast_1d(axes), models.items()):
        ConfusionMatrixDisplay.from_estimator(
            model, X_test, y_test, display_labels=["Stayed", "Churned"],
            cmap="Blues", ax=ax, colorbar=False,
        )
        ax.set_title(name)
        ax.grid(False)
    fig.tight_layout()
    fig.savefig(out, dpi=120)
    plt.close(fig)


def plot_single_confusion(model, X_test, y_test, title: str, out: Path) -> None:
    fig, ax = plt.subplots(figsize=(5, 4.2))
    ConfusionMatrixDisplay.from_estimator(
        model, X_test, y_test, display_labels=["Stayed", "Churned"],
        cmap="Blues", ax=ax, colorbar=False,
    )
    ax.set_title(title)
    ax.grid(False)
    fig.tight_layout()
    fig.savefig(out, dpi=120)
    plt.close(fig)


def plot_roc_curves(models: dict, X_test, y_test, out: Path) -> None:
    fig, ax = plt.subplots(figsize=(6.5, 5.5))
    for name, model in models.items():
        RocCurveDisplay.from_estimator(model, X_test, y_test, name=name, ax=ax)
    ax.plot([0, 1], [0, 1], "k--", lw=1, label="Chance")
    ax.set_title("ROC curves (test set)")
    ax.legend(loc="lower right")
    fig.tight_layout()
    fig.savefig(out, dpi=120)
    plt.close(fig)


def plot_feature_importance(importance: dict, title: str, out: Path) -> None:
    names = list(importance)[::-1]
    means = [importance[n]["mean"] for n in names]
    stds = [importance[n]["std"] for n in names]
    fig, ax = plt.subplots(figsize=(8, 5))
    ax.barh(names, means, xerr=stds, color="#DD5555")
    ax.set_xlabel("Mean drop in ROC-AUC when feature is shuffled")
    ax.set_title(title)
    fig.tight_layout()
    fig.savefig(out, dpi=120)
    plt.close(fig)


def write_model_report(metrics: dict, out: Path) -> None:
    res = metrics["models"]
    best = metrics["best_model"]
    lines = [
        "# Model Comparison Report",
        "",
        f"Dataset: `{metrics['data']['rows']:,}` customers after cleaning; "
        f"stratified {int((1 - metrics['split']['test_size']) * 100)}/"
        f"{int(metrics['split']['test_size'] * 100)} split "
        f"(train = {metrics['split']['train_rows']:,}, test = {metrics['split']['test_rows']:,}), "
        f"`random_state={metrics['split']['random_state']}`.",
        "",
        f"Hyper-parameters were tuned with {metrics['cv']['folds']}-fold stratified "
        f"cross-validation on the **training set only**, selecting on **{metrics['cv']['refit_metric'].upper()}** "
        "of the churn class. Class imbalance was handled with `class_weight='balanced'` "
        "(Logistic Regression, Random Forest) and balanced sample weights (Gradient Boosting). "
        "The test set was used once, for the final evaluation below.",
        "",
        "## Cross-validation (training set, mean ± std)",
        "",
        "| Model | F1 | Recall | Precision | ROC-AUC | Accuracy | Best params |",
        "|---|---|---|---|---|---|---|",
    ]
    for n, r in res.items():
        cv = r["cv"]
        params = ", ".join(f"{k}={v}" for k, v in cv["best_params"].items())
        lines.append(
            f"| {n} | " + " | ".join(
                f"{cv[m]['mean']:.4f} ± {cv[m]['std']:.4f}"
                for m in ["f1", "recall", "precision", "roc_auc", "accuracy"]
            ) + f" | {params} |"
        )
    lines += [
        "",
        "## Held-out test set",
        "",
        "| Model | Accuracy | Precision | Recall | F1 | ROC-AUC | TN | FP | FN | TP |",
        "|---|---|---|---|---|---|---|---|---|---|",
    ]
    for n, r in res.items():
        t = r["test"]
        (tn, fp), (fn, tp) = t["confusion_matrix"]
        lines.append(
            f"| {n} | {t['accuracy']:.4f} | {t['precision']:.4f} | {t['recall']:.4f} | "
            f"{t['f1']:.4f} | {t['roc_auc']:.4f} | {tn} | {fp} | {fn} | {tp} |"
        )
    lines += [
        "",
        f"## Selected model: **{best}**",
        "",
        metrics["selection_rationale"],
        "",
        "### Classification report (test set)",
        "",
        "```",
        res[best]["test"]["classification_report_text"].rstrip(),
        "```",
        "",
        "### Permutation feature importance (test set, drop in ROC-AUC)",
        "",
        "| Feature | Importance | Std |",
        "|---|---|---|",
    ]
    for f, v in metrics["feature_importance"].items():
        lines.append(f"| {f} | {v['mean']:.4f} | {v['std']:.4f} |")
    lines += [
        "",
        "## Figures",
        "",
        "- `figures/11_model_comparison.png`",
        "- `figures/12_confusion_matrices.png`",
        "- `figures/13_roc_curves.png`",
        "- `figures/14_feature_importance.png`",
        "- `figures/15_best_model_confusion_matrix.png`",
        "",
    ]
    out.write_text("\n".join(lines), encoding="utf-8")


def write_eda_report(quality: dict, eda: dict, out: Path) -> None:
    pct = lambda v: f"{v * 100:.1f}%"  # noqa: E731
    lines = [
        "# Exploratory Data Analysis Report",
        "",
        "## Data quality",
        "",
        f"- Rows: **{quality['rows']:,}**, columns after dropping identifiers: **{quality['columns']}**",
        f"- Missing values: **{quality['total_missing']}**",
        f"- Duplicate rows: **{quality['duplicate_rows']}**",
        f"- Identifier columns removed: `RowNumber`, `CustomerId`, `Surname` (no predictive meaning; "
        "`Surname` would also risk spurious memorisation).",
        "",
        "## Class balance",
        "",
        f"- Churned: **{quality['class_counts']['1']:,}** ({pct(eda['churn_rate'])}), "
        f"stayed: **{quality['class_counts']['0']:,}** ({pct(1 - eda['churn_rate'])}).",
        "- The data is imbalanced (~1 : 4), so accuracy alone is misleading - a model that "
        "always predicts \"stay\" would score ~80%. Recall, precision, F1 and ROC-AUC are reported.",
        "",
        "## Key findings",
        "",
        "| Segment | Churn rate |",
        "|---|---|",
    ]
    for k, v in eda["churn_rate_by_geography"].items():
        lines.append(f"| Geography = {k} | {pct(v)} |")
    for k, v in eda["churn_rate_by_gender"].items():
        lines.append(f"| Gender = {k} | {pct(v)} |")
    for k, v in eda["churn_rate_by_active"].items():
        lines.append(f"| IsActiveMember = {k} | {pct(v)} |")
    for k, v in eda["churn_rate_by_products"].items():
        lines.append(f"| NumOfProducts = {k} | {pct(v)} |")
    for k, v in eda["churn_rate_by_age_band"].items():
        lines.append(f"| Age {k} | {pct(v)} |")
    lines.append(f"| Balance = 0 | {pct(eda['zero_balance_churn_rate'])} |")
    lines.append(f"| Balance > 0 | {pct(eda['nonzero_balance_churn_rate'])} |")
    lines += [
        "",
        f"- Mean age: churned **{eda['mean_age']['Churned']}** vs stayed **{eda['mean_age']['Stayed']}**.",
        "",
        "### Correlation with `Exited` (Pearson, numeric columns)",
        "",
        "| Feature | r |",
        "|---|---|",
    ]
    for k, v in eda["correlation_with_target"].items():
        lines.append(f"| {k} | {v:+.3f} |")
    lines += [
        "",
        "Note: Pearson correlation only captures linear, monotonic relationships. "
        "`NumOfProducts` has a weak linear correlation but a strong non-linear effect "
        "(churn is lowest at 2 products and very high at 3-4), which tree-based models can capture.",
        "",
        "## Figures (`reports/figures/`)",
        "",
    ]
    for p in sorted(Path(out.parent / "figures").glob("0*.png")) + sorted(
        Path(out.parent / "figures").glob("10_*.png")
    ):
        lines.append(f"- `figures/{p.name}`")
    lines.append("")
    out.write_text("\n".join(lines), encoding="utf-8")
