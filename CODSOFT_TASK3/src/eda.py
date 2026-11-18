"""Exploratory data analysis figures and summary statistics."""
from __future__ import annotations

from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
import seaborn as sns  # noqa: E402

from .config import FIGURES_DIR, TARGET  # noqa: E402

STAY, CHURN = "#4C72B0", "#DD5555"
PALETTE = {0: STAY, 1: CHURN}
LABELS = {0: "Stayed", 1: "Churned"}


def _save(fig, out_dir: Path, name: str) -> Path:
    path = out_dir / name
    fig.tight_layout()
    fig.savefig(path, dpi=120)
    plt.close(fig)
    return path


def _rate_bar(df, col, ax, order=None, title=None):
    rates = df.groupby(col)[TARGET].mean().mul(100)
    if order is not None:
        rates = rates.reindex(order)
    counts = df[col].value_counts()
    sns.barplot(x=rates.index.astype(str), y=rates.values, ax=ax, color=CHURN)
    for i, (k, v) in enumerate(rates.items()):
        ax.text(i, v + 0.8, f"{v:.1f}%\n(n={counts[k]:,})", ha="center", fontsize=8)
    ax.set_ylabel("Churn rate (%)")
    ax.set_xlabel(col)
    ax.set_ylim(0, max(rates.values) * 1.3)
    ax.set_title(title or f"Churn rate by {col}")


def generate_eda(df: pd.DataFrame, out_dir: Path = FIGURES_DIR) -> dict:
    """Create all EDA figures from the cleaned dataframe. Returns key findings."""
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    sns.set_theme(style="whitegrid")
    d = df.copy()
    d["Churn"] = d[TARGET].map(LABELS)

    # 1. Churn distribution
    fig, ax = plt.subplots(figsize=(6, 4))
    counts = d[TARGET].value_counts().sort_index()
    ax.bar([LABELS[i] for i in counts.index], counts.values, color=[STAY, CHURN])
    for i, v in enumerate(counts.values):
        ax.text(i, v + 80, f"{v:,} ({v / len(d):.1%})", ha="center")
    ax.set_title("Churn distribution (class imbalance)")
    ax.set_ylabel("Customers")
    _save(fig, out_dir, "01_churn_distribution.png")

    # 2. Demographics overview
    fig, axes = plt.subplots(1, 3, figsize=(15, 4))
    sns.countplot(data=d, x="Geography", ax=axes[0], color=STAY)
    axes[0].set_title("Customers by geography")
    sns.countplot(data=d, x="Gender", ax=axes[1], color=STAY)
    axes[1].set_title("Customers by gender")
    sns.histplot(data=d, x="Age", bins=30, ax=axes[2], color=STAY)
    axes[2].set_title("Age distribution")
    _save(fig, out_dir, "02_demographics.png")

    # 3. Age vs churn
    fig, axes = plt.subplots(1, 2, figsize=(13, 4))
    sns.histplot(data=d, x="Age", hue="Churn", bins=30, stat="density",
                 common_norm=False, element="step", ax=axes[0],
                 palette={"Stayed": STAY, "Churned": CHURN})
    axes[0].set_title("Age distribution by churn")
    d["AgeBand"] = pd.cut(d["Age"], [17, 30, 40, 50, 60, 100],
                          labels=["18-30", "31-40", "41-50", "51-60", "60+"])
    _rate_bar(d, "AgeBand", axes[1], title="Churn rate by age band")
    _save(fig, out_dir, "03_age_vs_churn.png")

    # 4. Geography & 5. Gender
    fig, ax = plt.subplots(figsize=(6, 4))
    _rate_bar(d, "Geography", ax)
    _save(fig, out_dir, "04_geography_vs_churn.png")
    fig, ax = plt.subplots(figsize=(6, 4))
    _rate_bar(d, "Gender", ax)
    _save(fig, out_dir, "05_gender_vs_churn.png")

    # 6. Credit score
    fig, axes = plt.subplots(1, 2, figsize=(13, 4))
    sns.kdeplot(data=d, x="CreditScore", hue="Churn", common_norm=False, fill=True,
                ax=axes[0], palette={"Stayed": STAY, "Churned": CHURN})
    axes[0].set_title("Credit score density by churn")
    sns.boxplot(data=d, x="Churn", y="CreditScore", ax=axes[1],
                hue="Churn", palette={"Stayed": STAY, "Churned": CHURN}, legend=False)
    axes[1].set_title("Credit score by churn")
    _save(fig, out_dir, "06_credit_score.png")

    # 7. Balance & salary
    fig, axes = plt.subplots(1, 3, figsize=(17, 4))
    sns.boxplot(data=d, x="Churn", y="Balance", ax=axes[0], hue="Churn",
                palette={"Stayed": STAY, "Churned": CHURN}, legend=False)
    axes[0].set_title("Balance by churn")
    sns.boxplot(data=d, x="Churn", y="EstimatedSalary", ax=axes[1], hue="Churn",
                palette={"Stayed": STAY, "Churned": CHURN}, legend=False)
    axes[1].set_title("Estimated salary by churn")
    d["ZeroBalance"] = np.where(d["Balance"] == 0, "Zero balance", "Has balance")
    _rate_bar(d, "ZeroBalance", axes[2], title="Churn rate: zero vs non-zero balance")
    _save(fig, out_dir, "07_balance_salary.png")

    # 8. Tenure
    fig, ax = plt.subplots(figsize=(9, 4))
    _rate_bar(d, "Tenure", ax, order=sorted(d["Tenure"].unique()))
    _save(fig, out_dir, "08_tenure_vs_churn.png")

    # 9. Active membership, products, credit card
    fig, axes = plt.subplots(1, 3, figsize=(16, 4))
    _rate_bar(d, "IsActiveMember", axes[0], title="Churn rate by active membership")
    _rate_bar(d, "NumOfProducts", axes[1], title="Churn rate by number of products")
    _rate_bar(d, "HasCrCard", axes[2], title="Churn rate by credit card ownership")
    _save(fig, out_dir, "09_activity_products.png")

    # 10. Correlation
    num = df.select_dtypes("number")
    corr = num.corr()
    fig, ax = plt.subplots(figsize=(9, 7))
    sns.heatmap(corr, annot=True, fmt=".2f", cmap="coolwarm", center=0, ax=ax,
                mask=np.triu(np.ones_like(corr, dtype=bool), k=1))
    ax.set_title("Correlation matrix (numeric features)")
    _save(fig, out_dir, "10_correlation_heatmap.png")

    def rates(col):
        return {str(k): round(float(v), 4) for k, v in df.groupby(col)[TARGET].mean().items()}

    return {
        "churn_rate": round(float(df[TARGET].mean()), 4),
        "churn_rate_by_geography": rates("Geography"),
        "churn_rate_by_gender": rates("Gender"),
        "churn_rate_by_active": rates("IsActiveMember"),
        "churn_rate_by_products": rates("NumOfProducts"),
        "churn_rate_by_age_band": {str(k): round(float(v), 4) for k, v in
                                   d.groupby("AgeBand", observed=True)[TARGET].mean().items()},
        "mean_age": {LABELS[k]: round(float(v), 1) for k, v in df.groupby(TARGET)["Age"].mean().items()},
        "zero_balance_churn_rate": round(float(df.loc[df.Balance == 0, TARGET].mean()), 4),
        "nonzero_balance_churn_rate": round(float(df.loc[df.Balance > 0, TARGET].mean()), 4),
        "correlation_with_target": {k: round(float(v), 4) for k, v in
                                    corr[TARGET].drop(TARGET).sort_values(key=abs, ascending=False).items()},
    }
