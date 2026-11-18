"""Exploratory data analysis: summary statistics and figures."""

from collections import Counter
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
from sklearn.feature_extraction.text import ENGLISH_STOP_WORDS

from src.text_cleaning import clean_text

PALETTE = {"ham": "#2E86AB", "spam": "#E4572E"}
_PLACEHOLDERS = {"numtoken", "phonetoken", "urltoken", "emailtoken", "currencytoken"}


def add_text_features(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()
    out["char_count"] = out["message"].str.len()
    out["word_count"] = out["message"].str.split().str.len()
    out["digit_count"] = out["message"].str.count(r"\d")
    out["upper_ratio"] = out["message"].apply(
        lambda s: sum(c.isupper() for c in s) / max(1, sum(c.isalpha() for c in s))
    )
    return out


def length_summary(df: pd.DataFrame) -> pd.DataFrame:
    feats = add_text_features(df)
    return (
        feats.groupby("label")[["char_count", "word_count", "digit_count", "upper_ratio"]]
        .mean()
        .round(2)
    )


def top_words(df: pd.DataFrame, label: str, n: int = 15) -> list[tuple[str, int]]:
    counter: Counter = Counter()
    for msg in df.loc[df["label"] == label, "message"]:
        counter.update(
            w for w in clean_text(msg).split()
            if len(w) > 2 and w not in ENGLISH_STOP_WORDS and w not in _PLACEHOLDERS
        )
    return counter.most_common(n)


def plot_class_distribution(df: pd.DataFrame, path: Path) -> None:
    counts = df["label"].value_counts().reindex(["ham", "spam"])
    fig, axes = plt.subplots(1, 2, figsize=(9, 3.8))
    bars = axes[0].bar(["Legitimate (ham)", "Spam"], counts.values,
                       color=[PALETTE["ham"], PALETTE["spam"]])
    axes[0].bar_label(bars, padding=3)
    axes[0].set_ylabel("Messages")
    axes[0].set_title("Class counts")
    axes[0].set_ylim(0, counts.max() * 1.12)
    axes[1].pie(counts.values, labels=["Legitimate", "Spam"], autopct="%1.1f%%",
                colors=[PALETTE["ham"], PALETTE["spam"]], startangle=90,
                wedgeprops={"edgecolor": "white"})
    axes[1].set_title("Class share")
    fig.suptitle("Spam vs legitimate SMS (after cleaning)")
    fig.tight_layout()
    fig.savefig(path, dpi=130)
    plt.close(fig)


def plot_length_distribution(df: pd.DataFrame, path: Path) -> None:
    feats = add_text_features(df)
    fig, axes = plt.subplots(1, 2, figsize=(10, 3.8))
    for ax, col, title in [(axes[0], "char_count", "Characters per message"),
                           (axes[1], "word_count", "Words per message")]:
        sns.histplot(data=feats, x=col, hue="label", bins=50, stat="density",
                     common_norm=False, palette=PALETTE, element="step", ax=ax)
        ax.set_title(title)
        ax.set_xlim(0, feats[col].quantile(0.99))
    fig.tight_layout()
    fig.savefig(path, dpi=130)
    plt.close(fig)


def plot_top_words(df: pd.DataFrame, path: Path, n: int = 15) -> None:
    fig, axes = plt.subplots(1, 2, figsize=(10, 4.6))
    for ax, label in zip(axes, ["ham", "spam"]):
        words = top_words(df, label, n)
        ax.barh([w for w, _ in words][::-1], [c for _, c in words][::-1], color=PALETTE[label])
        ax.set_title(f"Top {n} words - {'legitimate' if label == 'ham' else 'spam'}")
        ax.set_xlabel("Frequency")
    fig.tight_layout()
    fig.savefig(path, dpi=130)
    plt.close(fig)


def run_eda(df: pd.DataFrame, figures_dir: Path) -> dict:
    figures_dir.mkdir(parents=True, exist_ok=True)
    plot_class_distribution(df, figures_dir / "class_distribution.png")
    plot_length_distribution(df, figures_dir / "message_length_distribution.png")
    plot_top_words(df, figures_dir / "top_words.png")
    return {
        "length_summary": length_summary(df).to_dict(orient="index"),
        "top_spam_words": top_words(df, "spam", 10),
        "top_ham_words": top_words(df, "ham", 10),
    }
