"""Exploratory data analysis on the training data (and unlabelled test text).

Test-solution labels are never used here.
"""

from __future__ import annotations

import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
import seaborn as sns  # noqa: E402
from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer  # noqa: E402

from src.preprocessing import clean_text  # noqa: E402

sns.set_theme(style="whitegrid")


def dataset_summary(train: pd.DataFrame, test: pd.DataFrame) -> dict:
    words = train["description"].str.split().str.len()
    return {
        "train_records": int(len(train)),
        "test_records": int(len(test)),
        "n_genres": int(train["genre"].nunique()),
        "columns_train": list(train.columns),
        "columns_test": list(test.columns),
        "missing_values_train": {k: int(v) for k, v in train.isna().sum().items()},
        "empty_descriptions_train": int((train["description"].str.strip() == "").sum()),
        "empty_descriptions_test": int((test["description"].str.strip() == "").sum()),
        "duplicate_ids_train": int(train["id"].duplicated().sum()),
        "duplicate_rows_train": int(train.duplicated(["title", "genre", "description"]).sum()),
        "duplicate_descriptions_train": int(train["description"].duplicated().sum()),
        "duplicate_descriptions_conflicting_genre": int(
            train[train["description"].duplicated(keep=False)]
            .groupby("description")["genre"]
            .nunique()
            .gt(1)
            .sum()
        ),
        "train_descriptions_also_in_test": int(train["description"].isin(set(test["description"])).sum()),
        "genre_counts": {k: int(v) for k, v in train["genre"].value_counts().items()},
        "imbalance_ratio_largest_to_smallest": float(
            train["genre"].value_counts().max() / train["genre"].value_counts().min()
        ),
        "description_words": {
            k: float(v) for k, v in words.describe(percentiles=[0.25, 0.5, 0.75, 0.95]).items()
        },
        "median_words_by_genre": {
            k: float(v) for k, v in train.assign(words=words).groupby("genre")["words"].median().items()
        },
    }


def plot_genre_distribution(train: pd.DataFrame, path: Path) -> None:
    counts = train["genre"].value_counts()
    fig, ax = plt.subplots(figsize=(11, 8))
    sns.barplot(x=counts.values, y=counts.index, ax=ax, color="#4C72B0")
    for i, v in enumerate(counts.values):
        ax.text(v + counts.max() * 0.006, i, f"{v:,} ({v / counts.sum():.1%})", va="center", fontsize=8)
    ax.set_title(f"Genre distribution – training set ({counts.sum():,} movies, {len(counts)} genres)")
    ax.set_xlabel("Number of movies")
    ax.set_ylabel("")
    ax.set_xlim(0, counts.max() * 1.2)
    fig.tight_layout()
    fig.savefig(path, dpi=110)
    plt.close(fig)


def plot_length_distribution(train: pd.DataFrame, test: pd.DataFrame, path: Path) -> None:
    fig, ax = plt.subplots(figsize=(10, 5))
    tr = train["description"].str.split().str.len()
    te = test["description"].str.split().str.len()
    bins = np.arange(0, 401, 10)
    ax.hist(tr.clip(upper=400), bins=bins, alpha=0.6, label=f"train (median {tr.median():.0f})")
    ax.hist(te.clip(upper=400), bins=bins, alpha=0.6, label=f"test (median {te.median():.0f})")
    ax.set_title("Plot description length (words, clipped at 400)")
    ax.set_xlabel("Words per description")
    ax.set_ylabel("Movies")
    ax.legend()
    fig.tight_layout()
    fig.savefig(path, dpi=110)
    plt.close(fig)


def plot_length_by_genre(train: pd.DataFrame, path: Path) -> None:
    data = train.assign(words=train["description"].str.split().str.len())
    order = data.groupby("genre")["words"].median().sort_values(ascending=False).index
    fig, ax = plt.subplots(figsize=(11, 8))
    groups = [data.loc[data["genre"] == g, "words"].values for g in order]
    box = ax.boxplot(groups, orientation="horizontal", showfliers=False, patch_artist=True, widths=0.6)
    for patch in box["boxes"]:
        patch.set_facecolor("#8fbcd4")
    ax.set_yticks(range(1, len(order) + 1), order)
    ax.invert_yaxis()
    ax.set_title("Description length by genre (outliers hidden)")
    ax.set_xlabel("Words per description")
    ax.set_ylabel("")
    fig.tight_layout()
    fig.savefig(path, dpi=110)
    plt.close(fig)


def top_words_by_genre(train: pd.DataFrame, n: int = 10) -> pd.DataFrame:
    """Most frequent words and most *distinctive* words (class-level TF-IDF) per genre."""
    docs = train.groupby("genre")["description"].apply(lambda s: " ".join(s))
    count = CountVectorizer(preprocessor=clean_text, stop_words="english", min_df=1)
    counts = count.fit_transform(docs.values)
    vocab = np.array(count.get_feature_names_out())
    # Distinctive words must appear in >= 50 individual plots, otherwise
    # character names that occur in a single film dominate the ranking.
    min_plots = min(50, max(2, len(train) // 100))
    common = CountVectorizer(preprocessor=clean_text, stop_words="english", min_df=min_plots, binary=True)
    common.fit(train["description"])
    tfidf = TfidfVectorizer(preprocessor=clean_text, vocabulary=common.vocabulary_, sublinear_tf=True)
    weights = tfidf.fit_transform(docs.values)
    tvocab = np.array(tfidf.get_feature_names_out())
    rows = []
    for i, genre in enumerate(docs.index):
        c = counts[i].toarray().ravel()
        w = weights[i].toarray().ravel()
        for rank, (fi, di) in enumerate(zip(np.argsort(c)[::-1][:n], np.argsort(w)[::-1][:n]), 1):
            rows.append(
                {
                    "min_plots_for_distinctive": min_plots,
                    "genre": genre,
                    "rank": rank,
                    "frequent_word": vocab[fi],
                    "frequency": int(c[fi]),
                    "distinctive_word": tvocab[di],
                    "class_tfidf": float(w[di]),
                }
            )
    return pd.DataFrame(rows)


def plot_top_words(words: pd.DataFrame, path: Path, n: int = 8) -> None:
    genres = sorted(words["genre"].unique())
    cols = 4
    rows = int(np.ceil(len(genres) / cols))
    fig, axes = plt.subplots(rows, cols, figsize=(16, rows * 2.4))
    for ax, genre in zip(axes.ravel(), genres):
        d = words[(words["genre"] == genre) & (words["rank"] <= n)].iloc[::-1]
        ax.barh(d["distinctive_word"], d["class_tfidf"], color="#55A868")
        ax.set_title(genre, fontsize=10, fontweight="bold")
        ax.tick_params(axis="y", labelsize=8)
        ax.tick_params(axis="x", labelsize=7)
    for ax in axes.ravel()[len(genres):]:
        ax.axis("off")
    min_plots = int(words["min_plots_for_distinctive"].iloc[0])
    fig.suptitle(
        f"Most distinctive words per genre (class-level TF-IDF over words used in {min_plots}+ plots, training set)",
        fontsize=14,
    )
    fig.tight_layout(rect=(0, 0, 1, 0.98))
    fig.savefig(path, dpi=100)
    plt.close(fig)


def plot_frequent_words(words: pd.DataFrame, train: pd.DataFrame, path: Path, n_genres: int = 8) -> None:
    genres = train["genre"].value_counts().index[:n_genres]
    fig, axes = plt.subplots(2, 4, figsize=(16, 6.5))
    for ax, genre in zip(axes.ravel(), genres):
        d = words[words["genre"] == genre].iloc[::-1]
        ax.barh(d["frequent_word"], d["frequency"], color="#4C72B0")
        ax.set_title(genre, fontsize=10, fontweight="bold")
        ax.tick_params(labelsize=8)
    fig.suptitle(f"Most common words in the {n_genres} largest genres (stop words removed)", fontsize=14)
    fig.tight_layout(rect=(0, 0, 1, 0.96))
    fig.savefig(path, dpi=100)
    plt.close(fig)


def run_eda(train: pd.DataFrame, test: pd.DataFrame, reports_dir: Path, figures_dir: Path) -> dict:
    reports_dir, figures_dir = Path(reports_dir), Path(figures_dir)
    figures_dir.mkdir(parents=True, exist_ok=True)
    summary = dataset_summary(train, test)
    (reports_dir / "eda_summary.json").write_text(json.dumps(summary, indent=2))
    plot_genre_distribution(train, figures_dir / "genre_distribution.png")
    plot_length_distribution(train, test, figures_dir / "description_length_distribution.png")
    plot_length_by_genre(train, figures_dir / "description_length_by_genre.png")
    words = top_words_by_genre(train)
    words.to_csv(reports_dir / "top_words_by_genre.csv", index=False)
    plot_top_words(words, figures_dir / "top_words_by_genre.png")
    plot_frequent_words(words, train, figures_dir / "common_words_by_genre.png")
    return summary
