"""Shared fixtures.

The tiny corpus below is a *synthetic test fixture* used only to exercise the
code paths quickly; the real model is trained on the provided dataset.
"""

from __future__ import annotations

import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from src.models import build_pipeline  # noqa: E402
from src.predictor import save_model  # noqa: E402

VOCAB = {
    "horror": "ghost haunted demon blood scream terror night curse zombie killer graveyard",
    "comedy": "funny hilarious jokes wedding party laugh prank silly awkward friends mishap",
    "documentary": "interviews footage history scientists explores real archival experts facts nature",
}
TINY_GENRES = sorted(VOCAB)


def make_corpus(n_per_genre: int = 12):
    rows = []
    idx = 1
    for genre, words in VOCAB.items():
        tokens = words.split()
        for i in range(n_per_genre):
            rot = tokens[i % len(tokens):] + tokens[: i % len(tokens)]
            rows.append((idx, f"{genre.title()} Film {i} ({2000 + i})", genre,
                         f"A story where {' '.join(rot[:7])}, and more: {rot[7]}!"))
            idx += 1
    return rows


@pytest.fixture(scope="session")
def tiny_data_dir(tmp_path_factory) -> Path:
    d = tmp_path_factory.mktemp("dataset")
    rows = make_corpus()
    test_rows = make_corpus(4)
    with open(d / "train_data.txt", "w", encoding="utf-8") as f:
        for i, t, g, desc in rows:
            f.write(f"{i} ::: {t} ::: {g} ::: {desc}\n")
    with open(d / "test_data.txt", "w", encoding="utf-8") as f, \
         open(d / "test_data_solution.txt", "w", encoding="utf-8") as s:
        for i, t, g, desc in test_rows:
            f.write(f"{i} ::: {t} ::: {desc}\n")
            s.write(f"{i} ::: {t} ::: {g} ::: {desc}\n")
    (d / "description.txt").write_text("Train data:\nID ::: TITLE ::: GENRE ::: DESCRIPTION\n")
    return d


TINY_VEC = {"min_df": 1, "max_df": 1.0}


@pytest.fixture(scope="session")
def tiny_texts_labels():
    from src.preprocessing import build_text

    rows = make_corpus()
    return [build_text(t, d) for _, t, _, d in rows], [g for _, _, g, _ in rows]


@pytest.fixture(scope="session")
def tiny_model_path(tmp_path_factory, tiny_texts_labels) -> Path:
    X, y = tiny_texts_labels
    pipe = build_pipeline("logistic_regression", vectorizer_params=TINY_VEC, C=10.0)
    pipe.fit(X, y)
    path = tmp_path_factory.mktemp("models") / "genre_classifier.joblib"
    save_model(pipe, path, {"model_name": "tiny test model", "n_training_records": len(X)})
    return path
