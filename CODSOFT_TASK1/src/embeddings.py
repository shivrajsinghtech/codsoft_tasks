"""Word-embedding features: a scikit-learn transformer that trains Word2Vec on
the training texts and represents each document as the mean of its word vectors.

Used for the supplementary "Word2Vec + Logistic Regression" experiment. Because
Word2Vec is trained inside ``fit``, it only ever sees the data the pipeline is
fitted on (no leakage from validation or test texts).
"""

from __future__ import annotations

import numpy as np
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.feature_extraction.text import ENGLISH_STOP_WORDS

from src.preprocessing import clean_text


def tokenize(text: object) -> list[str]:
    return [t for t in clean_text(text).split() if t not in ENGLISH_STOP_WORDS and len(t) > 1]


class Word2VecVectorizer(BaseEstimator, TransformerMixin):
    def __init__(
        self,
        vector_size: int = 100,
        window: int = 5,
        min_count: int = 3,
        epochs: int = 10,
        workers: int = 4,
        seed: int = 42,
    ) -> None:
        self.vector_size = vector_size
        self.window = window
        self.min_count = min_count
        self.epochs = epochs
        self.workers = workers
        self.seed = seed

    def fit(self, X, y=None):
        from gensim.models import Word2Vec  # imported lazily: optional dependency

        sentences = [tokenize(x) for x in X]
        model = Word2Vec(
            sentences=sentences,
            vector_size=self.vector_size,
            window=self.window,
            min_count=self.min_count,
            epochs=self.epochs,
            workers=self.workers,
            seed=self.seed,
        )
        self.wv_ = model.wv  # keep only the vectors – smaller to pickle
        return self

    def transform(self, X) -> np.ndarray:
        out = np.zeros((len(X), self.vector_size), dtype=np.float32)
        for i, text in enumerate(X):
            vectors = [self.wv_[t] for t in tokenize(text) if t in self.wv_.key_to_index]
            if vectors:
                out[i] = np.mean(vectors, axis=0)
        return out
