"""Genre label normalisation and integer encoding."""

from __future__ import annotations

from collections.abc import Iterable

import numpy as np
from sklearn.preprocessing import LabelEncoder


def normalize_genre(genre: object) -> str:
    """Canonical genre string: stripped and lower-cased (e.g. ``' Sci-Fi '`` -> ``'sci-fi'``)."""
    if genre is None or not isinstance(genre, str):
        raise ValueError(f"genre must be a non-empty string, got {genre!r}")
    value = genre.strip().lower()
    if not value:
        raise ValueError("genre must be a non-empty string")
    return value


class GenreLabelEncoder:
    """Thin wrapper around :class:`~sklearn.preprocessing.LabelEncoder` with
    normalisation and a clear error for genres never seen during fitting."""

    def __init__(self) -> None:
        self._encoder = LabelEncoder()
        self.fitted = False

    @property
    def classes_(self) -> np.ndarray:
        if not self.fitted:
            raise RuntimeError("GenreLabelEncoder is not fitted")
        return self._encoder.classes_

    def fit(self, genres: Iterable[str]) -> GenreLabelEncoder:
        self._encoder.fit([normalize_genre(g) for g in genres])
        self.fitted = True
        return self

    def transform(self, genres: Iterable[str]) -> np.ndarray:
        values = [normalize_genre(g) for g in genres]
        unknown = sorted(set(values) - set(self.classes_))
        if unknown:
            raise ValueError(f"Unknown genre(s): {unknown}")
        return self._encoder.transform(values)

    def fit_transform(self, genres: Iterable[str]) -> np.ndarray:
        genres = list(genres)
        return self.fit(genres).transform(genres)

    def inverse_transform(self, codes: Iterable[int]) -> np.ndarray:
        if not self.fitted:
            raise RuntimeError("GenreLabelEncoder is not fitted")
        return self._encoder.inverse_transform(np.asarray(list(codes), dtype=int))

    def __len__(self) -> int:
        return len(self.classes_)
