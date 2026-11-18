"""Text cleaning used by every model.

``clean_text`` is passed to ``TfidfVectorizer(preprocessor=...)`` so the exact
same cleaning runs during training and inference and is stored inside the
saved pipeline. It is deliberately light: lower-casing, accent folding, URL
removal and punctuation-to-space. Numbers are kept (years and ages carry
signal, e.g. for documentaries/history). Stop words are removed later by the
vectorizer; no stemming, so word forms keep their meaning for bigrams.
"""

from __future__ import annotations

import math
import re
import unicodedata

_URL_RE = re.compile(r"(https?://\S+|www\.\S+)")
_NON_ALNUM_RE = re.compile(r"[^a-z0-9\s]")
_WS_RE = re.compile(r"\s+")
# "(2009)", "(????)", "(2009/I)" – release-year markers in titles
_YEAR_RE = re.compile(r"\(\s*(?:\d{4}|\?{4})(?:/[IVXLC]+)?\s*\)")


def _to_str(text: object) -> str:
    if text is None:
        return ""
    if isinstance(text, float) and math.isnan(text):
        return ""
    return text if isinstance(text, str) else str(text)


def clean_text(text: object) -> str:
    """Normalise raw text for TF-IDF / embedding tokenisation."""
    text = _to_str(text)
    text = unicodedata.normalize("NFKD", text).encode("ascii", "ignore").decode("ascii")
    text = text.lower()
    text = _URL_RE.sub(" ", text)
    text = _NON_ALNUM_RE.sub(" ", text)
    return _WS_RE.sub(" ", text).strip()


def clean_title(title: object) -> str:
    """Remove the year marker and quotes from a dataset title.

    ``'"Quattro piccole donne" (1989)'`` -> ``'Quattro piccole donne'``
    """
    title = _YEAR_RE.sub(" ", _to_str(title)).replace('"', " ")
    return _WS_RE.sub(" ", title).strip()


def build_text(title: object, description: object) -> str:
    """Model input: cleaned title followed by the plot description."""
    parts = [clean_title(title), _to_str(description).strip()]
    return " ".join(p for p in parts if p)
