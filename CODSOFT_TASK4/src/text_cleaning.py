"""Text normalisation applied to every SMS before TF-IDF vectorisation.

The function is used as the ``preprocessor`` of the TfidfVectorizer, so it is
stored inside the saved pipeline and applied identically at training and
prediction time.
"""

import re

_URL_RE = re.compile(r"(https?://\S+|www\.\S+|\b\S+\.(com|co\.uk|net|org|biz)\S*)")
_EMAIL_RE = re.compile(r"\b[\w.+-]+@[\w-]+\.[\w.-]+\b")
_PHONE_RE = re.compile(r"\b\d{5,}\b")
_CURRENCY_RE = re.compile(r"[£$€]|\bgbp\b|\busd\b")
_NUMBER_RE = re.compile(r"\b\d+\b")
_NON_ALNUM_RE = re.compile(r"[^a-z0-9\s]")
_WHITESPACE_RE = re.compile(r"\s+")


def clean_text(text: str) -> str:
    """Lower-case the SMS and replace noisy tokens with generic placeholders.

    URLs, e-mail addresses, long digit sequences (phone / short codes),
    currency symbols and other numbers are strong spam signals in their
    *presence*, not their exact value, so they are mapped to single tokens.
    """
    if not isinstance(text, str):
        text = "" if text is None else str(text)

    text = text.lower()
    text = _URL_RE.sub(" urltoken ", text)
    text = _EMAIL_RE.sub(" emailtoken ", text)
    text = _CURRENCY_RE.sub(" currencytoken ", text)
    text = _PHONE_RE.sub(" phonetoken ", text)
    text = _NUMBER_RE.sub(" numtoken ", text)
    text = _NON_ALNUM_RE.sub(" ", text)
    return _WHITESPACE_RE.sub(" ", text).strip()
