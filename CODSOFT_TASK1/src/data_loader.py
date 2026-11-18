"""Loading and parsing of the ``:::``-separated Genre Classification Dataset.

Formats
-------
train / test solution : ``ID ::: TITLE ::: GENRE ::: DESCRIPTION``
test                  : ``ID ::: TITLE ::: DESCRIPTION``

The parser splits on the separator at most ``n_fields - 1`` times, so any
``:::`` or other punctuation that appears *inside* a description is preserved.
Malformed lines (too few fields, non-numeric ID) are recorded in a
:class:`ParseReport` instead of silently disappearing; no records are invented.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path

import pandas as pd

from src.config import (
    DATA_DIR,
    SEPARATOR,
    TEST_FILE,
    TEST_SOLUTION_FILE,
    TRAIN_FILE,
)
from src.label_encoding import normalize_genre

COLUMNS = {
    "train": ["id", "title", "genre", "description"],
    "solution": ["id", "title", "genre", "description"],
    "test": ["id", "title", "description"],
}


@dataclass
class ParseReport:
    """Summary of what happened while parsing one file."""

    path: str
    kind: str
    total_lines: int = 0
    parsed: int = 0
    blank_lines: int = 0
    empty_descriptions: int = 0
    malformed: list[tuple[int, str]] = field(default_factory=list)

    def as_dict(self) -> dict:
        return {
            "file": Path(self.path).name,
            "kind": self.kind,
            "total_lines": self.total_lines,
            "parsed_records": self.parsed,
            "blank_lines": self.blank_lines,
            "empty_descriptions": self.empty_descriptions,
            "malformed_lines": len(self.malformed),
            "malformed_examples": self.malformed[:10],
        }


def parse_line(line: str, n_fields: int) -> list[str]:
    """Split one raw line into exactly ``n_fields`` stripped fields.

    Raises ``ValueError`` if the line is blank or has too few fields.
    """
    if not isinstance(line, str):
        raise ValueError("line must be a string")
    line = line.rstrip("\r\n")
    if not line.strip():
        raise ValueError("blank line")
    parts = [p.strip() for p in line.split(SEPARATOR, n_fields - 1)]
    if len(parts) < n_fields:
        raise ValueError(f"expected {n_fields} fields, found {len(parts)}")
    if not parts[0].isdigit():
        raise ValueError(f"non-numeric id {parts[0]!r}")
    return parts


def load_dataset(path: str | Path, kind: str) -> tuple[pd.DataFrame, ParseReport]:
    """Parse a dataset file of the given ``kind`` (train | test | solution)."""
    if kind not in COLUMNS:
        raise ValueError(f"kind must be one of {sorted(COLUMNS)}")
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(f"Dataset file not found: {path}")

    columns = COLUMNS[kind]
    report = ParseReport(path=str(path), kind=kind)
    rows: list[list[str]] = []
    with path.open("r", encoding="utf-8", errors="replace") as handle:
        for line_no, line in enumerate(handle, start=1):
            report.total_lines += 1
            if not line.strip():
                report.blank_lines += 1
                continue
            try:
                parts = parse_line(line, len(columns))
            except ValueError as exc:
                report.malformed.append((line_no, str(exc)))
                continue
            rows.append(parts)

    df = pd.DataFrame(rows, columns=columns)
    df["id"] = df["id"].astype(int)
    if "genre" in df:
        df["genre"] = df["genre"].map(normalize_genre)
    report.parsed = len(df)
    report.empty_descriptions = int((df["description"].str.strip() == "").sum())
    return df, report


def load_train_data(data_dir: str | Path = DATA_DIR) -> tuple[pd.DataFrame, ParseReport]:
    return load_dataset(Path(data_dir) / TRAIN_FILE, "train")


def load_test_data(data_dir: str | Path = DATA_DIR) -> tuple[pd.DataFrame, ParseReport]:
    """Unlabelled test set – used only for generating predictions."""
    return load_dataset(Path(data_dir) / TEST_FILE, "test")


def load_test_solution(data_dir: str | Path = DATA_DIR) -> tuple[pd.DataFrame, ParseReport]:
    """Official held-out labels. Call ONLY for final evaluation after model selection."""
    return load_dataset(Path(data_dir) / TEST_SOLUTION_FILE, "solution")
