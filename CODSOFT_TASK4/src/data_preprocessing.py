"""Loading and cleaning of the raw SMS Spam Collection CSV."""

from pathlib import Path

import pandas as pd

from src.config import LABEL_MAP, RAW_DATA_PATH


def load_raw_data(path: Path = RAW_DATA_PATH) -> pd.DataFrame:
    """Read the raw CSV. The file is not UTF-8, so latin-1 is used."""
    if not Path(path).exists():
        raise FileNotFoundError(
            f"Dataset not found at {path}. Extract the provided ZIP into data/."
        )
    return pd.read_csv(path, encoding="latin-1")


def clean_dataset(raw: pd.DataFrame) -> tuple[pd.DataFrame, dict]:
    """Return a clean ``[label, message, target]`` frame and a cleaning summary.

    Steps:
    1. The ``Unnamed: 2..4`` columns contain the tail of messages that held
       extra commas (a CSV quoting artefact). That text is appended back to
       ``v2`` so nothing is lost, then the columns are dropped.
    2. Rows with a missing label or message are dropped.
    3. Exact duplicate (label, message) rows are dropped.
    4. ``ham`` -> 0 (LEGITIMATE), ``spam`` -> 1 (SPAM).
    """
    if not {"v1", "v2"}.issubset(raw.columns):
        raise ValueError("Expected columns 'v1' (label) and 'v2' (message).")

    df = raw.copy()
    summary = {"raw_rows": int(len(df)), "raw_columns": list(map(str, df.columns))}

    extra_cols = [c for c in df.columns if str(c).startswith("Unnamed")]
    summary["rows_with_overflow_text"] = int(df[extra_cols].notna().any(axis=1).sum()) if extra_cols else 0
    if extra_cols:
        overflow = df[extra_cols].apply(
            lambda row: ",".join(str(v) for v in row if pd.notna(v)), axis=1
        )
        has_overflow = overflow.str.len() > 0
        df.loc[has_overflow, "v2"] = df.loc[has_overflow, "v2"].astype(str) + "," + overflow[has_overflow]
        df = df.drop(columns=extra_cols)
    summary["dropped_columns"] = list(map(str, extra_cols))

    df = df.rename(columns={"v1": "label", "v2": "message"})
    df["label"] = df["label"].astype("string").str.strip().str.lower()
    df["message"] = df["message"].astype("string").str.strip()

    missing_mask = df["label"].isna() | df["message"].isna() | (df["message"] == "")
    summary["missing_rows_removed"] = int(missing_mask.sum())
    df = df[~missing_mask]

    invalid_label = ~df["label"].isin(LABEL_MAP.keys())
    summary["invalid_label_rows_removed"] = int(invalid_label.sum())
    df = df[~invalid_label]

    dup_mask = df.duplicated(subset=["label", "message"], keep="first")
    summary["duplicate_rows_removed"] = int(dup_mask.sum())
    df = df[~dup_mask]

    df["target"] = df["label"].map(LABEL_MAP).astype(int)
    df = df.reset_index(drop=True)[["label", "message", "target"]]
    df["label"] = df["label"].astype(str)
    df["message"] = df["message"].astype(str)

    summary["clean_rows"] = int(len(df))
    summary["class_counts"] = {k: int(v) for k, v in df["label"].value_counts().items()}
    return df, summary


def load_clean_data(path: Path = RAW_DATA_PATH) -> pd.DataFrame:
    """Convenience wrapper: load and clean in one call."""
    df, _ = clean_dataset(load_raw_data(path))
    return df
