"""Dataset loading and cleaning."""
from __future__ import annotations

from pathlib import Path

import pandas as pd

from .config import DATA_PATH, EXPECTED_COLUMNS, FEATURES, ID_COLUMNS, TARGET


def load_raw_data(path: str | Path = DATA_PATH) -> pd.DataFrame:
    """Load the raw CSV and check that every expected column is present."""
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(f"Dataset not found at {path}")
    df = pd.read_csv(path)
    missing = set(EXPECTED_COLUMNS) - set(df.columns)
    if missing:
        raise ValueError(f"Dataset is missing columns: {sorted(missing)}")
    return df


def clean_data(df: pd.DataFrame) -> pd.DataFrame:
    """Drop identifier columns and exact duplicate rows.

    Missing values are *not* imputed here: the source data has none, and any
    imputation needed at prediction time happens inside the model pipeline so
    that it is fitted on training data only (no leakage).
    """
    out = df.drop(columns=[c for c in ID_COLUMNS if c in df.columns])
    out = out.drop_duplicates().reset_index(drop=True)
    return out


def split_features_target(df: pd.DataFrame) -> tuple[pd.DataFrame, pd.Series]:
    """Return (X, y) using the configured feature list."""
    return df[FEATURES].copy(), df[TARGET].astype(int).copy()


def data_quality_report(df: pd.DataFrame) -> dict:
    """Summary of shape, missing values, duplicates and class balance."""
    return {
        "rows": int(df.shape[0]),
        "columns": int(df.shape[1]),
        "missing_values": {k: int(v) for k, v in df.isna().sum().items()},
        "total_missing": int(df.isna().sum().sum()),
        "duplicate_rows": int(df.duplicated().sum()),
        "churn_rate": float(df[TARGET].mean()) if TARGET in df else None,
        "class_counts": {str(k): int(v) for k, v in df[TARGET].value_counts().items()}
        if TARGET in df
        else None,
    }
