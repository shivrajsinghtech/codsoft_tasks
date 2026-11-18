import pandas as pd
import pytest

from src.data_preprocessing import clean_dataset
from src.text_cleaning import clean_text


def test_clean_text_placeholders():
    out = clean_text("WIN £500 now! Call 09061701461 or visit www.prize.com")
    assert out == out.lower()
    assert "currencytoken" in out
    assert "phonetoken" in out
    assert "urltoken" in out
    assert "!" not in out


def test_clean_text_handles_non_string():
    assert clean_text(None) == ""
    assert clean_text(123) == "numtoken"


def test_clean_dataset_on_synthetic_frame():
    raw = pd.DataFrame({
        "v1": ["ham", "spam", "ham", "ham", None, "spam"],
        "v2": ["hello there", "WIN cash", "hello there", "part one", "orphan", None],
        "Unnamed: 2": [None, None, None, " part two", None, None],
    })
    df, summary = clean_dataset(raw)
    assert list(df.columns) == ["label", "message", "target"]
    assert summary["duplicate_rows_removed"] == 1
    assert summary["missing_rows_removed"] == 2
    assert summary["dropped_columns"] == ["Unnamed: 2"]
    assert "part one, part two" in df["message"].tolist()
    assert set(df["target"]) == {0, 1}
    assert df.loc[df["label"] == "spam", "target"].eq(1).all()


def test_clean_dataset_requires_columns():
    with pytest.raises(ValueError):
        clean_dataset(pd.DataFrame({"a": [1]}))


def test_real_dataset_is_clean(clean_df):
    assert len(clean_df) > 5000
    assert not clean_df.duplicated(subset=["label", "message"]).any()
    assert clean_df.isna().sum().sum() == 0
    assert set(clean_df["label"]) == {"ham", "spam"}
    assert not any(c.startswith("Unnamed") for c in clean_df.columns)
