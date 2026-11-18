import pandas as pd
import pytest

from src.config import FEATURES, ID_COLUMNS, TARGET
from src.data import clean_data, data_quality_report, load_raw_data, split_features_target


def test_dataset_loads_with_expected_shape(raw_df):
    assert raw_df.shape == (10000, 14)
    assert TARGET in raw_df.columns


def test_missing_file_raises(tmp_path):
    with pytest.raises(FileNotFoundError):
        load_raw_data(tmp_path / "nope.csv")


def test_missing_columns_raise(tmp_path):
    p = tmp_path / "bad.csv"
    pd.DataFrame({"a": [1]}).to_csv(p, index=False)
    with pytest.raises(ValueError, match="missing columns"):
        load_raw_data(p)


def test_clean_drops_identifiers(clean_df):
    for col in ID_COLUMNS:
        assert col not in clean_df.columns
    assert clean_df.shape == (10000, 11)


def test_no_missing_or_duplicates(clean_df):
    q = data_quality_report(clean_df)
    assert q["total_missing"] == 0
    assert q["duplicate_rows"] == 0
    assert q["class_counts"] == {"0": 7963, "1": 2037}


def test_clean_removes_duplicates(raw_df):
    doubled = pd.concat([raw_df.head(5), raw_df.head(5)])
    assert len(clean_data(doubled)) == 5


def test_split_features_target(xy):
    X, y = xy
    assert list(X.columns) == FEATURES
    assert set(y.unique()) == {0, 1}
