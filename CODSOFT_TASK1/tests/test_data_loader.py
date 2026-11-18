import pandas as pd
import pytest

from src.config import DATA_DIR
from src.data_loader import load_dataset, load_test_data, load_test_solution, load_train_data, parse_line


def test_parse_line_train_format():
    parts = parse_line("7 ::: Cupid (1997) ::: thriller ::: A brother and sister... ok.\n", 4)
    assert parts == ["7", "Cupid (1997)", "thriller", "A brother and sister... ok."]


def test_parse_line_preserves_separator_and_punctuation_in_description():
    line = "3 ::: T (2001) ::: drama ::: He said: \"wait ::: now!\" — 50% off; done?"
    assert parse_line(line, 4)[3] == "He said: \"wait ::: now!\" — 50% off; done?"


@pytest.mark.parametrize("line", ["", "   \n", "1 ::: only title", "abc ::: t ::: g ::: d"])
def test_parse_line_rejects_malformed(line):
    with pytest.raises(ValueError):
        parse_line(line, 4)


def test_load_dataset_reports_malformed_and_blank_lines(tmp_path):
    f = tmp_path / "train.txt"
    f.write_text(
        "1 ::: A (2000) ::: Drama ::: Good plot.\n"
        "\n"
        "broken line without separators\n"
        "2 ::: B (2001) :::  COMEDY  ::: \n",
        encoding="utf-8",
    )
    df, report = load_dataset(f, "train")
    assert list(df.columns) == ["id", "title", "genre", "description"]
    assert len(df) == 2 and report.parsed == 2
    assert report.blank_lines == 1
    assert len(report.malformed) == 1 and report.malformed[0][0] == 3
    assert report.empty_descriptions == 1
    assert df["genre"].tolist() == ["drama", "comedy"]  # normalised
    assert pd.api.types.is_integer_dtype(df["id"])


def test_load_dataset_errors(tmp_path):
    with pytest.raises(FileNotFoundError):
        load_dataset(tmp_path / "missing.txt", "train")
    with pytest.raises(ValueError):
        load_dataset(tmp_path / "missing.txt", "unknown")


def test_load_tiny_dataset(tiny_data_dir):
    train, _ = load_train_data(tiny_data_dir)
    test, _ = load_test_data(tiny_data_dir)
    sol, _ = load_test_solution(tiny_data_dir)
    assert len(train) == 36 and train["genre"].nunique() == 3
    assert "genre" not in test.columns
    assert (test["id"].values == sol["id"].values).all()


real = pytest.mark.skipif(not (DATA_DIR / "train_data.txt").exists(), reason="real dataset not present")


@real
def test_real_dataset_record_counts():
    train, tr = load_train_data()
    test, te = load_test_data()
    sol, so = load_test_solution()
    assert (len(train), len(test), len(sol)) == (54214, 54200, 54200)
    assert not tr.malformed and not te.malformed and not so.malformed
    assert train["genre"].nunique() == 27
    assert set(sol["genre"]) <= set(train["genre"])
    assert (test["id"].values == sol["id"].values).all()
