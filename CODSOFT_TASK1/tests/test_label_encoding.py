import pytest

from src.label_encoding import GenreLabelEncoder, normalize_genre


def test_normalize_genre():
    assert normalize_genre("  Sci-Fi ") == "sci-fi"
    for bad in ["", "   ", None, 5]:
        with pytest.raises(ValueError):
            normalize_genre(bad)


def test_encoder_round_trip():
    enc = GenreLabelEncoder().fit(["drama", "comedy", "Drama", "war"])
    assert list(enc.classes_) == ["comedy", "drama", "war"]
    codes = enc.transform(["war", "comedy"])
    assert codes.tolist() == [2, 0]
    assert enc.inverse_transform(codes).tolist() == ["war", "comedy"]
    assert len(enc) == 3


def test_encoder_unknown_and_unfitted():
    with pytest.raises(RuntimeError):
        _ = GenreLabelEncoder().classes_
    enc = GenreLabelEncoder().fit(["drama"])
    with pytest.raises(ValueError, match="Unknown genre"):
        enc.transform(["horror"])
