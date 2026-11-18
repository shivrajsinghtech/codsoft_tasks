import math

import pytest

from src.preprocessing import build_text, clean_text, clean_title


def test_clean_text_lowercases_and_strips_punctuation():
    assert clean_text("Hello, WORLD!! It's 10-year-old Oscar.") == "hello world it s 10 year old oscar"


def test_clean_text_folds_accents_and_removes_urls():
    assert clean_text("Café — visit https://example.com now") == "cafe visit now"


@pytest.mark.parametrize("value", [None, math.nan, "", "   ", "!!!"])
def test_clean_text_empty_like_inputs(value):
    assert clean_text(value) == ""


def test_clean_text_non_string():
    assert clean_text(1999) == "1999"


@pytest.mark.parametrize(
    "title,expected",
    [
        ("Oscar et la dame rose (2009)", "Oscar et la dame rose"),
        ('"Quattro piccole donne" (1989)', "Quattro piccole donne"),
        ("Unknown (????)", "Unknown"),
        ("Remake (2009/II)", "Remake"),
    ],
)
def test_clean_title(title, expected):
    assert clean_title(title) == expected


def test_build_text_combines_title_and_description():
    assert build_text("Red Horizon (2140)", " Mars colony. ") == "Red Horizon Mars colony."
    assert build_text("", "Only plot") == "Only plot"
    assert build_text(None, None) == ""
