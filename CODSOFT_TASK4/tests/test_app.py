"""Streamlit app tests using Streamlit's built-in AppTest harness."""

from pathlib import Path

import pytest
from streamlit.testing.v1 import AppTest

APP = str(Path(__file__).resolve().parent.parent / "app.py")


@pytest.fixture
def app(model):  # `model` fixture ensures the trained model exists
    at = AppTest.from_file(APP, default_timeout=60)
    at.run()
    return at


def _predict_button(at):
    return next(b for b in at.button if b.label == "Predict")


def test_app_starts_without_errors(app):
    assert not app.exception
    assert len(app.error) == 0
    assert app.title[0].value.endswith("Spam SMS Detection")


def test_app_shows_metrics(app):
    labels = [m.label for m in app.metric]
    assert {"Accuracy", "Precision (spam)", "Recall (spam)", "F1-score (spam)"} <= set(labels)


def test_app_empty_input_warns(app):
    _predict_button(app).click().run()
    assert not app.exception
    assert any("empty" in w.value.lower() for w in app.warning)


def test_app_predicts_spam(app):
    app.text_area(key="sms_input").input(
        "URGENT! You have won a £1000 cash prize. Call 09061701461 now to claim!"
    )
    _predict_button(app).click().run()
    assert not app.exception
    html = " ".join(m.value for m in app.markdown)
    assert "SPAM" in html and "Confidence" in html


def test_app_predicts_legitimate(app):
    app.text_area(key="sms_input").input("Are we still on for dinner tonight at 7?")
    _predict_button(app).click().run()
    assert not app.exception
    html = " ".join(m.value for m in app.markdown)
    assert "LEGITIMATE" in html


def test_app_sample_button_loads_text(app):
    sample = next(b for b in app.button if b.key == "spam_0")
    sample.click().run()
    assert "WINNER" in app.text_area(key="sms_input").value
