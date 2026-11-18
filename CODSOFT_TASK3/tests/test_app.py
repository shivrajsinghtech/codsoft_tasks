from pathlib import Path

from streamlit.testing.v1 import AppTest

APP = str(Path(__file__).resolve().parent.parent / "app.py")


def _app():
    at = AppTest.from_file(APP, default_timeout=60)
    at.run()
    return at


def test_app_starts_without_errors():
    at = _app()
    assert not at.exception
    assert not at.error
    assert "Customer Churn Prediction" in at.title[0].value
    assert len(at.tabs) == 5


def test_valid_prediction_flow():
    at = _app()
    at.selectbox[0].select("Germany")
    at.selectbox[1].select("Female")
    at.button[0].click().run()
    assert not at.exception
    assert not at.error
    shown = " ".join(m.label for m in at.metric)
    assert "Churn probability" in shown and "Prediction confidence" in shown


def test_missing_input_shows_error():
    at = _app()
    # Geography and Gender are left unselected
    at.button[0].click().run()
    assert not at.exception
    assert at.error, "expected a validation error"
    text = " ".join(m.value for m in at.markdown)
    assert "Geography is required" in text and "Gender is required" in text
