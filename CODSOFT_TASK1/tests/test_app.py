"""Streamlit startup / interaction tests using streamlit.testing.AppTest."""

from pathlib import Path

import pytest
from streamlit.testing.v1 import AppTest

import src.config

APP = str(Path(__file__).resolve().parent.parent / "app.py")


@pytest.fixture
def app(monkeypatch, tiny_model_path):
    monkeypatch.setattr(src.config, "MODEL_PATH", tiny_model_path)
    at = AppTest.from_file(APP, default_timeout=60)
    at.run()
    return at


def test_app_starts_without_errors(app):
    assert not app.exception
    assert app.title[0].value == "Movie Genre Classifier"
    assert app.button[0].label == "Predict Genre"


def test_app_rejects_empty_plot(app):
    app.button[0].click().run()
    assert not app.exception
    assert any("enter a plot summary" in e.value for e in app.error)


def test_app_rejects_too_short_plot(app):
    app.text_area(key="plot_input").input("ghost")
    app.button[0].click().run()
    assert any("too short" in e.value for e in app.error)


def test_app_predicts(app):
    app.text_input(key="title_input").input("Night Curse")
    app.text_area(key="plot_input").input("A haunted ghost and a demon scream at night in the graveyard.")
    app.button[0].click().run()
    assert not app.exception and not app.error
    assert len(app.dataframe) == 1  # top-3 table
    assert app.dataframe[0].value.shape[0] == 3


def test_app_example_loader(app):
    app.selectbox(key="example").select("Wedding chaos (comedy)").run()
    assert "best man" in app.text_area(key="plot_input").value


@pytest.mark.parametrize("page", ["📊 Model Performance", "📈 Dataset Insights", "ℹ️ About"])
def test_other_pages_render(app, page):
    app.radio(key="page").set_value(page).run()
    assert not app.exception


def test_app_handles_missing_model(monkeypatch, tmp_path):
    monkeypatch.setattr(src.config, "MODEL_PATH", tmp_path / "missing.joblib")
    at = AppTest.from_file(APP, default_timeout=60)
    at.run()
    assert not at.exception
    assert any("not available" in e.value for e in at.error)
