import json

import pytest

from src.config import MODEL_PATH
from src.predictor import (
    GenrePredictor,
    InvalidInputError,
    load_model,
    metadata_path,
    save_model,
    validate_plot,
)
from tests.conftest import TINY_GENRES

HORROR = "A haunted house where a ghost and a demon scream at night in the graveyard."


@pytest.fixture(scope="module")
def predictor(tiny_model_path):
    return GenrePredictor.load(tiny_model_path)


def test_predict_returns_genre_and_confidence(predictor):
    result = predictor.predict(HORROR, title="Night Curse")
    assert result["genre"] == "horror"
    assert 0 < result["confidence"] <= 1
    assert result["genre"] == result["top_k"][0]["genre"]


def test_top3_sorted_and_valid(predictor):
    top = predictor.predict(HORROR, top_k=3)["top_k"]
    assert len(top) == 3
    scores = [t["score"] for t in top]
    assert scores == sorted(scores, reverse=True)
    assert {t["genre"] for t in top} == set(TINY_GENRES)
    assert sum(scores) == pytest.approx(1.0, abs=1e-6)


def test_top_k_is_capped_at_number_of_classes(predictor):
    assert len(predictor.predict(HORROR, top_k=10)["top_k"]) == 3


@pytest.mark.parametrize("plot", ["", "   ", "\n\t"])
def test_empty_input_rejected(predictor, plot):
    with pytest.raises(InvalidInputError, match="empty"):
        predictor.predict(plot)


@pytest.mark.parametrize("plot", [None, 123, ["ghost"], {"plot": "x"}])
def test_non_text_input_rejected(predictor, plot):
    with pytest.raises(InvalidInputError, match="text"):
        predictor.predict(plot)


@pytest.mark.parametrize("plot", ["12345 678", "!!! ??? ...", "ghost"])
def test_invalid_or_too_short_input_rejected(plot):
    with pytest.raises(InvalidInputError):
        validate_plot(plot)


def test_invalid_title_and_top_k(predictor):
    with pytest.raises(InvalidInputError):
        predictor.predict(HORROR, title=42)
    with pytest.raises(InvalidInputError):
        predictor.predict(HORROR, top_k=0)


def test_batch_prediction(predictor):
    out = predictor.predict_batch(["A", "B"], [HORROR, "funny wedding party with silly jokes and laugh"])
    assert [o["genre"] for o in out] == ["horror", "comedy"]


def test_save_and_load_round_trip(tmp_path, tiny_model_path):
    pipe = load_model(tiny_model_path)
    target = tmp_path / "sub" / "model.joblib"
    save_model(pipe, target, {"model_name": "copy"})
    assert target.exists()
    assert json.loads(metadata_path(target).read_text())["model_name"] == "copy"
    reloaded = GenrePredictor.load(target)
    assert reloaded.metadata["model_name"] == "copy"
    assert reloaded.predict(HORROR)["genre"] == GenrePredictor(pipe).predict(HORROR)["genre"]


def test_missing_model_raises(tmp_path):
    with pytest.raises(FileNotFoundError, match="train.py"):
        load_model(tmp_path / "nope.joblib")


@pytest.mark.skipif(not MODEL_PATH.exists(), reason="trained model not present")
def test_real_saved_model():
    p = GenrePredictor.load(MODEL_PATH)
    assert len(p.classes) == 27
    result = p.predict(
        "Rustlers attack the ranchers, and the marshal rides into the saloon with his revolver "
        "to face the outlaw gang at the cattle town.",
        title="Dust and Iron",
    )
    assert result["genre"] == "western"
    assert len(result["top_k"]) == 3
