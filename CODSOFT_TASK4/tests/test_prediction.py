import pytest

from src.predict import InvalidInputError, load_model, predict_message

SPAM = [
    "WINNER!! You have been selected to receive a £900 prize reward! To claim call 09061701461. Claim code KL341.",
    "URGENT! Your mobile number has been awarded a £2000 bonus caller prize. Call 09058094455 now!",
    "Free entry in 2 a wkly comp to win FA Cup final tkts. Text FA to 87121 to receive entry question",
]
HAM = [
    "Hey, are we still meeting for lunch at 1pm tomorrow?",
    "I'll be home late tonight, can you please pick up some milk on the way?",
    "Happy birthday! Hope you have a wonderful day. See you at dinner.",
]


def test_model_loads(model):
    assert hasattr(model, "predict_proba")
    assert list(model.classes_) == [0, 1]
    assert "tfidf" in model.named_steps


def test_load_model_missing_file(tmp_path):
    with pytest.raises(FileNotFoundError):
        load_model(tmp_path / "nope.joblib")


@pytest.mark.parametrize("msg", SPAM)
def test_spam_prediction(model, msg):
    result = predict_message(msg, model=model)
    assert result["label"] == "SPAM"
    assert result["is_spam"] is True
    assert 0.5 <= result["confidence"] <= 1.0


@pytest.mark.parametrize("msg", HAM)
def test_legitimate_prediction(model, msg):
    result = predict_message(msg, model=model)
    assert result["label"] == "LEGITIMATE"
    assert result["is_spam"] is False
    assert 0.5 <= result["confidence"] <= 1.0


@pytest.mark.parametrize("msg", ["", "   ", "\n\t", None])
def test_empty_input_rejected(model, msg):
    with pytest.raises(InvalidInputError, match="empty"):
        predict_message(msg, model=model)


@pytest.mark.parametrize("msg", [123, 4.5, ["hello"], {"text": "hi"}, b"bytes"])
def test_non_text_input_rejected(model, msg):
    with pytest.raises(InvalidInputError, match="must be text"):
        predict_message(msg, model=model)


def test_symbols_only_rejected(model):
    with pytest.raises(InvalidInputError, match="no letters or digits"):
        predict_message("!!! ??? ...", model=model)


def test_too_long_rejected(model):
    with pytest.raises(InvalidInputError, match="too long"):
        predict_message("a" * 6000, model=model)


def test_unicode_input_is_handled(model):
    result = predict_message("Café ☕ at 5? 😊", model=model)
    assert result["label"] in {"SPAM", "LEGITIMATE"}


def test_default_model_used_when_none_passed():
    result = predict_message("See you at the station at 6")
    assert result["label"] in {"SPAM", "LEGITIMATE"}
