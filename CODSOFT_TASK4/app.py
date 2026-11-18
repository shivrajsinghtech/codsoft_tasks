"""Streamlit web app for Spam SMS Detection.

Run with:
    streamlit run app.py
"""

import json

import pandas as pd
import streamlit as st

from src.config import FIGURES_DIR, METRICS_PATH, MODEL_PATH
from src.predict import InvalidInputError, load_model, predict_message

st.set_page_config(page_title="Spam SMS Detection", page_icon="📩", layout="wide")

SAMPLE_SPAM = [
    "WINNER!! As a valued network customer you have been selected to receive a £900 prize reward! "
    "To claim call 09061701461. Claim code KL341. Valid 12 hours only.",
    "Free entry in 2 a wkly comp to win FA Cup final tkts 21st May 2005. Text FA to 87121 to receive entry question",
    "URGENT! Your mobile number has been awarded a £2000 bonus caller prize. Call 09058094455 from a landline now!",
    "Congratulations! You've won a free holiday to Spain. Reply YES to 80082 to claim. T&Cs apply. Stop? txt STOP",
]
SAMPLE_HAM = [
    "Hey, are we still meeting for lunch at 1pm tomorrow?",
    "I'll be home late tonight, can you please pick up some milk on the way?",
    "Ok lar... Joking wif u oni...",
    "Happy birthday! Hope you have a wonderful day. See you at dinner.",
]

st.markdown(
    """
    <style>
      .result-card {padding: 1.1rem 1.3rem; border-radius: 12px; margin-top: .6rem;
                    border: 1px solid rgba(128,128,128,.25);}
      .result-spam {background: rgba(228,87,46,.12); border-left: 6px solid #E4572E;}
      .result-ham  {background: rgba(46,160,67,.12); border-left: 6px solid #2EA043;}
      .result-label {font-size: 1.6rem; font-weight: 700; margin: 0;}
      .result-sub {opacity: .8; margin: .2rem 0 0 0;}
    </style>
    """,
    unsafe_allow_html=True,
)


@st.cache_resource(show_spinner=False)
def get_model():
    return load_model(MODEL_PATH)


@st.cache_data(show_spinner=False)
def get_metrics():
    if not METRICS_PATH.exists():
        return None
    return json.loads(METRICS_PATH.read_text(encoding="utf-8"))


def set_sample(text: str) -> None:
    st.session_state["sms_input"] = text


def render_result(result: dict) -> None:
    is_spam = result["is_spam"]
    css = "result-spam" if is_spam else "result-ham"
    icon = "🚫" if is_spam else "✅"
    st.markdown(
        f"""<div class="result-card {css}">
              <p class="result-label">{icon} {result['label']}</p>
              <p class="result-sub">Confidence: <b>{result['confidence']:.1%}</b>
              &nbsp;·&nbsp; Spam probability: {result['spam_probability']:.1%}</p>
            </div>""",
        unsafe_allow_html=True,
    )
    st.progress(min(max(result["confidence"], 0.0), 1.0))


# ---------------------------------------------------------------- header
st.title("📩 Spam SMS Detection")
st.caption(
    "Classifies an SMS message as **SPAM** or **LEGITIMATE** using a TF-IDF + machine "
    "learning pipeline trained on the SMS Spam Collection dataset."
)

metrics = get_metrics()
try:
    model = get_model()
    model_error = None
except Exception as exc:  # noqa: BLE001 - shown to the user
    model, model_error = None, str(exc)

if model_error:
    st.error(f"Model could not be loaded: {model_error}")

tab_predict, tab_metrics, tab_about = st.tabs(["🔍 Predict", "📊 Model Performance", "ℹ️ About"])

# ---------------------------------------------------------------- predict tab
with tab_predict:
    left, right = st.columns([3, 2], gap="large")
    with left:
        st.text_area(
            "SMS message",
            key="sms_input",
            height=150,
            placeholder="Type or paste an SMS message here…",
        )
        predict_clicked = st.button("Predict", type="primary", width="stretch",
                                    disabled=model is None)
        if predict_clicked:
            text = st.session_state.get("sms_input", "")
            try:
                result = predict_message(text, model=model)
            except InvalidInputError as exc:
                st.warning(str(exc))
            except Exception as exc:  # noqa: BLE001
                st.error(f"Prediction failed: {exc}")
            else:
                render_result(result)

    with right:
        st.subheader("Try a sample")
        st.markdown("**Spam examples**")
        for i, msg in enumerate(SAMPLE_SPAM):
            st.button(msg[:70] + "…", key=f"spam_{i}", on_click=set_sample, args=(msg,),
                      width="stretch")
        st.markdown("**Legitimate examples**")
        for i, msg in enumerate(SAMPLE_HAM):
            st.button(msg[:70] + ("…" if len(msg) > 70 else ""), key=f"ham_{i}",
                      on_click=set_sample, args=(msg,), width="stretch")
        st.caption("Click a sample to load it, then press **Predict**.")

# ---------------------------------------------------------------- metrics tab
with tab_metrics:
    if metrics is None:
        st.info("No metrics found. Run `python train.py` to generate reports/metrics.json.")
    else:
        best = metrics["best_model"]
        m = metrics["models"][best]
        st.subheader(f"Selected model: {best}")
        st.caption(f"Held-out test set: {metrics['split']['test_size']} messages "
                   f"({metrics['split']['test_spam']} spam). Spam is the positive class.")
        c1, c2, c3, c4 = st.columns(4)
        c1.metric("Accuracy", f"{m['accuracy']:.2%}")
        c2.metric("Precision (spam)", f"{m['precision_spam']:.2%}")
        c3.metric("Recall (spam)", f"{m['recall_spam']:.2%}")
        c4.metric("F1-score (spam)", f"{m['f1_spam']:.2%}")

        col_cm, col_tbl = st.columns([2, 3], gap="large")
        with col_cm:
            st.markdown("**Confusion matrix**")
            cm_img = FIGURES_DIR / "best_model_confusion_matrix.png"
            if cm_img.exists():
                st.image(str(cm_img), width="stretch")
            cm = m["confusion_matrix"]
            st.dataframe(
                pd.DataFrame(cm, index=["Actual LEGITIMATE", "Actual SPAM"],
                             columns=["Pred LEGITIMATE", "Pred SPAM"]),
                width="stretch",
            )
        with col_tbl:
            st.markdown("**All models (test set)**")
            rows = []
            for name, mm in metrics["models"].items():
                rows.append({
                    "Model": name + (" ⭐" if name == best else ""),
                    "Accuracy": mm["accuracy"], "Precision": mm["precision_spam"],
                    "Recall": mm["recall_spam"], "F1": mm["f1_spam"],
                    "ROC-AUC": mm.get("roc_auc"), "CV F1": mm["cv_f1"],
                })
            st.dataframe(pd.DataFrame(rows).set_index("Model").style.format("{:.4f}"),
                         width="stretch")
            comp_img = FIGURES_DIR / "model_comparison.png"
            if comp_img.exists():
                st.image(str(comp_img), width="stretch")
        st.caption(f"Selection criterion: {metrics['selection_criterion']}.")

# ---------------------------------------------------------------- about tab
with tab_about:
    st.markdown(
        """
        **Pipeline:** text cleaning → TF-IDF (word uni-/bi-grams) → classifier.
        The TF-IDF vectoriser lives inside the scikit-learn pipeline, so it was fitted on
        training data only.

        **Models compared:** Multinomial Naive Bayes, Logistic Regression, and a linear
        Support Vector Machine (calibrated for probabilities).

        **Dataset:** SMS Spam Collection (`data/spam.csv`): 5,572 raw messages, 5,158 after
        removing duplicates.

        CodSoft Machine Learning Internship · Task 4
        """
    )
    for fig in ["class_distribution.png", "message_length_distribution.png", "top_words.png"]:
        p = FIGURES_DIR / fig
        if p.exists():
            st.image(str(p), width="stretch")
