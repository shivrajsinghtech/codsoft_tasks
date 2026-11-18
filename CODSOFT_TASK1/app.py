"""Movie Genre Classifier – Streamlit application.

Run with:  streamlit run app.py
"""

from __future__ import annotations

import json
from pathlib import Path

import pandas as pd
import streamlit as st

from src.config import FIGURES_DIR, MODEL_PATH, REPORTS_DIR
from src.predictor import GenrePredictor, InvalidInputError

APP_NAME = "Movie Genre Classifier"
REPO_URL = "https://github.com/shivrajsinghtech/CODSOFT_TASK1"
DISCLAIMER = (
    "Predictions are statistical estimates produced by a machine-learning model trained on "
    "plot summaries. They are **not** guaranteed or official movie classifications – many films "
    "belong to several genres, and the model can be wrong."
)

# Short plot summaries written for this demo (not taken from the dataset).
EXAMPLES = {
    "— choose an example —": ("", ""),
    "Haunted lighthouse (horror)": (
        "The Keeper's Light",
        "A young couple restores an abandoned lighthouse on a remote island. At night they hear "
        "whispers in the walls, the lamp turns on by itself, and the ghost of a drowned keeper "
        "begins hunting them one by one until they uncover the bloody secret buried beneath the rocks.",
    ),
    "Wildlife of the Arctic (documentary)": (
        "Frozen Frontier",
        "Filmed over three years, this film follows scientists and Inuit hunters as they track polar "
        "bears, walrus and migrating whales across the Arctic, interviewing experts about how "
        "climate change is transforming the region and the lives of the people who live there.",
    ),
    "Wedding chaos (comedy)": (
        "Best Man Down",
        "When a nervous best man accidentally loses the wedding rings the night before the ceremony, "
        "he and his hopeless friends race across town, crash a karaoke party, steal a goat and "
        "end up in hilarious trouble with the bride's grandmother.",
    ),
    "Outlaw on the frontier (western)": (
        "Dust and Iron",
        "In 1870s Texas a retired gunslinger is forced to pick up his revolver again when a gang of "
        "outlaws takes over the town, terrorises the ranchers and the sheriff is shot dead in the saloon.",
    ),
    "Mars colony (sci-fi)": (
        "Red Horizon",
        "In the year 2140 the first colony on Mars loses contact with Earth. An engineer discovers "
        "that the colony's artificial intelligence has been secretly altering oxygen levels and must "
        "shut it down before the spaceship returns.",
    ),
    "Championship season (sport)": (
        "The Last Inning",
        "A struggling high-school baseball team gets a new coach, a former major-league pitcher, "
        "who trains the players to believe in themselves as they fight their way to the state championship.",
    ),
}

st.set_page_config(page_title=APP_NAME, page_icon="🎬", layout="wide")

st.markdown(
    """
    <style>
      .block-container {padding-top: 2rem; max-width: 1200px;}
      .genre-card {padding: 1.1rem 1.4rem; border-radius: 12px; color: #fff;
                   background: linear-gradient(135deg, #1f3b73 0%, #3a6fd8 100%);}
      .genre-card .label {font-size: 0.85rem; opacity: 0.85; letter-spacing: .06em; text-transform: uppercase;}
      .genre-card .value {font-size: 2.2rem; font-weight: 700; text-transform: capitalize;}
    </style>
    """,
    unsafe_allow_html=True,
)


@st.cache_resource(show_spinner="Loading model ...")
def get_predictor(path: str) -> GenrePredictor:
    return GenrePredictor.load(path)


@st.cache_data
def read_json(path: str) -> dict:
    p = Path(path)
    return json.loads(p.read_text()) if p.exists() else {}


@st.cache_data
def read_csv(path: str) -> pd.DataFrame | None:
    p = Path(path)
    return pd.read_csv(p) if p.exists() else None


def show_figure(name: str, caption: str) -> None:
    path = FIGURES_DIR / name
    if path.exists():
        st.image(str(path), caption=caption, width="stretch")
    else:
        st.info(f"Figure `{name}` not found – run `python train.py` to generate it.")


def pct(value) -> str:
    return f"{value:.1%}" if isinstance(value, (int, float)) else "–"


metrics = read_json(str(REPORTS_DIR / "metrics.json"))
eda = read_json(str(REPORTS_DIR / "eda_summary.json"))
test_metrics = metrics.get("test", {}).get("selected_model_metrics", {})

predictor = None
model_error = None
try:
    predictor = get_predictor(str(MODEL_PATH))
except FileNotFoundError as exc:
    model_error = str(exc)

# ------------------------------------------------------------------ sidebar
with st.sidebar:
    st.title("🎬 " + APP_NAME)
    page = st.radio(
        "Navigate",
        ["🎯 Predict Genre", "📊 Model Performance", "📈 Dataset Insights", "ℹ️ About"],
        key="page",
    )
    st.divider()
    if predictor is not None:
        st.caption("Active model")
        st.write(f"**{predictor.metadata.get('model_name', 'Saved pipeline')}**")
        st.caption(f"{len(predictor.classes)} genres · trained on "
                   f"{predictor.metadata.get('n_training_records', '?'):,} movies")
    st.caption(f"[GitHub repository]({REPO_URL})")

st.title(APP_NAME)
st.caption("Predict a movie's genre from its plot summary using TF-IDF features and linear classifiers.")

c1, c2, c3, c4 = st.columns(4)
c1.metric("Genres", eda.get("n_genres", len(predictor.classes) if predictor else "–"))
c2.metric("Training movies", f"{eda.get('train_records', 0):,}" if eda else "–")
c3.metric("Test accuracy", pct(test_metrics.get("accuracy")))
c4.metric("Test macro-F1", pct(test_metrics.get("macro_f1")))

# ------------------------------------------------------------------ pages
if page == "🎯 Predict Genre":
    if model_error:
        st.error(f"The model is not available. {model_error}")

    def load_example() -> None:
        title, plot = EXAMPLES[st.session_state["example"]]
        st.session_state["title_input"] = title
        st.session_state["plot_input"] = plot

    st.selectbox("Example plot summaries", list(EXAMPLES), key="example", on_change=load_example)

    with st.form("predict_form"):
        title = st.text_input("Movie title (optional)", key="title_input", placeholder="e.g. Red Horizon")
        plot = st.text_area(
            "Plot summary",
            key="plot_input",
            height=180,
            placeholder="Describe the story of the movie in a few sentences ...",
        )
        submitted = st.form_submit_button("Predict Genre", type="primary", width="stretch")

    if submitted:
        if predictor is None:
            st.error("Cannot predict: the trained model file is missing. Run `python train.py`.")
        elif not plot or not plot.strip():
            st.error("Please enter a plot summary before predicting.")
        else:
            try:
                result = predictor.predict(plot, title=title, top_k=3)
            except InvalidInputError as exc:
                st.error(str(exc))
            else:
                left, right = st.columns([1, 1.4])
                with left:
                    st.markdown(
                        f"<div class='genre-card'><div class='label'>Predicted genre</div>"
                        f"<div class='value'>{result['genre']}</div>"
                        f"<div>Confidence: {result['confidence']:.1%}</div></div>",
                        unsafe_allow_html=True,
                    )
                    st.progress(min(max(result["confidence"], 0.0), 1.0))
                    if result["confidence"] < 0.40:
                        st.warning("Low confidence – the plot fits several genres. Check the alternatives.")
                with right:
                    st.subheader("Top 3 predicted genres")
                    top = pd.DataFrame(result["top_k"]).rename(columns={"genre": "Genre", "score": "Score"})
                    st.dataframe(
                        top,
                        hide_index=True,
                        width="stretch",
                        column_config={
                            "Score": st.column_config.ProgressColumn(
                                "Score", format="percent", min_value=0.0, max_value=1.0
                            )
                        },
                    )
    st.info(DISCLAIMER, icon="⚠️")

elif page == "📊 Model Performance":
    st.header("Model performance")
    if not metrics:
        st.info("No metrics found – run `python train.py`.")
    else:
        sel = metrics["selected_model"]
        st.markdown(
            f"**Selected model:** {sel['model_name']} with parameters `{json.dumps(sel['params'])}`, "
            f"chosen by **{sel['selection_metric']}** on a stratified 20% validation split of "
            "`train_data.txt`. The official test labels were used only once, after selection."
        )
        cols = ["model", "accuracy", "macro_precision", "macro_recall", "macro_f1", "weighted_f1"]
        fmt = {c: "{:.3f}" for c in cols[1:]}
        val = read_csv(str(REPORTS_DIR / "validation_model_comparison.csv"))
        test = read_csv(str(REPORTS_DIR / "test_model_comparison.csv"))
        tab1, tab2, tab3, tab4 = st.tabs(["Comparison", "Confusion matrix", "Per-genre", "Tuning log"])
        with tab1:
            if val is not None:
                st.subheader("Validation (used for model selection)")
                st.dataframe(val[cols].style.format(fmt), hide_index=True, width="stretch")
            if test is not None:
                st.subheader("Official held-out test set (54,200 movies)")
                st.dataframe(test[cols].style.format(fmt), hide_index=True, width="stretch")
            show_figure("model_comparison_test.png", "Test-set comparison of all models")
        with tab2:
            show_figure("confusion_matrix_test.png", "Row-normalised confusion matrix on the test set")
        with tab3:
            per_genre = read_csv(str(REPORTS_DIR / "per_genre_metrics_test.csv"))
            if per_genre is not None:
                st.dataframe(
                    per_genre.style.format({"precision": "{:.3f}", "recall": "{:.3f}", "f1": "{:.3f}"}),
                    hide_index=True,
                    width="stretch",
                )
            show_figure("per_genre_metrics_test.png", "Per-genre precision, recall and F1 (test set)")
        with tab4:
            tuning = read_csv(str(REPORTS_DIR / "validation_tuning_results.csv"))
            if tuning is not None:
                st.dataframe(tuning, hide_index=True, width="stretch")

elif page == "📈 Dataset Insights":
    st.header("Dataset insights")
    if eda:
        st.subheader("Genre distribution (training set)")
        show_figure("genre_distribution.png", "Movies per genre in train_data.txt (sorted by frequency)")
        st.caption(
            f"Strong class imbalance: the largest genre is "
            f"{eda['imbalance_ratio_largest_to_smallest']:.0f}× the smallest. "
            "Macro-averaged metrics are reported so rare genres are not hidden."
        )
        a, b, c = st.columns(3)
        a.metric("Median words per plot", f"{eda['description_words']['50%']:.0f}")
        b.metric("Empty descriptions", eda["empty_descriptions_train"])
        c.metric("Duplicate IDs", eda["duplicate_ids_train"])
    show_figure("description_length_distribution.png", "Description length distribution")
    show_figure("description_length_by_genre.png", "Description length by genre")
    show_figure("top_words_by_genre.png", "Most distinctive words per genre")

else:
    st.header("About the project")
    st.markdown(
        f"""
This application was built for **CodSoft Machine Learning Internship – Task 1: Movie Genre
Classification**. It predicts a movie's genre (27 classes) from its plot summary.

**Pipeline**
1. Parse the `ID ::: TITLE ::: GENRE ::: DESCRIPTION` files and combine the cleaned title with the plot.
2. Clean the text (lower-case, accent folding, punctuation removal) and vectorise with **TF-IDF**
   (unigrams + bigrams, sublinear term frequency, English stop words removed).
3. Compare **Multinomial Naive Bayes**, **Logistic Regression** and a **Linear SVM**
   (plus a supplementary **Word2Vec** embedding model), tuned on a stratified validation split
   with class weighting considered for rare genres.
4. Refit the selected model on the full training set and evaluate once on the official held-out test labels.

**Dataset:** Genre Classification Dataset (IMDb-derived), source documented as
`ftp://ftp.fu-berlin.de/pub/misc/movies/database/`.

**Repository:** [{REPO_URL}]({REPO_URL})
"""
    )
    st.info(DISCLAIMER, icon="⚠️")
