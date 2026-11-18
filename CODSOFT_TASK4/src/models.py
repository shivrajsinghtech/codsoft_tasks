"""Model pipelines: TF-IDF + {Naive Bayes, Logistic Regression, SVM}.

Each model is a scikit-learn ``Pipeline`` so that the TF-IDF vocabulary and
IDF weights are learned from the training fold only (no data leakage).
"""

from sklearn.calibration import CalibratedClassifierCV
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.naive_bayes import MultinomialNB
from sklearn.pipeline import Pipeline
from sklearn.svm import LinearSVC

from src.config import RANDOM_STATE
from src.text_cleaning import clean_text


def build_vectorizer() -> TfidfVectorizer:
    return TfidfVectorizer(
        preprocessor=clean_text,
        ngram_range=(1, 2),
        min_df=1,
        sublinear_tf=True,
    )


def build_pipelines() -> dict[str, Pipeline]:
    """Return the three untrained candidate pipelines."""
    return {
        "Multinomial Naive Bayes": Pipeline(
            [("tfidf", build_vectorizer()), ("clf", MultinomialNB())]
        ),
        "Logistic Regression": Pipeline(
            [
                ("tfidf", build_vectorizer()),
                (
                    "clf",
                    LogisticRegression(
                        max_iter=2000,
                        class_weight="balanced",
                        random_state=RANDOM_STATE,
                    ),
                ),
            ]
        ),
        # LinearSVC has no predict_proba; calibration provides real
        # probabilities for the app's confidence score.
        "Support Vector Machine": Pipeline(
            [
                ("tfidf", build_vectorizer()),
                (
                    "clf",
                    CalibratedClassifierCV(
                        LinearSVC(class_weight="balanced", max_iter=10000, random_state=RANDOM_STATE),
                        cv=5,
                        method="sigmoid",
                    ),
                ),
            ]
        ),
    }


# Small hyper-parameter grids searched with stratified CV on the training set.
PARAM_GRIDS: dict[str, dict] = {
    "Multinomial Naive Bayes": {
        "tfidf__min_df": [1, 2],
        "clf__alpha": [0.01, 0.05, 0.1, 0.5, 1.0],
    },
    "Logistic Regression": {
        "tfidf__min_df": [1, 2],
        "clf__C": [1.0, 10.0, 100.0, 1000.0],
    },
    "Support Vector Machine": {
        "tfidf__min_df": [1, 2],
        "clf__estimator__C": [0.1, 1.0, 10.0],
    },
}
