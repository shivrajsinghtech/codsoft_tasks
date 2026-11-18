"""Feature preprocessing with ColumnTransformer."""
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

from .config import BINARY_FEATURES, CATEGORICAL_FEATURES, NUMERIC_FEATURES


def build_preprocessor() -> ColumnTransformer:
    """Scale numeric features, pass binary flags through, one-hot encode categoricals.

    Imputers are included defensively so the saved pipeline can handle a missing
    value at inference time; they are fitted on the training fold only.
    """
    numeric = Pipeline(
        [
            ("impute", SimpleImputer(strategy="median")),
            ("scale", StandardScaler()),
        ]
    )
    binary = SimpleImputer(strategy="most_frequent")
    categorical = Pipeline(
        [
            ("impute", SimpleImputer(strategy="most_frequent")),
            ("onehot", OneHotEncoder(handle_unknown="ignore", sparse_output=False)),
        ]
    )
    return ColumnTransformer(
        [
            ("num", numeric, NUMERIC_FEATURES),
            ("bin", binary, BINARY_FEATURES),
            ("cat", categorical, CATEGORICAL_FEATURES),
        ],
        verbose_feature_names_out=False,
    )
