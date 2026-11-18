import numpy as np

from src.preprocessing import build_preprocessor


def test_encoding_and_scaling(xy):
    X, _ = xy
    pre = build_preprocessor().fit(X)
    names = list(pre.get_feature_names_out())
    for col in ["Geography_France", "Geography_Germany", "Geography_Spain",
                "Gender_Female", "Gender_Male"]:
        assert col in names
    out = pre.transform(X)
    assert out.shape == (len(X), 13)
    assert not np.isnan(out).any()
    # scaled numeric columns have ~zero mean / unit std
    num = out[:, :6]
    assert np.allclose(num.mean(axis=0), 0, atol=1e-6)
    assert np.allclose(num.std(axis=0), 1, atol=1e-6)
    # one-hot columns are 0/1 and each row has exactly one geography
    geo = out[:, names.index("Geography_France"):names.index("Geography_Spain") + 1]
    assert set(np.unique(geo)) == {0.0, 1.0}
    assert (geo.sum(axis=1) == 1).all()


def test_unknown_category_and_missing_value_handled(xy):
    X, _ = xy
    pre = build_preprocessor().fit(X)
    row = X.head(1).copy()
    row["Geography"] = "Italy"
    row["Balance"] = np.nan
    out = pre.transform(row)
    assert not np.isnan(out).any()
