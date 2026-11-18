import json

from src.predictor import GenrePredictor
from src.training import run_training
from tests.conftest import TINY_VEC

FAST_GRIDS = {
    "naive_bayes": [{"alpha": 0.1}, {"alpha": 1.0}],
    "logistic_regression": [{"C": 5.0}],
    "linear_svm": [{"C": 0.5, "calibration_cv": 2}],
}


def test_run_training_end_to_end(tiny_data_dir, tmp_path):
    model_path = tmp_path / "models" / "genre_classifier.joblib"
    reports = tmp_path / "reports"
    summary = run_training(
        data_dir=tiny_data_dir,
        model_path=model_path,
        reports_dir=reports,
        figures_dir=reports / "figures",
        models=("naive_bayes", "logistic_regression", "linear_svm"),
        param_grids=FAST_GRIDS,
        vectorizer_params=TINY_VEC,
        run_eda_step=True,
        n_jobs=1,
    )
    # model saved + loadable
    assert model_path.exists()
    predictor = GenrePredictor.load(model_path)
    assert predictor.metadata["selection_metric"] == "validation macro_f1"
    # selection is based on validation results only
    val_best = max(summary["validation"], key=lambda r: (r["macro_f1"], r["accuracy"]))
    assert val_best["key"] == summary["selected_model"]["model_key"]
    # every expected report exists
    for name in [
        "metrics.json", "eda_summary.json", "validation_tuning_results.csv",
        "validation_model_comparison.csv", "test_model_comparison.csv",
        "per_genre_metrics_test.csv", "test_predictions.csv",
    ]:
        assert (reports / name).exists(), name
    for fig in ["confusion_matrix_test.png", "genre_distribution.png", "model_comparison_test.png"]:
        assert (reports / "figures" / fig).exists(), fig
    metrics = json.loads((reports / "metrics.json").read_text())
    assert metrics["test"]["selected_model_metrics"]["accuracy"] >= 0.9
    assert metrics["dataset"]["train_split"] + metrics["dataset"]["validation_split"] == 36
