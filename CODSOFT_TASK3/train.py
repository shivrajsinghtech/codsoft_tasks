"""Train, compare and save customer-churn models.

Usage:
    python train.py            # full run: EDA + tuning + evaluation + save
    python train.py --skip-eda
"""
from __future__ import annotations

import argparse
import json
import time

from sklearn.model_selection import train_test_split

from src.config import (
    CV_FOLDS, DATA_PATH, FIGURES_DIR, METRICS_PATH, MODEL_PATH, RANDOM_STATE,
    REPORTS_DIR, TEST_SIZE,
)
from src.data import clean_data, data_quality_report, load_raw_data, split_features_target
from src.eda import generate_eda
from src.evaluate import evaluate_model, feature_importance
from src.models import REFIT_METRIC, cv_summary, get_model_specs, tune_model
from src.predict import load_model, predict_customer, save_model
from src import reporting


def main(skip_eda: bool = False) -> dict:
    start = time.time()
    raw = load_raw_data(DATA_PATH)
    print(f"Loaded {raw.shape[0]:,} rows x {raw.shape[1]} columns from {DATA_PATH.name}")
    raw_quality = data_quality_report(raw)
    df = clean_data(raw)
    quality = data_quality_report(df)
    print(f"Missing values: {quality['total_missing']}, duplicates: {raw_quality['duplicate_rows']}, "
          f"churn rate: {quality['churn_rate']:.2%}")

    FIGURES_DIR.mkdir(parents=True, exist_ok=True)
    eda = None
    if not skip_eda:
        eda = generate_eda(df, FIGURES_DIR)
        reporting.write_eda_report(quality, eda, REPORTS_DIR / "eda_report.md")
        print("EDA figures written to reports/figures/")

    X, y = split_features_target(df)
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=TEST_SIZE, stratify=y, random_state=RANDOM_STATE
    )

    results, fitted = {}, {}
    for spec in get_model_specs():
        t0 = time.time()
        search = tune_model(spec, X_train, y_train)
        fitted[spec.name] = search.best_estimator_
        results[spec.name] = {
            "cv": cv_summary(search),
            "test": evaluate_model(search.best_estimator_, X_test, y_test),
        }
        cv, t = results[spec.name]["cv"], results[spec.name]["test"]
        print(f"{spec.name:20s} CV F1={cv['f1']['mean']:.4f} CV AUC={cv['roc_auc']['mean']:.4f} | "
              f"test acc={t['accuracy']:.4f} prec={t['precision']:.4f} rec={t['recall']:.4f} "
              f"f1={t['f1']:.4f} auc={t['roc_auc']:.4f} ({time.time() - t0:.0f}s)")

    # Select on cross-validated churn-class F1 (training data only), never on test scores.
    best = max(results, key=lambda n: results[n]["cv"][REFIT_METRIC]["mean"])
    best_model = fitted[best]
    cvb = results[best]["cv"]
    rationale = (
        f"{best} achieved the highest mean cross-validated F1 for the churn class "
        f"({cvb['f1']['mean']:.4f}), with CV recall {cvb['recall']['mean']:.4f} and CV ROC-AUC "
        f"{cvb['roc_auc']['mean']:.4f}. F1 was used because the business cost is two-sided: "
        "missing a churner (false negative) loses a customer, while flagging a loyal customer "
        "(false positive) wastes retention budget. The selection was made on cross-validation "
        "scores; test-set scores are reported only as an unbiased final check."
    )
    print(f"Selected model: {best}")

    importance = feature_importance(best_model, X_test, y_test)

    reporting.plot_model_comparison(results, FIGURES_DIR / "11_model_comparison.png")
    reporting.plot_confusion_matrices(fitted, X_test, y_test, FIGURES_DIR / "12_confusion_matrices.png")
    reporting.plot_roc_curves(fitted, X_test, y_test, FIGURES_DIR / "13_roc_curves.png")
    reporting.plot_feature_importance(
        importance, f"Permutation feature importance - {best}",
        FIGURES_DIR / "14_feature_importance.png",
    )
    reporting.plot_single_confusion(
        best_model, X_test, y_test, f"{best} - test confusion matrix",
        FIGURES_DIR / "15_best_model_confusion_matrix.png",
    )

    save_model(best_model, MODEL_PATH)
    # Round-trip check: the saved pipeline must reload and predict identically.
    reloaded = load_model(MODEL_PATH)
    assert (reloaded.predict(X_test) == best_model.predict(X_test)).all()
    sample = predict_customer(reloaded, X_test.iloc[0].to_dict())
    print(f"Saved pipeline to {MODEL_PATH.relative_to(MODEL_PATH.parent.parent)}; "
          f"sample prediction: {sample['label']} (p={sample['churn_probability']:.3f})")

    metrics = {
        "data": quality,
        "split": {"test_size": TEST_SIZE, "random_state": RANDOM_STATE,
                  "train_rows": int(len(X_train)), "test_rows": int(len(X_test)),
                  "train_churn_rate": float(y_train.mean()), "test_churn_rate": float(y_test.mean())},
        "cv": {"folds": CV_FOLDS, "refit_metric": REFIT_METRIC},
        "models": results,
        "best_model": best,
        "selection_rationale": rationale,
        "feature_importance": importance,
        "eda": eda,
        "training_seconds": round(time.time() - start, 1),
    }
    METRICS_PATH.write_text(json.dumps(metrics, indent=2), encoding="utf-8")
    reporting.write_model_report(metrics, REPORTS_DIR / "model_comparison.md")
    print(f"Done in {metrics['training_seconds']}s. Reports in reports/")
    return metrics


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--skip-eda", action="store_true", help="skip EDA figure generation")
    main(skip_eda=parser.parse_args().skip_eda)
