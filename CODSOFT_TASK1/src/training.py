"""End-to-end training workflow.

1. Load ``train_data.txt`` and make a stratified 80/20 train/validation split.
2. Tune every model on the validation split. For the TF-IDF models the
   vectoriser is fitted on the training part only and reused for all grid
   points (equivalent to refitting the pipeline, and leak-free).
3. Select the best configuration per model, and the overall best model,
   by **validation macro-F1** (rare genres count as much as frequent ones).
4. Refit the chosen configurations as complete pipelines on all of
   ``train_data.txt`` and save the overall best with joblib.
5. Predict ``test_data.txt`` (unlabelled).
6. Only now load ``test_data_solution.txt`` and report held-out metrics.
   The selection in step 3 is already fixed and recorded, so test results
   cannot influence it.
"""

from __future__ import annotations

import json
import logging
import platform
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import sklearn
from joblib import Parallel, delayed
from sklearn.base import clone
from sklearn.model_selection import train_test_split

from src.config import FIGURES_DIR, MODEL_PATH, RANDOM_STATE, REPORTS_DIR, VALIDATION_SIZE
from src.data_loader import load_test_data, load_test_solution, load_train_data
from src.eda import run_eda
from src.evaluation import (
    compute_metrics,
    per_genre_report,
    plot_confusion_matrix,
    plot_model_comparison,
    plot_per_genre,
)
from src.label_encoding import GenreLabelEncoder
from src.models import (
    MODEL_DISPLAY_NAMES,
    PARAM_GRIDS,
    TFIDF_MODELS,
    build_classifier,
    build_pipeline,
    build_vectorizer,
)
from src.predictor import GenrePredictor, save_model
from src.preprocessing import build_text

log = logging.getLogger(__name__)
SELECTION_METRIC = "macro_f1"


def _fit_score(name: str, params: dict, X_tr, y_tr, X_val, y_val) -> dict[str, Any]:
    start = time.time()
    clf = build_classifier(name, **dict(params))
    clf.fit(X_tr, y_tr)
    metrics = compute_metrics(y_val, clf.predict(X_val))
    return {"model": name, "params": params, **metrics, "fit_seconds": round(time.time() - start, 1)}


def _fit_score_pipeline(name, params, emb_params, X_tr, y_tr, X_val, y_val) -> dict[str, Any]:
    start = time.time()
    pipe = build_pipeline(name, embedding_params=emb_params, **dict(params))
    pipe.fit(X_tr, y_tr)
    metrics = compute_metrics(y_val, pipe.predict(X_val))
    return {"model": name, "params": params, **metrics, "fit_seconds": round(time.time() - start, 1)}


def _rank_key(row: dict) -> tuple:
    return (row[SELECTION_METRIC], row["accuracy"])


def run_training(
    data_dir: str | Path,
    model_path: str | Path = MODEL_PATH,
    reports_dir: str | Path = REPORTS_DIR,
    figures_dir: str | Path = FIGURES_DIR,
    models: tuple[str, ...] = ("naive_bayes", "logistic_regression", "linear_svm", "word2vec_logreg"),
    param_grids: dict[str, list[dict]] | None = None,
    vectorizer_params: dict | None = None,
    embedding_params: dict | None = None,
    run_eda_step: bool = True,
    evaluate_on_test: bool = True,
    n_jobs: int = 4,
) -> dict[str, Any]:
    t0 = time.time()
    reports_dir, figures_dir = Path(reports_dir), Path(figures_dir)
    reports_dir.mkdir(parents=True, exist_ok=True)
    figures_dir.mkdir(parents=True, exist_ok=True)
    grids = {**PARAM_GRIDS, **(param_grids or {})}

    # ---------------------------------------------------------------- data
    train_df, train_report = load_train_data(data_dir)
    test_df, test_report = load_test_data(data_dir)
    log.info("Loaded %d training and %d test records", len(train_df), len(test_df))

    if run_eda_step:
        log.info("Running EDA ...")
        run_eda(train_df, test_df, reports_dir, figures_dir)

    train_df["text"] = [build_text(t, d) for t, d in zip(train_df["title"], train_df["description"])]
    usable = train_df["text"].str.strip() != ""
    dropped_empty = int((~usable).sum())
    train_df = train_df[usable].reset_index(drop=True)

    encoder = GenreLabelEncoder().fit(train_df["genre"])
    labels = list(encoder.classes_)

    X_tr, X_val, y_tr, y_val = train_test_split(
        train_df["text"].to_numpy(dtype=object),
        train_df["genre"].to_numpy(dtype=object),
        test_size=VALIDATION_SIZE,
        stratify=train_df["genre"].to_numpy(dtype=object),
        random_state=RANDOM_STATE,
    )
    log.info("Stratified split: %d train / %d validation", len(X_tr), len(X_val))

    # -------------------------------------------------------------- tuning
    results: list[dict] = []
    tfidf_models = [m for m in models if m in TFIDF_MODELS]
    if tfidf_models:
        vectorizer = build_vectorizer(**(vectorizer_params or {}))
        V_tr = vectorizer.fit_transform(X_tr)
        V_val = vectorizer.transform(X_val)
        log.info("TF-IDF features (fitted on train split only): %d", V_tr.shape[1])
        jobs = [(m, p) for m in tfidf_models for p in grids[m]]
        results += Parallel(n_jobs=n_jobs, verbose=0)(
            delayed(_fit_score)(m, p, V_tr, y_tr, V_val, y_val) for m, p in jobs
        )
    if "word2vec_logreg" in models:
        for p in grids["word2vec_logreg"]:
            results.append(
                _fit_score_pipeline("word2vec_logreg", p, embedding_params, X_tr, y_tr, X_val, y_val)
            )

    tuning = pd.DataFrame(results)
    tuning["params"] = tuning["params"].map(json.dumps)
    tuning.sort_values(["model", SELECTION_METRIC], ascending=[True, False]).to_csv(
        reports_dir / "validation_tuning_results.csv", index=False
    )

    best_per_model = {}
    for m in models:
        rows = [r for r in results if r["model"] == m]
        best_per_model[m] = max(rows, key=_rank_key)
    selected = max(best_per_model.values(), key=_rank_key)["model"]
    log.info("Selected on validation %s: %s", SELECTION_METRIC, selected)

    val_table = pd.DataFrame(
        [
            {"model": MODEL_DISPLAY_NAMES[m], "key": m, "best_params": json.dumps(r["params"]),
             **{k: r[k] for k in ("accuracy", "macro_precision", "macro_recall", "macro_f1", "weighted_f1")}}
            for m, r in best_per_model.items()
        ]
    ).sort_values("macro_f1", ascending=False)
    val_table.to_csv(reports_dir / "validation_model_comparison.csv", index=False)
    plot_model_comparison(
        val_table, figures_dir / "model_comparison_validation.png",
        "Validation set (20% stratified split of train_data.txt) – best configuration per model",
    )

    # -------------------------------------------- refit on full training set
    final_pipelines = {}
    for m in models:
        params = dict(best_per_model[m]["params"])
        pipe = build_pipeline(
            m, vectorizer_params=vectorizer_params, embedding_params=embedding_params, **params
        )
        log.info("Refitting %s on all %d training records", m, len(train_df))
        pipe.fit(train_df["text"].to_numpy(dtype=object), train_df["genre"].to_numpy(dtype=object))
        final_pipelines[m] = pipe

    best_pipeline = final_pipelines[selected]
    metadata = {
        "model_key": selected,
        "model_name": MODEL_DISPLAY_NAMES[selected],
        "params": best_per_model[selected]["params"],
        "selection_metric": f"validation {SELECTION_METRIC}",
        "validation_metrics": {k: best_per_model[selected][k] for k in
                               ("accuracy", "macro_precision", "macro_recall", "macro_f1", "weighted_f1")},
        "classes": labels,
        "n_training_records": int(len(train_df)),
        "input": "cleaned title + plot description",
        "trained_at": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "sklearn_version": sklearn.__version__,
        "python_version": platform.python_version(),
    }
    save_model(best_pipeline, model_path, metadata)
    log.info("Saved %s to %s", selected, model_path)

    # ------------------------------------------- predictions on test_data.txt
    predictor = GenrePredictor(best_pipeline, metadata)
    preds = predictor.predict_batch(test_df["title"], test_df["description"])
    pred_df = pd.DataFrame(
        {
            "id": test_df["id"],
            "title": test_df["title"],
            "predicted_genre": [p["genre"] for p in preds],
            "confidence": [round(p["confidence"], 4) for p in preds],
            "top3": [" | ".join(f"{t['genre']}:{t['score']:.3f}" for t in p["top_k"]) for p in preds],
        }
    )
    pred_df.to_csv(reports_dir / "test_predictions.csv", index=False)

    summary: dict[str, Any] = {
        "dataset": {
            "train": train_report.as_dict(),
            "test": test_report.as_dict(),
            "dropped_empty_text_records": dropped_empty,
            "train_split": int(len(X_tr)),
            "validation_split": int(len(X_val)),
            "n_genres": len(labels),
        },
        "selected_model": metadata,
        "validation": val_table.drop(columns="best_params").to_dict(orient="records"),
    }

    # --------------------------------- final held-out evaluation (once, last)
    if evaluate_on_test:
        solution, solution_report = load_test_solution(data_dir)
        merged = test_df.merge(solution[["id", "genre", "description"]], on="id", suffixes=("", "_sol"))
        if len(merged) != len(test_df) or not (merged["description"] == merged["description_sol"]).all():
            raise RuntimeError("test_data.txt and test_data_solution.txt are not aligned")
        y_test = merged["genre"].to_numpy(dtype=object)
        test_texts = [build_text(t, d) for t, d in zip(merged["title"], merged["description"])]

        rows = []
        for m, pipe in final_pipelines.items():
            y_hat = pred_df["predicted_genre"].to_numpy(dtype=object) if m == selected else pipe.predict(test_texts)
            rows.append({"model": MODEL_DISPLAY_NAMES[m], "key": m, "selected": m == selected,
                         **compute_metrics(y_test, y_hat)})
        test_table = pd.DataFrame(rows).sort_values("macro_f1", ascending=False)
        test_table.to_csv(reports_dir / "test_model_comparison.csv", index=False)
        plot_model_comparison(
            test_table, figures_dir / "model_comparison_test.png",
            "Official held-out test set (54,200 movies) – after model selection",
        )

        y_best = pred_df["predicted_genre"].to_numpy(dtype=object)
        genre_report = per_genre_report(y_test, y_best, labels)
        genre_report.to_csv(reports_dir / "per_genre_metrics_test.csv", index=False)
        plot_per_genre(genre_report, figures_dir / "per_genre_metrics_test.png",
                       f"Per-genre precision / recall / F1 – {MODEL_DISPLAY_NAMES[selected]} (test)")
        cm = plot_confusion_matrix(
            y_test, y_best, labels, figures_dir / "confusion_matrix_test.png",
            f"Normalised confusion matrix – {MODEL_DISPLAY_NAMES[selected]} (official test set)",
        )
        pd.DataFrame(cm, index=labels, columns=labels).to_csv(reports_dir / "confusion_matrix_test.csv")
        top3_hit = float(np.mean([y in {t["genre"] for t in p["top_k"]} for y, p in zip(y_test, preds)]))

        summary["test"] = {
            "solution_file": solution_report.as_dict(),
            "selected_model_metrics": {**compute_metrics(y_test, y_best), "top3_accuracy": top3_hit},
            "all_models": test_table.to_dict(orient="records"),
        }

    summary["runtime_seconds"] = round(time.time() - t0, 1)
    (reports_dir / "metrics.json").write_text(json.dumps(summary, indent=2, default=str))
    return summary
