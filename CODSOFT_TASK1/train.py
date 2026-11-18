"""Train, compare and evaluate the movie genre classifiers.

Usage:
    python train.py                       # full run (EDA + tuning + final test evaluation)
    python train.py --skip-eda
    python train.py --models naive_bayes linear_svm
    python train.py --data-dir "path/to/Genre Classification Dataset"
"""

from __future__ import annotations

import argparse
import json
import logging

from src.config import DATA_DIR, FIGURES_DIR, MODEL_PATH, REPORTS_DIR
from src.models import MODEL_DISPLAY_NAMES
from src.training import run_training


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--data-dir", default=str(DATA_DIR))
    parser.add_argument("--model-path", default=str(MODEL_PATH))
    parser.add_argument("--reports-dir", default=str(REPORTS_DIR))
    parser.add_argument("--models", nargs="+", choices=sorted(MODEL_DISPLAY_NAMES),
                        default=["naive_bayes", "logistic_regression", "linear_svm", "word2vec_logreg"])
    parser.add_argument("--skip-eda", action="store_true")
    parser.add_argument("--n-jobs", type=int, default=4)
    args = parser.parse_args()

    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    logging.getLogger("gensim").setLevel(logging.WARNING)
    reports_dir = args.reports_dir
    figures_dir = FIGURES_DIR if reports_dir == str(REPORTS_DIR) else f"{reports_dir}/figures"
    summary = run_training(
        data_dir=args.data_dir,
        model_path=args.model_path,
        reports_dir=reports_dir,
        figures_dir=figures_dir,
        models=tuple(args.models),
        run_eda_step=not args.skip_eda,
        n_jobs=args.n_jobs,
    )
    print(json.dumps({"selected_model": summary["selected_model"]["model_name"],
                      "validation": summary["selected_model"]["validation_metrics"],
                      "test": summary.get("test", {}).get("selected_model_metrics")}, indent=2))


if __name__ == "__main__":
    main()
