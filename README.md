# CodSoft Machine Learning Internship — Completed Tasks

This repository collects the three machine learning projects I completed during the
**CodSoft Machine Learning Internship**. Each project is self-contained in its own folder
with its source code, trained model, tests, notebook, reports and README.

| Folder | Project | Problem type | Selected model | Headline result (held-out test set) |
|---|---|---|---|---|
| [`CODSOFT_TASK1`](CODSOFT_TASK1/) | 🎬 **Movie Genre Classification** | Multi-class NLP (27 genres) | TF-IDF + Logistic Regression (balanced) | Macro-F1 **0.424** · Accuracy **59.2 %** · Top-3 accuracy **84.9 %** (54,200 movies) |
| [`CODSOFT_TASK3`](CODSOFT_TASK3/) | 📉 **Customer Churn Prediction** | Binary tabular classification | Random Forest | Churn F1 **0.619** · Recall **0.671** · ROC-AUC **0.858** · Accuracy **83.2 %** |
| [`CODSOFT_TASK4`](CODSOFT_TASK4/) | 📩 **Spam SMS Detection** | Binary NLP classification | TF-IDF + Logistic Regression | Spam F1 **0.954** · Recall **0.969** · ROC-AUC **0.999** · Accuracy **98.8 %** |

Every project uses leakage-free scikit-learn `Pipeline`s, selects its model on
training/validation data only (the test set is used once, for the final evaluation), and
includes a **Streamlit web app** and a **pytest** test suite.

---

## Contents

1. [Projects](#projects)
2. [Repository structure](#repository-structure)
3. [Requirements](#requirements)
4. [Running a project](#running-a-project)
5. [Testing](#testing)
6. [Datasets](#datasets)
7. [Original repositories](#original-repositories)

---

## Projects

### Task 1 — Movie Genre Classification
Predicts a movie's genre from its plot summary (and, optionally, its title). The project
compares TF-IDF features with Multinomial Naive Bayes, Logistic Regression and a
calibrated Linear SVM, plus a supplementary Word2Vec experiment. Selection is by validation
macro-F1, so that rare genres count as much as frequent ones. It includes a Streamlit app, a
command-line predictor (`predict.py`) and 64 tests.
→ [Full README](CODSOFT_TASK1/README.md) · [Evaluation report](CODSOFT_TASK1/reports/evaluation_report.md)

### Task 3 — Customer Churn Prediction
Estimates the probability that a bank customer will leave, using demographics and account
behaviour (10,000 customers). It compares Logistic Regression, Random Forest and Gradient
Boosting, each tuned with stratified 5-fold cross-validation for churn F1. The EDA finds
age, number of products, activity status and geography to be the main churn drivers.
It includes a Streamlit dashboard and 44 tests.
→ [Full README](CODSOFT_TASK3/README.md) · [EDA report](CODSOFT_TASK3/reports/eda_report.md) · [Model comparison](CODSOFT_TASK3/reports/model_comparison.md)

### Task 4 — Spam SMS Detection
Classifies SMS messages as **SPAM** or **LEGITIMATE**. Text is normalised (URLs, phone
numbers, currency and numbers become placeholder tokens) and vectorised with TF-IDF, then
Multinomial Naive Bayes, Logistic Regression and a calibrated SVM are compared, with
selection by cross-validated spam F1. It includes a Streamlit app and 40 tests.
→ [Full README](CODSOFT_TASK4/README.md) · [Evaluation report](CODSOFT_TASK4/reports/evaluation_report.md)

---

## Repository structure

```
codsoft_tasks/
├── README.md                  # this file
├── CODSOFT_TASK1/             # Movie Genre Classification
│   ├── app.py  train.py  predict.py  requirements.txt  README.md
│   ├── src/  tests/  notebooks/
│   ├── models/                # genre_classifier.joblib + metadata
│   ├── reports/               # metrics, evaluation report, figures, screenshots
│   └── data/README.md         # where to place the raw IMDb dataset (not committed)
├── CODSOFT_TASK3/             # Customer Churn Prediction
│   ├── app.py  train.py  requirements.txt  README.md
│   ├── src/  tests/  notebooks/
│   ├── data/Churn_Modelling.csv
│   ├── models/churn_pipeline.joblib
│   └── reports/               # EDA report, model comparison, metrics, figures, screenshots
└── CODSOFT_TASK4/             # Spam SMS Detection
    ├── app.py  train.py  requirements.txt  README.md
    ├── src/  tests/  notebooks/
    ├── data/spam.csv
    ├── models/spam_classifier.joblib
    └── reports/               # evaluation report, metrics, figures, screenshots
```

---

## Requirements

- **Python 3.13** (the version used to train and test all three projects)
- `pip` and `venv`
- About 300 MB of disk space for each project's virtual environment

Each project has its own `requirements.txt`. Tasks 1 and 3 pin exact versions so that the
saved `joblib` models load with the same scikit-learn version they were trained with
(scikit-learn 1.9.1). **Use a separate virtual environment for each project.**

---

## Running a project

Clone the repository once:

```bash
git clone https://github.com/shivrajsinghtech/codsoft_tasks.git
cd codsoft_tasks
```

Then, for the project you want to run (replace `CODSOFT_TASK3` with `CODSOFT_TASK1` or
`CODSOFT_TASK4` as needed):

```bash
cd CODSOFT_TASK3
python -m venv .venv
source .venv/bin/activate        # Windows: .venv\Scripts\activate
pip install -r requirements.txt
```

| Action | Command (run inside the project folder) | Notes |
|---|---|---|
| Launch the web app | `streamlit run app.py` | Opens at http://localhost:8501 and uses the committed trained model |
| Run the tests | `python -m pytest -v` | No dataset download needed |
| Retrain from scratch | `python train.py` | Regenerates the model, metrics, figures and reports |
| Open the notebook | `jupyter notebook notebooks/` | Install `jupyter` separately |

**Task 1 extras**

```bash
# Predict from the command line
python predict.py --title "Red Horizon" --plot "In 2140 the first colony on Mars loses contact with Earth..."

# Batch prediction (needs the raw dataset, see below)
python predict.py --input "data/raw/Genre Classification Dataset/test_data.txt" --output predictions.csv
```

Retraining Task 1 requires the raw IMDb Genre Classification dataset (≈105 MB), which is
not committed. See [`CODSOFT_TASK1/data/README.md`](CODSOFT_TASK1/data/README.md) for where
to put it. The web app, CLI and tests all work without it.

---

## Testing

All three test suites were run from this repository's folders with Python 3.13:

| Project | Result |
|---|---|
| CODSOFT_TASK1 | 63 passed, 1 skipped (the skipped test needs the uncommitted raw dataset) |
| CODSOFT_TASK3 | 44 passed |
| CODSOFT_TASK4 | 40 passed |

The committed models were also loaded and used for a sample prediction in each project.

---

## Datasets

| Project | Dataset | Included? |
|---|---|---|
| Task 1 | Genre Classification Dataset (IMDb plain-text data dumps, distributed on Kaggle) | No (≈105 MB). See `CODSOFT_TASK1/data/README.md` |
| Task 3 | "Churn Modelling" bank-customer dataset (public, Kaggle) | Yes, unmodified. The identifier columns (`RowNumber`, `CustomerId`, `Surname`) are dropped before training |
| Task 4 | SMS Spam Collection v.1 (UCI Machine Learning Repository, CC BY 4.0) | Yes, unmodified |

Dataset sources, licences and citations are given in each project's README.

---

## Original repositories

Each project was first developed in its own repository. The folders here are exact copies
of those repositories' files at the time this collection was created:

- Task 1: https://github.com/shivrajsinghtech/CODSOFT_TASK1
- Task 3: https://github.com/shivrajsinghtech/CODSOFT_TASK3
- Task 4: https://github.com/shivrajsinghtech/CODSOFT_TASK4

Task 1 is released under the MIT License (see [`CODSOFT_TASK1/LICENSE`](CODSOFT_TASK1/LICENSE)).
