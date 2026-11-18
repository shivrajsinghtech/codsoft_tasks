# Exploratory Data Analysis Report

## Data quality

- Rows: **10,000**, columns after dropping identifiers: **11**
- Missing values: **0**
- Duplicate rows: **0**
- Identifier columns removed: `RowNumber`, `CustomerId`, `Surname` (no predictive meaning; `Surname` would also risk spurious memorisation).

## Class balance

- Churned: **2,037** (20.4%), stayed: **7,963** (79.6%).
- The data is imbalanced (~1 : 4), so accuracy alone is misleading - a model that always predicts "stay" would score ~80%. Recall, precision, F1 and ROC-AUC are reported.

## Key findings

| Segment | Churn rate |
|---|---|
| Geography = France | 16.2% |
| Geography = Germany | 32.4% |
| Geography = Spain | 16.7% |
| Gender = Female | 25.1% |
| Gender = Male | 16.5% |
| IsActiveMember = 0 | 26.9% |
| IsActiveMember = 1 | 14.3% |
| NumOfProducts = 1 | 27.7% |
| NumOfProducts = 2 | 7.6% |
| NumOfProducts = 3 | 82.7% |
| NumOfProducts = 4 | 100.0% |
| Age 18-30 | 7.5% |
| Age 31-40 | 12.1% |
| Age 41-50 | 34.0% |
| Age 51-60 | 56.2% |
| Age 60+ | 24.8% |
| Balance = 0 | 13.8% |
| Balance > 0 | 24.1% |

- Mean age: churned **44.8** vs stayed **37.4**.

### Correlation with `Exited` (Pearson, numeric columns)

| Feature | r |
|---|---|
| Age | +0.285 |
| IsActiveMember | -0.156 |
| Balance | +0.118 |
| NumOfProducts | -0.048 |
| CreditScore | -0.027 |
| Tenure | -0.014 |
| EstimatedSalary | +0.012 |
| HasCrCard | -0.007 |

Note: Pearson correlation only captures linear, monotonic relationships. `NumOfProducts` has a weak linear correlation but a strong non-linear effect (churn is lowest at 2 products and very high at 3-4), which tree-based models can capture.

## Figures (`reports/figures/`)

- `figures/01_churn_distribution.png`
- `figures/02_demographics.png`
- `figures/03_age_vs_churn.png`
- `figures/04_geography_vs_churn.png`
- `figures/05_gender_vs_churn.png`
- `figures/06_credit_score.png`
- `figures/07_balance_salary.png`
- `figures/08_tenure_vs_churn.png`
- `figures/09_activity_products.png`
- `figures/10_correlation_heatmap.png`
