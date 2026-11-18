# Model Comparison Report

Dataset: `10,000` customers after cleaning; stratified 80/20 split (train = 8,000, test = 2,000), `random_state=42`.

Hyper-parameters were tuned with 5-fold stratified cross-validation on the **training set only**, selecting on **F1** of the churn class. Class imbalance was handled with `class_weight='balanced'` (Logistic Regression, Random Forest) and balanced sample weights (Gradient Boosting). The test set was used once, for the final evaluation below.

## Cross-validation (training set, mean ± std)

| Model | F1 | Recall | Precision | ROC-AUC | Accuracy | Best params |
|---|---|---|---|---|---|---|
| Logistic Regression | 0.4932 ± 0.0226 | 0.6859 ± 0.0408 | 0.3853 ± 0.0170 | 0.7673 ± 0.0204 | 0.7130 ± 0.0132 | C=0.01 |
| Random Forest | 0.6117 ± 0.0246 | 0.6534 ± 0.0453 | 0.5759 ± 0.0111 | 0.8537 ± 0.0117 | 0.8315 ± 0.0064 | max_depth=12, min_samples_leaf=1, n_estimators=300 |
| Gradient Boosting | 0.6077 ± 0.0171 | 0.7190 ± 0.0373 | 0.5267 ± 0.0095 | 0.8603 ± 0.0133 | 0.8111 ± 0.0054 | learning_rate=0.1, max_depth=3, n_estimators=300 |

## Held-out test set

| Model | Accuracy | Precision | Recall | F1 | ROC-AUC | TN | FP | FN | TP |
|---|---|---|---|---|---|---|---|---|---|
| Logistic Regression | 0.7180 | 0.3926 | 0.7052 | 0.5044 | 0.7771 | 1149 | 444 | 120 | 287 |
| Random Forest | 0.8320 | 0.5747 | 0.6708 | 0.6190 | 0.8575 | 1391 | 202 | 134 | 273 |
| Gradient Boosting | 0.8085 | 0.5207 | 0.7420 | 0.6120 | 0.8588 | 1315 | 278 | 105 | 302 |

## Selected model: **Random Forest**

Random Forest achieved the highest mean cross-validated F1 for the churn class (0.6117), with CV recall 0.6534 and CV ROC-AUC 0.8537. F1 was used because the business cost is two-sided: missing a churner (false negative) loses a customer, while flagging a loyal customer (false positive) wastes retention budget. The selection was made on cross-validation scores; test-set scores are reported only as an unbiased final check.

### Classification report (test set)

```
              precision    recall  f1-score   support

      Stayed       0.91      0.87      0.89      1593
     Churned       0.57      0.67      0.62       407

    accuracy                           0.83      2000
   macro avg       0.74      0.77      0.76      2000
weighted avg       0.84      0.83      0.84      2000
```

### Permutation feature importance (test set, drop in ROC-AUC)

| Feature | Importance | Std |
|---|---|---|
| Age | 0.1183 | 0.0055 |
| NumOfProducts | 0.1141 | 0.0102 |
| Balance | 0.0391 | 0.0058 |
| IsActiveMember | 0.0283 | 0.0034 |
| Geography | 0.0262 | 0.0026 |
| Gender | 0.0082 | 0.0028 |
| EstimatedSalary | -0.0006 | 0.0017 |
| CreditScore | -0.0007 | 0.0011 |
| Tenure | -0.0010 | 0.0011 |
| HasCrCard | -0.0014 | 0.0008 |

## Figures

- `figures/11_model_comparison.png`
- `figures/12_confusion_matrices.png`
- `figures/13_roc_curves.png`
- `figures/14_feature_importance.png`
- `figures/15_best_model_confusion_matrix.png`
