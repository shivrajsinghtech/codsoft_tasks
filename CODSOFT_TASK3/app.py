"""Customer Churn Prediction dashboard (Streamlit).

Run with:  streamlit run app.py
"""
from __future__ import annotations

import json

import pandas as pd
import streamlit as st

from src.config import FIGURES_DIR, METRICS_PATH, MODEL_PATH
from src.predict import (
    GENDERS, GEOGRAPHIES, NUMERIC_RULES, InvalidInputError, load_model, predict_customer,
)

st.set_page_config(page_title="Customer Churn Prediction", page_icon="📉", layout="wide")


@st.cache_resource(show_spinner="Loading trained model...")
def get_model():
    return load_model(MODEL_PATH)


@st.cache_data
def get_metrics() -> dict | None:
    if not METRICS_PATH.exists():
        return None
    return json.loads(METRICS_PATH.read_text(encoding="utf-8"))


def show_figure(name: str, caption: str | None = None) -> None:
    path = FIGURES_DIR / name
    if path.exists():
        st.image(str(path), caption=caption, width="stretch")
    else:
        st.info(f"Figure `{name}` not found. Run `python train.py` to generate it.")


st.title("📉 Customer Churn Prediction")
st.caption("CodSoft Machine Learning Internship · Task 3 · Bank customer churn "
           "(Churn_Modelling dataset, 10,000 customers)")

try:
    model = get_model()
    model_error = None
except Exception as exc:  # missing or corrupt model file
    model, model_error = None, str(exc)
metrics = get_metrics()

if model_error:
    st.error(f"The prediction model could not be loaded: {model_error}")

if metrics:
    best = metrics["best_model"]
    t = metrics["models"][best]["test"]
    st.markdown(f"**Deployed model:** {best} (selected by 5-fold cross-validated churn F1)")
    c = st.columns(5)
    c[0].metric("Accuracy", f"{t['accuracy']:.1%}")
    c[1].metric("Churn recall", f"{t['recall']:.1%}")
    c[2].metric("Churn precision", f"{t['precision']:.1%}")
    c[3].metric("F1 (churn)", f"{t['f1']:.3f}")
    c[4].metric("ROC-AUC", f"{t['roc_auc']:.3f}")
    st.caption("Metrics are measured on a held-out stratified test set "
               f"({metrics['split']['test_rows']:,} customers) never used during training or model selection.")
else:
    st.warning("Model metrics not found. Run `python train.py` to train the models.")

tab_predict, tab_perf, tab_eda, tab_insights, tab_about = st.tabs(
    ["🔮 Predict", "📊 Model performance", "🔍 Data exploration", "💡 Business insights", "ℹ️ About"]
)

# ------------------------------------------------------------------ Predict
with tab_predict:
    st.subheader("Customer information")
    with st.form("customer_form"):
        c1, c2, c3 = st.columns(3)
        with c1:
            credit_score = st.number_input(
                "Credit score", min_value=int(NUMERIC_RULES["CreditScore"][0]),
                max_value=int(NUMERIC_RULES["CreditScore"][1]), value=650, step=1,
                help="Typical range 350-850.")
            geography = st.selectbox("Geography", GEOGRAPHIES, index=None,
                                     placeholder="Select country")
            gender = st.selectbox("Gender", GENDERS, index=None, placeholder="Select gender")
            age = st.number_input("Age", min_value=18, max_value=100, value=38, step=1)
        with c2:
            tenure = st.number_input("Tenure (years with bank)", min_value=0, max_value=10,
                                     value=5, step=1)
            balance = st.number_input("Account balance", min_value=0.0, max_value=1_000_000.0,
                                      value=75_000.0, step=1_000.0, format="%.2f")
            num_products = st.selectbox("Number of products", [1, 2, 3, 4], index=1)
        with c3:
            has_card = st.radio("Has credit card?", ["Yes", "No"], horizontal=True)
            active = st.radio("Active member?", ["Yes", "No"], horizontal=True)
            salary = st.number_input("Estimated salary", min_value=0.0, max_value=1_000_000.0,
                                     value=100_000.0, step=1_000.0, format="%.2f")
        submitted = st.form_submit_button("Predict churn", type="primary",
                                          width="stretch")

    if submitted:
        customer = {
            "CreditScore": credit_score, "Geography": geography, "Gender": gender,
            "Age": age, "Tenure": tenure, "Balance": balance,
            "NumOfProducts": num_products, "HasCrCard": 1 if has_card == "Yes" else 0,
            "IsActiveMember": 1 if active == "Yes" else 0, "EstimatedSalary": salary,
        }
        if model is None:
            st.error("Cannot predict: the model is not available. Run `python train.py`.")
        else:
            try:
                result = predict_customer(model, customer)
            except InvalidInputError as exc:
                st.error("Please correct the following input problems:")
                for e in exc.errors:
                    st.markdown(f"- {e}")
            except Exception as exc:  # unexpected failure - never crash the app
                st.error(f"Prediction failed unexpectedly: {exc}")
            else:
                p = result["churn_probability"]
                r1, r2, r3 = st.columns(3)
                if result["will_churn"]:
                    r1.error(f"### ⚠️ {result['label']}")
                else:
                    r1.success(f"### ✅ {result['label']}")
                r2.metric("Churn probability", f"{p:.1%}")
                r3.metric("Prediction confidence", f"{result['confidence']:.1%}")
                st.progress(p, text=f"Estimated churn risk: {p:.1%}")
                risk = "High" if p >= 0.7 else "Medium" if p >= 0.4 else "Low"
                st.markdown(f"**Risk band:** {risk}")
                tips = []
                if customer["IsActiveMember"] == 0:
                    tips.append("Customer is inactive - engagement campaigns are the strongest lever.")
                if customer["NumOfProducts"] >= 3:
                    tips.append("Holds 3+ products - historically this group churns heavily; review product fit.")
                if customer["NumOfProducts"] == 1:
                    tips.append("Holds a single product - cross-selling a second product is associated with lower churn.")
                if customer["Age"] >= 45:
                    tips.append("Age 45+ - this segment shows elevated churn in the historical data.")
                if customer["Geography"] == "Germany":
                    tips.append("German customers churn at roughly twice the rate of France/Spain.")
                if tips:
                    st.markdown("**Observed risk factors (from historical data):**")
                    for tip in tips:
                        st.markdown(f"- {tip}")
    st.info("ℹ️ Predictions are statistical **estimates** based on historical patterns, "
            "not guarantees of customer behaviour. Use them to prioritise retention "
            "efforts alongside human judgement.")

# ------------------------------------------------------------------ Performance
with tab_perf:
    if metrics:
        st.subheader("Model comparison")
        rows = []
        for name, r in metrics["models"].items():
            rows.append({
                "Model": name,
                "CV F1": r["cv"]["f1"]["mean"], "CV Recall": r["cv"]["recall"]["mean"],
                "CV ROC-AUC": r["cv"]["roc_auc"]["mean"],
                "Test Accuracy": r["test"]["accuracy"], "Test Precision": r["test"]["precision"],
                "Test Recall": r["test"]["recall"], "Test F1": r["test"]["f1"],
                "Test ROC-AUC": r["test"]["roc_auc"],
            })
        st.dataframe(pd.DataFrame(rows).set_index("Model").style.format("{:.4f}")
                     .highlight_max(axis=0, color="#d4edda"), width="stretch")
        st.markdown(f"**Selected: {metrics['best_model']}.** {metrics['selection_rationale']}")
        c1, c2 = st.columns(2)
        with c1:
            show_figure("15_best_model_confusion_matrix.png", "Confusion matrix - selected model")
        with c2:
            show_figure("14_feature_importance.png", "Permutation feature importance")
        show_figure("11_model_comparison.png")
        c1, c2 = st.columns(2)
        with c1:
            show_figure("13_roc_curves.png")
        with c2:
            show_figure("12_confusion_matrices.png")
        st.subheader("Classification report (selected model, test set)")
        st.code(metrics["models"][metrics["best_model"]]["test"]["classification_report_text"])
    else:
        st.warning("No metrics available. Run `python train.py`.")

# ------------------------------------------------------------------ EDA
with tab_eda:
    st.subheader("Exploratory data analysis")
    figs = [
        ("01_churn_distribution.png", "Churn distribution"),
        ("02_demographics.png", "Customer demographics"),
        ("03_age_vs_churn.png", "Age vs churn"),
        ("04_geography_vs_churn.png", "Geography vs churn"),
        ("05_gender_vs_churn.png", "Gender vs churn"),
        ("06_credit_score.png", "Credit score analysis"),
        ("07_balance_salary.png", "Balance & salary analysis"),
        ("08_tenure_vs_churn.png", "Tenure vs churn"),
        ("09_activity_products.png", "Activity, products & credit card"),
        ("10_correlation_heatmap.png", "Correlation analysis"),
    ]
    for i in range(0, len(figs), 2):
        cols = st.columns(2)
        for col, (f, cap) in zip(cols, figs[i:i + 2]):
            with col:
                show_figure(f, cap)

# ------------------------------------------------------------------ Insights
with tab_insights:
    st.subheader("Business insights")
    if metrics and metrics.get("eda"):
        e = metrics["eda"]
        g, a, pr = e["churn_rate_by_geography"], e["churn_rate_by_active"], e["churn_rate_by_products"]
        st.markdown(f"""
- **Overall churn is {e['churn_rate']:.1%}** - roughly 1 in 5 customers left.
- **Age is the strongest driver.** Churners average {e['mean_age']['Churned']} years vs
  {e['mean_age']['Stayed']} for customers who stayed; churn peaks in the 51-60 band
  ({e['churn_rate_by_age_band'].get('51-60', 0):.1%}).
- **Product holding matters non-linearly.** Customers with 2 products churn at only
  {pr['2']:.1%}, versus {pr['1']:.1%} with 1 product and {pr['3']:.1%} / {pr['4']:.1%} with 3 / 4.
- **Inactive members churn almost twice as often** ({a['0']:.1%} vs {a['1']:.1%}).
- **Germany is a hotspot** ({g['Germany']:.1%}) vs France ({g['France']:.1%}) and Spain ({g['Spain']:.1%}).
- **Female customers churn more** ({e['churn_rate_by_gender']['Female']:.1%}) than male
  ({e['churn_rate_by_gender']['Male']:.1%}).
- Credit score, tenure, salary and credit-card ownership show little difference between groups.
""")
        st.markdown("""
**Recommended actions**
1. Prioritise re-engagement of inactive members, especially aged 40-65.
2. Encourage single-product customers to adopt a second product; review why 3-4 product holders leave.
3. Investigate the German market (pricing, service, competition).
4. Use the model's probability to rank customers and focus limited retention budget on the highest-risk group.
""")
    else:
        st.info("Run `python train.py` to compute insights.")

# ------------------------------------------------------------------ About
with tab_about:
    st.markdown("""
**Objective:** predict whether a bank customer will leave (`Exited = 1`) from demographics
and account-usage behaviour.

**Pipeline:** identifier columns removed → `ColumnTransformer` (median-impute + standard-scale
numeric features, pass binary flags, one-hot encode Geography/Gender) → classifier.
The whole pipeline is fitted inside cross-validation folds, so no information leaks from validation data.

**Models compared:** Logistic Regression, Random Forest and Gradient Boosting, each tuned with
5-fold stratified grid-search CV and class-imbalance weighting. The model with the best
cross-validated churn F1 is deployed.

**Limitations:** the data is a single historical snapshot from one bank in three countries;
predictions are estimates and should not be used as the sole basis for decisions about a customer.

Source code: [github.com/shivrajsinghtech/CODSOFT_TASK3](https://github.com/shivrajsinghtech/CODSOFT_TASK3)
""")
