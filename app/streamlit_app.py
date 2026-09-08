"""Streamlit UI for diabetes risk prediction.

Thin presentation layer only: collects input, calls into src.models.predict
for inference (and src.models.evaluate for coefficient display), and
renders the result. No preprocessing or training logic lives here.
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import pandas as pd
import streamlit as st

from src.models.predict import load_artifacts, predict_one, FEATURE_ORDER
from src.models.evaluate import coefficients_to_odds_ratios


st.set_page_config(
    page_title="Diabetes Risk Prediction",
    page_icon="🩺",
    layout="wide",
)


@st.cache_resource
def get_artifacts():
    return load_artifacts()


try:
    model, scaler = get_artifacts()
except FileNotFoundError as e:
    st.error(str(e))
    st.stop()


# ---------- Header ----------
st.title("Diabetes Risk Prediction")
st.markdown(
    "Estimate diabetes risk from clinical and demographic measurements, "
    "using a logistic regression model trained on the "
    "[Pima Indians Diabetes dataset](https://www.kaggle.com/datasets/mragpavank/diabetes)."
)
st.info(
    "**This project is for educational purposes only.** Not intended for clinical decision making.",
)

st.divider()


# ---------- Sidebar: input form ----------
with st.sidebar:
    st.header("Patient Measurements")
    st.caption("Enter the patient's clinical and demographic values below.")

    with st.form("prediction_form"):
        pregnancies = st.number_input(
            "Pregnancies", min_value=0, max_value=20, value=1, step=1,
            help="Number of times pregnant",
        )
        glucose = st.number_input(
            "Glucose (mg/dL)", min_value=0.0, max_value=300.0, value=120.0,
            help="Plasma glucose concentration (2-hour oral glucose tolerance test)",
        )
        blood_pressure = st.number_input(
            "Blood Pressure (mm Hg)", min_value=0.0, max_value=200.0, value=70.0,
            help="Diastolic blood pressure",
        )
        skin_thickness = st.number_input(
            "Skin Thickness (mm)", min_value=0.0, max_value=100.0, value=20.0,
            help="Triceps skinfold thickness",
        )
        insulin = st.number_input(
            "Insulin (mu U/ml)", min_value=0.0, max_value=900.0, value=80.0,
            help="2-hour serum insulin",
        )
        bmi = st.number_input(
            "BMI", min_value=0.0, max_value=70.0, value=28.0,
            help="Body mass index (weight in kg / (height in m)^2)",
        )
        dpf = st.number_input(
            "Diabetes Pedigree Function", min_value=0.0, max_value=3.0, value=0.5,
            format="%.3f", help="Family history-based diabetes risk score",
        )
        age = st.number_input(
            "Age (years)", min_value=1, max_value=120, value=33, step=1,
        )

        submitted = st.form_submit_button("Predict", type="primary", use_container_width=True)


# ---------- Main area: result ----------
result_col, info_col = st.columns([3, 2], gap="large")

with result_col:
    st.subheader("Prediction Result")

    if not submitted:
        st.markdown(
            "Fill in the patient's measurements in the sidebar and click "
            "**Predict** to see a risk estimate."
        )
    else:
        features = {
            "Pregnancies": pregnancies,
            "Glucose": glucose,
            "BloodPressure": blood_pressure,
            "SkinThickness": skin_thickness,
            "Insulin": insulin,
            "BMI": bmi,
            "DiabetesPedigreeFunction": dpf,
            "Age": age,
        }

        result = predict_one(features, model, scaler)
        probability = result["probability"]
        is_high_risk = result["prediction"] == 1

        metric_col, banner_col = st.columns([1, 2])
        with metric_col:
            st.metric("Estimated Probability", f"{probability:.1%}")
        with banner_col:
            if is_high_risk:
                st.warning("**Higher risk of diabetes**")
            else:
                st.success("**Lower risk of diabetes**")

        st.progress(probability)
        st.caption(
            "This probability reflects the model's estimate based on the "
            "patient measurements entered - it is not a diagnosis."
        )

with info_col:
    with st.expander("What drives this prediction?", expanded=False):
        st.write(
            "The model assigns each feature an **odds ratio**: values above "
            "1 increase estimated diabetes risk, values below 1 decrease it."
        )
        odds_df = coefficients_to_odds_ratios(model, pd.Index(FEATURE_ORDER))
        st.bar_chart(
            odds_df.set_index("Feature")["Odds Ratio (exp(coeff))"],
            horizontal=True,
        )
        st.caption(
            "Glucose and BMI are consistently the strongest predictors in "
            "this model, consistent with established medical research."
        )

    with st.expander("About this model", expanded=False):
        st.markdown(
            "- **Model:** Logistic Regression\n"
            "- **Test accuracy:** ~70.8%\n"
            "- **ROC-AUC:** ~0.81\n"
            "- **Dataset:** 768 patient records, Pima Indians Diabetes dataset\n"
        )

    with st.expander("About the Variables", expanded=False):
        st.markdown(
            "- **Pregnancies:** Number of times pregnant\n"
            "- **Glucose:** Plasma glucose concentration, measured via a 2-hour oral glucose tolerance test (mg/dL)\n"
            "- **Blood Pressure:** Diastolic blood pressure (mm Hg)\n"
            "- **Skin Thickness:** Triceps skinfold thickness, a measure of body fat (mm)\n"
            "- **Insulin:** 2-hour serum insulin level (mu U/ml)\n"
            "- **BMI:** Body mass index - weight in kg divided by height in m squared\n"
            "- **Diabetes Pedigree Function:** A score estimating diabetes risk based on family history\n"
            "- **Age:** Age in years\n"
        )

