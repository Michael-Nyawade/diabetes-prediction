"""Streamlit UI for diabetes risk prediction.

Thin presentation layer only: collects input, calls into src.models.predict
for inference, and displays the result. No preprocessing, training, or
business logic lives here.
"""

import sys
from pathlib import Path

# Allow running via `streamlit run app/streamlit_app.py` from the repo root
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import streamlit as st

from src.models.predict import load_artifacts, predict_one, FEATURE_ORDER


st.set_page_config(page_title="Diabetes Risk Prediction", page_icon="🩺")

st.title("Diabetes Risk Prediction")
st.write(
    "Estimate diabetes risk from clinical and demographic measurements, "
    "using a logistic regression model trained on the Pima Indians "
    "Diabetes dataset."
)
st.caption(
    "⚠️ Educational project only. Not intended for clinical decision making."
)


@st.cache_resource
def get_artifacts():
    return load_artifacts()


try:
    model, scaler = get_artifacts()
except FileNotFoundError as e:
    st.error(str(e))
    st.stop()

st.subheader("Patient Measurements")

col1, col2 = st.columns(2)

with col1:
    pregnancies = st.number_input("Pregnancies", min_value=0, max_value=20, value=1, step=1)
    glucose = st.number_input("Glucose (mg/dL)", min_value=0.0, max_value=300.0, value=120.0)
    blood_pressure = st.number_input("Blood Pressure (mm Hg)", min_value=0.0, max_value=200.0, value=70.0)
    skin_thickness = st.number_input("Skin Thickness (mm)", min_value=0.0, max_value=100.0, value=20.0)

with col2:
    insulin = st.number_input("Insulin (mu U/ml)", min_value=0.0, max_value=900.0, value=80.0)
    bmi = st.number_input("BMI", min_value=0.0, max_value=70.0, value=28.0)
    dpf = st.number_input("Diabetes Pedigree Function", min_value=0.0, max_value=3.0, value=0.5, format="%.3f")
    age = st.number_input("Age (years)", min_value=1, max_value=120, value=33, step=1)

if st.button("Predict", type="primary"):
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

    st.subheader("Result")
    if result["prediction"] == 1:
        st.warning(f"**Higher risk of diabetes** - estimated probability: {result['probability']:.1%}")
    else:
        st.success(f"**Lower risk of diabetes** - estimated probability: {result['probability']:.1%}")

    st.progress(result["probability"])
