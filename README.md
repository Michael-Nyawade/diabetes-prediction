# 🩺 Diabetes Prediction Using Logistic Regression

## 📌 Project Overview

This project applies exploratory data analysis (EDA), feature engineering, and machine learning to predict the likelihood of diabetes using health-related attributes. The goal is to build an interpretable yet effective predictive model and extract meaningful insights that align with known medical risk factors.

The analysis is performed in Python using Jupyter Notebook and leverages the publicly available **Pima Indians Diabetes dataset**, commonly used for research, analytics, and educational purposes.

---

## 🎯 Objectives

- Perform data cleaning and exploratory data analysis
- Handle missing or invalid health feature values appropriately
- Build a baseline model for performance comparison
- Train an interpretable Logistic Regression model
- Evaluate results using classification metrics and ROC-AUC
- Interpret coefficients to understand key risk predictors

---

## 🧠 Key Findings

- **Glucose level** was the strongest predictor of diabetes, with an odds ratio of approximately **3.26**, meaning higher glucose levels significantly increase risk.
- **BMI** and **Pregnancies** also played meaningful roles in predicting diabetes.
- The Logistic Regression model achieved:
  - **Accuracy:** ~70.8%
  - **ROC-AUC:** ~0.81
  - A substantial improvement over the majority-class baseline accuracy of ~64.9%

These findings align with existing medical knowledge, demonstrating both **validity** and **interpretability**.

---

## 🗂 Dataset Information

The dataset is publicly available from Kaggle and the UCI Machine Learning Repository.

- Features include: _Glucose, BMI, Insulin, Age, Skin Thickness, Blood Pressure, Pregnancies, Diabetes Pedigree Function_
- Target label: **Outcome** (1 = diabetic, 0 = non-diabetic)

📎 Dataset reference link:  
https://www.kaggle.com/datasets/uciml/pima-indians-diabetes-database

---

🛡 Disclaimer

This project is for educational purposes only and not intended for clinical decision making.
