# Diabetes Risk Prediction

A machine learning project that predicts diabetes risk using logistic
regression and interpretable health analytics, built as a reproducible,
testable, and deployable Python project.

## Project Overview

This project applies exploratory data analysis, data cleaning, and logistic
regression to predict the likelihood of diabetes from clinical and
demographic measurements.  

## Objectives

- Perform data cleaning and exploratory data analysis
- Handle missing or invalid health feature values appropriately
- Build a baseline model for performance comparison
- Train an interpretable logistic regression model
- Evaluate results using classification metrics and ROC-AUC
- Interpret coefficients to understand key risk predictors
- Provide a reproducible, testable, and deployable implementation

## Key Findings

- **Glucose** is the strongest predictor of diabetes (odds ratio ≈ 3.26).
- **BMI** (≈ 1.99) and **Pregnancies** (≈ 1.46) are also meaningful predictors.
- The logistic regression model achieves:
  - **Accuracy:** ~70.8%
  - **ROC-AUC:** ~0.81
  - A substantial improvement over the majority-class baseline (~65%)

## Dataset

This project uses the Pima Indians Diabetes dataset:
[`mragpavank/diabetes`](https://www.kaggle.com/datasets/mragpavank/diabetes)
on Kaggle.

- 768 observations, 8 clinical/demographic features, 1 binary target
  (`Outcome`: 1 = diabetic, 0 = non-diabetic)
- Features: `Pregnancies`, `Glucose`, `BloodPressure`, `SkinThickness`,
  `Insulin`, `BMI`, `DiabetesPedigreeFunction`, `Age`

The dataset is not included in this repository (see `.gitignore`). Download
it from the link above and place it at `data/raw/diabetes.csv` before
running training.

## Project Structure

```bash
.
├── app/
│ └── streamlit_app.py                  # Streamlit UI (thin layer over src/models/predict.py)
├── config.yaml                         # data paths, split, and model parameters
├── data/
│ ├── raw/                              # place diabetes.csv here (not tracked in git)
│ └── processed/                        
├── main.py                             # CLI entry point (train / predict)
├── models/                             # trained model + scaler artifacts (not tracked in git)
├── notebooks/                          # exploratory notebooks, calling into src/
│ ├── 01_data_cleaning.ipynb
│ ├── 02_exploratory_data_analysis.ipynb
│ ├── 03_feature_engineering.ipynb
│ └── 04_modeling.ipynb
├── presentation/                       # slide deck summarizing the project
├── requirements.txt
├── src/
│ ├── data/                             # load_data.py, preprocess.py
│ ├── features/                         # feature_engineering.py (split + scale)
│ ├── models/                           # train.py, predict.py, evaluate.py
│ └── visualization/                    # plots.py
└── tests/
├── test_preprocessing.py
└── test_model.py
```


## Setup

### Clone the repository

```bash
git clone https://github.com/Michael-Nyawade/diabetes-prediction.git
cd diabetes-prediction
```

### Linux / macOS

```bash
python3 -m venv venv
source venv/bin/activate
pip install -r requirements.txt
```

### Windows

```powershell
python -m venv venv
venv\Scripts\Activate
pip install -r requirements.txt
```

Download the dataset (see [Dataset](#dataset) above) and place it at
`data/raw/diabetes.csv`.

## Usage

### Train the model

```bash
python main.py train
```

Trains the logistic regression model, prints evaluation metrics, and saves
the model and scaler to `models/`.

### Predict for a single patient (CLI)

```bash
python main.py predict --pregnancies 2 --glucose 130 --blood-pressure 70 \
    --skin-thickness 25 --insulin 100 --bmi 28.5 --dpf 0.45 --age 33
```

### Run the Streamlit app

```bash
streamlit run app/streamlit_app.py
```

Opens an interactive form in your browser for entering patient measurements
and viewing the predicted diabetes risk.

### Run tests

```bash
python -m pytest tests/ -v
```

## Disclaimer

This project is for educational purposes only and is not intended for
clinical decision making.