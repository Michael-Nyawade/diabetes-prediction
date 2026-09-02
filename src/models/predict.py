"""Load persisted model artifacts and run inference on new input."""

from pathlib import Path

import joblib
import numpy as np
import pandas as pd

FEATURE_ORDER = [
    "Pregnancies",
    "Glucose",
    "BloodPressure",
    "SkinThickness",
    "Insulin",
    "BMI",
    "DiabetesPedigreeFunction",
    "Age",
]


def load_artifacts(models_dir: str = "models"):
    """Load the trained model and fitted scaler from disk."""
    models_path = Path(models_dir)
    model_path = models_path / "logistic_regression.joblib"
    scaler_path = models_path / "scaler.joblib"

    if not model_path.exists() or not scaler_path.exists():
        raise FileNotFoundError(
            f"Model artifacts not found in {models_path}. "
            "Run training (src.models.train) before predicting."
        )

    model = joblib.load(model_path)
    scaler = joblib.load(scaler_path)
    return model, scaler


def predict_one(features: dict, model, scaler) -> dict:
    """Predict diabetes risk for a single patient.

    Args:
        features: dict with keys matching FEATURE_ORDER, raw (unscaled) values.
        model: trained LogisticRegression instance.
        scaler: fitted StandardScaler instance.

    Returns:
        dict with 'prediction' (0 or 1) and 'probability' (float, P(Outcome=1)).
    """
    missing = [f for f in FEATURE_ORDER if f not in features]
    if missing:
        raise ValueError(f"Missing required features: {missing}")

    X = pd.DataFrame([features], columns=FEATURE_ORDER)
    X_scaled = scaler.transform(X)

    prediction = int(model.predict(X_scaled)[0])
    probability = float(model.predict_proba(X_scaled)[:, 1][0])

    return {"prediction": prediction, "probability": probability}
