"""Train the baseline and logistic regression models."""

from sklearn.dummy import DummyClassifier
from sklearn.linear_model import LogisticRegression


def train_baseline(X_train_scaled, y_train):
    """Majority-class baseline, for comparison against the real model."""
    baseline = DummyClassifier(strategy="most_frequent")
    baseline.fit(X_train_scaled, y_train)
    return baseline


def train_logistic_regression(X_train_scaled, y_train, config: dict):
    """Train the logistic regression model, params driven by config['model']."""
    model_cfg = config["model"]
    log_reg = LogisticRegression(
        max_iter=model_cfg["max_iter"],
        random_state=model_cfg["random_state"],
    )
    log_reg.fit(X_train_scaled, y_train)
    return log_reg


def save_artifacts(model, scaler, models_dir: str = "models"):
    """Persist the trained model and fitted scaler to disk."""
    import joblib
    from pathlib import Path

    models_path = Path(models_dir)
    models_path.mkdir(parents=True, exist_ok=True)

    joblib.dump(model, models_path / "logistic_regression.joblib")
    joblib.dump(scaler, models_path / "scaler.joblib")
