"""Tests for the full training and prediction pipeline."""

import numpy as np
import pytest

from src.data.load_data import load_config, load_data
from src.data.preprocess import clean_data
from src.features.feature_engineering import (
    split_features_target,
    split_train_test,
    scale_features,
)
from src.models.train import train_baseline, train_logistic_regression
from src.models.evaluate import evaluate_model
from src.models.predict import predict_one, FEATURE_ORDER


@pytest.fixture(scope="module")
def trained_pipeline():
    """Run the full pipeline once and share results across tests in this module."""
    config = load_config()
    df = load_data(config)
    df_clean = clean_data(df, config["data"]["invalid_zero_columns"])
    X, y = split_features_target(df_clean, config["data"]["target_column"])
    X_train, X_test, y_train, y_test = split_train_test(X, y, config)
    X_train_scaled, X_test_scaled, scaler = scale_features(X_train, X_test)

    baseline = train_baseline(X_train_scaled, y_train)
    log_reg = train_logistic_regression(X_train_scaled, y_train, config)

    return {
        "baseline": baseline,
        "log_reg": log_reg,
        "scaler": scaler,
        "X_test": X_test,
        "X_test_scaled": X_test_scaled,
        "y_test": y_test,
    }


def test_split_produces_known_class_balance(trained_pipeline):
    counts = trained_pipeline["y_test"].value_counts()
    assert counts[0] == 100
    assert counts[1] == 54


def test_logistic_regression_matches_original_accuracy(trained_pipeline):
    results = evaluate_model(
        trained_pipeline["log_reg"],
        trained_pipeline["X_test_scaled"],
        trained_pipeline["y_test"],
    )
    assert results["accuracy"] == pytest.approx(0.7077922077922078, abs=1e-9)
    assert results["roc_auc"] == pytest.approx(0.812962962962963, abs=1e-9)


def test_logistic_regression_beats_baseline(trained_pipeline):
    baseline_results = evaluate_model(
        trained_pipeline["baseline"],
        trained_pipeline["X_test_scaled"],
        trained_pipeline["y_test"],
        predict_proba=False,
    )
    lr_results = evaluate_model(
        trained_pipeline["log_reg"],
        trained_pipeline["X_test_scaled"],
        trained_pipeline["y_test"],
    )
    assert lr_results["accuracy"] > baseline_results["accuracy"]


def test_predict_one_returns_valid_output(trained_pipeline):
    row = trained_pipeline["X_test"].iloc[0]
    features = {col: row[col] for col in FEATURE_ORDER}

    result = predict_one(features, trained_pipeline["log_reg"], trained_pipeline["scaler"])

    assert result["prediction"] in (0, 1)
    assert 0.0 <= result["probability"] <= 1.0
    assert (result["probability"] > 0.5) == (result["prediction"] == 1)


def test_predict_one_raises_on_missing_feature(trained_pipeline):
    incomplete_features = {"Pregnancies": 1, "Glucose": 120}
    with pytest.raises(ValueError):
        predict_one(incomplete_features, trained_pipeline["log_reg"], trained_pipeline["scaler"])
