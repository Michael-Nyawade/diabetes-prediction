"""Evaluate trained models and interpret logistic regression coefficients."""

import numpy as np
import pandas as pd
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
    roc_auc_score,
)


def evaluate_model(model, X_test_scaled, y_test, predict_proba: bool = True) -> dict:
    """Compute accuracy, classification report, confusion matrix, and ROC-AUC."""
    y_pred = model.predict(X_test_scaled)

    results = {
        "accuracy": accuracy_score(y_test, y_pred),
        "classification_report": classification_report(
            y_test, y_pred, zero_division=0
        ),
        "confusion_matrix": confusion_matrix(y_test, y_pred),
    }

    if predict_proba:
        y_proba = model.predict_proba(X_test_scaled)[:, 1]
        results["roc_auc"] = roc_auc_score(y_test, y_proba)

    return results


def coefficients_to_odds_ratios(model, feature_names) -> pd.DataFrame:
    """Convert logistic regression coefficients to odds ratios, ranked."""
    coefficients = model.coef_[0]
    coef_df = pd.DataFrame(
        {
            "Feature": feature_names,
            "Coefficient": coefficients,
            "Odds Ratio (exp(coeff))": np.exp(coefficients),
        }
    ).sort_values(by="Odds Ratio (exp(coeff))", ascending=False)
    coef_df.reset_index(drop=True, inplace=True)
    return coef_df
