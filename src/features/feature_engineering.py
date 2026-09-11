"""Split cleaned data into train/test sets and scale features.

An 80/20 stratified train/test split, followed by StandardScaler 
fit on the training set only (to avoid scaling leakage) and applied 
to both sets.
"""

import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler


def split_features_target(df: pd.DataFrame, target_column: str):
    """Separate features (X) from the target column (y)."""
    X = df.drop(columns=[target_column])
    y = df[target_column]
    return X, y


def split_train_test(X: pd.DataFrame, y: pd.Series, config: dict):
    """Stratified train/test split, parameters driven by config['split']."""
    split_cfg = config["split"]
    stratify_arg = y if split_cfg.get("stratify", True) else None

    return train_test_split(
        X,
        y,
        test_size=split_cfg["test_size"],
        random_state=split_cfg["random_state"],
        stratify=stratify_arg,
    )


def scale_features(X_train: pd.DataFrame, X_test: pd.DataFrame):
    """Fit StandardScaler on the training set only; transform both sets.

    Returns the fitted scaler alongside the scaled arrays so it can be
    persisted and reused at inference time on new, unseen input.
    """
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)
    return X_train_scaled, X_test_scaled, scaler
