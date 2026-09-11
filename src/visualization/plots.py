"""Reusable plotting functions for exploratory data analysis.

Each function returns a matplotlib Figure and optionally saves it to disk,
so notebooks can both display figures inline and persist them to
reports/figures/ without duplicating plotting logic.
"""

from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns


def _save_if_path(fig, save_path):
    if save_path is not None:
        Path(save_path).parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(save_path, bbox_inches="tight")


def plot_outcome_distribution(df: pd.DataFrame, target_column: str, save_path=None):
    """Bar chart of class counts for the target column."""
    fig, ax = plt.subplots(figsize=(6, 4))
    sns.countplot(x=target_column, data=df, ax=ax)
    ax.set_title("Outcome Distribution")
    ax.set_xlabel("Outcome (0 = Non-diabetic, 1 = Diabetic)")
    ax.set_ylabel("Count")
    _save_if_path(fig, save_path)
    return fig


def plot_feature_distributions(df: pd.DataFrame, feature_columns: list[str], save_path=None):
    """Grid of histograms, one per feature."""
    n_cols = 3
    n_rows = -(-len(feature_columns) // n_cols)  # ceiling division
    fig, axes = plt.subplots(n_rows, n_cols, figsize=(15, 4 * n_rows))
    axes = axes.flatten()

    for i, col in enumerate(feature_columns):
        sns.histplot(df[col], kde=True, ax=axes[i])
        axes[i].set_title(col)

    # Hide any unused subplot axes when feature count isn't a multiple of n_cols
    for j in range(len(feature_columns), len(axes)):
        axes[j].set_visible(False)

    fig.suptitle("Feature Distributions", y=1.02)
    fig.tight_layout()
    _save_if_path(fig, save_path)
    return fig


def plot_correlation_heatmap(df: pd.DataFrame, save_path=None):
    """Correlation heatmap across all numeric columns."""
    fig, ax = plt.subplots(figsize=(9, 7))
    sns.heatmap(df.corr(), annot=True, fmt=".2f", cmap="coolwarm", ax=ax)
    ax.set_title("Feature Correlation Heatmap")
    _save_if_path(fig, save_path)
    return fig


def plot_boxplots_by_outcome(df: pd.DataFrame, feature_columns: list[str], target_column: str, save_path=None):
    """Grid of boxplots, one per feature, split by target class."""
    n_cols = 3
    n_rows = -(-len(feature_columns) // n_cols)
    fig, axes = plt.subplots(n_rows, n_cols, figsize=(15, 4 * n_rows))
    axes = axes.flatten()

    for i, col in enumerate(feature_columns):
        sns.boxplot(x=target_column, y=col, data=df, ax=axes[i])
        axes[i].set_title(col)

    for j in range(len(feature_columns), len(axes)):
        axes[j].set_visible(False)

    fig.suptitle("Feature Distributions by Outcome", y=1.02)
    fig.tight_layout()
    _save_if_path(fig, save_path)
    return fig
