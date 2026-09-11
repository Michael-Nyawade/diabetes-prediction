"""Tests for data loading and cleaning logic."""

import pandas as pd
import pytest

from src.data.preprocess import clean_data


INVALID_ZERO_COLUMNS = ["Glucose", "BloodPressure", "SkinThickness", "Insulin", "BMI"]


@pytest.fixture
def raw_sample_df():
    """A small synthetic dataframe with zeros in clinically-invalid columns."""
    return pd.DataFrame(
        {
            "Pregnancies": [1, 2, 3, 4],
            "Glucose": [120, 0, 130, 140],
            "BloodPressure": [70, 80, 0, 75],
            "SkinThickness": [20, 25, 30, 0],
            "Insulin": [100, 0, 150, 120],
            "BMI": [28.0, 30.0, 0.0, 26.0],
            "DiabetesPedigreeFunction": [0.5, 0.6, 0.3, 0.4],
            "Age": [25, 30, 35, 40],
            "Outcome": [0, 1, 0, 1],
        }
    )


def test_clean_data_replaces_zeros_with_median(raw_sample_df):
    cleaned = clean_data(raw_sample_df, INVALID_ZERO_COLUMNS)

    # No zeros should remain in the invalid-zero columns.
    for col in INVALID_ZERO_COLUMNS:
        assert (cleaned[col] == 0).sum() == 0

    # Glucose had one zero (row 1); it should be replaced with the
    # median of the non-zero values: median(120, 130, 140) = 130.
    assert cleaned.loc[1, "Glucose"] == 130


def test_clean_data_no_missing_values_after_cleaning(raw_sample_df):
    cleaned = clean_data(raw_sample_df, INVALID_ZERO_COLUMNS)
    assert cleaned.isnull().sum().sum() == 0


def test_clean_data_does_not_mutate_input(raw_sample_df):
    original = raw_sample_df.copy()
    clean_data(raw_sample_df, INVALID_ZERO_COLUMNS)
    pd.testing.assert_frame_equal(raw_sample_df, original)


def test_clean_data_leaves_other_columns_untouched(raw_sample_df):
    cleaned = clean_data(raw_sample_df, INVALID_ZERO_COLUMNS)
    pd.testing.assert_series_equal(cleaned["Pregnancies"], raw_sample_df["Pregnancies"])
    pd.testing.assert_series_equal(cleaned["Outcome"], raw_sample_df["Outcome"])
