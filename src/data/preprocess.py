"""Clean the raw diabetes dataset.

Certain clinical measurements cannot be zero for a living adult. Zeros in those 
columns are treated as missing values and imputed with the column median.
"""

import pandas as pd


def clean_data(df: pd.DataFrame, invalid_zero_columns: list[str]) -> pd.DataFrame:
    """Replace biologically-invalid zeros with NaN, then impute with the median.
    """
    df = df.copy()
    df[invalid_zero_columns] = df[invalid_zero_columns].replace(0, pd.NA)
    df[invalid_zero_columns] = df[invalid_zero_columns].fillna(
        df[invalid_zero_columns].median()
    )
    return df
