import pandas as pd
from typing import Tuple


def split_data(
    X: pd.DataFrame, y: pd.DataFrame, n_date_test: int
) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """
    Splits data into train and test sets based on the 'TS' column.

    """

    if "TS" not in X.columns:
        raise KeyError("'TS' column is missing in X.")

    # Extract numeric part of TS (assumes format 'xxx_number')
    date_numbers = X["TS"].str.split("_").apply(lambda x: int(x[1]))

    max_date = date_numbers.max()
    cutoff_date = max_date - n_date_test

    mask = date_numbers < cutoff_date

    return (
        X[mask].reset_index(drop=True),
        y[mask].reset_index(drop=True),
        X[~mask].reset_index(drop=True),
        y[~mask].reset_index(drop=True),
    )
