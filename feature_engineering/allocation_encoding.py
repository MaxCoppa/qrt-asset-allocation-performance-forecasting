import pandas as pd
from typing import Dict, Optional


def encode_allocation(df: pd.DataFrame) -> pd.DataFrame:
    """
    Encode the 'ALLOCATION' column by extracting the integer part after the underscore.

    """
    if "ALLOCATION" not in df.columns:
        raise KeyError("'ALLOCATION' column not found in DataFrame!")

    def _extract_allocation(value: str) -> int:
        try:
            return int(value.split("_")[-1])
        except (AttributeError, ValueError, IndexError):
            raise ValueError(f"Invalid ALLOCATION format: {value!r}")

    df["allocation_id"] = df["ALLOCATION"].apply(_extract_allocation)

    return df


def create_mean_allocation(
    data: pd.DataFrame, dict_mean: Optional[Dict[str, float]] = None
) -> pd.DataFrame:
    """
    Adds a column `mean_allocation` to the DataFrame by mapping the mean
    of the `target` column grouped by `ALLOCATION`.
    """
    # Validate required column
    if "ALLOCATION" not in data.columns:
        raise KeyError("'ALLOCATION' column not found in DataFrame!")

    # Compute mapping dictionary if not provided
    if dict_mean is None:
        if "target" not in data.columns:
            raise KeyError(
                "'target' column not found in DataFrame when computing means!"
            )
        dict_mean = data.groupby("ALLOCATION")["target"].mean().to_dict()

    # Map the dictionary to create new column
    data = data.copy()
    data["mean_allocation"] = data["ALLOCATION"].map(dict_mean)

    return data


def create_allocation_features(
    data: pd.DataFrame,
    window_sizes: list = [3, 5, 10, 15, 20],
) -> pd.DataFrame:
    """
    Adds a column `mean_allocation` to the DataFrame by mapping the mean
    of the `target` column grouped by `ALLOCATION`.
    """
    # Validate required column
    if "ALLOCATION" not in data.columns:
        raise KeyError("'ALLOCATION' column not found in DataFrame!")

    dict_mean = data.groupby("ALLOCATION")["AVG_DAILY_TURNOVER"].mean().to_dict()

    # Map the dictionary to create new column
    data["AVG_DAILY_TURNOVER_ALLOCATION"] = data["ALLOCATION"].map(dict_mean)

    for i in window_sizes:
        avg_col = f"RET_{i}"
        alloc_col = f"GROUP_ALLOCATION_PERF_{i}"

        # Compute group mean of these averages
        data[alloc_col] = data.groupby("ALLOCATION")[avg_col].transform("mean")

    return data
