import pandas as pd


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
