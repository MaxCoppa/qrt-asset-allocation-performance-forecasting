import pandas as pd


def add_average_perf_features(
    X: pd.DataFrame,
    RET_features: list,
    window_sizes: list = [3, 5, 10, 15, 20],
    group_col: str = "TS",
):
    """
    Add average performance and grouped average performance features
    to train and test datasets.

    """

    X = X.copy()

    for i in window_sizes:
        avg_col = f"AVERAGE_PERF_{i}"
        alloc_col = f"ALLOCATIONS_AVERAGE_PERF_{i}"

        # Compute average of first i return features
        X[avg_col] = X[RET_features[:i]].mean(axis=1)

        # Compute group mean of these averages
        X[alloc_col] = X.groupby(group_col)[avg_col].transform("mean")

    for i in window_sizes:
        avg_col = f"AVERAGE_PERF_{i}"
        alloc_col = f"ALLOCATIONS_AVERAGE_PERF_{i}"

    return X
