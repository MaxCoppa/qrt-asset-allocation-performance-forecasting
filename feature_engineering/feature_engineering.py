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

    return X


def add_average_volume_features(
    X: pd.DataFrame,
    SIGNED_VOLUME_features: list,
    window_sizes: list = [3, 5, 10, 15, 20],
    group_col: str = "TS",
):

    X = X.copy()

    for i in window_sizes:
        avg_col = f"AVERAGE_VOLUME_{i}"
        alloc_col = f"ALLOCATIONS_AVERAGE_VOLUME_{i}"

        # Compute average of first i return features
        X[avg_col] = X[SIGNED_VOLUME_features[:i]].mean(axis=1)

        # Compute group mean of these averages
        X[alloc_col] = X.groupby(group_col)[avg_col].transform("mean")

    return X


def add_average_volume_features(
    X: pd.DataFrame,
    SIGNED_VOLUME_features: list,
    window_sizes: list = [3, 5, 10, 15, 20],
    group_col: str = "TS",
):

    X = X.copy()

    for i in window_sizes:
        avg_col = f"AVERAGE_VOLUME_{i}"
        alloc_col = f"ALLOCATIONS_AVERAGE_VOLUME_{i}"

        # Compute average of first i return features
        X[avg_col] = X[SIGNED_VOLUME_features[:i]].mean(axis=1)

        # Compute group mean of these averages
        X[alloc_col] = X.groupby(group_col)[avg_col].transform("mean")

    return X


def add_ratio_difference_features(X: pd.DataFrame, features: list, index_pairs: list):
    """
    Add ratio-scaled difference features for specific feature index pairs.
    """

    X = X.copy()

    for i, j in index_pairs:
        fi, fj = features[i - 1], features[j - 1]  # 1-based indexing
        colname = f"{fi}_MINUS_{fj}_RATIO"
        X[colname] = (X[fi] - X[fj]) / X[fj].replace(0, pd.NA)

    return X


def add_near_time_comparison_features(
    X: pd.DataFrame,
    RET_features: list,
    SIGNED_VOLUME_features: list,
    pairs: list = [(1, 2)],  # default RET_1 vs RET_2
):
    """
    Add near-time comparison features (differences and ratios)
    between RETs, volumes, and optionally RET*volume impacts.

    """
    X = X.copy()

    for i, j in pairs:
        # Return comparisons
        X[f"RET_diff_{i}_{j}"] = X[RET_features[i]] - X[RET_features[j]]
        X[f"RET_ratio_{i}_{j}"] = X[RET_features[i]] / (X[RET_features[j]] + 1e-8)

        # Volume-only comparisons (optional)
        X[f"VOL_diff_{i}_{j}"] = (
            X[SIGNED_VOLUME_features[i]] - X[SIGNED_VOLUME_features[j]]
        )
        X[f"VOL_ratio_{i}_{j}"] = X[SIGNED_VOLUME_features[i]] / (
            X[SIGNED_VOLUME_features[j]] + 1e-8
        )

        # Impact features (RET * VOLUME)

        # Create per-lag impacts
        imp_i = X[RET_features[i]] * X[SIGNED_VOLUME_features[i]]
        imp_j = X[RET_features[j]] * X[SIGNED_VOLUME_features[j]]

        # Compare impacts
        X[f"IMPACT_diff_{i}_{j}"] = imp_i - imp_j
        X[f"IMPACT_ratio_{i}_{j}"] = imp_i / (imp_j + 1e-8)

    return X
