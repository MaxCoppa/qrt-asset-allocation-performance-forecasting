import pandas as pd
import numpy as np


import pandas as pd


def add_average_perf_features(
    X: pd.DataFrame,
    RET_features: list,
    window_sizes: list = [3, 5, 10],
    group_col: str = "TS",
):
    X = X.copy()
    for i in window_sizes:
        # rolling mean & std across lagged features
        avg_col = f"AVERAGE_PERF_{i}"

        X[avg_col] = X[RET_features[:i]].mean(axis=1)

        # group average (cross-sectional info)
        alloc_col = f"ALLOC_AVG_PERF_{i}"
        X[alloc_col] = X.groupby(group_col)[avg_col].transform("mean")

        # # spreads
        # last_ret = RET_features[i - 1]  # the "latest" return in that window
        # X[f"SPREAD_{last_ret}"] = (X[last_ret] - X[alloc_col]) / (
        #     X[alloc_col].abs() + 1e-6
        # )
        # X[f"SPREAD_LAST_{last_ret}"] = (X[last_ret] - X[avg_col]) / (
        #     X[avg_col].abs() + 1e-6
        # )

        # # NEW: z-score style feature (last return vs its rolling mean/std)
        # X[f"ZSCORE_{last_ret}"] = (X[last_ret] - X[avg_col]) / (X[std_col] + 1e-6)

    return X


def add_ret_minus_market(
    X: pd.DataFrame,
    RET_features: list,
    rolling_average: int = 1,
    group_col: str = "TS",
):
    n = len(RET_features)
    X = X.copy()
    for i in range(n - rolling_average - 1):

        avg_col = f"AVG_PAST_PERF_{i}"
        X[avg_col] = X[RET_features[i + 1 : i + rolling_average + 1]].mean(axis=1)

        alloc_col = f"ALLOC_AVG_PAST_PERF_{i}"
        std_col = f"ALLOC_STD_PAST_PERF_{i}"
        X[alloc_col] = X.groupby(group_col)[avg_col].transform("mean")  # Market
        X[std_col] = X.groupby(group_col)[avg_col].transform("std")

        # spreads
        last_ret = RET_features[i]  # the "today" return in that window
        X[f"SPREAD_{last_ret}"] = (X[last_ret] - X[alloc_col]) / X[std_col]

    if "target" in X.columns:
        avg_col = f"AVG_PAST_PERF"
        X[avg_col] = X[RET_features[1 : rolling_average + 1]].mean(axis=1)

        alloc_col = f"ALLOC_AVG_PAST_PERF"
        std_col = f"ALLOC_STD_PAST_PERF"
        X[alloc_col] = X.groupby(group_col)[avg_col].transform("mean")  # Market
        X[std_col] = X.groupby(group_col)[avg_col].transform("std")  # Market

        # spreads

        X[f"SPREAD_target"] = (X["target"] - X[alloc_col]) / X[std_col]

    return X


def add_average_volume_features(
    X: pd.DataFrame,
    SIGNED_VOLUME_features: list,
    window_sizes: list = [3, 5, 10, 15, 20],
    group_col: str = "TS",
):
    """
    Moving averages of signed volumes + within-group (day) averages.
    """
    X = X.copy()
    for i in window_sizes:
        avg_col = f"AVERAGE_VOLUME_{i}"
        alloc_col = f"ALLOCATIONS_AVERAGE_VOLUME_{i}"
        X[avg_col] = X[SIGNED_VOLUME_features[:i]].mean(axis=1)
        X[alloc_col] = X.groupby(group_col)[avg_col].transform("mean")
        X = X.drop(columns=avg_col)

        X[f"SPREAD_{SIGNED_VOLUME_features[i-1]}"] = (
            X[SIGNED_VOLUME_features[i - 1]] - X[alloc_col]
        ) / (X[alloc_col].abs() + 1e-6)
        X[f"ZSCORE_SIGNED_VOLUME_{i}"] = (
            X[SIGNED_VOLUME_features[i - 1]]
            - X[SIGNED_VOLUME_features[:i]].mean(axis=1)
        ) / (X[SIGNED_VOLUME_features[:i]].std(axis=1) + 1e-6)

    return X


def add_statistical_features(
    X: pd.DataFrame,
    RET_features: list,
    SIGNED_VOLUME_features: list,
    group_col: str = "TS",
):
    """
    Global statistics (std, skew, kurtosis) on returns and volumes,
    with within-group versions (deviation from the daily mean).
    """
    X = X.copy()

    # Returns
    X["RET_STD"] = X[RET_features].std(axis=1)
    X["RET_SKEW"] = X[RET_features].skew(axis=1)
    X["RET_KURT"] = X[RET_features].kurtosis(axis=1)

    # Volumes
    X["VOL_STD"] = X[SIGNED_VOLUME_features].std(axis=1)
    X["VOL_SKEW"] = X[SIGNED_VOLUME_features].skew(axis=1)
    X["VOL_KURT"] = X[SIGNED_VOLUME_features].kurtosis(axis=1)

    # Within-group spreads
    # for col in ["RET_STD", "VOL_STD"]:
    #     X[f"{col}_SPREAD"] = X[col] - X.groupby(group_col)[col].transform("mean")

    return X


def add_ratio_difference_features(
    X: pd.DataFrame, features: list, index_pairs: list = [(1, 20)]
):
    """
    Differences and ratios between specific features (e.g., RET_1 vs RET_20).
    """
    X = X.copy()
    for i, j in index_pairs:
        fi, fj = features[i - 1], features[j - 1]  # 1-based indexing
        colname_diff = f"{fi}_MINUS_{fj}"
        colname_ratio = f"{fi}_DIV_{fj}"
        X[colname_diff] = X[fi] - X[fj]
        X[colname_ratio] = X[fi] / (X[fj] + 1e-8)
    return X


def add_near_time_comparison_features(
    X: pd.DataFrame,
    RET_features: list,
    SIGNED_VOLUME_features: list,
    pairs: list = [(1, 2), (19, 20)],
):
    """
    Short-horizon comparisons: differences and ratios
    between returns, volumes, and impacts (return * volume).
    """
    X = X.copy()
    for i, j in pairs:
        # Returns
        X[f"RET_diff_{i}_{j}"] = X[RET_features[i - 1]] - X[RET_features[j - 1]]
        X[f"RET_ratio_{i}_{j}"] = X[RET_features[i - 1]] / (
            X[RET_features[j - 1]] + 1e-8
        )

        # Volumes
        X[f"VOL_diff_{i}_{j}"] = (
            X[SIGNED_VOLUME_features[i - 1]] - X[SIGNED_VOLUME_features[j - 1]]
        )
        X[f"VOL_ratio_{i}_{j}"] = X[SIGNED_VOLUME_features[i - 1]] / (
            X[SIGNED_VOLUME_features[j - 1]] + 1e-8
        )

        # Impacts
        imp_i = X[RET_features[i - 1]] * X[SIGNED_VOLUME_features[i - 1]]
        imp_j = X[RET_features[j - 1]] * X[SIGNED_VOLUME_features[j - 1]]
        X[f"IMPACT_diff_{i}_{j}"] = imp_i - imp_j
        X[f"IMPACT_ratio_{i}_{j}"] = imp_i / (imp_j + 1e-8)

    return X


def add_cross_sectional_features(
    X: pd.DataFrame, base_cols: list, group_col: str = "TS"
):
    """
    Adds ranks and relative spreads within each group (day).
    """
    X = X.copy()
    for col in base_cols:
        X[f"{col}_RANK"] = X.groupby(group_col)[col].rank(pct=True)
        X[f"{col}_SPREAD"] = X[col] - X.groupby(group_col)[col].transform("mean")
    return X


def scale_perf_features(
    X: pd.DataFrame,
    RET_features: list,
    SIGNED_VOLUME_features: list,
):
    """
    Moving averages of returns + within-group (day) averages.
    """

    X = X.copy()
    # X[RET_features] = (X[RET_features].sub(X[RET_features].mean(axis=1), axis=0)
    #  .div(X[RET_features].std(axis=1), axis=0))

    # X[SIGNED_VOLUME_features] = (X[SIGNED_VOLUME_features].sub(X[SIGNED_VOLUME_features].mean(axis=1), axis=0)
    #  .div(X[SIGNED_VOLUME_features].std(axis=1), axis=0))

    for col in SIGNED_VOLUME_features:
        X["SCALED_" + col] = X[col] * 1e-4

    return X


def add_mulitiply_col(
    X: pd.DataFrame,
    RET_features: list,
    SIGNED_VOLUME_features: list,
):
    X = X.copy()
    n = len(RET_features)
    for i in range(n):
        col_name = RET_features[i] + "_" + SIGNED_VOLUME_features[i]
        avg_col_name = "AVERAGE_" + col_name
        X[col_name] = X[RET_features[i]] * X[SIGNED_VOLUME_features[i]]
        X[avg_col_name] = X.groupby("TS")[col_name].transform("mean")

    return X
