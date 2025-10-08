import pandas as pd
import numpy as np


def add_average_perf_features(
    X: pd.DataFrame,
    RET_features: list,
    window_sizes: list = [3, 5, 10, 15, 20],
    group_col: str = "TS",
):
    """
    Moving averages of returns + within-group (day) averages.
    """
    X = X.copy()
    for i in window_sizes:
        avg_col = f"AVERAGE_PERF_{i}"
        alloc_col = f"ALLOCATIONS_AVERAGE_PERF_{i}"
        X[avg_col] = X[RET_features[:i]].mean(axis=1)
        X[alloc_col] = X.groupby(group_col)[avg_col].transform("mean")
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
    for col in ["RET_STD", "VOL_STD"]:
        X[f"{col}_SPREAD"] = X[col] - X.groupby(group_col)[col].transform("mean")

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
):
    """
    Moving averages of returns + within-group (day) averages.
    """

    X = X.copy()
    for col in RET_features:
        X[col] = np.sigmoid(X[col]) * np.log(np.abs(X[col]) * 1e4 + 1)
    if "target" in X.columns:
        X["target"] = np.sign(X["target"]) * np.log(np.abs(X[col]) * 1e4 + 1)
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
