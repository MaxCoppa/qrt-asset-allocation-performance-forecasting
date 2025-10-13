import pandas as pd
import numpy as np
from typing import List, Tuple


def add_average_perf_features(
    X: pd.DataFrame,
    RET_features: List[str],
    window_sizes: List[int] = [3, 5, 10],
    group_col: str = "TS",
    include_spreads: bool = False,
    include_zscore: bool = False,
) -> pd.DataFrame:
    """
    Rolling performance averages, group averages, and optionally spreads/z-scores.
    """

    for i in window_sizes:
        avg_col = f"AVERAGE_PERF_{i}"
        X[avg_col] = X[RET_features[:i]].mean(axis=1)

        alloc_col = f"ALLOC_AVG_PERF_{i}"
        X[alloc_col] = X.groupby(group_col)[avg_col].transform("mean")

        if include_spreads:
            last_ret = RET_features[i - 1]
            X[f"SPREAD_{last_ret}"] = (X[last_ret] - X[alloc_col]) / (
                X[alloc_col].abs() + 1e-6
            )
            X[f"SPREAD_LAST_{last_ret}"] = (X[last_ret] - X[avg_col]) / (
                X[avg_col].abs() + 1e-6
            )

        if include_zscore:
            std_col = f"STD_PERF_{i}"
            X[std_col] = X[RET_features[:i]].std(axis=1)
            last_ret = RET_features[i - 1]
            X[f"ZSCORE_{last_ret}"] = (X[last_ret] - X[avg_col]) / (X[std_col] + 1e-6)

    return X


def add_ret_minus_market(
    X: pd.DataFrame,
    RET_features: List[str],
    rolling_average: int = 1,
    group_col: str = "TS",
    include_target: bool = False,
) -> pd.DataFrame:
    """
    Adds return vs. market spread features.
    """
    n = len(RET_features)

    for i in range(n - rolling_average - 1):
        avg_col = f"AVG_PAST_PERF_{i}"
        X[avg_col] = X[RET_features[i + 1 : i + rolling_average + 1]].mean(axis=1)

        alloc_col = f"ALLOC_AVG_PAST_PERF_{i}"
        std_col = f"ALLOC_STD_PAST_PERF_{i}"
        X[alloc_col] = X.groupby(group_col)[avg_col].transform("mean")
        X[std_col] = X.groupby(group_col)[avg_col].transform("std")

        last_ret = RET_features[i]
        X[f"SPREAD_{last_ret}"] = (X[last_ret] - X[alloc_col]) / (X[std_col] + 1e-6)

    if include_target and "target" in X.columns:
        # Change the target Spread
        avg_col = f"AVG_PAST_PERF"
        X[avg_col] = X[RET_features[1 : rolling_average + 1]].mean(axis=1)

        alloc_col = f"ALLOC_AVG_PAST_PERF"
        std_col = f"ALLOC_STD_PAST_PERF"
        X[alloc_col] = X.groupby(group_col)[avg_col].transform("mean")
        X[std_col] = X.groupby(group_col)[avg_col].transform("std")

        X[f"SPREAD_target"] = (X["target"] - X[alloc_col]) / (X[std_col] + 1e-6)

    return X


def add_average_volume_features(
    X: pd.DataFrame,
    SIGNED_VOLUME_features: List[str],
    window_sizes: List[int] = [3, 5, 10, 15, 20],
    group_col: str = "TS",
    include_spreads: bool = False,
    include_zscore: bool = False,
) -> pd.DataFrame:
    """
    Moving averages of signed volumes + group averages + spreads/zscores.
    """

    for i in window_sizes:
        avg_col = f"AVERAGE_VOLUME_{i}"
        alloc_col = f"ALLOC_AVG_VOLUME_{i}"

        X[avg_col] = X[SIGNED_VOLUME_features[:i]].mean(axis=1)
        X[alloc_col] = X.groupby(group_col)[avg_col].transform("mean")

        if include_spreads:
            X[f"SPREAD_{SIGNED_VOLUME_features[i-1]}"] = (
                X[SIGNED_VOLUME_features[i - 1]] - X[alloc_col]
            ) / (X[alloc_col].abs() + 1e-6)

        if include_zscore:
            X[f"ZSCORE_SIGNED_VOLUME_{i}"] = (
                X[SIGNED_VOLUME_features[i - 1]]
                - X[SIGNED_VOLUME_features[:i]].mean(axis=1)
            ) / (X[SIGNED_VOLUME_features[:i]].std(axis=1) + 1e-6)

    return X


def add_statistical_features(
    X: pd.DataFrame,
    RET_features: List[str],
    SIGNED_VOLUME_features: List[str],
    group_col: str = "TS",
    include_group_spreads: bool = True,
) -> pd.DataFrame:
    """
    Adds std, skew, kurtosis for returns and volumes.
    Optionally include within-group spreads.
    """

    X["RET_STD"] = X[RET_features].std(axis=1)
    X["RET_SKEW"] = X[RET_features].skew(axis=1)
    X["RET_KURT"] = X[RET_features].kurtosis(axis=1)

    X["VOL_STD"] = X[SIGNED_VOLUME_features].std(axis=1)
    X["VOL_SKEW"] = X[SIGNED_VOLUME_features].skew(axis=1)
    X["VOL_KURT"] = X[SIGNED_VOLUME_features].kurtosis(axis=1)

    if include_group_spreads:
        for col in [
            "RET_STD",
            "RET_SKEW",
            "RET_KURT",
            "VOL_STD",
            "VOL_SKEW",
            "VOL_KURT",
        ]:
            X[f"{col}_SPREAD"] = X[col] - X.groupby(group_col)[col].transform("mean")

    return X


def add_ratio_difference_features(
    X: pd.DataFrame, features: List[str], index_pairs: List[Tuple[int, int]] = [(1, 20)]
) -> pd.DataFrame:
    """
    Differences and ratios between specific feature indices (1-based).
    """

    for i, j in index_pairs:
        fi, fj = features[i - 1], features[j - 1]
        X[f"{fi}_MINUS_{fj}"] = X[fi] - X[fj]
        X[f"{fi}_DIV_{fj}"] = X[fi] / (X[fj] + 1e-6)
    return X


def add_near_time_comparison_features(
    X: pd.DataFrame,
    RET_features: List[str],
    SIGNED_VOLUME_features: List[str],
    pairs: List[Tuple[int, int]] = [(1, 2), (19, 20)],
) -> pd.DataFrame:
    """
    Short-horizon comparisons: differences and ratios
    between returns, volumes, and impacts (return * volume).
    """

    for i, j in pairs:
        X[f"RET_diff_{i}_{j}"] = X[RET_features[i - 1]] - X[RET_features[j - 1]]
        X[f"RET_ratio_{i}_{j}"] = X[RET_features[i - 1]] / (
            X[RET_features[j - 1]] + 1e-6
        )

        X[f"VOL_diff_{i}_{j}"] = (
            X[SIGNED_VOLUME_features[i - 1]] - X[SIGNED_VOLUME_features[j - 1]]
        )
        X[f"VOL_ratio_{i}_{j}"] = X[SIGNED_VOLUME_features[i - 1]] / (
            X[SIGNED_VOLUME_features[j - 1]] + 1e-6
        )

        imp_i = X[RET_features[i - 1]] * X[SIGNED_VOLUME_features[i - 1]]
        imp_j = X[RET_features[j - 1]] * X[SIGNED_VOLUME_features[j - 1]]
        X[f"IMPACT_diff_{i}_{j}"] = imp_i - imp_j
        X[f"IMPACT_ratio_{i}_{j}"] = imp_i / (imp_j + 1e-6)

    return X


def add_cross_sectional_features(
    X: pd.DataFrame, base_cols: List[str], group_col: str = "TS"
) -> pd.DataFrame:
    """
    Adds ranks and relative spreads within each group.
    """

    for col in base_cols:
        X[f"{col}_RANK"] = X.groupby(group_col)[col].rank(pct=True)
        X[f"{col}_SPREAD"] = X[col] - X.groupby(group_col)[col].transform("mean")
    return X


def scale_features(
    X: pd.DataFrame,
    RET_features: List[str],
    SIGNED_VOLUME_features: List[str],
) -> pd.DataFrame:
    """
    Scales volume features and optionally adds z-score normalization.
    """

    for col in SIGNED_VOLUME_features:
        X["SCALED_" + col] = X[col] * 1e-2

    return X


def add_return_to_volume_ratio(
    X: pd.DataFrame,
    RET_features: List[str],
    SIGNED_VOLUME_features: List[str],
    group_col: str = "TS",
) -> pd.DataFrame:
    """
    Return-to-volume ratios + group averages.
    """

    for r, v in zip(RET_features, SIGNED_VOLUME_features):
        col_name = f"{r}_DIV_{v}"
        avg_col_name = f"AVG_{col_name}"
        X[col_name] = X[r] / (X[v] + 1e-6)
        X[avg_col_name] = X.groupby(group_col)[col_name].transform("mean")
    return X


def add_lagged_features(
    X: pd.DataFrame, features: List[str], lags: List[int] = [1, 2, 5]
) -> pd.DataFrame:
    """
    Adds lagged versions of selected features.
    """

    for col in features:
        for lag in lags:
            X[f"{col}_LAG{lag}"] = X[col].shift(lag)
    return X


def add_rolling_corr_features(
    X: pd.DataFrame, RET_features: List[str], VOL_features: List[str], window: int = 5
) -> pd.DataFrame:
    """
    Rolling correlations between return and volume.
    """

    for r, v in zip(RET_features, VOL_features):
        X[f"CORR_{r}_{v}_{window}"] = X[r].rolling(window).corr(X[v])
    return X


def add_vol_adjusted_returns(
    X: pd.DataFrame, RET_features: List[str], window: int = 5
) -> pd.DataFrame:
    """
    Volatility-adjusted returns (Sharpe-like).
    """

    for col in RET_features:
        vol = X[col].rolling(window).std() + 1e-6
        X[f"{col}_ADJ_VOL{window}"] = X[col] / vol
    return X
