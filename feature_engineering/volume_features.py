import pandas as pd


def add_strategy_features(
    X: pd.DataFrame,
    SIGNED_VOLUME_features: list,
):

    X = X.copy()

    # Compute average of first i return features
    X["is_long_short_term"] = (X[SIGNED_VOLUME_features[:2]].mean(axis=1) > 0).astype(
        int
    )
    X["is_long_middle_term"] = (X[SIGNED_VOLUME_features[:5]].mean(axis=1) > 0).astype(
        int
    )
    X["is_long_long_term"] = (X[SIGNED_VOLUME_features[:20]].mean(axis=1) > 0).astype(
        int
    )

    return X
