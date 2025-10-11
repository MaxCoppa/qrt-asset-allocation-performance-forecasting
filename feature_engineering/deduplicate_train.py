import pandas as pd


def find_unique_indices(X_alloc: pd.DataFrame, RET_features: list[str]) -> list[int]:
    """Find indices of unique values based on overlapping RET features."""
    seen = set()
    unique_idx = []
    n = len(RET_features)

    for i in range(n - 1):
        ref = X_alloc[RET_features[i]] + X_alloc[RET_features[i + 1]]
        for j in range(i + 1, n - 1):
            comp = X_alloc[RET_features[j]] + X_alloc[RET_features[j + 1]]

            common_vals = set(ref) & set(comp)
            denom = len(set(comp))

            if denom > 0:
                p = len(common_vals) / denom
                if p > 0.5:
                    for val in common_vals:
                        if val not in seen:
                            idx = ref[ref == val].index[0]
                            unique_idx.append(idx)
                            seen.add(val)

    return unique_idx


def get_unique_subset(train: pd.DataFrame, unique_idx: list[int]) -> pd.DataFrame:
    """Return unique subset of train based on TS column and unique indices."""
    unique_date = train[train.index.isin(unique_idx)]["TS"].unique()
    return train[train["TS"].isin(unique_date)].reset_index(drop=True)


def extract_unique_train(
    data: pd.DataFrame,
    allocation: str = "ALLOCATION_01",
    output_csv: str = "data/train_unique.csv",
    save: bool = True,
) -> pd.DataFrame:
    """
    Main pipeline: load data, filter by allocation, find unique rows,
    and optionally save the result.
    """
    train = data.copy()
    RET_features = [f"RET_{i}" for i in range(1, 21)]

    # Filter by allocation
    X_alloc = train[train["ALLOCATION"] == allocation]

    # Compute unique indices
    unique_idx = find_unique_indices(X_alloc, RET_features)

    # Get final subset
    X_train_unique = get_unique_subset(train, unique_idx)

    # Save if requested
    if save:
        X_train_unique.to_csv(output_csv, index=False)

    return X_train_unique
