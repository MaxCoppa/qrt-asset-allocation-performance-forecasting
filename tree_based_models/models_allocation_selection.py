import pandas as pd
import numpy as np
import seaborn as sns
from sklearn.model_selection import KFold

from .initialise_model import get_model
from .evaluate import evaluate_model

from tqdm import tqdm


def model_selection_by_allocation(
    X: pd.DataFrame,
    y: pd.DataFrame,
    target: str,
    features: list[str],
    model_type: str,
    unique_id: str = "ROW_ID",
):
    """
    Perform K-Fold cross-validation for model selection, separately for each ALLOCATION group.
    At the end, print overall aggregated results across all allocations.
    """
    train = X.merge(y, on=unique_id)

    all_results = {}
    all_models = {}
    all_accuracies = []  # collect accuracies from all allocations

    for alloc in tqdm(X["ALLOCATION"].unique(), desc="Processing allocations"):
        # Filter subset
        train_sub = train[train["ALLOCATION"] == alloc]

        # Prepare training sets
        X_train = train_sub[features].copy()
        y_train = train_sub[target].copy()
        unique_ids = train_sub[unique_id].unique()

        n_splits = 8
        metrics = {"accuracy": []}
        models = []

        # Define KFold splits
        splits = KFold(n_splits=n_splits, random_state=0, shuffle=True).split(
            unique_ids
        )

        for train_idx, test_idx in splits:
            X_local_train, y_local_train, _ = get_data(
                train_idx, unique_ids, train_sub[unique_id], X_train, y_train
            )
            X_local_test, y_local_test, _ = get_data(
                test_idx, unique_ids, train_sub[unique_id], X_train, y_train
            )

            model = get_model(model_type)
            model.fit(X_local_train, y_local_train)

            model_eval = evaluate_model(model=model, X=X_local_test, y=y_local_test)
            acc = model_eval["accuracy"]

            models.append(model)
            metrics["accuracy"].append(acc)
            all_accuracies.append(acc)  # store for global summary

        all_results[alloc] = metrics
        all_models[alloc] = models

    # Print final aggregated results across all allocations
    mean = np.mean(all_accuracies) * 100
    std = np.std(all_accuracies) * 100
    l, u = mean - std, mean + std
    print(
        f"\nOverall Accuracy: {mean:.2f}% [{l:.2f}% ; {u:.2f}%] "
        f"(± {std:.2f}%) across all allocations"
    )


def get_data(ids, unique_vals, col: pd.Series, X_data: pd.DataFrame, y_data: pd.Series):
    """
    Extract subset of X and y given selected indices of unique values.
    """
    selected = unique_vals[ids]
    mask = col.isin(selected)
    return X_data.loc[mask], y_data.loc[mask], mask
