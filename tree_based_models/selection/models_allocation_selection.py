import pandas as pd
import numpy as np
import seaborn as sns
from sklearn.model_selection import KFold
from datetime import datetime
from tqdm import tqdm

from ..models import get_model
from ..evaluation import evaluate_model


def model_selection_by_allocation(
    data: pd.DataFrame,
    target: str,
    features: list[str],
    model_type: str,
    unique_id: str = "ROW_ID",
    log: bool = False,
    log_note: str = None,
    n_splits: int = 4,
):
    """
    Perform K-Fold cross-validation for model selection, separately for each ALLOCATION group.
    At the end, print overall aggregated results across all allocations.
    """

    all_results = {}
    all_models = {}
    all_accuracies = []  # collect accuracies from all allocations

    for alloc in tqdm(data["ALLOCATION"].unique(), desc="Processing allocations"):
        train_sub = data[data["ALLOCATION"] == alloc]

        # Prepare training sets
        X_train = train_sub[features].copy()
        y_train = train_sub[target].copy()
        unique_ids = train_sub[unique_id].unique()

        metrics = {"accuracy": []}
        models = []

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

    # Final aggregated results
    accs = np.array(all_accuracies)
    mean = accs.mean() * 100
    std = accs.std() * 100
    min_acc = accs.min() * 100
    max_acc = accs.max() * 100

    print(
        f"\nOverall Accuracy: {mean:.2f}% (± {std:.2f}%) "
        f"[Min: {min_acc:.2f}% ; Max: {max_acc:.2f}%] across all allocations"
    )

    # Logging results
    if log:
        logfile = "predictions/model_selection.log"
        note_str = f" | Note: {log_note}" if log_note else ""
        with open(logfile, "a") as f:
            f.write(
                f"{datetime.now()} - Model Selection by Allocation ({model_type}): "
                f"Mean acc: {mean:.2f}% | Std: {std:.2f}% | "
                f"Min: {min_acc:.2f}% | Max: {max_acc:.2f}%{note_str}\n"
            )


def get_data(ids, unique_vals, col: pd.Series, X_data: pd.DataFrame, y_data: pd.Series):
    """
    Extract subset of X and y given selected indices of unique values.
    """
    selected = unique_vals[ids]
    mask = col.isin(selected)
    return X_data.loc[mask], y_data.loc[mask], mask
