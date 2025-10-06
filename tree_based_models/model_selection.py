"""
model_selection.py

Model selection utility using K-Fold cross-validation.

- A Random Forest (or another tree-based model) can be used as a benchmark.
- Missing values are filled with 0.
- KFold is applied on unique identifiers (e.g., IDs, dates) to ensure
  splits are consistent with grouped data.
"""

import pandas as pd
import numpy as np
import seaborn as sns
from sklearn.model_selection import KFold

from .initialise_model import get_model
from .evaluate import evaluate_model


def model_selection_using_kfold(
    data: pd.DataFrame,
    target: str,
    features: list[str],
    model_type: str,
    feat_engineering=None,
    unique_id: str = "ROW_ID",
    plot_ft_importance: bool = False,
    n_splits: int = 8,
):
    """
    Perform K-Fold cross-validation for model selection,
    splitting folds on unique values (e.g., dates).

    Parameters
    ----------
    X : pd.DataFrame
        Training dataset.
    y : pd.DataFrame
        Target dataset.
    target : str
        Target column name.
    features : list of str
        Feature column names.
    model_type : str
        One of {"rf", "xgb", "lgbm", "cat"}.
    feat_engineering : callable, optional
        Feature engineering function applied per fold.
    unique_id : str, default="ROW_ID"
        Column used to split folds (e.g., IDs, dates).
    plot_ft_importance : bool, default=False
        If True, plots average feature importance across folds.
    n_splits : int, default=4
        Number of KFold splits.
    """

    # Get the unique identifiers (e.g. dates)
    unique_vals = data[unique_id].unique()

    metrics = {"accuracy": []}
    models = []

    # Apply KFold *directly* on the unique values, not their array indices
    kf = KFold(n_splits=n_splits, random_state=0, shuffle=True)
    for i, (train_idx, test_idx) in enumerate(kf.split(unique_vals)):
        # Select the actual values for this fold
        train_vals = unique_vals[train_idx]
        test_vals = unique_vals[test_idx]
        # Build masks by checking membership in the unique_id column
        train_mask = data[unique_id].isin(train_vals)
        test_mask = data[unique_id].isin(test_vals)

        data_local_train = data.loc[train_mask].copy()
        data_local_test = data.loc[test_mask].copy()

        if feat_engineering:
            data_local_train = feat_engineering(data_local_train)
            data_local_test = feat_engineering(data_local_test)

        X_local_train = data_local_train[features]
        X_local_test = data_local_test[features]

        y_local_train = data_local_train[target]
        y_local_test = data_local_test[target]

        # Initialize and fit model
        model = get_model(model_type)
        model.fit(X_local_train, y_local_train)

        # Evaluate on local test split
        model_eval = evaluate_model(model=model, X=X_local_test, y=y_local_test)

        models.append(model)

        acc = model_eval["accuracy"]
        metrics["accuracy"].append(acc)

        print(f"Fold {i+1} - Accuracy: {acc*100:.2f}%")

    # Aggregate results
    for m in metrics:
        mean = np.mean(metrics[m]) * (100 if m == "accuracy" else 1)
        std = np.std(metrics[m]) * (100 if m == "accuracy" else 1)
        l, u = mean - std, mean + std
        unit = "%" if m == "accuracy" else ""
        print(
            f"{m.capitalize()}: {mean:.2f}{unit} [{l:.2f}{unit} ; {u:.2f}{unit}] "
            f"(± {std:.2f}{unit})"
        )

    if plot_ft_importance:
        plot_feature_importance(models, features)


def get_data(ids, unique_vals, col: pd.Series, X_data: pd.DataFrame, y_data: pd.Series):
    """
    Extract subset of X and y given selected indices of unique values.
    """
    selected = unique_vals[ids]
    mask = col.isin(selected)
    return X_data.loc[mask], y_data.loc[mask], mask


def plot_feature_importance(models, features):
    """
    Plot mean feature importance across trained models.
    """
    feature_importances = pd.DataFrame(
        [model.feature_importances_ for model in models], columns=features
    )

    sns.barplot(
        data=feature_importances,
        orient="h",
        order=feature_importances.mean().sort_values(ascending=False).index,
    )
    return True
