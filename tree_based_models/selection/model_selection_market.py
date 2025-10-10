import pandas as pd
import numpy as np
import seaborn as sns
from sklearn.model_selection import KFold
from datetime import datetime

from ..models import get_model
from ..evaluation import evaluate_model_market


def model_selection_respect_market(
    data: pd.DataFrame,
    target: str,
    true_target: str,
    features: list[str],
    model_type: str,
    unique_id: str = "ROW_ID",
    market_feature: str = None,
    n_splits: int = 4,
    plot_ft_importance: bool = False,
    log: bool = False,
    log_note: str = None,
):
    """
    Perform K-Fold cross-validation for model selection,
    splitting folds on unique values (e.g., dates).
    """

    unique_vals = data[unique_id].unique()
    metrics = {"accuracy": []}
    models = []

    kf = KFold(n_splits=n_splits, random_state=0, shuffle=True)
    for i, (train_idx, test_idx) in enumerate(kf.split(unique_vals)):
        train_vals = unique_vals[train_idx]
        test_vals = unique_vals[test_idx]

        train_mask = data[unique_id].isin(train_vals)
        test_mask = data[unique_id].isin(test_vals)

        data_local_train = data.loc[train_mask].copy()
        data_local_test = data.loc[test_mask].copy()

        X_local_train = data_local_train[features]
        X_local_test = data_local_test[features]

        y_local_train = data_local_train[[target, true_target]]
        y_local_test = data_local_test[[target, true_target]]

        y_local_market = data_local_test[market_feature]

        model = get_model(model_type)
        model.fit(X_local_train, y_local_train[target])

        model_eval = evaluate_model_market(
            model=model,
            X=X_local_test,
            y=y_local_test[true_target],
            y_market=y_local_market,
        )

        models.append(model)

        acc = model_eval["accuracy"]
        metrics["accuracy"].append(acc)

        print(f"Fold {i+1} - Accuracy: {acc*100:.2f}%")

    # Aggregate results
    accs = np.array(metrics["accuracy"])
    mean_acc = accs.mean() * 100
    std_acc = accs.std() * 100
    min_acc = accs.min() * 100
    max_acc = accs.max() * 100

    print(
        f"Accuracy: {mean_acc:.2f}% (± {std_acc:.2f}%) "
        f"[Min: {min_acc:.2f}% ; Max: {max_acc:.2f}%]"
    )

    # Logging results
    if log:
        logfile = "predictions/model_selection.log"
        note_str = f" | Note: {log_note}" if log_note else ""
        with open(logfile, "a") as f:
            f.write(
                f"{datetime.now()} - Model Selection ({model_type}): "
                f"Mean acc: {mean_acc:.2f}% | Std: {std_acc:.2f}% | "
                f"Min: {min_acc:.2f}% | Max: {max_acc:.2f}%{note_str}\n"
            )

    # Feature importance
    if plot_ft_importance:
        try:
            plot_feature_importance(models, features)
        except Exception:
            print("No possible to get feature importance for this model.")


def get_data(ids, unique_vals, col: pd.Series, X_data: pd.DataFrame, y_data: pd.Series):
    """Extract subset of X and y given selected indices of unique values."""
    selected = unique_vals[ids]
    mask = col.isin(selected)
    return X_data.loc[mask], y_data.loc[mask], mask


def plot_feature_importance(models, features):
    """
    Plot mean feature importance across trained models and log top features if required.
    """
    feature_importances = pd.DataFrame(
        [model.feature_importances_ for model in models], columns=features
    )
    mean_importance = feature_importances.mean().sort_values(ascending=False)

    top_features = mean_importance.head(10).index.tolist()

    print("\nTop 10 important features:")
    print(top_features)

    sns.barplot(
        data=feature_importances,
        orient="h",
        order=mean_importance.index,
    )
    return True
