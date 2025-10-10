from sklearn.linear_model import Ridge
from sklearn.metrics import accuracy_score
from xgboost import XGBRegressor
from sklearn.model_selection import KFold
import numpy as np
import pandas as pd
from datetime import datetime


def kfold_general_with_residuals(
    data: pd.DataFrame,
    target: str,
    features: list[str],
    features_res: list[str],
    unique_id: str = "ROW_ID",
    feat_engineering=None,
    n_splits: int = 5,
    general_model_cls=Ridge,
    general_params=None,
    residual_model_cls=XGBRegressor,
    residual_params=None,
    log: bool = False,
    log_note: str = None,
):
    """
    KFold cross-validation for the 'general + residual per ALLOCATION' approach.
    """

    general_params = general_params or {"alpha": 1.0, "fit_intercept": True}
    residual_params = residual_params or {
        "n_estimators": 50,
        "max_depth": 3,
        "learning_rate": 0.1,
        "subsample": 0.8,
        "colsample_bytree": 0.8,
        "random_state": 42,
    }

    missing = [f for f in features_res if f not in features]
    if missing:
        raise ValueError(f"Invalid features found: {missing}")

    unique_vals = data[unique_id].unique()
    metrics = {"accuracy": []}

    kf = KFold(n_splits=n_splits, shuffle=True, random_state=42)

    for fold, (train_idx, test_idx) in enumerate(kf.split(unique_vals)):
        train_vals = unique_vals[train_idx]
        test_vals = unique_vals[test_idx]

        train_mask = data[unique_id].isin(train_vals)
        test_mask = data[unique_id].isin(test_vals)

        train_df = data.loc[train_mask].copy()
        test_df = data.loc[test_mask].copy()

        if feat_engineering:
            train_df = feat_engineering(train_df)
            test_df = feat_engineering(test_df)

        X_train = train_df[features]
        y_train = train_df[target]

        X_test = test_df[features]
        y_test = test_df[target]

        # --- General model ---
        general_model = general_model_cls(**general_params)
        general_model.fit(X_train.drop(columns=["ALLOCATION"]), y_train)

        train_df["residuals"] = y_train - general_model.predict(
            X_train.drop(columns=["ALLOCATION"])
        )

        # --- Residual models per ALLOCATION ---
        residual_models = {}
        for alloc, group in train_df.groupby("ALLOCATION"):
            res_model = residual_model_cls(**residual_params)
            res_model.fit(
                group[features_res].drop(columns=["ALLOCATION"]),
                group["residuals"],
            )
            residual_models[alloc] = res_model

        # --- Combined prediction ---
        base_pred = general_model.predict(X_test.drop(columns=["ALLOCATION"]))
        corrections = np.zeros(len(X_test))

        for alloc, model in residual_models.items():
            mask = X_test["ALLOCATION"] == alloc
            if mask.any():
                X_group = X_test[features_res].loc[mask].drop(columns=["ALLOCATION"])
                corrections[mask] = model.predict(X_group)

        y_pred = base_pred + corrections

        # --- Evaluation ---
        y_true_bin = (y_test > 0).astype(int)
        y_pred_bin = (y_pred > 0).astype(int)

        acc = accuracy_score(y_true_bin, y_pred_bin)
        metrics["accuracy"].append(acc)

        print(f"Fold {fold+1} - Acc: {acc*100:.2f}%")

    # Aggregate results
    accs = np.array(metrics["accuracy"])
    mean_acc = accs.mean() * 100
    std_acc = accs.std() * 100
    min_acc = accs.min() * 100
    max_acc = accs.max() * 100

    print(
        f"\nAccuracy: {mean_acc:.2f}% (± {std_acc:.2f}%) "
        f"[Min: {min_acc:.2f}% ; Max: {max_acc:.2f}%]"
    )

    # Logging results
    if log:
        logfile = "predictions/model_selection.log"
        note_str = f" | Note: {log_note}" if log_note else ""
        with open(logfile, "a") as f:
            f.write(
                f"{datetime.now()} - General+Residual ({general_model_cls.__name__} + {residual_model_cls.__name__}): "
                f"Mean acc: {mean_acc:.2f}% | Std: {std_acc:.2f}% | "
                f"Min: {min_acc:.2f}% | Max: {max_acc:.2f}%{note_str}\n"
            )

    return metrics
