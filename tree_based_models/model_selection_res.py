from sklearn.linear_model import Ridge
from sklearn.metrics import accuracy_score, mean_squared_error
from xgboost import XGBRegressor
from sklearn.model_selection import KFold
import numpy as np
import pandas as pd


def kfold_general_with_residuals(
    data: pd.DataFrame,
    target: str,
    features: list[str],
    unique_id: str = "ROW_ID",
    feat_engineering=None,
    n_splits: int = 5,
    general_model_cls=Ridge,
    general_params=None,
    residual_model_cls=XGBRegressor,
    residual_params=None,
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
                group.drop(
                    columns=["ROW_ID", "TS", "target", "residuals", "ALLOCATION"]
                ),
                group["residuals"],
            )
            residual_models[alloc] = res_model

        # --- Combined prediction ---
        base_pred = general_model.predict(X_test.drop(columns=["ALLOCATION"]))
        corrections = np.zeros(len(X_test))

        for alloc, model in residual_models.items():
            mask = X_test["ALLOCATION"] == alloc
            if mask.any():
                X_group = X_test.loc[mask].drop(columns=["ALLOCATION"])
                corrections[mask] = model.predict(X_group)

        y_pred = base_pred + corrections

        # --- Evaluation ---
        y_true_bin = (y_test > 0).astype(int)
        y_pred_bin = (y_pred > 0).astype(int)

        acc = accuracy_score(y_true_bin, y_pred_bin)

        metrics["accuracy"].append(acc)

        print(f"Fold {fold+1} - Acc: {acc:.3f}")

    # Aggregate
    for m, vals in metrics.items():
        mean, std = np.mean(vals), np.std(vals)
        print(f"{m.capitalize()}: {mean:.3f} ± {std:.3f}")

    return metrics
