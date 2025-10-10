import numpy as np
import pandas as pd
from sklearn.model_selection import KFold
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_squared_error, accuracy_score
from pytorch_tabnet.tab_model import TabNetRegressor
import torch


def tabnet_model_selection_using_kfold(
    data: pd.DataFrame,
    target: str,
    features: list[str],
    n_splits: int = 4,
    feat_engineering=None,
    num_epochs: int = 100,
    patience: int = 20,
    batch_size: int = 1024,
    virtual_batch_size: int = 128,
    lr: float = 2e-2,
    device: str = "cuda" if torch.cuda.is_available() else "cpu",
):
    """
    KFold CV training with TabNet Regressor (supervised regression).
    Tracks both RMSE and accuracy (sign agreement: y>0 and pred>0).
    """

    unique_vals = data.index.values  # or replace with unique_id if needed
    metrics = {"rmse": [], "acc": []}
    models = []

    kf = KFold(n_splits=n_splits, random_state=42, shuffle=True)
    for i, (train_idx, val_idx) in enumerate(kf.split(unique_vals)):
        print(f"\n===== Fold {i+1} =====")

        train_df = data.iloc[train_idx].copy()
        val_df = data.iloc[val_idx].copy()

        if feat_engineering:
            X_train_np = feat_engineering(train_df)
            X_val_np = feat_engineering(val_df)
        else:
            X_train_np = train_df[features].values
            X_val_np = val_df[features].values

        y_train_np = train_df[target].values.reshape(-1, 1)
        y_val_np = val_df[target].values.reshape(-1, 1)

        # Scaling (optional, TabNet can handle raw features too)
        scaler = StandardScaler()
        X_train_np = scaler.fit_transform(X_train_np)
        X_val_np = scaler.transform(X_val_np)

        # Initialize TabNet model
        model = TabNetRegressor(
            optimizer_fn=torch.optim.Adam,
            optimizer_params=dict(lr=lr),
            mask_type="sparsemax",  # "sparsemax" or "entmax"
            device_name=device,
        )

        # Fit model
        model.fit(
            X_train=X_train_np,
            y_train=y_train_np,
            eval_set=[(X_val_np, y_val_np)],
            eval_name=["val"],
            eval_metric=["rmse"],
            max_epochs=num_epochs,
            patience=patience,
            batch_size=batch_size,
            virtual_batch_size=virtual_batch_size,
            num_workers=0,
            drop_last=False,
        )

        # Evaluate on validation set
        preds_val = model.predict(X_val_np)
        rmse = mean_squared_error(
            y_val_np,
            preds_val,
        )

        # Convert regression output to sign classification
        acc = accuracy_score((y_val_np > 0).astype(int), (preds_val > 0).astype(int))

        metrics["rmse"].append(rmse)
        metrics["acc"].append(acc)

        print(f"Fold {i+1} RMSE: {rmse:.4f} | Acc: {acc:.4f}")
        models.append(model)

    # Aggregate results
    print("\nFinal CV Results:")
    for m in metrics:
        mean = np.mean(metrics[m])
        std = np.std(metrics[m])
        print(f"{m.upper()}: {mean:.4f} ± {std:.4f}")

    return models, metrics
