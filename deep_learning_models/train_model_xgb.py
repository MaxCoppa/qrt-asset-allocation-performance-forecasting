import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
from sklearn.model_selection import KFold
from tqdm import tqdm
from sklearn.metrics import mean_squared_error, accuracy_score
import xgboost as xgb

from .bottleneck import AE_Bottleneck  # <-- use your AE_Bottleneck


def ae_xgb_model_selection_using_kfold(
    data: pd.DataFrame,
    target: str,
    features: list[str],
    enc_units: list[int],
    dec_units: list[int],
    dropout_rate: float = 0.2,
    unique_id: str = "ROW_ID",
    n_splits: int = 4,
    num_epochs: int = 10,
    batch_size: int = 128,
    lr: float = 1e-3,
    device: str = "cuda" if torch.cuda.is_available() else "cpu",
):
    """
    KFold CV training for AE bottleneck + XGBoost regressor.
    """

    unique_vals = data[unique_id].unique()
    metrics = {"rmse": []}
    models = []
    xgb_models = []

    kf = KFold(n_splits=n_splits, random_state=0, shuffle=True)
    for i, (train_idx, test_idx) in enumerate(kf.split(unique_vals)):
        print(f"\n===== Fold {i+1} =====")

        train_vals = unique_vals[train_idx]
        test_vals = unique_vals[test_idx]

        train_mask = data[unique_id].isin(train_vals)
        test_mask = data[unique_id].isin(test_vals)

        data_local_train = data.loc[train_mask].copy()
        data_local_test = data.loc[test_mask].copy()

        X_train_np = data_local_train[features].values.astype(np.float32)
        X_val_np = data_local_test[features].values.astype(np.float32)

        y_train = data_local_train[target].values.astype(np.float32)
        y_val = data_local_test[target].values.astype(np.float32)

        # Convert to tensors
        X_train = torch.tensor(X_train_np, dtype=torch.float32)
        X_val = torch.tensor(X_val_np, dtype=torch.float32)

        train_loader = DataLoader(
            TensorDataset(X_train), batch_size=batch_size, shuffle=True
        )
        val_loader = DataLoader(
            TensorDataset(X_val), batch_size=batch_size, shuffle=False
        )

        # Initialize AE
        model = AE_Bottleneck(
            num_columns=len(features),
            enc_units=enc_units,
            dec_units=dec_units,
            dropout_rate=dropout_rate,
        ).to(device)

        optimizer = optim.Adam(model.parameters(), lr=lr, weight_decay=0.0001)
        loss_decoder = nn.MSELoss(reduction="mean")

        # Train AE
        for epoch in range(num_epochs):
            model.train()
            train_loss = 0.0

            for (xb,) in tqdm(
                train_loader,
                desc=f"Fold {i+1} | Epoch {epoch+1}/{num_epochs}",
                leave=False,
            ):
                xb = xb.to(device)
                x_recon, _ = model(xb)
                loss = loss_decoder(x_recon, xb)

                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
                train_loss += loss.item() * xb.size(0)

            avg_train_loss = train_loss / len(train_loader.dataset)

            # Validation reconstruction loss
            model.eval()
            val_loss = 0.0
            with torch.no_grad():
                for (xb,) in val_loader:
                    xb = xb.to(device)
                    x_recon, _ = model(xb)
                    loss = loss_decoder(x_recon, xb)
                    val_loss += loss.item() * xb.size(0)

            print(
                f"Epoch {epoch+1}/{num_epochs} - "
                f"Train Loss: {avg_train_loss:.4f} - "
                f"Val Loss: {val_loss/len(val_loader.dataset):.4f}"
            )

        # --- Extract bottleneck features ---
        model.eval()
        with torch.no_grad():
            _, Z_train = model(X_train.to(device))  # use encoder only
            _, Z_val = model(X_val.to(device))

        Z_train = (
            Z_train[0].cpu().numpy()
            if isinstance(Z_train, tuple)
            else Z_train.cpu().numpy()
        )
        Z_val = (
            Z_val[0].cpu().numpy() if isinstance(Z_val, tuple) else Z_val.cpu().numpy()
        )

        # --- Train XGBoost ---
        xgb_model = xgb.XGBRegressor(
            n_estimators=500,
            learning_rate=0.05,
            max_depth=6,
            subsample=0.8,
            colsample_bytree=0.8,
            tree_method="hist",
        )

        xgb_model.fit(Z_train, y_train, eval_set=[(Z_val, y_val)], verbose=False)

        # Evaluate
        y_pred = xgb_model.predict(Z_val)
        rmse = mean_squared_error(y_val, y_pred)
        metrics["rmse"].append(rmse)
        preds = (np.vstack(y_pred) > 0).astype(int)
        true = (np.vstack(y_val) > 0).astype(int)

        print(accuracy_score(true, preds))
        print(f"Fold {i+1} RMSE: {rmse:.4f}")

        models.append(model)
        xgb_models.append(xgb_model)

    # Aggregate results
    print("\nFinal CV Results:")
    for m in metrics:
        mean = np.mean(metrics[m])
        std = np.std(metrics[m])
        print(f"{m.upper()}: {mean:.4f} ± {std:.4f}")

    return models, xgb_models, metrics
