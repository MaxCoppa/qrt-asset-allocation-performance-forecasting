import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
from sklearn.model_selection import KFold
from tqdm import tqdm
from sklearn.metrics import accuracy_score
from sklearn.preprocessing import StandardScaler

from .jane_street_model import AE_BottleneckMLP


def nn_model_selection_using_kfold(
    data: pd.DataFrame,
    target: str,
    features: list[str],
    enc_units: list[int],
    dec_units: list[int],
    mlp_units: list[int],
    dropout_rate: float = 0.2,
    unique_id: str = "ROW_ID",
    n_splits: int = 4,
    feat_engineering=None,
    num_epochs: int = 10,
    batch_size: int = 128,
    lr: float = 1e-3,
    alpha: float = 1.0,
    beta: float = 1.0,
    device: str = "cuda" if torch.cuda.is_available() else "cpu",
):
    """
    KFold CV training for supervised bottleneck autoencoder MLP in PyTorch (classification).
    """

    unique_vals = data[unique_id].unique()
    metrics = {"acc": []}
    models = []

    kf = KFold(n_splits=n_splits, random_state=0, shuffle=True)
    for i, (train_idx, test_idx) in enumerate(kf.split(unique_vals)):
        print(f"\n===== Fold {i+1} =====")

        train_vals = unique_vals[train_idx]
        test_vals = unique_vals[test_idx]

        train_mask = data[unique_id].isin(train_vals)
        test_mask = data[unique_id].isin(test_vals)

        data_local_train = data.loc[train_mask].copy()
        data_local_test = data.loc[test_mask].copy()

        if feat_engineering:
            X_train_np = feat_engineering(data_local_train)
            X_val_np = feat_engineering(data_local_test)

        X_train_np = data_local_train[features].values
        X_val_np = data_local_test[features].values

        # Convert to tensors
        X_train = torch.tensor(X_train_np, dtype=torch.float32)
        y_train = torch.tensor(
            data_local_train[target].values, dtype=torch.float32
        ).view(-1, 1)
        X_val = torch.tensor(X_val_np, dtype=torch.float32)
        y_val = torch.tensor(data_local_test[target].values, dtype=torch.float32).view(
            -1, 1
        )

        train_loader = DataLoader(
            TensorDataset(X_train, y_train), batch_size=batch_size, shuffle=False
        )
        val_loader = DataLoader(
            TensorDataset(X_val, y_val), batch_size=batch_size, shuffle=False
        )

        # Initialize model
        model = AE_BottleneckMLP(
            num_columns=len(features),
            enc_units=enc_units,
            dec_units=dec_units,
            mlp_units=mlp_units,
            dropout_rate=dropout_rate,
        ).to(device)

        optimizer = optim.Adam(model.parameters(), lr=lr, weight_decay=0.0001)
        loss_decoder = nn.MSELoss(reduction="mean")
        loss_supervised = nn.MSELoss(reduction="mean")

        # Training loop
        for epoch in range(num_epochs):
            model.train()
            train_loss = 0.0

            for xb, yb in tqdm(
                train_loader,
                desc=f"Fold {i+1} | Epoch {epoch+1}/{num_epochs}",
                leave=False,
            ):
                xb, yb = xb.to(device), yb.to(device)

                x_recon, y_pred = model(xb)
                loss = alpha * loss_decoder(x_recon, xb) + beta * loss_supervised(
                    y_pred, yb
                )

                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
                train_loss += loss.item() * xb.size(0)

            avg_train_loss = train_loss / len(train_loader.dataset)

            # Validation loop
            model.eval()
            val_loss, preds, true = 0.0, [], []
            for xb, yb in tqdm(
                val_loader, desc=f"Validation Fold {i+1} Epoch {epoch+1}", leave=False
            ):
                xb, yb = xb.to(device), yb.to(device)
                with torch.no_grad():
                    x_recon, y_pred = model(xb)
                    loss = alpha * loss_decoder(x_recon, xb) + beta * loss_supervised(
                        y_pred, yb
                    )
                val_loss += loss.item() * xb.size(0)

                preds.append(y_pred.cpu().numpy())
                true.append(yb.cpu().numpy())

            preds = (np.vstack(preds) > 0).astype(int)
            true = (np.vstack(true) > 0).astype(int)

            acc = accuracy_score(true, preds)
            metrics["acc"].append(acc)

            print(
                f"Epoch {epoch+1}/{num_epochs} - "
                f"Train Loss: {avg_train_loss:.4f} - "
                f"Val Loss: {val_loss/len(val_loader.dataset):.4f} - "
                f"Acc: {acc:.4f}"
            )

        models.append(model)

    # Aggregate results
    print("\nFinal CV Results:")
    for m in metrics:
        mean = np.mean(metrics[m])
        std = np.std(metrics[m])
        print(f"{m.upper()}: {mean:.4f} ± {std:.4f}")

    return models, metrics
