import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
from sklearn.model_selection import KFold
from tqdm import tqdm
from sklearn.metrics import accuracy_score

from ..architecture.js_model import AE_BottleneckMLP


def nn_model_selection_using_kfold(
    data: pd.DataFrame,
    allocation_name: str,
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
    KFold CV training for supervised bottleneck autoencoder MLP in PyTorch (binary classification).
    """

    unique_vals = data[unique_id].unique()
    metrics = {"acc": []}
    models = []

    target = allocation_name + "_target"

    # Columns to drop (all targets except the main one)
    target_cols = [col for col in data.columns if ("target" in col) and col != target]

    # Columns to reconstruct (subset of features related to this allocation)
    cols_reconstructed = [
        col for col in data.columns if (allocation_name in col) and col != target
    ]

    # Drop unused columns
    data = data.drop(columns=target_cols).copy()

    kf = KFold(n_splits=n_splits, random_state=0, shuffle=True)
    for i, (train_idx, test_idx) in enumerate(kf.split(unique_vals)):
        print(f"\n===== Fold {i+1} =====")
        print(test_idx.shape, train_idx.shape)
        train_vals = unique_vals[train_idx]
        test_vals = unique_vals[test_idx]

        train_mask = data[unique_id].isin(train_vals)
        test_mask = data[unique_id].isin(test_vals)

        data_train = data.loc[train_mask].copy()
        data_val = data.loc[test_mask].copy()

        # Features
        if feat_engineering:
            X_train_np = feat_engineering(data_train)
            X_val_np = feat_engineering(data_val)
        else:
            X_train_np = data_train[features].values
            X_val_np = data_val[features].values

        # Vectors to reconstruct
        X_train_recon = data_train[cols_reconstructed].values
        X_val_recon = data_val[cols_reconstructed].values

        # Targets
        y_train_np = data_train[target].values
        y_val_np = data_val[target].values

        # Convert to tensors
        X_train = torch.tensor(X_train_np, dtype=torch.float32)
        X_train_recon = torch.tensor(X_train_recon, dtype=torch.float32)
        y_train = torch.tensor(y_train_np, dtype=torch.float32).view(-1, 1)

        X_val = torch.tensor(X_val_np, dtype=torch.float32)
        X_val_recon = torch.tensor(X_val_recon, dtype=torch.float32)
        y_val = torch.tensor(y_val_np, dtype=torch.float32).view(-1, 1)

        train_loader = DataLoader(
            TensorDataset(X_train, X_train_recon, y_train),
            batch_size=batch_size,
            shuffle=True,
        )
        val_loader = DataLoader(
            TensorDataset(X_val, X_val_recon, y_val),
            batch_size=batch_size,
            shuffle=False,
        )

        # Initialize model
        model = AE_BottleneckMLP(
            num_columns=len(features),
            enc_units=enc_units,
            dec_units=dec_units,
            mlp_units=mlp_units,
            recon_dim=len(cols_reconstructed),  # only reconstruct selected vector
            dropout_rate=dropout_rate,
        ).to(device)

        optimizer = optim.Adam(model.parameters(), lr=lr, weight_decay=0.0001)
        loss_decoder = nn.MSELoss(reduction="mean")
        loss_supervised = nn.MSELoss(reduction="mean")  # binary classification

        # Training loop
        for epoch in range(num_epochs):
            model.train()
            train_loss = 0.0

            for xb, x_recon_target, yb in tqdm(
                train_loader,
                desc=f"Fold {i+1} | Epoch {epoch+1}/{num_epochs}",
                leave=False,
            ):
                xb, x_recon_target, yb = (
                    xb.to(device),
                    x_recon_target.to(device),
                    yb.to(device),
                )

                x_recon, y_pred = model(xb)

                loss = alpha * loss_decoder(
                    x_recon, x_recon_target
                ) + beta * loss_supervised(y_pred, yb)

                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
                train_loss += loss.item() * xb.size(0)

            avg_train_loss = train_loss / len(train_loader.dataset)

            # Validation loop
            model.eval()
            val_loss, preds, true = 0.0, [], []
            for xb, x_recon_target, yb in tqdm(
                val_loader, desc=f"Validation Fold {i+1} Epoch {epoch+1}", leave=False
            ):
                xb, x_recon_target, yb = (
                    xb.to(device),
                    x_recon_target.to(device),
                    yb.to(device),
                )
                with torch.no_grad():
                    x_recon, y_pred = model(xb)
                    loss = alpha * loss_decoder(
                        x_recon, x_recon_target
                    ) + beta * loss_supervised(y_pred, yb)
                val_loss += loss.item() * xb.size(0)

                preds.append(y_pred.cpu().numpy())
                true.append(yb.cpu().numpy())

            preds = (np.vstack(preds) > 0).astype(int)
            true = (np.vstack(true) > 0).astype(int)
            print(np.mean(preds))
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
