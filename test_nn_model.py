# %%
import pandas as pd
from deep_learning_models import nn_model_selection_using_kfold, AE_BottleneckMLP
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
from tqdm import tqdm
from sklearn.metrics import accuracy_score

# %% Load Data

train = pd.read_csv("data/train.csv")
X_val = pd.read_csv("data/X_val.csv")
y_val = pd.read_csv("data/y_val.csv")

# %%
features = [col for col in train.columns if col not in ["ROW_ID", "TS", "ALLOCATION"]]

# Transformation en format wide
df_wide = train.pivot(index="TS", columns="ALLOCATION", values=features)

# Les colonnes deviennent multi-index (feature, allocation)
# On aplatit en concaténant "ALLOCATION_FEATURE"
df_wide.columns = [f"{alloc}_{feat}" for feat, alloc in df_wide.columns]

# Reset index pour remettre TS comme colonne
df_wide = df_wide.reset_index()
df_wide
# %% Configuration

features = [
    col
    for col in df_wide.columns
    if col
    not in (
        ["ROW_ID", "TS", "ALLOCATION", "target"]
        + [f"ALLOCATION_{i}_target" for i in range(10, 66)]
    )
]
allocation_name = "ALLOCATION_01"
unique_id = "TS"

# %%
params = {
    "enc_units": [1024, 256, 64],  # encoder layers
    "dec_units": [64, 48],  # decoder layers (mirror or custom)
    "mlp_units": [32, 16, 2],  # classifier layers
    "dropout_rate": 0.1,
    "lr": 0.5,
}

# %%

data = df_wide.copy()
allocation_name = allocation_name
features = features
enc_units = params["enc_units"]
dec_units = params["dec_units"]
mlp_units = params["mlp_units"]
dropout_rate = params["dropout_rate"]
unique_id = unique_id
num_epochs = 40
batch_size = 16
lr = params["lr"]
device = "cpu"
alpha = 5
beta = 5

# %%


# =========================
# DATA PREP
# =========================
# Assume `data` is already loaded as a pandas DataFrame
target = allocation_name + "_target"

# Drop all other target columns
target_cols = [col for col in data.columns if ("targets" in col) and col != target]

# Columns to reconstruct (subset of features related to allocation)
cols_reconstructed = [
    col for col in data.columns if (allocation_name in col) and col != target
]

data = data.drop(columns=target_cols).copy()

# Simple split: 80% train / 20% validation
n_train = int(0.8 * len(data))
data_train = data.iloc[:n_train].copy()
data_val = data.iloc[n_train:].copy()

# Features
X_train_np = data_train[features].values
X_val_np = data_val[features].values

# Reconstruction targets
X_train_recon = data_train[cols_reconstructed].values
X_val_recon = data_val[cols_reconstructed].values

# Supervised targets (real-valued)
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

# =========================
# MODEL INIT
# =========================
model = AE_BottleneckMLP(
    num_columns=len(features),
    enc_units=enc_units,
    dec_units=dec_units,
    mlp_units=mlp_units,
    recon_dim=len(cols_reconstructed),
    dropout_rate=dropout_rate,
).to(device)

optimizer = optim.Adam(model.parameters(), lr=lr, weight_decay=0.0001)
loss_decoder = nn.MSELoss(reduction="mean")
loss_supervised = nn.MSELoss(reduction="mean")  # y ∈ ℝ
scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=10, gamma=0.1)

# =========================
# TRAINING LOOP
# =========================
for epoch in range(num_epochs):
    model.train()
    train_loss = 0.0

    for xb, x_recon_target, yb in tqdm(
        train_loader, desc=f"Epoch {epoch+1}/{num_epochs} [Train]", leave=False
    ):
        xb, x_recon_target, yb = xb.to(device), x_recon_target.to(device), yb.to(device)

        x_recon, y_pred = model(xb)

        loss = alpha * loss_decoder(x_recon, x_recon_target) + beta * loss_supervised(
            y_pred, yb
        )

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        train_loss += loss.item() * xb.size(0)

    avg_train_loss = train_loss / len(train_loader.dataset)
    scheduler.step()
    # =========================
    # VALIDATION LOOP
    # =========================
    model.eval()
    val_loss, preds, true = 0.0, [], []
    for xb, x_recon_target, yb in tqdm(
        val_loader, desc=f"Epoch {epoch+1}/{num_epochs} [Validation]", leave=False
    ):
        xb, x_recon_target, yb = xb.to(device), x_recon_target.to(device), yb.to(device)
        with torch.no_grad():
            x_recon, y_pred = model(xb)
            loss = alpha * loss_decoder(
                x_recon, x_recon_target
            ) + beta * loss_supervised(y_pred, yb)
        val_loss += loss.item() * xb.size(0)

        preds.append(y_pred.cpu().numpy())
        true.append(yb.cpu().numpy())

    preds = np.vstack(preds)
    true = np.vstack(true)

    # Accuracy by threshold: > 0 → 1 else 0
    preds_bin = (preds > 0).astype(int)
    true_bin = (true > 0).astype(int)
    print(np.mean(preds_bin))
    print(np.mean(true_bin))
    acc = accuracy_score(true_bin, preds_bin)

    print(
        f"Epoch {epoch+1}/{num_epochs} - "
        f"Train Loss: {avg_train_loss:.4f} - "
        f"Val Loss: {val_loss/len(val_loader.dataset):.4f} - "
        f"Val Acc: {acc:.4f}"
    )


# %%
