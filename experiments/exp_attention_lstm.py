# %% Import Packages
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset
import pandas as pd
import numpy as np
from tqdm import tqdm
from deep_learning_models import CrossSeriesModel

# %% Load Data and preprocess
train = pd.read_csv("../data/train.csv")

n_ts = train["TS"].nunique()
timesteps = 20
seq_features = ["RET", "SIGNED_VOLUME"]

# Stack sequence features (like you did, but grouped by TS)
rets = train[[f"RET_{i}" for i in range(timesteps, 0, -1)]].values
vols = train[[f"SIGNED_VOLUME_{i}" for i in range(timesteps, 0, -1)]].values
X_seq = np.stack([rets, vols], axis=2)  # (n_samples, timesteps, 2)

n_samples, timesteps, n_feat = X_seq.shape

# Targets
y = train["target"].values

# Reshape into (batch, n_ts, timesteps, features)
X_seq = X_seq.reshape(n_ts, -1, timesteps, 2)
y = y.reshape(n_ts, -1)


# %% Dataset Loader
X_seq_tensor = torch.tensor(X_seq, dtype=torch.float32)
y_tensor = torch.tensor(y, dtype=torch.float32)

dataset = TensorDataset(X_seq_tensor, y_tensor)
train_loader = DataLoader(dataset, batch_size=8, shuffle=True)

# %% Model Definition

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = CrossSeriesModel(seq_dim=2, hidden_dim=32, num_heads=1).to(device)

criterion = nn.MSELoss()
optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)

# %% Training
for epoch in range(10):
    model.train()
    epoch_loss, correct, total = 0, 0, 0

    for xb_seq, yb in tqdm(train_loader, desc=f"Epoch {epoch+1}"):
        xb_seq, yb = xb_seq.to(device), yb.to(device)
        pred = model(xb_seq)  # (batch, n_ts)
        loss = criterion(pred, yb)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        epoch_loss += loss.item() * len(yb) * 1e4

        # Directional accuracy
        pred_sign = (pred.detach() > 0).int()
        true_sign = (yb > 0).int()
        correct += (pred_sign == true_sign).sum().item()
        total += yb.numel()

    avg_loss = epoch_loss / len(train_loader.dataset)
    acc = correct / total
    print(f"Epoch {epoch+1}: Loss={avg_loss:.6f}, Acc={acc:.4f}")


# %%
