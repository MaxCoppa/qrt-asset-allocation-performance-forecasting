# %% Import Packages
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset
import pandas as pd
import numpy as np
from tqdm import tqdm
from deep_learning_models import GRUModel, LSTMModel

# %% Load Data and preprocess

train = pd.read_csv("../data/train.csv")
X_val = pd.read_csv("../data/X_val.csv")
y_val = pd.read_csv("../data/y_val.csv")


rets = train[[f"RET_{i}" for i in range(20, 0, -1)]].values
vols = train[[f"SIGNED_VOLUME_{i}" for i in range(20, 0, -1)]].values
X_seq = np.stack([rets, vols], axis=2)
X_turnover = train[["AVG_DAILY_TURNOVER"]].values
X_ret1 = train[["RET_1"]].values
y = train["target"].values

# %% Dataset Loader

X_seq_tensor = torch.tensor(X_seq, dtype=torch.float32)
X_turnover_tensor = torch.tensor(X_turnover, dtype=torch.float32)
X_ret1_tensor = torch.tensor(X_ret1, dtype=torch.float32)

y_tensor = torch.tensor(y, dtype=torch.float32)

dataset = TensorDataset(X_seq_tensor, X_turnover_tensor, X_ret1_tensor, y_tensor)
train_loader = DataLoader(dataset, batch_size=65, shuffle=False)


# %% Model Definition

model = LSTMModel(input_dim=2, hidden_dim=10, num_layers=5, seq_len=20)
criterion = nn.MSELoss()
optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)

# %% Training

for epoch in range(10):
    epoch_loss = 0
    correct = 0
    total = 0

    for xb_seq, xb_turnover, xb_ret1, yb in tqdm(
        train_loader, desc=f"Epoch {epoch+1}", leave=False
    ):
        pred = model(xb_seq, xb_turnover, xb_ret1)
        loss = criterion(pred, yb)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        epoch_loss += loss.item() * len(yb) * 1e4

        # Directional accuracy
        pred_sign = (pred.detach() > 0).int()
        true_sign = (yb > 0).int()
        correct += (pred_sign == true_sign).sum().item()
        total += len(yb)

    avg_loss = epoch_loss / total
    accuracy = correct / total
    print(f"Epoch {epoch+1}, Loss {avg_loss:.4f}, Acc: {accuracy:.4f}")


# %%
