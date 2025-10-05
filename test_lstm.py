# %%
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset
import pandas as pd
import numpy as np
from tqdm import tqdm

# %% Load Data

train = pd.read_csv("data/train.csv")
X_val = pd.read_csv("data/X_val.csv")
y_val = pd.read_csv("data/y_val.csv")


rets = train[[f"RET_{i}" for i in range(20, 0, -1)]].values
vols = train[[f"SIGNED_VOLUME_{i}" for i in range(20, 0, -1)]].values
X_seq = np.stack([rets, vols], axis=2)  # shape: (n_samples, 20, 2)
# %%
X_turnover = train[["AVG_DAILY_TURNOVER"]].values  # shape: (n_samples, 1)
X_ret1 = train[["RET_1"]].values  # shape: (n_samples, 1)

y = train["target"].values

X_seq_tensor = torch.tensor(X_seq, dtype=torch.float32)
X_turnover_tensor = torch.tensor(X_turnover, dtype=torch.float32)
X_ret1_tensor = torch.tensor(X_ret1, dtype=torch.float32)

y_tensor = torch.tensor(y, dtype=torch.float32)

dataset = TensorDataset(X_seq_tensor, X_turnover_tensor, X_ret1_tensor, y_tensor)
train_loader = DataLoader(dataset, batch_size=64, shuffle=True)


# %%
# === 2. Modèle LSTM ===
class LSTMModel(nn.Module):
    def __init__(self, input_dim=2, hidden_dim=64, num_layers=2, seq_len=10):
        super().__init__()
        self.lstm = nn.LSTM(input_dim, hidden_dim, num_layers, batch_first=True)
        # add seq_len * input_dim extra inputs to the MLP
        self.fc = nn.Sequential(
            nn.Linear(hidden_dim + 2 + seq_len * input_dim, 64),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(64, 1),
        )

    def forward(self, x_seq, x_turnover, x_ret1):
        out, _ = self.lstm(x_seq)  # (batch, timesteps, hidden)
        out = out[:, -1, :]  # last hidden state (batch, hidden)
        seq_flat = x_seq.reshape(
            x_seq.size(0), -1
        )  # flatten (batch, seq_len * features)
        combined = torch.cat([out, seq_flat, x_turnover, x_ret1], dim=1)
        return self.fc(combined).squeeze()


class GRUModel(nn.Module):
    def __init__(self, input_dim=2, hidden_dim=64, num_layers=2):
        super().__init__()
        self.gru = nn.GRU(input_dim, hidden_dim, num_layers, batch_first=True)
        self.fc = nn.Linear(hidden_dim + 2, 1)

    def forward(self, x_seq, x_turnover, x_ret1):
        out, _ = self.gru(x_seq)
        out = out[:, -1, :]  # last hidden state
        combined = torch.cat([out, x_turnover, x_ret1], dim=1)
        return self.fc(combined).squeeze()


model = LSTMModel(input_dim=2, hidden_dim=10, num_layers=5, seq_len=20)
criterion = nn.MSELoss()
optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)

# %%
# === 3. Training  ===

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
