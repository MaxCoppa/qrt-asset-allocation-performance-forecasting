# %%
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset
import pandas as pd
import numpy as np
from tqdm import tqdm
from sklearn.preprocessing import StandardScaler


# %% Preprocess into 4D tensor
train = pd.read_csv("data/train.csv")

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
# assuming dataset is sorted by TS and sample
X_seq = X_seq.reshape(n_ts, -1, timesteps, 2)
y = y.reshape(n_ts, -1)


# %%
X_seq_tensor = torch.tensor(X_seq, dtype=torch.float32)
y_tensor = torch.tensor(y, dtype=torch.float32)

dataset = TensorDataset(X_seq_tensor, y_tensor)
train_loader = DataLoader(dataset, batch_size=8, shuffle=True)


# %%


class CrossSeriesModel(nn.Module):
    def __init__(self, seq_dim=2, hidden_dim=64, num_heads=4, dropout=0.2):
        super().__init__()

        # Encode each TS with LSTM
        self.encoder = nn.LSTM(
            input_size=seq_dim,
            hidden_size=hidden_dim,
            num_layers=2,
            batch_first=True,
            dropout=dropout,
        )

        # Cross-series self-attention
        self.attn = nn.MultiheadAttention(
            embed_dim=hidden_dim, num_heads=num_heads, batch_first=True
        )

        # Projection for raw x_seq → hidden space
        self.proj_raw = nn.Linear(seq_dim, hidden_dim)

        # Final prediction head
        self.fc_out = nn.Sequential(
            nn.Linear(hidden_dim * 3, 64),  # raw + seq_emb + attn
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(64, 1),
        )

    def forward(self, x):
        """
        x: (B, N, T, F)
        B = batch size
        N = number of TS
        T = timesteps
        F = input features
        """
        B, N, T, F = x.shape

        # Encode each TS independently with LSTM
        x_flat = x.reshape(B * N, T, F)
        seq_out, _ = self.encoder(x_flat)
        seq_emb = seq_out[:, -1, :]  # (B*N, hidden_dim)
        seq_emb = seq_emb.reshape(B, N, -1)  # (B, N, hidden_dim)

        # Cross-series self-attention
        attn_out, _ = self.attn(seq_emb, seq_emb, seq_emb)

        # Residual connection (like Transformer: seq_emb + attn_out)
        seq_plus_attn = seq_emb + attn_out  # (B, N, hidden_dim)

        # Raw input features → hidden space (use last timestep as summary)
        raw_last = x[:, :, -1, :]  # (B, N, F)
        raw_emb = self.proj_raw(raw_last)  # (B, N, hidden_dim)

        # Concatenate raw info + seq_emb + attention
        h = torch.cat([raw_emb, seq_emb, seq_plus_attn], dim=-1)  # (B, N, 3*hidden_dim)

        # Per-TS prediction
        return self.fc_out(h).squeeze(-1)  # (B, N)


# %%
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = CrossSeriesModel(seq_dim=2, hidden_dim=32, num_heads=1).to(device)

criterion = nn.MSELoss()
optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)

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
