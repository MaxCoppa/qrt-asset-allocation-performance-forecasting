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
X_seq_2d = X_seq.reshape(-1, n_feat)  # flatten across time + samples

scaler_X = StandardScaler()
X_seq_scaled = scaler_X.fit_transform(X_seq_2d)

X_seq = X_seq_scaled.reshape(n_samples, timesteps, n_feat)

# Static features
alloc = (
    train["ALLOCATION"].apply(lambda x: int(str(x).split("_")[1])).values
)  # (n_samples, 1)

# Targets
y = train["target"].values

# Reshape into (batch, n_ts, timesteps, features)
# assuming dataset is sorted by TS and sample
X_seq = X_seq.reshape(-1, n_ts, timesteps, 2)
alloc = alloc.reshape(-1, n_ts, 1)
y = y.reshape(-1, n_ts)

# %%
X_seq_tensor = torch.tensor(X_seq, dtype=torch.float32)
alloc_tensor = torch.tensor(alloc, dtype=torch.float32)
y_tensor = torch.tensor(y, dtype=torch.float32)

dataset = TensorDataset(X_seq_tensor, alloc_tensor, y_tensor)
train_loader = DataLoader(dataset, batch_size=8, shuffle=True)


# %%
class CrossSeriesModel(nn.Module):
    def __init__(self, seq_dim=2, hidden_dim=64, num_ts=65, num_heads=4, dropout=0.2):
        super().__init__()
        self.num_ts = num_ts
        # shared LSTM encoder (like in LSTMModel)
        self.encoder = nn.LSTM(
            seq_dim, hidden_dim, num_layers=2, batch_first=True, dropout=dropout
        )

        # allocation embedding → map scalar allocation into hidden space
        self.fc_alloc = nn.Linear(1, hidden_dim)

        # cross-series attention
        self.attn = nn.MultiheadAttention(
            hidden_dim, num_heads=num_heads, batch_first=True
        )

        # per-TS prediction head (like your LSTMModel)
        self.fc_out = nn.Sequential(
            nn.Linear(hidden_dim * 2, 64),  # concat seq_emb + alloc_emb
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(64, 1),
        )

    def forward(self, x_seq, alloc):
        # x_seq: (batch, num_ts, timesteps, seq_dim)
        B, N, T, F = x_seq.shape

        # reshape to merge batch and TS
        x_seq_flat = x_seq.reshape(B * N, T, F)
        seq_out, _ = self.encoder(x_seq_flat)  # (B*N, T, hidden_dim)
        seq_emb = seq_out[:, -1, :]  # last hidden state (B*N, hidden_dim)
        seq_emb = seq_emb.reshape(B, N, -1)  # (B, N, hidden_dim)

        # allocation embedding
        alloc_emb = self.fc_alloc(alloc)  # (B, N, hidden_dim)

        # combine sequence + allocation (before attention)
        H = seq_emb + alloc_emb  # (B, N, hidden_dim)

        # cross-series attention: let each TS see others
        H_attn, _ = self.attn(H, H, H)  # (B, N, hidden_dim)

        # final per-TS representation: concat local + cross info
        H_final = torch.cat([H, H_attn], dim=-1)  # (B, N, hidden_dim*2)

        # predict per TS
        out = self.fc_out(H_final).squeeze(-1)  # (B, N)

        return out


# %%
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = CrossSeriesModel(seq_dim=2, hidden_dim=32, num_ts=n_ts, num_heads=1).to(device)

criterion = nn.MSELoss()
optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)

for epoch in range(10):
    model.train()
    epoch_loss, correct, total = 0, 0, 0

    for xb_seq, xb_alloc, yb in tqdm(train_loader, desc=f"Epoch {epoch+1}"):
        xb_seq, xb_alloc, yb = xb_seq.to(device), xb_alloc.to(device), yb.to(device)

        pred = model(xb_seq, xb_alloc)  # (batch, n_ts)
        loss = criterion(pred, yb)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        epoch_loss += loss.item() * len(yb)

        # Directional accuracy
        pred_sign = (pred.detach() > 0).int()
        true_sign = (yb > 0).int()
        correct += (pred_sign == true_sign).sum().item()
        total += yb.numel()

    avg_loss = epoch_loss / len(train_loader.dataset)
    acc = correct / total
    print(f"Epoch {epoch+1}: Loss={avg_loss:.6f}, Acc={acc:.4f}")

# %%
