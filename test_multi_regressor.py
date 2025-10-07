# %%
import pandas as pd

# Chargement
train = pd.read_csv("data/train.csv")

# Features et target
features = [
    col for col in train.columns if col not in ["ROW_ID", "TS", "ALLOCATION", "target"]
]
target_name = "target"

# %%
# Colonnes à pivoter (toutes sauf celles qui identifient TS et ALLOCATION)
features = [col for col in train.columns if col not in ["ROW_ID", "TS", "ALLOCATION"]]

# Transformation en format wide
df_wide = train.pivot(index="TS", columns="ALLOCATION", values=features)

# Les colonnes deviennent multi-index (feature, allocation)
# On aplatit en concaténant "ALLOCATION_FEATURE"
df_wide.columns = [f"{alloc}_{feat}" for feat, alloc in df_wide.columns]

# Reset index pour remettre TS comme colonne
df_wide = df_wide.reset_index()

# %%
target_col = [f"ALLOCATION_{i}_target" for i in range(1, 66)]
features = [col for col in df_wide.columns if col not in (target_col + ["TS"])]
X_train = df_wide[features]
y_train = df_wide[target_col]
# %%
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset

# Convert to torch tensors
X_tensor = torch.tensor(X_train.values, dtype=torch.float32)
y_tensor = torch.tensor(y_train.values, dtype=torch.float32)

# Dataset & DataLoader
dataset = TensorDataset(X_tensor, y_tensor)
loader = DataLoader(dataset, batch_size=64, shuffle=True)


# Define a neural network for multivariate regression
class MultiRegressor(nn.Module):
    def __init__(self, input_dim, output_dim):
        super(MultiRegressor, self).__init__()
        self.net = nn.Sequential(
            nn.Linear(input_dim, 256),
            nn.ReLU(),
            nn.Linear(256, 128),
            nn.ReLU(),
            nn.Linear(128, output_dim),  # final layer: 65 outputs
        )

    def forward(self, x):
        return self.net(x)


# Model, loss, optimizer
model = MultiRegressor(input_dim=X_train.shape[1], output_dim=y_train.shape[1])
criterion = nn.MSELoss()
optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)

# Training loop
for epoch in range(20):  # increase for better results
    for xb, yb in loader:
        optimizer.zero_grad()
        preds = model(xb)
        loss = criterion(preds, yb)
        loss.backward()
        optimizer.step()
    print(f"Epoch {epoch+1}, Loss: {loss.item():.4f}")

# Predictions
with torch.no_grad():
    y_pred = model(X_tensor).numpy()
print("Pred shape:", y_pred.shape)  # (n_samples, 65)
# %%
