# %%
import pandas as pd
from sklearn.metrics import accuracy_score

# %% Load Data

train = pd.read_csv("data/train.csv")
X_val = pd.read_csv("data/X_val.csv")
y_val = pd.read_csv("data/y_val.csv")

# %% Configuration

print(
    accuracy_score((train["RET_1"] > 0).astype(int), (train["target"] > 0).astype(int))
)

# %%
