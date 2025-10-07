# %%
import pandas as pd
from sklearn.metrics import accuracy_score

# %% Load Data

train = pd.read_csv("data/train.csv")
X_val = pd.read_csv("data/X_val.csv")
y_val = pd.read_csv("data/y_val.csv")


# %% Configuration

preds = (
    (0.9 * train["RET_1"] + 0.1 * train.groupby("TS")["RET_1"].transform("mean")) > 0
).astype(int)
print(accuracy_score(y_true=(train["target"] > 0).astype(int), y_pred=preds))


# %%
