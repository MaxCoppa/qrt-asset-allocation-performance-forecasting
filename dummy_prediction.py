# %%
import pandas as pd
from sklearn.metrics import accuracy_score
import numpy as np

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
dict_allocation = train.groupby("ALLOCATION")["target"].mean().to_dict()
preds = ((X_val["ALLOCATION"].map(dict_allocation)) > 0).astype(int)
print(accuracy_score(y_true=(y_val["target"] > 0).astype(int), y_pred=preds))


# %%
X_test = pd.read_csv("data/X_test.csv")
preds_sub = np.array(X_test["ALLOCATION"].map(dict_allocation))
preds_sub = pd.DataFrame(
    (preds_sub > 0).astype(int), index=X_test["ROW_ID"], columns=["target"]
)

# %%
preds_sub.to_csv("predictions/dummy_prediction_v2.csv")
# %%
