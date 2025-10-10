# %%

import pandas as pd
from feature_engineering import split_data

# %% Load Data

X_train = pd.read_csv("data/X_train.csv")
X_test = pd.read_csv("data/X_test.csv")
y_train = pd.read_csv("data/y_train.csv")

# %%
n_val = int(172445 * 0.1 / X_train["ALLOCATION"].nunique())

X_train, y_train, X_val, y_val = split_data(X_train, y_train, n_val)
# %%
X_train.shape, X_val.shape  # = ((162955, 44), (17290, 44))
# %%
train = X_train.merge(y_train, on="ROW_ID")

# %%
train.to_csv("data/train.csv", index=False)
X_val.to_csv("data/X_val.csv", index=False)
y_val.to_csv("data/y_val.csv", index=False)
# %%


# %%
