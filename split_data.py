# %%

import pandas as pd
from feature_engineering import split_data

# %% Load Data

X_train = pd.read_csv("data/X_train.csv")
X_test = pd.read_csv("data/X_test.csv")
y_train = pd.read_csv("data/y_train.csv")

# %%
n_date_test = X_test["TS"].nunique()

X_train, y_train, X_val, y_val = split_data(X_train, y_train, n_date_test)
# %%
X_train.shape, X_val.shape  # = ((172445, 44), (7800, 44))
# %%
train = X_train.merge(y_train, on="ROW_ID")

# %%
train.to_csv("data/train.csv", index=False)  # Data For Model Selection
X_val.to_csv("data/X_val.csv", index=False)  # Data For Model Validation
y_val.to_csv("data/y_val.csv", index=False)  # Data For Model Validation
# %%
