# %%
import pandas as pd
from data_engineering.data_preprocessing import split_data, extract_unique_train

# %% Configuration
SAVE_OUTPUTS = False
VAL_RATIO = 0.1

# %% Load Data
X_train = pd.read_csv("data/X_train.csv")
X_test = pd.read_csv("data/X_test.csv")
y_train = pd.read_csv("data/y_train.csv")

print(
    f"Loaded -> X_train: {X_train.shape}, X_test: {X_test.shape}, y_train: {y_train.shape}"
)

# %% Initial Train / Validation Split
n_val = int(len(X_train) * VAL_RATIO / X_train["ALLOCATION"].nunique())
X_train_split, y_train_split, X_val, y_val = split_data(X_train, y_train, n_val)

train = X_train_split.merge(y_train_split, on="ROW_ID")

print(
    f"Initial split -> Train: {train.shape}, X_val: {X_val.shape}, y_val: {y_val.shape}"
)

if SAVE_OUTPUTS:
    train.to_csv("data/train.csv", index=False)
    X_val.to_csv("data/X_val.csv", index=False)
    y_val.to_csv("data/y_val.csv", index=False)

# %% Extract Unique Allocations
train_full = X_train.merge(y_train, on="ROW_ID")

df_unique = extract_unique_train(
    data=train_full,
    allocation="ALLOCATION_01",
    output_csv="data/train_unique.csv",
    save=SAVE_OUTPUTS,
)

print(f"Unique dataset -> df_unique: {df_unique.shape}")

# %% Prepare Unique Train / Validation Split
X_train_unique = df_unique.drop(columns="target")
y_train_unique = df_unique[["ROW_ID", "target"]]

n_val_unique = int(
    len(X_train_unique) * VAL_RATIO / X_train_unique["ALLOCATION"].nunique()
)
X_train_split, y_train_split, X_val, y_val = split_data(
    X_train_unique, y_train_unique, n_val_unique
)

train_unique = X_train_split.merge(y_train_split, on="ROW_ID")

print(
    f"Unique split -> Train_unique: {train_unique.shape}, X_val: {X_val.shape}, y_val: {y_val.shape}"
)

if False:
    train_unique.to_csv("data/train.csv", index=False)
    X_val.to_csv("data/X_val.csv", index=False)
    y_val.to_csv("data/y_val.csv", index=False)

# %%
