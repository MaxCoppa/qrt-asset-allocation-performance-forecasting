# %%
from data_engineering import feature_engineering as fe
import pandas as pd
from data_engineering.data_preprocessing import split_data, extract_unique_train

# %% Load Data

X_train = pd.read_csv("data/X_train.csv")
X_test = pd.read_csv("data/X_test.csv")
y_train = pd.read_csv("data/y_train.csv")

save = True

# %%
n_val = int(172445 * 0.1 / X_train["ALLOCATION"].nunique())

X_train, y_train, X_val, y_val = split_data(X_train, y_train, n_val)
train = X_train.merge(y_train, on="ROW_ID")  # Create a merge DataFrame

# %%
if save:
    train.to_csv("data/train.csv", index=False)
    X_val.to_csv("data/X_val.csv", index=False)
    y_val.to_csv("data/y_val.csv", index=False)

# %%

df_unique = extract_unique_train(
    data=train,
    allocation="ALLOCATION_01",
    output_csv="data/train_unique.csv",
    save=save,
)

# %%
