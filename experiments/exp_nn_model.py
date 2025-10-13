# %%
import pandas as pd
from deep_learning_models import nn_model_selection_using_kfold

# %% Load Data

train = pd.read_csv("../data/train.csv")
X_val = pd.read_csv("../data/X_val.csv")
y_val = pd.read_csv("../data/y_val.csv")

# %% Creation Cross Data

features_init = [
    col for col in train.columns if col not in ["ROW_ID", "TS", "ALLOCATION"]
]
data = train.pivot(index="TS", columns="ALLOCATION", values=features_init)
data.columns = [f"{alloc}_{feat}" for feat, alloc in data.columns]
data = data.reset_index()

data.head()
# %% Configuration

features = [
    col
    for col in data.columns
    if col
    not in (
        ["ROW_ID", "TS", "ALLOCATION", "target"]
        + [col for col in data.columns if "target" in col]
    )
]
allocation_name = "ALLOCATION_01"
unique_id = "TS"

params = {
    "enc_units": [8],  # encoder layers
    "dec_units": [8],  # decoder layers (mirror or custom)
    "mlp_units": [2],  # classifier layers
    "dropout_rate": 0.1,
    "lr": 0.5,
    "num_epochs": 40,
    "batch_size": 16,
    "alpha": 5,
    "beta": 5,
    "device": "cpu",
}

# %% Cross Validation for NN
nn_model_selection_using_kfold(
    data=data.copy(),
    features=features,
    allocation_name=allocation_name,
    unique_id=unique_id,
    **params,
)
# %%
