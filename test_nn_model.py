# %%
import pandas as pd
from deep_learning_models import nn_model_selection_using_kfold

# %% Load Data

train = pd.read_csv("data/train.csv")
X_val = pd.read_csv("data/X_val.csv")
y_val = pd.read_csv("data/y_val.csv")

# %% Configuration

features = [
    col for col in train.columns if col not in ["ROW_ID", "TS", "ALLOCATION", "target"]
]
target_name = "target"
unique_id = "TS"

# %%
params = {
    "enc_units": [24, 8],  # encoder layers
    "dec_units": [8, 24],  # decoder layers (mirror or custom)
    "mlp_units": [8, 2],  # classifier layers
    "dropout_rate": 0.1,
    "lr": 0.001,
}
# %%
_ = nn_model_selection_using_kfold(
    data=train,
    target=target_name,
    features=features,
    enc_units=params["enc_units"],
    dec_units=params["dec_units"],
    mlp_units=params["mlp_units"],
    dropout_rate=params["dropout_rate"],
    unique_id=unique_id,
    n_splits=2,
    num_epochs=10,
    batch_size=325,
    lr=params["lr"],
    device="cpu",
    alpha=0.5,
    beta=0.5,
)
# %%
