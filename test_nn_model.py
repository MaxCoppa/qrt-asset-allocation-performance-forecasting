# %%
import pandas as pd
from deep_learning_models import tabnet_model_selection_using_kfold


# %% Load Data

train = pd.read_csv("data/train.csv")
X_val = pd.read_csv("data/X_val.csv")
y_val = pd.read_csv("data/y_val.csv")

# %%
RET_features = [f"RET_{i}" for i in range(1, 20)]
SIGNED_VOLUME_features = [f"SIGNED_VOLUME_{i}" for i in range(1, 20)]
TURNOVER_features = ["AVG_DAILY_TURNOVER"]

# %% Configuration

features = [
    col for col in train.columns if col not in ["ROW_ID", "TS", "ALLOCATION", "target"]
]
target_name = "target"
unique_id = "TS"

# %%
params = {
    "enc_units": [56, 24, 8],  # encoder layers
    "dec_units": [8, 24],  # decoder layers (mirror or custom)
    "mlp_units": [8, 2],  # classifier layers
    "dropout_rate": 0.2,
    "lr": 0.001,
}
# %%
_ = tabnet_model_selection_using_kfold(
    data=train,
    target=target_name,
    features=features,
    n_splits=4,
    feat_engineering=None,
    num_epochs=10,
    patience=20,
    batch_size=512,
    virtual_batch_size=128,
    lr=2e-2,
    device="cpu",
)


# %%

train.shape

# %%
