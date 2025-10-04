# %%
import pandas as pd
from tree_based_models import model_selection_by_allocation, get_model, evaluate_model

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
model_name = "xgb"

# %% Model Selection Evaluation

model_selection_by_allocation(
    data=train,
    target=target_name,
    features=features,
    model_type=model_name,
    unique_id=unique_id,
)

# %%
