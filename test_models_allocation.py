# %%
import pandas as pd
from tree_based_models import model_selection_by_allocation, get_model, evaluate_model

# %% Load Data

X_train = pd.read_csv("data/X_train.csv")
X_test = pd.read_csv("data/X_test.csv")
y_train = pd.read_csv("data/y_train.csv")

# %% Configuration

FEATURES = [col for col in X_train.columns if col not in ["ROW_ID", "TS", "ALLOCATION"]]
target_name = "target"
unique_id = "ROW_ID"
model_name = "xgb"

# %% Model Selection Evaluation

model_selection_by_allocation(
    X=X_train,
    y=y_train,
    target=target_name,
    features=FEATURES,
    model_type=model_name,
    unique_id=unique_id,
)

# %%
