# %%
import pandas as pd
from tree_based_models import get_model, evaluate_ensemble_model
from feature_engineering import (
    add_average_perf_features,
    create_mean_allocation,
)

# %% Load Data

train = pd.read_csv("data/train.csv")
X_val = pd.read_csv("data/X_val.csv")
y_val = pd.read_csv("data/y_val.csv")

# %% Feature Engineering

RET_features = [f"RET_{i}" for i in range(1, 20)]
SIGNED_VOLUME_features = [f"SIGNED_VOLUME_{i}" for i in range(1, 20)]
TURNOVER_features = ["AVG_DAILY_TURNOVER"]
window_sizes = [3, 5, 10, 15, 20]


mean_allocation_return = train.groupby("ALLOCATION")["target"].mean().to_dict()


def feature_engineering(X: pd.DataFrame, mean_allocation_return=None) -> pd.DataFrame:
    X = X.pipe(create_mean_allocation, dict_mean=mean_allocation_return).pipe(
        add_average_perf_features, RET_features=RET_features, window_sizes=window_sizes
    )
    return X


train = feature_engineering(train, mean_allocation_return=mean_allocation_return)
X_val = feature_engineering(X_val, mean_allocation_return=mean_allocation_return)

# %% Configuration

features = [
    col for col in train.columns if col not in ["ROW_ID", "TS", "ALLOCATION", "target"]
]
target_name = "target"
model_names = "xgb:lgbm"
models = []


# %% Train Model

for model_name in model_names.split(":"):
    model = get_model(model_name)
    model.fit(train[features], train[target_name])
    models.append(model)

_ = evaluate_ensemble_model(
    models=models, X=X_val[features], y=y_val[target_name], verbose=True, log=True
)
# %% Predicion

# (preds_sub > 0).astype(int).to_csv("data/preds_test.csv")
# %%
