# %%
import pandas as pd
from tree_based_models import model_selection_using_kfold, get_model, evaluate_model
from feature_engineering import (
    encode_allocation,
    add_average_perf_features,
    create_mean_allocation,
)

# %% Load Data

train = pd.read_csv("data/train.csv")
X_val = pd.read_csv("data/X_val.csv")
y_val = pd.read_csv("data/y_val.csv")

# %%
RET_features = [f"RET_{i}" for i in range(1, 20)]
SIGNED_VOLUME_features = [f"SIGNED_VOLUME_{i}" for i in range(1, 20)]
TURNOVER_features = ["AVG_DAILY_TURNOVER"]
window_sizes = [3, 5, 10, 15, 20]
# %% Feature Engineering

mean_allocation_return = train.groupby("ALLOCATION")["target"].mean().to_dict()


def feature_engineering(X: pd.DataFrame, mean_allocation_return=None) -> pd.DataFrame:
    X = X.pipe(create_mean_allocation, dict_mean=mean_allocation_return).pipe(
        add_average_perf_features, RET_features=RET_features, window_sizes=window_sizes
    )
    return X


X_val = feature_engineering(X_val, mean_allocation_return=mean_allocation_return)

# %% Configuration

features = [col for col in X_val.columns if col not in ["ROW_ID", "TS", "ALLOCATION"]]
target_name = "target"
unique_id = "TS"
model_name = "xgb"
# %% Model Selection Evaluation

model_selection_using_kfold(
    data=train,
    target=target_name,
    features=features,
    feat_engineering=feature_engineering,
    model_type=model_name,
    unique_id=unique_id,
    plot_ft_importance=True,
)
# %% Train Model

train = feature_engineering(train)

model = get_model(model_name)
model.fit(train[features], train[target_name])

_ = evaluate_model(
    model=model, X=X_val[features], y=y_val[target_name], verbose=True, log=True
)
# %% Predicion

X_train = pd.read_csv("data/X_train.csv")
y_train = pd.read_csv("data/y_train.csv")
X_test = pd.read_csv("data/X_test.csv")

mean_allocation_return = (
    X_train.merge(y_train, on="ROW_ID").groupby("ALLOCATION")["target"].mean().to_dict()
)

X_train = feature_engineering(X_train, mean_allocation_return)
X_test = feature_engineering(X_test, mean_allocation_return)


model = get_model(model_name)
model.fit(X_train[features], y_train[target_name])

preds_sub = model.predict(X_test[features])
preds_sub = pd.DataFrame(preds_sub, index=X_test[unique_id], columns=[target_name])

# (preds_sub > 0).astype(int).to_csv(f"predictions/preds_{model_name}.csv")
# %%
