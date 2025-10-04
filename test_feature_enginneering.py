# %%
import pandas as pd
from tree_based_models import model_selection_using_kfold, get_model, evaluate_model
from feature_engineering import (
    encode_allocation,
    add_average_perf_features,
    split_data,
    create_mean_allocation,
)

# %% Load Data

X_train = pd.read_csv("data/X_train.csv")
X_test = pd.read_csv("data/X_test.csv")
y_train = pd.read_csv("data/y_train.csv")

# %%
n_date_test = X_test["TS"].nunique()

X_train, y_train, X_val, y_val = split_data(X_train, y_train, n_date_test)

# %%
RET_features = [f"RET_{i}" for i in range(1, 20)]
SIGNED_VOLUME_features = [f"SIGNED_VOLUME_{i}" for i in range(1, 20)]
TURNOVER_features = ["AVG_DAILY_TURNOVER"]
window_sizes = [3, 5, 10, 15, 20]
# %% Feature Engineering

mean_allocation_return = (
    X_train.merge(y_train, on="ROW_ID").groupby("ALLOCATION")["target"].mean().to_dict()
)


def feature_engineering(X: pd.DataFrame) -> pd.DataFrame:
    X = X.pipe(create_mean_allocation, dict_mean=mean_allocation_return).pipe(
        add_average_perf_features, RET_features=RET_features, window_sizes=window_sizes
    )
    return X


X_test = feature_engineering(X_test)

# %% Configuration

features = [col for col in X_test.columns if col not in ["ROW_ID", "TS", "ALLOCATION"]]
target_name = "target"
unique_id = "TS"
model_name = "xgb"
# %% Model Selection Evaluation

model_selection_using_kfold(
    X=X_train,
    y=y_train,
    target=target_name,
    features=features,
    feat_engineering=feature_engineering,
    model_type=model_name,
    unique_id=unique_id,
    plot_ft_importance=True,
)
# %% Train Model

X_train = feature_engineering(X_train)
X_val = feature_engineering(X_val)

model = get_model(model_name)
model.fit(X_train[features], y_train[target_name])

_ = evaluate_model(
    model=model,
    X=X_val[features],
    y=y_val[target_name],
    verbose=True,
)
# %% Predicion

preds_sub = model.predict(X_test[features])
preds_sub = pd.DataFrame(preds_sub, index=X_test[unique_id], columns=[target_name])

# (preds_sub > 0).astype(int).to_csv(f"data/preds_{model_name}.csv")
# %%
