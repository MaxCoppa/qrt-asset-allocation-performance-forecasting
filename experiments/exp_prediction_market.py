# %%
import pandas as pd
from tree_based_models import (
    get_model,
    evaluate_model_market,
    model_selection_respect_market,
)
from data_engineering import feature_engineering as fe


# %% Load Data

train = pd.read_csv("../data/train.csv")
X_val = pd.read_csv("../data/X_val.csv")
y_val = pd.read_csv("../data/y_val.csv")

# %%
RET_features = [f"RET_{i}" for i in range(1, 21)]
SIGNED_VOLUME_features = [f"SIGNED_VOLUME_{i}" for i in range(1, 21)]
TURNOVER_features = ["AVG_DAILY_TURNOVER"]

window_sizes = [1, 3, 5, 10, 15, 20]
# %% Feature Engineering


def feature_engineering(
    X: pd.DataFrame,
) -> pd.DataFrame:
    X = X.pipe(
        fe.add_ret_minus_market,
        RET_features=RET_features,
        rolling_average=1,
        group_col="TS",
    )

    return X


X_feat = feature_engineering(train)
train = feature_engineering(train)
# %%
features = (
    [f"SPREAD_RET_{i}" for i in range(1, 14)]
    + TURNOVER_features
    + SIGNED_VOLUME_features
)
market_feature = "ALLOC_AVG_PAST_PERF"
# %%
target_name = "SPREAD_target"
unique_id = "TS"
model_name = "xgb_opt"
# %% Model Selection Evaluation


model_selection_respect_market(
    data=train,
    target=target_name,
    true_target="target",
    features=features,
    market_feature=market_feature,
    model_type=model_name,
    unique_id=unique_id,
    plot_ft_importance=True,
    n_splits=4,
)

# %% Train Model


val = X_val.merge(y_val, on="ROW_ID")
val = feature_engineering(val)

model = get_model(model_name)
model.fit(train[features], train[target_name])

_ = evaluate_model_market(
    model=model,
    X=val[features],
    y=val["target"],
    y_market=val[market_feature],
    verbose=True,
    log=False,
)
# %% Predicion

X_train = pd.read_csv("../data/X_train.csv")
y_train = pd.read_csv("../data/y_train.csv")
X_test = pd.read_csv("../data/X_test.csv")


X_train = feature_engineering(X_train)
X_test = feature_engineering(X_test)


model = get_model(model_name)
model.fit(X_train[features], y_train[target_name])

preds_sub = model.predict(X_test[features])
preds_sub = pd.DataFrame(preds_sub, index=X_test["ROW_ID"], columns=[target_name])
preds_sub["target"] = 1
(preds_sub > 0).astype(int).to_csv(f"predictions/preds_{model_name}.csv")
# %%
