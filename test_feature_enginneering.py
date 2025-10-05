# %%
import pandas as pd
from tree_based_models import model_selection_using_kfold, get_model, evaluate_model
from feature_engineering import (
    encode_allocation,
    add_average_perf_features,
    create_allocation_features,
    add_average_volume_features,
    add_near_time_comparison_features,
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

features = RET_features + TURNOVER_features + SIGNED_VOLUME_features

features = features + [f"AVERAGE_PERF_{i}" for i in window_sizes]
features = features + [f"ALLOCATIONS_AVERAGE_PERF_{i}" for i in window_sizes]
# features = features + ["RET_diff_1_2","RET_ratio_1_2","VOL_diff_1_2","VOL_ratio_1_2","IMPACT_diff_1_2","IMPACT_ratio_1_2"]
# features = features + ["AVG_DAILY_TURNOVER_ALLOCATION"]
# %% Feature Engineering


def feature_engineering(
    X: pd.DataFrame,
) -> pd.DataFrame:
    X = (
        X.pipe(
            add_average_perf_features,
            RET_features=RET_features,
            window_sizes=window_sizes,
            group_col="TS",
        )
        # .pipe(
        #     add_near_time_comparison_features,
        #     RET_features=RET_features,
        #     SIGNED_VOLUME_features=SIGNED_VOLUME_features,
        # )
    )
    return X


# %% Configuration

target_name = "target"
unique_id = "TS"
model_name = "lgbm"
# %% Model Selection Evaluation

model_selection_using_kfold(
    data=train,
    target=target_name,
    features=features,
    feat_engineering=feature_engineering,
    model_type=model_name,
    unique_id=unique_id,
    plot_ft_importance=True,
    n_splits=6,
)
# %% Train Model

train = feature_engineering(train)
X_val = feature_engineering(X_val)

model = get_model(model_name)
model.fit(train[features], train[target_name])

_ = evaluate_model(
    model=model, X=X_val[features], y=y_val[target_name], verbose=True, log=False
)
# %% Predicion

X_train = pd.read_csv("data/X_train.csv")
y_train = pd.read_csv("data/y_train.csv")
X_test = pd.read_csv("data/X_test.csv")


X_train = feature_engineering(X_train)
X_test = feature_engineering(X_test)


model = get_model(model_name)
model.fit(X_train[features], y_train[target_name])

preds_sub = model.predict(X_test[features])
preds_sub = pd.DataFrame(preds_sub, index=X_test[unique_id], columns=[target_name])

# (preds_sub > 0).astype(int).to_csv(f"predictions/preds_{model_name}.csv")
# %%
