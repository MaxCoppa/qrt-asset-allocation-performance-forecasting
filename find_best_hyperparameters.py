# %%
import optuna
import pandas as pd
import numpy as np
import xgboost as xgb
from sklearn.metrics import mean_squared_error

from feature_engineering import (
    encode_allocation,
    add_average_perf_features,
    create_allocation_features,
    add_average_volume_features,
    add_near_time_comparison_features,
    add_ratio_difference_features,
    create_mean_allocation,
    add_strategy_features,
    add_cross_sectional_features,
    add_statistical_features,
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

dict_mean = train.groupby("ALLOCATION")["target"].mean().to_dict()


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
        .pipe(
            add_statistical_features,
            RET_features=RET_features,
            SIGNED_VOLUME_features=SIGNED_VOLUME_features,
        )
        .pipe(
            add_average_volume_features, SIGNED_VOLUME_features=SIGNED_VOLUME_features
        )
        .pipe(create_mean_allocation, dict_mean=dict_mean)
        .pipe(add_cross_sectional_features, base_cols=["RET_1", "RET_3"])
    )

    return X


X_feat = feature_engineering(train)
features = [
    col
    for col in X_feat.columns
    if col not in ["ROW_ID", "TS", "ALLOCATION", "target"] + SIGNED_VOLUME_features
]
# %% Configuration

target_name = "target"
unique_id = "TS"
model_name = "xgb"

X_train = feature_engineering(train)[features]
y_train = train[target_name]

X_val = feature_engineering(X_val)

# Convert to DMatrix
dtrain = xgb.DMatrix(X_train, label=y_train)
dval = xgb.DMatrix(X_val[features], label=y_val[target_name])


def objective(trial):
    params = {
        "booster": "gbtree",
        "objective": "reg:squarederror",
        "eval_metric": "rmse",
        "eta": trial.suggest_float("learning_rate", 0.01, 0.3, log=True),
        "max_depth": trial.suggest_int("max_depth", 3, 12),
        "min_child_weight": trial.suggest_float("min_child_weight", 1.0, 10.0),
        "subsample": trial.suggest_float("subsample", 0.5, 1.0),
        "colsample_bytree": trial.suggest_float("colsample_bytree", 0.5, 1.0),
        "gamma": trial.suggest_float("gamma", 0, 5),
        "alpha": trial.suggest_float("reg_alpha", 1e-8, 10.0, log=True),  # L1
        "lambda": trial.suggest_float("reg_lambda", 1e-8, 10.0, log=True),  # L2
    }

    evals = [(dtrain, "train"), (dval, "valid")]

    model = xgb.train(
        params,
        dtrain,
        num_boost_round=5000,
        evals=evals,
        early_stopping_rounds=100,
        verbose_eval=False,
    )

    preds = model.predict(dval, iteration_range=(0, model.best_iteration + 1))
    rmse = np.sqrt(mean_squared_error(y_val[target_name], preds))
    return rmse


# Run optimization
study = optuna.create_study(direction="minimize")
study.optimize(objective, n_trials=50)

print("Best RMSE:", study.best_value)
print("Best params:", study.best_trial.params)

# %%
