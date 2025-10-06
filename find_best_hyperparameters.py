# %%
import optuna
import pandas as pd
import numpy as np
import xgboost as xgb
from sklearn.metrics import mean_squared_error

# Load data
train = pd.read_csv("data/train.csv")
X_val = pd.read_csv("data/X_val.csv")
y_val = pd.read_csv("data/y_val.csv")

# Features & target
features = [
    col for col in train.columns if col not in ["ROW_ID", "TS", "ALLOCATION", "target"]
]
target_name = "target"

X_train = train[features]
y_train = train[target_name]

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
