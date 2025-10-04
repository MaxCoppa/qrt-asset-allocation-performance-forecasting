"""
Utility for initializing tree-based classifiers (Random Forest, XGBoost,
LightGBM, CatBoost) with predefined hyperparameters.
"""

from sklearn.ensemble import RandomForestRegressor
from xgboost import XGBRegressor
from lightgbm import LGBMRegressor
from catboost import CatBoostRegressor


# Default hyperparameters for supported models
model_params = {
    "rf": {
        "n_estimators": 100,
        "max_depth": 5,
        "random_state": 42,
        "metric": "mse",
    },
    "xgb": {
        "n_estimators": 100,
        "max_depth": 5,
        "learning_rate": 0.05,
        "subsample": 0.8,
        "colsample_bytree": 0.8,
        "random_state": 42,
    },
    "lgbm": {
        "n_estimators": 100,
        "max_depth": -1,
        "learning_rate": 0.05,
        "subsample": 0.8,
        "colsample_bytree": 0.8,
        "random_state": 42,
        "verbose": 0,
        "metric": "mse",
    },
    "cat": {
        "iterations": 100,
        "depth": 6,
        "learning_rate": 0.05,
        "random_seed": 42,
        "verbose": 0,
        "train_dir": None,
    },
}


def get_model(model_type):
    if model_type == "rf":
        return RandomForestRegressor(**model_params["rf"])
    elif model_type == "xgb":
        return XGBRegressor(**model_params["xgb"])
    elif model_type == "lgbm":
        return LGBMRegressor(**model_params["lgbm"])
    elif model_type == "cat":
        return CatBoostRegressor(**model_params["cat"])
    else:
        raise ValueError("Invalid model_type. Choose 'rf', 'xgb', 'lgbm', or 'cat'.")
