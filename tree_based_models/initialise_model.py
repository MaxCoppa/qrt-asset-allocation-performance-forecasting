"""
Utility for initializing regression models:
- Simple linear models (LinearRegression, Ridge, Lasso)
- Tree-based models (Random Forest, XGBoost, LightGBM, CatBoost)
with predefined hyperparameters.
"""

from sklearn.linear_model import LinearRegression, Ridge, Lasso
from sklearn.ensemble import RandomForestRegressor
from xgboost import XGBRegressor
from lightgbm import LGBMRegressor
from catboost import CatBoostRegressor


# Default hyperparameters for supported models
model_params = {
    # Simple linear models
    "linear": {
        "fit_intercept": True,
    },
    "ridge": {
        "alpha": 1e-2,
        "fit_intercept": True,
        "random_state": 42,
    },
    "lasso": {
        "alpha": 0.1,
        "fit_intercept": True,
        "random_state": 42,
    },
    # Tree-based regressors
    "rf": {
        "n_estimators": 100,
        "max_depth": 5,
        "random_state": 42,
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
        "verbose": -1,
        "metric": "mse",
    },
    "cat": {
        "iterations": 100,
        "depth": 6,
        "learning_rate": 0.05,
        "random_seed": 42,
        "verbose": 0,
    },
}


def get_model(model_type: str):
    """
    Returns a regression model with predefined hyperparameters.

    Parameters
    ----------
    model_type : str
        One of {"linear", "ridge", "lasso", "rf", "xgb", "lgbm", "cat"}.

    Returns
    -------
    model : Regressor
        Instantiated regression model.
    """
    if model_type == "linear":
        return LinearRegression(**model_params["linear"])
    elif model_type == "ridge":
        return Ridge(**model_params["ridge"])
    elif model_type == "lasso":
        return Lasso(**model_params["lasso"])
    elif model_type == "rf":
        return RandomForestRegressor(**model_params["rf"])
    elif model_type == "xgb":
        return XGBRegressor(**model_params["xgb"])
    elif model_type == "lgbm":
        return LGBMRegressor(**model_params["lgbm"])
    elif model_type == "cat":
        return CatBoostRegressor(**model_params["cat"])
    else:
        raise ValueError(
            "Invalid model_type. Choose from 'linear', 'ridge', 'lasso', 'rf', 'xgb', 'lgbm', 'cat'."
        )
