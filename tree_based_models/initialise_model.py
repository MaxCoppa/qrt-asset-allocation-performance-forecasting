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
        "alpha": 1.0,
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
        "n_estimators": 10,  # number of boosting rounds
        "learning_rate": 0.1,  # shrinkage
        "max_depth": 5,  # tree depth
        "subsample": 0.8,  # row sampling
        "colsample_bytree": 0.8,  # feature sampling
        "random_state": 42,
    },
    # {
    #     "n_estimators": 100,
    #     "max_depth": 5,
    #     "learning_rate": 0.05,
    #     "subsample": 0.8,
    #     "colsample_bytree": 0.8,
    #     "random_state": 42,
    # },
    "lgbm": {
        "n_estimators": 100,
        "max_depth": -1,
        "learning_rate": 0.05,
        "subsample": 0.8,
        "colsample_bytree": 0.8,
        "random_state": 42,
        "verbose": -1,
        "metric": "mse",
        "scale_pos_weight": 1 / 2,
    },
    "cat": {
        "iterations": 100,
        "depth": 6,
        "learning_rate": 0.05,
        "random_seed": 42,
        "verbose": 0,
    },
    "lgbm_opt": {
        "learning_rate": 0.047090685234810956,
        "num_leaves": 209,
        "max_depth": 4,
        "min_child_samples": 28,
        "subsample": 0.7349165219161458,
        "colsample_bytree": 0.6563592672166201,
        "lambda_l1": 2.0966568965626345e-06,
        "lambda_l2": 7.814815853457007,
        "verbose": -1,
        "metric": "mse",
    },
    "xgb_opt": {
        "learning_rate": 0.2553671687393621,
        "max_depth": 12,
        "min_child_weight": 5.539596769759182,
        "subsample": 0.5533027565817632,
        "colsample_bytree": 0.6958073921678826,
        "gamma": 2.7772619502519524,
        "reg_alpha": 0.0006987666658294355,
        "reg_lambda": 0.029502980270596287,
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
    elif model_type == "xgb_opt":
        return XGBRegressor(**model_params["xgb"])
    elif model_type == "lgbm":
        return LGBMRegressor(**model_params["lgbm"])
    elif model_type == "lgbm_opt":
        return LGBMRegressor(**model_params["lgbm_opt"])
    elif model_type == "cat":
        return CatBoostRegressor(**model_params["cat"])
    else:
        raise ValueError(
            "Invalid model_type. Choose from 'linear', 'ridge', 'lasso', 'rf', 'xgb', 'lgbm', 'cat'."
        )
