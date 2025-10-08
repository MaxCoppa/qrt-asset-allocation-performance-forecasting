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
import numpy as np
from sklearn.ensemble import VotingRegressor


class EnsembleRegressor:
    def __init__(self, models, weights=None):
        self.models = models
        self.weights = weights if weights is not None else [1] * len(models)

    def fit(self, X, y):
        for model in self.models:
            model.fit(X, y)
        return self

    def predict(self, X):
        preds = np.column_stack([model.predict(X) for model in self.models])
        weights = np.array(self.weights) / np.sum(self.weights)
        return np.dot(preds, weights)


def weighted_mse(y_true, y_pred):
    weight = 1.0 / (np.abs(y_true) + 1e-6)
    grad = -2 * (y_true - y_pred) * weight
    hess = 2 * weight
    return grad, hess


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
    },
    "lgbm_chat": {
        "learning_rate": 0.05,
        "n_estimators": 2000,  # high cap; rely on early stopping
        "num_leaves": 64,
        "min_data_in_leaf": 100,
        "feature_fraction": 0.8,
        "bagging_fraction": 0.8,
        "bagging_freq": 1,
        "verbose": -1,
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
    "xgb_objective": {
        "n_estimators": 300,
        "max_depth": 5,
        "learning_rate": 0.05,
        "subsample": 0.8,
        "colsample_bytree": 0.8,
        "random_state": 42,
        "objective": weighted_mse,
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
        return XGBRegressor(**model_params["xgb_opt"])
    elif model_type == "xgb_objective":
        return XGBRegressor(**model_params["xgb_objective"])
    elif model_type == "lgbm":
        return LGBMRegressor(**model_params["lgbm"])
    elif model_type == "lgbm_opt":
        return LGBMRegressor(**model_params["lgbm_opt"])
    elif model_type == "lgbm_chat":
        return LGBMRegressor(**model_params["lgbm_chat"])
    elif model_type == "cat":
        return CatBoostRegressor(**model_params["cat"])
    elif model_type == "voting":
        return EnsembleRegressor(
            models=[
                LGBMRegressor(**model_params["lgbm"]),
                XGBRegressor(**model_params["xgb"]),
            ]
        )
    else:
        raise ValueError(
            "Invalid model_type. Choose from 'linear', 'ridge', 'lasso', 'rf', 'xgb', 'lgbm', 'cat'."
        )
