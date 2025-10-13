# %%
import pandas as pd
import numpy as np
from sklearn.linear_model import Ridge, ElasticNet, LinearRegression
from sklearn.ensemble import RandomForestRegressor

from sklearn.metrics import accuracy_score
from xgboost import XGBRegressor
from lightgbm import LGBMRegressor
from tree_based_models import kfold_general_with_residuals, ResidualModel
from data_engineering import feature_engineering as fe

# %% Load Data

train = pd.read_csv("data/train.csv")
X_val = pd.read_csv("data/X_val.csv")
y_val = pd.read_csv("data/y_val.csv")

# %%
RET_features = [f"RET_{i}" for i in range(1, 21)]
SIGNED_VOLUME_features = [f"SIGNED_VOLUME_{i}" for i in range(1, 21)]
TURNOVER_features = ["AVG_DAILY_TURNOVER"]

window_sizes = [3, 5, 10, 15, 20]

# %% Feature Engineering


def feature_engineering(
    X: pd.DataFrame,
) -> pd.DataFrame:
    X = (
        X
        # .pipe(
        #     fe.scale_perf_features,
        #     RET_features=RET_features,
        #     SIGNED_VOLUME_features=SIGNED_VOLUME_features,
        # )
        # .pipe(
        #     fe.add_mulitiply_col,
        #     RET_features=RET_features,
        #     SIGNED_VOLUME_features=SIGNED_VOLUME_features,
        # )
        .pipe(
            fe.add_average_perf_features,
            RET_features=RET_features,
            window_sizes=window_sizes,
            group_col="TS",
        )
        # .pipe(
        #     fe.add_statistical_features,
        #     RET_features=RET_features,
        #     SIGNED_VOLUME_features=SIGNED_VOLUME_features,
        # )
        # .pipe(
        #     fe.add_average_volume_features,
        #     SIGNED_VOLUME_features=SIGNED_VOLUME_features,
        #     window_sizes=[3, 5, 10],
        # )
        # .  pipe(fe.add_cross_sectional_features, base_cols=["RET_1", "RET_3"])
    )

    return X


X_feat = feature_engineering(train)
# %%


features = [col for col in X_feat.columns if col not in ["ROW_ID", "TS", "target"]]

features = [
    col
    for col in X_feat.columns
    if col not in ["ROW_ID", "TS", "target"] + SIGNED_VOLUME_features
]
features_res = features


# %%
target_name = "target"

ridge_params = {
    "alpha": 1e-2,
    "fit_intercept": True,
    "random_state": 42,
}

linear_params = {
    "fit_intercept": True,
    "positive": True,
}

ridge_params_2 = {
    "alpha": 100,
    "fit_intercept": True,
    "random_state": 42,
}

xgb_params = {
    "n_estimators": 100,
    "max_depth": 5,
    "learning_rate": 0.05,
    "subsample": 0.8,
    "colsample_bytree": 0.8,
    "random_state": 42,
}

xgb_params_init = {
    "n_estimators": 100,
    "max_depth": 10,
    "learning_rate": 0.001,
    "subsample": 0.8,
    "colsample_bytree": 0.8,
    "random_state": 42,
}

rf_params = {
    "n_estimators": 50,
    "max_depth": 5,
    "min_samples_split": 5,
    "min_samples_leaf": 3,
    # "max_features": "sqrt",
    "random_state": 42,
    "n_jobs": -1,
}

general_model_cls = Ridge
general_params = ridge_params
residual_model_cls = Ridge
residual_params = ridge_params_2
# %%
metrics = kfold_general_with_residuals(
    data=train,
    target=target_name,
    features=features,
    features_res=features_res,
    unique_id="TS",
    feat_engineering=feature_engineering,
    n_splits=5,
    general_model_cls=general_model_cls,
    general_params=general_params,
    residual_model_cls=residual_model_cls,
    residual_params=residual_params,
)


# %%
# Define features (exclude target + ID + TS but keep ALLOCATION)

if feature_engineering:
    train = feature_engineering(train)
    X_val = feature_engineering(X_val)

X_train = train[features]
y_train = train[target_name]
# %%

res_model = ResidualModel(
    general_model_cls=general_model_cls,
    general_params=general_params,
    residual_model_cls=residual_model_cls,
    residual_params=residual_params,
)
res_model.fit(train, target_name, features, features_res)

# %%
# Predictions
y_pred_val = res_model.predict(X_val, features, features_res)

# Accuracy on sign (>0)
y_true_bin = (y_val[target_name] > 0).astype(int)
y_pred_bin = (y_pred_val > 0).astype(int)

print("Residual Model accuracy:", accuracy_score(y_true_bin, y_pred_bin))
# %%
X_test = pd.read_csv("data/X_test.csv")
# X_test = X_test.fillna(0)


for i in range(1, X_test.shape[1] - 1):  # skip first and last column
    col = X_test.columns[i]
    left = X_test.columns[i - 1]
    right = X_test.columns[i + 1]
    if X_test[col].isna().sum() > 0:
        print(left, right)
        X_test[col] = X_test[col].fillna((X_test[left] + X_test[right]) / 2)


if feature_engineering:
    X_test = feature_engineering(X_test)


preds_sub = res_model.predict(X_test, features, features_res)
preds_sub = pd.DataFrame(preds_sub, index=X_test["ROW_ID"], columns=[target_name])
(preds_sub > 0).astype(int).to_csv("predictions/preds_res_model_last.csv")

# print("Prediction file saved.")
print("Positive rate:", (preds_sub > 0).mean().values[0])

# %%
