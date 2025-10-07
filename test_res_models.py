# %%
import pandas as pd
import numpy as np
from sklearn.linear_model import Ridge
from sklearn.metrics import accuracy_score
from xgboost import XGBRegressor
from tree_based_models import kfold_general_with_residuals
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

window_sizes = [1, 3, 5, 10, 15, 20]

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
        # .pipe(add_statistical_features,
        #     RET_features=RET_features,
        #     SIGNED_VOLUME_features=SIGNED_VOLUME_features)
    )

    return X


X_feat = feature_engineering(train)
# Load data
train = pd.read_csv("data/train.csv")
X_val = pd.read_csv("data/X_val.csv")
y_val = pd.read_csv("data/y_val.csv")

# %%
features = [col for col in X_feat.columns if col not in ["ROW_ID", "TS", "target"]]
target_name = "target"
ridge_params = {
    "alpha": 1.0,
    "fit_intercept": True,
    "random_state": 42,
}

xgb_params = {
    "n_estimators": 10,
    "max_depth": 5,
    "learning_rate": 0.05,
    "subsample": 0.8,
    "colsample_bytree": 0.8,
    "random_state": 42,
}
# %%
metrics = kfold_general_with_residuals(
    data=train,
    target=target_name,
    features=features,
    unique_id="TS",
    feat_engineering=feature_engineering,
    n_splits=5,
    general_model_cls=Ridge,
    general_params=ridge_params,
    residual_model_cls=XGBRegressor,
    residual_params=xgb_params,
)

# %%
# Define features (exclude target + ID + TS but keep ALLOCATION)

if feature_engineering:
    train = feature_engineering(train)
    X_val = feature_engineering(X_val)

X_train = train[features]
y_train = train[target_name]


# %%
# General Ridge model
general_model = Ridge(**ridge_params)
general_model.fit(X_train.drop(columns=["ALLOCATION"]), y_train)

# Compute residuals
train["residuals"] = y_train - general_model.predict(
    X_train.drop(columns=["ALLOCATION"])
)

# %%
# Train residual Ridge models per ALLOCATION
residual_models = {}
for alloc, df_group in train.groupby("ALLOCATION"):
    model = XGBRegressor(**xgb_params)
    model.fit(
        df_group.drop(columns=["ROW_ID", "TS", "target", "residuals", "ALLOCATION"]),
        df_group["residuals"],
    )
    residual_models[alloc] = model


# %%
# Combined prediction function (vectorized)
def combined_predict(df):
    base_pred = general_model.predict(df.drop(columns=["ALLOCATION"]))
    corrections = np.zeros(len(df))
    for alloc, model in residual_models.items():
        mask = df["ALLOCATION"] == alloc
        if mask.any():
            X_group = df.loc[mask].drop(columns=["ALLOCATION"])
            corrections[mask] = model.predict(X_group)
    return base_pred + corrections


# %%
# Predictions
y_pred_general = general_model.predict(X_val[features].drop(columns=["ALLOCATION"]))

y_pred_combined = combined_predict(X_val[features])

# Accuracy on sign (>0)
y_true_bin = (y_val[target_name] > 0).astype(int)
y_pred_general_bin = (y_pred_general > 0).astype(int)
y_pred_combined_bin = (y_pred_combined > 0).astype(int)

print("General Ridge accuracy:", accuracy_score(y_true_bin, y_pred_general_bin))
print("Combined Ridge accuracy:", accuracy_score(y_true_bin, y_pred_combined_bin))


# %%
X_test = pd.read_csv("data/X_test.csv")
X_test = X_test.fillna(0)

if feature_engineering:
    X_test = feature_engineering(X_test)


preds_sub = combined_predict(X_test[features])
preds_sub = pd.DataFrame(preds_sub, index=X_test["ROW_ID"], columns=[target_name])
(preds_sub > 0).astype(int).to_csv(f"predictions/preds_res_model.csv")
# %%
preds_sub
# %%
