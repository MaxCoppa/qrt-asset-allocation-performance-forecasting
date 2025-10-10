# %%
import pandas as pd
import numpy as np
from sklearn.linear_model import Ridge
from sklearn.metrics import accuracy_score
from xgboost import XGBRegressor
from lightgbm import LGBMRegressor
from tree_based_models import kfold_general_with_residuals
import feature_engineering as fe


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
        X.pipe(
            fe.scale_perf_features,
            RET_features=RET_features,
            SIGNED_VOLUME_features=SIGNED_VOLUME_features,
        )
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
        .pipe(
            fe.add_statistical_features,
            RET_features=RET_features,
            SIGNED_VOLUME_features=SIGNED_VOLUME_features,
        )
        .pipe(
            fe.add_average_volume_features,
            SIGNED_VOLUME_features=SIGNED_VOLUME_features,
            window_sizes=[3, 5, 10],
        )
        # .  pipe(fe.add_cross_sectional_features, base_cols=["RET_1", "RET_3"])
    )

    return X


X_feat = feature_engineering(train)

# %%
X_feat.columns
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
    "alpha": 1.0,
    "fit_intercept": True,
    "random_state": 42,
}

xgb_params = {
    "n_estimators": 10,
    "max_depth": 5,
    "learning_rate": 0.01,
    "subsample": 0.8,
    "colsample_bytree": 0.8,
    "random_state": 42,
}

xgb_params_init = {
    "n_estimators": 100,
    "max_depth": 5,
    "learning_rate": 0.01,
    "subsample": 0.8,
    "colsample_bytree": 0.8,
    "random_state": 42,
}


general_model_cls = Ridge
general_params = ridge_params
residual_model_cls = Ridge
residual_params = ridge_params
# %%
metrics = kfold_general_with_residuals(
    data=train,
    target=target_name,
    features=features,
    features_res=RET_features + TURNOVER_features + ["ALLOCATION"],
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
X_train.isna().sum().sort_values()
# %%
# General Ridge model
general_model = general_model_cls(**general_params)
general_model.fit(X_train.drop(columns=["ALLOCATION"]), y_train)

# Compute residuals
train["residuals"] = y_train - general_model.predict(
    X_train.drop(columns=["ALLOCATION"])
)

# %%
# Train residual Ridge models per ALLOCATION
residual_models = {}
for alloc, df_group in train.groupby("ALLOCATION"):
    model = residual_model_cls(**residual_params)
    model.fit(
        df_group[features_res].drop(columns=["ALLOCATION"]),
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
            X_group = df[features_res].loc[mask].drop(columns=["ALLOCATION"])
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

print("General Model accuracy:", accuracy_score(y_true_bin, y_pred_general_bin))
print("Combined Model accuracy:", accuracy_score(y_true_bin, y_pred_combined_bin))

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


preds_sub = combined_predict(X_test[features])
preds_sub = pd.DataFrame(preds_sub, index=X_test["ROW_ID"], columns=[target_name])
(preds_sub > 0).astype(int).to_csv(f"predictions/preds_res_model_v6.csv")
# %%
(preds_sub > 0).mean()

# %%
