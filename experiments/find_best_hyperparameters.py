# %%
import pandas as pd
import feature_engineering as fe
from tree_based_models import tune_model

# %% Load Data

train = pd.read_csv("../data/train.csv")
X_val = pd.read_csv("../data/X_val.csv")
y_val = pd.read_csv("../data/y_val.csv")

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
        )
        .pipe(fe.create_mean_allocation, dict_mean=dict_mean)
        .pipe(fe.add_cross_sectional_features, base_cols=["RET_1", "RET_3"])
    )

    return X


X_feat = train
features = [
    col
    for col in X_feat.columns
    if col not in ["ROW_ID", "TS", "ALLOCATION", "target"] + SIGNED_VOLUME_features
]
# %% Configuration

target_name = "target"
unique_id = "TS"
model_name = "xgb"

X_train = (train)[features]
y_train = train[target_name]

X_test = (X_val)[features]
y_test = y_val[target_name]

# %%
study = tune_model(
    model_name="xgb",
    X_train=X_train,
    y_train=y_train,
    X_val=X_test,
    y_val=y_test,
    n_trials=50,
)


# %%
