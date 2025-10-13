# %%
import pandas as pd
import numpy as np
from tree_based_models import model_selection_by_allocation, get_model, evaluate_model
from tqdm import tqdm

from data_engineering import feature_engineering as fe

# %% Load Data

train = pd.read_csv("../data/train.csv")
X_val = pd.read_csv("../data/X_val.csv")
y_val = pd.read_csv("../data/y_val.csv")

# %%
RET_features = [f"RET_{i}" for i in range(1, 21)]
SIGNED_VOLUME_features = [f"SIGNED_VOLUME_{i}" for i in range(1, 21)]
TURNOVER_features = ["AVG_DAILY_TURNOVER"]

window_sizes = [1, 3, 5, 10, 15, 20]

# %% Feature Engineering

dict_mean = train.groupby("ALLOCATION")["target"].mean().to_dict()


def feature_engineering(
    X: pd.DataFrame,
) -> pd.DataFrame:
    X = (
        X.pipe(
            fe.add_return_to_volume_ratio,
            RET_features=RET_features,
            SIGNED_VOLUME_features=SIGNED_VOLUME_features,
        )
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
        )
        # .pipe(add_cross_sectional_features, base_cols=["RET_1", "RET_3"])
    )

    return X


X_feat = feature_engineering(train)
features = [
    col
    for col in X_feat.columns
    if col not in ["ROW_ID", "TS", "ALLOCATION", "target"]  # + SIGNED_VOLUME_features
]

train = feature_engineering(train)
X_val = feature_engineering(X_val)

# %% Configuration

features = [
    col for col in train.columns if col not in ["ROW_ID", "TS", "ALLOCATION", "target"]
]

target_name = "target"
unique_id = "TS"
model_name = "ridge"

# %% Model Selection Evaluation

model_selection_by_allocation(
    data=train,
    target=target_name,
    features=features,
    model_type=model_name,
    unique_id=unique_id,
)

# %%
