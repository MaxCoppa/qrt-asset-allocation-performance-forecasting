import pandas as pd
import numpy as np
import seaborn as sns
from sklearn.model_selection import KFold
from datetime import datetime
from tqdm import tqdm

from tree_based_models import get_model, evaluate_model
from tree_based_models import model_selection_by_allocation, get_model, evaluate_model

import feature_engineering as fe

# %% Load Data

train = pd.read_csv("data/train.csv")
X_val = pd.read_csv("data/X_val.csv")
y_val = pd.read_csv("data/y_val.csv")

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
        X
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
# %% Load Data

train = feature_engineering(train)
X_val = feature_engineering(X_val)

# %% Configuration

features = [
    col for col in train.columns if col not in ["ROW_ID", "TS", "ALLOCATION", "target"]
]
# features = [f"RET_{i}" for i in range(1, 21)]
target_name = "target"
unique_id = "TS"
model_name = "rf"

data = train
target = target_name
features = features
model_type = model_name
unique_id = unique_id
n_splits = 2


def get_data(ids, unique_vals, col: pd.Series, X_data: pd.DataFrame, y_data: pd.Series):
    """
    Extract subset of X and y given selected indices of unique values.
    """
    selected = unique_vals[ids]
    mask = col.isin(selected)
    return X_data.loc[mask], y_data.loc[mask], mask


all_results = {}
all_models = {}
all_accuracies = []  # collect accuracies from all allocations

for alloc in tqdm(data["ALLOCATION"].unique(), desc="Processing allocations"):
    train_sub = data[data["ALLOCATION"] == alloc]
    corr = train_sub[
        [col for col in train_sub.columns if col not in ["ROW_ID", "TS", "ALLOCATION"]]
    ].corr()

    features = (
        (corr["target"].drop("target") * 100)
        .abs()
        .sort_values(ascending=False)
        .head(30)
        .index
    )
    # Prepare training sets
    X_train = train_sub[features].copy()
    y_train = train_sub[target].copy()
    unique_ids = train_sub[unique_id].unique()

    metrics = {"accuracy": []}
    models = []

    splits = KFold(n_splits=n_splits, random_state=0, shuffle=True).split(unique_ids)

    for train_idx, val_idx in splits:
        X_local_train, y_local_train, _ = get_data(
            train_idx, unique_ids, train_sub[unique_id], X_train, y_train
        )
        X_local_val, y_local_val, _ = get_data(
            val_idx, unique_ids, train_sub[unique_id], X_train, y_train
        )

        model = get_model(model_type)
        model.fit(X_local_train, y_local_train)

        model_eval = evaluate_model(model=model, X=X_local_val, y=y_local_val)
        acc = model_eval["accuracy"]

        models.append(model)
        metrics["accuracy"].append(acc)
        all_accuracies.append(acc)  # store for global summary

    all_results[alloc] = metrics
    all_models[alloc] = models

# Final aggregated results
accs = np.array(all_accuracies)
mean = accs.mean() * 100
std = accs.std() * 100
min_acc = accs.min() * 100
max_acc = accs.max() * 100

print(
    f"\nOverall Accuracy: {mean:.2f}% (± {std:.2f}%) "
    f"[Min: {min_acc:.2f}% ; Max: {max_acc:.2f}%] across all allocations"
)


# %%

all_results = {}
all_models = {}
all_accuracies = []  # collect accuracies from all allocations
features_dict = {}
val = X_val.merge(y_val, on="ROW_ID")

for alloc in tqdm(data["ALLOCATION"].unique(), desc="Processing allocations"):
    train_sub = data[data["ALLOCATION"] == alloc]
    val_sub = val[val["ALLOCATION"] == alloc]
    corr = train_sub[
        [col for col in train_sub.columns if col not in ["ROW_ID", "TS", "ALLOCATION"]]
    ].corr()

    features = (
        (corr["target"].drop("target") * 100)
        .abs()
        .sort_values(ascending=False)
        .head(10)
        .index
    )
    features_dict[alloc] = features
    # Prepare training sets
    X_train = train_sub[features].copy()
    y_train = train_sub[target].copy()
    X_val = val_sub[features].copy()
    y_val = val_sub[target].copy()

    metrics = {"accuracy": []}
    models = []

    model = get_model(model_type)
    model.fit(X_train, y_train)

    model_eval = evaluate_model(model=model, X=X_val, y=y_val)
    acc = model_eval["accuracy"]

    models.append(model)
    metrics["accuracy"].append(acc)
    all_accuracies.append(acc)  # store for global summary

    all_results[alloc] = metrics
    all_models[alloc] = models

# Final aggregated results
accs = np.array(all_accuracies)
mean = accs.mean() * 100
std = accs.std() * 100
min_acc = accs.min() * 100
max_acc = accs.max() * 100

print(
    f"\nOverall Accuracy: {mean:.2f}% (± {std:.2f}%) "
    f"[Min: {min_acc:.2f}% ; Max: {max_acc:.2f}%] across all allocations"
)
# %%
val["pred"] = pd.NA
for alloc, model in all_models.items():
    mask = val["ALLOCATION"] == alloc
    if mask.any():
        features_res = features_dict[alloc]
        X_group = val.loc[mask, features_res]
        val.loc[mask, "pred"] = model[0].predict(X_group)

# %%
from sklearn.metrics import accuracy_score

accuracy_score(val["pred"] > 0, val["target"] > 0)
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

X_test["pred"] = pd.NA
for alloc, model in all_models.items():
    mask = X_test["ALLOCATION"] == alloc
    if mask.any():
        features_res = features_dict[alloc]
        X_group = X_test.loc[mask, features_res]
        X_test.loc[mask, "pred"] = model[0].predict(X_group)

# %%
X_test[target_name] = (X_test["pred"] > 0).astype(int)
preds_sub = X_test[["ROW_ID", target_name]].set_index("ROW_ID")
(preds_sub > 0).mean()

# %%
preds_sub.to_csv("predictions/model_allocation_corr.csv")
# %%
val["ALLOCATION"] == alloc
# %%
