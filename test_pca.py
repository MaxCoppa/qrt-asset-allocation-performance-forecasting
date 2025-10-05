# %%
import pandas as pd
from tree_based_models import model_selection_using_kfold, get_model, evaluate_model
from feature_engineering import (
    encode_allocation,
    add_average_perf_features,
    create_allocation_features,
    add_average_volume_features,
)

from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA


def add_pca_features(
    df, n_components=8, drop_cols=["ROW_ID", "TS", "ALLOCATION", "target"], prefix="PC"
):
    """
    Feature engineering function: add PCA components to dataset.

    """
    if drop_cols is None:
        drop_cols = []

    # Select numeric features (exclude drop_cols)
    features = df[[col for col in df.columns if col not in drop_cols]]

    # Standardize
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(features)

    # PCA
    pca = PCA(n_components=n_components)
    X_pca = pca.fit_transform(X_scaled)

    # Build PCA DataFrame
    pca_df = pd.DataFrame(
        X_pca, columns=[f"{prefix}{i+1}" for i in range(n_components)], index=df.index
    )

    # Append to original df
    df_out = pd.concat([df.copy(), pca_df], axis=1)

    return df_out


# %% Load Data

train = pd.read_csv("data/train.csv")
X_val = pd.read_csv("data/X_val.csv")
y_val = pd.read_csv("data/y_val.csv")

# %%
RET_features = [f"RET_{i}" for i in range(1, 20)]
SIGNED_VOLUME_features = [f"SIGNED_VOLUME_{i}" for i in [1, 3, 5, 10, 15, 20]]
TURNOVER_features = ["AVG_DAILY_TURNOVER"]

window_sizes = [3, 5, 10, 15, 20]

features = RET_features + TURNOVER_features + SIGNED_VOLUME_features
# features = features +  [f"PC{i}"for i in range(1,4)]
features = features + [f"AVERAGE_PERF_{i}" for i in window_sizes]
features = features + [f"ALLOCATIONS_AVERAGE_PERF_{i}" for i in window_sizes]

# %% Feature Engineering


def feature_engineering(
    X: pd.DataFrame,
) -> pd.DataFrame:
    X = X.pipe(add_pca_features).pipe(
        add_average_perf_features,
        RET_features=RET_features,
        window_sizes=window_sizes,
        group_col="TS",
    )
    return X


# %% Configuration

target_name = "target"
unique_id = "TS"
model_name = "lgbm"
# %% Model Selection Evaluation

model_selection_using_kfold(
    data=train,
    target=target_name,
    features=features,
    feat_engineering=feature_engineering,
    model_type=model_name,
    unique_id=unique_id,
    plot_ft_importance=True,
    n_splits=8,
)
# %% Train Model

# (preds_sub > 0).astype(int).to_csv(f"predictions/preds_{model_name}.csv")
# %%
