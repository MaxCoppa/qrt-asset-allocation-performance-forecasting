# %%
import pandas as pd
from tree_based_models import model_selection_using_kfold, get_model, evaluate_model
from feature_engineering import encode_allocation, add_average_perf_features

# %% Load Data

X_train = pd.read_csv("data/X_train.csv")
X_test = pd.read_csv("data/X_test.csv")
y_train = pd.read_csv("data/y_train.csv")

# %%
RET_features = [f"RET_{i}" for i in range(1, 20)]
SIGNED_VOLUME_features = [f"SIGNED_VOLUME_{i}" for i in range(1, 20)]
TURNOVER_features = ["AVG_DAILY_TURNOVER"]
window_sizes = [3, 5, 10, 15, 20]
# %% Feature Engineering


def feature_engineering(X: pd.DataFrame) -> pd.DataFrame:

    X = X.pipe(
        add_average_perf_features, RET_features=RET_features, window_sizes=window_sizes
    )

    return X


X_test = feature_engineering(X_test)

# %% Configuration

features = [col for col in X_test.columns if col not in ["ROW_ID", "TS", "ALLOCATION"]]
target_name = "target"
unique_id = "ROW_ID"
model_name = "lgbm"
# %% Model Selection Evaluation

model_selection_using_kfold(
    X=X_train,
    y=y_train,
    target=target_name,
    features=features,
    feat_engineering=feature_engineering,
    model_type=model_name,
    unique_id=unique_id,
    plot_ft_importance=True,
)
# %% Train Model

model = get_model(model_name)
model.fit(X_train[features], y_train[target_name])

_ = evaluate_model(
    model=model,
    X=X_train[features],
    y=y_train[target_name],
    verbose=True,
)
# %% Predicion

preds_sub = model.predict(X_test[features])
preds_sub = pd.DataFrame(preds_sub, index=X_test[unique_id], columns=[target_name])

# (preds_sub > 0).astype(int).to_csv(f"data/preds_{model_name}.csv")
# %%
