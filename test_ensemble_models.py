# %%
import pandas as pd
from tree_based_models import get_model, evaluate_ensemble_model
from feature_engineering import split_data

# %% Load Data

X_train = pd.read_csv("data/X_train.csv")
X_test = pd.read_csv("data/X_test.csv")
y_train = pd.read_csv("data/y_train.csv")

# %%
n_date_test = X_test["TS"].nunique()

X_train, y_train, X_val, y_val = split_data(X_train, y_train, n_date_test)
# %% Configuration

FEATURES = [col for col in X_train.columns if col not in ["ROW_ID", "TS", "ALLOCATION"]]
target_name = "target"
model_names = "xgb:lgbm"
models = []
# %% Train Model

for model_name in model_names.split(":"):
    model = get_model(model_name)
    model.fit(X_train[FEATURES], y_train[target_name])
    models.append(model)

_ = evaluate_ensemble_model(
    models=models,
    X=X_val[FEATURES],
    y=y_val[target_name],
    verbose=True,
)
# %% Predicion

# (preds_sub > 0).astype(int).to_csv("data/preds_test.csv")
# %%
