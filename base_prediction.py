# %%
import pandas as pd
from tree_based_models import model_selection_using_kfold, get_model, evaluate_model
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
unique_id = "TS"
model_name = "cat"

# %% Model Selection Evaluation

model_selection_using_kfold(
    X=X_train,
    y=y_train,
    target=target_name,
    features=FEATURES,
    model_type=model_name,
    feat_engineering=None,
    unique_id=unique_id,
    plot_ft_importance=False,
)
# %% Train Model

model = get_model(model_name)
model.fit(X_train[FEATURES], y_train[target_name])

_ = evaluate_model(
    model=model,
    X=X_val[FEATURES],
    y=y_val[target_name],
    verbose=True,
)
# %% Predicion

preds_sub = model.predict(X_test[FEATURES])
preds_sub = pd.DataFrame(preds_sub, index=X_test[unique_id], columns=[target_name])

# (preds_sub > 0).astype(int).to_csv("data/preds_test.csv")
# %%
