# %%
import pandas as pd
from tree_based_models import model_selection_using_kfold, get_model, evaluate_model
from feature_engineering import split_data

# %% Load Data

train = pd.read_csv("../data/train_unique.csv")
X_val = pd.read_csv("../data/X_val.csv")
y_val = pd.read_csv("../data/y_val.csv")

# %% Configuration

features = [
    col for col in train.columns if col not in ["ROW_ID", "TS", "ALLOCATION", "target"]
]

target_name = "target"
unique_id = "TS"
model_name = "xgb_opt"

# %% Model Selection Evaluation

model_selection_using_kfold(
    data=train,
    target=target_name,
    features=features,
    model_type=model_name,
    feat_engineering=None,
    unique_id=unique_id,
    plot_ft_importance=False,
    n_splits=5,
)

# %% Evaluation on the Val DataSet

model = get_model(model_name)
model.fit(train[features], train[target_name])

_ = evaluate_model(
    model=model,
    X=X_val[features],
    y=y_val[target_name],
    verbose=True,
    log=True,
)
# %% Predicion and train
X_train = pd.read_csv("../data/X_train.csv")
y_train = pd.read_csv("../data/y_train.csv")
X_test = pd.read_csv("../data/X_test.csv")

model = get_model(model_name)
model.fit(X_train[features], y_train[target_name])

preds_sub = model.predict(X_test[features])
preds_sub = pd.DataFrame(preds_sub, index=X_test[unique_id], columns=[target_name])

# (preds_sub > 0).astype(int).to_csv("../data/preds_test.csv")
# %%
