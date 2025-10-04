# %%
import pandas as pd
from tree_based_models import model_selection_using_kfold, get_model, evaluate_model

# %%
X_train = pd.read_csv("data/X_train.csv")
X_test = pd.read_csv("data/X_test.csv")
y_train = pd.read_csv("data/y_train.csv")

# %%
FEATURES = [col for col in X_train.columns if col not in ["ROW_ID", "TS", "ALLOCATION"]]

# %%
model_selection_using_kfold(
    X=X_train,
    y=y_train,
    target="target",
    features=FEATURES,
    model_type="lgbm",
    unique_id="ROW_ID",
    plot_ft_importance=False,
)
# %%
model = get_model("lgbm")
model.fit(X_train[FEATURES], y_train["target"])

evaluate_model(model, X_train[FEATURES], y_train["target"])
# %%

preds_lgbm = model.predict(X_test[FEATURES])
preds_lgbm = pd.DataFrame(preds_lgbm, index=X_test["ROW_ID"], columns=["target"])
(preds_lgbm > 0).astype(int).to_csv("preds_test.csv")
# %%
