# %%
import pandas as pd
from pytorch_tabnet.tab_model import TabNetClassifier
from pytorch_tabnet.pretraining import TabNetPretrainer
from sklearn.metrics import accuracy_score

# %%

train = pd.read_csv("../data/train.csv")
X_val = pd.read_csv("../data/X_val.csv")
y_val = pd.read_csv("../data/y_val.csv")
# %%
features = [
    col for col in train.columns if col not in ["ROW_ID", "TS", "ALLOCATION", "target"]
]
target_name = "target"

X_train = train[features].copy()
y_train = (train[target_name].copy() > 0).astype(int)
X_test = X_val[features].copy()
y_test = (y_val[target_name].copy() > 0).astype(int)


# %%
unsup_model = TabNetPretrainer(
    n_d=8,
    n_a=8,
    n_steps=3,
    gamma=1.3,
    seed=42,
    verbose=1,
    device_name="cpu",  # put "cuda" if GPU available
)

unsup_model.fit(
    X_train.to_numpy(),
    eval_set=[X_test.to_numpy()],
    max_epochs=20,
    patience=10,
    batch_size=1024,
    virtual_batch_size=128,
)

clf = TabNetClassifier(
    n_d=8, n_a=8, n_steps=3, gamma=1.3, seed=42, verbose=1, device_name="cpu"
)
# %%
clf.fit(
    X_train.to_numpy(),
    y_train.to_numpy(),
    eval_set=[(X_test.to_numpy(), y_test.to_numpy())],
    eval_name=["valid"],
    eval_metric=["accuracy"],
    max_epochs=20,
    patience=20,
    batch_size=1024,
    from_unsupervised=unsup_model,
)

# %%
y_pred = clf.predict(X_test.to_numpy())
print("Accuracy :", accuracy_score(y_test, y_pred))

# %%
