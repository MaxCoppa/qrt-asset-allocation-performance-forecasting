# %%
from sklearn.neural_network import MLPRegressor
from sklearn.preprocessing import StandardScaler
import pandas as pd
from sklearn.metrics import accuracy_score

# %% Load Data

train = pd.read_csv("data/train.csv")
X_val = pd.read_csv("data/X_val.csv")
y_val = pd.read_csv("data/y_val.csv")

# %% Configuration

features = [
    col for col in train.columns if col not in ["ROW_ID", "TS", "ALLOCATION", "target"]
]
target_name = "target"
unique_id = "TS"
# %%
scaler = StandardScaler(with_mean=True)
X_train = scaler.fit_transform(train[features])
y_train = train[target_name]


X_val = scaler.transform(X_val[features])
y_val = y_val[target_name]

m = MLPRegressor(
    hidden_layer_sizes=(100,50,),
    solver="adam",
    learning_rate_init=0.5,
    alpha=100,
    activation="tanh",
    tol=1e-2,
    n_iter_no_change=25,
)
m.fit(X_train, y_train)
y_pred = m.predict(X_val)

# %%
accuracy_score(
    y_true=(y_val > 0).astype(int),
    y_pred=(y_pred > 0).astype(int),
)
# %%
