# %%
import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.pipeline import Pipeline
from sklearn.linear_model import LinearRegression
from sklearn.metrics import accuracy_score, r2_score

from tree_based_models import model_selection_using_kfold, get_model, evaluate_model

# %% Load Data
train = pd.read_csv("data/train.csv")
X_val = pd.read_csv("data/X_val.csv")
y_val = pd.read_csv("data/y_val.csv")

target = "target"
id_col = "ROW_ID"
time_col = "TS"
alloc_col = "ALLOCATION"

features = [c for c in train.columns if c not in [id_col, time_col, alloc_col, target]]

X_train = train[features].copy()
y_train = train[target].copy()
X_val = X_val[features].copy()
y_val = y_val[target].copy()

# %% Pivot to wide format
wide_features = [c for c in train.columns if c not in [id_col, time_col, alloc_col]]
df_wide = train.pivot(index=time_col, columns=alloc_col, values=wide_features)
df_wide.columns = [f"{a}_{f}" for f, a in df_wide.columns]
df_wide = df_wide.reset_index()

print(df_wide.shape)
print(df_wide.head())

# %% PCA Pipeline
pca_pipeline = Pipeline(
    [
        ("scaler", StandardScaler()),
        ("pca", PCA(n_components=0.5)),  # keep enough comps to explain ~95% variance
    ]
)

X_train_pca = pca_pipeline.fit_transform(X_train)
X_val_pca = pca_pipeline.transform(X_val)

print("Train PCA shape:", X_train_pca.shape)
print("Validation PCA shape:", X_val_pca.shape)
print(
    "Explained variance:",
    pca_pipeline.named_steps["pca"].explained_variance_ratio_.sum(),
)

# %% Baseline Linear Regression
reg = LinearRegression()
reg.fit(X_train_pca, y_train)

y_train_pred = reg.predict(X_train_pca)
y_val_pred = reg.predict(X_val_pca)

print("Validation Accuracy:", accuracy_score(y_val > 0, y_val_pred > 0))
print("Validation R2:", r2_score(y_val, y_val_pred))

# %% Model Selection with K-Fold
model_name = "xgb"
model_selection_using_kfold(
    data=train,
    target=target,
    features=features,
    model_type=model_name,
    feat_engineering=None,
    unique_id=id_col,
    plot_ft_importance=False,
)

# %% Evaluation on Validation Set
X_train_pca_df = pd.DataFrame(
    X_train_pca,
    columns=[f"PC{i+1}" for i in range(X_train_pca.shape[1])],
    index=X_train.index,
)

X_val_pca_df = pd.DataFrame(
    X_val_pca,
    columns=[f"PC{i+1}" for i in range(X_val_pca.shape[1])],
    index=X_val.index,
)

pca_features = [f"PC{i+1}" for i in range(X_train_pca.shape[1])]
model = get_model("xgb")
model.fit(X_train_pca_df[pca_features], y_train)

_ = evaluate_model(
    model=model,
    X=X_val_pca_df[pca_features],
    y=y_val,
    verbose=True,
    log=True,
)

# %% Train Final Model and Predict on Test
X_train = pd.read_csv("data/X_train.csv")
y_train = pd.read_csv("data/y_train.csv")
X_test = pd.read_csv("data/X_test.csv")

model = get_model(model_name)
model.fit(X_train[features], y_train[target])

preds = model.predict(X_test[features])
preds = pd.DataFrame(preds, index=X_test[id_col], columns=[target])

# preds.to_csv("data/preds_test.csv")
# %%
