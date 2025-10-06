# %%
import pandas as pd
from tree_based_models import model_selection_using_kfold, get_model, evaluate_model
from feature_engineering import split_data

# %% Load Data

import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.pipeline import Pipeline

# Chargement
train = pd.read_csv("data/train.csv")
X_val = pd.read_csv("data/X_val.csv")
y_val = pd.read_csv("data/y_val.csv")

# Features et target
features = [
    col for col in train.columns if col not in ["ROW_ID", "TS", "ALLOCATION", "target"]
]
target_name = "target"

X_train = train[features].copy()
y_train = train[target_name].copy()
X_val = X_val[features].copy()
y_val = y_val[target_name].copy()
# %%
# Colonnes à pivoter (toutes sauf celles qui identifient TS et ALLOCATION)
features = [col for col in train.columns if col not in ["ROW_ID", "TS", "ALLOCATION"]]

# Transformation en format wide
df_wide = train.pivot(index="TS", columns="ALLOCATION", values=features)

# Les colonnes deviennent multi-index (feature, allocation)
# On aplatit en concaténant "ALLOCATION_FEATURE"
df_wide.columns = [f"{alloc}_{feat}" for feat, alloc in df_wide.columns]

# Reset index pour remettre TS comme colonne
df_wide = df_wide.reset_index()

print(df_wide.shape)
print(df_wide.head())


# %%
# --- Pipeline PCA ---
pca_pipeline = Pipeline(
    [
        ("scaler", StandardScaler()),  # Standardisation
        (
            "pca",
            PCA(n_components=0.5),
        ),  # Garde assez de composantes pour expliquer 95% de variance
    ]
)

# %%
X_train_pca = pca_pipeline.fit_transform(df_wide.drop(columns=["TS"]))
print("Dimensions train PCA:", X_train_pca.shape)

explained_variance = pca_pipeline.named_steps["pca"].explained_variance_ratio_
print("Variance expliquée cumulée :", explained_variance.sum())
# %%
# Fit uniquement sur TRAIN
X_train_pca = pca_pipeline.fit_transform(X_train)

# Transform sur VALIDATION (sans refit)
X_val_pca = pca_pipeline.transform(X_val)

print("Dimensions train PCA:", X_train_pca.shape)
print("Dimensions val PCA:", X_val_pca.shape)

explained_variance = pca_pipeline.named_steps["pca"].explained_variance_ratio_
print("Variance expliquée cumulée :", explained_variance.sum())

# %%
from sklearn.linear_model import LinearRegression
from sklearn.metrics import accuracy_score, r2_score

# Modèle de régression
reg = LinearRegression()

# Fit sur PCA
reg.fit(X_train_pca, y_train)

# Prédictions
y_train_pred = reg.predict(X_train_pca)
y_val_pred = reg.predict(X_val_pca)

# Évaluation
accuracy_score(y_val > 0, y_val_pred > 0)
# %%

unique_id = "ROW_ID"
model_name = "xgb"

# %% Model Selection Evaluation

model_selection_using_kfold(
    data=train,
    target=target_name,
    features=features,
    model_type=model_name,
    feat_engineering=None,
    unique_id=unique_id,
    plot_ft_importance=False,
)

# %% Evaluation on the Val DataSet

X_train_pca_df = pd.DataFrame(
    X_train_pca,
    columns=[f"PC{i+1}" for i in range(X_train_pca.shape[1])],
    index=X_train.index,
)

# For validation
X_val_pca_df = pd.DataFrame(
    X_val_pca,
    columns=[f"PC{i+1}" for i in range(X_val_pca.shape[1])],
    index=X_val.index,
)

# %%
y_train

# %%

features = [f"PC{i+1}" for i in range(X_train_pca.shape[1])]
model = get_model("xgb")
model.fit(X_train_pca_df[features], train[target_name])

_ = evaluate_model(
    model=model,
    X=X_val_pca_df[features],
    y=y_val,
    verbose=True,
    log=True,
)
# %% Predicion and train
X_train = pd.read_csv("data/X_train.csv")
y_train = pd.read_csv("data/y_train.csv")
X_test = pd.read_csv("data/X_test.csv")

model = get_model(model_name)
model.fit(X_train[features], y_train[target_name])

preds_sub = model.predict(X_test[features])
preds_sub = pd.DataFrame(preds_sub, index=X_test[unique_id], columns=[target_name])

# (preds_sub > 0).astype(int).to_csv("data/preds_test.csv")
# %%
