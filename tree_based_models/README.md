## tree_based_models Submodule

This submodule provides utilities for training, evaluating, and combining
tree-based machine learning models.

### Models Used

The following classifiers are supported with default hyperparameters (defined in `initialise_model.py`):

- **Random Forest (rf)** – A bagging ensemble of decision trees, useful as a benchmark due to its robustness and interpretability.  
- **XGBoost (xgb)** – Gradient boosting framework optimized for performance and regularization, widely used in structured/tabular data problems.  
- **LightGBM (lgbm)** – Gradient boosting method designed for efficiency and scalability on large datasets.  
- **CatBoost (cat)** – Gradient boosting library with strong support for categorical variables and built-in handling of overfitting.  

Each model comes with a reasonable default configuration, but parameters can be tuned as needed.

---

### Available Functions

- **get_model(model_type)**  
  Initialize a tree-based classifier (`rf`, `xgb`, `lgbm`, `cat`) with preset parameters.

- **model_selection_using_kfold(train, target, features, model_type, ...)**  
  Perform K-Fold cross-validation for a given model and dataset, returning evaluation metrics.

- **evaluate_model(model, X, y, ...)**  
  Evaluate a single trained model (accuracy, ROC AUC, F1 score).

- **evaluate_ensemble_model(models, X, y, ...)**  
  Evaluate an ensemble of models using averaged predictions.

- **predict_ensembler_models(models, X, ...)**  
  Combine predictions from multiple models via simple probability averaging.

---

### Customization

- **Changing hyperparameters**  
  Default parameters for Random Forest, XGBoost, LightGBM, and CatBoost are defined in  
  `initialise_model.py` inside the `model_params` dictionary.  
  You can edit these values or add new model configurations.

- **Adding new models**  
  To integrate another classifier, extend `get_model` in `initialise_model.py`.

- **Modifying cross-validation**  
  By default, `model_selection_using_kfold` uses a 4-fold KFold split.  
  You can change the number of folds (`n_splits`) or replace `KFold` with another  
  splitting strategy depending on your dataset.

- **Custom ensemble methods**  
  Currently, `predict_ensembler_models` averages probabilities equally.  
  If you want weighted averaging or more advanced stacking/blending,  
  you can modify `ensemble.py`.

- **Evaluation metrics**  
  Evaluation in `evaluate.py` is based on Accuracy, ROC AUC, and F1.  
  To include other metrics (e.g., Precision, Recall, Log Loss),  
  add them in the results dictionary inside `evaluate_model` and `evaluate_ensemble_model`.
