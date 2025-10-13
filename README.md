# qrt-asset-allocation-performance-forecasting
Can you predict whether a given asset allocation is worth following—or shorting?

---

## Challenge

- Predict the sign of returns using:
  - Signed Volume (lagged up to 20 periods)
  - Returns (lagged up to 20 periods)
  - Average Daily Turnover (ADT)

- Dataset: multivariate, 65 allocations  
  Columns:  
  `['ROW_ID', 'TS', 'ALLOCATION', 'RET_20', 'RET_19', ..., 'RET_1', 'SIGNED_VOLUME_20', ..., 'SIGNED_VOLUME_1', 'AVG_DAILY_TURNOVER']`

- Objective:  
  Predict the sign of the most recent return from the given lagged features.

### Key Lessons
- Careful model validation and awareness of data leakage are essential.
- Focus on extracting signal rather than fitting noise; avoid overly complex models prone to overfitting.
- Data engineering and preprocessing strongly influence model stability.
- Leaderboards can be misleading; always validate results rigorously.

---

## Data

- Train: 2773 dates, shape (180,245 × 44)  
- y_train: shape (180,245 × 2)  
- Test: shape (7,735 × 44)

Preliminary insights:
- Volumes are correlated across allocations.
- Turnover differs substantially by allocation style/management.

---

## Model Validation

Validation strategies tested:

- **K-Fold on `row_id`** → leakage due to shared timestamps between train/validation.  
- **Time-series split** → more realistic, but leakage remained due to duplicate rows.  
- **Hold-out validation set** → slightly improved robustness but still imperfect.  
- **Final approach**: removal of duplicate rows and construction of a cleaner training set with strict time splits.

**Takeaway**: strict temporal validation and deduplication are essential to avoid inflated performance estimates.

---

## Feature Engineering

A restrained feature engineering strategy was adopted to minimize noise amplification:

- **Market-level features**: benchmark signals summarizing average trends across allocations.  
- **Statistical descriptors**: rolling averages, spreads, and short-term descriptive statistics.  
- **Signed volume**: explored through categorical encodings (long/short), but no consistent benefit; ultimately dropped.  
- **Allocation identifiers**: one-hot encoded to capture allocation-specific biases.  
- **Average Daily Turnover**: important feature retained in raw form.

This design prioritized interpretability and stability over aggressive transformations.

---

## Model Approaches

### Main Research Directions
- Unified model across all allocations to capture general market trends.  
- Allocation-specific models to capture strategy/style deviations.  
- Improved formulations of market–allocation decomposition.

### Deep Learning Experiments
- **Autoencoder + MLP**: tested for denoising and latent representation learning.  
- **LSTM with attention**: aimed at capturing short-term dependencies.  

Both failed to converge meaningfully due to limited sequence length (20 days) and insufficient per-allocation data (~2000 samples). More extensive data and advanced architectures would be required.

### Machine Learning Experiments
- Complex tree ensembles overfit and performed poorly under robust validation.  
- Simpler, regularized models (ridge regression, linear regression) achieved more stable results.  
- Regularization was critical in controlling noise and variance.

### Residual Modeling (Final Approach)
The modeling framework assumed:

\[
\text{Return} = \text{Market Component} + \text{Allocation-Specific Component}
\]

1. Train a global model across all allocations to capture the market component.  
2. Train allocation-specific models on residuals (\(y - y_{market}\)).  
3. Combine both predictions.

This mirrored the underlying market structure while remaining data-efficient.

---

## Results

- **Competition model (Leaderboard: 52.883)**  
  - Market: Ridge regression (α = 1e-2)  
  - Allocation: Ridge regression (α = 100)  

- **Final best private leaderboard model (Leaderboard: 53.193)**  
  - Market: Linear regression (fit_intercept = True, positive = True)  
  - Allocation: Shallow, regularized random forest  

Both solutions were consistent: one fully linear and strongly regularized, the other combining linear market modeling with lightweight non-linear residual learning.

---

## Project Structure

- `data_engineering/`  
  - `data_preprocessing/`: preprocessing utilities  
  - `feature_engineering/`: feature construction functions  

- `tree_based_models/`  
  - `evaluation/`: evaluation utilities  
  - `models/`: model initialization and parameters  
  - `selection/`: model selection procedures  
  - `tuning/`: hyperparameter tuning modules  

- `deep_learning_models/`: deep learning experiments  

- `data/`: dataset folder  
- `predictions/`: model predictions  
- `experiments/`: scripts for experimental workflows  

- `pyproject.toml`  

- `base_prediction.py`: baseline predictions on raw dataset  
- `exp_feature_engineering.py`: feature engineering experiments  
- `exp_res_models.py`: residual modeling experiments  
- `preprocess_data.py`: preprocessing and validation dataset construction  

- `benchmark_submission.ipynb`: QRT benchmark baseline  
- `visualise_data.ipynb`: exploratory visualization of raw data
