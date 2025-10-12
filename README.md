# qrt-asset-allocation-performance-forecasting
Can you predict whether a given asset allocation is worth following—or shorting?

---

## Challenge

- Predict the sign of returns using:
  - Signed Volume (lagged -20)
  - Returns (lagged -20)
  - Average Daily Turnover (ADT)
- Multivariate dataset of 65 allocations

---

## Data

- Train: 2773 dates, shape (180245, 44)  
- y_train: shape (180245, 2)  
- Test: shape (7735, 44)

First insights:
- Volumes correlated across allocations
- Strong differences in turnover depending on allocation style/management

---

## Model Validation

Cross Validation:

- K-Fold on row_id → leakage due to shared TS between train/val  
- Time-series split → more realistic, but still some leakage from duplicate rows  
- Holdout a validation dataset → slightly better but not fully robust  
- Final approach: identify duplicates and build a cleaner training set

Takeaway: strict time splits and deduplication are essential. Find the best data to train and validate !

---

## Feature Engineering

Feature engineering focused on simple, robust signals rather than aggressive transformations, since the data were noisy and prone to overfitting.  

- Market-level information: benchmark features were created to incorporate average market trends across allocations, helping the model account for common dynamics.  
- Statistical features: rolling averages, spreads, and other descriptive statistics were tested to summarize short-term patterns and relative movements.  
- Signed volume: explored through one-hot encodings (long/short) and considered as a link between traded volume and allocation returns. However, this did not lead to meaningful improvements and was eventually dropped.  
- Allocation identifiers: one-hot encoded, allowing the models to recognize and adjust for allocation-specific styles or biases.  
- Average daily turnover: identified as an important input feature, though additional transformations did not improve results; it was kept in raw form.  


This restrained approach kept the features interpretable and avoided introducing unstable noise.

---

## Model Approach : 

Main ideas : 
- Predict General Trend on all the Data ? 
- Predict "Model" per allocation : strategy of the allocation ? 
- Improve Modelisation of the market ?

### Deep Learning Experiments
Two directions were tested:  
- Autoencoder + MLP, inspired by Jane Street’s market prediction approaches, to denoise the data and learn latent structures of allocations.  
- LSTM with attention to capture short time-series dependencies and cross-allocation information.  

Both approaches failed to converge meaningfully. The 20-day rolling window was too short, and the limited per-allocation data (~2000 rows) prevented stable learning. More data and deeper expertise in time-series deep learning would be needed for such methods to succeed.  

### Machine Learning Models
- Highly tuned and deep tree-based models overfit and failed under robust validation.  
- Simpler models with stronger regularization, like ridge or linear regression, consistently performed better.  
- The high noise level made these regularized approaches more stable and reliable.  

### Residual Modeling (Final Approach)
The final model design assumed that return = market component + allocation-specific component.  

1. Train a model across all allocations to capture the market component (general pattern).  
2. For each allocation, train a second model on the residuals (y − y_market) to capture allocation-specific deviations.  
3. Combine both predictions for the final forecast.  

This framework mirrored a key structure of the problem: market conditions shape overall movement, while allocations add their own style or strategy on top. It also matched the limited amount of per-allocation data by using a simple but structured approach.

### Best socres


- My selected model for competition  (LB = 52.883) 
  - Market component: Ridge regression (alpha = 1e-2)  
  - Allocation component: Ridge regression (alpha = 100)  

- My final best private leaderboard model (LB = 53.193)  
  - Market component: Linear regression (fit_intercept = True, positive = True)  
  - Allocation component: Random forest (shallow, regularized)  

These represent consistent approaches: one purely linear and regularized (ridge + ridge), and one combining a simple linear market baseline with a lightweight non-linear learner for allocation residuals.  
