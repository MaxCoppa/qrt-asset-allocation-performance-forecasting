"""
Utility for combining multiple models into an ensemble prediction
using simple probability averaging.
"""

import numpy as np


def predict_ensembler_models(models, X, threshold: float = 0.5):
    """
    Perform an average ensemble of model predictions.
    """
    # Collect predicted probabilities from each model
    preds = [(m.predict(X) > 0).astype(int) for m in models]

    # Average across models
    avg_preds = np.mean(preds, axis=0)

    return avg_preds
