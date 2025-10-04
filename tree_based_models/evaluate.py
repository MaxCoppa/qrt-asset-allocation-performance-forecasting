"""
Evaluation utilities for single models and ensembles.
"""

from sklearn.metrics import f1_score, roc_auc_score, accuracy_score
from .ensemble import predict_ensembler_models


from sklearn.metrics import accuracy_score
from datetime import datetime


def evaluate_model(model, X, y, verbose: bool = False, log: bool = False) -> dict:
    """
    Evaluate a single model on given data.
    """
    preds = model.predict(X)
    preds = (preds > 0).astype(int)
    labels = (y > 0).astype(int)

    results = {
        "accuracy": accuracy_score(labels, preds),
    }

    msg = " | ".join(f"{k}: {v*100:.2f} %" for k, v in results.items())

    if verbose:
        print("Model evaluation:", msg)

    if log:
        logfile = "predictions/evaluation.log"
        with open(logfile, "a") as f:
            f.write(f"{datetime.now()} - Model evaluation: {msg}\n")

    return results


def evaluate_ensemble_model(
    models, X, y, verbose: bool = False, log: bool = False
) -> dict:
    """
    Evaluate an ensemble of models using averaged predictions.

    """
    avg_preds = (predict_ensembler_models(models=models, X=X) > 0).astype(int)
    labels = (y > 0).astype(int)

    results = {
        "accuracy": accuracy_score(labels, avg_preds),
    }

    msg = " | ".join(f"{k}: {v*100:.2f} %" for k, v in results.items())

    if verbose:
        print("Ensemble Model evaluation:", msg)

    if log:
        logfile = "predictions/evaluation.log"
        with open(logfile, "a") as f:
            f.write(f"{datetime.now()} - Model evaluation: {msg}\n")

    return results
