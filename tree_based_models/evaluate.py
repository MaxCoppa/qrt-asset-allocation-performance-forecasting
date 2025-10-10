"""
Evaluation utilities for single models and ensembles.
"""

from sklearn.metrics import f1_score, roc_auc_score, accuracy_score
from .ensemble import predict_ensembler_models
from datetime import datetime


def evaluate_model(
    model, X, y, verbose: bool = False, log: bool = False, log_note: str = None
) -> dict:
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
        note_str = f" | Note: {log_note}" if log_note else ""
        with open(logfile, "a") as f:
            f.write(f"{datetime.now()} - Model evaluation: {msg}{note_str}\n")

    return results
