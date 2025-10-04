"""
Evaluation utilities for single models and ensembles.
"""

from sklearn.metrics import f1_score, roc_auc_score, accuracy_score
from .ensemble import predict_ensembler_models


def evaluate_model(model, X, y, threshold: float = 0.5, verbose: bool = False) -> dict:
    """
    Evaluate a single model on given data.
    """
    preds = model.predict(X)
    preds = (preds > 0).astype(int)
    labels = (y > 0).astype(int)

    results = {
        "f1": f1_score(labels, preds),
        "accuracy": accuracy_score(labels, preds),
    }

    if verbose:
        print(
            "Model evaluation:",
            " | ".join(f"{k}: {v:.3f}" for k, v in results.items()),
        )

    return results


def evaluate_ensemble_model(
    models, X, y, threshold: float = 0.5, verbose: bool = False
) -> dict:
    """
    Evaluate an ensemble of models using averaged predictions.

    """
    avg_preds = predict_ensembler_models(models=models, X=X, threshold=threshold)
    labels = (y > 0).astype(int)

    results = {
        "f1": f1_score(labels, avg_preds),
        "accuracy": accuracy_score(labels, avg_preds),
    }

    if verbose:
        print(
            "Ensemble evaluation:",
            " | ".join(f"{k}: {v:.3f}" for k, v in results.items()),
        )

    return results
