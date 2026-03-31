"""
Evaluation Metrics Module

Computes detection performance metrics:
- AUROC, F1, Precision, Recall
- Calibration metrics
- Per-class metrics for 3-class output

Author: SynthDetect Team
"""

import numpy as np
from typing import Dict, List, Optional
from sklearn.metrics import (
    roc_auc_score,
    f1_score,
    precision_score,
    recall_score,
    accuracy_score,
    confusion_matrix,
    classification_report,
)


def compute_binary_metrics(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    y_proba: Optional[np.ndarray] = None,
) -> Dict[str, float]:
    """
    Compute binary classification metrics.

    Args:
        y_true: Ground truth labels (0=human, 1=AI).
        y_pred: Predicted labels.
        y_proba: Predicted probabilities for positive class.

    Returns:
        Dictionary with accuracy, precision, recall, f1, and optionally auroc.
    """
    metrics = {
        "accuracy": float(accuracy_score(y_true, y_pred)),
        "precision": float(precision_score(y_true, y_pred, zero_division=0)),
        "recall": float(recall_score(y_true, y_pred, zero_division=0)),
        "f1": float(f1_score(y_true, y_pred, zero_division=0)),
    }

    if y_proba is not None:
        try:
            metrics["auroc"] = float(roc_auc_score(y_true, y_proba))
        except ValueError:
            metrics["auroc"] = 0.0  # Single class in y_true

    return metrics


def compute_multiclass_metrics(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    labels: Optional[List[str]] = None,
) -> Dict:
    """
    Compute 3-class metrics (fully_ai, collaborative, fully_human).

    Args:
        y_true: Ground truth labels.
        y_pred: Predicted labels.
        labels: Class label names.

    Returns:
        Dictionary with per-class and macro metrics.
    """
    if labels is None:
        labels = ["fully_human", "collaborative", "fully_ai"]

    report = classification_report(
        y_true, y_pred, target_names=labels, output_dict=True, zero_division=0
    )

    cm = confusion_matrix(y_true, y_pred)

    return {
        "classification_report": report,
        "confusion_matrix": cm.tolist(),
        "macro_f1": float(report["macro avg"]["f1-score"]),
        "weighted_f1": float(report["weighted avg"]["f1-score"]),
    }


def compute_calibration_error(
    y_true: np.ndarray,
    y_proba: np.ndarray,
    n_bins: int = 10,
) -> Dict[str, float]:
    """
    Compute Expected Calibration Error (ECE).

    Measures how well predicted probabilities match actual frequencies.

    Args:
        y_true: Ground truth binary labels.
        y_proba: Predicted probabilities.
        n_bins: Number of calibration bins.

    Returns:
        Dictionary with ECE and per-bin statistics.
    """
    bin_boundaries = np.linspace(0, 1, n_bins + 1)
    ece = 0.0
    total = len(y_true)

    for i in range(n_bins):
        mask = (y_proba >= bin_boundaries[i]) & (y_proba < bin_boundaries[i + 1])
        bin_size = mask.sum()

        if bin_size > 0:
            bin_accuracy = y_true[mask].mean()
            bin_confidence = y_proba[mask].mean()
            ece += (bin_size / total) * abs(bin_accuracy - bin_confidence)

    return {"ece": float(ece), "n_bins": n_bins}
