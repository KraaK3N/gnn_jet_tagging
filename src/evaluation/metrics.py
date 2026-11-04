"""
Evaluation Metrics for Jet Tagging
==================================
Physics Metrics:
----------------
1. ROC AUC (Receiver Operating Characteristic - Area Under Curve)
2. Background Rejection R_X:
     R_X = 1 / epsilon_B   at   epsilon_S = X%
   where epsilon_S is true positive rate (signal efficiency) and
   epsilon_B is false positive rate (background misidentification rate).
   In Quark vs Gluon tagging: Signal = Quark (0), Background = Gluon (1).
"""

from typing import Dict, Any, Tuple
import time
import numpy as np
import torch
from sklearn.metrics import roc_curve, auc, accuracy_score, precision_recall_fscore_support


def compute_jet_metrics(
    y_true: np.ndarray,
    y_scores: np.ndarray,
    y_pred: np.ndarray,
) -> Dict[str, Any]:
    """
    Computes all standard collider physics metrics for binary jet tagging.

    Args:
        y_true: (N,) binary true labels (0 = Quark, 1 = Gluon).
        y_scores: (N,) probability of signal class (Quark, label 0).
        y_pred: (N,) predicted class labels (0 or 1).

    Returns:
        Dictionary with AUC, Accuracy, R50, R30, FPR, TPR.
    """
    # Convention: Signal is Quark (y_true == 0)
    # Binary indicator where 1 = Quark (Signal), 0 = Gluon (Background)
    signal_true = (y_true == 0).astype(int)

    # Compute ROC curve using signal probabilities
    fpr, tpr, thresholds = roc_curve(signal_true, y_scores)
    roc_auc = auc(fpr, tpr)

    # Calculate background rejection: 1 / FPR at TPR = 0.50 and TPR = 0.30
    r50 = compute_background_rejection(fpr, tpr, target_eff=0.50)
    r30 = compute_background_rejection(fpr, tpr, target_eff=0.30)

    # Accuracy, Precision, Recall
    acc = accuracy_score(y_true, y_pred) * 100.0
    prec, recall, f1, _ = precision_recall_fscore_support(y_true, y_pred, average="binary", pos_label=0)

    return {
        "accuracy": float(acc),
        "auc": float(roc_auc),
        "r50": float(r50),
        "r30": float(r30),
        "precision": float(prec),
        "recall": float(recall),
        "f1": float(f1),
        "fpr": fpr,
        "tpr": tpr,
    }


def compute_background_rejection(fpr: np.ndarray, tpr: np.ndarray, target_eff: float = 0.50) -> float:
    """
    Finds the background rejection 1 / FPR at a specified signal efficiency (TPR).
    """
    # Find the point on the ROC curve closest to target_eff
    idx = np.argmin(np.abs(tpr - target_eff))
    selected_fpr = fpr[idx]
    if selected_fpr <= 0.0:
        return 1e4  # Cap to avoid division by zero
    return float(1.0 / selected_fpr)


def benchmark_inference_latency(
    model: torch.nn.Module,
    sample_features: torch.Tensor,
    sample_coords: torch.Tensor,
    sample_mask: torch.Tensor,
    n_runs: int = 50,
) -> float:
    """
    Measures average inference latency per jet in milliseconds.
    """
    model.eval()
    device = next(model.parameters()).device
    feat = sample_features.to(device)
    coords = sample_coords.to(device)
    mask = sample_mask.to(device)

    # Warmup
    with torch.no_grad():
        for _ in range(5):
            _ = model(feat, coords, mask)

    # Timed run
    start_time = time.perf_counter()
    with torch.no_grad():
        for _ in range(n_runs):
            _ = model(feat, coords, mask)
    end_time = time.perf_counter()

    total_time_ms = (end_time - start_time) * 1000.0
    num_jets = feat.shape[0] * n_runs
    latency_per_jet_ms = total_time_ms / num_jets
    return latency_per_jet_ms
