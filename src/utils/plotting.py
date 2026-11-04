"""
Plotting Utilities for Jet Tagging Evaluation
==============================================
Generates publication-quality figures for:
  1. Combined ROC curves with AUC annotations
  2. Background rejection curves (1/FPR vs TPR)
  3. Training and validation loss/accuracy curves
"""

import os
from typing import Dict, Any
import numpy as np
import matplotlib
matplotlib.use("Agg")  # Headless backend for terminal / server execution
import matplotlib.pyplot as plt


def plot_roc_curves(
    results: Dict[str, Dict[str, Any]],
    save_path: str = "results/figures/roc_curve.png",
    title: str = "Quark vs. Gluon Jet Tagging - ROC Curves",
):
    """
    Plots comparative ROC curves for multiple models.

    Args:
        results: Dictionary mapping model_name -> {
            'fpr': np.ndarray,
            'tpr': np.ndarray,
            'auc': float,
            'r50': float
        }
        save_path: Output file path.
        title: Figure title.
    """
    os.makedirs(os.path.dirname(save_path), exist_ok=True)

    plt.figure(figsize=(8, 6), dpi=200)
    plt.plot([0, 1], [0, 1], linestyle="--", color="gray", label="Random Guess (AUC = 0.5000)", alpha=0.7)

    colors = ["#2b5c8f", "#d95f02", "#7570b3", "#1b9e77"]
    for i, (name, res) in enumerate(results.items()):
        color = colors[i % len(colors)]
        fpr = res["fpr"]
        tpr = res["tpr"]
        auc = res["auc"]
        r50 = res.get("r50", 0.0)

        label = f"{name} (AUC = {auc:.4f}, $1/\\varepsilon_B$ @ 50% = {r50:.1f})"
        plt.plot(fpr, tpr, color=color, lw=2.2, label=label)

    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel("False Positive Rate (Gluon Misidentification $\\varepsilon_B$)", fontsize=12)
    plt.ylabel("True Positive Rate (Quark Signal Efficiency $\\varepsilon_S$)", fontsize=12)
    plt.title(title, fontsize=13, fontweight="bold", pad=12)
    plt.grid(True, linestyle=":", alpha=0.6)
    plt.legend(loc="lower right", fontsize=10, frameon=True, framealpha=0.9)
    plt.tight_layout()
    plt.savefig(save_path)
    plt.close()
    print(f"ROC curve figure saved to: {save_path}")


def plot_training_history(
    history: Dict[str, list],
    model_name: str,
    save_path: str,
):
    """
    Plots training and validation loss and accuracy over epochs.
    """
    os.makedirs(os.path.dirname(save_path), exist_ok=True)

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 4.5), dpi=150)
    epochs = range(1, len(history["train_loss"]) + 1)

    # Loss subplot
    ax1.plot(epochs, history["train_loss"], "o-", label="Train Loss", color="#1f77b4")
    ax1.plot(epochs, history["val_loss"], "s--", label="Val Loss", color="#ff7f0e")
    ax1.set_xlabel("Epoch", fontsize=11)
    ax1.set_ylabel("Cross Entropy Loss", fontsize=11)
    ax1.set_title(f"{model_name} - Loss vs Epoch", fontweight="bold")
    ax1.grid(True, linestyle=":", alpha=0.6)
    ax1.legend()

    # Accuracy subplot
    ax2.plot(epochs, history["train_acc"], "o-", label="Train Acc", color="#2ca02c")
    ax2.plot(epochs, history["val_acc"], "s--", label="Val Acc", color="#d62728")
    ax2.set_xlabel("Epoch", fontsize=11)
    ax2.set_ylabel("Accuracy (%)", fontsize=11)
    ax2.set_title(f"{model_name} - Accuracy vs Epoch", fontweight="bold")
    ax2.grid(True, linestyle=":", alpha=0.6)
    ax2.legend()

    plt.tight_layout()
    plt.savefig(save_path)
    plt.close()
    print(f"Training history figure saved to: {save_path}")
