"""
Model Evaluation and Benchmark Comparison Pipeline
===================================================
Evaluates model checkpoints on the held-out test dataset:
  - Accuracy, Precision, Recall, F1
  - Receiver Operating Characteristic (ROC) AUC
  - Background Rejection R_50 (1 / epsilon_B at epsilon_S = 50%)
  - Inference Latency per jet (milliseconds)
  - Number of trainable parameters

Outputs:
  - results/metrics_summary.json
  - results/figures/roc_curve.png
"""

import os
import json
import argparse
from typing import Dict, Any, List
import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader

from src.utils.config import load_config
from src.utils.plotting import plot_roc_curves
from src.data.dataset import load_jet_dataset
from src.training.train import build_model
from src.evaluation.metrics import compute_jet_metrics, benchmark_inference_latency


def evaluate_checkpoint(
    checkpoint_path: str,
    test_loader: DataLoader,
    device: torch.device,
) -> Dict[str, Any]:
    """Evaluates a single saved model checkpoint on the test set."""
    if not os.path.exists(checkpoint_path):
        raise FileNotFoundError(f"Checkpoint not found at: {checkpoint_path}")

    checkpoint = torch.load(checkpoint_path, map_location=device)
    config = checkpoint["config"]
    model = build_model(config, device)
    model.load_state_dict(checkpoint["model_state_dict"])
    model.eval()

    all_scores = []
    all_preds = []
    all_targets = []

    with torch.no_grad():
        for features, coordinates, mask, labels in test_loader:
            features = features.to(device)
            coordinates = coordinates.to(device)
            mask = mask.to(device)

            logits = model(features, coordinates, mask)
            probs = F.softmax(logits, dim=-1)

            # Score for Signal (Quark = 0)
            quark_probs = probs[:, 0].cpu().numpy()
            preds = logits.argmax(dim=-1).cpu().numpy()

            all_scores.extend(quark_probs)
            all_preds.extend(preds)
            all_targets.extend(labels.numpy())

    y_true = np.array(all_targets)
    y_scores = np.array(all_scores)
    y_pred = np.array(all_preds)

    metrics = compute_jet_metrics(y_true, y_scores, y_pred)

    # Benchmark latency
    sample_feat, sample_coords, sample_mask, _ = next(iter(test_loader))
    latency_ms = benchmark_inference_latency(
        model, sample_feat, sample_coords, sample_mask, n_runs=30
    )
    n_params = sum(p.numel() for p in model.parameters() if p.requires_grad)

    metrics["latency_ms_per_jet"] = float(latency_ms)
    metrics["num_parameters"] = int(n_params)
    metrics["model_name"] = config.get("model_name", "Model")
    return metrics


def run_comparative_evaluation(
    checkpoints: List[str],
    data_path: str = "data/sample_jets.npz",
    output_json: str = "results/metrics_summary.json",
    output_roc: str = "results/figures/roc_curve.png",
):
    """
    Evaluates multiple checkpoints and prints a side-by-side comparison table.
    """
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    splits = load_jet_dataset(data_path, seed=42)
    test_loader = DataLoader(splits["test"], batch_size=32, shuffle=False)

    results_for_plot = {}
    summary_to_save = {}

    print("\n" + "=" * 78)
    print("           JET TAGGING BENCHMARK: COMPARATIVE EVALUATION")
    print("=" * 78)
    print(f"{'Model':<30} | {'Acc (%)':<9} | {'ROC AUC':<9} | {'1/eps_B@50%':<11} | {'Latency':<8} | {'Params'}")
    print("-" * 78)

    for ckpt in checkpoints:
        if not os.path.exists(ckpt):
            print(f"Skipping {ckpt} (not found)")
            continue

        metrics = evaluate_checkpoint(ckpt, test_loader, device)
        name = metrics["model_name"]

        print(f"{name:<30} | "
              f"{metrics['accuracy']:<9.2f} | "
              f"{metrics['auc']:<9.4f} | "
              f"{metrics['r50']:<11.1f} | "
              f"{metrics['latency_ms_per_jet']:<6.3f}ms | "
              f"{metrics['num_parameters']:,}")

        results_for_plot[name] = {
            "fpr": metrics["fpr"],
            "tpr": metrics["tpr"],
            "auc": metrics["auc"],
            "r50": metrics["r50"],
        }

        # Serializable summary
        summary_to_save[name] = {
            "accuracy": metrics["accuracy"],
            "auc": metrics["auc"],
            "r50_background_rejection": metrics["r50"],
            "r30_background_rejection": metrics["r30"],
            "latency_ms_per_jet": metrics["latency_ms_per_jet"],
            "num_parameters": metrics["num_parameters"],
            "f1_score": metrics["f1"],
        }

    print("=" * 78)

    # Save summary JSON
    os.makedirs(os.path.dirname(output_json) or ".", exist_ok=True)
    with open(output_json, "w") as f:
        json.dump(summary_to_save, f, indent=2)
    print(f"Metrics summary written to: {output_json}")

    # Plot comparative ROC curve
    if results_for_plot:
        plot_roc_curves(results_for_plot, save_path=output_roc)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Evaluate model checkpoints on test dataset.")
    parser.add_argument("--checkpoints", nargs="+", default=["checkpoints/deepsets_best.pt", "checkpoints/edgeconv_best.pt"])
    parser.add_argument("--data_path", type=str, default="data/sample_jets.npz")
    args = parser.parse_args()

    run_comparative_evaluation(args.checkpoints, data_path=args.data_path)
