#!/usr/bin/env python3
"""
Canned Demonstration Script for CS6012 Teaching Team
====================================================
Topic: Graph Neural Networks for Particle Jet Tagging
Group 11: Pradipta Sadhukhan (25D1080) & Rahul Kumar (25M2015)
Supervisor: Prof. Sharat Chandran

Description:
------------
This script provides a 1-click executable demo evaluating both:
  1. Deep Sets Baseline (no edges)
  2. Dynamic EdgeConv GNN (ParticleNet-lite)

It runs inference on the bundled benchmark test slice, computes standard
high-energy physics metrics (ROC AUC, Background Rejection 1/eps_B at 50% signal efficiency),
measures latency per jet, prints an ASCII summary table, and saves the
comparative ROC curve plot to `results/figures/demo_roc_curve.png`.

Usage:
------
./run_demo.sh
python demo.py
python demo.py --train_quick   (runs 2 micro-epochs if training test is desired)
"""

import os
import sys
import time
import argparse
import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader

# Ensure src is discoverable
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from src.utils.seed import set_seed
from src.utils.plotting import plot_roc_curves
from src.data.dataset import load_jet_dataset
from src.training.train import build_model, run_training
from src.utils.config import load_config
from src.evaluation.metrics import compute_jet_metrics, benchmark_inference_latency


def print_banner():
    print("""
================================================================================
   CS6012: GRAPH NEURAL NETWORKS FOR PARTICLE JET TAGGING (CANNED DEMO)
   Group 11: Pradipta Sadhukhan (25D1080) & Rahul Kumar (25M2015)
   Supervisor: Prof. Sharat Chandran | IIT Bombay
================================================================================
    """)


def run_demo(data_path: str = "data/sample_jets.npz", train_quick: bool = False):
    print_banner()
    set_seed(42)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"[*] Execution Device : {device}")

    # Ensure dataset exists
    if not os.path.exists(data_path):
        print(f"[*] Benchmark slice not found at {data_path}. Generating now...")
        from src.data.synthetic import save_synthetic_dataset
        os.makedirs(os.path.dirname(data_path) or ".", exist_ok=True)
        save_synthetic_dataset(data_path, n_samples=1000, seed=42)
    else:
        print(f"[*] Loaded Benchmark Dataset : {data_path}")

    # If user explicitly wants a quick training test
    if train_quick:
        print("\n[*] Running 2-epoch training verification test for Deep Sets and GNN...")
        cfg_ds = load_config("configs/baseline.yaml")
        run_training(cfg_ds, quick=True)
        cfg_gnn = load_config("configs/gnn.yaml")
        run_training(cfg_gnn, quick=True)

    # Checkpoints
    checkpoints = {
        "Deep Sets (No Edges Baseline)": "checkpoints/deepsets_best.pt",
        "Dynamic EdgeConv (ParticleNet-lite)": "checkpoints/edgeconv_best.pt",
    }

    # Verify checkpoints exist; if not, train them briefly
    for name, path in checkpoints.items():
        if not os.path.exists(path):
            print(f"[!] Checkpoint {path} not found. Training model automatically...")
            cfg_file = "configs/baseline.yaml" if "Deep" in name else "configs/gnn.yaml"
            cfg = load_config(cfg_file)
            run_training(cfg, quick=False)

    # Load test split
    splits = load_jet_dataset(data_path, seed=42)
    test_loader = DataLoader(splits["test"], batch_size=32, shuffle=False)
    print(f"[*] Test Set Size            : {len(splits['test'])} jets")

    results_for_plot = {}
    table_rows = []

    print("\n" + "=" * 85)
    print(f"{'Model Architecture':<34} | {'Accuracy':<9} | {'ROC AUC':<9} | {'1/eps_B@50%':<11} | {'Latency':<8} | {'Params'}")
    print("=" * 85)

    for model_label, ckpt_path in checkpoints.items():
        ckpt = torch.load(ckpt_path, map_location=device)
        model = build_model(ckpt["config"], device)
        model.load_state_dict(ckpt["model_state_dict"])
        model.eval()

        all_scores = []
        all_preds = []
        all_targets = []

        with torch.no_grad():
            for feat, coords, mask, labels in test_loader:
                feat = feat.to(device)
                coords = coords.to(device)
                mask = mask.to(device)

                logits = model(feat, coords, mask)
                probs = F.softmax(logits, dim=-1)

                # Quark probability (signal = 0)
                all_scores.extend(probs[:, 0].cpu().numpy())
                all_preds.extend(logits.argmax(dim=-1).cpu().numpy())
                all_targets.extend(labels.numpy())

        y_true = np.array(all_targets)
        y_scores = np.array(all_scores)
        y_pred = np.array(all_preds)

        metrics = compute_jet_metrics(y_true, y_scores, y_pred)

        # Benchmark latency
        sample_feat, sample_coords, sample_mask, _ = next(iter(test_loader))
        lat = benchmark_inference_latency(model, sample_feat, sample_coords, sample_mask, n_runs=25)
        n_params = sum(p.numel() for p in model.parameters() if p.requires_grad)

        print(f"{model_label:<34} | "
              f"{metrics['accuracy']:<8.2f}% | "
              f"{metrics['auc']:<9.4f} | "
              f"{metrics['r50']:<11.1f} | "
              f"{lat:<6.3f}ms | "
              f"{n_params:,}")

        results_for_plot[model_label] = {
            "fpr": metrics["fpr"],
            "tpr": metrics["tpr"],
            "auc": metrics["auc"],
            "r50": metrics["r50"],
        }

    print("=" * 85)

    # Save ROC curve figure
    demo_roc_path = "results/figures/demo_roc_curve.png"
    plot_roc_curves(results_for_plot, save_path=demo_roc_path, title="CS6012 Jet Tagging: Deep Sets vs. Dynamic GNN")

    print(f"\n[✓] High-resolution comparative ROC curve saved to:")
    print(f"    --> {demo_roc_path}")
    print("\n[✓] Demo executed successfully with zero configuration required.")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="CS6012 Canned Demo")
    parser.add_argument("--data_path", type=str, default="data/sample_jets.npz")
    parser.add_argument("--train_quick", action="store_true", help="Run quick 2-epoch training verification")
    args = parser.parse_args()

    run_demo(data_path=args.data_path, train_quick=args.train_quick)
