"""
Model Training Pipeline for Jet Tagging
=======================================
Supports training:
  1. Deep Sets Baseline (no edges)
  2. Dynamic EdgeConv GNN (ParticleNet-lite)

Usage:
------
python -m src.training.train --config configs/baseline.yaml
python -m src.training.train --config configs/gnn.yaml
python -m src.training.train --config configs/gnn.yaml --quick
"""

import os
import argparse
import time
from typing import Dict, Any, Tuple
import torch
import torch.nn as nn
from torch.utils.data import DataLoader

from src.utils.config import load_config
from src.utils.seed import set_seed
from src.utils.plotting import plot_training_history
from src.data.dataset import load_jet_dataset
from src.models.baseline import DeepSetsJetTagger
from src.models.gnn import ParticleNetLite
from src.training.losses import get_loss_function


def build_model(config: Dict[str, Any], device: torch.device) -> nn.Module:
    """Instantiates model according to config parameters."""
    m_type = config.get("model_type", "deepsets").lower()
    m_cfg = config.get("model", {})

    if m_type == "deepsets":
        model = DeepSetsJetTagger(
            input_dim=m_cfg.get("input_dim", 7),
            hidden_dims=m_cfg.get("hidden_dims", [64, 128, 128]),
            classifier_dims=m_cfg.get("classifier_dims", [128, 64]),
            pooling=m_cfg.get("pooling", "mean_max"),
            dropout=m_cfg.get("dropout", 0.1),
            num_classes=m_cfg.get("num_classes", 2),
        )
    elif m_type in ["gnn", "particlenet", "edgeconv"]:
        conv_params = m_cfg.get("conv_params", [(64, 16), (128, 16), (128, 16)])
        # Ensure tuple format
        conv_params = [tuple(p) for p in conv_params]
        model = ParticleNetLite(
            input_dim=m_cfg.get("input_dim", 7),
            num_classes=m_cfg.get("num_classes", 2),
            conv_params=conv_params,
            fc_params=m_cfg.get("fc_params", [128, 64]),
            k=m_cfg.get("k", 16),
            dynamic=m_cfg.get("dynamic", True),
            dropout=m_cfg.get("dropout", 0.1),
        )
    else:
        raise ValueError(f"Unsupported model type: {m_type}")

    return model.to(device)


def train_one_epoch(
    model: nn.Module,
    loader: DataLoader,
    optimizer: torch.optim.Optimizer,
    criterion: nn.Module,
    device: torch.device,
) -> Tuple[float, float]:
    """Runs a single training epoch."""
    model.train()
    total_loss = 0.0
    correct = 0
    total = 0

    for features, coordinates, mask, labels in loader:
        features = features.to(device)
        coordinates = coordinates.to(device)
        mask = mask.to(device)
        labels = labels.to(device)

        optimizer.zero_grad()
        logits = model(features, coordinates, mask)
        loss = criterion(logits, labels)
        loss.backward()
        optimizer.step()

        total_loss += loss.item() * len(labels)
        preds = logits.argmax(dim=-1)
        correct += (preds == labels).sum().item()
        total += len(labels)

    epoch_loss = total_loss / max(total, 1)
    epoch_acc = (correct / max(total, 1)) * 100.0
    return epoch_loss, epoch_acc


def evaluate_epoch(
    model: nn.Module,
    loader: DataLoader,
    criterion: nn.Module,
    device: torch.device,
) -> Tuple[float, float]:
    """Evaluates validation loss and accuracy."""
    model.eval()
    total_loss = 0.0
    correct = 0
    total = 0

    with torch.no_grad():
        for features, coordinates, mask, labels in loader:
            features = features.to(device)
            coordinates = coordinates.to(device)
            mask = mask.to(device)
            labels = labels.to(device)

            logits = model(features, coordinates, mask)
            loss = criterion(logits, labels)

            total_loss += loss.item() * len(labels)
            preds = logits.argmax(dim=-1)
            correct += (preds == labels).sum().item()
            total += len(labels)

    epoch_loss = total_loss / max(total, 1)
    epoch_acc = (correct / max(total, 1)) * 100.0
    return epoch_loss, epoch_acc


def run_training(config: Dict[str, Any], quick: bool = False):
    """Executes full training loop."""
    t_cfg = config.get("training", {})
    out_cfg = config.get("output", {})
    seed = t_cfg.get("seed", 42)
    set_seed(seed)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"\n=======================================================")
    print(f"Training Model : {config.get('model_name', 'JetTagger')}")
    print(f"Device         : {device}")
    print(f"=======================================================")

    # Data
    data_path = config.get("data_path", "data/sample_jets.npz")
    if not os.path.exists(data_path):
        from src.data.synthetic import save_synthetic_dataset
        print(f"Dataset not found at {data_path}, generating synthetic benchmark...")
        os.makedirs(os.path.dirname(data_path) or ".", exist_ok=True)
        save_synthetic_dataset(data_path, n_samples=1000, seed=seed)

    splits = load_jet_dataset(data_path, seed=seed)
    batch_size = t_cfg.get("batch_size", 32)
    train_loader = DataLoader(splits["train"], batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(splits["val"], batch_size=batch_size, shuffle=False)

    print(f"Dataset splits: Train={len(splits['train'])}, Val={len(splits['val'])}, Test={len(splits['test'])}")

    # Model
    model = build_model(config, device)
    n_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"Trainable Parameters: {n_params:,}")

    # Training settings
    epochs = 2 if quick else t_cfg.get("epochs", 15)
    lr = float(t_cfg.get("learning_rate", 1e-3))
    weight_decay = float(t_cfg.get("weight_decay", 1e-4))

    optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=weight_decay)
    scheduler = torch.optim.lr_scheduler.StepLR(
        optimizer,
        step_size=t_cfg.get("scheduler_step", 5),
        gamma=t_cfg.get("scheduler_gamma", 0.5),
    )
    criterion = get_loss_function()

    checkpoint_dir = out_cfg.get("checkpoint_dir", "checkpoints")
    os.makedirs(checkpoint_dir, exist_ok=True)
    best_ckpt_path = os.path.join(checkpoint_dir, out_cfg.get("checkpoint_filename", "model_best.pt"))

    history = {"train_loss": [], "train_acc": [], "val_loss": [], "val_acc": []}
    best_val_acc = 0.0
    start_time = time.time()

    for epoch in range(1, epochs + 1):
        ep_start = time.time()
        train_loss, train_acc = train_one_epoch(model, train_loader, optimizer, criterion, device)
        val_loss, val_acc = evaluate_epoch(model, val_loader, criterion, device)
        scheduler.step()
        ep_duration = time.time() - ep_start

        history["train_loss"].append(train_loss)
        history["train_acc"].append(train_acc)
        history["val_loss"].append(val_loss)
        history["val_acc"].append(val_acc)

        # Checkpointing
        is_best = val_acc > best_val_acc
        if is_best:
            best_val_acc = val_acc
            torch.save({
                "epoch": epoch,
                "model_state_dict": model.state_dict(),
                "val_acc": val_acc,
                "val_loss": val_loss,
                "config": config,
            }, best_ckpt_path)

        star = " ★ (Best)" if is_best else ""
        print(f"Epoch [{epoch:02d}/{epochs:02d}] ({ep_duration:.2f}s) | "
              f"Train Loss: {train_loss:.4f}, Acc: {train_acc:.2f}% | "
              f"Val Loss: {val_loss:.4f}, Acc: {val_acc:.2f}%{star}")

    total_time = time.time() - start_time
    print(f"\nTraining completed in {total_time:.2f} seconds.")
    print(f"Best Validation Accuracy: {best_val_acc:.2f}%")
    print(f"Checkpoint saved to: {best_ckpt_path}")

    # Plot training curves
    figure_dir = out_cfg.get("figure_dir", "results/figures")
    m_type = config.get("model_type", "model")
    hist_path = os.path.join(figure_dir, f"{m_type}_training_history.png")
    plot_training_history(history, config.get("model_name", "Model"), hist_path)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train jet tagging model.")
    parser.add_argument("--config", type=str, required=True, help="Path to YAML config file")
    parser.add_argument("--quick", action="store_true", help="Run 2 micro-epochs for fast testing")
    args = parser.parse_args()

    cfg = load_config(args.config)
    run_training(cfg, quick=args.quick)
