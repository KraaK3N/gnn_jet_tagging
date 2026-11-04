"""
Loss Functions for Jet Classification
=====================================
"""

import torch
import torch.nn as nn


def get_loss_function(label_smoothing: float = 0.0, class_weights: list = None) -> nn.Module:
    """
    Returns the standard CrossEntropyLoss configured with optional label smoothing.
    """
    weight_tensor = torch.tensor(class_weights, dtype=torch.float32) if class_weights else None
    return nn.CrossEntropyLoss(weight=weight_tensor, label_smoothing=label_smoothing)
