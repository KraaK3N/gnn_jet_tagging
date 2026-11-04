"""
Training package.
"""

from src.training.losses import get_loss_function
from src.training.train import run_training

__all__ = ["get_loss_function", "run_training"]
