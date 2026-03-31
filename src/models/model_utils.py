"""
Model Utilities

Training helpers: checkpointing, early stopping, learning rate scheduling.

Author: SynthDetect Team
"""

import torch
import json
from pathlib import Path
from typing import Optional, Dict

from src.utils.logger import get_logger

logger = get_logger("models.utils")


class EarlyStopping:
    """Stop training when validation loss stops improving."""

    def __init__(self, patience: int = 5, min_delta: float = 0.001):
        self.patience = patience
        self.min_delta = min_delta
        self.counter = 0
        self.best_loss = float("inf")
        self.should_stop = False

    def __call__(self, val_loss: float) -> bool:
        if val_loss < self.best_loss - self.min_delta:
            self.best_loss = val_loss
            self.counter = 0
        else:
            self.counter += 1
            if self.counter >= self.patience:
                self.should_stop = True
                logger.info(f"Early stopping triggered after {self.counter} epochs")
        return self.should_stop


def save_checkpoint(
    model: torch.nn.Module,
    optimizer: torch.optim.Optimizer,
    epoch: int,
    loss: float,
    path: str,
    metadata: Optional[Dict] = None,
) -> None:
    """
    Save model checkpoint.

    Args:
        model: PyTorch model.
        optimizer: Optimizer state.
        epoch: Current epoch number.
        loss: Current loss value.
        path: Save path.
        metadata: Additional metadata to save.
    """
    checkpoint = {
        "epoch": epoch,
        "model_state_dict": model.state_dict(),
        "optimizer_state_dict": optimizer.state_dict(),
        "loss": loss,
    }
    if metadata:
        checkpoint["metadata"] = metadata

    Path(path).parent.mkdir(parents=True, exist_ok=True)
    torch.save(checkpoint, path)
    logger.info(f"Checkpoint saved: {path} (epoch={epoch}, loss={loss:.4f})")


def load_checkpoint(
    model: torch.nn.Module,
    path: str,
    optimizer: Optional[torch.optim.Optimizer] = None,
    device: str = "cpu",
) -> Dict:
    """
    Load model checkpoint.

    Args:
        model: PyTorch model to load weights into.
        path: Checkpoint path.
        optimizer: Optional optimizer to restore state.
        device: Device to map tensors to.

    Returns:
        Checkpoint dictionary with epoch, loss, and metadata.
    """
    checkpoint = torch.load(path, map_location=device, weights_only=False)
    model.load_state_dict(checkpoint["model_state_dict"])

    if optimizer and "optimizer_state_dict" in checkpoint:
        optimizer.load_state_dict(checkpoint["optimizer_state_dict"])

    logger.info(
        f"Checkpoint loaded: {path} "
        f"(epoch={checkpoint.get('epoch', '?')}, loss={checkpoint.get('loss', '?'):.4f})"
    )

    return checkpoint


def save_model_registry(
    model_name: str,
    version: str,
    metrics: Dict,
    path: str = "data/models/model_registry.json",
) -> None:
    """Save model metadata to registry for tracking."""
    registry_path = Path(path)
    registry_path.parent.mkdir(parents=True, exist_ok=True)

    registry = {}
    if registry_path.exists():
        with open(registry_path) as f:
            registry = json.load(f)

    registry[f"{model_name}_v{version}"] = {
        "name": model_name,
        "version": version,
        "metrics": metrics,
    }

    with open(registry_path, "w") as f:
        json.dump(registry, f, indent=2)

    logger.info(f"Model registered: {model_name} v{version}")
