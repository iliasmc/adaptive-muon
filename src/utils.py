"""
Utility functions for PyTorch training and data handling.

This module provides helper functions for common tasks in PyTorch projects.
"""

import random
from typing import Optional

import numpy as np
import torch


def set_seed(seed: int = 42) -> None:
    """
    Set random seeds for reproducibility.

    This function sets seeds for Python's random module, NumPy, and PyTorch
    to ensure reproducible results across runs.

    Args:
        seed (int): Seed value. Default: 42

    Example:
        >>> set_seed(123)
        >>> torch.rand(1)
        tensor([0.5328])
    """
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False


def get_device(device_id: Optional[int] = None) -> torch.device:
    """
    Get the appropriate device (CPU or CUDA).

    Args:
        device_id (Optional[int]): Specific CUDA device ID. If None, uses device 0.
                                   If CUDA is not available, returns CPU.

    Returns:
        torch.device: Device object for computation.

    Example:
        >>> device = get_device()
        >>> print(device)
        cuda:0  # or cpu
    """
    if torch.cuda.is_available():
        if device_id is not None:
            return torch.device(f"cuda:{device_id}")
        return torch.device("cuda:0")
    return torch.device("cpu")


def count_parameters(model: torch.nn.Module) -> int:
    """
    Count the number of trainable parameters in a model.

    Args:
        model (torch.nn.Module): PyTorch model.

    Returns:
        int: Number of trainable parameters.

    Example:
        >>> from src.model import SimpleNet
        >>> model = SimpleNet(784, 128, 10)
        >>> num_params = count_parameters(model)
        >>> print(f"Model has {num_params:,} parameters")
        Model has 101,770 parameters
    """
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


def save_checkpoint(
    model: torch.nn.Module,
    optimizer: torch.optim.Optimizer,
    epoch: int,
    loss: float,
    filepath: str,
) -> None:
    """
    Save a model checkpoint.

    Args:
        model (torch.nn.Module): Model to save.
        optimizer (torch.optim.Optimizer): Optimizer state to save.
        epoch (int): Current epoch number.
        loss (float): Current loss value.
        filepath (str): Path to save the checkpoint.

    Example:
        >>> from src.model import SimpleNet
        >>> model = SimpleNet(784, 128, 10)
        >>> optimizer = torch.optim.Adam(model.parameters())
        >>> save_checkpoint(model, optimizer, 10, 0.5, "checkpoint.pth")
    """
    checkpoint = {
        "epoch": epoch,
        "model_state_dict": model.state_dict(),
        "optimizer_state_dict": optimizer.state_dict(),
        "loss": loss,
    }
    torch.save(checkpoint, filepath)


def load_checkpoint(
    model: torch.nn.Module,
    optimizer: torch.optim.Optimizer,
    filepath: str,
    device: torch.device,
) -> int:
    """
    Load a model checkpoint.

    Args:
        model (torch.nn.Module): Model to load weights into.
        optimizer (torch.optim.Optimizer): Optimizer to load state into.
        filepath (str): Path to the checkpoint file.
        device (torch.device): Device to load the model on.

    Returns:
        int: Epoch number from the checkpoint.

    Example:
        >>> from src.model import SimpleNet
        >>> model = SimpleNet(784, 128, 10)
        >>> optimizer = torch.optim.Adam(model.parameters())
        >>> device = get_device()
        >>> epoch = load_checkpoint(model, optimizer, "checkpoint.pth", device)
    """
    checkpoint = torch.load(filepath, map_location=device)
    model.load_state_dict(checkpoint["model_state_dict"])
    optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
    return checkpoint["epoch"]
