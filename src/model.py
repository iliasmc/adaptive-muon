"""
Example PyTorch model demonstrating code quality standards.

This module shows proper type hints, docstrings, and PyTorch conventions.
"""

from typing import Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F


class SimpleNet(nn.Module):
    """
    A simple neural network for demonstration purposes.

    This network consists of two fully connected layers with ReLU activation.
    It demonstrates proper PyTorch module structure and documentation.

    Args:
        input_size (int): Size of input features.
        hidden_size (int): Size of hidden layer.
        output_size (int): Size of output layer.
        dropout_rate (float): Dropout probability. Default: 0.5

    Attributes:
        fc1 (nn.Linear): First fully connected layer.
        fc2 (nn.Linear): Second fully connected layer.
        dropout (nn.Dropout): Dropout layer for regularization.

    Example:
        >>> model = SimpleNet(input_size=784, hidden_size=128, output_size=10)
        >>> x = torch.randn(32, 784)
        >>> output = model(x)
        >>> output.shape
        torch.Size([32, 10])
    """

    def __init__(
        self,
        input_size: int,
        hidden_size: int,
        output_size: int,
        dropout_rate: float = 0.5,
    ) -> None:
        """Initialize the network layers."""
        super().__init__()
        self.fc1 = nn.Linear(input_size, hidden_size)
        self.fc2 = nn.Linear(hidden_size, output_size)
        self.dropout = nn.Dropout(dropout_rate)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass through the network.

        Args:
            x (torch.Tensor): Input tensor of shape (batch_size, input_size).

        Returns:
            torch.Tensor: Output tensor of shape (batch_size, output_size).
        """
        # First layer with ReLU activation
        x = F.relu(self.fc1(x))

        # Apply dropout for regularization
        x = self.dropout(x)

        # Output layer
        x = self.fc2(x)

        return x


def train_step(
    model: nn.Module,
    data: torch.Tensor,
    target: torch.Tensor,
    optimizer: torch.optim.Optimizer,
    criterion: nn.Module,
    device: torch.device,
) -> Tuple[float, torch.Tensor]:
    """
    Perform a single training step.

    Args:
        model (nn.Module): The neural network model.
        data (torch.Tensor): Input data batch.
        target (torch.Tensor): Target labels.
        optimizer (torch.optim.Optimizer): Optimizer for updating weights.
        criterion (nn.Module): Loss function.
        device (torch.device): Device to run computation on (CPU or CUDA).

    Returns:
        Tuple[float, torch.Tensor]: Loss value and model predictions.
    """
    model.train()

    # Move data to device
    data, target = data.to(device), target.to(device)

    # Zero gradients
    optimizer.zero_grad()

    # Forward pass
    output = model(data)

    # Compute loss
    loss = criterion(output, target)

    # Backward pass
    loss.backward()

    # Update weights
    optimizer.step()

    return loss.item(), output


def evaluate(
    model: nn.Module,
    data_loader: torch.utils.data.DataLoader,
    criterion: nn.Module,
    device: torch.device,
) -> Tuple[float, float]:
    """
    Evaluate model performance on a dataset.

    Args:
        model (nn.Module): The neural network model.
        data_loader (torch.utils.data.DataLoader): DataLoader for evaluation data.
        criterion (nn.Module): Loss function.
        device (torch.device): Device to run computation on.

    Returns:
        Tuple[float, float]: Average loss and accuracy.
    """
    model.eval()
    total_loss = 0.0
    correct = 0
    total = 0

    with torch.no_grad():
        for data, target in data_loader:
            data, target = data.to(device), target.to(device)

            # Forward pass
            output = model(data)

            # Compute loss
            loss = criterion(output, target)
            total_loss += loss.item()

            # Compute accuracy
            _, predicted = torch.max(output.data, 1)
            total += target.size(0)
            correct += (predicted == target).sum().item()

    avg_loss = total_loss / len(data_loader)
    accuracy = 100.0 * correct / total

    return avg_loss, accuracy
