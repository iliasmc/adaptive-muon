"""Tests for the utils module."""

import tempfile
from pathlib import Path

import torch

from src.model import SimpleNet
from src.utils import count_parameters, get_device, load_checkpoint, save_checkpoint, set_seed


def test_set_seed():
    """Test seed setting for reproducibility."""
    set_seed(42)
    x1 = torch.rand(10)

    set_seed(42)
    x2 = torch.rand(10)

    assert torch.allclose(x1, x2)


def test_get_device():
    """Test device selection."""
    device = get_device()

    assert isinstance(device, torch.device)
    # Device should be either CPU or CUDA
    assert device.type in ["cpu", "cuda"]


def test_count_parameters():
    """Test parameter counting."""
    model = SimpleNet(input_size=784, hidden_size=128, output_size=10)

    num_params = count_parameters(model)

    # Expected: 784*128 + 128 + 128*10 + 10 = 101,770
    expected = 784 * 128 + 128 + 128 * 10 + 10
    assert num_params == expected


def test_save_and_load_checkpoint():
    """Test checkpoint saving and loading."""
    model = SimpleNet(input_size=10, hidden_size=20, output_size=2)
    optimizer = torch.optim.Adam(model.parameters())
    device = torch.device("cpu")

    # Save checkpoint
    with tempfile.TemporaryDirectory() as tmpdir:
        filepath = Path(tmpdir) / "test_checkpoint.pth"
        save_checkpoint(model, optimizer, epoch=5, loss=0.5, filepath=str(filepath))

        # Create new model and optimizer
        new_model = SimpleNet(input_size=10, hidden_size=20, output_size=2)
        new_optimizer = torch.optim.Adam(new_model.parameters())

        # Load checkpoint
        epoch = load_checkpoint(new_model, new_optimizer, str(filepath), device)

        assert epoch == 5
        # Check that model weights are the same
        for p1, p2 in zip(model.parameters(), new_model.parameters()):
            assert torch.allclose(p1, p2)
