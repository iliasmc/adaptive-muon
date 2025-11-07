"""Tests for the model module."""

import torch

from src.model import SimpleNet, evaluate, train_step


def test_simplenet_initialization():
    """Test SimpleNet initialization."""
    model = SimpleNet(input_size=784, hidden_size=128, output_size=10)

    assert isinstance(model, torch.nn.Module)
    assert model.fc1.in_features == 784
    assert model.fc1.out_features == 128
    assert model.fc2.in_features == 128
    assert model.fc2.out_features == 10


def test_simplenet_forward():
    """Test SimpleNet forward pass."""
    model = SimpleNet(input_size=784, hidden_size=128, output_size=10)
    batch_size = 32
    x = torch.randn(batch_size, 784)

    output = model(x)

    assert output.shape == (batch_size, 10)
    assert not torch.isnan(output).any()


def test_train_step():
    """Test training step."""
    model = SimpleNet(input_size=10, hidden_size=20, output_size=2)
    optimizer = torch.optim.Adam(model.parameters())
    criterion = torch.nn.CrossEntropyLoss()
    device = torch.device("cpu")

    data = torch.randn(4, 10)
    target = torch.randint(0, 2, (4,))

    loss, output = train_step(model, data, target, optimizer, criterion, device)

    assert isinstance(loss, float)
    assert output.shape == (4, 2)
    assert not torch.isnan(output).any()


def test_evaluate():
    """Test evaluation function."""
    model = SimpleNet(input_size=10, hidden_size=20, output_size=2)
    criterion = torch.nn.CrossEntropyLoss()
    device = torch.device("cpu")

    # Create a simple dataset and dataloader
    dataset = torch.utils.data.TensorDataset(torch.randn(20, 10), torch.randint(0, 2, (20,)))
    data_loader = torch.utils.data.DataLoader(dataset, batch_size=4)

    avg_loss, accuracy = evaluate(model, data_loader, criterion, device)

    assert isinstance(avg_loss, float)
    assert isinstance(accuracy, float)
    assert 0 <= accuracy <= 100
