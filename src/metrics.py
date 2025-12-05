import itertools

import numpy as np
import torch
from backpack import backpack, extend
from backpack.extensions import DiagHessian
from curvlinops import HessianLinearOperator, hutchinson_squared_fro
from pyhessian import hessian
from scipy.sparse.linalg import eigsh

from .train_model import CifarLoader, CifarNet


def analyze_hessian_eigenvalues(weight_path, ev_num, batch_size=2000):
    """
    Load a .pth file and compute Hessian eigenvalues

    Args:
        weight_path: Path to the .pth file
        dataloader: DataLoader for your dataset
        criterion: Loss function
    """

    # Load the model
    model = CifarNet().to(torch.device("mps"))

    train_loader = CifarLoader(
        "cifar10", train=True, batch_size=batch_size, aug=dict(flip=True, translate=2)
    )

    # Load the checkpoint
    checkpoint = torch.load(weight_path, map_location="mps")

    # Load weights into model
    if "state_dict" in checkpoint:
        model.load_state_dict(checkpoint["state_dict"])
    else:
        # If the file directly contains state_dict
        model.load_state_dict(checkpoint)

    model.eval()  # Set to evaluation mode

    print(f"Model loaded from {weight_path}")

    inputs, targets = next(iter(train_loader))
    crit = torch.nn.CrossEntropyLoss()
    hessian_comp = hessian(model, crit, data=(inputs, targets), cuda=False)

    # Compute top eigenvalues (e.g., top 10)
    top_eigenvalues, top_eigenvector = hessian_comp.eigenvalues(top_n=ev_num)

    # ev_list = []
    print(f"Top {ev_num} Hessian eigenvalues:")
    for i, eigenval in enumerate(top_eigenvalues):
        print(f"Eigenvalue {i+1}: {eigenval:.6f}")

    # Compute the trace (sum of all eigenvalues)
    trace = hessian_comp.trace()
    print(f"\nTrace of Hessian: {trace:.6f}")

    # Compute density of eigenvalues
    eigenvalues, density = hessian_comp.density()

    return top_eigenvalues, trace, density


def analyze_weight_matrices(weight_path):
    """Analyze eigenvalues of weight matrices directly"""

    checkpoint = torch.load(weight_path, map_location="cpu")
    state_dict = checkpoint["state_dict"] if "state_dict" in checkpoint else checkpoint

    for name, param in state_dict.items():
        if "weight" in name and len(param.shape) == 2:  # 2D weight matrices
            # Convert to numpy for eigenvalue computation
            weight_matrix = param.detach().numpy()

            # Compute eigenvalues
            eigenvalues = np.linalg.eigvals(weight_matrix)

            print(f"\n{name}:")
            print(f"  Shape: {weight_matrix.shape}")
            print(f"  Top 5 eigenvalues: {eigenvalues[:5]}")
            print(f"  Spectral radius: {np.max(np.abs(eigenvalues)):.4f}")


# pip install backpack-for-pytorch


def compute_diagonal_hessian(model, dataloader, criterion):
    """Compute diagonal of Hessian - much more memory efficient"""

    model = extend(model)  # Extend model with Backpack functionality

    data_iter = iter(dataloader)
    inputs, targets = next(data_iter)

    outputs = model(inputs)
    loss = criterion(outputs, targets)

    with backpack(DiagHessian()):
        loss.backward()

        # Access diagonal Hessian for each parameter
        diag_hessian = []
        for param in model.parameters():
            diag_h = param.diag_h
            diag_hessian.append(diag_h.flatten())

        full_diag = torch.cat(diag_hessian)

        # Rank estimation from diagonal (approximate)
        threshold = 1e-6
        rank_estimate = torch.sum(full_diag > threshold).item()

        return full_diag, rank_estimate


class HessianRankEstimator:
    def __init__(self, model, dataloader, criterion):
        self.model = model
        self.dataloader = dataloader
        self.criterion = criterion

    def hutchinson_trace_estimator(self, num_samples=100):
        """Estimate Hessian trace using Hutchinson method"""
        model = self.model
        model.eval()

        # Get a batch of data
        data_iter = iter(self.dataloader)
        inputs, targets = next(data_iter)

        # Compute loss and gradients
        outputs = model(inputs)
        loss = self.criterion(outputs, targets)

        # First gradient computation
        grad_params = torch.autograd.grad(loss, model.parameters(), create_graph=True)
        flat_grad = torch.cat([g.view(-1) for g in grad_params])

        trace_estimate = 0.0

        for i in range(num_samples):
            # Random vector with same dimension as parameters
            v = torch.randn_like(flat_grad)

            # Compute Hv = ∇²f v
            Hv = torch.autograd.grad(
                flat_grad, model.parameters(), grad_outputs=v, retain_graph=True
            )
            Hv_flat = torch.cat([h.view(-1) for h in Hv])

            # vᵀ H v
            trace_estimate += torch.dot(v, Hv_flat).item()

            # Clear intermediate gradients
            for p in model.parameters():
                if p.grad is not None:
                    p.grad.zero_()

        return trace_estimate / num_samples

    def power_iteration(self, num_iterations=100):
        """Estimate top eigenvalue using power iteration"""
        model = self.model
        model.eval()

        data_iter = iter(self.dataloader)
        inputs, targets = next(data_iter)

        # Initialize random vector
        total_params = sum(p.numel() for p in model.parameters())
        v = torch.randn(total_params)
        v = v / torch.norm(v)

        eigenvalues = []

        for i in range(num_iterations):
            # Compute loss and gradients
            outputs = model(inputs)
            loss = self.criterion(outputs, targets)

            # Compute gradient
            grad_params = torch.autograd.grad(loss, model.parameters(), create_graph=True)
            flat_grad = torch.cat([g.view(-1) for g in grad_params])

            # Compute Hessian-vector product: Hv
            Hv = torch.autograd.grad(
                flat_grad, model.parameters(), grad_outputs=v, retain_graph=True
            )
            Hv_flat = torch.cat([h.view(-1) for h in Hv])

            # Rayleigh quotient: vᵀ H v / vᵀ v
            eigenvalue = torch.dot(v, Hv_flat).item()
            eigenvalues.append(eigenvalue)

            # Update vector for next iteration
            v = Hv_flat / torch.norm(Hv_flat)

            # Clear gradients
            for p in model.parameters():
                if p.grad is not None:
                    p.grad.zero_()

        return eigenvalues

    def estimate_rank(self, threshold=1e-6, num_samples=50):
        """Estimate Hessian rank using stochastic approximation"""
        # Use power iteration to get eigenvalue distribution
        eigenvalues = self.power_iteration(num_samples)
        eigenvalues = np.array(eigenvalues)

        # Count eigenvalues above threshold
        rank = np.sum(np.abs(eigenvalues) > threshold)

        return rank, eigenvalues


# https://medium.com/@ph_singer/handling-huge-matrices-in-python-dff4e31d4417
# https://github.com/EigenPro/EigenPro-pytorch
# https://github.com/touqir14/LUP-rank-computer
# https://www.cs.ubc.ca/~nickhar/W12/Lecture15Notes.pdf


def analyze_hessian_stable_rank(
    weight_path, batch_size=2000, num_matvecs=10, top_k=1, distribution="rademacher", device="cpu"
):
    """
    Load a .pth file and compute Hessian stable rank using CurvLinOps

    Args:
        weight_path: Path to the .pth file
        batch_size: Batch size for data loading
        num_matvecs: Total number of matrix-vector products to use for Frobenius norm estimation.
                    Must be smaller than the minimum dimension of the matrix.
        top_k: Number of top eigenvalues to compute
        distribution: Distribution of random vectors for trace estimation.
                     Can be either 'rademacher' or 'normal'. Default: 'rademacher'
    Returns:
        tuple[stable_rank, sharpness]
    """

    # Load the model (assuming CifarNet architecture)
    model = CifarNet().to(torch.device(device))

    # Create data loaders (using your existing setup)
    train_loader = CifarLoader(
        "cifar10", train=True, batch_size=batch_size, aug=dict(flip=True, translate=2)
    )
    # test_loader = CifarLoader("cifar10", train=False, batch_size=batch_size)

    # Load the checkpoint
    checkpoint = torch.load(weight_path, map_location=device)

    # Load weights into model
    if "state_dict" in checkpoint:
        model.load_state_dict(checkpoint["state_dict"])
    else:
        model.load_state_dict(checkpoint)

    model.eval()  # Set to evaluation mode

    print(f"Model loaded from {weight_path}")

    # Get a batch of data
    inputs, targets = next(iter(train_loader))
    data_subset = list(itertools.islice(train_loader, batch_size))

    # Move data to correct device
    data_subset = [(x.to(device), y.to(device)) for x, y in data_subset]
    criterion = torch.nn.CrossEntropyLoss()
    # Create Hessian linear operator using CurvLinOps
    H = HessianLinearOperator(
        model_func=model,
        loss_func=criterion,
        data=data_subset,
        params=[p for p in model.parameters() if p.requires_grad],
    )
    print(f"Hessian dimension: {H.shape}")
    print(f"Total parameters: {H.shape[1]}")

    # Check if num_matvecs is valid
    min_dimension = min(H.shape)
    if num_matvecs >= min_dimension:
        print(
            f"Warning: num_matvecs ({num_matvecs}) should be smaller than minimum dimension ({min_dimension})"
        )
        print(f"Reducing num_matvecs to {min_dimension - 1}")
        num_matvecs = min_dimension - 1

    # Compute squared Frobenius norm using CurvLinOps built-in function
    print(
        f"Computing squared Frobenius norm using {num_matvecs} matrix-vector products ({distribution} distribution)..."
    )
    frob_norm_sq = hutchinson_squared_fro(
        A=H,
        num_matvecs=num_matvecs,
        distribution=distribution,
    )
    print(f"Squared Frobenius norm (||H||_F²): {frob_norm_sq:.6e}")

    # Compute top eigenvalues for spectral norm and rank analysis
    print(f"Computing top {top_k} eigenvalues...")
    H_sp = H.to_scipy()
    eigenvalues, _ = eigsh(H_sp, k=top_k, which="LM", tol=1e-2)
    top_eigenvalue = np.max(np.abs(eigenvalues))
    spectral_norm_sq = top_eigenvalue**2

    print(f"Squared spectral norm (||H||_2²): {spectral_norm_sq:.6e}")

    # Compute stable rank: r(H) = ||H||_F² / ||H||_2²
    stable_rank = frob_norm_sq / spectral_norm_sq
    print(f"\nStable rank: {stable_rank:.6f}")

    return stable_rank, top_eigenvalue
