"""
https://medium.com/@ph_singer/handling-huge-matrices-in-python-dff4e31d4417
https://github.com/EigenPro/EigenPro-pytorch
https://github.com/touqir14/LUP-rank-computer
https://www.cs.ubc.ca/~nickhar/W12/Lecture15Notes.pdf
"""
import itertools
import os
import time

import matplotlib.pyplot as plt
import numpy as np
import torch
from backpack import backpack, extend
from backpack.extensions import DiagHessian
from curvlinops import HessianLinearOperator, hutchinson_squared_fro
from pyhessian import hessian
from scipy.sparse.linalg import eigsh

from .train_model import CifarLoader, CifarNet


def analyze_hessian_eigenvalues(model, batch, ev_num, device="cuda"):
    """
    Load a .pth file and compute Hessian eigenvalues

    Args:
        weight_path: Path to the .pth file
        dataloader: DataLoader for your dataset
        criterion: Loss function
    Returns:
        top_eigenvalues, trace, density
    """

    crit = torch.nn.CrossEntropyLoss()

    hessian_comp = hessian(model, crit, data=batch, cuda=(device=="cuda"))

    # Compute top eigenvalues (e.g., top 10)
    top_eigenvalues, top_eigenvector = hessian_comp.eigenvalues(top_n=ev_num)

    # ev_list = []
    print(f"Top {ev_num} Hessian eigenvalues:")
    for i, eigenval in enumerate(top_eigenvalues):
        print(f"Eigenvalue {i+1}: {eigenval:.6f}")

    # Compute the trace (sum of all eigenvalues)
    trace = hessian_comp.trace()
    print(trace)
    trace = np.mean(trace)
    print(f"\nTrace of Hessian: {trace:.6f}")

    # Compute density of eigenvalues
    # hessian_density = hessian_comp.density()

    return top_eigenvalues, trace, #hessian_density


def analyze_hessian_stable_rank(
    model, batch, num_matvecs=100, top_k=1, distribution="rademacher", device="cpu"
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
    criterion = torch.nn.CrossEntropyLoss()
    # Create Hessian linear operator using CurvLinOps
    # Disable determinism check since data augmentation causes slight variations
    H = HessianLinearOperator(
        model_func=model,
        loss_func=criterion,
        data=[batch],
        params=[p for p in model.parameters() if p.requires_grad],
        check_deterministic=False,
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
    eigenvalues, _ = eigsh(H_sp, k=top_k, which="LM", tol=1e-4)
    top_eigenvalue = np.max(np.abs(eigenvalues))
    spectral_norm_sq = top_eigenvalue**2

    print(f"Squared spectral norm (||H||_2²): {spectral_norm_sq:.6e}")

    # Compute stable rank: r(H) = ||H||_F² / ||H||_2²
    stable_rank = frob_norm_sq / spectral_norm_sq
    print(f"\nStable rank: {stable_rank:.6f}")

    # Convert to Python floats for compatibility with matplotlib
    if isinstance(stable_rank, torch.Tensor):
        stable_rank = stable_rank.cpu().item()
    else:
        stable_rank = float(stable_rank)
    
    if isinstance(top_eigenvalue, torch.Tensor):
        top_eigenvalue = top_eigenvalue.cpu().item()
    else:
        top_eigenvalue = float(top_eigenvalue)

    return stable_rank, top_eigenvalue


if __name__ == "__main__":
    weights_path = "./model_weights/"  # Path to model weights directory
    device = "cuda"

    # Get weight files and sort by epoch number
    weight_files = sorted(
        os.listdir(weights_path),
        key=lambda x: int(x.split("_")[-1].split(".")[0])  # Extract epoch number
    )

    stable_ranks = []
    sharpnesses = []
    
    traces = []
    densities = []
    for weight_file in weight_files:
        weight_path = os.path.join(weights_path, weight_file)

        start_time = time.time()

        # Load the model at this checkpoint
        model = CifarNet().to(torch.device(device))
        checkpoint = torch.load(weight_path, map_location=torch.device(device))
        if "state_dict" in checkpoint:
            model.load_state_dict(checkpoint["state_dict"])
        else:
            model.load_state_dict(checkpoint)
        model.eval() 

        # Load a batch to compute L
        train_loader = CifarLoader(
            "cifar10", train=True, batch_size=500, aug=dict(flip=True, translate=2)
        )
        batch = next(iter(train_loader))

        # Compute the metrics
        stable_rank, sharpness = analyze_hessian_stable_rank(model, batch, device=device)
        top_eigenvalues, trace = analyze_hessian_eigenvalues(model, batch, ev_num=10, device=device)
        traces.append(trace)
        stable_ranks.append(stable_rank)
        sharpnesses.append(sharpness)
        elapsed_time = time.time() - start_time
        print(f"Processed {weight_file} in {elapsed_time:.2f} seconds")

    # Create plot with three subplots
    epochs = list(range(len(stable_ranks)))

    fig, (ax1, ax2, ax3) = plt.subplots(1, 3, figsize=(16, 5))

    # Plot trace
    ax1.plot(epochs, traces, marker="o", linewidth=2, markersize=6)
    ax1.set_xlabel("Epoch")
    ax1.set_ylabel("Trace")
    ax1.set_title("Hessian Trace vs Epoch")
    ax1.grid(True, alpha=0.3)

    # Plot stable rank
    ax2.plot(epochs, stable_ranks, marker="o", linewidth=2, markersize=6)
    ax2.set_xlabel("Epoch")
    ax2.set_ylabel("Stable Rank")
    ax2.set_title("Hessian Stable Rank vs Epoch")
    ax2.grid(True, alpha=0.3)

    # Plot sharpness (top eigenvalue)
    ax3.plot(epochs, sharpnesses, marker="s", linewidth=2, markersize=6, color="orange")
    ax3.set_xlabel("Epoch")
    ax3.set_ylabel("Sharpness")
    ax3.set_title("Sharpness vs Epoch")
    ax3.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig("hessian_metrics.png", dpi=150, bbox_inches="tight")
    print("Plot saved to hessian_metrics.png")

###########################################################################
############ Cemetary of previous code which we might use later ###########
###########################################################################
# def analyze_weight_matrices(weight_path):
#     """Analyze eigenvalues of weight matrices directly"""

#     checkpoint = torch.load(weight_path, map_location="cpu")
#     state_dict = checkpoint["state_dict"] if "state_dict" in checkpoint else checkpoint

#     for name, param in state_dict.items():
#         if "weight" in name and len(param.shape) == 2:  # 2D weight matrices
#             # Convert to numpy for eigenvalue computation
#             weight_matrix = param.detach().numpy()

#             # Compute eigenvalues
#             eigenvalues = np.linalg.eigvals(weight_matrix)

#             print(f"\n{name}:")
#             print(f"  Shape: {weight_matrix.shape}")
#             print(f"  Top 5 eigenvalues: {eigenvalues[:5]}")
#             print(f"  Spectral radius: {np.max(np.abs(eigenvalues)):.4f}")


# def compute_diagonal_hessian(weight_path, batch_size=2000, device="cuda"):
#     """Compute diagonal of Hessian - much more memory efficient"""
#     # Load the model
#     model = CifarNet().to(torch.device(device))
#     train_loader = CifarLoader(
#         "cifar10", train=True, batch_size=batch_size, aug=dict(flip=True, translate=2)
#     )
#     checkpoint = torch.load(weight_path, map_location=torch.device(device))
#     if "state_dict" in checkpoint:
#         model.load_state_dict(checkpoint["state_dict"])
#     else:
#         model.load_state_dict(checkpoint)
#     model.eval()  # Set to evaluation mode

#     model = extend(model)  # Extend model with Backpack functionality
#     crit = extend(torch.nn.CrossEntropyLoss())
#     data_iter = iter(train_loader)

#     inputs, targets = next(data_iter)
#     outputs = model(inputs)
#     loss = crit(outputs, targets)

#     with backpack(DiagHessian()):
#         loss.backward()

#     # Access diagonal Hessian for each parameter
#     diag_hessian = []
#     for name, param in model.named_parameters():
#         diag_h = param.diag_h
#         diag_hessian.append(diag_h.flatten())

#     full_diag = torch.cat(diag_hessian)

#     # Rank estimation from diagonal (approximate)
#     threshold = 1e-6
#     rank_estimate = torch.sum(full_diag > threshold).item()

#     return full_diag, rank_estimate