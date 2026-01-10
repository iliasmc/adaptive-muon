import gc
import os
import time

import matplotlib.pyplot as plt
import numpy as np
import torch
from curvlinops import HessianLinearOperator, hutchinson_squared_fro, hutchinson_trace
from scipy.sparse.linalg import eigsh

# Assuming local imports
from .train_model import CifarLoader, CifarNet


# ==========================================
# HELPER: SAFETY CHECKS
# ==========================================
def check_model_integrity(model):
    """Checks if model weights have exploded to NaN/Inf."""
    for name, param in model.named_parameters():
        if torch.isnan(param).any() or torch.isinf(param).any():
            print(f"   [WARNING] Model weights corrupted (NaN/Inf) at layer: {name}")
            return False
    return True


def clear_memory(device):
    gc.collect()
    if device == "cuda":
        torch.cuda.empty_cache()
    elif device == "mps":
        try:
            torch.mps.empty_cache()
        except:
            pass


# ==========================================
# CORE ANALYSIS
# ==========================================
def analyze_hessian_metrics(model, batch, num_eigenvalues=1, num_trace_vecs=100):
    """
    Computes metrics using Float64 on CPU for numerical stability.
    """
    # 1. Setup: Force CPU and Double Precision (Float64)
    #    Hessian analysis is extremely sensitive to precision.
    device = "cpu"
    model = model.to(device).double()  # Critical for stability

    # Unpack and cast batch to double
    inputs, targets = batch
    inputs = inputs.to(device).double()
    targets = targets.to(device)

    criterion = torch.nn.CrossEntropyLoss()

    # 2. Check for NaN loss before starting expensive HVP
    #    If the loss is already NaN, the Hessian is undefined.
    with torch.no_grad():
        initial_loss = criterion(model(inputs), targets)
        if torch.isnan(initial_loss) or torch.isinf(initial_loss):
            raise ValueError("Loss is NaN/Inf on this batch. Model has likely diverged.")

    # 3. Create Linear Operator
    H_op = HessianLinearOperator(
        model_func=model,
        loss_func=criterion,
        data=[(inputs, targets)],
        params=[p for p in model.parameters() if len(p.shape) == 4 and p.requires_grad],
        check_deterministic=False,
    )

    # 4. Compute Top Eigenvalues (Spectral Norm)
    print(f"   -> Computing Top {num_eigenvalues} Eigenvalues (Lanczos)...")
    H_scipy = H_op.to_scipy()

    # Use 'LM' (Largest Magnitude) to catch sharp negative curvature
    eigenvalues, _ = eigsh(H_scipy, k=num_eigenvalues, which="LM", tol=1e-4)

    # Sort and take max absolute value
    eigenvalues = np.sort(np.abs(eigenvalues))[::-1]
    lambda_max = float(eigenvalues[0])

    # 5. Compute Trace
    print("   -> Computing Trace (Hutchinson)...")
    trace_tensor = hutchinson_trace(
        H_op,
        num_matvecs=num_trace_vecs,
        distribution="rademacher",  # Rademacher (+1/-1) is strictly better than Gaussian for trace
    )
    trace_est = trace_tensor.item()  # No detach/cpu needed as we are already on CPU

    # 6. Compute Squared Frobenius Norm
    print("   -> Computing Frobenius Norm...")
    frob_sq_tensor = hutchinson_squared_fro(
        H_op, num_matvecs=num_trace_vecs, distribution="rademacher"
    )
    frob_sq = frob_sq_tensor.item()

    # 7. Cleanup
    del H_op
    del H_scipy

    # 8. Derived Metrics
    spectral_sq = lambda_max**2
    stable_rank = frob_sq / (spectral_sq + 1e-12)  # epsilon to prevent div/0

    return {"sharpness": lambda_max, "trace": trace_est, "stable_rank": stable_rank}


# ==========================================
# MAIN
# ==========================================
if __name__ == "__main__":
    weights_path = "./model_weights/"

    # NOTE: We use CPU for analysis to ensure Float64 support,
    # but we can check what the original training device was.
    print(f"Analysis forced to CPU (Float64) for numerical stability.")

    weight_files = sorted(
        os.listdir(weights_path), key=lambda x: int(x.split("_")[-1].split(".")[0])
    )

    history = {"epochs": [], "trace": [], "stable_rank": [], "sharpness": []}

    # Load Data (CPU)
    # We use a smaller batch size for analysis to keep it fast on CPU
    train_loader = CifarLoader(
        "cifar10", train=True, batch_size=128, aug=dict(flip=True, translate=2)
    )
    # Grab one fixed batch
    fixed_inputs, fixed_targets = next(iter(train_loader))

    for weight_file in weight_files:
        epoch_num = int(weight_file.split("_")[-1].split(".")[0])
        history["epochs"].append(epoch_num)

        weight_path = os.path.join(weights_path, weight_file)
        print(f"\nProcessing Epoch {epoch_num} ({weight_file})...")
        start_time = time.time()

        try:
            # Load Model
            # Load onto CPU immediately
            checkpoint = torch.load(weight_path, map_location="cpu")
            model = CifarNet()

            if "state_dict" in checkpoint:
                model.load_state_dict(checkpoint["state_dict"])
            else:
                model.load_state_dict(checkpoint)
            model.eval()

            # Integrity Check
            if not check_model_integrity(model):
                print(f"   [SKIP] Model weights invalid at epoch {epoch_num}")
                # Append NaNs so plots don't break alignment
                history["trace"].append(np.nan)
                history["stable_rank"].append(np.nan)
                history["sharpness"].append(np.nan)
                continue

            # Analyze
            metrics = analyze_hessian_metrics(
                model, (fixed_inputs, fixed_targets), num_eigenvalues=1, num_trace_vecs=100
            )

            history["trace"].append(metrics["trace"])
            history["stable_rank"].append(metrics["stable_rank"])
            history["sharpness"].append(metrics["sharpness"])

            print(
                f"   [Done] Sharpness: {metrics['sharpness']:.4f} | Trace: {metrics['trace']:.4f} | Stable Rank: {metrics['stable_rank']:.4f}"
            )

        except ValueError as ve:
            print(f"   [SKIP] Calculation Error: {ve}")
            history["trace"].append(np.nan)
            history["stable_rank"].append(np.nan)
            history["sharpness"].append(np.nan)

        except Exception as e:
            print(f"   [ERROR] Unexpected error: {e}")
            raise e

        # Cleanup
        del model
        gc.collect()  # Python GC
        print(f"   Time taken: {time.time() - start_time:.2f}s")

    # --- PLOTTING ---

    print("\nGenerating Plots...")
    fig, (ax1, ax2, ax3) = plt.subplots(1, 3, figsize=(18, 5))

    epochs = history["epochs"]

    # Filter out NaNs for plotting
    valid_mask = ~np.isnan(history["trace"])
    valid_epochs = np.array(epochs)[valid_mask]

    ax1.plot(valid_epochs, np.array(history["trace"])[valid_mask], marker="o", color="teal")
    ax1.set_title("Hessian Trace")
    ax1.set_xlabel("Epoch")
    ax1.grid(True, alpha=0.3)

    ax2.plot(valid_epochs, np.array(history["stable_rank"])[valid_mask], marker="^", color="purple")
    ax2.set_title("Stable Rank")
    ax2.set_xlabel("Epoch")
    ax2.grid(True, alpha=0.3)

    ax3.plot(valid_epochs, np.array(history["sharpness"])[valid_mask], marker="s", color="orange")
    ax3.set_title("Sharpness (Max Eigenvalue)")
    ax3.set_xlabel("Epoch")
    ax3.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig("hessian_metrics_cpu.png", dpi=150)
    print("Saved to hessian_metrics_cpu.png")
