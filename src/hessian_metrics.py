"""
Script to extract approximate loss Hessian metrics per layer, but also globally.
"""
import gc
import os
import time
import sys
from tqdm import tqdm
import argparse
import wandb
import numpy as np

import matplotlib.pyplot as plt
import torch
from curvlinops import HessianLinearOperator, hutchinson_squared_fro, hutchinson_trace
from scipy.sparse.linalg import eigsh
from scipy.stats import entropy

from src.utils import check_model_integrity, get_conv_layers, get_experiment_args

from .train_model import CifarLoader, CifarNet


#############################################
#             Select PyTorch device         #
#############################################
if torch.cuda.is_available():
    device = torch.device("cuda")
    torch.backends.cudnn.benchmark = True
elif torch.backends.mps.is_available():
    device = torch.device("mps")
else:
    device = torch.device("cpu")


#############################################
#           Core Hessian Metric Logic       #
#############################################
def approximate_effective_rank(eigenvalues, k=100):
    """
    Estimates effective rank using the top 'k' singular values.
    k should be large enough to capture the 'energy' of the matrix.
    """
    s_norm = eigenvalues / np.sum(eigenvalues)

    #Calculate Shannon Entropy
    #H = -sum(p * log(p))
    ent = entropy(s_norm)

    #Effective Rank is the exponential of the entropy
    eff_rank = np.exp(ent)

    return eff_rank

def analyze_layer_hessian(model, layer_name, layer_params, batch, num_eigenvalues=100, num_trace_vecs=100):
    """
    Computes metrics for a specific layer (or globally, for all layers combined)
    """
    # Ensure model and inputs are double (should be handled by caller, but safety check)
    
    inputs, targets = batch

    criterion = torch.nn.CrossEntropyLoss()

    # Get number of parameters to normalize for
    num_params = sum(p.numel() for p in layer_params)
    print(f"   Analyzing {layer_name}... ({num_params} params)")
    
    # 1. Create Linear Operator for specific params
    H_op = HessianLinearOperator(
        model_func=model,
        loss_func=criterion,
        data=[(inputs, targets)],
        params=layer_params,
        check_deterministic=False,
    )

    # 2. Compute Top Eigenvalues (Spectral Norm)
    H_scipy = H_op.to_scipy()

    eigenvalues = np.array([])
    try:
        eigenvalues, _ = eigsh(H_scipy, k=num_eigenvalues, which="LM", tol=1e-4)
        eigenvalues = np.sort(np.abs(eigenvalues))[::-1]
        lambda_max = float(eigenvalues[0])
    except Exception as e:
        print(f"      [Warning] eigsh failed for {layer_name}: {e}")
        lambda_max = np.nan

    # 3. Compute Trace
    trace_tensor = hutchinson_trace(
        H_op,
        num_matvecs=num_trace_vecs,
    )
    trace_est = trace_tensor.item()

    # 4. Compute Squared Frobenius Norm
    frob_sq_tensor = hutchinson_squared_fro(
        H_op, num_matvecs=num_trace_vecs,
    )
    frob_sq = frob_sq_tensor.item()

    # Rank Metrics
    # Metric 1: Stable Rank
    # Interpretation: "How many directions are as sharp as the max direction?"
    # Heavily penalized by spectral outliers.
    stable_rank_spectral = frob_sq / (lambda_max**2 + 1e-12)

    # Metric 2: Effective Rank (Trace-based)
    # Interpretation: "Effective dimensionality of the curvature."
    # usually yields higher, more informative numbers for "flatness".
    # effective_rank = (trace_est**2) / (frob_sq + 1e-12)
    effective_rank = approximate_effective_rank(eigenvalues)

    # Cleanup
    del H_op
    del H_scipy

    return {
        "sharpness": lambda_max, 
        "trace": trace_est, 
        "stable_rank": stable_rank_spectral,
        "effective_rank": effective_rank,
        "num_params": num_params,
        "normalized_stable_rank": stable_rank_spectral / num_params,
        "normalized_effective_rank": effective_rank / num_params,
    }


#############################################
#           Run analysis on all layers      #
#############################################
def _load_model(weight_path, experiment_args):
    """Helper to load model from checkpoint."""
    checkpoint = torch.load(weight_path, map_location=device)
    model = CifarNet(block1=experiment_args.block1,
                     block2=experiment_args.block2,
                     block3=experiment_args.block3,
                     apply_whitening=experiment_args.whitening)
    if "state_dict" in checkpoint:
        model.load_state_dict(checkpoint["state_dict"])
    else:
        model.load_state_dict(checkpoint)

    for param in model.parameters():
        param.data = param.data.contiguous()
    
    model = model.to(device)
    model.eval()
    return model


def _compute_epoch_metrics(model, fixed_batch):
    """Computes Hessian metrics for all layers in the model."""
    # Layer-wise Hessian analysis
    target_layers = get_conv_layers(model)
    epoch_data = {}
    for layer_name, layer_params in target_layers:
        metrics = analyze_layer_hessian(
            model, layer_name, layer_params, fixed_batch
        )
        print(f"     -> Sharpness: {metrics['sharpness']:.4f}, Trace: {metrics['trace']:.4f}, Stable Rank: {metrics['stable_rank']:.4f}, Effective Rank: {metrics['effective_rank']:.4f}")
        epoch_data[layer_name] = metrics

    # Global Hessian analysis
    # TODO: issue with trace and stable rank approaching inf for some reason -> perhaps mps issue
    # filter_params = [p for p in model.parameters() if len(p.shape) == 4 and p.requires_grad]
    # if filter_params:
    #     metrics = analyze_layer_hessian(
    #         model, "global", filter_params, fixed_batch
    #     )
    #     print(f"     -> Sharpness: {metrics['sharpness']:.4f}, Trace: {metrics['trace']:.4f}, Stable Rank: {metrics['stable_rank']:.4f}, Effective Rank: {metrics['effective_rank']:.4f}")
    #     epoch_data["global"] = metrics

    return epoch_data


def _update_history(history, epoch_data):
    """Updates the history dictionary with new epoch data."""
    for layer_name, metrics in epoch_data.items():
        if layer_name not in history:
            history[layer_name] = {k: [] for k in metrics}
        
        for k, v in metrics.items():
            history[layer_name][k].append(v)


def _generate_plots(history, processed_epochs, run_path):
    """Generates and saves the summary plots."""
    if not history:
        return

    print("\nGenerating Line Plots...")
    fig1, (ax1, ax2, ax3, ax4) = plt.subplots(1, 4, figsize=(24, 5))
    colors = plt.cm.viridis(np.linspace(0, 1, len(history)))
    
    # Sort layers by name to ensure consistent legend order
    sorted_layers = sorted(history.keys())
    
    for layer_name, color in zip(sorted_layers, colors):
        metrics = history[layer_name]
        ax1.plot(processed_epochs, metrics["trace"], marker='o', label=layer_name, color=color)
        ax2.plot(processed_epochs, metrics["stable_rank"], marker='^', label=layer_name, color=color)
        ax3.plot(processed_epochs, metrics["sharpness"], marker='s', label=layer_name, color=color)
        ax4.plot(processed_epochs, metrics["effective_rank"], marker='d', label=layer_name, color=color)

    for ax, title in zip([ax1, ax2, ax3, ax4], ["Trace", "Stable Rank", "Sharpness", "Effective Rank Trace"]):
        ax.set_title(title)
        ax.set_xlabel("Epoch")
        ax.grid(True, alpha=0.3)
        ax.legend()
    
    fig1.suptitle("Hessian Metrics Summary")
    plt.tight_layout()
    save_path = os.path.join(run_path, "hessian_metrics.png")
    plt.savefig(save_path, dpi=150)
    print(f"Saved Summary Plot to {save_path}.")


def run_hessian_analysis(run_path, batch_size, experiment_args):
    weights_path = os.path.join(run_path, "model_weights")

    print(f"Running Hessian metrics on experiment with arguments: {experiment_args}")

    # Load the model weights
    if not os.path.exists(weights_path):
        print(f"Error: {weights_path} does not exist.")
        return

    weight_files = sorted(
        [f for f in os.listdir(weights_path) if f.startswith("model_epoch_") and f.endswith(".pt")], 
        key=lambda x: int(x.split("_")[-1].split(".")[0])
    )
    if not weight_files:
        print("No model weights found.")
        return

    # Load Data 
    train_loader = CifarLoader(
        "cifar10", train=True, batch_size=batch_size, aug=dict(flip=False, translate=0)
    )
    fixed_inputs, fixed_targets = next(iter(train_loader))
    fixed_inputs = fixed_inputs.to(device=device).contiguous()
    fixed_targets = fixed_targets.to(device=device).contiguous()
    
    history = {}
    processed_epochs = []
    
    for weight_file in tqdm(weight_files):
        epoch_num = int(weight_file.split("_")[-1].split(".")[0])
        weight_path = os.path.join(weights_path, weight_file)
        print(f"\nProcessing Epoch {epoch_num} ({weight_file})...")
        
        model = None
        try:
            model = _load_model(weight_path, experiment_args)

            if not check_model_integrity(model):
                print(f"   [SKIP] Model weights invalid at epoch {epoch_num}")
                continue

            # Identify layers and compute metrics
            epoch_data = _compute_epoch_metrics(model, (fixed_inputs, fixed_targets))

            # Commit data
            processed_epochs.append(epoch_num)
            _update_history(history, epoch_data)

            # Log to WandB
            wandb_log_data = {"epoch": epoch_num}
            for layer_name, metrics in epoch_data.items():
                for metric_name, value in metrics.items():
                    wandb_log_data[f"{layer_name}/{metric_name}"] = value
            wandb.log(wandb_log_data)

        except Exception as e:
            print(f"   [ERROR] Failed to process epoch {epoch_num}: {e}")
            import traceback
            traceback.print_exc()
        finally:
            if model is not None:
                del model
            gc.collect()

    # Generate plots
    _generate_plots(history, processed_epochs, run_path)


if __name__ == "__main__":
    # Parse the arguments for the hessian experiment.
    parser = argparse.ArgumentParser(description="Hessian Metrics Runner")
    parser.add_argument("--run_path", type=str, help="Path to run directory, which includes the model weights to analyze.")
    parser.add_argument("--batch_size", type=int, default=512, help="Number of items to use to compute the Hessian. Compute time scales linearly.")
    args = parser.parse_args()

    # Load experiment args
    experiment_args = get_experiment_args(args.run_path)

    # Setup WandB for tracking experiments ##
    wandb.init(
        entity="adaptive-muon",
        project="hessian-metrics", 
        config=vars(args),
        name=experiment_args.run_name,
        mode="disabled" if not experiment_args.wandb else "online"
    )

    run_hessian_analysis(args.run_path, args.batch_size, experiment_args)
    wandb.finish()

