"""
The main file for running experiments. Adjusted from https://github.com/KellerJordan/cifar10-airbench.
"""

#############################################
#                  Setup                    #
#############################################

import argparse
import json
import os
import time
from datetime import datetime
from math import ceil, cos, pi

import matplotlib.pyplot as plt
import torch
import torch.nn.functional as F
from tqdm import tqdm

import wandb
import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from src.data_loader import CifarLoader
from src.model import CifarNet
from src.utils import str2bool

from src.muon_original import Muon

#############################################
#           Parse experiment params         #
#############################################

parser = argparse.ArgumentParser(description="Adaptive Muon CNN Training Runner")
# High level args
parser.add_argument("--timestamp", type=str, default=None, help="Timestamp for the run (overrides auto-generation)")
parser.add_argument("--wandb", type=str2bool, default=True, help="Whether to store results in WANDB")

# CNN architecture params
parser.add_argument("--block1", type=int, default=24, help="Width of CNN block 1")
parser.add_argument("--block2", type=int, default=48, help="Width of CNN block 2")
parser.add_argument("--block3", type=int, default=48, help="Width of CNN block 3")
parser.add_argument("--whitening", type=str2bool, default=True, help="Whether to apply whitening")

# Dataset/batch size
parser.add_argument("--batch_size", type=int, default=512, help="Batch size for training")
parser.add_argument("--val_split", type=float, default=0.1, help="Validation split (0.0 to 1.0)")

# Run config
parser.add_argument("--num_epochs", type=int, default=100, help="Number of training epochs")

# Optimizer hyperparameters
parser.add_argument("--bias_lr", type=float, default=0.01, help="Initial learning rate for the bias parameters with SGD")
parser.add_argument("--head_lr", type=float, default=0.6, help="Initial learning rate for the head parameters with SGD")
parser.add_argument("--conv_optimizer", type=str, default="muon", choices=["muon", "sgd", "adaptive"], help="The optimizer to use for the convolutional layers")
parser.add_argument("--conv_lr", type=float, default=0.01, help="Initial learning rate for the optimizer chosen for the convolutional layers")
parser.add_argument("--conv_momentum", type=float, default=0.85, help="Momentum for the convolutional layers")
parser.add_argument("--scheduler", type=str, choices=["cosine", "linear"], default="cosine", help="LR scheduler type")
parser.add_argument("--sgd_momentum", type=float, default=0.85, help="Momentum for SGD (optimizer1)")

# Directory under which to store artifacts
parser.add_argument("--results_root", type=str, default=".", help="Root directory to store results in")
parser.add_argument("--run_name", type=str, default="", help="The name to store the run by in WANDB. If no value passed, a collection of all previous args will be used.")

if __name__ == "__main__":
    args = parser.parse_args()
else:
    args = parser.parse_args([])

# Set global timestamp
if args.timestamp:
    timestamp = args.timestamp
else:
    timestamp = datetime.now().strftime("%Y%m%d-%H%M%S")

# Have all args in a single string for easy comparison through WANDB
run_name = (
    f"blk-{args.block1}-{args.block2}-{args.block3},"
    f"bs-{args.batch_size},"
    f"val-{args.val_split},"
    f"whtn-{int(args.whitening)},"
    f"ep-{args.num_epochs},"
    f"blr-{args.bias_lr},"
    f"hlr-{args.head_lr},"
    f"opt-{args.conv_optimizer},"
    f"clr-{args.conv_lr},"
    f"cmom-{args.conv_momentum},"
    f"sch-{args.scheduler},"
    f"smom-{args.sgd_momentum}"
)

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


############################################
#               Evaluation                 #
############################################


def infer(model, loader, tta_level=0):
    # Test-time augmentation strategy (for tta_level=2):
    # 1. Flip/mirror the image left-to-right (50% of the time).
    # 2. Translate the image by one pixel either up-and-left or down-and-right (50% of the time,
    #    i.e. both happen 25% of the time).
    #
    # This creates 6 views per image (left/right times the two translations and no-translation),
    # which we evaluate and then weight according to the given probabilities.

    def infer_basic(inputs, net):
        return net(inputs).clone()

    def infer_mirror(inputs, net):
        return 0.5 * net(inputs) + 0.5 * net(inputs.flip(-1))

    def infer_mirror_translate(inputs, net):
        logits = infer_mirror(inputs, net)
        pad = 1
        padded_inputs = F.pad(inputs, (pad,) * 4, "reflect")
        inputs_translate_list = [
            padded_inputs[:, :, 0:32, 0:32],
            padded_inputs[:, :, 2:34, 2:34],
        ]
        logits_translate_list = [
            infer_mirror(inputs_translate, net) for inputs_translate in inputs_translate_list
        ]
        logits_translate = torch.stack(logits_translate_list).mean(0)
        return 0.5 * logits + 0.5 * logits_translate

    model.eval()
    test_images = loader.normalize(loader.images)
    infer_fn = [infer_basic, infer_mirror, infer_mirror_translate][tta_level]
    with torch.no_grad():
        return torch.cat([infer_fn(inputs, model) for inputs in test_images.split(2000)])


def evaluate(model, loader, tta_level=0):
    logits = infer(model, loader, tta_level)
    return (logits.argmax(1) == loader.labels).float().mean().item()


def main(run, model, config, device, is_sweep=False):
    batch_size = config.batch_size
    bias_lr = config.bias_lr #0.053
    head_lr = config.head_lr #0.67
    wd = 2e-6 * batch_size

    # Use a specific split for validation (e.g., 20% of the training data)
    val_split = config.val_split

    test_loader = CifarLoader("cifar10", device=device, train=False, batch_size=2000)
    val_loader = CifarLoader(
        "cifar10", device=device, train=True, batch_size=2000, val_split=val_split, part="val"
    )
    train_loader = CifarLoader(
        "cifar10",
        device=device,
        train=True,
        batch_size=batch_size,
        aug=dict(flip=True, translate=4),
        val_split=val_split,
        part="train",
    )
    num_epochs = config.num_epochs

    if run == "warmup":
        # The only purpose of the first run is to warmup the compiled model, so we can use dummy data
        train_loader.labels = torch.randint(
            0, 10, size=(len(train_loader.labels),), device=train_loader.labels.device
        )
    total_train_steps = ceil(num_epochs * len(train_loader))
    whiten_bias_train_steps = ceil(3 * len(train_loader))

    # Start total training timer
    total_start_time = time.time()

    # Create optimizers and learning rate schedulers
    filter_params = [p for p in model.parameters() if len(p.shape) == 4 and p.requires_grad]
    norm_biases = [p for n, p in model.named_parameters() if "norm" in n and p.requires_grad]
    param_configs = [
        dict(params=norm_biases, lr=bias_lr, weight_decay=wd / bias_lr),
        dict(params=[model.head.weight], lr=head_lr, weight_decay=wd / head_lr),
    ]
    if config.whitening:
        param_configs.append(dict(params=[model.whiten.bias], lr=bias_lr, weight_decay=wd / bias_lr))
    
    # Use fused optimizer only on CUDA
    use_fused = torch.cuda.is_available()
    optimizer1 = torch.optim.SGD(param_configs, momentum=config.sgd_momentum, nesterov=True, fused=use_fused)
    if config.conv_optimizer == "muon":
        optimizer2 = Muon(filter_params, lr=config.conv_lr, momentum=config.conv_momentum, nesterov=True)
    elif config.conv_optimizer == "sgd":
        optimizer2 = torch.optim.SGD(filter_params, lr=config.conv_lr, momentum=config.conv_momentum, nesterov=True, weight_decay=wd/config.conv_lr, fused=use_fused)
    else:
        raise NotImplementedError("Need to put our adaptive muon optimizer here")
    optimizers = [optimizer1, optimizer2]
    for opt in optimizers:
        for group in opt.param_groups:
            group["initial_lr"] = group["lr"]

    model.reset()
    step = 0
    train_accs = []
    val_accs = []
    test_accs = []

    # Initialize the whitening layer using training images
    if config.whitening:
        train_images = train_loader.normalize(train_loader.images[:5000])
        model.init_whiten(train_images)

    for epoch in tqdm(range(ceil(total_train_steps / len(train_loader)))):
        print(f"At epoch: {epoch}")

        # Start epoch timer
        epoch_start_time = time.time()

        ####################
        #     Training     #
        ####################

        model.train()
        epoch_train_loss = 0.0
        for inputs, labels in train_loader:
            outputs = model(inputs, whiten_bias_grad=(step < whiten_bias_train_steps))
            ce_loss = F.cross_entropy(outputs, labels, label_smoothing=0.2, reduction="sum")
            ce_loss.backward()

            epoch_train_loss += ce_loss.item()

            # The LR for the whitening layer params
            if config.whitening:
                whitening_group = optimizer1.param_groups[-1]
                whitening_group["lr"] = whitening_group["initial_lr"] * (1 - step / whiten_bias_train_steps)
            # The LR for the rest of the params
            for group in optimizer1.param_groups[:-1] + optimizer2.param_groups:
                if config.scheduler == "cosine":
                    decay = 0.5 * (1 + cos(pi * (step / total_train_steps)))
                else: # linear
                    decay = (1 - step / total_train_steps)
                group["lr"] = group["initial_lr"] * decay
            for opt in optimizers:
                opt.step()
            model.zero_grad(set_to_none=True)
            step += 1
            if step >= total_train_steps:
                break

        # End epoch timer
        epoch_end_time = time.time()
        epoch_duration = epoch_end_time - epoch_start_time
        print(f"Epoch {epoch} completed in {epoch_duration:.4f} seconds")

        # Save model weights after each epoch into the model_weights/ directory
        if not is_sweep:
            try:
                os.makedirs(f"{config.results_root}/{timestamp}/model_weights", exist_ok=True)
                weight_path = os.path.join(config.results_root, timestamp, "model_weights", f"model_epoch_{epoch}.pt")
                torch.save({"epoch": epoch, "state_dict": model.state_dict()}, weight_path)
                print(f"Saved model weights to {os.path.abspath(weight_path)}")
            except Exception as e:
                print(f"Warning: failed to save model weights for epoch {epoch}: {e}")

        ####################
        #    Evaluation    #
        ####################
        model.eval()

        # Calculate Test Loss
        epoch_test_loss = 0.0
        with torch.no_grad():
            for inputs, labels in test_loader:
                outputs = model(inputs)
                loss = F.cross_entropy(outputs, labels, reduction="sum")
                epoch_test_loss += loss.item()
        
        # Calculate Validation Loss
        epoch_val_loss = 0.0
        if len(val_loader.images) > 0:
            with torch.no_grad():
                for inputs, labels in val_loader:
                    outputs = model(inputs)
                    loss = F.cross_entropy(outputs, labels, reduction="sum")
                    epoch_val_loss += loss.item()
        else:
            epoch_val_loss = 0.0

        # Get all accuracies
        train_acc = evaluate(model, train_loader, tta_level=0)
        train_accs.append(train_acc)
        if len(val_loader.images) > 0:
            val_acc = evaluate(model, val_loader, tta_level=0)
        else:
            val_acc = 0
        val_accs.append(val_acc)
        test_acc = evaluate(model, test_loader, tta_level=0)
        test_accs.append(test_acc)
        print(
            f"Epoch {epoch} - Train Acc: {train_acc:.4f}, Val Acc: {val_acc:.4f}, Test Acc: {test_acc:.4f}\n"
        )
        # Log to WANDB
        avg_train_loss = epoch_train_loss / len(train_loader.images)
        avg_test_loss = epoch_test_loss / len(test_loader.images)
        avg_val_loss = epoch_val_loss / len(val_loader.images) if len(val_loader.images) > 0 else 0
        wandb.log({
            "epoch": epoch,
            "train/loss": avg_train_loss,
            "train/acc": train_acc,
            "test/loss": avg_test_loss,
            "test/acc": test_acc,
            "val/loss": avg_val_loss,
            "val/acc": val_acc,
            "conv_lr": optimizer2.param_groups[0]["lr"], # Tracks the main LR
            "epoch_duration": epoch_duration
        })

    if not is_sweep:
        plt.figure(figsize=(10, 6))
        plt.plot(range(len(train_accs)), train_accs, marker="s", linestyle="--", color="orange", label="Training Accuracy")
        plt.plot(range(len(test_accs)), test_accs, marker="o", linestyle="-", color="blue", label="Test Accuracy")
        plt.xlabel("Epoch")
        plt.ylabel("Accuracy")
        plt.title("Training vs. Test Accuracy over Epochs")
        plt.legend()
        plt.grid(True)
        plt.savefig(f"{config.results_root}/{timestamp}/accuracy_comparison.png", dpi=150, bbox_inches="tight")
        plt.close()

    # Calculate and print total training time
    total_end_time = time.time()
    total_training_time = total_end_time - total_start_time
    print(f"Total training time: {total_training_time:.4f} seconds")



if __name__ == "__main__":
    # Find whether we are in a sweep
    is_sweep = "WANDB_SWEEP_ID" in os.environ
    print(f"Is sweep: {is_sweep}")

    config_dict = vars(args)
    config_dict['timestamp'] = timestamp
    config_dict['run_name'] = args.run_name if args.run_name != "" else run_name

    # Store config
    if not is_sweep:
        print("\n--- Running Experiment with Configuration ---")
        print(json.dumps(config_dict, indent=4))
        print("--------------------------------------------\n")
        output_dir = os.path.join(args.results_root, timestamp)
        os.makedirs(output_dir, exist_ok=True)
        config_path = os.path.join(output_dir, "experimental_config.json")
        with open(config_path, "w") as f:
            json.dump(config_dict, f, indent=4)

    # Setup WandB for tracking experiments
    wandb.init(
        entity="adaptive-muon",
        project="cnn-training", 
        # config=config_dict,
        name=config_dict['run_name'],
        mode="disabled" if not config_dict["wandb"] else "online"
    )
    wandb.config.update(config_dict, allow_val_change=True)
    config = wandb.config
    # Training code
    model = CifarNet(block1=config.block1,
                     block2=config.block2,
                     block3=config.block3,
                     apply_whitening=config.whitening).to(device)
    print(model)
    total_params = sum(p.numel() for p in model.parameters())
    print(f"Total parameters: {total_params}")

    # Only compile on CUDA systems (fix on cluster)
    if torch.cuda.is_available():
        major, minor = torch.cuda.get_device_capability()
        if major >= 7:
            model.compile(mode="max-autotune")
        else:
            print(f"Warning: GPU capability {major}.{minor} < 7.0. Skipping torch.compile (Triton not supported).")
    else:
        print(
            "Warning: Running without torch.compile (CUDA not available). Performance will be significantly slower."
        )

    # main("warmup", model, config, device, is_sweep)
    main(run=0, 
         model=model, 
         config=config, 
         device=device, 
         is_sweep=is_sweep)
    wandb.finish()
