"""
The main file for running experiments. Adjusted from https://github.com/KellerJordan/cifar10-airbench.
"""

#############################################
#                  Setup                    #
#############################################

import os
import argparse 
import json
import wandb

import time
from math import ceil, cos, pi
from tqdm import tqdm

import matplotlib.pyplot as plt
import torch
import torch.nn.functional as F
import torchvision
import torchvision.transforms as T
from torch import nn
from datetime import datetime

from .muon_original import Muon

#############################################
#           Parse experiment params         #
#############################################
parser = argparse.ArgumentParser(description="Adaptive Muon CNN Training Runner")
# High level args
parser.add_argument("--timestamp", type=str, default=None, help="Timestamp for the run (overrides auto-generation)")
parser.add_argument("--wandb", action=argparse.BooleanOptionalAction, default=True, help="Whether to store results in WANDB")

# CNN architecture params
parser.add_argument("--block1", type=int, default=24, help="Width of CNN block 1")
parser.add_argument("--block2", type=int, default=48, help="Width of CNN block 2")
parser.add_argument("--block3", type=int, default=48, help="Width of CNN block 3")
parser.add_argument("--whitening", action=argparse.BooleanOptionalAction, default=True, help="Whether to apply whitening")

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


#############################################
#                DataLoader                 #
#############################################

CIFAR_MEAN = torch.tensor((0.4914, 0.4822, 0.4465)).to(device)
CIFAR_STD = torch.tensor((0.2470, 0.2435, 0.2616)).to(device)


def batch_flip_lr(inputs):
    flip_mask = (torch.rand(len(inputs), device=inputs.device) < 0.5).view(-1, 1, 1, 1)
    return torch.where(flip_mask, inputs.flip(-1), inputs)


def batch_crop(images, crop_size):
    r = (images.size(-1) - crop_size) // 2
    shifts = torch.randint(-r, r + 1, size=(len(images), 2), device=images.device)
    images_out = torch.empty(
        (len(images), 3, crop_size, crop_size), device=images.device, dtype=images.dtype
    )
    # The two cropping methods in this if-else produce equivalent results, but the second is faster for r > 2.
    if r <= 2:
        for sy in range(-r, r + 1):
            for sx in range(-r, r + 1):
                mask = (shifts[:, 0] == sy) & (shifts[:, 1] == sx)
                images_out[mask] = images[
                    mask, :, r + sy : r + sy + crop_size, r + sx : r + sx + crop_size
                ]
    else:
        images_tmp = torch.empty(
            (len(images), 3, crop_size, crop_size + 2 * r), device=images.device, dtype=images.dtype
        )
        for s in range(-r, r + 1):
            mask = shifts[:, 0] == s
            images_tmp[mask] = images[mask, :, r + s : r + s + crop_size, :]
        for s in range(-r, r + 1):
            mask = shifts[:, 1] == s
            images_out[mask] = images_tmp[mask, :, :, r + s : r + s + crop_size]
    return images_out


class CifarLoader:
    def __init__(self, path, train=True, batch_size=500, aug=None, val_split=0.0, part="train"):
        data_path = os.path.join(path, "train.pt" if train else "test.pt")
        if not os.path.exists(data_path):
            dset = torchvision.datasets.CIFAR10(path, download=True, train=train)
            images = torch.tensor(dset.data)
            labels = torch.tensor(dset.targets)

            # Reshuffle the loaded data
            indices = torch.randperm(images.size(0))
            images = images[indices]
            labels = labels[indices]

            torch.save({"images": images, "labels": labels, "classes": dset.classes}, data_path)

        data = torch.load(data_path, map_location=device)
        self.images, self.labels, self.classes = data["images"], data["labels"], data["classes"]

        if train:
            assert 0.0 <= val_split < 1.0, "val_split must be between 0 and 1"
            num_val = int(len(self.images) * val_split)
            num_train = len(self.images) - num_val
            if part == "train":
                self.images = self.images[:num_train]
                self.labels = self.labels[:num_train]
            elif part == "val":
                self.images = self.images[num_train:]
                self.labels = self.labels[num_train:]
            else:
                raise ValueError(f"Invalid part '{part}'")

        # It's faster to load+process uint8 data than to load preprocessed fp16 data
        self.images = (
            (self.images.float() / 255).permute(0, 3, 1, 2).to(memory_format=torch.channels_last)
        )

        self.normalize = T.Normalize(CIFAR_MEAN, CIFAR_STD)
        self.proc_images = {}  # Saved results of image processing to be done on the first epoch
        self.epoch = 0

        self.aug = aug or {}
        for k in self.aug.keys():
            assert k in ["flip", "translate"], "Unrecognized key: %s" % k

        self.batch_size = batch_size
        self.drop_last = train
        self.shuffle = train

    def __len__(self):
        return (
            len(self.images) // self.batch_size
            if self.drop_last
            else ceil(len(self.images) / self.batch_size)
        )

    def __iter__(self):
        if self.epoch == 0:
            images = self.proc_images["norm"] = self.normalize(self.images)
            # Pre-flip images in order to do every-other epoch flipping scheme
            if self.aug.get("flip", False):
                images = self.proc_images["flip"] = batch_flip_lr(images)
            # Pre-pad images to save time when doing random translation
            pad = self.aug.get("translate", 0)
            if pad > 0:
                self.proc_images["pad"] = F.pad(images, (pad,) * 4, "reflect")

        if self.aug.get("translate", 0) > 0:
            images = batch_crop(self.proc_images["pad"], self.images.shape[-2])
        elif self.aug.get("flip", False):
            images = self.proc_images["flip"]
        else:
            images = self.proc_images["norm"]
        # Flip all images together every other epoch. This increases diversity relative to random flipping
        if self.aug.get("flip", False):
            if self.epoch % 2 == 1:
                images = images.flip(-1)

        self.epoch += 1

        indices = (torch.randperm if self.shuffle else torch.arange)(
            len(images), device=images.device
        )
        for i in range(len(self)):
            idxs = indices[i * self.batch_size : (i + 1) * self.batch_size]
            yield (images[idxs], self.labels[idxs])


#############################################
#            Network Definition             #
#############################################


# note the use of low BatchNorm stats momentum
class BatchNorm(nn.BatchNorm2d):
    def __init__(self, num_features, momentum=0.6, eps=1e-12):
        super().__init__(num_features, eps=eps, momentum=1 - momentum)
        self.weight.requires_grad = False
        # Note that PyTorch already initializes the weights to one and bias to zero


class Conv(nn.Conv2d):
    def __init__(self, in_channels, out_channels):
        super().__init__(in_channels, out_channels, kernel_size=3, padding="same", bias=False)

    def reset_parameters(self):
        super().reset_parameters()
        w = self.weight.data
        torch.nn.init.dirac_(w[: w.size(1)])


class ConvGroup(nn.Module):
    def __init__(self, channels_in, channels_out):
        super().__init__()
        self.conv1 = Conv(channels_in, channels_out)
        self.pool = nn.MaxPool2d(2, return_indices=True)
        self.norm1 = BatchNorm(channels_out)
        self.conv2 = Conv(channels_out, channels_out)
        self.norm2 = BatchNorm(channels_out)
        self.activ = nn.GELU()

    def forward(self, x):
        x = self.conv1(x)
        x = self.pool(x)
        if isinstance(x, tuple):
            x = x[0]
        x = self.norm1(x)
        x = self.activ(x)
        x = self.conv2(x)
        x = self.norm2(x)
        x = self.activ(x)
        return x


class CifarNet(nn.Module):
    def __init__(self, block1:int, block2:int, block3:int, apply_whitening:bool):
        super().__init__()
        self.apply_whitening = apply_whitening
        widths = dict(block1=block1, block2=block2, block3=block3)
        if self.apply_whitening:
            whiten_kernel_size = 2
            whiten_width = 2 * 3 * whiten_kernel_size**2
            self.whiten = nn.Conv2d(3, whiten_width, whiten_kernel_size, padding=0, bias=True)
            self.whiten.weight.requires_grad = False
            self.layers = nn.Sequential(
                nn.GELU(),
                ConvGroup(whiten_width, widths["block1"]),
                ConvGroup(widths["block1"], widths["block2"]),
                ConvGroup(widths["block2"], widths["block3"]),
                nn.MaxPool2d(3, return_indices=True),
            )
        else:
            self.layers = nn.Sequential(
                nn.GELU(),
                ConvGroup(3, widths["block1"]),
                ConvGroup(widths["block1"], widths["block2"]),
                ConvGroup(widths["block2"], widths["block3"]),
                nn.MaxPool2d(3, return_indices=True),
            )
        self.head = nn.Linear(widths["block3"], 10, bias=False)
        for mod in self.modules():
            if isinstance(mod, BatchNorm):
                mod.float()
            else:
                mod.float()

    def reset(self):
        for m in self.modules():
            if type(m) in (nn.Conv2d, Conv, BatchNorm, nn.Linear):
                m.reset_parameters()
        w = self.head.weight.data
        w *= 1 / w.std()

    def init_whiten(self, train_images, eps=5e-4):
        c, (h, w) = train_images.shape[1], self.whiten.weight.shape[2:]
        patches = (
            train_images.unfold(2, h, 1)
            .unfold(3, w, 1)
            .transpose(1, 3)
            .reshape(-1, c, h, w)
            .float()
        )
        patches_flat = patches.view(len(patches), -1)
        est_patch_covariance = (patches_flat.T @ patches_flat) / len(patches_flat)
        
        # FIX: Move to CPU for eigenvalue decomposition (not supported on MPS)
        # We store the original device to move results back later
        orig_device = est_patch_covariance.device
        cov_cpu = est_patch_covariance.cpu()
        
        eigenvalues, eigenvectors = torch.linalg.eigh(cov_cpu, UPLO="U")
        
        # Move results back to the original device (MPS or CUDA)
        eigenvalues = eigenvalues.to(orig_device)
        eigenvectors = eigenvectors.to(orig_device)
        
        eigenvectors_scaled = eigenvectors.T.reshape(-1, c, h, w) / torch.sqrt(
            eigenvalues.view(-1, 1, 1, 1) + eps
        )
        self.whiten.weight.data[:] = torch.cat((eigenvectors_scaled, -eigenvectors_scaled))
    
    
    def forward(self, x, whiten_bias_grad=True):
        b = self.whiten.bias
        if self.apply_whitening:
            x = F.conv2d(x, self.whiten.weight, b if whiten_bias_grad else b.detach())
        x = self.layers(x)
        # Unpack tuple if return_indices=True was used in MaxPool2d
        if isinstance(x, tuple):
            x = x[0]
        x = x.reshape(len(x), -1)
        return self.head(x) / x.size(-1)


############################################
#                 Logging                  #
############################################


def print_columns(columns_list, is_head=False, is_final_entry=False):
    print_string = ""
    for col in columns_list:
        print_string += "|  %s  " % col
    print_string += "|"
    if is_head:
        print("-" * len(print_string))
    print(print_string)
    if is_head or is_final_entry:
        print("-" * len(print_string))


logging_columns_list = ["run   ", "epoch", "train_acc", "val_acc", "test_acc", "time_seconds"]


def print_training_details(variables, is_final_entry):
    formatted = []
    for col in logging_columns_list:
        var = variables.get(col.strip(), None)
        if type(var) in (int, str):
            res = str(var)
        elif type(var) is float:
            res = "{:0.4f}".format(var)
        else:
            assert var is None
            res = ""
        formatted.append(res.rjust(len(col)))
    print_columns(formatted, is_final_entry=is_final_entry)


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


############################################
#                Training                  #
############################################


def main(run, model):
    batch_size = args.batch_size
    bias_lr = args.bias_lr #0.053
    head_lr = args.head_lr #0.67
    wd = 2e-6 * batch_size

    # Use a specific split for validation (e.g., 20% of the training data)
    val_split = args.val_split

    test_loader = CifarLoader("cifar10", train=False, batch_size=2000)
    val_loader = CifarLoader(
        "cifar10", train=True, batch_size=2000, val_split=val_split, part="val"
    )
    train_loader = CifarLoader(
        "cifar10",
        train=True,
        batch_size=batch_size,
        aug=dict(flip=True, translate=4),
        val_split=val_split,
        part="train",
    )
    num_epochs = args.num_epochs

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
        dict(params=[model.whiten.bias], lr=bias_lr, weight_decay=wd / bias_lr),
        dict(params=norm_biases, lr=bias_lr, weight_decay=wd / bias_lr),
        dict(params=[model.head.weight], lr=head_lr, weight_decay=wd / head_lr),
    ]
    # Use fused optimizer only on CUDA
    use_fused = torch.cuda.is_available()
    optimizer1 = torch.optim.SGD(param_configs, momentum=args.sgd_momentum, nesterov=True, fused=use_fused)
    if args.conv_optimizer == "muon":
        optimizer2 = Muon(filter_params, lr=args.conv_lr, momentum=args.conv_momentum, nesterov=True)
    elif args.conv_optimizer == "sgd":
        optimizer2 = torch.optim.SGD(filter_params, lr=args.conv_lr, momentum=args.conv_momentum, nesterov=True, weight_decay=wd/args.conv_lr, fused=use_fused)
    else:
        raise NotImplementedError("Need to put our adaptive muon optimizer here")
    optimizers = [optimizer1, optimizer2]
    for opt in optimizers:
        for group in opt.param_groups:
            group["initial_lr"] = group["lr"]

    # For accurately timing GPU code
    # starter = torch.cuda.Event(enable_timing=True)
    # ender = torch.cuda.Event(enable_timing=True)
    # time_seconds = 0.0
    # def start_timer():
    #     starter.record()
    # def stop_timer():
    #     ender.record()
    #     torch.cuda.synchronize()
    #     nonlocal time_seconds
    #     time_seconds += 1e-3 * starter.elapsed_time(ender)

    model.reset()
    step = 0
    train_accs = []
    val_accs = []
    test_accs = []

    # Initialize the whitening layer using training images
    # start_timer()
    if args.whitening:
        train_images = train_loader.normalize(train_loader.images[:5000])
        model.init_whiten(train_images)
    # stop_timer()

    for epoch in tqdm(range(ceil(total_train_steps / len(train_loader)))):
        print(f"At epoch: {epoch}")

        # Start epoch timer
        epoch_start_time = time.time()

        ####################
        #     Training     #
        ####################

        # start_timer()
        model.train()
        epoch_train_loss = 0.0
        for inputs, labels in train_loader:
            outputs = model(inputs, whiten_bias_grad=(step < whiten_bias_train_steps))
            ce_loss = F.cross_entropy(outputs, labels, label_smoothing=0.2, reduction="sum")
            ce_loss.backward()

            epoch_train_loss += ce_loss.item()

            # The LR for the whitening layer params
            for group in optimizer1.param_groups[:1]:
                group["lr"] = group["initial_lr"] * (1 - step / whiten_bias_train_steps)
            # The LR for the rest of the params
            for group in optimizer1.param_groups[1:] + optimizer2.param_groups:
                if args.scheduler == "cosine":
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

        # Store model weights
        # Save model weights after each epoch into the model_weights/ directory
        try:
            os.makedirs(f"{args.results_root}/{timestamp}/model_weights", exist_ok=True)
            weight_path = os.path.join(args.results_root, timestamp, "model_weights", f"model_epoch_{epoch}.pt")
            torch.save({"epoch": epoch, "state_dict": model.state_dict()}, weight_path)
            print(f"Saved model weights to {os.path.abspath(weight_path)}")
        except Exception as e:
            print(f"Warning: failed to save model weights for epoch {epoch}: {e}")

        # stop_timer()

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

    plt.figure(figsize=(10, 6))
    plt.plot(range(len(train_accs)), train_accs, marker="s", linestyle="--", color="orange", label="Training Accuracy")
    plt.plot(range(len(test_accs)), test_accs, marker="o", linestyle="-", color="blue", label="Test Accuracy")
    plt.xlabel("Epoch")
    plt.ylabel("Accuracy")
    plt.title("Training vs. Test Accuracy over Epochs")
    plt.legend()
    plt.grid(True)
    plt.savefig(f"{args.results_root}/{timestamp}/accuracy_comparison.png", dpi=150, bbox_inches="tight")
    plt.close()

    ####################
    #  TTA Evaluation  #
    ####################

    # start_timer()
    test_acc = evaluate(model, test_loader, tta_level=2)
    # stop_timer()
    epoch_label = "eval"
    print_training_details({**locals(), "epoch": epoch_label}, is_final_entry=True)

    # Calculate and print total training time
    total_end_time = time.time()
    total_training_time = total_end_time - total_start_time
    print(f"Total training time: {total_training_time:.4f} seconds")

    return test_acc


if __name__ == "__main__":
    ### Store experimental dict
    config_dict = vars(args)
    config_dict['timestamp'] = timestamp
    config_dict['run_name'] = args.run_name if args.run_name != "" else run_name
    print("\n--- Running Experiment with Configuration ---")
    print(json.dumps(config_dict, indent=4))
    print("--------------------------------------------\n")
    output_dir = os.path.join(args.results_root, timestamp)
    os.makedirs(output_dir, exist_ok=True)
    config_path = os.path.join(output_dir, "experimental_config.json")
    with open(config_path, "w") as f:
        json.dump(config_dict, f, indent=4)

    ## Setup WandB for tracking experiments ##
    wandb.init(
        entity="adaptive-muon",
        project="cnn-training", 
        config=config_dict,
        name=config_dict['run_name'],
        mode="disabled" if not args.wandb else "online"
    )
    ### Training code ###
    model = CifarNet(block1=args.block1,
                     block2=args.block2,
                     block3=args.block3,
                     apply_whitening=args.whitening).to(device)
    print(model)
    total_params = sum(p.numel() for p in model.parameters())
    print(f"Total parameters: {total_params}")

    # Only compile on CUDA systems
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
    print("About to print columns")

    print_columns(logging_columns_list, is_head=True)
    # main("warmup", model)
    accs = torch.tensor([main(run, model) for run in range(1)])
    # print("Mean: %.4f    Std: %.4f" % (accs.mean(), accs.std()))
    wandb.finish()
