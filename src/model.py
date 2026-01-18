import torch
import torch.nn.functional as F
from torch import nn

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