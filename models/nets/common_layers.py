import math
import numpy as np
import torch
from torch import nn
from torch.nn import functional as F


def sobel_layer(x: torch.Tensor) -> torch.Tensor:
    device, dtype = x.device, x.dtype

    num_1 = torch.tensor([[1., 2., 1.],
                          [2., 4., 2.],
                          [1., 2., 1.]],
                         device=device, dtype=dtype)
    num_2 = torch.zeros((3, 3), device=device, dtype=dtype)
    num_3 = -num_1

    k = torch.zeros((3, 1, 3, 3, 3), device=device, dtype=dtype)
    # axis-0 (depth)
    k[0, 0, 0] = num_1
    k[0, 0, 1] = num_2
    k[0, 0, 2] = num_3
    # axis-1 (height)
    k[1, 0, :, 0] = num_1
    k[1, 0, :, 1] = num_2
    k[1, 0, :, 2] = num_3
    # axis-2 (width)
    k[2, 0, :, :, 0] = num_1
    k[2, 0, :, :, 1] = num_2
    k[2, 0, :, :, 2] = num_3

    x_pad = F.pad(x, (1, 1, 1, 1, 1, 1), mode="constant", value=-1.0)
    sob = F.conv3d(x_pad, k, padding=0, groups=1) / 4.0  # (B,3,D,H,W)

    # original: n,c,h,w,l = fake_sobel.size(); fake = norm(...)/c*3
    c = sob.shape[1]  # here c=3
    mag = torch.norm(sob, p=2, dim=1, keepdim=True) / c * 3.0  # (B,1,D,H,W)

    out = torch.tanh(mag) * 2.0 - 1.0
    return out

class RMSNorm(nn.Module):
    def __init__(self, hidden_size, eps=1e-6):
        """
        LlamaRMSNorm is equivalent to T5LayerNorm
        """
        super().__init__()
        self.weight = nn.Parameter(torch.ones(hidden_size))
        self.variance_epsilon = eps

    def forward(self, hidden_states):
        input_dtype = hidden_states.dtype
        hidden_states = hidden_states.to(torch.float32)
        variance = hidden_states.pow(2).mean(-1, keepdim=True)
        hidden_states = hidden_states * torch.rsqrt(variance + self.variance_epsilon)
        return (self.weight * hidden_states).to(input_dtype)

class MLP(nn.Module):
    def __init__(self, in_channels, exp_r=4, out_channels=None, drop=0.0):
        super().__init__()
        self.fc1 = nn.Conv3d(
            in_channels=in_channels,
            out_channels=exp_r * in_channels,
            kernel_size=1,
            stride=1,
            padding=0,
        )
        self.act = nn.GELU()
        self.fc2 = nn.Conv3d(
            in_channels=exp_r * in_channels,
            out_channels=out_channels,
            kernel_size=1,
            stride=1,
            padding=0,
        )
        self.drop = nn.Dropout(drop)

    def forward(self, x):
        x = self.fc1(x)
        x = self.act(x)
        x = self.drop(x)
        x = self.fc2(x)
        x = self.drop(x)
        return x


def get_norm_layer(in_channels, norm_type="instance", dim="3d"):
    assert dim in ["2d", "3d"]
    if norm_type == "bn":
        if dim == "2d":
            return nn.BatchNorm2d(in_channels)
        else:
            return nn.BatchNorm3d(in_channels)
    elif norm_type == "sync_bn":
        return nn.SyncBatchNorm(in_channels)
    elif norm_type == "instance":
        if dim == "2d":
            return nn.InstanceNorm2d(in_channels, affine=True)
        else:
            return nn.InstanceNorm3d(in_channels, affine=True)
    elif norm_type == "ln":
        return nn.LayerNorm(in_channels)
    elif norm_type == "gn":
        return nn.GroupNorm(
            num_groups=min(32, in_channels // 4), num_channels=in_channels
        )


def stem(in_channels=1, out_channels=32, kernel_size=3, dim="3d", norm_type="instance"):
    assert dim in ["2d", "3d"]
    dim = dim
    if dim == "2d":
        conv = nn.Conv2d
    else:
        conv = nn.Conv3d
    return nn.Sequential(
        conv(
            in_channels,
            out_channels,
            kernel_size=kernel_size,
            stride=1,
            padding=kernel_size // 2,
            bias=False,
        ),
        get_norm_layer(
            out_channels, dim=dim, norm_type="instance" if dim == "3d" else norm_type
        ),
        # nn.GELU(),
    )

class Stem(nn.Module):
    def __init__(self, in_channels=1, out_channels=32, kernel_size=1, norm_type="instance"):
        super().__init__()
        self.conv = nn.Conv3d(
            in_channels,
            out_channels,
            kernel_size=kernel_size,
            stride=1,
            padding=kernel_size // 2,
            bias=False,
        )
        self.norm_type = norm_type.lower()
        self.norm = nn.InstanceNorm3d(out_channels, affine=True)

    def forward(self, x):
        x = self.conv(x)
        x = self.norm(x)
        return x


def fuse_conv(in_channels, out_channels, dim="3d"):
    assert dim in ["2d", "3d"]
    dim = dim
    if dim == "2d":
        conv = nn.Conv2d
    else:
        conv = nn.Conv3d
    conv_layer = conv(
        in_channels, out_channels, kernel_size=1, stride=1, padding=0, bias=True
    )
    return conv_layer


class OutBlock(nn.Module):
    def __init__(self, in_channels, n_classes, dim, last_conv_kernel_size=1):
        super().__init__()
        if dim == "2d":
            conv = nn.Conv2d
        elif dim == "3d":
            conv = nn.Conv3d
        self.conv_out = conv(
            in_channels,
            n_classes,
            kernel_size=last_conv_kernel_size,
            bias=True,
            padding=last_conv_kernel_size // 2,
        )

    def forward(self, x):
        return self.conv_out(x)


def count_parameters(model, detailed=True):
    total_params = 0
    trainable_params = 0

    if detailed:
        print(f"{'Layer':40s} {'Params':>12s} {'Trainable':>12s}")
        print("=" * 70)
        for name, param in model.named_parameters():
            n_params = param.numel()
            total_params += n_params
            if param.requires_grad:
                trainable_params += n_params
                trainable_flag = "Yes"
            else:
                trainable_flag = "No"
            print(f"{name:40s} {n_params:12,d} {trainable_flag:>12s}")
        print("=" * 70)

    print(f"Total parameters: {total_params:,} ({total_params / 1e6:.2f} M)")
    print(f"Trainable parameters: {trainable_params:,} ({trainable_params / 1e6:.2f} M)")
    print(f"Parameter memory (float32): {total_params * 4 / 1024**2:.2f} MB")
    return total_params, trainable_params


class InputConditioner:
    def __init__(self, eps=1e-5, return_affine=True):
        self.eps = eps
        self.return_affine = return_affine
        if not self.return_affine:
            self.param_free_norm = nn.InstanceNorm3d(1, affine=False)

    @torch.no_grad()
    def __call__(self, x):
        if self.return_affine:
            mean = x.mean(dim=(2, 3, 4), keepdim=True)  # [B, C, 1, 1, 1]
            std = x.std(dim=(2, 3, 4), keepdim=True)
            x_norm = (x - mean) / (std + self.eps)
            return x_norm, (mean.squeeze(), std.squeeze())
        else:
            return self.param_free_norm(x), x


class PAIN(nn.Module):
    """ Patch-level Adaptive Instance Normalization"""
    def __init__(self, num_features, hidden_ratio=1.):
        super().__init__()
        self.param_free_norm = nn.InstanceNorm3d(num_features, affine=False)
        hidden_dim = int(num_features * hidden_ratio)
        self.fc_mu = nn.Sequential(
            nn.Linear(1, hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, num_features),
        )
        self.fc_std = nn.Sequential(
            nn.Linear(1, hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, num_features)
        )

    def forward(self, x: torch.Tensor, patch_mean: torch.Tensor, patch_std: torch.Tensor):
        """
        Args:
            x: [B, C, D, H, W]
            patch_mean: [B] or [B, 1] — mean intensity of the patch
            patch_std:  [B] or [B, 1] — std intensity of the patch
        """
        x_norm = self.param_free_norm(x)

        # ensure [B, 1]
        patch_mean = patch_mean.view(-1, 1)
        patch_std = patch_std.view(-1, 1)

        # project to per-channel modulation
        delta_mu = self.fc_mu(patch_mean)  # [B, C]
        delta_std = self.fc_std(patch_std)  # [B, C]

        # apply modulation
        x_mod = x_norm * (1 + delta_std[:, :, None, None, None]) + delta_mu[:, :, None, None, None]
        return x_mod


class SPADE(nn.Module):
    def __init__(self, norm_nc, label_nc, ks=1, use_act=False):
        super().__init__()
        self.param_free_norm = nn.InstanceNorm3d(norm_nc, affine=False)
        nhidden = norm_nc # // 2
        self.mlp_shared = nn.Sequential(
            nn.Conv3d(label_nc, nhidden, kernel_size=3, padding=1, groups=label_nc), nn.GELU()
        )
        self.mlp_gamma = nn.Conv3d(nhidden, norm_nc, kernel_size=ks, padding=(ks - 1) // 2)
        self.mlp_beta = nn.Conv3d(nhidden, norm_nc, kernel_size=ks, padding=(ks - 1) // 2)
        self.use_act = use_act

    def forward(self, x_feat, cond_map):
        # Part 1. generate parameter-free normalized activations
        normalized = self.param_free_norm(x_feat)

        # Part 2. produce scaling and bias conditioned on conditional map
        cond_map = F.interpolate(
            cond_map, size=x_feat.size()[2:], mode="trilinear", align_corners=False
        )
        actv = self.mlp_shared(cond_map)
        gamma = self.mlp_gamma(actv)
        beta = self.mlp_beta(actv)

        # apply scale and bias
        out = normalized * (1 + gamma) + beta

        if self.use_act:
            out = F.gelu(out)
        return out


# class SpatialPosEmbLayer(nn.Module):
#     """
#     Encode 3D positional map into a compact global embedding.
#     Steps:
#         pos_map [B, 3, D, H, W]
#           ↓ Encoding (Fourier or Sine)
#         [B, emb_dim, D, H, W]
#           ↓ Per-axis pooling
#         [B, emb_dim, 3]
#           ↓ Flatten + MLP
#         [B, emb_dim* 4]
#     """
#
#     def __init__(self, emb_dim=32, coding_type="fourier", fourier_scale=16., learnable_scale=True,):
#         super().__init__()
#         self.emb_dim = emb_dim
#         self.coding_type = coding_type.lower()
#
#         # fixed random frequencies for Fourier encoding
#         if self.coding_type == "fourier":
#             W = torch.randn(3, emb_dim // 2) * fourier_scale
#             self.W = nn.Parameter(W, requires_grad=learnable_scale)
#
#         # projection MLP
#         self.mlp = nn.Sequential(
#             nn.Linear(emb_dim * 3, emb_dim * 4),
#             nn.GELU(),
#             nn.Linear(emb_dim * 4, emb_dim * 4),
#         )
#
#     def _fourier_encode(self, pos_map: torch.Tensor):
#         """Gaussian Fourier encoding"""
#         pos_map = pos_map.permute(0, 2, 3, 4, 1)  # [B,D,H,W,3]
#         proj = torch.matmul(pos_map, self.W) * 2 * np.pi  # [B,D,H,W,emb_dim/2]
#         pos_emb = torch.cat([torch.sin(proj), torch.cos(proj)], dim=-1)
#         return pos_emb.permute(0, 4, 1, 2, 3).contiguous()  # [B,emb_dim,D,H,W]
#
#     def _positional_encode(self, pos_map: torch.Tensor):
#         """Transformer-style positional encoding"""
#         half_dim = self.emb_dim // 2
#         div_term = torch.exp(
#             torch.arange(0, half_dim, device=pos_map.device) *
#             (-math.log(10000.0) / half_dim)
#         )
#         W = div_term.repeat(3, 1)  # [3, half_dim]
#         pos = pos_map.permute(0, 2, 3, 4, 1)
#         proj = torch.matmul(pos, W)
#         emb = torch.cat([torch.sin(proj), torch.cos(proj)], dim=-1)
#         return emb.permute(0, 4, 1, 2, 3).contiguous()
#
#     def forward(self, pos_map: torch.Tensor):
#         """pos_map: [B, 3, D, H, W]"""
#         if self.coding_type == "fourier":
#             emb = self._fourier_encode(pos_map)
#         elif self.coding_type == "positional":
#             emb = self._positional_encode(pos_map)
#         else:
#             raise ValueError(f"Unknown coding_type: {self.coding_type}")
#
#         # choose pooling
#         reduce_fn = torch.mean
#
#         # per-axis pooling
#         x_pool = reduce_fn(emb, dim=(3, 4)).mean(dim=2)
#         y_pool = reduce_fn(emb, dim=(2, 4)).mean(dim=2)
#         z_pool = reduce_fn(emb, dim=(2, 3)).mean(dim=2)
#
#         per_axis = torch.stack([x_pool, y_pool, z_pool], dim=-1)
#         flat = per_axis.flatten(start_dim=1)
#         out = self.mlp(flat)
#         return out


class Embedding(nn.Module):
    def __init__(
        self,
        in_channels: int,
        embed_dim: int,
        out_dim: int = None,
    ):
        super().__init__()

        self.linear_1 = nn.Linear(in_channels, embed_dim, True)

        self.act = nn.GELU()

        if out_dim is not None:
            embed_dim = out_dim
        else:
            embed_dim = embed_dim

        self.linear_2 = nn.Linear(embed_dim, embed_dim, True)

    def forward(self, sample):
        return self.linear_2(self.act(self.linear_1(sample)))


class GaussianFourierProjection(nn.Module):
    """Gaussian Fourier embeddings for continuous conditioning variables.
    Supports both scalar [B] and vector [B, D] inputs.
    """

    def __init__(self, embedding_size: int = 256, input_dim: int = 1, scale: float = 1.0, log: bool = False):
        super().__init__()
        self.weight = nn.Parameter(
            torch.randn(input_dim, embedding_size) * scale,
            requires_grad=False,
        )
        self.log = log

    def forward(self, x: torch.Tensor):
        # ensure input shape [B, D]
        if x.dim() == 1:
            x = x.unsqueeze(-1)

        # optional log-space transform
        if self.log:
            x = torch.log(x + 1e-8)  # avoid log(0)

        # project
        proj = 2 * np.pi * x @ self.weight  # [B, emb_dim]
        emb = torch.cat([torch.sin(proj), torch.cos(proj)], dim=-1)
        return emb


def get_positional_embedding(
    input_tensor: torch.Tensor,
    embedding_dim: int,
    input_dim: int = 1,
    downscale_freq_shift: float = 1.0,
    max_period: int = 10000,
) -> torch.Tensor:
    """
    Generalized sinusoidal positional embedding (Diffusers-style)
    input_tensor: [B, D] or [B]
    return: [B, embedding_dim]
    """
    if input_tensor.dim() == 1:
        input_tensor = input_tensor.unsqueeze(-1)

    half_dim = embedding_dim // 2
    freqs = torch.exp(
        -math.log(max_period)
        * torch.arange(0, half_dim, dtype=torch.float32, device=input_tensor.device)
        / (half_dim - downscale_freq_shift)
    )  # [half_dim]

    W = freqs.repeat(input_dim, 1)  # [input_dim, half_dim]
    proj = input_tensor @ W  # [B, half_dim]

    emb = torch.cat([torch.sin(proj), torch.cos(proj)], dim=-1)
    return emb


class SinusoidalPositionalProjection(nn.Module):
    def __init__(self, embedding_size: int, input_dim: int = 1):
        super().__init__()
        self.embedding_size = embedding_size
        self.input_dim = input_dim

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        emb = get_positional_embedding(
            x,
            embedding_dim=self.embedding_size,
            input_dim=self.input_dim,
        )
        return emb

class PosEmbedding(nn.Module):
    def __init__(self, emb_dim=32, embedding_type="fourier", concat_pos=False):
        super().__init__()
        self.emb_dim = emb_dim
        self.embedding_type = embedding_type.lower()
        self.concat_pos = concat_pos

        # Separate random frequencies for center and spacing
        if self.embedding_type == "fourier":
            self.project = GaussianFourierProjection(embedding_size=emb_dim, input_dim=6)
            in_dim = emb_dim * 2 + (6 if concat_pos else 0)
        elif self.embedding_type == "positional":
            self.project = SinusoidalPositionalProjection(embedding_size=emb_dim, input_dim=6)
            in_dim = emb_dim + (6 if concat_pos else 0)
        elif self.embedding_type == "simple_project":
            in_dim = 6
        else:
            raise NotImplementedError(f"Embedding type {self.embedding_type} not implemented")

        self.embedding = Embedding(in_dim, emb_dim  * 4)

    def forward(self, pos_map: torch.Tensor):
        """
        pos_map: [B, 3, D, H, W] (cropped patch, coords are global)
        """
        B, C, D, H, W = pos_map.shape
        assert C == 3, f"Expected 3 channels (x,y,z), got {C}"

        center_z, center_y, center_x = D // 2, H // 2, W // 2
        center = pos_map[:, :, center_z, center_y, center_x]  # [B, 3]

        spacing = torch.stack([
            (pos_map[:, 0, 1, 0, 0] - pos_map[:, 0, 0, 0, 0]).abs(),  # Δz
            (pos_map[:, 1, 0, 1, 0] - pos_map[:, 1, 0, 0, 0]).abs(),  # Δy
            (pos_map[:, 2, 0, 0, 1] - pos_map[:, 2, 0, 0, 0]).abs(),  # Δx
        ], dim=-1)  # [B, 3]

        emb = torch.cat([center, spacing], dim=-1)  # [B, 6]

        if self.embedding_type == "fourier" or self.embedding_type == "positional":
            emb = self.project(emb)
            if self.concat_pos:
                emb = torch.cat([emb, center, spacing], dim=-1)

        return self.embedding(emb)

