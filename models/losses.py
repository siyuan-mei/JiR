from typing import List, Dict, Union

import numpy as np
import timm
import torch
import torch.nn.functional as F
from lpips import lpips, spatial_average
from monai.losses import PerceptualLoss, SSIMLoss
from monai.losses.perceptual import torchvision_zscore_norm
from torch import nn
from torch.nn import L1Loss, MSELoss, SmoothL1Loss

LOSS_REGISTRY = {
    "mae": lambda: L1Loss(),
    "mse": lambda: MSELoss(),
    "ssim": lambda: SSIMLoss(),
    "smooth_l1": lambda: SmoothL1Loss(),
    "perceptual_medicalnet": lambda: PerceptualLoss(
        spatial_dims=3, network_type="medicalnet_resnet50_23datasets", is_fake_3d=False
    ),
    "perceptual_vgg": lambda: PerceptualLoss(
        spatial_dims=3, network_type="vgg", is_fake_3d=True, fake_3d_ratio=0.5
    ),
    "perceptual_convnext": lambda: NewPerceptualLoss(
        spatial_dims=3,
        network_type="convnext_base_in22ft1k",
        is_fake_3d=True,
        fake_3d_ratio=0.5,
    ),
    "perceptual_vgg_2d": lambda: PerceptualLoss(
        spatial_dims=2, network_type="vgg", is_fake_3d=False
    ),
    "perceptual_convnext_2d": lambda: NewPerceptualLoss(
        spatial_dims=2, network_type="convnext_base_in22ft1k", is_fake_3d=False
    ),
    "gram": lambda: GramLoss(),
    "patchclip": lambda: PatchClipLoss(learnable_scale=True),
    "clsclip": lambda: ClsClipLoss(learnable_scale=True, init_temperature=10),
}


class PatchClipLoss(nn.Module):
    def __init__(self, learnable_scale: bool = False, init_temperature: float = 1 / 0.07):
        super().__init__()
        self.learnable_scale = learnable_scale
        self.init_temperature = init_temperature

    def forward(self, a: torch.Tensor, b: torch.Tensor, logit_scale=None) -> torch.Tensor:
        # a,b: [B, N, C]
        B, N, C = a.shape
        device = a.device

        a = a.reshape(B * N, C)
        b = b.reshape(B * N, C)

        a = a / (a.norm(dim=-1, keepdim=True) + 1e-6)
        b = b / (b.norm(dim=-1, keepdim=True) + 1e-6)

        if self.learnable_scale and logit_scale is not None:
            logit_scale = logit_scale.exp().clamp(max=100.0)
        else:
            logit_scale = self.init_temperature

        logits = logit_scale * (a @ b.t())  # [f, f], f=B*N
        labels = torch.arange(B * N, device=device)

        loss_ab = F.cross_entropy(logits, labels)
        loss_ba = F.cross_entropy(logits.t(), labels)
        return 0.5 * (loss_ab + loss_ba)

class ClsClipLoss(nn.Module):
    def __init__(self, learnable_scale: bool = False, init_temperature: float = 1 / 0.07):
        super().__init__()
        self.learnable_scale = learnable_scale
        self.init_temperature = init_temperature

    def forward(self, a: torch.Tensor, b: torch.Tensor, logit_scale=None) -> torch.Tensor:
        # a,b: [B, N, C]
        B, C = a.shape
        device = a.device

        a = a / (a.norm(dim=-1, keepdim=True) + 1e-6)
        b = b / (b.norm(dim=-1, keepdim=True) + 1e-6)

        if self.learnable_scale and logit_scale is not None:
            logit_scale = logit_scale.exp().clamp(max=100.0)
        else:
            logit_scale = self.init_temperature

        logits = logit_scale * (a @ b.t())  # [f, f], f=B*N
        labels = torch.arange(B, device=device)

        loss_ab = F.cross_entropy(logits, labels)
        loss_ba = F.cross_entropy(logits.t(), labels)
        return 0.5 * (loss_ab + loss_ba)


class GramLoss(nn.Module):
    """Implementation of the gram loss"""

    def __init__(
        self,
        apply_norm=True,
        remove_neg=False,
        remove_only_teacher_neg=False,
    ):
        super().__init__()
        # Loss
        self.mse_loss = torch.nn.MSELoss()
        # Parameters
        self.apply_norm = apply_norm
        self.remove_neg = remove_neg
        self.remove_only_teacher_neg = remove_only_teacher_neg
        if self.remove_neg or self.remove_only_teacher_neg:
            assert self.remove_neg != self.remove_only_teacher_neg

    def forward(self, output_feats, target_feats, img_level=True):
        """Compute the MSE loss between the gram matrix of the input and target features.

        Args:
            output_feats: Pytorch tensor (B, N, dim) or (B*N, dim) if img_level == False
            target_feats: Pytorch tensor (B, N, dim) or (B*N, dim) if img_level == False
            img_level: bool, if true gram computed at the image level only else over the entire batch
        Returns:
            loss: scalar
        """

        # Dimensions of the tensor should be (B, N, dim)
        if img_level:
            assert len(target_feats.shape) == 3 and len(output_feats.shape) == 3

        # Float casting
        output_feats = output_feats.float()
        target_feats = target_feats.float()

        # SSL correlation
        if self.apply_norm:
            target_feats = F.normalize(target_feats, dim=-1)

        if not img_level and len(target_feats.shape) == 3:
            # Flatten (B, N, D) into  (B*N, D)
            target_feats = target_feats.flatten(0, 1)

        # Compute similarities
        target_sim = torch.matmul(target_feats, target_feats.transpose(-1, -2))

        # Patch correlation
        if self.apply_norm:
            output_feats = F.normalize(output_feats, dim=-1)

        if not img_level and len(output_feats.shape) == 3:
            # Flatten (B, N, D) into  (B*N, D)
            output_feats = output_feats.flatten(0, 1)

        # Compute similarities
        student_sim = torch.matmul(output_feats, output_feats.transpose(-1, -2))

        if self.remove_neg:
            target_sim[target_sim < 0] = 0.0
            student_sim[student_sim < 0] = 0.0

        elif self.remove_only_teacher_neg:
            # Remove only the negative sim values of the teacher
            target_sim[target_sim < 0] = 0.0
            student_sim[(student_sim < 0) & (target_sim < 0)] = 0.0

        return self.mse_loss(student_sim, target_sim)


class TimmPretrainedModel(nn.Module):
    def __init__(
        self,
        model_name="convnext_base_in22ft1k",
        requires_grad=False,
    ):
        super().__init__()
        self.net = timm.create_model(model_name=model_name, pretrained=True)
        self.net.eval()  # Set the model to evaluation mode
        if not requires_grad:
            for param in self.parameters():
                param.requires_grad = False

    def forward(self, x):
        return self.net.forward_intermediates(x, intermediates_only=True)


class TimmModelPerceptualSimilarity(nn.Module):
    def __init__(
        self,
        net: str = "convnext_base_in22ft1k",
    ) -> None:
        super().__init__()
        supported_networks = ["convnext_base_in22ft1k"]
        if net not in supported_networks:
            raise NotImplementedError(
                f"'net' {net} is not supported, please select a network from {supported_networks}."
            )

        self.net = TimmPretrainedModel(model_name=net)

        self.net.eval()

        for param in self.parameters():
            param.requires_grad = False

    def forward(self, input: torch.Tensor, target: torch.Tensor):
        """
        We expect that the input is normalised between [0, 1]. Given the preprocessing performed during the training at
        https://pytorch.org/vision/main/models/generated/torchvision.models.resnet50.html#torchvision.models.ResNet50_Weights,
        we make sure that the input and target have 3 channels, and then do Z-Score normalization.
        The outputs are normalised across the channels, and we obtain the mean from the spatial dimensions (similar
        approach to the lpips package).
        """
        # If input has just 1 channel, repeat channel to have 3 channels
        if input.shape[1] == 1 and target.shape[1] == 1:
            input = input.repeat(1, 3, 1, 1)
            target = target.repeat(1, 3, 1, 1)

        # Input normalization
        input = torchvision_zscore_norm(input)
        target = torchvision_zscore_norm(target)

        # Get model outputs
        outs0, outs1 = self.net.forward(input), self.net.forward(target)
        feats0, feats1, diffs = {}, {}, {}

        for kk in range(len(outs0)):
            feats0[kk], feats1[kk] = (
                lpips.normalize_tensor(outs0[kk]),
                lpips.normalize_tensor(outs1[kk]),
            )
            diffs[kk] = (feats0[kk] - feats1[kk]) ** 2

        res = [
            spatial_average(diffs[kk].sum(dim=1, keepdim=True), keepdim=True)
            for kk in range(len(outs0))
        ]

        val = 0
        for l in range(len(outs0)):
            val += res[l]

        return val


class NewPerceptualLoss(PerceptualLoss):
    def __init__(
        self,
        spatial_dims: int = 3,
        network_type: str = "convnext_base_in22ft1k",
        is_fake_3d: bool = True,
        fake_3d_ratio: float = 0.5,
    ):
        super().__init__(
            spatial_dims=spatial_dims,
            network_type="vgg",
            is_fake_3d=is_fake_3d,
            fake_3d_ratio=fake_3d_ratio,
        )
        self.perceptual_function = TimmModelPerceptualSimilarity(network_type)


def get_loss_fn(loss_names: List[str]) -> Dict[str, torch.nn.Module]:
    loss_fn = {}
    for name in loss_names:
        if (
            name not in LOSS_REGISTRY
            and name.removeprefix("contrastive_") not in LOSS_REGISTRY
            and name.removeprefix("recon_") not in LOSS_REGISTRY
            and name.removeprefix("cycle_") not in LOSS_REGISTRY
            and name.removeprefix("syn_") not in LOSS_REGISTRY
        ):
            raise ValueError(f"Unsupported loss name: {name}")
        if name.startswith("contrastive_"):
            loss_fn[name] = LOSS_REGISTRY[name.removeprefix("contrastive_")]()
        elif name.startswith("recon_"):
            loss_fn[name] = LOSS_REGISTRY[name.removeprefix("recon_")]()
        elif name.startswith("cycle_"):
            loss_fn[name] = LOSS_REGISTRY[name.removeprefix("cycle_")]()
        elif name.startswith("syn_"):
            loss_fn[name] = LOSS_REGISTRY[name.removeprefix("syn_")]()
        else:
            loss_fn[name] = LOSS_REGISTRY[name]()
    return loss_fn
