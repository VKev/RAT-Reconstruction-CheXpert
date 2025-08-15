#!/usr/bin/env python
# -*- coding: utf-8 -*-

from typing import Any

import torch
import torch.nn as nn


class FocalRegionLoss(nn.Module):
    """Focal Region Loss used by RAT.

    Re-implemented here to avoid importing from rat.losses.
    It computes per-pixel L1, then averages weights per region index in a mask
    and reweights the pixel losses accordingly.

    Args:
        beta: scales the region weights contribution.
    """

    def __init__(self, beta: float = 1.0, epsilon: float = 1e-3) -> None:
        super().__init__()
        self.beta = float(beta)
        self.epsilon2 = float(epsilon) * float(epsilon)
        self.l1 = nn.L1Loss(reduction="none")

    def forward(self, input: torch.Tensor, target: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
        # input/target: [B, C, H, W]; mask: [B, H, W] (integers of region ids)
        if mask.dim() != 3:
            raise ValueError(f"mask must be [B,H,W], got {mask.shape}")
        if input.shape[-2:] != target.shape[-2:] or input.shape[-2:] != mask.shape[-2:]:
            raise ValueError("spatial sizes of input, target, and mask must match")

        loss_metric = self.l1(input, target)  # [B, C, H, W]
        weight = loss_metric.detach().clone()  # [B, C, H, W]

        batch_size, channels, height, width = weight.shape
        device = weight.device

        for b in range(batch_size):
            mask_b = mask[b].to(device)
            # Ensure integer region ids starting from 0..K-1
            max_region = int(mask_b.max().item())
            total_area = 0
            for region_id in range(max_region + 1):
                region = (mask_b == region_id)
                area = int(region.sum().item())
                if area > 0:
                    avg_region = weight[b, :, region].mean()
                    weight[b, :, region] = avg_region
                    total_area += area
            if total_area != height * width:
                raise ValueError("Total Area Error in FocalRegionLoss: mask does not cover all pixels")

        # Normalize weights to [0,1]
        max_w = weight.max()
        if max_w > 0:
            weight = weight / max_w
        weight = torch.clamp(weight, 0.0, 1.0)

        return torch.mean(loss_metric * (weight * self.beta + 1.0))


def kl_divergence(mu: torch.Tensor, logvar: torch.Tensor) -> torch.Tensor:
    """KL divergence term for VAEs.

    Returns the mean over the batch.
    """
    return -0.5 * torch.mean(1 + logvar - mu.pow(2) - logvar.exp())


def make_criterion(name: str) -> nn.Module:
    name = name.lower()
    if name == "mse":
        return nn.MSELoss()
    if name == "l1":
        return nn.L1Loss()
    if name in ("bce", "bcewithlogits"):
        return nn.BCEWithLogitsLoss()
    if name in ("focal", "focal_region"):
        return FocalRegionLoss(beta=1.0)
    raise ValueError(f"Unknown loss '{name}'. Choose from: mse|l1|bce")

