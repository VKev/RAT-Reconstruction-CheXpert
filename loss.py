#!/usr/bin/env python
# -*- coding: utf-8 -*-

from typing import Any

import torch
import torch.nn as nn


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
    raise ValueError(f"Unknown loss '{name}'. Choose from: mse|l1|bce")

