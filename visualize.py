#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
Visualize reconstructions after training.

Usage examples:
  # Auto-find latest best checkpoint
  python visualize.py --num-samples 2

  # Explicit checkpoint path
  python visualize.py --ckpt-path outputs/lightning_logs/version_0/checkpoints/best-00.ckpt

  # Optionally point to a specific checkpoint
"""

import argparse
import os
import sys
from glob import glob
from typing import Optional

import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision
from torchvision import transforms as T
from torchvision.utils import save_image

# Reuse helpers and LightningModule from train.py
from train import LitAutoModule
from utils import get_recon
from model import SmallVAE


def find_latest_best_ckpt(root_dir: str = "outputs") -> Optional[str]:
    """Return the newest 'best-*.ckpt' path under outputs/lightning_logs/**/checkpoints.

    If none is found, returns None.
    """
    pattern = os.path.join(root_dir, "lightning_logs", "**", "checkpoints", "best-*.ckpt")
    candidates = glob(pattern, recursive=True)
    if not candidates:
        return None
    candidates.sort(key=lambda p: os.path.getmtime(p), reverse=True)
    return candidates[0]


def build_base_model() -> nn.Module:
    return SmallVAE(latent_dim=128)


def main():
    parser = argparse.ArgumentParser(description="Visualize reconstructions from a trained checkpoint")
    parser.add_argument("--ckpt-path", type=str, default=None, help="Path to .ckpt (if omitted, auto-detect latest best)")

    # Model: always use SmallVAE from model.py

    # Data / output
    parser.add_argument("--data-dir", type=str, default="./data")
    parser.add_argument("--num-samples", type=int, default=2)
    parser.add_argument("--output-dir", type=str, default="outputs/visualize")
    parser.add_argument("--seed", type=int, default=42)

    args = parser.parse_args()

    os.makedirs(args.output_dir, exist_ok=True)

    # Precision hint for Ampere+ GPUs
    try:
        torch.set_float32_matmul_precision("high")
    except Exception:
        pass

    torch.manual_seed(args.seed)
    torch.cuda.manual_seed_all(args.seed)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Auto-detect checkpoint if not provided
    ckpt_path = args.ckpt_path or find_latest_best_ckpt(root_dir=os.path.dirname(args.output_dir) or "outputs")
    if ckpt_path is None or not os.path.exists(ckpt_path):
        print("[visualize] Could not locate checkpoint. Provide --ckpt-path or ensure training saved a best-*.ckpt.")
        sys.exit(1)
    print(f"[visualize] Using checkpoint: {ckpt_path}")

    # Build base model and Lightning module, then load weights
    base_model = build_base_model()

    lit: LitAutoModule = LitAutoModule.load_from_checkpoint(ckpt_path, model=base_model, map_location=device)
    lit.eval()
    lit.to(device)

    # Prepare a tiny train sampler from CIFAR10
    transform = T.Compose([T.ToTensor()])
    train_set = torchvision.datasets.CIFAR10(args.data_dir, train=True, transform=transform, download=True)
    if args.num_samples <= 0:
        print("[visualize] --num-samples must be >= 1")
        sys.exit(1)
    loader = torch.utils.data.DataLoader(train_set, batch_size=args.num_samples, shuffle=True)

    # Fetch a single batch
    try:
        x, _ = next(iter(loader))
    except StopIteration:
        print("[visualize] Empty dataset.")
        sys.exit(1)
    x = x.to(device)

    # Forward and reconstruct
    with torch.no_grad():
        out = lit(x)
        recon = get_recon(out).clamp(0, 1)

    # Optionally compute PSNR as a quick quality indicator
    mse_per_sample = F.mse_loss(recon, x, reduction="none").mean(dim=(1, 2, 3))
    psnr_per_sample = -10.0 * torch.log10(mse_per_sample + 1e-8)
    print(f"[visualize] PSNR (mean over {args.num_samples} samples): {psnr_per_sample.mean().item():.2f} dB")

    # Interleave originals and reconstructions vertically in the saved grid
    grid = torch.cat([x, recon], dim=0)
    save_path = os.path.join(args.output_dir, "recon_grid.png")
    save_image(grid, save_path, nrow=args.num_samples, padding=2)
    print(f"[visualize] Saved visualization to: {save_path}")


if __name__ == "__main__":
    main()

