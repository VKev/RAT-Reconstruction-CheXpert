#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
Visualize reconstructions after training with RAT model for flowers102 dataset.

Usage examples:
  # Auto-find latest best checkpoint for flowers102
  python visualize.py --dataset flowers102 --num-samples 2

  # Explicit checkpoint path
  python visualize.py --dataset flowers102 --ckpt-path outputs/lightning_logs/version_0/checkpoints/best-00.ckpt

  # CIFAR-10 with SmallVAE (legacy)
  python visualize.py --dataset cifar10 --num-samples 2
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
from utils import get_recon, generate_region_mask
from model import SmallVAE
from rat.Model_RAT import RAT
from datamodule import ImageDataModule


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


def build_base_model(dataset: str) -> nn.Module:
    """Build the appropriate model based on dataset."""
    if dataset == "flowers102":
        # RAT wrapper for flowers102 (224x224x3)
        class RATWrapper(nn.Module):
            def __init__(self):
                super().__init__()
                self.rat = RAT(scale=1, img_channel=3, width=64, middle_blk_num=12,
                                enc_blk_nums=[2, 2], dec_blk_nums=[2, 2], loss_fun=None)

            def forward(self, x):
                b, _, hh, ww = x.shape
                # Simple block mask: 14x14 grid regions over 224x224
                mask = generate_region_mask(b, hh, ww, grid_h=14, grid_w=14, device=x.device)
                return self.rat(x, mask)
        
        return RATWrapper()
    else:
        # SmallVAE for CIFAR-10 (32x32x3)
        return SmallVAE(latent_dim=128)


def main():
    parser = argparse.ArgumentParser(description="Visualize reconstructions from a trained checkpoint")
    parser.add_argument("--ckpt-path", type=str, default=None, help="Path to .ckpt (if omitted, auto-detect latest best)")

    # Dataset and model selection
    parser.add_argument("--dataset", type=str, default="flowers102", choices=["cifar10", "flowers102"],
                        help="Dataset name. 'cifar10' uses SmallVAE; 'flowers102' uses RAT")

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
    base_model = build_base_model(args.dataset)

    lit: LitAutoModule = LitAutoModule.load_from_checkpoint(ckpt_path, model=base_model, map_location=device)
    lit.eval()
    lit.to(device)

    # Prepare dataset using DataModule (same as training)
    dm = ImageDataModule(data_dir=args.data_dir, batch_size=args.num_samples, num_workers=0,
                         dataset=args.dataset)
    dm.prepare_data()
    dm.setup("test")  # Use test set for visualization
    
    if args.num_samples <= 0:
        print("[visualize] --num-samples must be >= 1")
        sys.exit(1)
    loader = dm.test_dataloader()

    # Fetch a single batch
    try:
        x, _ = next(iter(loader))
    except StopIteration:
        print("[visualize] Empty dataset.")
        sys.exit(1)
    x = x.to(device)

    # Register a forward hook to capture latent feature maps
    captured = {"feat": None}

    def _hook_fn(_module, _inp, _out):
        try:
            captured["feat"] = _out.detach()
        except Exception:
            captured["feat"] = None

    hooks = []
    try:
        if args.dataset == "flowers102":
            # Prefer deeper encoder output (smaller spatial size, more semantic)
            if hasattr(base_model, "rat") and hasattr(base_model.rat, "encoders") and len(base_model.rat.encoders) > 0:
                hooks.append(base_model.rat.encoders[-1].register_forward_hook(_hook_fn))
            elif hasattr(base_model, "rat") and hasattr(base_model.rat, "intro"):
                hooks.append(base_model.rat.intro.register_forward_hook(_hook_fn))
        else:
            # SmallVAE: grab last encoder activation (B,128,4,4)
            if hasattr(base_model, "enc"):
                hooks.append(base_model.enc.register_forward_hook(_hook_fn))
    except Exception:
        hooks = []

    # Forward and reconstruct
    with torch.no_grad():
        out = lit(x)
        recon = get_recon(out).clamp(0, 1)

    # Remove hooks
    for h in hooks:
        try:
            h.remove()
        except Exception:
            pass

    # Optionally compute PSNR as a quick quality indicator
    mse_per_sample = F.mse_loss(recon, x, reduction="none").mean(dim=(1, 2, 3))
    psnr_per_sample = -10.0 * torch.log10(mse_per_sample + 1e-8)
    print(f"[visualize] PSNR (mean over {args.num_samples} samples): {psnr_per_sample.mean().item():.2f} dB")

    # Compute latent activation heatmaps (mean/max) for the first sample
    try:
        feat = captured["feat"]
        if isinstance(feat, torch.Tensor) and feat.dim() == 4 and feat.size(0) >= 1:
            # Upsample to input spatial size
            feat_up = F.interpolate(feat, size=x.shape[-2:], mode="bilinear", align_corners=False)
            f0 = feat_up[0]  # (C,H,W)
            mean_map = f0.mean(0)
            max_map = f0.amax(0)

            # Normalize to [0,1]
            mean_map = (mean_map - mean_map.min()) / (mean_map.max() - mean_map.min() + 1e-8)
            max_map = (max_map - max_map.min()) / (max_map.max() - max_map.min() + 1e-8)

            # Build simple red-tint overlays for quick viewing
            x0 = x[0].clamp(0, 1)
            mean_rgb = torch.stack([mean_map, torch.zeros_like(mean_map), torch.zeros_like(mean_map)], dim=0)
            max_rgb = torch.stack([max_map, torch.zeros_like(max_map), torch.zeros_like(max_map)], dim=0)

            overlay_mean = (0.6 * x0 + 0.4 * mean_rgb).clamp(0, 1)
            overlay_max = (0.6 * x0 + 0.4 * max_rgb).clamp(0, 1)

            # Save overlays and raw maps
            save_image(overlay_mean, os.path.join(args.output_dir, f"overlay_mean_{args.dataset}.png"))
            save_image(overlay_max, os.path.join(args.output_dir, f"overlay_max_{args.dataset}.png"))

            # Also save grayscale heatmaps as 3-channel images for convenience
            mean_gray3 = mean_map.unsqueeze(0).repeat(3, 1, 1)
            max_gray3 = max_map.unsqueeze(0).repeat(3, 1, 1)
            save_image(mean_gray3, os.path.join(args.output_dir, f"heat_mean_{args.dataset}.png"))
            save_image(max_gray3, os.path.join(args.output_dir, f"heat_max_{args.dataset}.png"))

            print("[visualize] Saved latent activation heatmaps (mean/max) and overlays.")
        else:
            print("[visualize] Warning: Could not capture latent feature maps for heatmap visualization.")
    except Exception as e:
        print(f"[visualize] Heatmap generation failed: {e}")

    # Interleave originals and reconstructions vertically in the saved grid
    grid = torch.cat([x, recon], dim=0)
    save_path = os.path.join(args.output_dir, f"recon_grid_{args.dataset}.png")
    save_image(grid, save_path, nrow=args.num_samples, padding=2)
    print(f"[visualize] Dataset: {args.dataset}")
    print(f"[visualize] Model: {'RAT' if args.dataset == 'flowers102' else 'SmallVAE'}")
    print(f"[visualize] Image size: {x.shape[-2:]} x {x.shape[1]} channels")
    print(f"[visualize] Saved visualization to: {save_path}")


if __name__ == "__main__":
    main()

