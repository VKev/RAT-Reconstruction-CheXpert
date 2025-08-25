import argparse
from typing import Union

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader

import lightning as L
from lightning.pytorch.callbacks import ModelCheckpoint, LearningRateMonitor, TQDMProgressBar
from lightning.pytorch.loggers import WandbLogger

from datamodule import ImageDataModule
from model import SmallVAE
from rat.Model_RAT import RAT
from loss import make_criterion, kl_divergence, FocalRegionLoss, MultiLabelFocalLossWithLogits
from save import SavePredictionsCallback, SegmentedBestCheckpointCallback, ClassificationAccEvalCallback
from utils import get_recon, generate_region_mask, ssim_tensor


# -------------------------------
# Default tiny VAE baseline
# -------------------------------



# -------------------------------
# Loss helpers
# -------------------------------



# -------------------------------
# Lightning wrapper
# -------------------------------

class LitAutoModule(L.LightningModule):
    def __init__(self, model: nn.Module, lr: float = 1e-3, loss: str = "mse",
                 add_kld_if_available: bool = True, beta: float = 0.001,
                 input_noise_std: float = 0.0,
                 use_default_grid_mask: bool = True, grid_h: int = 14, grid_w: int = 14,
                 phase: int = 1, num_classes: int = 14, cls_weight: float = 1.0, recon_weight: float = 1.0,
                 mlp_hidden: int = 512, mlp_dropout: float = 0.2):
        super().__init__()
        self.model = model
        self.save_hyperparameters(ignore=["model"])

        self.crit = make_criterion(loss)
        self.add_kld_if_available = add_kld_if_available
        self.beta = beta
        self.example_input_array = torch.randn(2, 3, 32, 32)
        # Phase 2 components
        self.phase = int(phase)
        self.num_classes = int(num_classes)
        self.cls_weight = float(cls_weight)
        self.recon_weight = float(recon_weight)
        self.mlp_hidden = int(mlp_hidden)
        self.mlp_dropout = float(mlp_dropout)
        if self.phase == 2:
            # Prefer a Lazy Linear-based head so optimizer includes its params before first batch
            self._cls_head = nn.Sequential(
                nn.AdaptiveAvgPool2d(1),
                nn.Flatten(),
                nn.LazyLinear(256),
                nn.GELU(),
                nn.Dropout(self.mlp_dropout),
                nn.Linear(256, 256),
                nn.GELU(),
                nn.Dropout(self.mlp_dropout),
                nn.Linear(256, self.num_classes),
            )  # type: ignore[attr-defined]
            # Focal loss for imbalanced multi-label classification
            self.cls_loss = MultiLabelFocalLossWithLogits(alpha=0.25, gamma=2.0, reduction="mean")

    def _make_cls_head(self, in_channels: int, prefix: str, device: torch.device) -> nn.Sequential:
        # Build a 2-layer MLP head with activation and dropout; print shape after Flatten
        head = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Flatten(),
            nn.Linear(in_channels, 256),
            nn.GELU(),
            nn.Dropout(self.mlp_dropout),
            nn.Linear(256, 256),
            nn.GELU(),
            nn.Dropout(self.mlp_dropout),
            nn.Linear(256, self.num_classes),
        )
        return head.to(device)
        # Cache for default grid masks per (device,b,h,w,grid_h,grid_w)
        self._mask_cache: dict[tuple, torch.Tensor] = {}

    def _get_default_mask(self, batch_size: int, height: int, width: int, device: torch.device) -> torch.Tensor:
        # Safety: ensure cache exists even if model was restored in a way that skipped __init__ assignments
        if not hasattr(self, "_mask_cache") or self._mask_cache is None:
            try:
                self._mask_cache = {}
            except Exception:
                self.__dict__["_mask_cache"] = {}
        key = (device.type, int(getattr(device, 'index', -1) or -1), batch_size, height, width,
               int(getattr(self.hparams, "grid_h", 14)), int(getattr(self.hparams, "grid_w", 14)))
        m = self._mask_cache.get(key, None)
        if m is None or m.device != device:
            m = generate_region_mask(
                batch_size, height, width,
                grid_h=int(getattr(self.hparams, "grid_h", 14)),
                grid_w=int(getattr(self.hparams, "grid_w", 14)),
                device=device,
            )
            self._mask_cache[key] = m
        return m

    # -----------------
    # Logging helpers
    # -----------------
    def _log_both(self, base: str, value: torch.Tensor, prog_bar_step: bool = False, prog_bar_epoch: bool = False):
        try:
            self.log(f"{base}/step", value, prog_bar=prog_bar_step, on_step=True, on_epoch=False)
        except Exception:
            pass
        try:
            self.log(f"{base}/epoch", value, prog_bar=prog_bar_epoch, on_step=False, on_epoch=True)
        except Exception:
            pass

    def forward(self, x, mask: torch.Tensor | None = None):
        try:
            return self.model(x, mask)
        except TypeError:
            return self.model(x)

    def _compute_loss(self, x, out, mask_for_loss: torch.Tensor | None = None) -> torch.Tensor:
        recon = get_recon(out)
        # If using FocalRegionLoss (RAT), require a region mask at input resolution
        if isinstance(self.crit, FocalRegionLoss):
            if mask_for_loss is None and bool(getattr(self.hparams, "use_default_grid_mask", True)):
                b, _, hh, ww = x.shape
                mask_for_loss = generate_region_mask(
                    b, hh, ww,
                    grid_h=int(getattr(self.hparams, "grid_h", 14)),
                    grid_w=int(getattr(self.hparams, "grid_w", 14)),
                    device=x.device,
                )
            rec = self.crit(recon, x, mask_for_loss)
        else:
            rec = self.crit(recon, x)
        total = rec

        # Add KL term if (mu, logvar) are provided by the model
        mu, logvar = None, None
        if isinstance(out, (list, tuple)):
            if len(out) >= 3 and all(isinstance(t, torch.Tensor) for t in out[:3]):
                _, mu, logvar = out[:3]
        elif isinstance(out, dict):
            mu = out.get("mu", None)
            logvar = out.get("logvar", None)

        if self.add_kld_if_available and mu is not None and logvar is not None:
            kld = kl_divergence(mu, logvar)
            total = total + self.beta * kld
            self._log_both("loss/kld", kld, prog_bar_step=False, prog_bar_epoch=False)

        self._log_both("loss/recon", rec, prog_bar_step=True, prog_bar_epoch=True)
        return total

    def training_step(self, batch, batch_idx):
        x, y = batch
        x_in = x
        try:
            noise_std = float(getattr(self.hparams, "input_noise_std", 0.0) or 0.0)
        except Exception:
            noise_std = 0.0
        if noise_std > 0.0:
            noise = torch.randn_like(x) * noise_std
            x_in = (x + noise).clamp(0, 1)
        # Prefer channels_last for faster CUDA convolutions
        try:
            x_in = x_in.contiguous(memory_format=torch.channels_last)
        except Exception:
            pass
        # Choose mask strategy
        mask = None
        labels = None
        use_default = bool(getattr(self.hparams, "use_default_grid_mask", True))
        if isinstance(y, dict):
            labels = y.get("labels", None)
            mask = y.get("mask", None)
        if mask is None:
            if use_default:
                b, _, hh, ww = x_in.shape
                mask = self._get_default_mask(b, hh, ww, device=x_in.device)
            else:
                try:
                    if torch.is_tensor(y) and y.dim() >= 2:
                        mask = y
                except Exception:
                    mask = None

        out = self(x_in, mask=mask)
        # Reconstruction loss as before (rec)
        loss_recon = self._compute_loss(x, out, mask_for_loss=mask)
        total = loss_recon if self.phase == 1 else torch.tensor(0.0, device=x.device)

        if self.phase == 2:
            # Middle features for classification
            features = None
            try:
                rat = getattr(self.model, "rat", None) or getattr(self, "rat", None)
                if rat is not None and hasattr(rat, "extract_pre_post_features"):
                    pre_f, post_f = rat.extract_pre_post_features(x_in, mask if mask is not None else generate_region_mask(x_in.size(0), x_in.size(2), x_in.size(3), device=x_in.device))
                    # Concatenate along channel dimension
                    features = torch.cat([pre_f, post_f], dim=1)
                elif rat is not None and hasattr(rat, "extract_middle_features"):
                    features = rat.extract_middle_features(x_in, mask if mask is not None else generate_region_mask(x_in.size(0), x_in.size(2), x_in.size(3), device=x_in.device))
            except Exception:
                features = None
            if features is None:
                # Fallback: use reconstruction tensor as features (not ideal)
                features = get_recon(out)
            # Ensure classification head exists (created lazily in __init__ with LazyLinear)
            if getattr(self, "_cls_head", None) is None:
                # Extremely defensive: fallback to non-lazy head if missing for any reason
                in_ch = int(features.size(1))
                self._cls_head = self._make_cls_head(in_ch, prefix="[train]", device=features.device)
            logits = self._cls_head(features)
            if labels is None and isinstance(y, dict) and "labels" in y:
                labels = y["labels"]
            if labels is None:
                # Try zeros if labels unavailable
                labels = torch.zeros(x.size(0), self.num_classes, device=logits.device)
            labels = labels.to(logits.device).float()
            # Ensure multi-label shape [B, C]
            if labels.dim() == 1:
                labels = labels.unsqueeze(0)
            # If flatten size matches logits, reshape
            if labels.numel() == logits.numel() and labels.shape != logits.shape:
                labels = labels.view_as(logits)
            # Align class dimension
            if labels.shape[1] != logits.shape[1]:
                if labels.shape[1] > logits.shape[1]:
                    labels = labels[:, : logits.shape[1]]
                else:
                    pad = torch.zeros(labels.shape[0], logits.shape[1] - labels.shape[1], device=labels.device)
                    labels = torch.cat([labels, pad], dim=1)
            loss_cls = self.cls_loss(logits, labels)
            self._log_both("loss/cls", loss_cls, prog_bar_step=True, prog_bar_epoch=True)
            # Weighted combination for phase 2
            total = self.recon_weight * loss_recon + self.cls_weight * loss_cls

        self._log_both("loss/train", total, prog_bar_step=True, prog_bar_epoch=True)
        return total

    def validation_step(self, batch, batch_idx):
        x, y = batch
        with torch.no_grad():
            # Choose mask strategy
            mask = None
            labels = None
            use_default = bool(getattr(self.hparams, "use_default_grid_mask", True))
            if isinstance(y, dict):
                labels = y.get("labels", None)
                mask = y.get("mask", None)
            if mask is None:
                if use_default:
                    b, _, hh, ww = x.shape
                    mask = self._get_default_mask(b, hh, ww, device=x.device)
                else:
                    try:
                        if torch.is_tensor(y) and y.dim() >= 2:
                            mask = y
                    except Exception:
                        mask = None
            # Prefer channels_last for faster CUDA convolutions
            try:
                x = x.contiguous(memory_format=torch.channels_last)
            except Exception:
                pass
            out = self(x, mask=mask)
        loss_recon = self._compute_loss(x, out, mask_for_loss=mask)
        total = loss_recon if self.phase == 1 else torch.tensor(0.0, device=x.device)

        if self.phase == 2:
            # Classification validation loss
            try:
                rat = getattr(self.model, "rat", None) or getattr(self, "rat", None)
                if rat is not None and hasattr(rat, "extract_pre_post_features"):
                    pre_f, post_f = rat.extract_pre_post_features(x, mask if mask is not None else generate_region_mask(x.size(0), x.size(2), x.size(3), device=x.device))
                    features = torch.cat([pre_f, post_f], dim=1)
                else:
                    features = rat.extract_middle_features(x, mask if mask is not None else generate_region_mask(x.size(0), x.size(2), x.size(3), device=x.device))
                # Ensure head exists and matches channel count
                if getattr(self, "_cls_head", None) is None:
                    in_ch = int(features.size(1))
                    self._cls_head = self._make_cls_head(in_ch, prefix="[val]", device=features.device)
                logits = self._cls_head(features)
                if labels is None and isinstance(y, dict) and "labels" in y:
                    labels = y["labels"]
                if labels is None:
                    labels = torch.zeros(x.size(0), self.num_classes, device=logits.device)
                labels = labels.to(logits.device).float()
                if labels.dim() == 1:
                    labels = labels.unsqueeze(0)
                if labels.numel() == logits.numel() and labels.shape != logits.shape:
                    labels = labels.view_as(logits)
                if labels.shape[1] != logits.shape[1]:
                    if labels.shape[1] > logits.shape[1]:
                        labels = labels[:, : logits.shape[1]]
                    else:
                        pad = torch.zeros(labels.shape[0], logits.shape[1] - labels.shape[1], device=labels.device)
                        labels = torch.cat([labels, pad], dim=1)
                loss_cls = self.cls_loss(logits, labels)
                self._log_both("loss/cls_val", loss_cls, prog_bar_step=False, prog_bar_epoch=True)
                total = self.recon_weight * loss_recon + self.cls_weight * loss_cls
            except Exception:
                pass

        # Log both batch-level (for step plots) and epoch-level aggregation
        self._log_both("loss/val", total, prog_bar_step=False, prog_bar_epoch=True)

        # PSNR for a quick quality signal (assuming outputs are in [0,1])
        recon = get_recon(out).clamp(0, 1)
        mse = F.mse_loss(recon, x, reduction="none").mean(dim=(1,2,3))
        psnr = -10.0 * torch.log10(mse + 1e-8)
        self._log_both("metric/psnr", psnr.mean(), prog_bar_step=False, prog_bar_epoch=True)

    def configure_optimizers(self):
        return torch.optim.AdamW(self.parameters(), lr=self.hparams.lr)


# -------------------------------
# Callback: save sample reconstructions
# -------------------------------



# -------------------------------
# CLI / main
# -------------------------------

def parse_devices(dev_str: str) -> Union[str, int, list]:
    if dev_str.lower() == "auto":
        return "auto"
    # Allow comma-separated GPU indices like "0,1"
    if "," in dev_str:
        return [int(x) for x in dev_str.split(",") if x.strip().isdigit()]
    if dev_str.isdigit():
        return int(dev_str)
    return "auto"

def main():
    parser = argparse.ArgumentParser(description="Training (Lightning) with CIFAR-10, Flowers102, or CheXpert")
    # Model: always use SmallVAE from model.py

    # Training options
    parser.add_argument("--data-dir", type=str, default="./data")
    parser.add_argument("--dataset", type=str, default="chexpert", choices=["cifar10", "flowers102", "chexpert"],
                        help="Dataset name. 'cifar10' keeps 32x32; 'flowers102' and 'chexpert' are resized to 224x224")
    parser.add_argument("--batch-size", type=int, default=12)
    parser.add_argument("--num-workers", type=int, default=4)
    parser.add_argument("--max-epochs", type=int, default=1)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--loss", type=str, default="mse", choices=["mse","l1","bce","focal"],
                        help="Loss function. Use 'focal' for RAT on flowers102 to enable FocalRegionLoss")
    parser.add_argument("--beta", type=float, default=0.001, help="KLD weight (if available)")
    parser.add_argument("--add-kld-if-available", action="store_true", help="Add KL term if model returns (mu, logvar)")
    parser.add_argument("--input-noise-std", type=float, default=0.0, help="Std of Gaussian noise added to inputs during training (denoising)")
    # RAT hyperparameters (declare upfront to avoid unrecognized args)
    parser.add_argument("--rat-width", type=int, default=64, help="Base channel width for RAT")
    parser.add_argument("--rat-skip-strength", type=float, default=1.0, help="Strength multiplier for decoder skip connections (0 disables skips)")
    parser.add_argument("--rat-middle-blk-num", type=int, default=6, help="Total basic layers in the middle stage (multiple of 3)")
    parser.add_argument("--rat-enc-depths", type=str, default="1,1", help="Comma-separated depths per encoder stage, e.g., '1,1'")
    parser.add_argument("--rat-dec-depths", type=str, default="1,1", help="Comma-separated depths per decoder stage, e.g., '1,1'")
    # Default mask options
    parser.add_argument("--use-default-grid-mask", action="store_true", help="Use default 14x14 region mask for training/validation")
    parser.add_argument("--grid-h", type=int, default=14, help="Grid height for default region mask")
    parser.add_argument("--grid-w", type=int, default=14, help="Grid width for default region mask")
    parser.add_argument("--seed", type=int, default=42)
    # Phase control
    parser.add_argument("--phase", type=int, default=1, choices=[1,2], help="Phase 1: reconstruction; Phase 2: classification+reconstruction")
    parser.add_argument("--phase2-ckpt", type=str, default=None, help="Checkpoint path to initialize RAT weights for Phase 2")
    parser.add_argument("--num-classes", type=int, default=14, help="Number of disease classes for CheXpert")
    parser.add_argument("--cls-weight", type=float, default=0.8, help="[Phase 2] Weight of classification loss in total loss")
    parser.add_argument("--recon-weight", type=float, default=0.2, help="[Phase 2] Weight of reconstruction loss in total loss")
    parser.add_argument("--mlp-hidden", type=int, default=512, help="Hidden dim for MLP classification head (unused when using 2x256)")
    parser.add_argument("--mlp-dropout", type=float, default=0.2, help="Dropout for MLP classification head in phase 2")

    # CheXpert-specific options
    parser.add_argument("--chexpert-train-csv", type=str, default=r"C:\Vkev\Repos\Region-Attention-Transformer-for-Medical-Image-Restoration\data\archive\train.csv",
                        help="Path to CheXpert training CSV (with 'Path' column)")
    parser.add_argument("--chexpert-val-csv", type=str, default=r"C:\Vkev\Repos\Region-Attention-Transformer-for-Medical-Image-Restoration\data\archive\valid.csv",
                        help="Path to CheXpert validation CSV (with 'Path' column)")
    parser.add_argument("--chexpert-test-csv", type=str, default=r"C:\Vkev\Repos\Region-Attention-Transformer-for-Medical-Image-Restoration\data\ChetXpert_Test\content\chexlocalize\chexlocalize\CheXpert\test_labels.csv",
                        help="Path to CheXpert test CSV (with 'Path' column)")
    parser.add_argument("--chexpert-root", type=str, default=None,
                        help="Root directory to resolve image paths from the CSV (optional if CSV has absolute paths)")
    parser.add_argument("--chexpert-train-root", type=str, default=r"C:\Vkev\Repos\Region-Attention-Transformer-for-Medical-Image-Restoration\data\archive",
                        help="Override root for train split (joined as <root>/train/<suffix> if needed)")
    parser.add_argument("--chexpert-valid-root", type=str, default=r"C:\Vkev\Repos\Region-Attention-Transformer-for-Medical-Image-Restoration\data\archive",
                        help="Override root for valid split (joined as <root>/valid/<suffix> if needed)")
    parser.add_argument("--chexpert-test-root", type=str, default=r"C:\Vkev\Repos\Region-Attention-Transformer-for-Medical-Image-Restoration\data\ChetXpert_Test\content\chexlocalize\chexlocalize\CheXpert",
                        help="Override root for test split (joined as <root>/test/<suffix> if needed)")
    parser.add_argument("--chexpert-policy", type=str, default="ones", choices=["ones", "zeroes"],
                        help="How to map uncertain labels (-1): 'ones' or 'zeroes'")
    parser.add_argument("--chexpert-exclude-support-devices", action="store_true",
                        help="Exclude samples labeled with Support Devices from all splits")
    parser.add_argument("--chexpert-mask-dir", type=str, default=None,
                        help="Directory containing offline masks (saved as .pt matching train/valid/test structure)")
    parser.add_argument("--chexpert-mask-file", type=str, default=None,
                        help="Single .pt file containing all masks (generated by gen_masks.py --single-file)")

    # Trainer options
    parser.add_argument("--devices", type=str, default="auto", help="'auto', an int like '1', or list '0,1'")
    parser.add_argument("--precision", type=str, default="32-true", help="e.g., 32-true, 16-mixed, bf16-mixed")
    parser.add_argument("--accumulate-grad-batches", type=int, default=1)
    parser.add_argument("--log-every-n-steps", type=int, default=50)
    parser.add_argument("--limit-train-batches", type=float, default=0.001)
    parser.add_argument("--limit-val-batches", type=float, default=0.01)
    parser.add_argument("--limit-test-batches", type=float, default=0.01)
    parser.add_argument("--ckpt-path", type=str, default=None, help="Resume from checkpoint path")
    parser.add_argument("--output-dir", type=str, default="outputs")
    parser.add_argument("--save-samples", action="store_true", help="Save sample reconstructions each val epoch")
    # Weights & Biases logging
    parser.add_argument("--wandb-api-key", type=str, default=None, help="W&B API key to enable online logging")
    parser.add_argument("--wandb-project", type=str, default="RAT-Medical-Image-Restoration", help="W&B project name")
    parser.add_argument("--wandb-entity", type=str, default=None, help="W&B entity (team/user)")
    parser.add_argument("--wandb-run-name", type=str, default=None, help="W&B run name")
    # Evaluation/test controls
    parser.add_argument("--eval-cls-every-epoch", action="store_true",
                        help="[Phase 2] Evaluate and print classification metrics during validation each epoch")
    parser.add_argument("--run-test", action="store_true",
                        help="Run trainer.test after training and print test metrics")

    args = parser.parse_args()

    # Seed & data
    # Use Tensor Cores on Ampere+ for faster matmul at slight precision trade-off
    try:
        torch.set_float32_matmul_precision('high')
    except Exception:
        pass
    L.seed_everything(args.seed, workers=True)
    dm = ImageDataModule(
        data_dir=args.data_dir,
        batch_size=args.batch_size,
        num_workers=args.num_workers,
        dataset=args.dataset,
        chexpert_train_csv=args.chexpert_train_csv,
        chexpert_val_csv=args.chexpert_val_csv,
        chexpert_test_csv=args.chexpert_test_csv,
        chexpert_root=args.chexpert_root,
        chexpert_train_root=args.chexpert_train_root,
        chexpert_valid_root=args.chexpert_valid_root,
        chexpert_test_root=args.chexpert_test_root,
        chexpert_policy=args.chexpert_policy,
        chexpert_exclude_support_devices=args.chexpert_exclude_support_devices,
        chexpert_mask_dir=args.chexpert_mask_dir,
        chexpert_mask_file=args.chexpert_mask_file,
        chexpert_return_labels=(args.phase == 2),
    )
    dm.prepare_data()
    dm.setup("fit")

    # Print dataset sizes for CheXpert (before limiting batches)
    if args.dataset == "chexpert":
        try:
            train_set = getattr(dm, "train_set", None)
            if hasattr(train_set, "num_rows_before_filter") and hasattr(train_set, "num_rows_after_filter"):
                before = train_set.num_rows_before_filter
                after = train_set.num_rows_after_filter
                filtered = getattr(train_set, "num_filtered_out", max(0, before - after))
                print(f"[chexpert] train rows: {after} (filtered out Support Devices: {filtered} of {before})")
            else:
                print(f"[chexpert] train rows: {len(dm.train_set)}")
        except Exception as e:
            print(f"[chexpert] Could not report train dataset size: {e}")

    # Model
    # Choose model based on dataset image shape
    # CIFAR-10 -> (3, 32, 32), Flowers102 -> (3, 224, 224)
    c = getattr(dm, "num_channels", 3)
    h, w = getattr(dm, "image_size", (32, 32))

    if (c, h, w) == (3, 224, 224):
        # Wrap RAT to auto-generate a simple region mask per image
        class RATWrapper(nn.Module):
            def __init__(self, rat_width: int = 64, rat_skip_strength: float = 1.0,
                         middle_blk_num: int = 6, enc_depths: list[int] | None = None, dec_depths: list[int] | None = None):
                super().__init__()
                if enc_depths is None:
                    enc_depths = [1, 1]
                if dec_depths is None:
                    dec_depths = [1, 1]
                self.rat = RAT(scale=1, img_channel=3, width=rat_width, middle_blk_num=middle_blk_num,
                                enc_blk_nums=enc_depths, dec_blk_nums=dec_depths, loss_fun=None,
                                skip_connection_strength=rat_skip_strength)

            def forward(self, x):
                b, _, hh, ww = x.shape
                # If batch provides masks (from DataModule), use them; else generate
                if isinstance(x, dict) and "mask" in x:
                    mask = x["mask"].to(x.device)
                else:
                    mask = generate_region_mask(b, hh, ww, grid_h=14, grid_w=14, device=x.device)
                return self.rat(x, mask)

        # Parse depths
        try:
            enc_depths = [int(x) for x in str(args.rat_enc_depths).split(',') if x.strip() != '']
        except Exception:
            enc_depths = [1, 1]
        try:
            dec_depths = [int(x) for x in str(args.rat_dec_depths).split(',') if x.strip() != '']
        except Exception:
            dec_depths = [1, 1]
        base_model = RATWrapper(rat_width=args.rat_width,
                                rat_skip_strength=args.rat_skip_strength,
                                middle_blk_num=int(args.rat_middle_blk_num),
                                enc_depths=enc_depths,
                                dec_depths=dec_depths)
    else:
        base_model = SmallVAE(latent_dim=128)

    lit = LitAutoModule(
        model=base_model,
        lr=args.lr,
        loss=args.loss,
        add_kld_if_available=args.add_kld_if_available,
        beta=args.beta,
        input_noise_std=args.input_noise_std,
        use_default_grid_mask=args.use_default_grid_mask,
        grid_h=args.grid_h,
        grid_w=args.grid_w,
        phase=args.phase,
        num_classes=args.num_classes,
        cls_weight=args.cls_weight,
        recon_weight=args.recon_weight,
        mlp_hidden=args.mlp_hidden,
        mlp_dropout=args.mlp_dropout,
    )
    # Align example input shape with dataset
    try:
        lit.example_input_array = torch.randn(2, c, h, w)
    except Exception:
        pass

    # Phase 2: optionally load checkpoint and freeze decoder
    if args.phase == 2 and hasattr(base_model, "rat"):
        if args.phase2_ckpt is not None:
            try:
                state = torch.load(args.phase2_ckpt, map_location="cpu")
                if "state_dict" in state:
                    # Lightning checkpoint
                    sd = state["state_dict"]
                    # Filter keys to RAT only
                    rat_prefixes = ["model.rat.", "rat."]
                    cleaned = {}
                    for k, v in sd.items():
                        for p in rat_prefixes:
                            if k.startswith(p):
                                cleaned[k[len(p):]] = v
                    base_model.rat.load_state_dict(cleaned, strict=False)
                else:
                    # Plain state dict of model
                    base_model.rat.load_state_dict(state, strict=False)
                print(f"[phase2] Loaded RAT weights from {args.phase2_ckpt}")
            except Exception as e:
                print(f"[phase2] Failed to load RAT weights from {args.phase2_ckpt}: {e}")
        # Freeze decoder, keep encoder+middle blocks trainable
        for p in base_model.rat.decoders.parameters():
            p.requires_grad = False
        for p in base_model.rat.ups.parameters():
            p.requires_grad = False
        for p in base_model.rat.ending.parameters():
            p.requires_grad = False
        # Keep intro, encoders, downs, middle_blks, ending trainable
        # (ending is small conv producing reconstruction; keep it to allow recon loss)

    # Callbacks
    callbacks = [
        ModelCheckpoint(monitor="loss/val/epoch", mode="min", save_top_k=1, filename="best-{epoch:02d}"),
        SegmentedBestCheckpointCallback(segments=10),
        LearningRateMonitor(logging_interval="step"),
        TQDMProgressBar(),
    ]
    if args.save_samples:
        callbacks.append(SavePredictionsCallback(out_dir=args.output_dir, num_samples=16))
    if args.phase == 2 and args.eval_cls_every_epoch:
        callbacks.append(ClassificationAccEvalCallback(segments=10, max_eval_batches=50))

    # Optional W&B logger
    wandb_logger = None
    if args.wandb_api_key:
        try:
            import wandb
            wandb.login(key=args.wandb_api_key)
            wandb_logger = WandbLogger(
                project=args.wandb_project,
                entity=args.wandb_entity if args.wandb_entity else None,
                name=args.wandb_run_name if args.wandb_run_name else None,
                save_dir=args.output_dir,
                log_model=False,
            )
            try:
                run_url = getattr(wandb_logger.experiment, 'url', None)
            except Exception:
                run_url = None
            print(f"[wandb] Logging enabled - project={wandb_logger.project_name}, entity={wandb_logger.experiment.entity if hasattr(wandb_logger, 'experiment') else args.wandb_entity}")
            if run_url:
                print(f"[wandb] run: {run_url}")
        except Exception as e:
            print(f"[wandb] Failed to initialize logging: {e}")

    trainer = L.Trainer(
        accelerator="auto",
        devices=parse_devices(args.devices),
        max_epochs=args.max_epochs,
        precision=args.precision,
        accumulate_grad_batches=args.accumulate_grad_batches,
        log_every_n_steps=args.log_every_n_steps,
        limit_train_batches=args.limit_train_batches,
        limit_val_batches=args.limit_val_batches,
        limit_test_batches=args.limit_test_batches,
        default_root_dir=args.output_dir,
        callbacks=callbacks,
        logger=wandb_logger if wandb_logger is not None else True,
    )

    trainer.fit(lit, datamodule=dm, ckpt_path=args.ckpt_path)

    # Optionally run test phase and print metrics
    if args.run_test:
        try:
            dm.setup("test")
        except Exception:
            pass
        results = trainer.test(lit, datamodule=dm)
        try:
            print("[test] metrics:")
            for i, res in enumerate(results):
                print(f"  dataloader#{i}: {res}")
        except Exception:
            pass

    # Optional quick test to export a sample grid from the test split too
    if args.save_samples:
        callbacks[-1].on_validation_epoch_end(trainer, lit)


if __name__ == "__main__":
    main()
