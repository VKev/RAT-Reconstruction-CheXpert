import argparse
from typing import Union

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader

import lightning as L
from lightning.pytorch.callbacks import ModelCheckpoint, LearningRateMonitor, TQDMProgressBar

from datamodule import ImageDataModule
from model import SmallVAE
from rat.Model_RAT import RAT
from loss import make_criterion, kl_divergence, FocalRegionLoss
from save import SavePredictionsCallback
from utils import get_recon, generate_region_mask


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
                 use_default_grid_mask: bool = True, grid_h: int = 14, grid_w: int = 14):
        super().__init__()
        self.model = model
        self.save_hyperparameters(ignore=["model"])

        self.crit = make_criterion(loss)
        self.add_kld_if_available = add_kld_if_available
        self.beta = beta
        self.example_input_array = torch.randn(2, 3, 32, 32)

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
            self.log("loss/kld", kld, prog_bar=False, on_step=True, on_epoch=True)

        self.log("loss/recon", rec, prog_bar=True, on_step=True, on_epoch=True)
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
        # Choose mask strategy
        mask = None
        use_default = bool(getattr(self.hparams, "use_default_grid_mask", True))
        if use_default:
            b, _, hh, ww = x_in.shape
            mask = generate_region_mask(
                b, hh, ww,
                grid_h=int(getattr(self.hparams, "grid_h", 14)),
                grid_w=int(getattr(self.hparams, "grid_w", 14)),
                device=x_in.device,
            )
        else:
            try:
                if torch.is_tensor(y) and y.dim() >= 2:
                    mask = y
            except Exception:
                mask = None
        out = self(x_in, mask=mask)
        loss = self._compute_loss(x, out, mask_for_loss=mask)
        self.log("loss/train", loss, prog_bar=True, on_step=True, on_epoch=True)
        return loss

    def validation_step(self, batch, batch_idx):
        x, y = batch
        with torch.no_grad():
            # Choose mask strategy
            mask = None
            use_default = bool(getattr(self.hparams, "use_default_grid_mask", True))
            if use_default:
                b, _, hh, ww = x.shape
                mask = generate_region_mask(
                    b, hh, ww,
                    grid_h=int(getattr(self.hparams, "grid_h", 14)),
                    grid_w=int(getattr(self.hparams, "grid_w", 14)),
                    device=x.device,
                )
            else:
                try:
                    if torch.is_tensor(y) and y.dim() >= 2:
                        mask = y
                except Exception:
                    mask = None
            out = self(x, mask=mask)
        loss = self._compute_loss(x, out, mask_for_loss=mask)
        self.log("loss/val", loss, prog_bar=True, on_step=False, on_epoch=True)

        # PSNR for a quick quality signal (assuming outputs are in [0,1])
        recon = get_recon(out).clamp(0, 1)
        mse = F.mse_loss(recon, x, reduction="none").mean(dim=(1,2,3))
        psnr = -10.0 * torch.log10(mse + 1e-8)
        self.log("metric/psnr", psnr.mean(), prog_bar=True, on_step=False, on_epoch=True)

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
    # Default mask options
    parser.add_argument("--use-default-grid-mask", action="store_true", help="Use default 14x14 region mask for training/validation")
    parser.add_argument("--grid-h", type=int, default=14, help="Grid height for default region mask")
    parser.add_argument("--grid-w", type=int, default=14, help="Grid width for default region mask")
    parser.add_argument("--seed", type=int, default=42)

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
    parser.add_argument("--limit-train-batches", type=float, default=0.0005)
    parser.add_argument("--limit-val-batches", type=float, default=0.01)
    parser.add_argument("--limit-test-batches", type=float, default=1.0)
    parser.add_argument("--ckpt-path", type=str, default=None, help="Resume from checkpoint path")
    parser.add_argument("--output-dir", type=str, default="outputs")
    parser.add_argument("--save-samples", action="store_true", help="Save sample reconstructions each val epoch")

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
            def __init__(self, rat_width: int = 64, rat_skip_strength: float = 1.0):
                super().__init__()
                self.rat = RAT(scale=1, img_channel=3, width=rat_width, middle_blk_num=12,
                                enc_blk_nums=[2, 2], dec_blk_nums=[2, 2], loss_fun=None,
                                skip_connection_strength=rat_skip_strength)

            def forward(self, x):
                b, _, hh, ww = x.shape
                # If batch provides masks (from DataModule), use them; else generate
                if isinstance(x, dict) and "mask" in x:
                    mask = x["mask"].to(x.device)
                else:
                    mask = generate_region_mask(b, hh, ww, grid_h=14, grid_w=14, device=x.device)
                return self.rat(x, mask)

        base_model = RATWrapper(rat_width=args.rat_width, rat_skip_strength=args.rat_skip_strength)
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
    )
    # Align example input shape with dataset
    try:
        lit.example_input_array = torch.randn(2, c, h, w)
    except Exception:
        pass

    # Callbacks
    callbacks = [
        ModelCheckpoint(monitor="loss/val", mode="min", save_top_k=1, filename="best-{epoch:02d}"),
        LearningRateMonitor(logging_interval="step"),
        TQDMProgressBar(),
    ]
    if args.save_samples:
        callbacks.append(SavePredictionsCallback(out_dir=args.output_dir, num_samples=16))

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
    )

    trainer.fit(lit, datamodule=dm, ckpt_path=args.ckpt_path)

    # Optional quick test to export a sample grid from the test split too
    if args.save_samples:
        callbacks[-1].on_validation_epoch_end(trainer, lit)


if __name__ == "__main__":
    main()
