#!/usr/bin/env python
# -*- coding: utf-8 -*-

import os
import torch
import lightning as L
from torchvision.utils import save_image

from utils import get_recon


class SavePredictionsCallback(L.Callback):
    def __init__(self, out_dir: str = "outputs", num_samples: int = 16):
        super().__init__()
        self.out_dir = out_dir
        self.num_samples = num_samples
        os.makedirs(self.out_dir, exist_ok=True)

    def on_validation_epoch_end(self, trainer: L.Trainer, pl_module: L.LightningModule) -> None:
        dm = trainer.datamodule
        if dm is None:
            return
        dm.setup("validate")
        loader = dm.val_dataloader()
        try:
            batch = next(iter(loader))
        except StopIteration:
            return
        x, _ = batch
        x = x.to(pl_module.device)[: self.num_samples]
        with torch.no_grad():
            out = pl_module(x)
            recon = get_recon(out).clamp(0, 1)

        # Interleave original and recon
        grid = torch.cat([x, recon], dim=0)
        save_path = os.path.join(self.out_dir, f"epoch{trainer.current_epoch:03d}_recon.png")
        save_image(grid, save_path, nrow=self.num_samples, padding=2)
        trainer.strategy.barrier()  # avoid multi-process write conflicts
        if trainer.is_global_zero:
            print(f"[SaveReconstructionsCallback] Wrote {save_path}")

