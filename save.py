#!/usr/bin/env python
# -*- coding: utf-8 -*-

import os
import torch
import lightning as L
from torchvision.utils import save_image

from utils import get_recon


class SegmentedBestCheckpointCallback(L.Callback):
    """Save the best checkpoint within each segment of an epoch.

    The epoch is divided into `segments` equal parts (default 10 → ~10% each).
    We track the best training loss per batch within the current segment and,
    when the segment boundary is reached, we save that segment's best model
    weights as a new checkpoint file. Then we reset for the next segment.

    Notes:
    - Monitors per-batch training loss (the value returned by training_step).
    - Checkpoints are saved alongside the main ModelCheckpoint directory if
      available; otherwise under ``<default_root_dir>/checkpoints``.
    - Filenames: ``epoch{epoch:02d}_seg{seg:02d}_best.ckpt``.
    """

    def __init__(self, segments: int = 10):
        super().__init__()
        assert segments >= 1
        self.segments = segments
        self._dirpath = None
        self._boundaries = None
        self._segment_idx = 0
        self._segment_best_value = float("inf")
        self._segment_best_state = None
        self._restored_from_ckpt = False
        self._temp_best_relpath = None  # relative file name under dirpath

    def _resolve_dirpath(self, trainer: L.Trainer):
        # Prefer the directory used by an existing ModelCheckpoint callback
        dirpath = None
        try:
            # Support list of callbacks
            for cb in trainer.callbacks:
                if hasattr(cb, "dirpath") and isinstance(getattr(cb, "dirpath"), str):
                    dirpath = cb.dirpath
                    break
        except Exception:
            dirpath = None

        if not dirpath:
            # Fallback to default_root_dir/checkpoints
            import os
            dirpath = os.path.join(trainer.default_root_dir or "outputs", "checkpoints")
        os.makedirs(dirpath, exist_ok=True)
        self._dirpath = dirpath

    def on_fit_start(self, trainer: L.Trainer, pl_module: L.LightningModule) -> None:
        self._resolve_dirpath(trainer)

    def on_train_epoch_start(self, trainer: L.Trainer, pl_module: L.LightningModule) -> None:
        # Determine number of batches for current epoch (respects limit_train_batches)
        import math
        try:
            num_batches = int(getattr(trainer, "num_training_batches", 0))
        except Exception:
            num_batches = 0
        if not num_batches:
            try:
                loader = trainer.datamodule.train_dataloader() if trainer.datamodule else trainer.train_dataloader
                num_batches = len(loader)
            except Exception:
                num_batches = 0

        num_batches = max(1, num_batches)
        # If we are not restoring from ckpt with existing boundaries, compute them now
        if not self._boundaries:
            boundaries = []
            for i in range(1, self.segments + 1):
                boundary = math.ceil(num_batches * (i / self.segments))
                boundary = max(1, min(boundary, num_batches))
                if not boundaries or boundary > boundaries[-1]:
                    boundaries.append(boundary)
            self._boundaries = boundaries
        # If not restored, this is a fresh epoch; reset tracking
        if not self._restored_from_ckpt:
            self._segment_idx = 0
            self._segment_best_value = float("inf")
            self._segment_best_state = None
        # Clear the flag so following epochs behave normally
        self._restored_from_ckpt = False

    def on_train_batch_end(self, trainer: L.Trainer, pl_module: L.LightningModule, outputs, batch, batch_idx: int) -> None:
        # Extract scalar training loss from outputs where possible
        value = None
        try:
            if isinstance(outputs, torch.Tensor):
                value = float(outputs.detach().item())
            elif isinstance(outputs, dict):
                if "loss" in outputs and isinstance(outputs["loss"], torch.Tensor):
                    value = float(outputs["loss"].detach().item())
        except Exception:
            value = None

        if value is not None and value < self._segment_best_value:
            self._segment_best_value = value
            try:
                # Store a CPU copy of the best state_dict for this segment
                self._segment_best_state = {k: v.detach().clone().cpu() for k, v in pl_module.state_dict().items()}
                # Persist to a small temp file so we can resume mid-segment
                try:
                    epoch = int(trainer.current_epoch)
                    seg_num = int(self._segment_idx + 1)
                    tmp_name = f".tmp_epoch{epoch:02d}_seg{seg_num:02d}_best_state.pt"
                    tmp_path = os.path.join(self._dirpath, tmp_name)
                    if trainer.is_global_zero:
                        torch.save(self._segment_best_state, tmp_path)
                    try:
                        trainer.strategy.barrier()
                    except Exception:
                        pass
                    self._temp_best_relpath = tmp_name
                except Exception:
                    # If temp persist fails, keep in memory only
                    pass
            except Exception:
                self._segment_best_state = None

        if self._boundaries is None or self._segment_idx >= len(self._boundaries):
            return

        # 1-indexed batch count within epoch
        current_batch_in_epoch = int(batch_idx) + 1
        if current_batch_in_epoch >= self._boundaries[self._segment_idx]:
            # Reached the boundary; save the best state for this segment
            self._save_segment_best(trainer, pl_module)
            # Prepare for next segment
            self._segment_idx += 1
            self._segment_best_value = float("inf")
            self._segment_best_state = None

    def _save_segment_best(self, trainer: L.Trainer, pl_module: L.LightningModule):
        import os
        if self._segment_best_state is None:
            return

        epoch = int(trainer.current_epoch)
        seg_num = int(self._segment_idx + 1)
        filename = f"epoch{epoch:02d}_seg{seg_num:02d}_best.ckpt"
        filepath = os.path.join(self._dirpath, filename)

        # Synchronize processes to avoid race conditions
        try:
            trainer.strategy.barrier()
        except Exception:
            pass

        # Temporarily swap in the best weights to save a full checkpoint, then restore
        try:
            current_state = {k: v.detach().clone() for k, v in pl_module.state_dict().items()}
            if trainer.is_global_zero:
                with torch.no_grad():
                    pl_module.load_state_dict(self._segment_best_state, strict=True)
                    trainer.save_checkpoint(filepath)
                    print(f"[SegmentedBestCheckpoint] Saved {filepath} (loss={self._segment_best_value:.6f})")
            # Barrier to ensure rank0 completes save before others move on
            try:
                trainer.strategy.barrier()
            except Exception:
                pass
        finally:
            # Restore original state on all ranks
            with torch.no_grad():
                pl_module.load_state_dict(current_state, strict=True)
        # Clean up temp file after successful save
        try:
            if self._temp_best_relpath:
                tmp_path = os.path.join(self._dirpath, self._temp_best_relpath)
                if trainer.is_global_zero and os.path.isfile(tmp_path):
                    os.remove(tmp_path)
                self._temp_best_relpath = None
            try:
                trainer.strategy.barrier()
            except Exception:
                pass
        except Exception:
            pass

    # -------------
    # Checkpoint I/O
    # -------------
    def on_save_checkpoint(self, trainer: L.Trainer, pl_module: L.LightningModule) -> dict:
        # Do NOT store _segment_best_state to avoid duplicating full model weights
        return {
            "segments": int(self.segments),
            "boundaries": list(self._boundaries or []),
            "segment_idx": int(self._segment_idx),
            "segment_best_value": float(self._segment_best_value),
            "temp_best_relpath": self._temp_best_relpath,
        }

    def on_load_checkpoint(self, trainer: L.Trainer, pl_module: L.LightningModule, callback_state: dict) -> None:
        try:
            self.segments = int(callback_state.get("segments", self.segments))
            b = callback_state.get("boundaries", None)
            self._boundaries = list(b) if isinstance(b, (list, tuple)) else None
            self._segment_idx = int(callback_state.get("segment_idx", 0))
            self._segment_best_value = float(callback_state.get("segment_best_value", float("inf")))
            # Try to restore temp best state if available
            self._segment_best_state = None
            self._temp_best_relpath = callback_state.get("temp_best_relpath", None)
            if self._temp_best_relpath and self._dirpath:
                tmp_path = os.path.join(self._dirpath, self._temp_best_relpath)
                if os.path.isfile(tmp_path):
                    try:
                        obj = torch.load(tmp_path, map_location="cpu")
                        if isinstance(obj, dict):
                            self._segment_best_state = obj
                    except Exception:
                        self._segment_best_state = None
            self._restored_from_ckpt = True
        except Exception:
            # Fall back to fresh state if anything goes wrong
            self._boundaries = None
            self._segment_idx = 0
            self._segment_best_value = float("inf")
            self._segment_best_state = None
            self._restored_from_ckpt = False



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

