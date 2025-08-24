#!/usr/bin/env python
# -*- coding: utf-8 -*-

import os
import torch
import torch.nn.functional as F
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
        # If we were restored, try to reload temp best state from disk now that dirpath is known
        if self._restored_from_ckpt and self._temp_best_relpath and self._dirpath:
            try:
                tmp_path = os.path.join(self._dirpath, self._temp_best_relpath)
                if os.path.isfile(tmp_path):
                    obj = torch.load(tmp_path, map_location="cpu")
                    if isinstance(obj, dict):
                        self._segment_best_state = obj
            except Exception:
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
    def on_save_checkpoint(self, trainer: L.Trainer, pl_module: L.LightningModule, checkpoint: dict) -> None:
        # Lightning v2 passes `checkpoint` to mutate in-place. We keep state in state_dict instead.
        # Intentionally no-op to avoid duplicating weights/state here.
        return None

    def state_dict(self) -> dict:
        # Do NOT store _segment_best_state to avoid duplicating full model weights
        return {
            "segments": int(self.segments),
            "boundaries": list(self._boundaries or []),
            "segment_idx": int(self._segment_idx),
            "segment_best_value": float(self._segment_best_value),
            "temp_best_relpath": self._temp_best_relpath,
        }

    def load_state_dict(self, state_dict: dict) -> None:
        try:
            self.segments = int(state_dict.get("segments", self.segments))
            b = state_dict.get("boundaries", None)
            self._boundaries = list(b) if isinstance(b, (list, tuple)) else None
            self._segment_idx = int(state_dict.get("segment_idx", 0))
            self._segment_best_value = float(state_dict.get("segment_best_value", float("inf")))
            # Defer loading of temp best state until dirpath is resolved
            self._segment_best_state = None
            self._temp_best_relpath = state_dict.get("temp_best_relpath", None)
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


class ClassificationAccEvalCallback(L.Callback):
    """Evaluate classification accuracy on the test set every 1/segments of an epoch (Phase 2 only)."""

    def __init__(self, segments: int = 10, max_eval_batches: int | None = None):
        super().__init__()
        assert segments >= 1
        self.segments = segments
        self.max_eval_batches = max_eval_batches
        self._boundaries = None
        self._segment_idx = 0
        # Metrics accumulators
        self._sum_correct = 0.0
        self._sum_count = 0.0
        self._auroc_scores = []
        self._f1_scores = []

    def on_train_epoch_start(self, trainer: L.Trainer, pl_module: L.LightningModule) -> None:
        # compute batch boundaries for this epoch
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
        boundaries = []
        for i in range(1, self.segments + 1):
            b = math.ceil(num_batches * (i / self.segments))
            b = max(1, min(b, num_batches))
            if not boundaries or b > boundaries[-1]:
                boundaries.append(b)
        self._boundaries = boundaries
        self._segment_idx = 0
        self._sum_correct = 0.0
        self._sum_count = 0.0
        self._auroc_scores = []
        self._f1_scores = []

    @torch.no_grad()
    def _eval_cls_acc(self, trainer: L.Trainer, pl_module: L.LightningModule) -> dict | None:
        # Only valid for phase 2 with a classification head
        if not hasattr(pl_module, "phase") or int(getattr(pl_module, "phase", 1)) != 2:
            return None
        # Accept either attribute name
        head = getattr(pl_module, "_cls_head", None)
        if head is None:
            head = getattr(pl_module, "cls_head", None)
        dm = trainer.datamodule
        if dm is None:
            return None
        try:
            dm.setup("test")
            loader = dm.test_dataloader()
        except Exception:
            return None
        if loader is None:
            return None

        device = pl_module.device
        was_training = pl_module.training
        pl_module.eval()

        total_correct = 0.0
        total_count = 0.0
        all_logits = []
        all_labels = []

        # Helper to get mask
        from utils import generate_region_mask

        max_batches = self.max_eval_batches or len(loader)

        pbar = None
        use_pbar = False
        if trainer.is_global_zero:
            try:
                from tqdm import tqdm as _tqdm
                pbar = _tqdm(total=max_batches, desc="Phase2 test acc", leave=False)
                use_pbar = True
            except Exception:
                pbar = None
                use_pbar = False

        for bi, (x, y) in enumerate(loader):
            if bi >= max_batches:
                break
            try:
                labels = None
                mask = None
                if isinstance(y, dict):
                    labels = y.get("labels", None)
                    mask = y.get("mask", None)
                # Build default mask if missing
                if mask is None:
                    b, _, hh, ww = x.shape
                    mask = generate_region_mask(b, hh, ww, grid_h=int(getattr(pl_module.hparams, "grid_h", 14)), grid_w=int(getattr(pl_module.hparams, "grid_w", 14)), device=device)

                x = x.to(device)
                if labels is None:
                    # if labels absent, skip batch
                    continue
                labels = labels.to(device).float()

                # Extract features from RAT; prefer pre+post concat if available
                rat = getattr(pl_module.model, "rat", None)
                if rat is None:
                    continue
                feats = None
                if hasattr(rat, "extract_pre_post_features"):
                    pre_f, post_f = rat.extract_pre_post_features(x, mask)
                    feats = torch.cat([pre_f, post_f], dim=1)
                elif hasattr(rat, "extract_middle_features"):
                    feats = rat.extract_middle_features(x, mask)
                else:
                    continue
                # Ensure head exists and matches channel count
                if head is None:
                    make_head = getattr(pl_module, "_make_cls_head", None)
                    if callable(make_head):
                        head = make_head(int(feats.size(1)), prefix="[eval]", device=device)
                        try:
                            pl_module._cls_head = head
                        except Exception:
                            pass
                    else:
                        return None
                logits = head(feats)
                probs = torch.sigmoid(logits)
                preds = (probs > 0.5).float()

                if labels.dim() == 1:
                    labels = labels.unsqueeze(0)
                # Align shapes to [B, C]
                if labels.shape[1] != preds.shape[1]:
                    if labels.shape[1] > preds.shape[1]:
                        labels = labels[:, : preds.shape[1]]
                    else:
                        pad = torch.zeros(labels.shape[0], preds.shape[1] - labels.shape[1], device=labels.device)
                        labels = torch.cat([labels, pad], dim=1)
                labels = labels.expand_as(preds)
                correct = (preds == labels).float().sum().item()
                count = float(preds.numel())
                total_correct += correct
                total_count += count
                # Collect for AUROC/F1
                all_logits.append(probs.detach().cpu())
                all_labels.append(labels.detach().cpu())
            except Exception:
                continue
            finally:
                if use_pbar and pbar is not None:
                    try:
                        pbar.update(1)
                    except Exception:
                        pass

        if use_pbar and pbar is not None:
            try:
                pbar.close()
            except Exception:
                pass

        if was_training:
            pl_module.train()

        if total_count == 0:
            return None
        acc = float(total_correct / total_count)
        # Compute AUROC and F1 (macro) if possible
        auroc_macro = None
        f1_macro = None
        try:
            import numpy as _np
            from sklearn.metrics import roc_auc_score, f1_score
            y_true = torch.cat(all_labels, dim=0).numpy()
            y_prob = torch.cat(all_logits, dim=0).numpy()
            # Handle classes with constant labels by ignoring errors
            with _np.errstate(all='ignore'):
                aurocs = []
                for ci in range(y_true.shape[1]):
                    yt = y_true[:, ci]
                    yp = y_prob[:, ci]
                    if (yt.max() == yt.min()):
                        continue
                    try:
                        aurocs.append(roc_auc_score(yt, yp))
                    except Exception:
                        pass
                auroc_macro = float(_np.mean(aurocs)) if len(aurocs) > 0 else None
            # F1 with threshold 0.5
            y_pred = (y_prob > 0.5).astype('float32')
            f1_macro = float(f1_score(y_true, y_pred, average='macro', zero_division=0))
        except Exception:
            pass
        return {"acc": acc, "auroc_macro": auroc_macro, "f1_macro": f1_macro}

    def on_train_batch_end(self, trainer: L.Trainer, pl_module: L.LightningModule, outputs, batch, batch_idx: int) -> None:
        if self._boundaries is None or self._segment_idx >= len(self._boundaries):
            return
        current = int(batch_idx) + 1
        if current >= self._boundaries[self._segment_idx]:
            metrics = self._eval_cls_acc(trainer, pl_module)
            if metrics is not None:
                try:
                    pl_module.log("metric/test/acc", metrics["acc"], prog_bar=True, on_step=True, on_epoch=False)
                    if metrics.get("auroc_macro", None) is not None:
                        pl_module.log("metric/test/auroc_macro", metrics["auroc_macro"], prog_bar=False, on_step=True, on_epoch=False)
                    if metrics.get("f1_macro", None) is not None:
                        pl_module.log("metric/test/f1_macro", metrics["f1_macro"], prog_bar=False, on_step=True, on_epoch=False)
                except Exception:
                    pass
                if trainer.is_global_zero:
                    msg = f"[Eval@{current} batches] Test acc: {metrics['acc']:.4f}"
                    if metrics.get("auroc_macro", None) is not None:
                        msg += f", AUROC-macro: {metrics['auroc_macro']:.4f}"
                    if metrics.get("f1_macro", None) is not None:
                        msg += f", F1-macro: {metrics['f1_macro']:.4f}"
                    print(msg)
            self._segment_idx += 1
