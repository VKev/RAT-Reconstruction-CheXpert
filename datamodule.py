#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
Generic LightningDataModule with support for CIFAR-10 (32x32) and Flowers102 (resized to 224x224).
Pass the dataset name via the constructor.
"""

from typing import Optional, Tuple, List

import lightning as L
import torchvision
from torch.utils.data import DataLoader, random_split
from torchvision import transforms as T
import torch
import os
import pandas as pd
from PIL import Image


class ImageDataModule(L.LightningDataModule):
    def __init__(self, data_dir: str = "./data", batch_size: int = 128, num_workers: int = 4, val_split: int = 5000,
                 dataset: str = "cifar10",
                 # CheXpert-specific args
                 chexpert_train_csv: Optional[str] = None,
                 chexpert_val_csv: Optional[str] = None,
                 chexpert_test_csv: Optional[str] = None,
                 chexpert_root: Optional[str] = None,
                 chexpert_train_root: Optional[str] = None,
                 chexpert_valid_root: Optional[str] = None,
                 chexpert_test_root: Optional[str] = None,
                 chexpert_policy: str = "ones",
                 chexpert_exclude_support_devices: bool = False,
                 chexpert_only_support_devices: bool = False,
                 chexpert_mask_dir: Optional[str] = None,
                 chexpert_mask_file: Optional[str] = None,
                 chexpert_return_labels: bool = False,
                 ):
        super().__init__()
        self.data_dir = data_dir
        self.batch_size = batch_size
        self.num_workers = num_workers
        self.val_split = val_split
        self.dataset = dataset.lower()
        self.chexpert_train_csv = chexpert_train_csv
        self.chexpert_val_csv = chexpert_val_csv
        self.chexpert_test_csv = chexpert_test_csv
        self.chexpert_root = chexpert_root
        self.chexpert_train_root = chexpert_train_root
        self.chexpert_valid_root = chexpert_valid_root
        self.chexpert_test_root = chexpert_test_root
        self.chexpert_policy = chexpert_policy
        self.chexpert_exclude_support_devices = chexpert_exclude_support_devices
        self.chexpert_only_support_devices = chexpert_only_support_devices
        self.chexpert_mask_dir = chexpert_mask_dir
        self.chexpert_mask_file = chexpert_mask_file
        self.chexpert_return_labels = chexpert_return_labels

        # Keep pixels in [0,1]; models should output [0,1] (we use Sigmoid in the default VAE).
        if self.dataset == "cifar10":
            self.image_size: Tuple[int, int] = (32, 32)
            self.num_channels: int = 3
            self.transform = T.Compose([
                T.ToTensor(),
            ])
        elif self.dataset == "flowers102":
            # Standardize to 224x224x3
            self.image_size = (224, 224)
            self.num_channels = 3
            self.transform = T.Compose([
                T.Resize(self.image_size, antialias=True),
                T.ToTensor(),
            ])
        elif self.dataset == "chexpert":
            # Standardize to 224x224x3
            self.image_size = (224, 224)
            self.num_channels = 3
            self.transform = T.Compose([
                T.Resize(self.image_size, antialias=True),
                T.ToTensor(),
            ])
        else:
            raise ValueError(f"Unsupported dataset '{self.dataset}'. Use 'cifar10', 'flowers102', or 'chexpert'.")

    def prepare_data(self) -> None:
        if self.dataset == "cifar10":
            torchvision.datasets.CIFAR10(self.data_dir, train=True, download=True)
            torchvision.datasets.CIFAR10(self.data_dir, train=False, download=True)
        elif self.dataset == "flowers102":
            # Download all splits once
            torchvision.datasets.Flowers102(self.data_dir, split='train', download=True)
            torchvision.datasets.Flowers102(self.data_dir, split='val', download=True)
            torchvision.datasets.Flowers102(self.data_dir, split='test', download=True)
        elif self.dataset == "chexpert":
            # No download; expects CSV paths provided
            missing = []
            if not self.chexpert_train_csv:
                missing.append("chexpert_train_csv")
            if not self.chexpert_val_csv:
                missing.append("chexpert_val_csv")
            if not self.chexpert_test_csv:
                missing.append("chexpert_test_csv")
            if missing:
                raise ValueError("For 'chexpert', please provide: " + ", ".join(missing))

    def setup(self, stage: Optional[str] = None) -> None:
        if stage in (None, "fit"):
            if self.dataset == "cifar10":
                full_train = torchvision.datasets.CIFAR10(self.data_dir, train=True, transform=self.transform, download=False)
                lengths = [len(full_train) - self.val_split, self.val_split]
                self.train_set, self.val_set = random_split(full_train, lengths, generator=torch.Generator().manual_seed(42))
            elif self.dataset == "flowers102":
                self.train_set = torchvision.datasets.Flowers102(self.data_dir, split='train', transform=self.transform, download=False)
                self.val_set = torchvision.datasets.Flowers102(self.data_dir, split='val', transform=self.transform, download=False)
            elif self.dataset == "chexpert":
                self.train_set = CheXpertDataset(
                    csv_path=self.chexpert_train_csv,
                    root_dir=self.chexpert_root,
                    train_root=self.chexpert_train_root or self.chexpert_root,
                    valid_root=self.chexpert_valid_root or self.chexpert_root,
                    test_root=self.chexpert_test_root,
                    split="train",
                    transform=self.transform,
                    policy=self.chexpert_policy,
                    exclude_support_devices=self.chexpert_exclude_support_devices,
                    include_only_support_devices=self.chexpert_only_support_devices,
                    mask_dir=self.chexpert_mask_dir,
                    mask_file=self.chexpert_mask_file,
                    return_labels=self.chexpert_return_labels,
                )
                self.val_set = CheXpertDataset(
                    csv_path=self.chexpert_val_csv,
                    root_dir=self.chexpert_root,
                    train_root=self.chexpert_train_root or self.chexpert_root,
                    valid_root=self.chexpert_valid_root or self.chexpert_root,
                    test_root=self.chexpert_test_root,
                    split="valid",
                    transform=self.transform,
                    policy=self.chexpert_policy,
                    exclude_support_devices=self.chexpert_exclude_support_devices,
                    include_only_support_devices=self.chexpert_only_support_devices,
                    mask_dir=self.chexpert_mask_dir,
                    mask_file=self.chexpert_mask_file,
                    return_labels=self.chexpert_return_labels,
                )

        if stage in (None, "test", "validate"):
            if self.dataset == "cifar10":
                self.test_set = torchvision.datasets.CIFAR10(self.data_dir, train=False, transform=self.transform, download=False)
            elif self.dataset == "flowers102":
                self.test_set = torchvision.datasets.Flowers102(self.data_dir, split='test', transform=self.transform, download=False)
            elif self.dataset == "chexpert":
                self.test_set = CheXpertDataset(
                    csv_path=self.chexpert_test_csv,
                    root_dir=self.chexpert_root,
                    train_root=self.chexpert_train_root or self.chexpert_root,
                    valid_root=self.chexpert_valid_root or self.chexpert_root,
                    test_root=self.chexpert_test_root or self.chexpert_root,
                    split="test",
                    transform=self.transform,
                    policy=self.chexpert_policy,
                    exclude_support_devices=self.chexpert_exclude_support_devices,
                    include_only_support_devices=self.chexpert_only_support_devices,
                    mask_dir=self.chexpert_mask_dir,
                    mask_file=self.chexpert_mask_file,
                    return_labels=self.chexpert_return_labels,
                )

    def train_dataloader(self) -> DataLoader:
        return DataLoader(self.train_set, batch_size=self.batch_size, shuffle=True, num_workers=self.num_workers, pin_memory=True, persistent_workers=self.num_workers > 0)

    def val_dataloader(self) -> DataLoader:
        return DataLoader(self.val_set, batch_size=self.batch_size, shuffle=False, num_workers=self.num_workers, pin_memory=True, persistent_workers=self.num_workers > 0)

    def test_dataloader(self) -> DataLoader:
        return DataLoader(self.test_set, batch_size=self.batch_size, shuffle=False, num_workers=self.num_workers, pin_memory=True, persistent_workers=self.num_workers > 0)



class CheXpertDataset(torch.utils.data.Dataset):
    """
    Minimal CheXpert dataset wrapper for image restoration/reconstruction tasks.
    - Expects a CSV with a 'Path' column and label columns (ignored here).
    - Returns (image_tensor, dummy_label) where dummy_label is 0 to fit (x, _) training API.
    - If 'root_dir' is provided, image paths are resolved relative to it.
    - Uncertain labels policy: 'ones' (default) or 'zeroes' to replace -1.
    """

    LABEL_COLUMNS: List[str] = [
        "No Finding", "Enlarged Cardiomediastinum", "Cardiomegaly", "Lung Opacity",
        "Lung Lesion", "Edema", "Consolidation", "Pneumonia", "Atelectasis",
        "Pneumothorax", "Pleural Effusion", "Pleural Other", "Fracture", "Support Devices",
    ]

    def __init__(self, csv_path: str, root_dir: Optional[str] = None, transform: Optional[T.Compose] = None,
                 policy: str = "ones", split: Optional[str] = None,
                 train_root: Optional[str] = None, valid_root: Optional[str] = None, test_root: Optional[str] = None,
                 exclude_support_devices: bool = False, include_only_support_devices: bool = False,
                 mask_dir: Optional[str] = None, mask_file: Optional[str] = None,
                 return_labels: bool = False):
        super().__init__()
        self.csv_path = csv_path
        self.root_dir = root_dir
        self.transform = transform
        self.policy = policy
        self.split = (split or "").lower() if split is not None else None
        self.train_root = train_root
        self.valid_root = valid_root
        self.test_root = test_root
        self.exclude_support_devices = exclude_support_devices
        self.include_only_support_devices = include_only_support_devices
        self.num_rows_before_filter: int = 0
        self.num_rows_after_filter: int = 0
        self.num_filtered_out: int = 0
        self.mask_dir = mask_dir
        self.mask_file = mask_file
        self.mask_mapping = None
        self.return_labels = return_labels

        df = pd.read_csv(csv_path)
        self.num_rows_before_filter = len(df)

        # Normalize uncertain and missing labels, though labels are not used downstream
        cols_present = [c for c in self.LABEL_COLUMNS if c in df.columns]
        if cols_present:
            if policy == "ones":
                df[cols_present] = df[cols_present].replace(-1, 1)
            elif policy == "zeroes":
                df[cols_present] = df[cols_present].replace(-1, 0)
            df[cols_present] = df[cols_present].fillna(0)

        # Optionally filter by Support Devices (after applying uncertainty policy)
        # Precedence: include_only_support_devices > exclude_support_devices
        if self.include_only_support_devices and ("Support Devices" in df.columns):
            try:
                before = len(df)
                df = df[df["Support Devices"].fillna(0) == 1]
                after = len(df)
                self.num_rows_after_filter = after
                self.num_filtered_out = max(0, before - after)
            except Exception:
                self.num_rows_after_filter = len(df)
                self.num_filtered_out = 0
        elif self.exclude_support_devices and ("Support Devices" in df.columns):
            try:
                before = len(df)
                df = df[df["Support Devices"].fillna(0) == 0]
                after = len(df)
                self.num_rows_after_filter = after
                self.num_filtered_out = max(0, before - after)
            except Exception:
                # If filter fails for any reason, fall back to unfiltered
                self.num_rows_after_filter = len(df)
                self.num_filtered_out = 0
        else:
            self.num_rows_after_filter = len(df)
            self.num_filtered_out = 0

        # Keep relative paths for mask lookup
        self.rel_paths: List[str] = df["Path"].astype(str).tolist()
        self.paths: List[str] = self.rel_paths
        # Optionally keep label matrix for classification
        self.labels_tensor = None
        cols_present = [c for c in self.LABEL_COLUMNS if c in df.columns]
        if self.return_labels and cols_present:
            import numpy as _np
            arr = df[cols_present].values.astype(_np.float32)
            self.labels_tensor = torch.from_numpy(arr)
        # If a single mask file is provided, load once
        if self.mask_file is not None and os.path.exists(self.mask_file):
            try:
                # If JSONL (RLE), parse as lightweight single-file
                if self.mask_file.lower().endswith('.jsonl'):
                    import json
                    self.mask_mapping = {}
                    with open(self.mask_file, 'r', encoding='utf-8') as f:
                        for line in f:
                            line = line.strip()
                            if not line:
                                continue
                            rec = json.loads(line)
                            key = str(rec.get('path', '')).replace('\\', '/')
                            self.mask_mapping[key] = rec.get('rle')
                else:
                    obj = torch.load(self.mask_file, map_location="cpu")
                    mapping = obj.get("masks", obj)
                    self.mask_mapping = {str(k).replace("\\", "/"): torch.as_tensor(v) for k, v in mapping.items()}
            except Exception:
                self.mask_mapping = None

    def _resolve_path(self, rel_or_abs: str) -> str:
        # If absolute (Windows or POSIX) and exists, return
        try:
            if (os.path.isabs(rel_or_abs) and os.path.exists(rel_or_abs)):
                return rel_or_abs
        except Exception:
            pass

        # Normalize slashes for matching
        norm = rel_or_abs.replace("\\", "/")

        # Determine roots for train/valid/test
        train_root = self.train_root or self.root_dir
        valid_root = self.valid_root or self.root_dir
        test_root = self.test_root or self.root_dir

        # Match Kaggle-like CSV content, mirroring the original logic
        if norm.startswith("train/") or "chexpert-v1.0-small/train" in norm.lower():
            suffix = norm.split("train/", 1)[-1]
            base = train_root
            if base:
                candidate = os.path.join(base, "train", suffix)
                if os.path.exists(candidate):
                    return candidate
        elif norm.startswith("valid/") or "chexpert-v1.0-small/valid" in norm.lower():
            suffix = norm.split("valid/", 1)[-1]
            base = valid_root
            if base:
                candidate = os.path.join(base, "valid", suffix)
                if os.path.exists(candidate):
                    return candidate
        elif norm.startswith("test/") or "/chexpert/test" in norm.lower():
            suffix = norm.split("test/", 1)[-1]
            base = test_root
            if base:
                candidate = os.path.join(base, "test", suffix)
                if os.path.exists(candidate):
                    return candidate

        # If a generic root_dir is provided, try simple join
        if self.root_dir is not None:
            candidate = os.path.join(self.root_dir, rel_or_abs)
            if os.path.exists(candidate):
                return candidate

        # Fallback to original string
        return rel_or_abs

    def __len__(self) -> int:
        return len(self.paths)

    def __getitem__(self, idx: int):
        rel_path = self.rel_paths[idx]
        path = self._resolve_path(rel_path)
        if not os.path.exists(path):
            raise FileNotFoundError(f"CheXpert image not found: {path}")
        img = Image.open(path).convert("RGB")
        if self.transform is not None:
            img = self.transform(img)
        # Try load offline mask if provided
        mask_tensor = None
        # Try global mask mapping first
        if self.mask_mapping is not None and self.split in ("train", "valid", "test"):
            try:
                norm_key = rel_path.replace("\\", "/")
                if not (norm_key.startswith("train/") or norm_key.startswith("valid/") or norm_key.startswith("test/")):
                    norm_key = f"{self.split}/{norm_key}"
                m = self.mask_mapping.get(norm_key, None)
                if m is None:
                    m = self.mask_mapping.get(norm_key.split("/", 1)[-1], None)
                if m is not None:
                    # If value is RLE dict, decode
                    if isinstance(m, dict) and 'rle' in m and 'shape' in m:
                        from utils import rle_decode_binary
                        import torch as _torch
                        mask_np = rle_decode_binary(m)
                        mask_tensor = _torch.from_numpy(mask_np.astype('int64'))
                    elif isinstance(m, list):
                        # Multi-class mask stored as nested list
                        import numpy as np
                        mask_np = np.array(m, dtype='int64')
                        mask_tensor = torch.from_numpy(mask_np)
                    else:
                        mask_tensor = torch.as_tensor(m)
            except Exception:
                mask_tensor = None
        if mask_tensor is None and self.mask_dir is not None:
            try:
                norm = rel_path.replace("\\", "/")
                # ensure split prefix present
                if not (norm.startswith("train/") or norm.startswith("valid/") or norm.startswith("test/")):
                    # best effort: try to detect any split keyword and reconstruct
                    if "train/" in norm:
                        norm = norm.split("train/", 1)[-1]
                        norm = "train/" + norm
                    elif "valid/" in norm:
                        norm = norm.split("valid/", 1)[-1]
                        norm = "valid/" + norm
                    elif "test/" in norm:
                        norm = norm.split("test/", 1)[-1]
                        norm = "test/" + norm
                mask_path = os.path.join(self.mask_dir, norm)
                base, _ = os.path.splitext(mask_path)
                mask_path = base + ".pt"
                if os.path.exists(mask_path):
                    mask_tensor = torch.load(mask_path)
                    if isinstance(mask_tensor, dict) and "mask" in mask_tensor:
                        mask_tensor = mask_tensor["mask"]
                    if not torch.is_tensor(mask_tensor):
                        mask_tensor = torch.as_tensor(mask_tensor)
                    if mask_tensor.dim() == 3 and mask_tensor.size(0) == 1:
                        mask_tensor = mask_tensor.squeeze(0)
                    if mask_tensor.dtype != torch.long:
                        mask_tensor = mask_tensor.long()
                    # Resize to match image if needed
                    h_img, w_img = img.shape[-2:]
                    if mask_tensor.shape[-2:] != (h_img, w_img):
                        mask_tensor = torch.nn.functional.interpolate(mask_tensor.unsqueeze(0).unsqueeze(0).float(),
                                                                       size=(h_img, w_img), mode="nearest").squeeze(0).squeeze(0).long()
            except Exception:
                mask_tensor = None
        # Final normalization for either source (mapping or dir): ensure type/shape
        if mask_tensor is not None:
            if not torch.is_tensor(mask_tensor):
                mask_tensor = torch.as_tensor(mask_tensor)
            if mask_tensor.dim() == 3 and mask_tensor.size(0) == 1:
                mask_tensor = mask_tensor.squeeze(0)
            if mask_tensor.dtype != torch.long:
                mask_tensor = mask_tensor.long()
            h_img, w_img = img.shape[-2:]
            if mask_tensor.shape[-2:] != (h_img, w_img):
                mask_tensor = torch.nn.functional.interpolate(mask_tensor.unsqueeze(0).unsqueeze(0).float(),
                                                               size=(h_img, w_img), mode="nearest").squeeze(0).squeeze(0).long()
        if self.return_labels and self.labels_tensor is not None:
            y_vec = self.labels_tensor[idx]
            # return both classification labels and optional mask if present
            if mask_tensor is None:
                return img, {"labels": y_vec}
            else:
                return img, {"labels": y_vec, "mask": mask_tensor}
        else:
            if mask_tensor is None:
                # Fallback: dummy label
                return img, torch.tensor(0, dtype=torch.long)
            else:
                return img, mask_tensor
