#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
Generic LightningDataModule with support for CIFAR-10 (32x32) and Flowers102 (resized to 224x224).
Pass the dataset name via the constructor.
"""

from typing import Optional, Tuple

import lightning as L
import torchvision
from torch.utils.data import DataLoader, random_split
from torchvision import transforms as T
import torch


class ImageDataModule(L.LightningDataModule):
    def __init__(self, data_dir: str = "./data", batch_size: int = 128, num_workers: int = 4, val_split: int = 5000,
                 dataset: str = "cifar10"):
        super().__init__()
        self.data_dir = data_dir
        self.batch_size = batch_size
        self.num_workers = num_workers
        self.val_split = val_split
        self.dataset = dataset.lower()

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
        else:
            raise ValueError(f"Unsupported dataset '{self.dataset}'. Use 'cifar10' or 'flowers102'.")

    def prepare_data(self) -> None:
        if self.dataset == "cifar10":
            torchvision.datasets.CIFAR10(self.data_dir, train=True, download=True)
            torchvision.datasets.CIFAR10(self.data_dir, train=False, download=True)
        elif self.dataset == "flowers102":
            # Download all splits once
            torchvision.datasets.Flowers102(self.data_dir, split='train', download=True)
            torchvision.datasets.Flowers102(self.data_dir, split='val', download=True)
            torchvision.datasets.Flowers102(self.data_dir, split='test', download=True)

    def setup(self, stage: Optional[str] = None) -> None:
        if stage in (None, "fit"):
            if self.dataset == "cifar10":
                full_train = torchvision.datasets.CIFAR10(self.data_dir, train=True, transform=self.transform, download=False)
                lengths = [len(full_train) - self.val_split, self.val_split]
                self.train_set, self.val_set = random_split(full_train, lengths, generator=torch.Generator().manual_seed(42))
            elif self.dataset == "flowers102":
                self.train_set = torchvision.datasets.Flowers102(self.data_dir, split='train', transform=self.transform, download=False)
                self.val_set = torchvision.datasets.Flowers102(self.data_dir, split='val', transform=self.transform, download=False)

        if stage in (None, "test", "validate"):
            if self.dataset == "cifar10":
                self.test_set = torchvision.datasets.CIFAR10(self.data_dir, train=False, transform=self.transform, download=False)
            elif self.dataset == "flowers102":
                self.test_set = torchvision.datasets.Flowers102(self.data_dir, split='test', transform=self.transform, download=False)

    def train_dataloader(self) -> DataLoader:
        return DataLoader(self.train_set, batch_size=self.batch_size, shuffle=True, num_workers=self.num_workers, pin_memory=True, persistent_workers=self.num_workers > 0)

    def val_dataloader(self) -> DataLoader:
        return DataLoader(self.val_set, batch_size=self.batch_size, shuffle=False, num_workers=self.num_workers, pin_memory=True, persistent_workers=self.num_workers > 0)

    def test_dataloader(self) -> DataLoader:
        return DataLoader(self.test_set, batch_size=self.batch_size, shuffle=False, num_workers=self.num_workers, pin_memory=True, persistent_workers=self.num_workers > 0)

