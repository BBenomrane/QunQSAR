from __future__ import annotations

import math
import os
from dataclasses import dataclass
from typing import Optional, Tuple

import torch
from torch.utils.data import Dataset, DataLoader


class TextDataset(Dataset):
    """Simple next-token prediction dataset over raw byte IDs.

    Builds overlapping sequences of length `block_size` where the target is the
    input shifted by one position (teacher forcing).
    """

    def __init__(self, data_bytes: bytes, block_size: int) -> None:
        if block_size < 2:
            raise ValueError("block_size must be >= 2")
        if len(data_bytes) <= block_size:
            raise ValueError(
                f"data is too small: {len(data_bytes)} bytes for block_size {block_size}"
            )
        self.block_size = int(block_size)
        # Store as a single LongTensor for efficient slicing
        self.data = torch.frombuffer(bytearray(data_bytes), dtype=torch.uint8).to(torch.long)

    def __len__(self) -> int:
        # Number of possible starting indices for sequences of length block_size
        return self.data.size(0) - self.block_size

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, torch.Tensor]:
        chunk = self.data[idx : idx + self.block_size + 1]
        x = chunk[:-1]
        y = chunk[1:]
        return x, y


@dataclass
class LoaderConfig:
    file_path: str
    block_size: int = 256
    batch_size: int = 64
    val_fraction: float = 0.1
    num_workers: int = 0
    seed: int = 1337
    shuffle: bool = True


def _split_bytes(data_bytes: bytes, val_fraction: float) -> tuple[bytes, bytes]:
    n = len(data_bytes)
    split = max(0, min(n, int(n * (1.0 - val_fraction))))
    train_bytes = data_bytes[:split]
    val_bytes = data_bytes[split:]
    # Ensure both splits are adequate for at least one sample
    if len(train_bytes) <= 1024 or len(val_bytes) <= 1024:
        # If validation is too small, fold it back to train; users can adjust val_fraction
        val_bytes = data_bytes[-max(1024, len(data_bytes) // 10) :]
        train_bytes = data_bytes[:-len(val_bytes)]
    return train_bytes, val_bytes


def create_dataloaders(cfg: LoaderConfig) -> tuple[DataLoader, DataLoader]:
    if not os.path.isfile(cfg.file_path):
        raise FileNotFoundError(f"No such file: {cfg.file_path}")
    with open(cfg.file_path, "rb") as f:
        data_bytes = f.read()

    train_bytes, val_bytes = _split_bytes(data_bytes, cfg.val_fraction)

    train_ds = TextDataset(train_bytes, cfg.block_size)
    val_ds = TextDataset(val_bytes, cfg.block_size)

    generator = torch.Generator()
    generator.manual_seed(cfg.seed)

    train_loader = DataLoader(
        train_ds,
        batch_size=cfg.batch_size,
        shuffle=cfg.shuffle,
        num_workers=cfg.num_workers,
        pin_memory=True,
        drop_last=True,
        generator=generator,
    )

    val_loader = DataLoader(
        val_ds,
        batch_size=cfg.batch_size,
        shuffle=False,
        num_workers=cfg.num_workers,
        pin_memory=True,
        drop_last=True,
    )

    return train_loader, val_loader
