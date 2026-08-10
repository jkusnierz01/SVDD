"""CLAM benchmark dataset and Lightning DataModule."""

from __future__ import annotations

import collections
import json
import logging
from pathlib import Path

import lightning as L
import torch
from torch.utils.data import DataLoader, Dataset

from src.preprocessing.clam_align import CLAM_LAYER_DIM, CLAM_NUM_LAYERS
from src.utils.dataset import parse_split_label

SPLITS = {"train", "valid", "test"}


class CLAMDataset(Dataset):
    """Loads pre-aligned CLAM all-layer features from index.json entries."""

    def __init__(
        self,
        index_path: str | Path,
        split: str | None = None,
        *,
        strict: bool = True,
    ):
        super().__init__()
        index_path = Path(index_path)
        with open(index_path, encoding="utf-8") as f:
            data = json.load(f)

        self.entries: list[dict] = []
        for item in data:
            stem = item.get("stem")
            w2v_path = item.get("clam_w2v")
            mert_path = item.get("clam_mert")
            if not stem or not w2v_path or not mert_path:
                continue

            item_split, label = parse_split_label(stem)
            if split is not None and item_split != split:
                continue
            if item_split not in SPLITS:
                continue

            self.entries.append(
                {
                    "stem": stem,
                    "clam_w2v": w2v_path,
                    "clam_mert": mert_path,
                    "label": label,
                }
            )

        self.strict = strict
        if not self.entries:
            raise ValueError(f"No CLAM entries found in {index_path} (split={split})")

    def __len__(self) -> int:
        return len(self.entries)

    def __getitem__(self, index: int):
        entry = self.entries[index]
        # Keep fp16 on CPU to cut RAM (~2x); Trainer uses precision=16-mixed on GPU.
        w2v = torch.load(entry["clam_w2v"], map_location="cpu", weights_only=True)
        mert = torch.load(entry["clam_mert"], map_location="cpu", weights_only=True)

        if self.strict:
            expected = (w2v.shape[0], CLAM_NUM_LAYERS, CLAM_LAYER_DIM)
            if w2v.shape != expected:
                raise ValueError(
                    f"W2V shape {tuple(w2v.shape)} != expected {expected} for {entry['stem']}"
                )
            if mert.shape != w2v.shape:
                raise ValueError(
                    f"MERT/W2V mismatch for {entry['stem']}: "
                    f"w2v={tuple(w2v.shape)} mert={tuple(mert.shape)}"
                )

        label = torch.tensor(entry["label"], dtype=torch.long)
        return (w2v, mert), label


class ClamDataModule(L.LightningDataModule):
    def __init__(
        self,
        data_dir: str,
        batch_size: int,
        num_workers: int,
        transform: object = None,
    ):
        super().__init__()
        self.data_dir = Path(data_dir)
        self.batch_size = batch_size
        self.num_workers = num_workers
        self.transform = transform  # reserved; CLAM features are not augmented here

    def _index_path(self) -> Path:
        matches = list(self.data_dir.rglob("index.json"))
        if not matches:
            raise ValueError(f"No index.json under {self.data_dir}")
        return matches[0]

    def setup(self, stage: str | None = None) -> None:
        index_path = self._index_path()
        self.training_dataset = CLAMDataset(index_path, split="train")
        self.valid_dataset = CLAMDataset(index_path, split="valid")
        self.test_dataset = CLAMDataset(index_path, split="test")

        logging.info(
            "CLAM splits: train=%d valid=%d test=%d",
            len(self.training_dataset),
            len(self.valid_dataset),
            len(self.test_dataset),
        )
        train_labels = [e["label"] for e in self.training_dataset.entries]
        valid_labels = [e["label"] for e in self.valid_dataset.entries]
        logging.info("TRAIN labels: %s", collections.Counter(train_labels))
        logging.info("VALID labels: %s", collections.Counter(valid_labels))

    def _loader_kwargs(self) -> dict:
        kwargs = {
            "batch_size": self.batch_size,
            "shuffle": False,
            "num_workers": self.num_workers,
        }
        if self.num_workers > 0:
            kwargs["prefetch_factor"] = 1
        return kwargs

    def train_dataloader(self) -> DataLoader:
        kwargs = self._loader_kwargs()
        kwargs["shuffle"] = True
        if self.num_workers > 0:
            kwargs["pin_memory"] = True
        return DataLoader(self.training_dataset, **kwargs)

    def val_dataloader(self) -> DataLoader:
        kwargs = self._loader_kwargs()
        if self.num_workers > 0:
            kwargs["pin_memory"] = True
        return DataLoader(self.valid_dataset, **kwargs)

    def test_dataloader(self) -> DataLoader:
        return DataLoader(self.test_dataset, **self._loader_kwargs())
