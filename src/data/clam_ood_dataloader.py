"""Test-only CLAM DataModule for OOD datasets (M6/MoM)."""

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


class CLAMOODDataset(Dataset):
    """Loads pre-aligned CLAM all-layer features from index.json test entries."""

    def __init__(self, index_path: str | Path, *, strict: bool = True):
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

            split, label = parse_split_label(stem)
            if split != "test":
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
            raise ValueError(f"No CLAM OOD test entries found in {index_path}")

    def __len__(self) -> int:
        return len(self.entries)

    def __getitem__(self, index: int):
        entry = self.entries[index]
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


class ClamOODDataModule(L.LightningDataModule):
    """Hydra-friendly test-only DataModule for CLAM OOD evaluation."""

    def __init__(self, data_dir: str, batch_size: int, num_workers: int, transform: object = None):
        super().__init__()
        self.data_dir = Path(data_dir)
        self.batch_size = batch_size
        self.num_workers = num_workers
        self.transform = transform

    def _index_path(self) -> Path:
        matches = list(self.data_dir.rglob("index.json"))
        if not matches:
            raise ValueError(f"No index.json under {self.data_dir}")
        return matches[0]

    def setup(self, stage: str | None = None) -> None:
        self.test_dataset = CLAMOODDataset(self._index_path())
        labels = [e["label"] for e in self.test_dataset.entries]
        logging.info("CLAM OOD TEST labels: %s", collections.Counter(labels))

    def test_dataloader(self) -> DataLoader:
        kwargs = {
            "dataset": self.test_dataset,
            "batch_size": self.batch_size,
            "shuffle": False,
            "num_workers": self.num_workers,
        }
        if self.num_workers > 0:
            kwargs["prefetch_factor"] = 1
            kwargs["pin_memory"] = True
        return DataLoader(**kwargs)
