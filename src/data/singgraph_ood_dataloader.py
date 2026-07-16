"""Test-only SingGraph DataModule for OOD datasets (M6/MoM)."""

from __future__ import annotations

import collections
import json
import logging
from pathlib import Path

import lightning as L
import torch
from torch.utils.data import DataLoader, Dataset

from src.utils.dataset import parse_split_label

SINGGRAPH_NUM_WINDOWS = 30
SINGGRAPH_FEAT_DIM = 1024


class SingGraphOODDataset(Dataset):
    """Loads pre-aligned SingGraph vocals/instrumental features from test entries."""

    def __init__(
        self,
        index_path: str | Path,
        *,
        strict: bool = True,
        num_windows: int = SINGGRAPH_NUM_WINDOWS,
        feat_dim: int = SINGGRAPH_FEAT_DIM,
    ):
        super().__init__()
        index_path = Path(index_path)
        with open(index_path, encoding="utf-8") as f:
            data = json.load(f)

        self.num_windows = num_windows
        self.feat_dim = feat_dim
        self.entries: list[dict] = []
        for item in data:
            stem = item.get("stem")
            vocals_path = item.get("singgraph_vocals")
            instrumental_path = item.get("singgraph_instrumental")
            if not stem or not vocals_path or not instrumental_path:
                continue

            split, label = parse_split_label(stem)
            if split != "test":
                continue

            self.entries.append(
                {
                    "stem": stem,
                    "singgraph_vocals": vocals_path,
                    "singgraph_instrumental": instrumental_path,
                    "label": label,
                }
            )

        self.strict = strict
        if not self.entries:
            raise ValueError(f"No SingGraph OOD test entries found in {index_path}")

    def __len__(self) -> int:
        return len(self.entries)

    def __getitem__(self, index: int):
        entry = self.entries[index]
        vocals = torch.load(entry["singgraph_vocals"], map_location="cpu", weights_only=True)
        instrumental = torch.load(
            entry["singgraph_instrumental"], map_location="cpu", weights_only=True
        )

        if self.strict:
            self._check_shape(vocals, entry["stem"], "vocals")
            self._check_shape(instrumental, entry["stem"], "instrumental")
            if vocals.shape != instrumental.shape:
                raise ValueError(
                    f"Vocals/instrumental mismatch for {entry['stem']}: "
                    f"vocals={tuple(vocals.shape)} instrumental={tuple(instrumental.shape)}"
                )

        label = torch.tensor(entry["label"], dtype=torch.long)
        return (vocals, instrumental), label

    def _check_shape(self, tensor: torch.Tensor, stem: str, name: str) -> None:
        if tensor.ndim != 3:
            raise ValueError(f"{name} rank {tensor.ndim} != 3 for {stem}: {tuple(tensor.shape)}")
        if tensor.shape[0] != self.num_windows or tensor.shape[2] != self.feat_dim:
            raise ValueError(
                f"{name} shape {tuple(tensor.shape)} != "
                f"({self.num_windows}, T, {self.feat_dim}) for {stem}"
            )


class SingGraphOODDataModule(L.LightningDataModule):
    """Hydra-friendly test-only DataModule for SingGraph OOD evaluation."""

    def __init__(
        self,
        data_dir: str,
        batch_size: int,
        num_workers: int,
        transform: object = None,
        num_windows: int = SINGGRAPH_NUM_WINDOWS,
        feat_dim: int = SINGGRAPH_FEAT_DIM,
    ):
        super().__init__()
        self.data_dir = Path(data_dir)
        self.batch_size = batch_size
        self.num_workers = num_workers
        self.transform = transform
        self.num_windows = num_windows
        self.feat_dim = feat_dim

    def _index_path(self) -> Path:
        matches = list(self.data_dir.rglob("index.json"))
        if not matches:
            raise ValueError(f"No index.json under {self.data_dir}")
        return matches[0]

    def setup(self, stage: str | None = None) -> None:
        self.test_dataset = SingGraphOODDataset(
            self._index_path(),
            num_windows=self.num_windows,
            feat_dim=self.feat_dim,
        )
        labels = [e["label"] for e in self.test_dataset.entries]
        logging.info("SingGraph OOD TEST labels: %s", collections.Counter(labels))

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
