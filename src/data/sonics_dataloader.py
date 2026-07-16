import lightning as L
from pathlib import Path
import json
from src.utils.dataset import parse_split_label
from src.data.base_datasets import FeaturesDataset, PooledFeaturesDataset
from torch.utils.data import DataLoader
import collections
import logging

SPLITS = {"train", "valid", "test"}


def _detect_dataset_mode(data: list[dict]) -> str:
    if not data:
        raise ValueError("Empty index.json")
    item = data[0]
    if item.get("pooled"):
        return "pooled"
    if item.get("wav2vec") and item.get("mert"):
        return "raw"
    raise ValueError(f"Unknown index entry keys: {list(item.keys())}")


# this is sonic datamodule to take precomuted features!
class SonicsDataModule(L.LightningDataModule):
    def __init__(self, data_dir: str, batch_size: int, num_workers: int, transform: object):
        super().__init__()
        self.data_dir = Path(data_dir)
        self.batch_size = batch_size
        self.num_workers = num_workers
        self.transform = transform
        self.dataset_mode = "raw"

    def setup(self, stage):
        groups = {k: {"files": [], "labels": []} for k in SPLITS}

        index_file = list(self.data_dir.rglob("index.json"))[0]
        if not index_file:
            raise ValueError("No index file found!")

        with open(str(index_file), "r") as file:
            data = json.load(file)

        self.dataset_mode = _detect_dataset_mode(data)
        mode_labels = {
            "pooled": "pooled (PooledFeaturesDataset)",
            "raw": "raw (FeaturesDataset)",
        }
        logging.info(f"Dataset mode: {mode_labels[self.dataset_mode]}")

        for item in data:
            stem = item.get("stem", None)
            if not stem:
                continue

            split, label = parse_split_label(stem)
            if split not in groups:
                continue

            if self.dataset_mode == "pooled":
                pooled_path = item.get("pooled", None)
                if not pooled_path:
                    continue
                groups[split]["files"].append(pooled_path)
            else:
                w2v_path = item.get("wav2vec", None)
                mert_path = item.get("mert", None)
                if not all([w2v_path, mert_path]):
                    continue
                groups[split]["files"].append((w2v_path, mert_path))

            groups[split]["labels"].append(label)

        logging.info("TRAIN Labels distribution:", collections.Counter(groups["train"]["labels"]))
        logging.info("VALID Labels distribution:", collections.Counter(groups["valid"]["labels"]))

        if self.dataset_mode == "pooled":
            self.training_dataset = PooledFeaturesDataset(
                files=groups["train"]["files"],
                labels=groups["train"]["labels"],
                transform=self.transform,
            )
            self.valid_dataset = PooledFeaturesDataset(
                files=groups["valid"]["files"],
                labels=groups["valid"]["labels"],
            )
            self.test_dataset = PooledFeaturesDataset(
                files=groups["test"]["files"],
                labels=groups["test"]["labels"],
            )
        else:
            self.training_dataset = FeaturesDataset(
                files=groups["train"]["files"],
                labels=groups["train"]["labels"],
                transform=self.transform,
            )
            self.valid_dataset = FeaturesDataset(
                files=groups["valid"]["files"],
                labels=groups["valid"]["labels"],
            )
            self.test_dataset = FeaturesDataset(
                files=groups["test"]["files"],
                labels=groups["test"]["labels"],
            )

    def _loader_kwargs(self) -> dict:
        return {
            "batch_size": self.batch_size,
            "shuffle": False,
            "num_workers": self.num_workers,
        }

    def train_dataloader(self):
        kwargs = self._loader_kwargs()
        kwargs["shuffle"] = True
        if self.num_workers > 0:
            kwargs["pin_memory"] = True
            kwargs["persistent_workers"] = True
        return DataLoader(dataset=self.training_dataset, **kwargs)

    def val_dataloader(self):
        kwargs = self._loader_kwargs()
        if self.num_workers > 0:
            kwargs["pin_memory"] = True
        return DataLoader(dataset=self.valid_dataset, **kwargs)

    def test_dataloader(self):
        return DataLoader(dataset=self.test_dataset, **self._loader_kwargs())


class SonicsSpectttraLoader(L.LightningDataModule):
    def __init__(self, data_dir: str, batch_size: int, num_workers: int):
        super().__init__()
        self.data_dir = Path(data_dir)
        self.batch_size = batch_size
        self.num_workers = num_workers

    def setup(self, stage):
        groups = {k: {"files": [], "labels": []} for k in SPLITS}

        for file in self.data_dir.iterdir():
            stem = file.stem
            elements = stem.strip("_")

    def test_dataloader(self):
        return DataLoader(
            dataset=self.test_dataset,
            batch_size=self.batch_size,
            shuffle=False,
            num_workers=self.num_workers,
        )
