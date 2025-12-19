import lightning as L
from torch.utils.data import DataLoader
from pathlib import Path
import os
from .base_datasets import LongAudioDataset, SingFakeShortDataModule
from .base_datasets import collate_fn


LABEL_MAP = {"bonafide": 1, "spoof": 0}


class SingFakeDataModule(L.LightningDataModule):
    def __init__(
        self,
        data_dir: str,
        batch_size: int,
        num_workers: int,
        drop_last: bool,
        loader_type: str = "precomputed",  # wav or spectrogram
        **dataset_kwargs,
    ):
        super().__init__()
        self.data_dir = data_dir
        self.batch_size = batch_size
        self.num_workers = num_workers
        self.loader_type = loader_type
        self.drop_last = drop_last
        self.is_mixture = False

        self.dataset_kwargs = dataset_kwargs

    def setup(self, stage):
        keys = ["Training", "Validation", "T01", "T02", "T03", "T04"]
        groups = {k: {"files": [], "labels": []} for k in keys}

        data_path = Path(self.data_dir)
        for full_filename in data_path.iterdir():
            if not full_filename.is_file():
                continue
            filename = full_filename.stem
            parts = filename.split("_")
            if len(parts) == 0:
                continue
            key = parts[0]
            label = parts[-1].lower()
            label_int = LABEL_MAP.get(label, -1)
            if label_int == -1:
                print(f"Unknown label in file: {filename}, last part: {label}")
                continue
            if key in groups:
                groups[key]["files"].append(full_filename)
                groups[key]["labels"].append(label_int)
        if self.loader_type == "precomputed":
            self.training_dataset = LongAudioDataset(
                groups["Training"]["files"],
                groups["Training"]["labels"],
                **self.dataset_kwargs,
            )
            self.validation_dataset = LongAudioDataset(
                groups["Validation"]["files"],
                groups["Validation"]["labels"],
                **self.dataset_kwargs,
            )
            self.test_t01 = LongAudioDataset(
                groups["T01"]["files"],
                groups["T01"]["labels"],
                **self.dataset_kwargs,
            )
            self.test_t02 = LongAudioDataset(
                groups["T02"]["files"],
                groups["T02"]["labels"],
                **self.dataset_kwargs,
            )
            self.test_t03 = LongAudioDataset(
                groups["T03"]["files"],
                groups["T03"]["labels"],
                **self.dataset_kwargs,
            )
            self.test_t04 = LongAudioDataset(
                groups["T04"]["files"],
                groups["T04"]["labels"],
                **self.dataset_kwargs,
            )
        else:
            self.training_dataset = SingFakeShortDataModule(
                data_dir=self.data_dir,
                is_mixture=self.is_mixture,
                target_sr=16000,
                mode=self.loader_type,
            )

    def train_dataloader(self):
        return DataLoader(
            self.training_dataset,
            batch_size=self.batch_size,
            shuffle=True,
            drop_last=self.drop_last,
            num_workers=self.num_workers,
            collate_fn=collate_fn,
            pin_memory=True,
        )

    def val_dataloader(self):
        return DataLoader(
            self.validation_dataset,
            batch_size=self.batch_size,
            num_workers=self.num_workers,
            collate_fn=collate_fn,
        )

    def test_t01_dataloader(self):
        return DataLoader(
            self.test_t01, batch_size=self.batch_size, num_workers=self.num_workers
        )

    def test_t02_dataloader(self):
        return DataLoader(
            self.test_t02, batch_size=self.batch_size, num_workers=self.num_workers
        )

    def test_t03_dataloader(self):
        return DataLoader(
            self.test_t03, batch_size=self.batch_size, num_workers=self.num_workers
        )

    def test_t04_dataloader(self):
        return DataLoader(
            self.test_t04, batch_size=self.batch_size, num_workers=self.num_workers
        )
