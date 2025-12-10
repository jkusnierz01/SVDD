import lightning as L
from torch.utils.data import DataLoader
from pathlib import Path
from .base import AudioDataset
from torch.nn.utils.rnn import pad_sequence
import torch

LABEL_MAP = {"bonafide": 1, "spoof": 0}


def collate_fn(batch):
    # batch to lista krotek (mel_spec, label)
    features, labels = zip(*batch)

    # Zapisz oryginalne długości (potrzebne dla Mamby/Maskowania!)
    lengths = torch.tensor([f.size(0) for f in features])

    # Paduj sekwencje do najdłuższej w batchu (batch_first=True -> [Batch, Time, Feat])
    features_padded = pad_sequence(features, batch_first=True, padding_value=0.0)

    labels = torch.stack(labels)

    return features_padded, labels, lengths


class SingFakeDataModule(L.LightningDataModule):
    def __init__(
        self, data_dir: str, batch_size: int, num_workers: int, **dataset_kwargs
    ):
        super().__init__()
        self.data_dir = data_dir
        self.batch_size = batch_size
        self.num_workers = num_workers
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

        self.training_dataset = AudioDataset(
            groups["Training"]["files"],
            groups["Training"]["labels"],
            **self.dataset_kwargs,
        )
        self.validation_dataset = AudioDataset(
            groups["Validation"]["files"],
            groups["Validation"]["labels"],
            **self.dataset_kwargs,
        )
        self.test_t01 = AudioDataset(
            groups["T01"]["files"],
            groups["T01"]["labels"],
            **self.dataset_kwargs,
        )
        self.test_t02 = AudioDataset(
            groups["T02"]["files"],
            groups["T02"]["labels"],
            **self.dataset_kwargs,
        )
        self.test_t03 = AudioDataset(
            groups["T03"]["files"],
            groups["T03"]["labels"],
            **self.dataset_kwargs,
        )
        self.test_t04 = AudioDataset(
            groups["T04"]["files"],
            groups["T04"]["labels"],
            **self.dataset_kwargs,
        )

    def train_dataloader(self):
        return DataLoader(
            self.training_dataset,
            batch_size=self.batch_size,
            shuffle=True,
            num_workers=self.num_workers,
            collate_fn=collate_fn,
            pin_memory=True
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
