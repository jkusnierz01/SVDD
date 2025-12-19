import lightning as L
from torch.utils.data import DataLoader
from pathlib import Path
from .base_datasets import AudioDataset
from .base_datasets import collate_fn

LABEL_MAP = {"bonafide": 1, "spoof": 0, "deepfake": 0}


class WildSVDDDataModule(L.LightningDataModule):
    def __init__(
        self, data_dir: str, batch_size: int, num_workers: int, **dataset_kwargs
    ):
        super().__init__()
        self.data_dir = data_dir
        self.batch_size = batch_size
        self.num_workers = num_workers
        self.dataset_kwargs = dataset_kwargs

    def setup(self, stage):
        keys = ["Training", "TestA", "TestB"]
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
        self.test_a_dataset = AudioDataset(
            groups["TestA"]["files"],
            groups["TestA"]["labels"],
            **self.dataset_kwargs,
        )
        self.test_b_dataset = AudioDataset(
            groups["TestB"]["files"],
            groups["TestB"]["labels"],
            **self.dataset_kwargs,
        )

    def train_dataloader(self):
        return DataLoader(
            self.training_dataset,
            batch_size=self.batch_size,
            shuffle=True,
            num_workers=self.num_workers,
            collate_fn=collate_fn,
            pin_memory=True,
        )

    def test_dataloader(self):
        return [
            DataLoader(
                self.test_a_dataset,
                batch_size=self.batch_size,
                num_workers=self.num_workers,
                collate_fn=collate_fn,
                pin_memory=True,
            ),
            DataLoader(
                self.test_b_dataset,
                batch_size=self.batch_size,
                num_workers=self.num_workers,
                collate_fn=collate_fn,
                pin_memory=True,
            ),
        ]
