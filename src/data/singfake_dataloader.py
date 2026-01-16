import lightning as L
from torch.utils.data import DataLoader
from pathlib import Path
from .base_datasets import LongAudioDataset, SingFakeShortDataset
from .base_datasets import collate_fn

LABEL_MAP = {"bonafide": 1, "spoof": 0}
SPLITS = {"Training", "Validation", "T01", "T02", "T03", "T04"}


def base_stem(stem: str) -> str:
    return stem.split("__seg", 1)[0]

def parse_split_label(stem: str):
    base = base_stem(stem)
    parts = base.split("_")
    split = parts[0]
    label = parts[-1].lower()
    return split, LABEL_MAP[label]


class SingFakeDataModule(L.LightningDataModule):
    def __init__(
        self,
        data_dir: str,
        batch_size: int,
        num_workers: int,
        drop_last: bool = False,
        loader_type: str = "precomputed",  # wav or spectrogram
        use_collate: bool = False,
        **dataset_kwargs,
    ):
        super().__init__()
        self.data_dir = data_dir
        self.batch_size = batch_size
        self.num_workers = num_workers
        self.loader_type = loader_type
        self.drop_last = drop_last
        self.is_mixture = False
        self.use_collate = use_collate

        self.dataset_kwargs = dataset_kwargs

    def setup(self, stage):
        groups = {k: {"files": [], "labels": []} for k in SPLITS}

        if self.loader_type == "precomputed":
            data_path = Path(self.data_dir)
        else:
            data_path = Path(self.data_dir) / (
                "mixtures" if self.is_mixture else "vocals"
            )

        for full_filename in data_path.iterdir():
            if not full_filename.is_file():
                continue
            filename = full_filename.stem
            try:
                key, label_int = parse_split_label(filename)
            except Exception as e:
                print(f"Bad filename: {filename} ({e})")
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
            self.training_dataset = SingFakeShortDataset(
                groups["Training"]["files"],
                groups["Training"]["labels"],
                target_sr=16000,
                mode=self.loader_type,
                **self.dataset_kwargs,
            )

            self.validation_dataset = SingFakeShortDataset(
                groups["Validation"]["files"],
                groups["Validation"]["labels"],
                target_sr=16000,
                mode=self.loader_type,
                **self.dataset_kwargs,
            )
            self.test_t01 = SingFakeShortDataset(
                groups["T01"]["files"],
                groups["T01"]["labels"],
                target_sr=16000,
                mode=self.loader_type,
                **self.dataset_kwargs,
            )
            self.test_t02 = SingFakeShortDataset(
                groups["T02"]["files"],
                groups["T02"]["labels"],
                target_sr=16000,
                mode=self.loader_type,
                **self.dataset_kwargs,
            )
            self.test_t03 = SingFakeShortDataset(
                groups["T03"]["files"],
                groups["T03"]["labels"],
                target_sr=16000,
                mode=self.loader_type,
                **self.dataset_kwargs,
            )
            self.test_t04 = SingFakeShortDataset(
                groups["T04"]["files"],
                groups["T04"]["labels"],
                target_sr=16000,
                mode=self.loader_type,
                **self.dataset_kwargs,
            )

    def train_dataloader(self):
        return DataLoader(
            self.training_dataset,
            batch_size=self.batch_size,
            shuffle=True,
            drop_last=self.drop_last,
            num_workers=self.num_workers,
            collate_fn=collate_fn if self.use_collate else None,
            pin_memory=True,
        )

    def val_dataloader(self):
        return DataLoader(
            self.validation_dataset,
            batch_size=self.batch_size,
            num_workers=self.num_workers,
            collate_fn=collate_fn if self.use_collate else None,
        )

    def test_dataloader(self):
        return [
            DataLoader(
                self.test_t01,
                batch_size=self.batch_size,
                num_workers=self.num_workers,
                collate_fn=collate_fn if self.use_collate else None,
            ),
            DataLoader(
                self.test_t02,
                batch_size=self.batch_size,
                num_workers=self.num_workers,
                collate_fn=collate_fn if self.use_collate else None,
            ),
            DataLoader(
                self.test_t03,
                batch_size=self.batch_size,
                num_workers=self.num_workers,
                collate_fn=collate_fn if self.use_collate else None,
            ),
            DataLoader(
                self.test_t04,
                batch_size=self.batch_size,
                num_workers=self.num_workers,
                collate_fn=collate_fn if self.use_collate else None,
            ),
        ]
