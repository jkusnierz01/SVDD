import lightning as L
from pathlib import Path
import json
from src.utils.dataset import parse_split_label
from src.data.base_datasets import FeaturesDataset, PooledFeaturesDataset
from torch.utils.data import DataLoader
import collections
import logging

SPLITS = {"train", "valid", "test"}

# this is sonic datamodule to take precomuted features!
class SonicsDataModule(L.LightningDataModule):
    def __init__(self, data_dir:str, batch_size: int, num_workers: int, transform: object):
        super().__init__()
        self.data_dir = Path(data_dir)
        self.batch_size = batch_size
        self.num_workers = num_workers
        self.transform = transform
        
    
    def setup(self, stage):
        groups = {k: {"files": [], "labels": []} for k in SPLITS}
        
        index_file = list(self.data_dir.rglob("index.json"))[0]
        if not index_file:
            raise ValueError("No index file found!")
        
        with open(str(index_file), "r") as file:
            data = json.load(file)
            
        use_pooled = data and "pooled" in data[0]
        logging.info(f"Dataset mode: {'pooled (PooledFeaturesDataset)' if use_pooled else 'raw (FeaturesDataset)'}")

        for item in data:
            stem = item.get("stem", None)
            if not stem:
                continue

            split, label = parse_split_label(stem)
            if split not in groups:
                continue

            if use_pooled:
                pooled_path = item.get("pooled", None)
                if not pooled_path:
                    continue
                groups[split]['files'].append(pooled_path)
            else:
                w2v_path = item.get("wav2vec", None)
                mert_path = item.get("mert", None)
                if not all([w2v_path, mert_path]):
                    continue
                groups[split]['files'].append((w2v_path, mert_path))

            groups[split]['labels'].append(label)

        logging.info("TRAIN Labels distribution:", collections.Counter(groups['train']['labels']))
        logging.info("VALID Labels distribution:", collections.Counter(groups['valid']['labels']))

        DatasetClass = PooledFeaturesDataset if use_pooled else FeaturesDataset

        self.training_dataset = DatasetClass(
            files=groups['train']['files'],
            labels=groups['train']['labels'],
            transform=self.transform
        )

        self.valid_dataset = DatasetClass(
            files=groups['valid']['files'],
            labels=groups['valid']['labels']
        )

        self.test_dataset = DatasetClass(
            files=groups['test']['files'],
            labels=groups['test']['labels']
        )
                
            
    def train_dataloader(self):
        return DataLoader(
            dataset=self.training_dataset,
            batch_size=self.batch_size,
            shuffle=True,
            num_workers=self.num_workers,
            pin_memory=True,
            persistent_workers=True
        )
        
    def val_dataloader(self):
        return DataLoader(
            dataset=self.valid_dataset,
            batch_size=self.batch_size,
            shuffle=False,
            num_workers=self.num_workers,
            pin_memory=True,
        )


    def test_dataloader(self):
        return DataLoader(
            dataset=self.test_dataset,
            batch_size=self.batch_size,
            shuffle=False,
            num_workers=self.num_workers,
        )
        
        
class SonicsSpectttraLoader(L.LightningDataModule):
    def __init__(self, data_dir:str, batch_size: int, num_workers: int):
        super().__init__()
        self.data_dir = Path(data_dir)
        self.batch_size = batch_size
        self.num_workers = num_workers
        
    def setup(self, stage):
        groups = {k: {"files": [], "labels": []} for k in SPLITS}
            
        for file in self.data_dir.iterdir():
            stem = file.stem
            elements = stem.strip("_")
            
    #     stem = item.get("stem", None)
    #     w2v_path = item.get("wav2vec", None)
    #     mert_path = item.get("mert", None)
        
    #     if not all([w2v_path, mert_path, stem]):
    #         continue
        
    #     split, label = parse_split_label(stem)
        
    #     if split in groups:
    #         groups[split]['files'].append((w2v_path, mert_path))
    #         groups[split]['labels'].append(label)
    
    def test_dataloader(self):
        return DataLoader(
            dataset=self.test_dataset,
            batch_size=self.batch_size,
            shuffle=False,
            num_workers=self.num_workers
        )