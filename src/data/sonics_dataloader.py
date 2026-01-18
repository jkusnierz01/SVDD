import lightning as L
from pathlib import Path
import json
from src.utils.dataset import parse_split_label
from src.data.base_datasets import FeaturesDataset
from torch.utils.data import DataLoader

SPLITS = {"train", "valid", "test"}

# this is sonic datamodule to take precomuted features!
class SonicsDataModule(L.LightningDataModule):
    def __init__(self, data_dir:str, batch_size: int, num_workers: int):
        super().__init__()
        self.data_dir = Path(data_dir)
        self.batch_size = batch_size
        self.num_workers = num_workers
        
    
    def setup(self, stage):
        groups = {k: {"files": [], "labels": []} for k in SPLITS}
        
        index_file = list(self.data_dir.rglob("index.json"))[0]
        if not index_file:
            raise ValueError("No index file found!")
        
        with open(index_file, "r") as file:
            data = json.load(file)
            
        for item in data:
            stem = item.get("stem", None)
            w2v_path = item.get("wav2vec", None)
            mert_path = item.get("mert", None)
            
            if not all([w2v_path, mert_path, stem]):
                continue
            
            split, label = parse_split_label(stem)
            
            if split in groups:
                groups[split]['files'].append((w2v_path, mert_path))
                groups[split]['labels'].append(label)
                
        self.training_dataset = FeaturesDataset(
            files=groups['train']['files'],
            labels=groups['train']['labels']
        )
            
        self.valid_dataset = FeaturesDataset(
            files=groups['valid']['files'],
            labels=groups['valid']['labels']
        )
            
        self.valid_dataset = FeaturesDataset(
            files=groups['test']['files'],
            labels=groups['test']['labels']
        )
                
            
    def train_dataloader(self):
        return DataLoader(
            dataset=self.training_dataset,
            batch_size=self.batch_size,
            shuffle=True,
            num_workers=self.num_workers
        )
        
    def val_dataloader(self):
        return DataLoader(
            dataset=self.valid_dataset,
            batch_size=self.batch_size,
            shuffle=False,
            num_workers=self.num_workers
        )
        
        
    def test_dataloader(self):
        return DataLoader(
            dataset=self.test_dataset,
            batch_size=self.batch_size,
            shuffle=False,
            num_workers=self.num_workers
        )