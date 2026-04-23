import lightning as L
from pathlib import Path
import json
import logging
from src.utils.dataset import parse_split_label_mom
from src.data.base_datasets import FeaturesDataset, PooledFeaturesDataset
from torch.utils.data import DataLoader
import collections

SPLITS = {"test"}

class MoMDataModule(L.LightningDataModule):
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

            split, label = parse_split_label_mom(stem)
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

        logging.info(f"TEST Labels distribution: {collections.Counter(groups['test']['labels'])}")

        DatasetClass = PooledFeaturesDataset if use_pooled else FeaturesDataset

        self.test_dataset = DatasetClass(
            files=groups['test']['files'],
            labels=groups['test']['labels']
        )
                
    def test_dataloader(self):
        return DataLoader(
            dataset=self.test_dataset,
            batch_size=self.batch_size,
            shuffle=False,
            num_workers=self.num_workers,
        )