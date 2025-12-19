import lightning as L
from torch.utils.data import DataLoader, ConcatDataset
from .base_datasets import AudioDataset, collate_fn
from .wildsvdd_dataloader import WildSVDDDataModule
from .singfake_dataloader import SingFakeDataModule
import torch



class CombinedAudioDataModule(L.LightningDataModule):
    def __init__(
        self, 
        wild_data_dir: str, 
        sing_data_dir: str, 
        batch_size: int, 
        num_workers: int, 
        **dataset_kwargs
    ):
        super().__init__()
        self.batch_size = batch_size
        self.num_workers = num_workers
        self.dataset_kwargs = dataset_kwargs
        
        self.wild_dm = WildSVDDDataModule(
            data_dir=wild_data_dir, 
            batch_size=batch_size, 
            num_workers=num_workers, 
            **dataset_kwargs
        )
        
        self.sing_dm = SingFakeDataModule(
            data_dir=sing_data_dir, 
            batch_size=batch_size, 
            num_workers=num_workers, 
            **dataset_kwargs
        )

    def prepare_data(self):
        self.wild_dm.prepare_data()
        self.sing_dm.prepare_data()

    def setup(self, stage=None):
        self.wild_dm.setup(stage)
        self.sing_dm.setup(stage)

        if stage == "fit" or stage is None:
            self.train_dataset = ConcatDataset([
                self.wild_dm.training_dataset, 
                self.sing_dm.training_dataset
            ])
            

            self.validation_dataset = self.sing_dm.validation_dataset

       
        if stage == "test" or stage is None:
            pass 

    def train_dataloader(self):
        return DataLoader(
            self.train_dataset,
            batch_size=self.batch_size,
            shuffle=True, 
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
            pin_memory=True,
        )

    def test_dataloader(self):
        loaders = []
        
        loaders.extend(self.wild_dm.test_dataloader())
        
        sing_loaders = [
            self.sing_dm.test_t01_dataloader(),
            self.sing_dm.test_t02_dataloader(),
            self.sing_dm.test_t03_dataloader(),
            self.sing_dm.test_t04_dataloader()
        ]
        loaders.extend(sing_loaders)
        
        return loaders