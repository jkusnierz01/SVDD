import lightning as L


class SonicsDataModule(L.LightningDataModule):
    def __init__(self):
        super().__init__()
    
    def setup(self, stage):
        ...
        
        
    def train_dataloader(self):
        ...
        
    def val_dataloader(self):
        ...
        
        
    def test_dataloader(self):
        ...