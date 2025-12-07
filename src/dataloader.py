import lightning as L
from torch.utils.data import DataLoader
from torch.utils.data import Dataset
import torchaudio
import torch
from pathlib import Path

LABEL_MAP = {
    "bonafide": 1,  
    "spoof": 0      
}


class AudioDataset(Dataset):
    def __init__(self, files, labels):
        self.files = files
        self.labels = labels
        self.max_len = 4000000

    def __getitem__(self, index):
        wav, sr = torchaudio.load(self.files[index])
        label = self.labels[index]
        real_len = wav.shape[1]
        
        if real_len < self.max_len:
            pad_amt = self.max_len - real_len
            wav = torch.nn.functional.pad(wav, (0, pad_amt))
        else:
            wav = wav[:, :self.max_len]
            real_len = self.max_len
        return wav, torch.tensor(label, dtype=torch.float), real_len
    
    def __len__(self):
        return len(self.files)


class SingFakeDataModule(L.LightningDataModule):
    def __init__(self, data_dir:str, batch_size: int):
        super().__init__()
        self.data_dir = data_dir
        self.batch_size = batch_size
        
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
        
        self.training_dataset = AudioDataset(groups['Training']['files'], groups['Training']['labels'])
        self.validation_dataset = AudioDataset(groups['Validation']['files'], groups['Validation']['labels'])
        self.test_t01 = AudioDataset(groups['T01']['files'], groups['T01']['labels'])
        self.test_t02 = AudioDataset(groups['T02']['files'], groups['T02']['labels'])
        self.test_t03 = AudioDataset(groups['T03']['files'], groups['T03']['labels'])
        self.test_t04 = AudioDataset(groups['T04']['files'], groups['T04']['labels'])
                    
            
            
    
    def train_dataloader(self):
        return DataLoader(self.training_dataset, batch_size=self.batch_size, shuffle=True, num_workers=4)
    
    def val_dataloader(self):
        return DataLoader(self.validation_dataset, batch_size=self.batch_size, num_workers=4)
    
    def test_t01_dataloader(self):
        return DataLoader(self.test_t01, batch_size=self.batch_size)
    
    def test_t02_dataloader(self):
        return DataLoader(self.test_t02, batch_size=self.batch_size)
    
    def test_t03_dataloader(self):
        return DataLoader(self.test_t03, batch_size=self.batch_size)
    
    def test_t04_dataloader(self):
        return DataLoader(self.test_t04, batch_size=self.batch_size)