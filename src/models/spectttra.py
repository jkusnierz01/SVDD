from sonics import HFAudioClassifier
import torch

import lightning as L

class SpectttraModel(L.LightningModule):
    def __init__(self, device: str, model_name:str = "awsaf49/sonics-spectttra-alpha-120s"):
        self.model = self.load_model(model_name)
        self.device = device
        
    @staticmethod
    def load_model(model_name:str):
        return HFAudioClassifier.from_pretrained(model_name)
    
    def __post_init__(self):
        self.model.eval()
        self.model.to(self.device)
    
    
    def forward(self, x: torch.Tensor):
        self.model(x)
        ...
        
    def test_step(self):
        ...
        
        
    def on_test_epoch_end(self):
        ...
        
    # TO DO
