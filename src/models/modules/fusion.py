import lightning as L
import torch
import torch.nn as nn
import torch.nn.functional as F


class GMU_per_feature(nn.Module):
    """
    Generates hidden_dim independent scalars for each timestamps.
    Controls each specific dimenstion in fusion space.
    """
    def __init__(self, dim_w2v: int = 1024, dim_mert: int = 1024, hidden_dim: int = 1024) -> None:
        super().__init__()
        
        self.proj_mert = nn.Linear(dim_mert, hidden_dim)
        self.proj_w2v = nn.Linear(dim_w2v, hidden_dim)

        self.gate = nn.Linear(dim_mert + dim_w2v, hidden_dim)

    def forward(self, x_w2v, x_mert):
        # x_mert: [B, 1500, 1024]
        # x_w2v:  [B, 1500, 1024]
        
        h_mert = torch.tanh(self.proj_mert(x_mert))  
        h_w2v  = torch.tanh(self.proj_w2v(x_w2v))  
        
        z = torch.sigmoid(self.gate(torch.cat([x_mert, x_w2v], dim=-1)))  
        
        return z * h_w2v + (1.0 - z) * h_mert  # [B, 1500, 1024]
    
class ScalarGMU(nn.Module):
    """
    Generates one scalar for each timestep. 
    It decides how much of wav2vec or mert embeddings to use in timestep.
    """
    def __init__(self, dim_w2v: int = 1024, dim_mert: int = 1024, hidden_dim: int = 1024) -> None:
        super().__init__()
        
        self.proj_mert = nn.Linear(dim_mert, hidden_dim)
        self.proj_w2v = nn.Linear(dim_w2v, hidden_dim)
        self.gate = nn.Linear(dim_mert + dim_w2v, 1)

    def forward(self, x_w2v, x_mert):
        # x_mert: [B, 1500, 1024]
        # x_w2v:  [B, 1500, 1024]
        
        h_mert = torch.tanh(self.proj_mert(x_mert))  
        h_w2v  = torch.tanh(self.proj_w2v(x_w2v))
        z = torch.sigmoid(self.gate(torch.cat([x_mert, x_w2v], dim=-1)))
        return z * h_w2v + (1.0 - z) * h_mert