from mamba_ssm.modules.mamba_simple import Mamba
from src.models.modules.fusion import GMU_per_feature, ScalarGMU
from src.models.modules.base import SmoothedBCEWithLogitsLoss
import lightning as L
import torch
import torch.nn as nn
import torchmetrics
import wandb
from src.utils.utils import wandb_log_cm
from typing import Callable, Optional
from src.models.mamba import BaseDeepfakeModel




class ControlModel(BaseDeepfakeModel):
    def __init__(
        self,
        w2v_dim: int = 1024,
        optimizer: Callable = None,
        label_smoothing: float = 0.0,
        scheduler: Optional[Callable] = None,
    ):
        super().__init__(
            optimizer=optimizer, scheduler=scheduler, label_smoothing=label_smoothing
        )
        self.save_hyperparameters(ignore=["optimizer", "scheduler"])
        self.w2v_dim = w2v_dim
        self.layer_one = nn.Linear(w2v_dim, 1)
        self.norm = nn.LayerNorm(w2v_dim)

    def forward(self, vector: torch.Tensor):
        vector_w2v, vector_mert = self._split_modalities(vector)

        vector_mean = vector_w2v.mean(dim=1)
        v_normalized = self.norm(vector_mean)
        logit = self.layer_one(v_normalized)
        return logit