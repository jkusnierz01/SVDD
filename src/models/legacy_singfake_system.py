import lightning as L
from hydra.utils import instantiate
from typing import Callable
import numpy as np
import torch


class AntiSpoofSystem(L.LightningModule):
    def __init__(self, model_config: dict, loss_fn: Callable, optimizer: Callable, scheduler: dict):
        super().__init__()
        self.save_hyperparameters()

        if "weight" in loss_fn.keywords:
            raw_weights = loss_fn.keywords["weight"]
            self.register_buffer("loss_weight", torch.tensor(raw_weights, dtype=torch.float32))
            
            self.loss_fn = loss_fn(weight=self.loss_weight)
        else:
            self.loss_fn = loss_fn()

        self.loss_fn = instantiate(loss_fn)
        self.model = instantiate(model_config)
        self.optimizer_fn = optimizer
        self.scheduler = scheduler
        
        

    def forward(self, x):
        x = self.model(x)
        return x

    def configure_optimizers(self):
        optimizer = self.optimizer_fn(self.parameters())

        if self.scheduler.type == "cosine":
            total_steps = self.trainer.estimated_stepping_batches
            base_lr = self.hparams.optimizer.lr
            scheduler = torch.optim.lr_scheduler.LambdaLR(
                optimizer=optimizer,
                lr_lambda=lambda step: cosine_annealing(
                    step=step,
                    total_steps=total_steps,
                    lr_max=1.0, 
                    lr_min=self.scheduler.lr_min / base_lr,
                ),
            )
            return {
                "optimizer": optimizer,
                "lr_scheduler": {
                    "scheduler": scheduler,
                    "interval": "step",
                },
            }


    def training_step(self, train_batch, batch_idx):
        x = self.forward(train_batch)


def cosine_annealing(step, total_steps, lr_max, lr_min):
    """Cosine Annealing for learning rate decay scheduler"""
    return lr_min + (lr_max - lr_min) * 0.5 * (1 + np.cos(step / total_steps * np.pi))
