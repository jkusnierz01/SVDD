from mamba_ssm.modules.mamba_simple import Mamba
import lightning as L
import torch
import torch.nn as nn
import torchmetrics
from typing import Callable, Optional


class MambaBlock(L.LightningModule):
    def __init__(
        self,
        d_model: int,
        d_state: int,
        d_conv: int,
        expand: int,
        dropout: int
    ):
        super().__init__()
        self.norm_layer = nn.LayerNorm(d_model)
        self.mamba_layer = Mamba(
            d_model=d_model,
            d_state=d_state,
            d_conv=d_conv,
            expand=expand,
        )
        
        self.dropout = nn.Dropout(dropout)
        
    def forward(self, x: torch.Tensor):
        x_norm = self.norm_layer(x)
        x_mamba = self.mamba_layer(x_norm)
        x = self.dropout(x)
        x_out = x_mamba + x
        return x_out


class MambaDeepfakeModel(L.LightningModule):
    def __init__(
        self,
        n_mels: int,
        hop_length: int,
        d_model: int,
        d_state: int,
        d_conv: int,
        expand: int,
        n_layers: int,
        dropout: int,
        optimizer: Callable,
        scheduler: Optional[Callable] = None,
    ):
        super().__init__()
        self.save_hyperparameters()
        self.hop_len = hop_length
        self.n_layers = n_layers
        
        self.optimizer_fn = optimizer
        self.scheduler_fn = scheduler       

        self.loss_fn = nn.BCEWithLogitsLoss()
        self.accuracy = torchmetrics.Accuracy(task="binary")

        ## Model
        self.linear = nn.Linear(n_mels, d_model)
        self.layers = nn.ModuleList(
            [
                MambaBlock(
                    d_model=d_model, d_state=d_state, d_conv=d_conv, expand=expand, dropout=dropout
                )
                for _ in range(self.n_layers)
            ]
        )
        self.linear_2 = nn.Linear(d_model, 1)

    def forward(self, x: torch.Tensor, lengths: torch.Tensor = None):
        x = self.linear(x)
        for layer in self.layers:
            x = layer(x)
        if lengths is not None:
            batch_idx = torch.arange(x.size(0), device=x.device)
            last = x[batch_idx, lengths - 1, :]
        else:
            last = x[:, -1, :]

        logit = self.linear_2(last)
        return logit

    def configure_optimizers(self):
        optimizer = self.optimizer_fn(self.parameters())

        if self.scheduler_fn is not None:
            scheduler = self.scheduler_fn(optimizer=optimizer)
            return {
                "optimizer": optimizer,
                "lr_scheduler": {
                    "scheduler": scheduler,
                    "monitor": "val/loss",
                },
            }

        return {"optimizer": optimizer}


    def training_step(self, train_batch, batch_idx):
        x, y_true, x_len = train_batch
        y_pred = self.forward(x, lengths=x_len).squeeze(-1)
        loss = self.loss_fn(y_pred, y_true)
        probs = torch.sigmoid(y_pred)
        preds = (probs > 0.5).int()
        acc = self.accuracy(preds, y_true)
        self.log("train/acc", acc, on_epoch=True, prog_bar=True)
        self.log("train/loss", loss, on_epoch=True, prog_bar=True)
        return loss

    def validation_step(self, val_batch, batch_idx):
        x, y_true, x_len = val_batch
        y_pred = self.forward(x, lengths=x_len).squeeze(-1)
        loss = self.loss_fn(y_pred, y_true)
        probs = torch.sigmoid(y_pred)
        preds = (probs > 0.5).int()
        acc = self.accuracy(preds, y_true)
        self.log("val/acc", acc, on_epoch=True, prog_bar=True)
        self.log("val/loss", loss, on_epoch=True, prog_bar=True)
        return loss
