from mamba_ssm.modules.mamba_simple import Mamba
import lightning as L
import torch
import torch.nn as nn
import torchmetrics
import wandb
from src.utils.utils import wandb_log_cm
from typing import Callable, Optional


class MambaBlock(nn.Module):
    def __init__(
        self,
        d_model: int,
        d_state: int,
        d_conv: int,
        expand: int,
        dropout: int = 0.0
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
        residual = x
        x = self.norm_layer(x)
        x = self.mamba_layer(x)
        x = self.dropout(x)
        output = residual + x
        return output

class BaseDeepfakeModel(L.LightningModule):
    def __init__(self, optimizer: Callable, scheduler: Optional[Callable] = None):
        super().__init__()
        
        self.optimizer_fn = optimizer
        self.scheduler_fn = scheduler
        
        self.loss_fn = nn.BCEWithLogitsLoss()
        self.accuracy = torchmetrics.Accuracy(task="binary")
        self.precision = torchmetrics.Precision(task="binary")
        self.recall = torchmetrics.Recall(task="binary")
        self.f1 = torchmetrics.F1Score(task="binary")
        self.eer = torchmetrics.classification.EER(task="binary")
        self.conf_matrix = torchmetrics.ConfusionMatrix(task='binary')

        self.validation_outputs = {'predictions': [], 'targets': []}
        self.test_outputs = {}
    
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
        #to work with short dataset
        # x -> [batch, z1, z2]
        x, y_true = train_batch
        y_pred = self(x).squeeze(-1)
        loss = self.loss_fn(y_pred, y_true.float())
        probs = torch.sigmoid(y_pred)
        preds = (probs > 0.5).int()
        acc = self.accuracy(preds, y_true)
        self.log("train/acc", acc, on_epoch=True, prog_bar=True)
        self.log("train/loss", loss, on_epoch=True, prog_bar=True)
        return loss
    
    def validation_step(self, val_batch, batch_idx):
        #to work with short dataset
        x, y_true = val_batch
        y_pred = self(x).squeeze(-1)
        loss = self.loss_fn(y_pred, y_true.float())
        probs = torch.sigmoid(y_pred)
        preds = (probs > 0.5).int()
        precision = self.precision(preds, y_true)
        recall = self.recall(preds, y_true)
        self.log("val/precision", precision, on_epoch=True, prog_bar=True)
        self.log("val/recall", recall, on_epoch=True, prog_bar=True)
        self.log("val/loss", loss, on_epoch=True, prog_bar=True)
        
        # Append, not overwrite
        self.validation_outputs['predictions'].append(y_pred)
        self.validation_outputs['targets'].append(y_true)
        return loss
    
    def on_validation_epoch_end(self):
        all_logits = torch.cat(self.validation_outputs['predictions'], dim=0)
        all_targets = torch.cat(self.validation_outputs['targets'], dim=0)
        
        # Convert to probs and preds
        probs = torch.sigmoid(all_logits)
        preds = (probs > 0.5).int()
        
        # Update and compute only what you want (e.g., EER and F1)
        # self.conf_matrix.update(preds, all_targets)
        # cm = self.conf_matrix.compute()
        
        self.eer.update(probs, all_targets)
        eer_val = self.eer.compute()
        self.log('val/eer', eer_val)
        
        self.f1.update(preds, all_targets)
        f1_val = self.f1.compute()
        self.log('val/f1', f1_val)
        
        # Reset
        self.eer.reset()
        self.f1.reset()
        self.conf_matrix.reset()
        
        self.validation_outputs = {'predictions': [], 'targets': []}
    
    def test_step(self, batch, batch_idx, dataloader_idx: int = 0):
        x, y_true = batch
        y_true = y_true.float()
        y_pred = self(x).squeeze(-1)
        loss = self.loss_fn(y_pred, y_true.float())
        
        if dataloader_idx not in self.test_outputs:
            self.test_outputs[dataloader_idx] = {'predictions': [], 'targets': []}
        self.test_outputs[dataloader_idx]['predictions'].append(y_pred)
        self.test_outputs[dataloader_idx]['targets'].append(y_true)
        return loss
    
    def on_test_epoch_end(self):
        if wandb.run is None:
            self.test_outputs.clear()
            return

        rows = []

        for dataloader_idx, outputs in self.test_outputs.items():
            all_logits = torch.cat(outputs["predictions"], dim=0)
            all_targets = torch.cat(outputs["targets"], dim=0)

            # Convert to probs and preds
            probs = torch.sigmoid(all_logits)
            preds = (probs > 0.5).int()

            # Confusion matrix (zostaje)
            self.conf_matrix.update(preds, all_targets)
            cm = self.conf_matrix.compute()
            wandb_log_cm(cm, f"test/dataloader_{dataloader_idx}/confusion_matrix")

            # Metrics (zostaje)
            self.eer.update(probs, all_targets)
            self.f1.update(preds, all_targets)
            self.precision.update(preds, all_targets)
            self.recall.update(preds, all_targets)

            rows.append([
                int(dataloader_idx),
                float(self.eer.compute()),
                float(self.f1.compute()),
                float(self.precision.compute()),
                float(self.recall.compute()),
            ])

            # Reset (zostaje)
            self.eer.reset()
            self.f1.reset()
            self.precision.reset()
            self.recall.reset()
            self.conf_matrix.reset()

        # Table + bar plots
        table = wandb.Table(
            columns=["dataloader", "eer", "f1", "precision", "recall"],
            data=rows
        )

        wandb.log({
            "test/metrics_table": table,
            "test/eer_bar": wandb.plot.bar(table, "dataloader", "eer", title="Test EER per dataloader"),
            "test/f1_bar": wandb.plot.bar(table, "dataloader", "f1", title="Test F1 per dataloader"),
            "test/precision_bar": wandb.plot.bar(table, "dataloader", "precision", title="Test Precision per dataloader"),
            "test/recall_bar": wandb.plot.bar(table, "dataloader", "recall", title="Test Recall per dataloader"),
        })

        self.test_outputs.clear()

class MambaDeepfakeModel(BaseDeepfakeModel):
    def __init__(
        self,
        input_dim: int,
        d_model: int,
        d_state: int,
        d_conv: int,
        expand: int,
        n_layers: int,
        dropout: int,
        optimizer: Callable,
        scheduler: Optional[Callable] = None,
    ):
        super().__init__(optimizer=optimizer, scheduler=scheduler)
        print(f"DEBUG INICJALIZACJA MODELU: n_layers={n_layers}")
        self.save_hyperparameters()
        
        ## MOdel
        self.n_layers = n_layers
        self.linear = nn.Linear(input_dim, d_model)
        self.layers = nn.ModuleList(
            [
                MambaBlock(
                    d_model=d_model, d_state=d_state, d_conv=d_conv, expand=expand, dropout=dropout
                )
                for _ in range(self.n_layers)
            ]
        )
        self.linear_2 = nn.Linear(d_model, 1)

    def forward(self, x: torch.Tensor):
        x = self.linear(x)
        for layer in self.layers:
            x = layer(x)
        last = x[:, -1, :]
        logit = self.linear_2(last)
        return logit

    
        
class BidirectionalMambaModel(BaseDeepfakeModel):
    def __init__(
        self,
        input_dim: int,
        d_model: int,
        d_state: int,
        d_conv: int,
        expand: int,
        n_layers: int,
        dropout: int,
        optimizer: Callable,
        scheduler: Optional[Callable] = None,
    ):
        super().__init__(optimizer=optimizer, scheduler=scheduler)
        self.save_hyperparameters()

        ## Model
        self.n_layers = n_layers
        self.linear = nn.Linear(input_dim, d_model)
        self.mamba_forward = nn.ModuleList(
            [
                MambaBlock(
                    d_model=d_model, d_state=d_state, d_conv=d_conv, expand=expand, dropout=dropout
                )
                for _ in range(self.n_layers)
            ]
        )
        self.mamba_backward = nn.ModuleList(
            [
                MambaBlock(
                    d_model=d_model, d_state=d_state, d_conv=d_conv, expand=expand, dropout=dropout
                )
                for _ in range(self.n_layers)
            ]
        )
        self.linear_2 = nn.Linear(d_model, 1)
        
    
    # [batch_size, 6000, 2048]
    def forward(self, vector: torch.Tensor):
        backward = torch.flip(vector, [1])
        x_forward = self.linear(vector)
        x_backward = self.linear(backward)
        
        for layer in self.mamba_forward:
            x_forward = layer(x_forward)
        
        for layer in self.mamba_backward:
            x_backward = layer(x_backward)
        
        x_backward = torch.flip(x_backward, [1])
        out = x_forward + x_backward
        out = out.mean(dim=1)
        
        logit = self.linear_2(out)
        return logit
