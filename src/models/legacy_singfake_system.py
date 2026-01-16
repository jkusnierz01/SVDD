import lightning as L
from typing import Callable
import numpy as np
import torch
import torchmetrics
import wandb
from src.utils.utils import wandb_log_cm

class AntiSpoofSystem(L.LightningModule):
    def __init__(self, model_config: dict, loss_fn: Callable, optimizer: Callable, scheduler: dict):
        super().__init__()
        self.save_hyperparameters(ignore=["model_config"])

        if "weight" in loss_fn.keywords:
            raw_weights = loss_fn.keywords["weight"]
            self.register_buffer("loss_weight", torch.tensor(raw_weights, dtype=torch.float32))
            
            self.loss_fn = loss_fn(weight=self.loss_weight)
        else:
            self.loss_fn = loss_fn()

        self.optimizer_fn = optimizer
        self.scheduler = scheduler
        
        self.validation_outputs = {'predictions': [], 'targets': []}
        self.test_outputs = {}
        self.model = model_config
        self.accuracy = torchmetrics.Accuracy(task="binary")
        self.precision = torchmetrics.Precision(task="binary")
        self.recall = torchmetrics.Recall(task="binary")
        self.f1 = torchmetrics.F1Score(task="binary")
        self.eer = torchmetrics.classification.EER(task="binary")
        self.conf_matrix = torchmetrics.ConfusionMatrix(task='binary')
        
        
    def forward(self, x):
        x = self.model(x)
        return x

    def configure_optimizers(self):
        optimizer = self.optimizer_fn(self.parameters())

        if self.scheduler.type == "cosine":
            total_steps = self.trainer.estimated_stepping_batches
            base_lr = self.scheduler.lr_max
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
        x, y_true = train_batch
        y_pred = self(x)
        loss = self.loss_fn(y_pred, y_true)
        self.log("train/loss", loss, on_epoch=True, prog_bar=True)
        return loss
    
    def validation_step(self, val_batch, batch_idx):
        x, y_true = val_batch
        y_pred = self(x)
        loss = self.loss_fn(y_pred, y_true)
        
        preds = torch.argmax(y_pred, dim=1)
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
        probs = torch.softmax(all_logits, dim=1)[:, 1]  # prob for positive class
        preds = torch.argmax(all_logits, dim=1)  # zamiast (torch.sigmoid(y_pred) > 0.5).int()
        
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
        
        # Keep _save_results
        # self._save_results(all_logits, all_targets)
        self.validation_outputs = {'predictions': [], 'targets': []}

    def test_step(self, batch, batch_idx, dataloader_idx: int = 0):
        x, y_true = batch
        y_pred = self(x)
        loss = self.loss_fn(y_pred, y_true)
        
        if dataloader_idx not in self.test_outputs:
            self.test_outputs[dataloader_idx] = {'predictions': [], 'targets': []}
        self.test_outputs[dataloader_idx]['predictions'].append(y_pred)
        self.test_outputs[dataloader_idx]['targets'].append(y_true)
        return loss
    
    def on_test_start(self):
        print("num test dataloaders:", len(self.trainer.test_dataloaders))
        for i, dl in enumerate(self.trainer.test_dataloaders):
            try:
                print(f"test dl {i}: len(dataset)={len(dl.dataset)}, len(dataloader)={len(dl)}")
            except Exception as e:
                print(f"test dl {i}: cannot get len -> {e}")
        
    def on_test_epoch_end(self):
        if wandb.run is None:
            self.test_outputs.clear()
            return

        rows = []

        for dataloader_idx, outputs in self.test_outputs.items():
            all_logits = torch.cat(outputs["predictions"], dim=0)
            all_targets = torch.cat(outputs["targets"], dim=0)

            # Convert to probs and preds
            probs = torch.softmax(all_logits, dim=1)[:, 1]
            preds = torch.argmax(all_logits, dim=1)

            # Confusion matrix (zostaje)
            self.conf_matrix.update(preds, all_targets)
            cm = self.conf_matrix.compute()
            wandb_log_cm(cm, f"test/dataloader_{dataloader_idx}/confusion_matrix")

            # Metrics (zostaje)
            self.eer.update(probs, all_targets)
            eer_val = self.eer.compute()

            self.f1.update(preds, all_targets)
            f1_val = self.f1.compute()

            self.precision.update(preds, all_targets)
            precision_val = self.precision.compute()

            self.recall.update(preds, all_targets)
            recall_val = self.recall.compute()

            rows.append([
                int(dataloader_idx),
                float(eer_val),
                float(f1_val),
                float(precision_val),
                float(recall_val),
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


def cosine_annealing(step, total_steps, lr_max, lr_min):
    """Cosine Annealing for learning rate decay scheduler"""
    return lr_min + (lr_max - lr_min) * 0.5 * (1 + np.cos(step / total_steps * np.pi))
