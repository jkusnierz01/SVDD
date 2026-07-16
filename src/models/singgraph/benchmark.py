"""Lightning wrapper for SingGraph on precomputed benchmark features."""

from __future__ import annotations

from typing import Callable, Optional

import torch
import torch.nn as nn
import torchmetrics

from src.models.mamba import BaseDeepfakeModel
from src.models.singgraph.backbone import SingGraphBackbone


class SingGraphBenchmarkModel(BaseDeepfakeModel):
    """SingGraph with utterance-level max over 30 windows; CE loss on track logits."""

    def __init__(
        self,
        feat_dim: int = 1024,
        num_windows: int = 30,
        enc_dim: int = 128,
        gat_dims: list[int] | None = None,
        pool_ratios: list[float] | None = None,
        temperatures: list[float] | None = None,
        num_nodes_s: int = 42,
        dropout: float = 0.5,
        drop_way: float = 0.2,
        optimizer: Callable = None,
        scheduler: Optional[Callable] = None,
        label_smoothing: float = 0.0,
    ):
        super().__init__(
            optimizer=optimizer,
            scheduler=scheduler,
            label_smoothing=0.0,
        )
        self.save_hyperparameters(ignore=["optimizer", "scheduler"])

        self.num_windows = num_windows
        self.backbone = SingGraphBackbone(
            feat_dim=feat_dim,
            enc_dim=enc_dim,
            gat_dims=gat_dims,
            pool_ratios=pool_ratios,
            temperatures=temperatures,
            num_nodes_s=num_nodes_s,
            dropout=dropout,
            drop_way=drop_way,
        )

        self.loss_fn = (
            nn.CrossEntropyLoss(label_smoothing=label_smoothing)
            if label_smoothing > 0.0
            else nn.CrossEntropyLoss()
        )
        self.accuracy = torchmetrics.Accuracy(task="multiclass", num_classes=2)

    def _unpack_batch(self, batch):
        (vocals, instrumental), labels = batch
        return vocals, instrumental, labels

    @staticmethod
    def _flatten_windows(x: torch.Tensor) -> torch.Tensor:
        b, w, t, d = x.shape
        return x.reshape(b * w, t, d)

    def forward_windows(
        self, vocals: torch.Tensor, instrumental: torch.Tensor
    ) -> torch.Tensor:
        return self.backbone(vocals, instrumental)

    def forward_utterance(
        self, vocals: torch.Tensor, instrumental: torch.Tensor
    ) -> torch.Tensor:
        v = self._flatten_windows(vocals)
        i = self._flatten_windows(instrumental)
        window_logits = self.forward_windows(v, i)
        window_logits = window_logits.view(-1, self.num_windows, 2)
        return window_logits.max(dim=1).values

    def forward(self, batch) -> torch.Tensor:
        vocals, instrumental, _ = self._unpack_batch(batch)
        return self.forward_utterance(vocals, instrumental)

    def _compute_loss(self, batch, log_prefix: str):
        vocals, instrumental, labels = self._unpack_batch(batch)
        logits = self.forward_utterance(vocals, instrumental)
        loss = self.loss_fn(logits, labels)

        preds = logits.argmax(dim=-1)
        acc = self.accuracy(preds, labels)

        self.log(f"{log_prefix}/loss", loss, on_epoch=True, prog_bar=True)
        self.log(f"{log_prefix}/acc", acc, on_epoch=True, prog_bar=True)
        return loss

    def training_step(self, train_batch, batch_idx):
        return self._compute_loss(train_batch, "train")

    def validation_step(self, val_batch, batch_idx):
        vocals, instrumental, labels = self._unpack_batch(val_batch)
        logits = self.forward_utterance(vocals, instrumental)
        loss = self.loss_fn(logits, labels)
        self.log("val/loss", loss, on_epoch=True, prog_bar=True)

        fake_probs = torch.softmax(logits, dim=-1)[:, 1]
        self.validation_outputs["predictions"].append(fake_probs)
        self.validation_outputs["targets"].append(labels)
        return loss

    def on_validation_epoch_end(self):
        all_probs = torch.cat(self.validation_outputs["predictions"], dim=0)
        all_targets = torch.cat(self.validation_outputs["targets"], dim=0)

        preds = (all_probs > 0.5).int()

        self.eer.update(all_probs, all_targets)
        self.f1.update(preds, all_targets)
        self.precision.update(preds, all_targets)
        self.recall.update(preds, all_targets)

        self.log("val/eer", self.eer.compute(), prog_bar=True)
        self.log("val/f1", self.f1.compute(), prog_bar=True)
        self.log("val/precision", self.precision.compute())
        self.log("val/recall", self.recall.compute())

        self.eer.reset()
        self.f1.reset()
        self.precision.reset()
        self.recall.reset()
        self.conf_matrix.reset()

        self.validation_outputs = {"predictions": [], "targets": []}

    def test_step(self, batch, batch_idx, dataloader_idx: int = 0):
        vocals, instrumental, labels = self._unpack_batch(batch)
        logits = self.forward_utterance(vocals, instrumental)
        loss = self.loss_fn(logits, labels)

        fake_probs = torch.softmax(logits, dim=-1)[:, 1]
        if dataloader_idx not in self.test_outputs:
            self.test_outputs[dataloader_idx] = {"predictions": [], "targets": []}
        self.test_outputs[dataloader_idx]["predictions"].append(fake_probs)
        self.test_outputs[dataloader_idx]["targets"].append(labels)
        return loss
