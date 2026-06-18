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
from enum import Enum


class MambaBlock(nn.Module):
    def __init__(
        self, d_model: int, d_state: int, d_conv: int, expand: int, dropout: int = 0.0
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
    def __init__(
        self,
        optimizer: Callable,
        scheduler: Optional[Callable] = None,
        label_smoothing: float = 0.1,
    ):
        super().__init__()

        self.optimizer_fn = optimizer
        self.scheduler_fn = scheduler

        self.loss_fn = SmoothedBCEWithLogitsLoss(smoothing=label_smoothing) if label_smoothing > 0.0 else nn.BCEWithLogitsLoss()
        self.accuracy = torchmetrics.Accuracy(task="binary")
        self.precision = torchmetrics.Precision(task="binary")
        self.recall = torchmetrics.Recall(task="binary")
        self.f1 = torchmetrics.F1Score(task="binary")
        self.eer = torchmetrics.classification.EER(task="binary")
        self.conf_matrix = torchmetrics.ConfusionMatrix(task="binary")

        self.validation_outputs = {"predictions": [], "targets": []}
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
        x, y_true = train_batch
        if batch_idx == 0:
            print(
                f"DEBUG Batch 0 stats: Mean={x.mean():.4f}, Std={x.std():.4f}, Min={x.min():.4f}, Max={x.max():.4f}"
            )
            if x.std() < 1e-5:
                print(
                    "WARNING: Input features have zero variance! Model collapse imminent."
                )

        y_pred = self(x).squeeze(-1)
        loss = self.loss_fn(y_pred, y_true.float())
        probs = torch.sigmoid(y_pred)
        preds = (probs > 0.5).int()
        acc = self.accuracy(preds, y_true)
        self.log("train/acc", acc, on_epoch=True, prog_bar=True)
        self.log("train/loss", loss, on_epoch=True, prog_bar=True)
        return loss

    def on_train_epoch_end(self):
        self.accuracy.reset()

    def validation_step(self, val_batch, batch_idx):
        x, y_true = val_batch
        y_pred = self(x).squeeze(-1)
        loss = self.loss_fn(y_pred, y_true.float())
        self.log("val/loss", loss, on_epoch=True, prog_bar=True)
        self.validation_outputs["predictions"].append(y_pred)
        self.validation_outputs["targets"].append(y_true)
        return loss

    def on_validation_epoch_end(self):
        all_logits = torch.cat(self.validation_outputs["predictions"], dim=0)
        all_targets = torch.cat(self.validation_outputs["targets"], dim=0)

        probs = torch.sigmoid(all_logits)
        preds = (probs > 0.5).int()

        self.eer.update(probs, all_targets)
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
        x, y_true = batch
        y_true = y_true.float()
        y_pred = self(x).squeeze(-1)
        loss = self.loss_fn(y_pred, y_true.float())

        if dataloader_idx not in self.test_outputs:
            self.test_outputs[dataloader_idx] = {"predictions": [], "targets": []}
        self.test_outputs[dataloader_idx]["predictions"].append(y_pred)
        self.test_outputs[dataloader_idx]["targets"].append(y_true)
        return loss

    def on_test_epoch_end(self):
        if wandb.run is None:
            self.test_outputs.clear()
            return

        rows = []

        for dataloader_idx, outputs in self.test_outputs.items():
            all_logits = torch.cat(outputs["predictions"], dim=0)
            all_targets = torch.cat(outputs["targets"], dim=0).long()

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

            rows.append(
                [
                    int(dataloader_idx),
                    float(self.eer.compute()),
                    float(self.f1.compute()),
                    float(self.precision.compute()),
                    float(self.recall.compute()),
                ]
            )

            # Reset (zostaje)
            self.eer.reset()
            self.f1.reset()
            self.precision.reset()
            self.recall.reset()
            self.conf_matrix.reset()

        # Table + bar plots
        table = wandb.Table(
            columns=["dataloader", "eer", "f1", "precision", "recall"], data=rows
        )

        wandb.log(
            {
                "test/metrics_table": table,
                "test/eer_bar": wandb.plot.bar(
                    table, "dataloader", "eer", title="Test EER per dataloader"
                ),
                "test/f1_bar": wandb.plot.bar(
                    table, "dataloader", "f1", title="Test F1 per dataloader"
                ),
                "test/precision_bar": wandb.plot.bar(
                    table,
                    "dataloader",
                    "precision",
                    title="Test Precision per dataloader",
                ),
                "test/recall_bar": wandb.plot.bar(
                    table, "dataloader", "recall", title="Test Recall per dataloader"
                ),
            }
        )

        self.test_outputs.clear()

    @staticmethod
    def _instance_norm(x: torch.Tensor) -> torch.Tensor:
        """Per-sample norm over time and features (legacy `global` / `per_modality`)."""
        mean = x.mean(dim=(1, 2), keepdim=True)
        std = x.std(dim=(1, 2), keepdim=True) + 1e-6
        return (x - mean) / std

    def _init_embedding_norm_layers(
        self,
        norm_type: str,
        w2v_dim: int,
        mert_dim: int,
        fused_dim: int,
    ) -> None:
        valid = {
            "none",
            "global",
            "per_modality",
            "layer_norm",
            "layer_norm_per_modality",
        }
        if norm_type not in valid:
            raise ValueError(
                f"Unknown norm_type={norm_type!r}. Expected one of {sorted(valid)}"
            )
        self.norm_type = norm_type
        self.ln_w2v = None
        self.ln_mert = None
        self.ln_fused = None
        if norm_type == "layer_norm_per_modality":
            self.ln_w2v = nn.LayerNorm(w2v_dim)
            self.ln_mert = nn.LayerNorm(mert_dim)
        elif norm_type == "layer_norm":
            self.ln_fused = nn.LayerNorm(fused_dim)

    def _normalize_modalities(
        self, vector_w2v: torch.Tensor, vector_mert: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor]:
        if self.norm_type == "per_modality":
            return (
                self._instance_norm(vector_w2v),
                self._instance_norm(vector_mert),
            )
        if self.norm_type == "layer_norm_per_modality":
            return self.ln_w2v(vector_w2v), self.ln_mert(vector_mert)
        return vector_w2v, vector_mert

    def _normalize_fused(self, vector: torch.Tensor) -> torch.Tensor:
        if self.norm_type == "global":
            return self._instance_norm(vector)
        if self.norm_type == "layer_norm":
            return self.ln_fused(vector)
        return vector

    def _split_modalities(self, vector: torch.Tensor):
        w2v = vector[..., : self.w2v_dim]
        mert = vector[..., self.w2v_dim :]
        return w2v, mert


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
                    d_model=d_model,
                    d_state=d_state,
                    d_conv=d_conv,
                    expand=expand,
                    dropout=dropout,
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
        w2v_dim: int,
        mert_dim: int,
        fusion_dim: int,
        d_model: int,
        d_state: int,
        d_conv: int,
        expand: int,
        n_layers: int,
        dropout: float,
        label_smoothing: float = 0.1,
        optimizer: Callable = None,
        scheduler: Optional[Callable] = None,
        norm_type: str = "global",
        test_seq_len: int = 1500,
        fusion_type: str = "concat",
        modality_dropout: bool = False,
        only_modality: str | None = None,
        aggregation: str | None = "mean"
    ):
        super().__init__(
            optimizer=optimizer, scheduler=scheduler, label_smoothing=label_smoothing
        )
        self.save_hyperparameters(ignore=["optimizer", "scheduler"])
        self.modality_dropout = modality_dropout
        self.fusion_type = fusion_type
        self.w2v_dim = w2v_dim
        self.mert_dim = mert_dim
        self.only_modality = only_modality
        self.aggregation = aggregation

        if self.only_modality == "w2v":
            encoder_input_dim = w2v_dim
        elif self.only_modality == "mert":
            encoder_input_dim = mert_dim
        elif fusion_type == "concat":
            encoder_input_dim = w2v_dim + mert_dim
        elif fusion_type == "gmu":
            self.fusion = GMU_per_feature(
                dim_w2v=w2v_dim, dim_mert=mert_dim, hidden_dim=fusion_dim
            )
            encoder_input_dim = fusion_dim
        elif fusion_type == "scalar-gmu":
            self.fusion = ScalarGMU(
                dim_w2v=w2v_dim, dim_mert=mert_dim, hidden_dim=fusion_dim
            )
            encoder_input_dim = fusion_dim
        elif fusion_type == "add":
            self.w2v_proj = nn.Linear(w2v_dim, fusion_dim)
            self.mert_proj = nn.Linear(mert_dim, fusion_dim)
            encoder_input_dim = fusion_dim
        else:
            raise ValueError(f"Unknown fusion type: {fusion_type}")

        # -----------------------
        ##  MODEL
        # -----------------------
        self.n_layers = n_layers
        # self.pool = nn.AvgPool1d(kernel_size=4, stride=4)
        self.linear_forward = nn.Linear(encoder_input_dim, d_model)
        self.linear_backward = nn.Linear(encoder_input_dim, d_model)
        self.mamba_forward = nn.ModuleList(
            [
                MambaBlock(
                    d_model=d_model,
                    d_state=d_state,
                    d_conv=d_conv,
                    expand=expand,
                    dropout=dropout,
                )
                for _ in range(self.n_layers)
            ]
        )
        self.mamba_backward = nn.ModuleList(
            [
                MambaBlock(
                    d_model=d_model,
                    d_state=d_state,
                    d_conv=d_conv,
                    expand=expand,
                    dropout=dropout,
                )
                for _ in range(self.n_layers)
            ]
        )
        self.linear_2 = nn.Linear(2 * d_model, 1)
        self._init_embedding_norm_layers(
            norm_type=norm_type,
            w2v_dim=w2v_dim,
            mert_dim=mert_dim,
            fused_dim=encoder_input_dim,
        )
        self.test_seq_len = test_seq_len

    # ----------------------------------------------------------------
    # [batch_size, 1500, 2048]  (pre-pooled by precompute_pooled.py)
    # [batch_size, 6000, 2048] if without pooling in precomputation
    # Feature layout: [..., :modality_split] = wav2vec, [..., modality_split:] = MERT
    def forward(self, vector: torch.Tensor):
        # spliting concated vectors
        vector_w2v, vector_mert = self._split_modalities(vector)

        # MODALITY DROPOUT
        if self.training and self.modality_dropout and self.only_modality is None:
            rand_val = torch.rand(1).item()
            if rand_val < 0.15:
                vector_w2v = torch.zeros_like(vector_w2v)
            elif rand_val < 0.30:
                vector_mert = torch.zeros_like(vector_mert)

        vector_w2v, vector_mert = self._normalize_modalities(vector_w2v, vector_mert)

        if self.only_modality == "w2v":
            vector = vector_w2v
        elif self.only_modality == "mert":
            vector = vector_mert
        elif self.fusion_type == "gmu":
            vector = self.fusion(vector_w2v, vector_mert)
        elif self.fusion_type == "scalar-gmu":
            vector = self.fusion(vector_w2v, vector_mert)
        elif self.fusion_type == "add":
            vector = self.w2v_proj(vector_w2v) + self.mert_proj(vector_mert)
        elif self.fusion_type == "concat":
            vector = torch.cat([vector_w2v, vector_mert], dim=-1)
        else:
            raise ValueError(f"Unknown fusion type: {self.fusion_type}")

        vector = self._normalize_fused(vector)

        backward = torch.flip(vector, [1])
        x_forward = self.linear_forward(vector)
        x_backward = self.linear_backward(backward)

        for layer in self.mamba_forward:
            x_forward = layer(x_forward)

        for layer in self.mamba_backward:
            x_backward = layer(x_backward)

        x_backward = torch.flip(x_backward, [1])
        
        # AGGREGATION
        if self.aggregation == "last_vector":
            last_backward = x_backward[:,0,:]
            last_forward = x_forward[:,-1,:]
            out = torch.cat([last_forward, last_backward], dim=-1)
            logit = self.linear_2(out)
        elif self.aggregation == "mean":
            out = torch.cat([x_forward, x_backward], dim=-1)  # [B, 1500, 1024]
            out = out.mean(dim=1)  # [B, 1024]
            logit = self.linear_2(out)
        return logit

    def test_step(self, batch, batch_idx, dataloader_idx: int = 0):
        x, y_true = batch
        if self.test_seq_len < 1500:
            x = x[:, : self.test_seq_len, :]
        y_true = y_true.float()
        y_pred = self(x).squeeze(-1)
        loss = self.loss_fn(y_pred, y_true.float())

        if dataloader_idx not in self.test_outputs:
            self.test_outputs[dataloader_idx] = {"predictions": [], "targets": []}
        self.test_outputs[dataloader_idx]["predictions"].append(y_pred)
        self.test_outputs[dataloader_idx]["targets"].append(y_true)
        return loss


class LateFusionBidirectionalMambaModel(BaseDeepfakeModel):
    def __init__(
        self,
        w2v_dim: int,
        mert_dim: int,
        fusion_dim: int,
        d_model: int,
        d_state: int,
        d_conv: int,
        expand: int,
        n_layers: int,
        dropout: float,
        label_smoothing: float = 0.1,
        optimizer: Callable = None,
        scheduler: Optional[Callable] = None,
        norm_type: str | None = None,
        test_seq_len: int = 1500,
        fusion_type: str = "concat",
        modality_dropout: bool = False,
        aggregation: str | None = "mean"
    ):
        super().__init__(
            optimizer=optimizer, scheduler=scheduler, label_smoothing=label_smoothing
        )
        self.save_hyperparameters(ignore=["optimizer", "scheduler"])
        
        self.modality_dropout = modality_dropout
        self.n_layers = n_layers
        self.fusion_type = fusion_type
        self.w2v_dim = w2v_dim
        self.mert_dim = mert_dim
        self.test_seq_len = test_seq_len
        self.aggregation = aggregation

        self.linear_forward_w2v = nn.Linear(w2v_dim, d_model)
        self.linear_forward_mert = nn.Linear(mert_dim, d_model)

        self.linear_backward_w2v = nn.Linear(w2v_dim, d_model)
        self.linear_backward_mert = nn.Linear(mert_dim, d_model)

        self.mamba_forward_w2v = nn.ModuleList(
            [
                MambaBlock(
                    d_model=d_model,
                    d_state=d_state,
                    d_conv=d_conv,
                    expand=expand,
                    dropout=dropout,
                )
                for _ in range(self.n_layers)
            ]
        )
        self.mamba_backward_w2v = nn.ModuleList(
            [
                MambaBlock(
                    d_model=d_model,
                    d_state=d_state,
                    d_conv=d_conv,
                    expand=expand,
                    dropout=dropout,
                )
                for _ in range(self.n_layers)
            ]
        )

        self.mamba_forward_mert = nn.ModuleList(
            [
                MambaBlock(
                    d_model=d_model,
                    d_state=d_state,
                    d_conv=d_conv,
                    expand=expand,
                    dropout=dropout,
                )
                for _ in range(self.n_layers)
            ]
        )
        self.mamba_backward_mert = nn.ModuleList(
            [
                MambaBlock(
                    d_model=d_model,
                    d_state=d_state,
                    d_conv=d_conv,
                    expand=expand,
                    dropout=dropout,
                )
                for _ in range(self.n_layers)
            ]
        )
        if self.aggregation == "last_vector" and fusion_type in ["gmu", "scalar-gmu"]:
            raise ValueError("last vector is not working with gmu or scalar-gmu fusion")

        if fusion_type == "concat":
            linear_input_dim = 4 * d_model
        elif fusion_type == "gmu":
            self.fusion = GMU_per_feature(
                dim_w2v=2 * d_model,
                dim_mert=2 * d_model,
                hidden_dim=fusion_dim,
            )
            linear_input_dim = fusion_dim
        elif fusion_type == "scalar-gmu":
            self.fusion = ScalarGMU(
                dim_w2v=2 * d_model,
                dim_mert=2 * d_model,
                hidden_dim=fusion_dim,
            )
            linear_input_dim = fusion_dim
        elif fusion_type == "add":
            self.w2v_proj = nn.Linear(w2v_dim, fusion_dim)
            self.mert_proj = nn.Linear(mert_dim, fusion_dim)
            linear_input_dim = fusion_dim
        else:
            raise ValueError(f"Unknown fusion type: {fusion_type}")

        self.linear_back = nn.Linear(linear_input_dim, 1)
        if norm_type == "layer_norm":
            raise ValueError(
                "Late fusion: use norm_type=layer_norm_per_modality (or per_modality), "
                "not layer_norm on a single fused tensor."
            )
        self._init_embedding_norm_layers(
            norm_type=norm_type,
            w2v_dim=w2v_dim,
            mert_dim=mert_dim,
            fused_dim=w2v_dim,
        )

    def forward(self, vector: torch.Tensor):
        # spliting concated vectors
        vector_w2v, vector_mert = self._split_modalities(vector)
        
        # MODALITY DROPOUT
        if self.training and self.modality_dropout:
            rand_val = torch.rand(1).item()
            if rand_val < 0.15:
                vector_w2v = torch.zeros_like(vector_w2v)
            elif rand_val < 0.30:
                vector_mert = torch.zeros_like(vector_mert)

        vector_w2v, vector_mert = self._normalize_modalities(vector_w2v, vector_mert)

        # WAV2VEC

        backward_w2v = torch.flip(vector_w2v, [1])
        x_forward_w2v = self.linear_forward_w2v(vector_w2v)
        x_backward_w2v = self.linear_backward_w2v(backward_w2v)

        # w2v forward pass
        for layer in self.mamba_forward_w2v:
            x_forward_w2v = layer(x_forward_w2v)

        # backward pass
        for layer in self.mamba_backward_w2v:
            x_backward_w2v = layer(x_backward_w2v)

        x_backward_w2v = torch.flip(x_backward_w2v, [1])
        
        if self.aggregation == "last_vector":
            x_forward_w2v = x_forward_w2v[:,-1,:]
            x_backward_w2v = x_backward_w2v[:,0,:]

        out_w2v = torch.cat([x_forward_w2v, x_backward_w2v], dim=-1)  # [B, 1500, 2 * d_model]

        # MERT

        backward_mert = torch.flip(vector_mert, [1])
        x_forward_mert = self.linear_forward_mert(vector_mert)
        x_backward_mert = self.linear_backward_mert(backward_mert)

        # mert forward pass
        for layer in self.mamba_forward_mert:
            x_forward_mert = layer(x_forward_mert)

        # backward pass
        for layer in self.mamba_backward_mert:
            x_backward_mert = layer(x_backward_mert)

        x_backward_mert = torch.flip(x_backward_mert, [1])
        
        if self.aggregation == "last_vector":
            x_forward_mert = x_forward_mert[:,-1,:]
            x_backward_mert = x_backward_mert[:,0,:]
        
        out_mert = torch.cat(
            [x_forward_mert, x_backward_mert], dim=-1
        )  # [B, 1500, 2 * d_model]

        # FUSION
        if self.fusion_type == "gmu":
            out = self.fusion(out_w2v, out_mert)
        elif self.fusion_type == "scalar-gmu":
            out = self.fusion(out_w2v, out_mert)
        elif self.fusion_type == "add":
            ...
        elif self.fusion_type == "concat":
            out = torch.cat([out_w2v, out_mert], dim=-1)  # [B, 1500, 4 * d_model]

        
        if self.aggregation != "last_vector":
            out = out.mean(dim=1)  # [B, 4 * d_model]
        logit = self.linear_back(out)
        return logit
