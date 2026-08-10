import torch
import torch.nn as nn
from typing import Callable, Optional

from src.models.mamba import BaseDeepfakeModel


class AdaptedCLAMModel(BaseDeepfakeModel):
    def __init__(
        self,
        w2v_dim: int = 1024,
        mert_dim: int = 1024,
        proj_dim: int = 512,
        num_heads: int = 4,
        triplet_weight: float = 0.5,
        margin: float = 1.0,
        optimizer: Callable = None,
        scheduler: Optional[Callable] = None,
        label_smoothing: float = 0.1,
    ):
        super().__init__(
            optimizer=optimizer, scheduler=scheduler, label_smoothing=label_smoothing
        )
        self.save_hyperparameters(ignore=["optimizer", "scheduler"])

        self.w2v_dim = w2v_dim
        self.mert_dim = mert_dim
        self.proj_dim = proj_dim
        self.triplet_weight = triplet_weight

        # Normalization layers before Attention
        self.norm_w2v = nn.LayerNorm(w2v_dim)
        self.norm_mert = nn.LayerNorm(mert_dim)

        # Temporal Intra-Stream Self-Attention
        self.attn_w2v = nn.MultiheadAttention(
            embed_dim=w2v_dim, num_heads=num_heads, batch_first=True
        )
        self.attn_mert = nn.MultiheadAttention(
            embed_dim=mert_dim, num_heads=num_heads, batch_first=True
        )

        # Projection
        self.proj_w2v_layer = nn.Linear(w2v_dim, proj_dim)
        self.proj_mert_layer = nn.Linear(mert_dim, proj_dim)

        # Classification
        self.classifier = nn.Linear(2 * proj_dim, 1)

        # Triplet Loss
        self.triplet_loss = nn.TripletMarginLoss(margin=margin)

    def _split_modalities(self, vector: torch.Tensor):
        w2v = vector[..., : self.w2v_dim]
        mert = vector[..., self.w2v_dim :]
        return w2v, mert

    def forward_features(self, x: torch.Tensor):
        w2v_emb, mert_emb = self._split_modalities(x)

        # Pre-normalization
        w2v_norm = self.norm_w2v(w2v_emb)
        mert_norm = self.norm_mert(mert_emb)

        # Self-attention
        # MultiheadAttention returns (attn_output, attn_output_weights)
        w2v_attn, _ = self.attn_w2v(w2v_norm, w2v_norm, w2v_norm)
        mert_attn, _ = self.attn_mert(mert_norm, mert_norm, mert_norm)

        # Aggregation (Mean pooling along time dim=1)
        w2v_pooled = w2v_attn.mean(dim=1)
        mert_pooled = mert_attn.mean(dim=1)

        # Projection
        proj_w2v = self.proj_w2v_layer(w2v_pooled)
        proj_mert = self.proj_mert_layer(mert_pooled)

        # Fusion
        fused = torch.cat([proj_w2v, proj_mert], dim=-1)

        # Classification
        logit = self.classifier(fused)

        return logit, proj_w2v, proj_mert

    def forward(self, x: torch.Tensor):
        # We only return logit during standard forward (e.g. for validation/test step in BaseDeepfakeModel)
        logit, _, _ = self.forward_features(x)
        return logit

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

        logit, proj_w2v, proj_mert = self.forward_features(x)
        y_pred = logit.squeeze(-1)

        bce_loss = self.loss_fn(y_pred, y_true.float())

        # Triplet Loss only for authentic (label == 0)
        real_indices = (y_true == 0).nonzero(as_tuple=True)[0]

        if len(real_indices) > 1:
            anchor = proj_mert[real_indices]
            positive = proj_w2v[real_indices]

            # Rolled positive for negative
            negative = torch.roll(positive, shifts=1, dims=0)

            triplet_loss = self.triplet_loss(anchor, positive, negative)
            loss = bce_loss + self.triplet_weight * triplet_loss
            self.log("train/triplet_loss", triplet_loss, on_epoch=True, prog_bar=True)
        else:
            loss = bce_loss
            if len(real_indices) == 1:
                # Log 0 if only one real sample to avoid metric disappearing
                self.log("train/triplet_loss", 0.0, on_epoch=True, prog_bar=True)

        probs = torch.sigmoid(y_pred)
        preds = (probs > 0.5).int()
        acc = self.accuracy(preds, y_true)

        self.log("train/bce_loss", bce_loss, on_epoch=True, prog_bar=False)
        self.log("train/acc", acc, on_epoch=True, prog_bar=True)
        self.log("train/loss", loss, on_epoch=True, prog_bar=True)

        return loss
