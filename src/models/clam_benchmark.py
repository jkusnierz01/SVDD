import torch
import torch.nn as nn
from typing import Callable, Optional

from src.models.mamba import BaseDeepfakeModel


class ClamBenchmarkModel(BaseDeepfakeModel):
    """CLAM on pre-extracted all-layer features [B, T, 13, 768] per modality."""

    def __init__(
        self,
        num_layers: int = 13,
        layer_dim: int = 768,
        proj_dim: int = 512,
        num_heads: int = 4,
        triplet_weight: float = 0.5,
        margin: float = 1.0,
        dropout: float = 0.3,
        classifier_hidden: int = 256,
        optimizer: Callable = None,
        scheduler: Optional[Callable] = None,
        label_smoothing: float = 0.1,
    ):
        super().__init__(
            optimizer=optimizer, scheduler=scheduler, label_smoothing=label_smoothing
        )
        self.save_hyperparameters(ignore=["optimizer", "scheduler"])

        self.num_layers = num_layers
        self.layer_dim = layer_dim
        self.proj_dim = proj_dim
        self.triplet_weight = triplet_weight

        self.wca_w2v = nn.Conv1d(num_layers, 1, kernel_size=1)
        self.wca_mert = nn.Conv1d(num_layers, 1, kernel_size=1)

        self.norm_w2v = nn.LayerNorm(layer_dim)
        self.norm_mert = nn.LayerNorm(layer_dim)

        self.attn_w2v = nn.MultiheadAttention(
            embed_dim=layer_dim, num_heads=num_heads, batch_first=True
        )
        self.attn_mert = nn.MultiheadAttention(
            embed_dim=layer_dim, num_heads=num_heads, batch_first=True
        )

        self.proj_w2v_layer = nn.Linear(layer_dim, proj_dim)
        self.proj_mert_layer = nn.Linear(layer_dim, proj_dim)

        self.classifier = nn.Sequential(
            nn.Linear(2 * proj_dim, classifier_hidden),
            nn.Dropout(dropout),
            nn.Linear(classifier_hidden, 1),
        )

        self.triplet_loss = nn.TripletMarginLoss(margin=margin)

    def _wca(self, x: torch.Tensor, conv: nn.Conv1d) -> torch.Tensor:
        """[B, T, L, D] -> [B, T, D] via layer-wise Conv1d."""
        b, t, layers, dim = x.shape
        flat = x.reshape(b * t, layers, dim)
        out = conv(flat).squeeze(1)
        return out.view(b, t, dim)

    def _pad_mask(self, lengths: torch.Tensor, max_len: int) -> torch.Tensor:
        """True for padding positions (MHA key_padding_mask)."""
        return torch.arange(max_len, device=lengths.device)[None, :] >= lengths[:, None]

    def _masked_mean(self, x: torch.Tensor, lengths: torch.Tensor) -> torch.Tensor:
        """Mean pool over time dim=1 ignoring padding."""
        b, t, _ = x.shape
        mask = torch.arange(t, device=x.device)[None, :] < lengths[:, None]
        mask = mask.unsqueeze(-1).float()
        summed = (x * mask).sum(dim=1)
        return summed / lengths.unsqueeze(-1).float().clamp(min=1)

    def forward_features(
        self,
        w2v: torch.Tensor,
        mert: torch.Tensor,
        lengths: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        w2v_emb = self._wca(w2v, self.wca_w2v)
        mert_emb = self._wca(mert, self.wca_mert)

        w2v_norm = self.norm_w2v(w2v_emb)
        mert_norm = self.norm_mert(mert_emb)

        pad_mask = self._pad_mask(lengths, w2v_norm.shape[1])

        w2v_attn, _ = self.attn_w2v(
            w2v_norm, w2v_norm, w2v_norm, key_padding_mask=pad_mask
        )
        mert_attn, _ = self.attn_mert(
            mert_norm, mert_norm, mert_norm, key_padding_mask=pad_mask
        )

        w2v_pooled = self._masked_mean(w2v_attn, lengths)
        mert_pooled = self._masked_mean(mert_attn, lengths)

        proj_w2v = self.proj_w2v_layer(w2v_pooled)
        proj_mert = self.proj_mert_layer(mert_pooled)

        fused = torch.cat([proj_w2v, proj_mert], dim=-1)
        logit = self.classifier(fused)

        return logit, proj_w2v, proj_mert

    def _unpack_batch(self, batch):
        if len(batch) == 3:
            (w2v, mert), y_true, lengths = batch
        else:
            (w2v, mert), y_true = batch
            lengths = torch.full(
                (w2v.shape[0],), w2v.shape[1], dtype=torch.long, device=w2v.device
            )
        return w2v, mert, y_true, lengths

    def forward(self, batch):
        w2v, mert, _, lengths = self._unpack_batch(batch)
        logit, _, _ = self.forward_features(w2v, mert, lengths)
        return logit

    def _compute_loss(self, batch, log_prefix: str):
        w2v, mert, y_true, lengths = self._unpack_batch(batch)

        logit, proj_w2v, proj_mert = self.forward_features(w2v, mert, lengths)
        y_pred = logit.squeeze(-1)

        bce_loss = self.loss_fn(y_pred, y_true.float())

        real_indices = (y_true == 0).nonzero(as_tuple=True)[0]

        if len(real_indices) > 1:
            anchor = proj_mert[real_indices]
            positive = proj_w2v[real_indices]
            negative = torch.roll(positive, shifts=1, dims=0)
            triplet_loss = self.triplet_loss(anchor, positive, negative)
            loss = bce_loss + self.triplet_weight * triplet_loss
            self.log(f"{log_prefix}/triplet_loss", triplet_loss, on_epoch=True, prog_bar=True)
        else:
            loss = bce_loss
            if len(real_indices) == 1:
                self.log(f"{log_prefix}/triplet_loss", 0.0, on_epoch=True, prog_bar=True)

        probs = torch.sigmoid(y_pred)
        preds = (probs > 0.5).int()
        acc = self.accuracy(preds, y_true)

        self.log(f"{log_prefix}/bce_loss", bce_loss, on_epoch=True, prog_bar=False)
        self.log(f"{log_prefix}/acc", acc, on_epoch=True, prog_bar=True)
        self.log(f"{log_prefix}/loss", loss, on_epoch=True, prog_bar=True)

        return loss

    def training_step(self, train_batch, batch_idx):
        w2v, mert, _, _ = self._unpack_batch(train_batch)

        if batch_idx == 0:
            x = torch.cat([w2v, mert], dim=-1)
            print(
                f"DEBUG Batch 0 stats: Mean={x.mean():.4f}, Std={x.std():.4f}, "
                f"Min={x.min():.4f}, Max={x.max():.4f}, shape={tuple(w2v.shape)}"
            )
            if x.std() < 1e-5:
                print(
                    "WARNING: Input features have zero variance! Model collapse imminent."
                )

        return self._compute_loss(train_batch, "train")

    def validation_step(self, val_batch, batch_idx):
        loss = self._compute_loss(val_batch, "val")
        w2v, mert, y_true, lengths = self._unpack_batch(val_batch)
        logit, _, _ = self.forward_features(w2v, mert, lengths)
        y_pred = logit.squeeze(-1)
        self.validation_outputs["predictions"].append(y_pred)
        self.validation_outputs["targets"].append(y_true)
        return loss

    def test_step(self, batch, batch_idx, dataloader_idx: int = 0):
        w2v, mert, y_true, lengths = self._unpack_batch(batch)
        y_true = y_true.float()
        logit, _, _ = self.forward_features(w2v, mert, lengths)
        y_pred = logit.squeeze(-1)
        loss = self.loss_fn(y_pred, y_true.float())

        if dataloader_idx not in self.test_outputs:
            self.test_outputs[dataloader_idx] = {"predictions": [], "targets": []}
        self.test_outputs[dataloader_idx]["predictions"].append(y_pred)
        self.test_outputs[dataloader_idx]["targets"].append(y_true)
        return loss
