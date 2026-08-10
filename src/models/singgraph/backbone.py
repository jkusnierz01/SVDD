"""SingGraph backbone on precomputed vocals/instrumental SSL features."""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F

from src.models.singgraph.layers import (
    GraphAttentionLayer,
    GraphPool,
    HtrgGraphAttentionLayer,
    Residual_block,
)


class SingGraphBackbone(nn.Module):
    """AASIST-style graph backend without live SSL; expects [N, T, 1024] per stream."""

    def __init__(
        self,
        feat_dim: int = 1024,
        enc_dim: int = 128,
        gat_dims: list[int] | None = None,
        pool_ratios: list[float] | None = None,
        temperatures: list[float] | None = None,
        num_nodes_s: int = 42,
        dropout: float = 0.5,
        drop_way: float = 0.2,
    ):
        super().__init__()
        gat_dims = gat_dims or [64, 32]
        pool_ratios = pool_ratios or [0.5, 0.5, 0.5, 0.5]
        temperatures = temperatures or [2.0, 2.0, 100.0, 100.0]
        filts = [128, [1, 32], [32, 32], [32, 64], [64, 64]]
        node_dim = filts[-1][-1]

        self.input_proj = nn.Linear(2 * feat_dim, enc_dim)
        self.first_bn = nn.BatchNorm2d(num_features=1)
        self.first_bn1 = nn.BatchNorm2d(num_features=node_dim)
        self.drop = nn.Dropout(dropout, inplace=True)
        self.drop_way = nn.Dropout(drop_way, inplace=True)
        self.selu = nn.SELU(inplace=True)

        self.encoder = nn.Sequential(
            nn.Sequential(Residual_block(nb_filts=filts[1], first=True)),
            nn.Sequential(Residual_block(nb_filts=filts[2])),
            nn.Sequential(Residual_block(nb_filts=filts[3])),
            nn.Sequential(Residual_block(nb_filts=filts[4])),
            nn.Sequential(Residual_block(nb_filts=filts[4])),
            nn.Sequential(Residual_block(nb_filts=filts[4])),
        )

        self.attention = nn.Sequential(
            nn.Conv2d(node_dim, 128, kernel_size=(1, 1)),
            nn.SELU(inplace=True),
            nn.BatchNorm2d(128),
            nn.Conv2d(128, node_dim, kernel_size=(1, 1)),
        )

        self.pos_S = nn.Parameter(torch.randn(1, num_nodes_s, node_dim))
        self.master1 = nn.Parameter(torch.randn(1, 1, gat_dims[0]))
        self.master2 = nn.Parameter(torch.randn(1, 1, gat_dims[0]))

        self.GAT_layer_S = GraphAttentionLayer(
            node_dim, gat_dims[0], temperature=temperatures[0]
        )
        self.GAT_layer_T = GraphAttentionLayer(
            node_dim, gat_dims[0], temperature=temperatures[1]
        )

        self.HtrgGAT_layer_ST11 = HtrgGraphAttentionLayer(
            gat_dims[0], gat_dims[1], temperature=temperatures[2]
        )
        self.HtrgGAT_layer_ST12 = HtrgGraphAttentionLayer(
            gat_dims[1], gat_dims[1], temperature=temperatures[2]
        )
        self.HtrgGAT_layer_ST21 = HtrgGraphAttentionLayer(
            gat_dims[0], gat_dims[1], temperature=temperatures[2]
        )
        self.HtrgGAT_layer_ST22 = HtrgGraphAttentionLayer(
            gat_dims[1], gat_dims[1], temperature=temperatures[2]
        )

        self.pool_S = GraphPool(pool_ratios[0], gat_dims[0], 0.3)
        self.pool_T = GraphPool(pool_ratios[1], gat_dims[0], 0.3)
        self.pool_hS1 = GraphPool(pool_ratios[2], gat_dims[1], 0.3)
        self.pool_hT1 = GraphPool(pool_ratios[2], gat_dims[1], 0.3)
        self.pool_hS2 = GraphPool(pool_ratios[2], gat_dims[1], 0.3)
        self.pool_hT2 = GraphPool(pool_ratios[2], gat_dims[1], 0.3)

        self.out_layer = nn.Linear(5 * gat_dims[1], 2)

    def _spectral_pos_encoding(self, e_s: torch.Tensor) -> torch.Tensor:
        """Match learnable pos_S to the current number of spectral nodes."""
        if e_s.size(1) == self.pos_S.size(1):
            return self.pos_S
        pos = F.interpolate(
            self.pos_S.transpose(1, 2),
            size=e_s.size(1),
            mode="linear",
            align_corners=False,
        ).transpose(1, 2)
        return pos

    def forward(self, vocals: torch.Tensor, instrumental: torch.Tensor) -> torch.Tensor:
        """
        Args:
            vocals: [N, T, feat_dim]
            instrumental: [N, T, feat_dim]

        Returns:
            window_logits: [N, 2]
        """
        x = torch.cat([vocals, instrumental], dim=-1)
        x = self.input_proj(x)
        x = x.transpose(1, 2).unsqueeze(1)
        x = F.max_pool2d(x, (3, 3))
        x = self.first_bn(x)
        x = self.selu(x)

        x = self.encoder(x)
        x = self.first_bn1(x)
        x = self.selu(x)

        w = self.attention(x)

        w1 = F.softmax(w, dim=-1)
        m = torch.sum(x * w1, dim=-1)
        e_s = m.transpose(1, 2) + self._spectral_pos_encoding(m.transpose(1, 2))

        gat_s = self.GAT_layer_S(e_s)
        out_s = self.pool_S(gat_s)

        w2 = F.softmax(w, dim=-2)
        m1 = torch.sum(x * w2, dim=-2)
        e_t = m1.transpose(1, 2)

        gat_t = self.GAT_layer_T(e_t)
        out_t = self.pool_T(gat_t)

        out_t1, out_s1, master1 = self.HtrgGAT_layer_ST11(
            out_t, out_s, master=self.master1
        )
        out_s1 = self.pool_hS1(out_s1)
        out_t1 = self.pool_hT1(out_t1)

        out_t_aug, out_s_aug, master_aug = self.HtrgGAT_layer_ST12(
            out_t1, out_s1, master=master1
        )
        out_t1 = out_t1 + out_t_aug
        out_s1 = out_s1 + out_s_aug
        master1 = master1 + master_aug

        out_t2, out_s2, master2 = self.HtrgGAT_layer_ST21(
            out_t, out_s, master=self.master2
        )
        out_s2 = self.pool_hS2(out_s2)
        out_t2 = self.pool_hT2(out_t2)

        out_t_aug, out_s_aug, master_aug = self.HtrgGAT_layer_ST22(
            out_t2, out_s2, master=master2
        )
        out_t2 = out_t2 + out_t_aug
        out_s2 = out_s2 + out_s_aug
        master2 = master2 + master_aug

        out_t1 = self.drop_way(out_t1)
        out_t2 = self.drop_way(out_t2)
        out_s1 = self.drop_way(out_s1)
        out_s2 = self.drop_way(out_s2)
        master1 = self.drop_way(master1)
        master2 = self.drop_way(master2)

        out_t = torch.max(out_t1, out_t2)
        out_s = torch.max(out_s1, out_s2)
        master = torch.max(master1, master2)

        t_max, _ = torch.max(torch.abs(out_t), dim=1)
        t_avg = torch.mean(out_t, dim=1)
        s_max, _ = torch.max(torch.abs(out_s), dim=1)
        s_avg = torch.mean(out_s, dim=1)

        last_hidden = torch.cat([t_max, t_avg, s_max, s_avg, master.squeeze(1)], dim=1)
        last_hidden = self.drop(last_hidden)
        return self.out_layer(last_hidden)
