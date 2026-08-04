#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Benchmark model architectures for XNA/PC 6-mer classification.

This public version removes the dependency on the private ``config_*.py``
files while preserving the original module, class and parameter names. Existing
checkpoints produced by the original script therefore remain load-compatible.

Model input is a three-tensor tuple:

    seq_onehot:  [batch, sequence_length, 5]
    merged_feat: [batch, sequence_length, 13]
    raw_rect:    [batch, sequence_length, raw_signal_length]

``seq_onehot`` is retained for compatibility. Its information is already
included in ``merged_feat``.
"""

from __future__ import annotations

from typing import Dict, Tuple, Type

import torch
import torch.nn as nn
import torch.nn.functional as F

DEFAULT_NUM_CLASSES = 2
DEFAULT_SEQUENCE_LENGTH = 6
DEFAULT_RAW_SIGNAL_LENGTH = 30

ModelInput = Tuple[torch.Tensor, torch.Tensor, torch.Tensor]


class RawSignalEncoder(nn.Module):
    """Encode each base-level rectified raw-signal segment independently."""

    def __init__(self, signal_len: int, out_dim: int = 32):
        super().__init__()
        if signal_len <= 0:
            raise ValueError("signal_len must be greater than 0")
        self.signal_len = signal_len
        self.net = nn.Sequential(
            nn.Conv1d(1, 16, kernel_size=5, padding=2),
            nn.GELU(),
            nn.BatchNorm1d(16),
            nn.Conv1d(16, 32, kernel_size=5, padding=2),
            nn.GELU(),
            nn.BatchNorm1d(32),
            nn.Conv1d(32, out_dim, kernel_size=3, padding=1),
            nn.GELU(),
            nn.AdaptiveAvgPool1d(1),
        )

    def forward(self, raw_rect: torch.Tensor) -> torch.Tensor:
        if raw_rect.ndim != 3:
            raise ValueError(
                "raw_rect must have shape [batch, sequence_length, signal_length]"
            )
        if raw_rect.size(-1) != self.signal_len:
            raise ValueError(
                f"Expected raw signal length {self.signal_len}, "
                f"received {raw_rect.size(-1)}"
            )
        bsz, kmer_len, sig_len = raw_rect.shape
        x = raw_rect.reshape(bsz * kmer_len, 1, sig_len)
        x = self.net(x).squeeze(-1)
        x = x.reshape(bsz, kmer_len, -1)
        return x


class AttnPool(nn.Module):
    def __init__(self, dim: int):
        super().__init__()
        self.score = nn.Linear(dim, 1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        attn = torch.softmax(self.score(x).squeeze(-1), dim=1)
        pooled = torch.sum(x * attn.unsqueeze(-1), dim=1)
        return pooled


class FusionStem(nn.Module):
    """Common stem: 5-dim ATCGN one-hot + 8 scalars + raw signal."""

    def __init__(
        self,
        raw_signal_len: int,
        feature_dim: int = 13,
        raw_dim: int = 32,
        model_dim: int = 96,
    ):
        super().__init__()
        self.feature_dim = feature_dim
        self.raw_encoder = RawSignalEncoder(raw_signal_len, out_dim=raw_dim)
        self.feature_proj = nn.Sequential(
            nn.Linear(feature_dim, 32),
            nn.LayerNorm(32),
            nn.GELU(),
        )
        self.fuse_proj = nn.Sequential(
            nn.Linear(32 + raw_dim, model_dim),
            nn.LayerNorm(model_dim),
            nn.GELU(),
        )

    def forward(
        self,
        merged_feat: torch.Tensor,
        raw_rect: torch.Tensor,
    ) -> torch.Tensor:
        if merged_feat.ndim != 3:
            raise ValueError(
                "merged_feat must have shape [batch, sequence_length, feature_dim]"
            )
        if merged_feat.size(-1) != self.feature_dim:
            raise ValueError(
                f"Expected merged feature dimension {self.feature_dim}, "
                f"received {merged_feat.size(-1)}"
            )
        if merged_feat.shape[:2] != raw_rect.shape[:2]:
            raise ValueError(
                "merged_feat and raw_rect must share batch and sequence dimensions"
            )

        feat_embed = self.feature_proj(merged_feat)
        raw_embed = self.raw_encoder(raw_rect)
        x = torch.cat([feat_embed, raw_embed], dim=-1)
        x = self.fuse_proj(x)
        return x


class FusionBiGRU(nn.Module):
    def __init__(
        self,
        num_classes: int = DEFAULT_NUM_CLASSES,
        seq_len: int = DEFAULT_SEQUENCE_LENGTH,
        raw_signal_len: int = DEFAULT_RAW_SIGNAL_LENGTH,
        model_dim: int = 96,
    ):
        super().__init__()
        del seq_len
        self.stem = FusionStem(raw_signal_len=raw_signal_len, model_dim=model_dim)
        self.encoder = nn.GRU(
            input_size=model_dim,
            hidden_size=64,
            num_layers=2,
            batch_first=True,
            dropout=0.1,
            bidirectional=True,
        )
        self.pool = AttnPool(128)
        self.cls = nn.Sequential(
            nn.Linear(256, 128),
            nn.GELU(),
            nn.Dropout(0.2),
            nn.Linear(128, num_classes),
        )

    def forward(self, inputs: ModelInput) -> Tuple[torch.Tensor, torch.Tensor]:
        _, merged_feat, raw_rect = inputs
        x = self.stem(merged_feat, raw_rect)
        x, _ = self.encoder(x)
        attn_vec = self.pool(x)
        max_vec = x.max(dim=1).values
        feat = torch.cat([attn_vec, max_vec], dim=-1)
        logits = self.cls(feat)
        return logits, feat


class FusionTransformer(nn.Module):
    def __init__(
        self,
        num_classes: int = DEFAULT_NUM_CLASSES,
        seq_len: int = DEFAULT_SEQUENCE_LENGTH,
        raw_signal_len: int = DEFAULT_RAW_SIGNAL_LENGTH,
        model_dim: int = 96,
        nhead: int = 4,
        num_layers: int = 2,
    ):
        super().__init__()
        self.seq_len = seq_len
        self.stem = FusionStem(raw_signal_len=raw_signal_len, model_dim=model_dim)
        self.pos_embed = nn.Parameter(torch.zeros(1, seq_len, model_dim))
        enc_layer = nn.TransformerEncoderLayer(
            d_model=model_dim,
            nhead=nhead,
            dim_feedforward=model_dim * 2,
            dropout=0.1,
            activation="gelu",
            batch_first=True,
            norm_first=True,
        )
        self.encoder = nn.TransformerEncoder(enc_layer, num_layers=num_layers)
        self.pool = AttnPool(model_dim)
        self.cls = nn.Sequential(
            nn.Linear(model_dim * 2, 128),
            nn.GELU(),
            nn.Dropout(0.2),
            nn.Linear(128, num_classes),
        )

    def forward(self, inputs: ModelInput) -> Tuple[torch.Tensor, torch.Tensor]:
        _, merged_feat, raw_rect = inputs
        x = self.stem(merged_feat, raw_rect)
        if x.size(1) != self.seq_len:
            raise ValueError(
                f"Transformer was configured for sequence length {self.seq_len}, "
                f"received {x.size(1)}"
            )
        x = x + self.pos_embed
        x = self.encoder(x)
        mean_vec = x.mean(dim=1)
        attn_vec = self.pool(x)
        feat = torch.cat([mean_vec, attn_vec], dim=-1)
        logits = self.cls(feat)
        return logits, feat


class ConvMixerBlock(nn.Module):
    def __init__(
        self,
        dim: int,
        kernel_size: int = 3,
        dropout: float = 0.1,
    ):
        super().__init__()
        self.dw = nn.Conv1d(
            dim,
            dim,
            kernel_size=kernel_size,
            padding=kernel_size // 2,
            groups=dim,
        )
        self.pw = nn.Conv1d(dim, dim, kernel_size=1)
        self.norm1 = nn.BatchNorm1d(dim)
        self.norm2 = nn.BatchNorm1d(dim)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        residual = x
        x = self.dw(x)
        x = F.gelu(self.norm1(x))
        x = self.pw(x)
        x = self.dropout(F.gelu(self.norm2(x)))
        return x + residual


class FusionConvMixer(nn.Module):
    def __init__(
        self,
        num_classes: int = DEFAULT_NUM_CLASSES,
        seq_len: int = DEFAULT_SEQUENCE_LENGTH,
        raw_signal_len: int = DEFAULT_RAW_SIGNAL_LENGTH,
        model_dim: int = 96,
        num_blocks: int = 4,
    ):
        super().__init__()
        del seq_len
        self.stem = FusionStem(raw_signal_len=raw_signal_len, model_dim=model_dim)
        self.blocks = nn.Sequential(
            *[
                ConvMixerBlock(model_dim, kernel_size=3)
                for _ in range(num_blocks)
            ]
        )
        self.cls = nn.Sequential(
            nn.Linear(model_dim * 2, 128),
            nn.GELU(),
            nn.Dropout(0.2),
            nn.Linear(128, num_classes),
        )

    def forward(self, inputs: ModelInput) -> Tuple[torch.Tensor, torch.Tensor]:
        _, merged_feat, raw_rect = inputs
        x = self.stem(merged_feat, raw_rect)
        x = x.transpose(1, 2)
        x = self.blocks(x)
        avg_vec = x.mean(dim=-1)
        max_vec = x.max(dim=-1).values
        feat = torch.cat([avg_vec, max_vec], dim=-1)
        logits = self.cls(feat)
        return logits, feat


class LiteMambaBlock(nn.Module):
    """Dependency-free Mamba-style gated mixing block.

    This is not the official ``mamba_ssm`` implementation. It preserves the
    original study code's gated depthwise-convolutional mixing design.
    """

    def __init__(
        self,
        dim: int,
        kernel_size: int = 3,
        dropout: float = 0.1,
    ):
        super().__init__()
        self.norm = nn.LayerNorm(dim)
        self.in_proj = nn.Linear(dim, dim * 2)
        self.dw_conv = nn.Conv1d(
            dim,
            dim,
            kernel_size=kernel_size,
            padding=kernel_size // 2,
            groups=dim,
        )
        self.out_proj = nn.Linear(dim, dim)
        self.dropout = nn.Dropout(dropout)
        self.ffn = nn.Sequential(
            nn.LayerNorm(dim),
            nn.Linear(dim, dim * 2),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(dim * 2, dim),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        residual = x
        z = self.norm(x)
        u, g = self.in_proj(z).chunk(2, dim=-1)
        u = self.dw_conv(u.transpose(1, 2)).transpose(1, 2)
        u = F.silu(u)
        g = torch.sigmoid(g)
        x = residual + self.dropout(self.out_proj(u * g))
        x = x + self.dropout(self.ffn(x))
        return x


class FusionLiteMamba(nn.Module):
    def __init__(
        self,
        num_classes: int = DEFAULT_NUM_CLASSES,
        seq_len: int = DEFAULT_SEQUENCE_LENGTH,
        raw_signal_len: int = DEFAULT_RAW_SIGNAL_LENGTH,
        model_dim: int = 96,
        num_blocks: int = 4,
    ):
        super().__init__()
        del seq_len
        self.stem = FusionStem(raw_signal_len=raw_signal_len, model_dim=model_dim)
        self.blocks = nn.ModuleList(
            [
                LiteMambaBlock(model_dim, kernel_size=3)
                for _ in range(num_blocks)
            ]
        )
        self.pool = AttnPool(model_dim)
        self.cls = nn.Sequential(
            nn.Linear(model_dim * 2, 128),
            nn.GELU(),
            nn.Dropout(0.2),
            nn.Linear(128, num_classes),
        )

    def forward(self, inputs: ModelInput) -> Tuple[torch.Tensor, torch.Tensor]:
        _, merged_feat, raw_rect = inputs
        x = self.stem(merged_feat, raw_rect)
        for blk in self.blocks:
            x = blk(x)
        attn_vec = self.pool(x)
        last_vec = x[:, -1, :]
        feat = torch.cat([attn_vec, last_vec], dim=-1)
        logits = self.cls(feat)
        return logits, feat


MODEL_REGISTRY: Dict[str, Type[nn.Module]] = {
    "bigru": FusionBiGRU,
    "transformer": FusionTransformer,
    "convmixer": FusionConvMixer,
    "litemamba": FusionLiteMamba,
}


def build_model(
    model_type: str,
    num_classes: int,
    seq_len: int,
    raw_signal_len: int,
) -> nn.Module:
    """Build one registered benchmark model."""

    key = model_type.lower()
    if key not in MODEL_REGISTRY:
        available = ", ".join(sorted(MODEL_REGISTRY))
        raise ValueError(
            f"Unsupported model_type {model_type!r}. Available: {available}"
        )
    if num_classes < 2:
        raise ValueError("num_classes must be at least 2")
    if seq_len <= 0:
        raise ValueError("seq_len must be greater than 0")
    if raw_signal_len <= 0:
        raise ValueError("raw_signal_len must be greater than 0")

    cls = MODEL_REGISTRY[key]
    return cls(
        num_classes=num_classes,
        seq_len=seq_len,
        raw_signal_len=raw_signal_len,
    )
