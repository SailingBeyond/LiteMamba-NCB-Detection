"""Comparable six-token fusion models for six-class XNA classification.

All four backbones share the input stem, target-aware readout and classifier.
``litemamba`` uses a real input-dependent selective state-space recurrence;
``legacy_gatedconv`` is available explicitly as a non-SSM ablation.
The state-space sequence dimension is the six base-level tokens, not the
individual current samples within one base-level signal segment.
The fixed class order is A, T, C, G, X, Y. Sequence masking belongs to the
dataset: the fourth one-hot position is N while its measured features remain.
The original submitted LiteMamba used a gated convolution rather than an SSM;
the explicit legacy variant preserves that mixer for an honest ablation.
"""

from __future__ import annotations

import math
from typing import Dict, Tuple

import torch
from torch import nn
import torch.nn.functional as F


BASE_DIM = 5
CLASS_NAMES = ("A", "T", "C", "G", "X", "Y")
ONE_HOT_BASE_ORDER = ("A", "C", "T", "G", "N")
SCALAR_FEATURES = (
    "mean", "std", "median", "dwell", "quality", "mismatch", "insertion", "deletion"
)
MERGED_FEATURE_DIM = BASE_DIM + len(SCALAR_FEATURES)
READOUT_CHOICES = ("target_attention", "legacy_last_attention", "global_attention_only")
LITEMAMBA_VARIANTS = ("selective_ssm", "legacy_gatedconv")
FEATURE_GROUPS = {
    "context": (0, 5),
    "statistics": (5, 9),
    "responses": (9, 13),
}
FEATURE_SCHEMES = (
    "full", "raw_only", "context_only", "responses_only", "statistics_only",
    "no_raw", "no_context", "no_responses", "no_statistics",
)
_SCHEME_GROUPS = {
    "full": (True, ("context", "statistics", "responses")),
    "raw_only": (True, ()),
    "context_only": (False, ("context",)),
    "responses_only": (False, ("responses",)),
    "statistics_only": (False, ("statistics",)),
    "no_raw": (False, ("context", "statistics", "responses")),
    "no_context": (True, ("statistics", "responses")),
    "no_responses": (True, ("context", "statistics")),
    "no_statistics": (True, ("context", "responses")),
}


class RawSignalEncoder(nn.Module):
    """Encode each position's deterministically resampled signal independently."""

    def __init__(self, signal_len: int = 30, out_dim: int = 32):
        super().__init__()
        self.signal_len = signal_len
        self.net = nn.Sequential(
            nn.Conv1d(1, 16, kernel_size=5, padding=2), nn.GELU(), nn.BatchNorm1d(16),
            nn.Conv1d(16, 32, kernel_size=5, padding=2), nn.GELU(), nn.BatchNorm1d(32),
            nn.Conv1d(32, out_dim, kernel_size=3, padding=1), nn.GELU(),
            nn.AdaptiveAvgPool1d(1),
        )

    def forward(self, raw_rect: torch.Tensor) -> torch.Tensor:
        batch, length, samples = raw_rect.shape
        x = self.net(raw_rect.reshape(batch * length, 1, samples)).squeeze(-1)
        return x.reshape(batch, length, -1)


class FusionStem(nn.Module):
    """ACTGN channel-order one-hot + eight scalars, fused with a signal CNN."""

    def __init__(self, raw_signal_len: int = 30, feature_dim: int = MERGED_FEATURE_DIM,
                 raw_dim: int = 32, model_dim: int = 96, feature_scheme: str = "full"):
        super().__init__()
        if feature_dim != MERGED_FEATURE_DIM:
            raise ValueError("merged features must contain five one-hot channels and eight scalar channels")
        if feature_scheme not in FEATURE_SCHEMES:
            raise ValueError(f"Unsupported feature_scheme: {feature_scheme}. Available: {FEATURE_SCHEMES}")
        self.feature_scheme = feature_scheme
        self.raw_dim = raw_dim
        self.use_raw, groups = _SCHEME_GROUPS[feature_scheme]
        self.use_tabular = bool(groups)
        mask = torch.zeros(MERGED_FEATURE_DIM)
        for name in groups:
            begin, end = FEATURE_GROUPS[name]
            mask[begin:end] = 1.0
        self.register_buffer("feature_mask", mask)
        self.raw_encoder = RawSignalEncoder(raw_signal_len, raw_dim)
        self.feature_proj = nn.Sequential(
            nn.Linear(feature_dim, 32), nn.LayerNorm(32), nn.GELU(),
        )
        self.fuse_proj = nn.Sequential(
            nn.Linear(32 + raw_dim, model_dim), nn.LayerNorm(model_dim), nn.GELU(),
        )
        # Excluded branches are neither evaluated (including BatchNorm state)
        # nor counted as trainable parameters in feature-ablation reports.
        if not self.use_raw:
            self.raw_encoder.requires_grad_(False)
        if not self.use_tabular:
            self.feature_proj.requires_grad_(False)

    def forward(self, merged_feat: torch.Tensor, raw_rect: torch.Tensor) -> torch.Tensor:
        if merged_feat.ndim != 3 or merged_feat.size(-1) != MERGED_FEATURE_DIM:
            raise ValueError(f"Expected merged features (batch, six positions, {MERGED_FEATURE_DIM})")
        if raw_rect.ndim != 3 or raw_rect.shape[:2] != merged_feat.shape[:2]:
            raise ValueError("raw signal must have shape (batch, positions, samples) matching merged features")
        shape = merged_feat.shape[:2]
        features = (
            self.feature_proj(merged_feat * self.feature_mask)
            if self.use_tabular else merged_feat.new_zeros(*shape, 32)
        )
        raw = (
            self.raw_encoder(raw_rect)
            if self.use_raw else merged_feat.new_zeros(*shape, self.raw_dim)
        )
        return self.fuse_proj(torch.cat((features, raw), dim=-1))


class TargetAttentionReadout(nn.Module):
    """Concatenate global attention pooling and the fourth-position representation.

    The fixed target is indexed after contextual encoding. ``legacy_last_attention``
    instead uses the sixth/last position as a deliberately named readout ablation.
    All architectures use this same module and the same output dimensionality.
    ``global_attention_only`` removes the direct target branch by replacing it
    with zeros. The classifier keeps its original width and initialization;
    only its first half input columns can receive a data-dependent gradient.
    """

    def __init__(self, dim: int = 96, target_index: int = 3, mode: str = "target_attention"):
        super().__init__()
        if mode not in READOUT_CHOICES:
            raise ValueError(f"Unknown readout {mode}; choose from {READOUT_CHOICES}")
        self.score = nn.Linear(dim, 1)
        self.target_index = int(target_index)
        self.mode = mode

    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        if not 0 <= self.target_index < x.size(1):
            raise ValueError("target_index is outside the sequence")
        attention = torch.softmax(self.score(x).squeeze(-1).float(), dim=1).to(x.dtype)
        pooled = (attention.unsqueeze(-1) * x).sum(dim=1)
        if self.mode == "global_attention_only":
            embedding = torch.cat((pooled, torch.zeros_like(pooled)), dim=-1)
            return embedding, attention
        index = self.target_index if self.mode == "target_attention" else x.size(1) - 1
        embedding = torch.cat((pooled, x[:, index, :]), dim=-1)
        return embedding, attention


class _FusionBase(nn.Module):
    def __init__(self, num_classes: int = 6, seq_len: int = 6, raw_signal_len: int = 30,
                 model_dim: int = 96, target_index: int = 3, dropout: float = 0.1,
                 readout: str = "target_attention", feature_scheme: str = "full",
                 classifier_dropout: float = 0.2):
        super().__init__()
        if num_classes < 2 or seq_len < 1 or raw_signal_len < 1 or model_dim < 1:
            raise ValueError("num_classes >= 2, seq_len/raw_signal_len/model_dim >= 1 are required")
        if not 0 <= target_index < seq_len:
            raise ValueError("target_index must be a zero-based valid sequence index")
        if not 0 <= dropout < 1 or not 0 <= classifier_dropout < 1:
            raise ValueError("dropout probabilities must lie in [0, 1)")
        self.seq_len = seq_len
        self.stem = FusionStem(raw_signal_len, model_dim=model_dim, feature_scheme=feature_scheme)
        self.readout = TargetAttentionReadout(model_dim, target_index, readout)
        self.cls = nn.Sequential(
            nn.Linear(model_dim * 2, 128), nn.GELU(), nn.Dropout(classifier_dropout),
            nn.Linear(128, num_classes),
        )
        self.model_config: Dict[str, object] = dict(
            num_classes=num_classes, seq_len=seq_len, raw_signal_len=raw_signal_len,
            class_names=list(CLASS_NAMES) if num_classes == len(CLASS_NAMES) else [str(i) for i in range(num_classes)],
            one_hot_base_order=list(ONE_HOT_BASE_ORDER),
            model_dim=model_dim, target_index=target_index, dropout=dropout,
            classifier_dropout=classifier_dropout, readout=readout,
            feature_scheme=feature_scheme, base_dim=BASE_DIM,
            scalar_features=list(SCALAR_FEATURES),
        )
        if readout == "global_attention_only":
            self.model_config.update(
                direct_target_readout_enabled=False,
                direct_target_branch="zeros; fourth token remains in global attention and contextual encoding",
                classifier_input_width=model_dim * 2,
                classifier_data_active_input_width=model_dim,
                classifier_data_inactive_input_weight_count=model_dim * 128,
                parameter_count_note="Nominal/trainable parameter counts retain the original classifier width; its zero-input half has zero data gradient, so effective capacity is not identical.",
            )

    def _stem(self, inputs):
        if len(inputs) != 3:
            raise ValueError("model inputs must be (sequence_onehot, merged_features, raw_signal)")
        _, merged_feat, raw_rect = inputs
        if merged_feat.size(1) != self.seq_len:
            raise ValueError(f"Expected {self.seq_len} base-level tokens, got {merged_feat.size(1)}")
        return self.stem(merged_feat, raw_rect)

    def _finish(self, tokens: torch.Tensor, return_aux: bool = False):
        embedding, attention = self.readout(tokens)
        logits = self.cls(embedding)
        if return_aux:
            return logits, embedding, {"attention": attention}
        return logits, embedding


class FusionBiGRU(_FusionBase):
    def __init__(self, num_classes=6, seq_len=6, raw_signal_len=30, model_dim=96,
                 gru_hidden=64, num_layers=2, **kwargs):
        super().__init__(num_classes, seq_len, raw_signal_len, model_dim, **kwargs)
        if gru_hidden < 1 or num_layers < 1:
            raise ValueError("gru_hidden and num_layers must be positive")
        dropout = float(self.model_config["dropout"])
        self.encoder = nn.GRU(
            model_dim, gru_hidden, num_layers=num_layers, batch_first=True,
            dropout=dropout if num_layers > 1 else 0.0, bidirectional=True,
        )
        self.output_proj = nn.Linear(gru_hidden * 2, model_dim)
        self.model_config.update(model_type="bigru", gru_hidden=gru_hidden, num_layers=num_layers)

    def forward(self, inputs, return_aux: bool = False):
        tokens, _ = self.encoder(self._stem(inputs))
        return self._finish(self.output_proj(tokens), return_aux)


class FusionTransformer(_FusionBase):
    def __init__(self, num_classes=6, seq_len=6, raw_signal_len=30, model_dim=96,
                 nhead=4, num_layers=2, **kwargs):
        super().__init__(num_classes, seq_len, raw_signal_len, model_dim, **kwargs)
        if nhead < 1 or num_layers < 1:
            raise ValueError("nhead and num_layers must be positive")
        if model_dim % nhead != 0:
            raise ValueError("model_dim must be divisible by nhead")
        self.pos_embed = nn.Parameter(torch.zeros(1, seq_len, model_dim))
        layer = nn.TransformerEncoderLayer(
            d_model=model_dim, nhead=nhead, dim_feedforward=model_dim * 2,
            dropout=float(self.model_config["dropout"]), activation="gelu",
            batch_first=True, norm_first=True,
        )
        self.encoder = nn.TransformerEncoder(layer, num_layers=num_layers)
        self.model_config.update(model_type="transformer", nhead=nhead, num_layers=num_layers)

    def forward(self, inputs, return_aux: bool = False):
        tokens = self._stem(inputs) + self.pos_embed
        return self._finish(self.encoder(tokens), return_aux)


class ConvMixerBlock(nn.Module):
    def __init__(self, dim: int, kernel_size: int = 3, dropout: float = 0.1):
        super().__init__()
        self.dw = nn.Conv1d(dim, dim, kernel_size, padding=kernel_size // 2, groups=dim)
        self.pw = nn.Conv1d(dim, dim, 1)
        self.norm1, self.norm2 = nn.BatchNorm1d(dim), nn.BatchNorm1d(dim)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        residual = x
        x = F.gelu(self.norm1(self.dw(x)))
        return residual + self.dropout(F.gelu(self.norm2(self.pw(x))))


class FusionConvMixer(_FusionBase):
    def __init__(self, num_classes=6, seq_len=6, raw_signal_len=30, model_dim=96,
                 num_blocks=4, **kwargs):
        super().__init__(num_classes, seq_len, raw_signal_len, model_dim, **kwargs)
        if num_blocks < 1:
            raise ValueError("num_blocks must be positive")
        self.blocks = nn.Sequential(*[
            ConvMixerBlock(model_dim, dropout=float(self.model_config["dropout"]))
            for _ in range(num_blocks)
        ])
        self.model_config.update(model_type="convmixer", num_blocks=num_blocks)

    def forward(self, inputs, return_aux: bool = False):
        tokens = self.blocks(self._stem(inputs).transpose(1, 2)).transpose(1, 2)
        return self._finish(tokens, return_aux)


class SelectiveStateSpaceMixer(nn.Module):
    """Small selective SSM with an explicit, differentiable PyTorch scan.

    For each base-level token t and each hidden channel:
        delta_t = softplus(W_delta u_t + b_delta)
        B_t, C_t = input-dependent projections of u_t
        h_t = exp(delta_t * A) h_(t-1) + delta_t * B_t * u_t
        y_t = sum(C_t * h_t) + D * u_t
    A = -exp(A_log) ensures nonpositive continuous-time state eigenvalues.
    The input term uses the usual Mamba delta-B discretization. This is a
    transparent reference implementation, not the official optimized CUDA kernel.
    Shared parameters are used for forward and reverse scans when bidirectional.
    """

    def __init__(self, dim: int, state_dim: int = 16, kernel_size: int = 3,
                 bidirectional: bool = True):
        super().__init__()
        if state_dim < 1 or kernel_size < 1:
            raise ValueError("state_dim and kernel_size must be positive")
        self.dim, self.state_dim = dim, state_dim
        self.kernel_size, self.bidirectional = kernel_size, bidirectional
        self.dt_rank = max(1, math.ceil(dim / 16))
        self.in_proj = nn.Linear(dim, dim * 2)
        # Causal convolution in each direction; both flanks are available after averaging.
        self.dw_conv = nn.Conv1d(dim, dim, kernel_size, padding=kernel_size - 1, groups=dim)
        self.x_proj = nn.Linear(dim, self.dt_rank + state_dim * 2, bias=False)
        self.dt_proj = nn.Linear(self.dt_rank, dim, bias=True)
        self.A_log = nn.Parameter(torch.arange(1, state_dim + 1).float().log().repeat(dim, 1))
        self.D = nn.Parameter(torch.ones(dim))
        self.out_proj = nn.Linear(dim, dim)
        nn.init.uniform_(self.dt_proj.weight, -self.dt_rank ** -0.5, self.dt_rank ** -0.5)
        dt = torch.exp(torch.rand(dim) * (math.log(0.1) - math.log(0.001)) + math.log(0.001))
        with torch.no_grad():
            self.dt_proj.bias.copy_(dt + torch.log(-torch.expm1(-dt)))

    def _direction(self, x: torch.Tensor) -> torch.Tensor:
        u, gate = self.in_proj(x).chunk(2, dim=-1)
        u = self.dw_conv(u.transpose(1, 2))[..., :x.size(1)].transpose(1, 2)
        u = F.silu(u)
        delta_low, B, C = self.x_proj(u).split((self.dt_rank, self.state_dim, self.state_dim), dim=-1)
        # Recurrence in float32 also under AMP, to avoid half-precision state overflow.
        delta = F.softplus(self.dt_proj(delta_low).float())
        u_float, B_float, C_float = u.float(), B.float(), C.float()
        A = -self.A_log.float().exp()
        state = torch.zeros(x.size(0), self.dim, self.state_dim, dtype=torch.float32, device=x.device)
        outputs = []
        for t in range(x.size(1)):
            dt_t = delta[:, t, :].unsqueeze(-1)
            transition = torch.exp(dt_t * A.unsqueeze(0))
            drive = dt_t * B_float[:, t, :].unsqueeze(1) * u_float[:, t, :].unsqueeze(-1)
            state = transition * state + drive
            y = (state * C_float[:, t, :].unsqueeze(1)).sum(-1)
            outputs.append(y + self.D.float() * u_float[:, t, :])
        mixed = torch.stack(outputs, dim=1).to(u.dtype)
        return self.out_proj(mixed * F.silu(gate))

    def forward(self, x):
        forward = self._direction(x)
        if not self.bidirectional:
            return forward
        reverse = self._direction(x.flip(1)).flip(1)
        return (forward + reverse) * 0.5


class LiteMambaBlock(nn.Module):
    """Residual selective state-space mixer followed by a channel FFN."""

    def __init__(self, dim=96, state_dim=16, dropout=0.1, bidirectional=True):
        super().__init__()
        self.norm = nn.LayerNorm(dim)
        self.mixer = SelectiveStateSpaceMixer(dim, state_dim=state_dim, bidirectional=bidirectional)
        self.dropout = nn.Dropout(dropout)
        self.ffn = nn.Sequential(
            nn.LayerNorm(dim), nn.Linear(dim, dim * 2), nn.GELU(),
            nn.Dropout(dropout), nn.Linear(dim * 2, dim),
        )

    def forward(self, x):
        x = x + self.dropout(self.mixer(self.norm(x)))
        return x + self.dropout(self.ffn(x))


class LegacyGatedConvBlock(nn.Module):
    """Original gated convolution block; this block does not implement an SSM."""

    def __init__(self, dim=96, kernel_size=3, dropout=0.1):
        super().__init__()
        self.norm = nn.LayerNorm(dim)
        self.in_proj = nn.Linear(dim, dim * 2)
        self.dw_conv = nn.Conv1d(dim, dim, kernel_size, padding=kernel_size // 2, groups=dim)
        self.out_proj = nn.Linear(dim, dim)
        self.dropout = nn.Dropout(dropout)
        self.ffn = nn.Sequential(
            nn.LayerNorm(dim), nn.Linear(dim, dim * 2), nn.GELU(),
            nn.Dropout(dropout), nn.Linear(dim * 2, dim),
        )

    def forward(self, x):
        u, gate = self.in_proj(self.norm(x)).chunk(2, dim=-1)
        u = F.silu(self.dw_conv(u.transpose(1, 2)).transpose(1, 2))
        x = x + self.dropout(self.out_proj(u * torch.sigmoid(gate)))
        return x + self.dropout(self.ffn(x))


class FusionLiteMamba(_FusionBase):
    def __init__(self, num_classes=6, seq_len=6, raw_signal_len=30, model_dim=96,
                 num_blocks=4, state_dim=16, ssm_bidirectional=True,
                 litemamba_variant="selective_ssm", **kwargs):
        super().__init__(num_classes, seq_len, raw_signal_len, model_dim, **kwargs)
        if num_blocks < 1 or state_dim < 1:
            raise ValueError("num_blocks and state_dim must be positive")
        if litemamba_variant not in LITEMAMBA_VARIANTS:
            raise ValueError(f"Unknown LiteMamba variant: {litemamba_variant}")
        dropout = float(self.model_config["dropout"])
        if litemamba_variant == "selective_ssm":
            blocks = [LiteMambaBlock(model_dim, state_dim, dropout, ssm_bidirectional) for _ in range(num_blocks)]
        else:
            blocks = [LegacyGatedConvBlock(model_dim, dropout=dropout) for _ in range(num_blocks)]
        self.blocks = nn.ModuleList(blocks)
        self.model_config.update(
            model_type="litemamba", num_blocks=num_blocks, state_dim=state_dim,
            ssm_bidirectional=ssm_bidirectional, litemamba_variant=litemamba_variant,
            ssm_sequence_axis="base_level_tokens" if litemamba_variant == "selective_ssm" else None,
            ssm_parameters_shared_between_directions=(litemamba_variant == "selective_ssm" and ssm_bidirectional),
        )

    def forward(self, inputs, return_aux: bool = False):
        tokens = self._stem(inputs)
        for block in self.blocks:
            tokens = block(tokens)
        return self._finish(tokens, return_aux)


MODEL_REGISTRY = {
    "bigru": FusionBiGRU, "transformer": FusionTransformer,
    "convmixer": FusionConvMixer, "litemamba": FusionLiteMamba,
}


def build_model(model_type: str, num_classes: int = 6, seq_len: int = 6,
                raw_signal_len: int = 30, **kwargs):
    """Build a model; common kwargs are identical across the four backbones.

    Backbone-specific optional kwargs can be supplied in a common configuration;
    only those applicable to the selected backbone are used. Unknown names raise.
    The exact resolved settings are available in ``model.model_config``.
    """
    key = model_type.lower()
    if key not in MODEL_REGISTRY:
        raise ValueError(f"Unsupported model_type: {model_type}. Available: {sorted(MODEL_REGISTRY)}")
    common = {"model_dim", "target_index", "dropout", "classifier_dropout", "readout", "feature_scheme"}
    specific = {
        "bigru": {"gru_hidden", "num_layers"},
        "transformer": {"nhead", "num_layers"},
        "convmixer": {"num_blocks"},
        "litemamba": {"num_blocks", "state_dim", "ssm_bidirectional", "litemamba_variant"},
    }
    known = common.union(*(names for names in specific.values()))
    unknown = set(kwargs) - known
    if unknown:
        raise TypeError(f"Unknown model options: {sorted(unknown)}")
    options = {name: value for name, value in kwargs.items() if name in common | specific[key]}
    return MODEL_REGISTRY[key](num_classes, seq_len, raw_signal_len, **options)
