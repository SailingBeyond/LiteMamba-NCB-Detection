#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Evaluate the two binary LiteMamba classifiers used for CB-vs-X and CB-vs-Y.

This evaluator intentionally keeps the binary-model architecture used by the
original checkpoints: A/C/T/G four-dimensional one-hot encoding plus eight
scalar features (12 dimensions per base). It is therefore separate from the
masked-kmer multiclass architecture in scripts/06_training/model_benchmark_all.py,
which uses A/C/T/G/N five-dimensional one-hot encoding.

All input, checkpoint, and output paths are supplied through command-line
arguments. The script writes per-threshold metrics, optional predictions,
confusion matrices, ROC/PR curves, and Figure 2B/2F source values.
"""

import os
import csv
import gc
import json
import time
import random
import argparse
from contextlib import nullcontext
from pathlib import Path
from typing import Iterator, Optional, List, Tuple, Dict, Any

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import IterableDataset, DataLoader

import matplotlib
matplotlib.use("Agg")
import matplotlib as mpl
import matplotlib.pyplot as plt
from matplotlib.ticker import MultipleLocator, FormatStrFormatter
from matplotlib.lines import Line2D

from sklearn.metrics import (
    confusion_matrix,
    ConfusionMatrixDisplay,
    classification_report,
    accuracy_score,
    balanced_accuracy_score,
    precision_recall_fscore_support,
    cohen_kappa_score,
    matthews_corrcoef,
    roc_curve,
    roc_auc_score,
    precision_recall_curve,
    average_precision_score,
    log_loss,
    brier_score_loss,
)

# =========================
# 0) 全局绘图风格
# =========================
mpl.rcParams["font.family"] = "Arial"
mpl.rcParams["font.sans-serif"] = ["Arial"]
mpl.rcParams["pdf.fonttype"] = 42
mpl.rcParams["ps.fonttype"] = 42
mpl.rcParams["axes.unicode_minus"] = False

CM_AXIS_LABEL_FONTSIZE = 30
CM_TICK_FONTSIZE = 30
CM_TEXT_FONTSIZE = 40
CM_CBAR_FONTSIZE = 30

CURVE_AXIS_LABEL_FONTSIZE = 25
CURVE_TICK_FONTSIZE = 25
CURVE_TITLE_FONTSIZE = 22
CURVE_LEGEND_FONTSIZE = 22

FIG2F_AXIS_LABEL_FONTSIZE = 25
FIG2F_TICK_FONTSIZE = 22
FIG2F_LEGEND_FONTSIZE = 19
FIG2F_ANNOTATION_FONTSIZE = 14
FIG2F_LINEWIDTH = 3.5
FIG2F_MARKERSIZE = 10
FIG2F_COLORS = {
    "X": "#377DB8",
    "Y": "#D98484",
}

FIG2B_AXIS_LABEL_FONTSIZE = 24
FIG2B_TICK_FONTSIZE = 20
FIG2B_XTICK_FONTSIZE = 16
FIG2B_TEXT_FONTSIZE = 14
FIG2B_SUBPANEL_FONTSIZE = 19
FIG2B_LINEWIDTH = 2.8
FIG2B_POINTSIZE = 180
FIG2B_COLORS = {
    "POS_X": "#377DB8",
    "NEG_Y": "#D98484",
}

LABEL_COLORS_BINARY = {
    0: "#377DB8",
    1: "#D98484",
}

# =========================
# 1) Binary LiteMamba model architecture
# =========================
class RawSignalEncoder(nn.Module):
    def __init__(self, signal_len: int, out_dim: int = 32):
        super().__init__()
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
    def __init__(self, raw_signal_len: int, feature_dim: int = 12, raw_dim: int = 32, model_dim: int = 96):
        super().__init__()
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

    def forward(self, merged_feat: torch.Tensor, raw_rect: torch.Tensor) -> torch.Tensor:
        feat_embed = self.feature_proj(merged_feat)
        raw_embed = self.raw_encoder(raw_rect)
        x = torch.cat([feat_embed, raw_embed], dim=-1)
        x = self.fuse_proj(x)
        return x


class LiteMambaBlock(nn.Module):
    def __init__(self, dim: int, kernel_size: int = 3, dropout: float = 0.1):
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
    def __init__(self, num_classes: int = 2, seq_len: int = 6,
                 raw_signal_len: int = 30, model_dim: int = 96,
                 num_blocks: int = 4):
        super().__init__()
        self.stem = FusionStem(raw_signal_len=raw_signal_len, model_dim=model_dim)
        self.blocks = nn.ModuleList([
            LiteMambaBlock(model_dim, kernel_size=3) for _ in range(num_blocks)
        ])
        self.pool = AttnPool(model_dim)
        self.cls = nn.Sequential(
            nn.Linear(model_dim * 2, 128),
            nn.GELU(),
            nn.Dropout(0.2),
            nn.Linear(128, num_classes),
        )

    def forward(self, inputs):
        _, merged_feat, raw_rect = inputs
        x = self.stem(merged_feat, raw_rect)
        for blk in self.blocks:
            x = blk(x)
        attn_vec = self.pool(x)
        last_vec = x[:, -1, :]
        feat = torch.cat([attn_vec, last_vec], dim=-1)
        logits = self.cls(feat)
        return logits, feat

# =========================
# 3) IterableDataset
# =========================
class LineIterableDataset(IterableDataset):
    def __init__(self, file_path: str):
        super().__init__()
        self.file_path = file_path

    def _iter_range(self, f, start: int, end: Optional[int]) -> Iterator[str]:
        f.seek(start)
        if start != 0:
            _ = f.readline()
        while True:
            pos = f.tell()
            if end is not None and pos >= end:
                break
            line = f.readline()
            if not line:
                break
            yield line.rstrip("\n")

    def __iter__(self) -> Iterator[str]:
        path = self.file_path
        file_size = os.path.getsize(path)
        worker = torch.utils.data.get_worker_info()
        f = open(path, "r", encoding="utf-8", buffering=1 << 20)

        if worker is None:
            try:
                yield from self._iter_range(f, 0, None)
            finally:
                f.close()
        else:
            n = worker.num_workers
            wid = worker.id
            start = (file_size * wid) // n
            end = (file_size * (wid + 1)) // n if (wid + 1) < n else None
            try:
                yield from self._iter_range(f, start, end)
            finally:
                f.close()

# =========================
# 4) 数据解析
# =========================
_BASE_MAP = {"A": 0, "C": 1, "T": 2, "G": 3}

def encode_seq_block(seq: str, mer_len: int) -> Optional[np.ndarray]:
    seq = seq.strip().upper()
    if len(seq) != mer_len:
        return None
    out = np.zeros((mer_len, 4), dtype=np.float32)
    for i, ch in enumerate(seq):
        v = _BASE_MAP.get(ch, None)
        if v is None:
            return None
        out[i, v] = 1.0
    return out

def fast_array_from_pipe(s: str, mer_len: int) -> Optional[np.ndarray]:
    arr = np.fromstring(s, sep="|", dtype=np.float32)
    if arr.size != mer_len:
        return None
    return arr

def get_signals_rect_from_str(signal_str: str, mer_len: int, signals_len: int) -> Optional[np.ndarray]:
    segs = signal_str.split("|")
    if len(segs) != mer_len:
        return None

    signals_rect = np.empty((mer_len, signals_len), dtype=np.float32)
    for i, seg in enumerate(segs):
        arr = np.fromstring(seg, sep="*", dtype=np.float32) if seg else np.empty(0, dtype=np.float32)
        arr = np.around(arr, decimals=4)
        if arr.size < signals_len:
            pad0_len = signals_len - arr.size
            pad_left = pad0_len // 2
            pad_right = pad0_len - pad_left
            arr = np.pad(arr, (pad_left, pad_right), mode="constant", constant_values=0.0)
        elif arr.size > signals_len:
            idx = sorted(random.sample(range(arr.size), signals_len))
            arr = arr[idx]
        signals_rect[i] = arr.astype(np.float32, copy=False)

    return signals_rect

def build_merged_features(seq_onehot_np: np.ndarray,
                          mean_np: np.ndarray,
                          std_np: np.ndarray,
                          median_np: np.ndarray,
                          dwell_np: np.ndarray,
                          quality_np: np.ndarray,
                          mismatch_np: np.ndarray,
                          insertion_np: np.ndarray,
                          deletion_np: np.ndarray,
                          dwell_divisor: float,
                          quality_divisor: float) -> np.ndarray:
    dwell_np = (dwell_np / dwell_divisor).astype(np.float32, copy=False)
    quality_np = (quality_np / quality_divisor).astype(np.float32, copy=False)

    scalar_part = np.stack([
        mean_np,
        std_np,
        median_np,
        dwell_np,
        quality_np,
        mismatch_np,
        insertion_np,
        deletion_np,
    ], axis=1).astype(np.float32, copy=False)

    merged = np.concatenate([seq_onehot_np.astype(np.float32, copy=False), scalar_part], axis=1)
    return merged.astype(np.float32, copy=False)

def parse_batch_lines(lines: List[str],
                      mer_len: int,
                      raw_signal_len_per_base: int,
                      dwell_divisor: float,
                      quality_divisor: float) -> Tuple[Tuple[torch.Tensor, torch.Tensor, torch.Tensor], torch.Tensor]:
    seq_onehot_list = []
    merged_feat_list = []
    raw_rect_list = []
    labels = []

    for ln in lines:
        if not ln:
            continue
        parts = ln.rstrip("\n").split(",")
        if len(parts) != 11:
            continue
        if parts[0].strip().lower() == "kmer" or parts[-1].strip().lower() == "label":
            continue

        seq_s, mean_s, std_s, median_s, dwell_s, quality_s, mismatch_s, insertion_s, deletion_s, signal_s, label_s = parts

        seq_onehot_np = encode_seq_block(seq_s, mer_len)
        if seq_onehot_np is None:
            continue

        mean_np = fast_array_from_pipe(mean_s, mer_len)
        std_np = fast_array_from_pipe(std_s, mer_len)
        median_np = fast_array_from_pipe(median_s, mer_len)
        dwell_np = fast_array_from_pipe(dwell_s, mer_len)
        quality_np = fast_array_from_pipe(quality_s, mer_len)
        mismatch_np = fast_array_from_pipe(mismatch_s, mer_len)
        insertion_np = fast_array_from_pipe(insertion_s, mer_len)
        deletion_np = fast_array_from_pipe(deletion_s, mer_len)
        raw_rect_np = get_signals_rect_from_str(signal_s, mer_len, raw_signal_len_per_base)

        if any(x is None for x in [
            mean_np, std_np, median_np, dwell_np, quality_np,
            mismatch_np, insertion_np, deletion_np, raw_rect_np
        ]):
            continue

        try:
            label = int(float(label_s))
        except Exception:
            continue
        if label not in (0, 1):
            continue

        merged_feat = build_merged_features(
            seq_onehot_np,
            mean_np,
            std_np,
            median_np,
            dwell_np,
            quality_np,
            mismatch_np,
            insertion_np,
            deletion_np,
            dwell_divisor=dwell_divisor,
            quality_divisor=quality_divisor,
        )

        seq_onehot_list.append(seq_onehot_np)
        merged_feat_list.append(merged_feat)
        raw_rect_list.append(raw_rect_np)
        labels.append(label)

    if not seq_onehot_list:
        empty_x = (
            torch.empty(0, mer_len, 4, dtype=torch.float32),
            torch.empty(0, mer_len, 12, dtype=torch.float32),
            torch.empty(0, mer_len, raw_signal_len_per_base, dtype=torch.float32),
        )
        empty_y = torch.empty(0, dtype=torch.long)
        return empty_x, empty_y

    seq_onehot_arr = np.stack(seq_onehot_list, axis=0)
    merged_arr = np.stack(merged_feat_list, axis=0)
    raw_arr = np.stack(raw_rect_list, axis=0)
    y_arr = np.asarray(labels, dtype=np.int64)

    x = (
        torch.from_numpy(seq_onehot_arr.astype(np.float32, copy=False)),
        torch.from_numpy(merged_arr.astype(np.float32, copy=False)),
        torch.from_numpy(raw_arr.astype(np.float32, copy=False)),
    )
    y = torch.from_numpy(y_arr)
    return x, y

class CollateFn:
    def __init__(self, mer_len: int, raw_signal_len: int, dwell_divisor: float, quality_divisor: float):
        self.mer_len = mer_len
        self.raw_signal_len = raw_signal_len
        self.dwell_divisor = dwell_divisor
        self.quality_divisor = quality_divisor

    def __call__(self, lines: List[str]):
        return parse_batch_lines(
            lines,
            mer_len=self.mer_len,
            raw_signal_len_per_base=self.raw_signal_len,
            dwell_divisor=self.dwell_divisor,
            quality_divisor=self.quality_divisor,
        )

# =========================
# 5) 工具函数
# =========================
def get_device(device_arg: str) -> torch.device:
    if device_arg.lower() == "auto":
        return torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    return torch.device(device_arg)

def set_seed(seed: int):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)

def enable_speedup():
    try:
        torch.backends.cudnn.benchmark = True
        torch.backends.cuda.matmul.allow_tf32 = True
        torch.backends.cudnn.allow_tf32 = True
        torch.set_float32_matmul_precision("high")
    except Exception:
        pass

def strip_state_dict_prefix(state: Dict[str, torch.Tensor]) -> Dict[str, torch.Tensor]:
    if not state:
        return state
    prefixes = ["module.", "_orig_mod."]
    new_state = state
    changed = True
    while changed:
        changed = False
        keys = list(new_state.keys())
        for prefix in prefixes:
            if keys and all(k.startswith(prefix) for k in keys):
                new_state = {k[len(prefix):]: v for k, v in new_state.items()}
                changed = True
                break
    return new_state

def load_litemamba_model(model_path: Path,
                         device: torch.device,
                         raw_signal_len: int,
                         num_classes: int = 2,
                         seq_len: int = 6) -> FusionLiteMamba:
    model_path = Path(model_path)
    if not model_path.exists():
        raise FileNotFoundError(f"模型权重不存在: {model_path}")

    print(f"加载模型: {model_path}")
    model = FusionLiteMamba(
        num_classes=num_classes,
        seq_len=seq_len,
        raw_signal_len=raw_signal_len,
    )
    ckpt = torch.load(str(model_path), map_location=device)
    if isinstance(ckpt, dict) and "state_dict" in ckpt:
        state = ckpt["state_dict"]
    elif isinstance(ckpt, dict) and "model_state_dict" in ckpt:
        state = ckpt["model_state_dict"]
    else:
        state = ckpt

    if not isinstance(state, dict):
        raise TypeError(f"无法识别的 checkpoint 格式: {type(state)}")

    state = strip_state_dict_prefix(state)
    model.load_state_dict(state, strict=True)
    model.to(device)
    model.eval()
    print("  ✓ LiteMamba 权重加载成功")
    return model

def seed_worker(worker_id: int) -> None:
    worker_seed = torch.initial_seed() % (2 ** 32)
    random.seed(worker_seed)
    np.random.seed(worker_seed)


def get_loader(csv_path: Path,
               batch_size: int,
               num_workers: int,
               mer_len: int,
               raw_signal_len: int,
               dwell_divisor: float,
               quality_divisor: float) -> DataLoader:
    ds = LineIterableDataset(str(csv_path))
    params = dict(
        batch_size=batch_size,
        shuffle=False,
        drop_last=False,
        collate_fn=CollateFn(
            mer_len=mer_len,
            raw_signal_len=raw_signal_len,
            dwell_divisor=dwell_divisor,
            quality_divisor=quality_divisor,
        ),
        num_workers=num_workers,
        pin_memory=torch.cuda.is_available(),
        persistent_workers=(num_workers > 0),
        worker_init_fn=seed_worker if num_workers > 0 else None,
    )
    return DataLoader(ds, **params)

def _is_plain_dna_kmer(s: str, mer_len: int) -> bool:
    s = str(s).strip().upper()
    return len(s) == mer_len and all(ch in {"A", "C", "G", "T"} for ch in s)

def get_kmer_from_file(csv_path: Path, mer_len: int = 6) -> str:
    stem = Path(csv_path).stem.strip().upper()
    if _is_plain_dna_kmer(stem, mer_len):
        return stem
    try:
        with open(csv_path, "r", encoding="utf-8") as f:
            for line in f:
                parts = line.rstrip("\n").split(",")
                if not parts:
                    continue
                cand = parts[0].strip().upper()
                if cand.lower() == "kmer":
                    continue
                if _is_plain_dna_kmer(cand, mer_len):
                    return cand
    except Exception:
        pass
    return stem

def get_label_names(csv_path: Path,
                    modified_label_name: str,
                    mer_len: int = 6,
                    natural_label_name: str = "CB",
                    use_kmer_base_label: bool = False) -> Dict[int, str]:
    if use_kmer_base_label:
        kmer = get_kmer_from_file(csv_path, mer_len=mer_len)
        base4 = kmer[3] if len(kmer) >= 4 else natural_label_name
        return {0: base4, 1: modified_label_name}
    return {0: natural_label_name, 1: modified_label_name}

def condition_name(threshold: Optional[float]) -> str:
    if threshold is None:
        return "argmax_all"
    return f"T{threshold:.2f}".replace(".", "p")

def condition_label(threshold: Optional[float]) -> str:
    if threshold is None:
        return "Argmax all"
    return f"max prob ≥ {threshold:.2f}"

def safe_float(x: Any) -> float:
    try:
        v = float(x)
        if np.isnan(v) or np.isinf(v):
            return float("nan")
        return v
    except Exception:
        return float("nan")

def _safe_div(a: float, b: float) -> float:
    return float(a) / float(b) if b else 0.0

def expected_calibration_error(confidence: np.ndarray, correctness: np.ndarray, n_bins: int = 15) -> float:
    confidence = np.asarray(confidence, dtype=np.float64)
    correctness = np.asarray(correctness, dtype=np.float64)
    if confidence.size == 0:
        return float("nan")
    bins = np.linspace(0.0, 1.0, n_bins + 1)
    ece = 0.0
    n = confidence.size
    for i in range(n_bins):
        left, right = bins[i], bins[i + 1]
        if i == n_bins - 1:
            mask = (confidence >= left) & (confidence <= right)
        else:
            mask = (confidence >= left) & (confidence < right)
        if not np.any(mask):
            continue
        bin_conf = float(np.mean(confidence[mask]))
        bin_acc = float(np.mean(correctness[mask]))
        ece += (float(mask.sum()) / float(n)) * abs(bin_acc - bin_conf)
    return float(ece)

# =========================
# 6) 推理
# =========================
@torch.no_grad()
def predict_one_file(model: FusionLiteMamba,
                     csv_path: Path,
                     device: torch.device,
                     args) -> Tuple[np.ndarray, np.ndarray]:
    loader = get_loader(
        csv_path=csv_path,
        batch_size=args.batch_size,
        num_workers=args.workers,
        mer_len=args.mer_len,
        raw_signal_len=args.raw_signal_len,
        dwell_divisor=args.dwell_divisor,
        quality_divisor=args.quality_divisor,
    )

    y_true_list = []
    y_prob_list = []
    total_seen = 0
    use_amp = bool(args.amp and torch.cuda.is_available() and "cuda" in str(device).lower())

    for data_cpu, labels_cpu in loader:
        total_seen += int(labels_cpu.numel())
        if labels_cpu.numel() == 0:
            continue
        data = tuple(t.to(device, non_blocking=True) for t in data_cpu)
        labels_cpu_np = labels_cpu.detach().cpu().numpy().astype(np.int64, copy=False)

        amp_context = torch.amp.autocast(device_type="cuda", enabled=True) if use_amp else nullcontext()
        with amp_context:
            logits, _ = model(data)

        probs = torch.softmax(logits.float(), dim=1).detach().cpu().numpy().astype(np.float32, copy=False)
        y_true_list.append(labels_cpu_np)
        y_prob_list.append(probs)
        del data, data_cpu, labels_cpu, logits

    if not y_true_list:
        raise RuntimeError(
            f"没有从文件中解析到有效样本: {csv_path}\n"
            f"DataLoader seen labels={total_seen}. 请检查 CSV 是否为 11 列格式。"
        )

    y_true = np.concatenate(y_true_list, axis=0).astype(np.int64, copy=False)
    y_prob = np.concatenate(y_prob_list, axis=0).astype(np.float32, copy=False)

    if y_prob.ndim != 2 or y_prob.shape[1] != 2:
        raise RuntimeError(f"模型输出概率维度异常: {y_prob.shape}, 期望 [N, 2]")

    return y_true, y_prob

# =========================
# 7) 指标计算
# =========================
def compute_metrics_for_mask(y_true_all: np.ndarray,
                             y_prob_all: np.ndarray,
                             threshold: Optional[float],
                             label_names: Dict[int, str],
                             total_original: int) -> Tuple[Dict[str, Any], np.ndarray, np.ndarray, np.ndarray]:
    y_pred_all = np.argmax(y_prob_all, axis=1).astype(np.int64, copy=False)
    max_prob_all = np.max(y_prob_all, axis=1)

    if threshold is None:
        mask = np.ones_like(y_true_all, dtype=bool)
    else:
        mask = max_prob_all >= float(threshold)

    y_true = y_true_all[mask]
    y_prob = y_prob_all[mask]
    y_pred = y_pred_all[mask]
    max_prob = max_prob_all[mask]

    kept = int(mask.sum())
    rejected = int(total_original - kept)
    coverage = _safe_div(kept, total_original)

    cond = condition_name(threshold)
    cond_label = condition_label(threshold)

    base_payload: Dict[str, Any] = {
        "condition": cond,
        "condition_label": cond_label,
        "threshold": "None" if threshold is None else float(threshold),
        "total_original": int(total_original),
        "kept_samples": kept,
        "rejected_samples": rejected,
        "coverage": coverage,
        "n_true_0": int(np.sum(y_true == 0)) if kept else 0,
        "n_true_1": int(np.sum(y_true == 1)) if kept else 0,
        "n_pred_0": int(np.sum(y_pred == 0)) if kept else 0,
        "n_pred_1": int(np.sum(y_pred == 1)) if kept else 0,
        "label0_name": label_names[0],
        "label1_name": label_names[1],
    }

    if kept == 0:
        base_payload.update({
            "status": "EMPTY_AFTER_THRESHOLD",
            "accuracy": float("nan"),
            "balanced_accuracy": float("nan"),
            "macro_precision": float("nan"),
            "macro_recall": float("nan"),
            "macro_f1": float("nan"),
            "weighted_precision": float("nan"),
            "weighted_recall": float("nan"),
            "weighted_f1": float("nan"),
            "mcc": float("nan"),
            "cohen_kappa": float("nan"),
            "roc_auc_pos": float("nan"),
            "average_precision_pos": float("nan"),
            "log_loss": float("nan"),
            "brier_score_pos": float("nan"),
            "ece_15bins": float("nan"),
        })
        return base_payload, y_true, y_pred, y_prob

    labels = [0, 1]
    cm = confusion_matrix(y_true, y_pred, labels=labels)
    tn, fp, fn, tp = cm.ravel()

    precision_arr, recall_arr, f1_arr, support_arr = precision_recall_fscore_support(
        y_true, y_pred, labels=labels, zero_division=0
    )
    macro_p, macro_r, macro_f1, _ = precision_recall_fscore_support(
        y_true, y_pred, labels=labels, average="macro", zero_division=0
    )
    weighted_p, weighted_r, weighted_f1, _ = precision_recall_fscore_support(
        y_true, y_pred, labels=labels, average="weighted", zero_division=0
    )

    acc = float(accuracy_score(y_true, y_pred))
    try:
        bal_acc = float(balanced_accuracy_score(y_true, y_pred))
    except Exception:
        bal_acc = float("nan")
    try:
        kappa = float(cohen_kappa_score(y_true, y_pred))
    except Exception:
        kappa = float("nan")
    try:
        mcc = float(matthews_corrcoef(y_true, y_pred))
    except Exception:
        mcc = float("nan")

    specificity = _safe_div(tn, tn + fp)
    sensitivity = _safe_div(tp, tp + fn)
    precision_pos = _safe_div(tp, tp + fp)
    precision_neg = _safe_div(tn, tn + fn)
    recall_neg = _safe_div(tn, tn + fp)
    f1_pos = _safe_div(2 * precision_pos * sensitivity, precision_pos + sensitivity) if (precision_pos + sensitivity) else 0.0
    f1_neg = _safe_div(2 * precision_neg * recall_neg, precision_neg + recall_neg) if (precision_neg + recall_neg) else 0.0
    npv = _safe_div(tn, tn + fn)
    fpr = _safe_div(fp, fp + tn)
    fnr = _safe_div(fn, tp + fn)
    fdr = _safe_div(fp, tp + fp)
    false_omission_rate = _safe_div(fn, tn + fn)

    if len(np.unique(y_true)) == 2:
        try:
            roc_auc_pos = float(roc_auc_score(y_true, y_prob[:, 1]))
        except Exception:
            roc_auc_pos = float("nan")
        try:
            ap_pos = float(average_precision_score(y_true, y_prob[:, 1]))
        except Exception:
            ap_pos = float("nan")
        try:
            roc_auc_neg = float(roc_auc_score(1 - y_true, y_prob[:, 0]))
        except Exception:
            roc_auc_neg = float("nan")
        try:
            ap_neg = float(average_precision_score(1 - y_true, y_prob[:, 0]))
        except Exception:
            ap_neg = float("nan")
    else:
        roc_auc_pos = roc_auc_neg = ap_pos = ap_neg = float("nan")

    try:
        ll = float(log_loss(y_true, y_prob, labels=[0, 1]))
    except Exception:
        ll = float("nan")
    try:
        brier_pos = float(brier_score_loss((y_true == 1).astype(int), y_prob[:, 1]))
    except Exception:
        brier_pos = float("nan")

    correctness = (y_true == y_pred).astype(np.float32)
    ece = expected_calibration_error(max_prob, correctness, n_bins=15)

    per_class = {
        "0": {
            "label_name": label_names[0],
            "precision": float(precision_arr[0]),
            "recall": float(recall_arr[0]),
            "f1": float(f1_arr[0]),
            "support": int(support_arr[0]),
            "auroc_ovr": roc_auc_neg,
            "auprc_ovr": ap_neg,
        },
        "1": {
            "label_name": label_names[1],
            "precision": float(precision_arr[1]),
            "recall": float(recall_arr[1]),
            "f1": float(f1_arr[1]),
            "support": int(support_arr[1]),
            "auroc_ovr": roc_auc_pos,
            "auprc_ovr": ap_pos,
        },
    }

    base_payload.update({
        "status": "OK",
        "TN": int(tn),
        "FP": int(fp),
        "FN": int(fn),
        "TP": int(tp),
        "accuracy": acc,
        "balanced_accuracy": bal_acc,
        "macro_precision": float(macro_p),
        "macro_recall": float(macro_r),
        "macro_f1": float(macro_f1),
        "weighted_precision": float(weighted_p),
        "weighted_recall": float(weighted_r),
        "weighted_f1": float(weighted_f1),
        "precision_pos_label1": precision_pos,
        "recall_sensitivity_TPR_pos_label1": sensitivity,
        "f1_pos_label1": f1_pos,
        "precision_label0": precision_neg,
        "recall_specificity_TNR_label0": recall_neg,
        "f1_label0": f1_neg,
        "specificity_TNR": specificity,
        "NPV": npv,
        "FPR": fpr,
        "FNR": fnr,
        "FDR": fdr,
        "FOR": false_omission_rate,
        "mcc": mcc,
        "cohen_kappa": kappa,
        "roc_auc_pos": roc_auc_pos,
        "roc_auc_neg": roc_auc_neg,
        "roc_auc_macro": float(np.nanmean([roc_auc_neg, roc_auc_pos])),
        "average_precision_pos": ap_pos,
        "average_precision_neg": ap_neg,
        "average_precision_macro": float(np.nanmean([ap_neg, ap_pos])),
        "log_loss": ll,
        "brier_score_pos": brier_pos,
        "mean_max_prob_kept": float(np.mean(max_prob)),
        "median_max_prob_kept": float(np.median(max_prob)),
        "ece_15bins": ece,
        "confusion_matrix_raw": cm.tolist(),
        "per_class": per_class,
        "classification_report_dict": classification_report(
            y_true,
            y_pred,
            labels=labels,
            target_names=[label_names[0], label_names[1]],
            digits=4,
            output_dict=True,
            zero_division=0,
        ),
    })

    return base_payload, y_true, y_pred, y_prob

# =========================
# 8) 绘图函数
# =========================
def style_axes_frame(ax, spine_lw: float = 3.0, tick_lw: float = 3.0, tick_len: float = 6.0):
    ax.tick_params(axis="both", direction="out", length=tick_len, width=tick_lw, colors="black")
    for side in ["left", "right", "top", "bottom"]:
        ax.spines[side].set_linewidth(spine_lw)
        ax.spines[side].set_color("black")

def plot_confusion_matrix_norm(y_true: np.ndarray,
                               y_pred: np.ndarray,
                               out_path: Path,
                               label_names: Dict[int, str]):
    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    labels = [0, 1]
    display_labels = [label_names[0], label_names[1]]
    cm = confusion_matrix(y_true, y_pred, labels=labels)

    with np.errstate(divide="ignore", invalid="ignore"):
        cm_norm = cm.astype(np.float64) / cm.sum(axis=1, keepdims=True)
        cm_norm = np.nan_to_num(cm_norm)

    fig, ax = plt.subplots(figsize=(8, 7))
    cmd = ConfusionMatrixDisplay(confusion_matrix=cm_norm, display_labels=display_labels)
    cmd.plot(
        cmap=plt.cm.Blues,
        ax=ax,
        values_format=".2f",
        colorbar=True,
        text_kw={"fontsize": CM_TEXT_FONTSIZE, "fontweight": "bold"},
    )

    style_axes_frame(ax, spine_lw=3, tick_lw=3, tick_len=6)

    if cmd.text_ is not None:
        for txt in cmd.text_.ravel():
            txt.set_fontsize(CM_TEXT_FONTSIZE)
            txt.set_fontweight("bold")

    cmd.im_.set_clim(0.0, 1.0)

    ax.set_xticks(range(len(labels)))
    ax.set_xticklabels(display_labels, rotation=45, ha="right", fontsize=CM_TICK_FONTSIZE)
    for i, tick in enumerate(ax.get_xticklabels()):
        tick.set_color(LABEL_COLORS_BINARY[labels[i]])

    ax.set_yticks(range(len(labels)))
    ax.set_yticklabels(display_labels, fontsize=CM_TICK_FONTSIZE)
    for i, tick in enumerate(ax.get_yticklabels()):
        tick.set_color(LABEL_COLORS_BINARY[labels[i]])

    ax.set_xlabel("Predicted Label", fontsize=CM_AXIS_LABEL_FONTSIZE)
    ax.set_ylabel("True Label", fontsize=CM_AXIS_LABEL_FONTSIZE)

    cbar = getattr(cmd.im_, "colorbar", None)
    if cbar is not None:
        cbar.ax.tick_params(labelsize=CM_CBAR_FONTSIZE, width=3, length=6, colors="black")
        cbar.set_label("Proportion", fontsize=CM_CBAR_FONTSIZE)
        try:
            cbar.outline.set_linewidth(3)
            cbar.outline.set_edgecolor("black")
        except Exception:
            pass
        for spine in cbar.ax.spines.values():
            spine.set_linewidth(3)
            spine.set_color("black")

    plt.tight_layout()
    fig.savefig(out_path, dpi=600, bbox_inches="tight")
    plt.close(fig)

def force_ticks_01(ax, xlim=(0.0, 1.0), ylim=(0.0, 1.0), step=0.1):
    ax.set_xlim(*xlim)
    ax.set_ylim(*ylim)
    xticks = np.round(np.arange(xlim[0], xlim[1] + 1e-9, step), 2)
    yticks = np.round(np.arange(ylim[0], ylim[1] + 1e-9, step), 2)
    ax.set_xticks(xticks)
    ax.set_yticks(yticks)
    ax.xaxis.set_major_formatter(FormatStrFormatter("%.1f"))
    ax.yaxis.set_major_formatter(FormatStrFormatter("%.1f"))
    ax.xaxis.set_major_locator(MultipleLocator(step))
    ax.yaxis.set_major_locator(MultipleLocator(step))

def fix_corner_ticklabel_overlap(ax):
    ax.tick_params(axis="x", pad=10)
    ax.tick_params(axis="y", pad=10)
    fig = ax.figure
    fig.canvas.draw()
    xlabels = ax.get_xticklabels()
    ylabels = ax.get_yticklabels()
    if len(xlabels) > 0:
        xlabels[0].set_horizontalalignment("left")
    if len(ylabels) > 0:
        ylabels[0].set_verticalalignment("bottom")

def plot_single_roc_pr(y_true: np.ndarray,
                       y_prob: np.ndarray,
                       out_roc: Path,
                       out_pr: Path,
                       label_names: Dict[int, str],
                       title_prefix: str,
                       roc_xlim=(0.0, 1.0),
                       roc_ylim=(0.0, 1.0),
                       pr_xlim=(0.0, 1.0),
                       pr_ylim=(0.0, 1.0)):
    out_roc = Path(out_roc)
    out_pr = Path(out_pr)
    out_roc.parent.mkdir(parents=True, exist_ok=True)
    out_pr.parent.mkdir(parents=True, exist_ok=True)

    y_true = np.asarray(y_true)
    y_score = np.asarray(y_prob)[:, 1]
    pos_name = label_names[1]

    fig, ax = plt.subplots(figsize=(8, 8))
    if len(np.unique(y_true)) == 2:
        fpr, tpr, _ = roc_curve(y_true, y_score)
        auc = roc_auc_score(y_true, y_score)
        ax.plot(fpr, tpr, linewidth=3, color=LABEL_COLORS_BINARY[1], label=f"{pos_name} AUC={auc:.3f}")
    else:
        ax.text(0.5, 0.5, "Only one class retained\nROC is undefined", ha="center", va="center", fontsize=18)
    ax.plot([0, 1], [0, 1], linestyle="--", linewidth=1.5, color="gray", alpha=0.8)
    ax.set_xlabel("False Positive Rate", fontsize=CURVE_AXIS_LABEL_FONTSIZE)
    ax.set_ylabel("True Positive Rate", fontsize=CURVE_AXIS_LABEL_FONTSIZE)
    ax.set_title(f"{title_prefix} ROC", fontsize=CURVE_TITLE_FONTSIZE, fontweight="bold", pad=12)
    force_ticks_01(ax, xlim=roc_xlim, ylim=roc_ylim, step=0.1)
    try:
        ax.set_box_aspect(1)
    except Exception:
        ax.set_aspect("equal", adjustable="box")
    ax.tick_params(direction="out", length=6, width=3, labelsize=CURVE_TICK_FONTSIZE)
    for spine in ax.spines.values():
        spine.set_linewidth(3)
    fix_corner_ticklabel_overlap(ax)
    ax.legend(loc="lower right", prop={"size": CURVE_LEGEND_FONTSIZE}, frameon=False)
    plt.tight_layout()
    fig.savefig(out_roc, dpi=600, bbox_inches="tight")
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(8, 8))
    if len(np.unique(y_true)) == 2:
        precision, recall, _ = precision_recall_curve(y_true, y_score)
        ap = average_precision_score(y_true, y_score)
        ax.plot(recall, precision, linewidth=3, color=LABEL_COLORS_BINARY[1], label=f"{pos_name} AP={ap:.3f}")
    else:
        ax.text(0.5, 0.5, "Only one class retained\nPR is undefined", ha="center", va="center", fontsize=18)
    ax.set_xlabel("Recall", fontsize=CURVE_AXIS_LABEL_FONTSIZE)
    ax.set_ylabel("Precision", fontsize=CURVE_AXIS_LABEL_FONTSIZE)
    ax.set_title(f"{title_prefix} PR", fontsize=CURVE_TITLE_FONTSIZE, fontweight="bold", pad=12)
    force_ticks_01(ax, xlim=pr_xlim, ylim=pr_ylim, step=0.1)
    try:
        ax.set_box_aspect(1)
    except Exception:
        ax.set_aspect("equal", adjustable="box")
    ax.tick_params(direction="out", length=6, width=3, labelsize=CURVE_TICK_FONTSIZE)
    for spine in ax.spines.values():
        spine.set_linewidth(3)
    fix_corner_ticklabel_overlap(ax)
    ax.legend(loc="lower left", prop={"size": CURVE_LEGEND_FONTSIZE}, frameon=False)
    plt.tight_layout()
    fig.savefig(out_pr, dpi=600, bbox_inches="tight")
    plt.close(fig)

def plot_multi_threshold_roc_pr(y_true_all: np.ndarray,
                                y_prob_all: np.ndarray,
                                thresholds: List[Optional[float]],
                                out_roc: Path,
                                out_pr: Path,
                                label_names: Dict[int, str],
                                title_prefix: str,
                                min_kept_for_curve: int,
                                roc_xlim=(0.0, 1.0),
                                roc_ylim=(0.0, 1.0),
                                pr_xlim=(0.0, 1.0),
                                pr_ylim=(0.0, 1.0)):
    out_roc = Path(out_roc)
    out_pr = Path(out_pr)
    out_roc.parent.mkdir(parents=True, exist_ok=True)
    out_pr.parent.mkdir(parents=True, exist_ok=True)

    max_prob = y_prob_all.max(axis=1)
    total = int(y_true_all.size)
    pos_name = label_names[1]

    fig, ax = plt.subplots(figsize=(8, 8))
    plotted = 0
    for thr in thresholds:
        if thr is None:
            mask = np.ones_like(y_true_all, dtype=bool)
        else:
            mask = max_prob >= float(thr)
        kept = int(mask.sum())
        if kept < min_kept_for_curve:
            continue
        yt = y_true_all[mask]
        yp = y_prob_all[mask]
        if len(np.unique(yt)) < 2:
            continue
        fpr, tpr, _ = roc_curve(yt, yp[:, 1])
        auc = roc_auc_score(yt, yp[:, 1])
        keep_ratio = _safe_div(kept, total) * 100.0
        lab = "All" if thr is None else f"T≥{thr:.2f}"
        ax.plot(fpr, tpr, linewidth=3, label=f"{lab} | keep {keep_ratio:.1f}% | AUC {auc:.3f}")
        plotted += 1
    ax.plot([0, 1], [0, 1], linestyle="--", linewidth=1.5, color="gray", alpha=0.8)
    if plotted == 0:
        ax.text(0.5, 0.5, "No valid ROC curve", ha="center", va="center", fontsize=18)
    ax.set_xlabel("False Positive Rate", fontsize=CURVE_AXIS_LABEL_FONTSIZE)
    ax.set_ylabel("True Positive Rate", fontsize=CURVE_AXIS_LABEL_FONTSIZE)
    ax.set_title(f"{title_prefix} ROC ({pos_name})", fontsize=CURVE_TITLE_FONTSIZE, fontweight="bold", pad=12)
    force_ticks_01(ax, xlim=roc_xlim, ylim=roc_ylim, step=0.1)
    try:
        ax.set_box_aspect(1)
    except Exception:
        ax.set_aspect("equal", adjustable="box")
    ax.tick_params(direction="out", length=6, width=3, labelsize=CURVE_TICK_FONTSIZE)
    for spine in ax.spines.values():
        spine.set_linewidth(3)
    fix_corner_ticklabel_overlap(ax)
    ax.legend(loc="lower right", prop={"size": CURVE_LEGEND_FONTSIZE}, frameon=False)
    plt.tight_layout()
    fig.savefig(out_roc, dpi=600, bbox_inches="tight")
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(8, 8))
    plotted = 0
    for thr in thresholds:
        if thr is None:
            mask = np.ones_like(y_true_all, dtype=bool)
        else:
            mask = max_prob >= float(thr)
        kept = int(mask.sum())
        if kept < min_kept_for_curve:
            continue
        yt = y_true_all[mask]
        yp = y_prob_all[mask]
        if len(np.unique(yt)) < 2:
            continue
        precision, recall, _ = precision_recall_curve(yt, yp[:, 1])
        ap = average_precision_score(yt, yp[:, 1])
        keep_ratio = _safe_div(kept, total) * 100.0
        lab = "All" if thr is None else f"T≥{thr:.2f}"
        ax.plot(recall, precision, linewidth=3, label=f"{lab} | keep {keep_ratio:.1f}% | AP {ap:.3f}")
        plotted += 1
    if plotted == 0:
        ax.text(0.5, 0.5, "No valid PR curve", ha="center", va="center", fontsize=18)
    ax.set_xlabel("Recall", fontsize=CURVE_AXIS_LABEL_FONTSIZE)
    ax.set_ylabel("Precision", fontsize=CURVE_AXIS_LABEL_FONTSIZE)
    ax.set_title(f"{title_prefix} PR ({pos_name})", fontsize=CURVE_TITLE_FONTSIZE, fontweight="bold", pad=12)
    force_ticks_01(ax, xlim=pr_xlim, ylim=pr_ylim, step=0.1)
    try:
        ax.set_box_aspect(1)
    except Exception:
        ax.set_aspect("equal", adjustable="box")
    ax.tick_params(direction="out", length=6, width=3, labelsize=CURVE_TICK_FONTSIZE)
    for spine in ax.spines.values():
        spine.set_linewidth(3)
    fix_corner_ticklabel_overlap(ax)
    ax.legend(loc="lower left", prop={"size": CURVE_LEGEND_FONTSIZE}, frameon=False)
    plt.tight_layout()
    fig.savefig(out_pr, dpi=600, bbox_inches="tight")
    plt.close(fig)

# =========================
# 9) Figure 2F
# =========================
def _numeric_threshold(value: Any) -> Optional[float]:
    if value is None:
        return None
    if isinstance(value, str) and value.strip().lower() in {"", "none", "nan"}:
        return None
    try:
        v = float(value)
    except Exception:
        return None
    if not np.isfinite(v):
        return None
    return v


def _collect_figure2f_rows(global_rows: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    """
    收集 Figure 2F 所需的阈值结果。

    Figure 2F 仅使用：
      - threshold：置信度阈值
      - accuracy：阈值筛选后保留样本的准确率
      - coverage：剩余率，即 kept_samples / total_original
    """
    group_to_label = {"POS_X": "X", "NEG_Y": "Y"}
    rows: List[Dict[str, Any]] = []

    for row in global_rows:
        group = str(row.get("group", ""))
        series = group_to_label.get(group)
        if series is None or row.get("status") != "OK":
            continue

        threshold = _numeric_threshold(row.get("threshold"))
        if threshold is None:
            # Figure 2F 只画明确的数值阈值，不包含 argmax_all。
            continue

        accuracy = safe_float(row.get("accuracy"))
        coverage = safe_float(row.get("coverage"))
        if not np.isfinite(accuracy) or not np.isfinite(coverage):
            continue

        rows.append({
            "series": series,
            "group": group,
            "threshold": float(threshold),
            "accuracy": float(accuracy),
            "coverage": float(coverage),
            "remaining_rate": float(coverage),
            "rejection_rate": float(1.0 - coverage),
            "total_original": int(row.get("total_original", 0) or 0),
            "kept_samples": int(row.get("kept_samples", 0) or 0),
            "rejected_samples": int(row.get("rejected_samples", 0) or 0),
        })

    rows.sort(key=lambda r: (r["series"], r["threshold"]))
    return rows


def write_figure2f_values_csv(rows: List[Dict[str, Any]], out_path: Path):
    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fields = [
        "series", "group", "threshold", "accuracy",
        "coverage", "remaining_rate", "rejection_rate",
        "total_original", "kept_samples", "rejected_samples",
    ]
    with open(out_path, "w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fields)
        writer.writeheader()
        for row in rows:
            writer.writerow({k: row.get(k, "") for k in fields})


def _auto_figure2f_ylim(
        values: np.ndarray,
        requested_min: Optional[float] = None,
        requested_max: Optional[float] = None,
        lower_bound: float = 0.0,
        upper_bound: float = 1.0) -> Tuple[float, float]:
    """根据数据自动生成紧凑、可读的纵轴范围。"""
    values = np.asarray(values, dtype=float)
    values = values[np.isfinite(values)]
    if values.size == 0:
        return lower_bound, upper_bound

    data_min = float(np.min(values))
    data_max = float(np.max(values))
    data_span = data_max - data_min
    padding = max(0.005, data_span * 0.20)

    if requested_min is None:
        ymin = max(lower_bound, data_min - padding)
        ymin = np.floor(ymin / 0.005) * 0.005
    else:
        ymin = float(requested_min)

    if requested_max is None:
        ymax = min(upper_bound, data_max + padding)
        ymax = np.ceil(ymax / 0.005) * 0.005
    else:
        ymax = float(requested_max)

    # 当所有数值完全相同或范围过小时，至少保留 0.01 的显示跨度。
    if ymax - ymin < 0.01:
        center = (ymax + ymin) / 2.0
        ymin = max(lower_bound, center - 0.005)
        ymax = min(upper_bound, center + 0.005)

    if ymax <= ymin:
        ymin = max(lower_bound, data_min - 0.01)
        ymax = min(upper_bound, max(data_max + 0.01, ymin + 0.01))

    return float(ymin), float(ymax)


def _set_figure2f_y_ticks(ax, ymin: float, ymax: float):
    span = ymax - ymin
    if span <= 0.03:
        step = 0.005
        fmt = "%.3f"
    elif span <= 0.10:
        step = 0.01
        fmt = "%.2f"
    elif span <= 0.30:
        step = 0.05
        fmt = "%.2f"
    else:
        step = 0.10
        fmt = "%.1f"
    ax.yaxis.set_major_locator(MultipleLocator(step))
    ax.yaxis.set_major_formatter(FormatStrFormatter(fmt))


def _plot_single_figure2f_series(
        rows: List[Dict[str, Any]],
        series: str,
        out_dir: Path,
        accuracy_ymin: Optional[float],
        accuracy_ymax: Optional[float]) -> Dict[str, str]:
    """
    为 X 或 Y 单独绘制 Figure 2F：
      横轴：Threshold
      左纵轴：Accuracy
      右纵轴：Remaining Rate（coverage）
    """
    series_rows = sorted(
        [row for row in rows if row.get("series") == series],
        key=lambda row: float(row["threshold"]),
    )
    if not series_rows:
        print(f"  ⚠ Figure 2F {series} 未绘制：没有有效阈值结果。")
        return {}

    thresholds = np.asarray([row["threshold"] for row in series_rows], dtype=float)
    accuracy = np.asarray([row["accuracy"] for row in series_rows], dtype=float)
    remaining_rate = np.asarray([row["remaining_rate"] for row in series_rows], dtype=float)

    acc_ymin, acc_ymax = _auto_figure2f_ylim(
        accuracy,
        requested_min=accuracy_ymin,
        requested_max=accuracy_ymax,
    )
    remain_ymin, remain_ymax = _auto_figure2f_ylim(
        remaining_rate,
        requested_min=None,
        requested_max=1.0,
    )

    series_color = FIG2F_COLORS[series]
    remaining_color = "#555555"

    fig, ax_accuracy = plt.subplots(figsize=(8.8, 7.2))
    ax_remaining = ax_accuracy.twinx()

    accuracy_line, = ax_accuracy.plot(
        thresholds,
        accuracy,
        color=series_color,
        linewidth=FIG2F_LINEWIDTH,
        marker="o",
        markersize=FIG2F_MARKERSIZE,
        markeredgewidth=2.0,
        markeredgecolor="white",
        label="Accuracy",
        zorder=4,
    )

    remaining_line, = ax_remaining.plot(
        thresholds,
        remaining_rate,
        color=remaining_color,
        linewidth=FIG2F_LINEWIDTH,
        linestyle="--",
        marker="s",
        markersize=FIG2F_MARKERSIZE - 1,
        markeredgewidth=1.8,
        markeredgecolor="white",
        label="Remaining Rate",
        zorder=3,
    )

    # 阈值横轴。
    if thresholds.size == 1:
        x_padding = 0.05
    else:
        x_padding = max(0.02, float(np.min(np.diff(np.unique(thresholds)))) * 0.25)
    ax_accuracy.set_xlim(float(np.min(thresholds) - x_padding), float(np.max(thresholds) + x_padding))
    ax_accuracy.set_xticks(thresholds)
    if np.allclose(thresholds * 10.0, np.round(thresholds * 10.0), atol=1e-8):
        ax_accuracy.xaxis.set_major_formatter(FormatStrFormatter("%.1f"))
    else:
        ax_accuracy.xaxis.set_major_formatter(FormatStrFormatter("%.2f"))

    ax_accuracy.set_ylim(acc_ymin, acc_ymax)
    ax_remaining.set_ylim(remain_ymin, remain_ymax)
    _set_figure2f_y_ticks(ax_accuracy, acc_ymin, acc_ymax)
    _set_figure2f_y_ticks(ax_remaining, remain_ymin, remain_ymax)

    ax_accuracy.set_xlabel("Threshold", fontsize=FIG2F_AXIS_LABEL_FONTSIZE)
    ax_accuracy.set_ylabel(
        "Accuracy",
        fontsize=FIG2F_AXIS_LABEL_FONTSIZE,
        color=series_color,
        labelpad=10,
    )
    ax_remaining.set_ylabel(
        "Remaining Rate",
        fontsize=FIG2F_AXIS_LABEL_FONTSIZE,
        color=remaining_color,
        labelpad=12,
    )

    # 左、右纵轴刻度分别与对应曲线同色，便于识别双纵轴。
    ax_accuracy.tick_params(
        axis="x", direction="out", length=7, width=3,
        labelsize=FIG2F_TICK_FONTSIZE, pad=8, colors="black",
    )
    ax_accuracy.tick_params(
        axis="y", direction="out", length=7, width=3,
        labelsize=FIG2F_TICK_FONTSIZE, pad=8, colors=series_color,
    )
    ax_remaining.tick_params(
        axis="y", direction="out", length=7, width=3,
        labelsize=FIG2F_TICK_FONTSIZE, pad=8, colors=remaining_color,
    )

    # 双纵轴共用上下边框；左边框对应 Accuracy，右边框对应 Remaining Rate。
    for side in ["bottom", "top"]:
        ax_accuracy.spines[side].set_linewidth(3)
        ax_accuracy.spines[side].set_color("black")
    ax_accuracy.spines["left"].set_linewidth(3)
    ax_accuracy.spines["left"].set_color(series_color)
    ax_accuracy.spines["right"].set_visible(False)

    ax_remaining.spines["right"].set_linewidth(3)
    ax_remaining.spines["right"].set_color(remaining_color)
    ax_remaining.spines["left"].set_visible(False)
    ax_remaining.spines["top"].set_visible(False)
    ax_remaining.spines["bottom"].set_visible(False)

    # 用图内的 X/Y 标记区分拆分后的两张图，不额外设置大标题。
    ax_accuracy.text(
        0.02, 0.97, series,
        transform=ax_accuracy.transAxes,
        ha="left", va="top",
        fontsize=FIG2F_AXIS_LABEL_FONTSIZE,
        color=series_color,
        fontweight="bold",
    )

    ax_accuracy.legend(
        handles=[accuracy_line, remaining_line],
        labels=["Accuracy", "Remaining Rate"],
        loc="best",
        frameon=False,
        prop={"size": FIG2F_LEGEND_FONTSIZE, "weight": "bold"},
        handlelength=2.6,
    )

    fig.subplots_adjust(left=0.17, right=0.83, bottom=0.15, top=0.97)

    stem = f"Figure2F_{series}_accuracy_remaining_rate"
    png_path = out_dir / f"{stem}.png"
    pdf_path = out_dir / f"{stem}.pdf"
    svg_path = out_dir / f"{stem}.svg"

    fig.savefig(png_path, dpi=600, bbox_inches="tight")
    fig.savefig(pdf_path, bbox_inches="tight")
    fig.savefig(svg_path, bbox_inches="tight")
    plt.close(fig)

    print(f"  ✓ Figure 2F {series} PNG: {png_path}")
    print(f"  ✓ Figure 2F {series} PDF: {pdf_path}")
    print(f"  ✓ Figure 2F {series} SVG: {svg_path}")

    return {
        "png": str(png_path),
        "pdf": str(pdf_path),
        "svg": str(svg_path),
    }


def generate_figure2f(global_rows: List[Dict[str, Any]], args) -> Dict[str, Any]:
    out_dir = Path(args.figure2f_out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    rows = _collect_figure2f_rows(global_rows)
    if not rows:
        print("  ⚠ Figure 2F 未绘制：没有有效的 X/Y 阈值指标行。")
        return {}

    values_csv = out_dir / "Figure2F_accuracy_remaining_rate_values.csv"
    write_figure2f_values_csv(rows, values_csv)

    results: Dict[str, Any] = {"values_csv": str(values_csv)}
    for series in ["X", "Y"]:
        paths = _plot_single_figure2f_series(
            rows=rows,
            series=series,
            out_dir=out_dir,
            accuracy_ymin=args.figure2f_ymin,
            accuracy_ymax=args.figure2f_ymax,
        )
        if paths:
            results[series] = paths

    return results


# =========================
# 10) Figure 2B
# =========================
def _collect_figure2b_rows(global_rows: List[Dict[str, Any]], condition: str = "argmax_all") -> List[Dict[str, Any]]:
    rows: List[Dict[str, Any]] = []
    for r in global_rows:
        if r.get("status") != "OK":
            continue
        if r.get("condition") != condition:
            continue
        group = str(r.get("group", ""))
        if group not in {"POS_X", "NEG_Y"}:
            continue
        rows.append(r)
    return rows

def write_figure2b_values_csv(rows: List[Dict[str, Any]], out_path: Path):
    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fields = [
        "group", "task", "model", "condition",
        "accuracy", "precision_pos_label1", "recall_sensitivity_TPR_pos_label1",
        "f1_pos_label1", "roc_auc_pos", "average_precision_pos",
        "sensitivity_TPR", "specificity_TNR", "mcc",
    ]
    with open(out_path, "w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fields)
        writer.writeheader()
        for row in rows:
            task = "CB_vs_X" if row.get("group") == "POS_X" else "CB_vs_Y"
            writer.writerow({
                "group": row.get("group", ""),
                "task": task,
                "model": "LiteMamba",
                "condition": row.get("condition", ""),
                "accuracy": row.get("accuracy", ""),
                "precision_pos_label1": row.get("precision_pos_label1", ""),
                "recall_sensitivity_TPR_pos_label1": row.get("recall_sensitivity_TPR_pos_label1", ""),
                "f1_pos_label1": row.get("f1_pos_label1", ""),
                "roc_auc_pos": row.get("roc_auc_pos", ""),
                "average_precision_pos": row.get("average_precision_pos", ""),
                "sensitivity_TPR": row.get("recall_sensitivity_TPR_pos_label1", ""),
                "specificity_TNR": row.get("specificity_TNR", ""),
                "mcc": row.get("mcc", ""),
            })

def _figure2b_metric_items() -> List[Tuple[str, str]]:
    return [
        ("accuracy", "Accuracy"),
        ("precision_pos_label1", "Precision"),
        ("recall_sensitivity_TPR_pos_label1", "Recall"),
        ("f1_pos_label1", "F1-score"),
        ("roc_auc_pos", "AUROC"),
        ("average_precision_pos", "AUPRC"),
        ("specificity_TNR", "Specificity"),
        ("mcc", "MCC"),
    ]

def _auto_figure2b_ylim_from_values(values: List[float]) -> Tuple[float, float]:
    vals = [v for v in values if np.isfinite(v)]
    if not vals:
        return 0.0, 1.05

    vmin = min(vals)
    vmax = max(vals)

    if vmin >= 0.90:
        ymin = max(0.80, np.floor((vmin - 0.03) * 100.0) / 100.0)
        tick_step = 0.02
    elif vmin >= 0.75:
        ymin = max(0.60, np.floor((vmin - 0.05) * 20.0) / 20.0)
        tick_step = 0.05
    elif vmin >= 0.50:
        ymin = max(0.40, np.floor((vmin - 0.08) * 10.0) / 10.0)
        tick_step = 0.10
    else:
        ymin = min(0.0, np.floor((vmin - 0.10) * 10.0) / 10.0)
        tick_step = 0.10

    data_span = max(vmax - ymin, tick_step)
    top_padding = max(0.025, data_span * 0.12)
    ymax = vmax + top_padding
    ymax = np.ceil(ymax / tick_step) * tick_step
    ymax = max(ymax, 1.02 if vmax >= 0.98 else vmax + top_padding)

    return float(ymin), float(ymax)

def _plot_single_figure2b_task(
        row: Dict[str, Any],
        group_key: str,
        task_label: str,
        out_dir: Path,
        metric_items: List[Tuple[str, str]]) -> Dict[str, str]:
    color = FIG2B_COLORS[group_key]
    x = np.arange(len(metric_items), dtype=float)
    vals = np.asarray([safe_float(row.get(key)) for key, _ in metric_items], dtype=float)

    if not np.any(np.isfinite(vals)):
        print(f"  ⚠ Figure 2B {task_label} 未绘制：指标值均无效。")
        return {}

    ymin, ymax = _auto_figure2b_ylim_from_values(list(vals))
    y_range = ymax - ymin

    fig, ax = plt.subplots(figsize=(13.5, 6.8))

    ax.vlines(x, ymin, vals, colors=color, linewidth=FIG2B_LINEWIDTH, zorder=2)
    ax.scatter(
        x, vals, s=FIG2B_POINTSIZE, color=color,
        edgecolor="white", linewidth=2.0, zorder=3,
    )

    for xi, yi in zip(x, vals):
        if not np.isfinite(yi):
            continue
        ax.annotate(
            f"{yi:.3f}",
            xy=(xi, yi),
            xytext=(0, 3),
            textcoords="offset points",
            ha="center", va="bottom",
            fontsize=FIG2B_TEXT_FONTSIZE,
            color=color, fontweight="bold",
            clip_on=True, zorder=4,
        )

    ax.text(
        0.015, 0.965, task_label,
        transform=ax.transAxes,
        ha="left", va="top",
        fontsize=FIG2B_SUBPANEL_FONTSIZE,
        color=color, fontweight="bold",
    )

    ax.set_xlim(-0.55, len(metric_items) - 0.45)
    ax.set_ylim(ymin, ymax)
    ax.set_ylabel("Score", fontsize=FIG2B_AXIS_LABEL_FONTSIZE)
    ax.set_xlabel("Metric", fontsize=FIG2B_AXIS_LABEL_FONTSIZE)
    ax.set_xticks(x)
    ax.set_xticklabels(
        [label for _, label in metric_items],
        rotation=35, ha="right", fontsize=FIG2B_XTICK_FONTSIZE,
    )

    if y_range <= 0.18:
        ax.yaxis.set_major_locator(MultipleLocator(0.02))
    elif y_range <= 0.35:
        ax.yaxis.set_major_locator(MultipleLocator(0.05))
    else:
        ax.yaxis.set_major_locator(MultipleLocator(0.10))
    ax.yaxis.set_major_formatter(FormatStrFormatter("%.2f"))

    style_axes_frame(ax, spine_lw=3, tick_lw=3, tick_len=7)
    ax.tick_params(axis="y", labelsize=FIG2B_TICK_FONTSIZE, pad=8)
    ax.tick_params(axis="x", pad=8)

    fig.subplots_adjust(left=0.10, right=0.985, bottom=0.27, top=0.97)

    task_suffix = "X" if group_key == "POS_X" else "Y"
    stem = f"Figure2B_litemamba_{task_suffix}_overall_performance"
    png_path = out_dir / f"{stem}.png"
    pdf_path = out_dir / f"{stem}.pdf"
    svg_path = out_dir / f"{stem}.svg"
    fig.savefig(png_path, dpi=600, bbox_inches="tight")
    fig.savefig(pdf_path, bbox_inches="tight")
    fig.savefig(svg_path, bbox_inches="tight")
    plt.close(fig)

    return {
        "png": str(png_path),
        "pdf": str(pdf_path),
        "svg": str(svg_path),
    }

def _plot_combined_figure2b_tasks(
        rows: List[Dict[str, Any]],
        out_dir: Path,
        metric_items: List[Tuple[str, str]]) -> Dict[str, str]:
    row_x = next((r for r in rows if r.get("group") == "POS_X"), None)
    row_y = next((r for r in rows if r.get("group") == "NEG_Y"), None)

    if row_x is None or row_y is None:
        print("  ⚠ Figure 2B combined 总体性能图未绘制：缺少 X 或 Y 的结果。")
        return {}

    vals_x = np.asarray([safe_float(row_x.get(key)) for key, _ in metric_items], dtype=float)
    vals_y = np.asarray([safe_float(row_y.get(key)) for key, _ in metric_items], dtype=float)

    finite_vals = [float(v) for v in np.concatenate([vals_x, vals_y]) if np.isfinite(v)]
    if not finite_vals:
        print("  ⚠ Figure 2B combined 总体性能图未绘制：X 和 Y 的指标值均无效。")
        return {}

    ymin, ymax = _auto_figure2b_ylim_from_values(finite_vals)
    y_range = ymax - ymin

    x = np.arange(len(metric_items), dtype=float)
    offset = 0.15

    fig, ax = plt.subplots(figsize=(13.5, 6.8))

    x_pos_x = x - offset
    x_pos_y = x + offset

    color_x = FIG2B_COLORS["POS_X"]
    color_y = FIG2B_COLORS["NEG_Y"]

    ax.vlines(x_pos_x, ymin, vals_x, colors=color_x, linewidth=FIG2B_LINEWIDTH, zorder=2)
    ax.scatter(
        x_pos_x, vals_x, s=FIG2B_POINTSIZE, color=color_x,
        edgecolor="white", linewidth=2.0, zorder=3,
    )

    ax.vlines(x_pos_y, ymin, vals_y, colors=color_y, linewidth=FIG2B_LINEWIDTH, zorder=2)
    ax.scatter(
        x_pos_y, vals_y, s=FIG2B_POINTSIZE, color=color_y,
        edgecolor="white", linewidth=2.0, zorder=3,
    )

    # =========================
    # 数值标注位置微调
    # 左侧 X：稍微向左上移动
    # 右侧 Y：稍微向右下移动
    # 这样当 X/Y 指标值接近时，不容易上下或左右重叠。
    # 移动幅度设置得较小，避免破坏原图整体布局。
    # =========================
    x_label_offset = (-4, 4)    # 左侧数字：左上
    y_label_offset = (4, -4)    # 右侧数字：右下

    for xi, yi in zip(x_pos_x, vals_x):
        if np.isfinite(yi):
            ax.annotate(
                f"{yi:.3f}",
                xy=(xi, yi),
                xytext=x_label_offset,
                textcoords="offset points",
                ha="right",
                va="bottom",
                fontsize=FIG2B_TEXT_FONTSIZE,
                color=color_x,
                fontweight="bold",
                clip_on=False,
                zorder=4,
            )

    for xi, yi in zip(x_pos_y, vals_y):
        if np.isfinite(yi):
            ax.annotate(
                f"{yi:.3f}",
                xy=(xi, yi),
                xytext=y_label_offset,
                textcoords="offset points",
                ha="left",
                va="top",
                fontsize=FIG2B_TEXT_FONTSIZE,
                color=color_y,
                fontweight="bold",
                clip_on=False,
                zorder=4,
            )

    ax.set_xlim(-0.60, len(metric_items) - 0.40)
    ax.set_ylim(ymin, ymax)
    ax.set_ylabel("Score", fontsize=FIG2B_AXIS_LABEL_FONTSIZE)
    ax.set_xlabel("Metric", fontsize=FIG2B_AXIS_LABEL_FONTSIZE)
    ax.set_xticks(x)
    ax.set_xticklabels(
        [label for _, label in metric_items],
        rotation=35,
        ha="right",
        fontsize=FIG2B_XTICK_FONTSIZE,
    )

    if y_range <= 0.18:
        ax.yaxis.set_major_locator(MultipleLocator(0.02))
    elif y_range <= 0.35:
        ax.yaxis.set_major_locator(MultipleLocator(0.05))
    else:
        ax.yaxis.set_major_locator(MultipleLocator(0.10))
    ax.yaxis.set_major_formatter(FormatStrFormatter("%.2f"))

    style_axes_frame(ax, spine_lw=3, tick_lw=3, tick_len=7)
    ax.tick_params(axis="y", labelsize=FIG2B_TICK_FONTSIZE, pad=8)
    ax.tick_params(axis="x", pad=8)

    legend_handles = [
        Line2D(
            [0], [0],
            color=color_x,
            linewidth=FIG2B_LINEWIDTH,
            marker="o",
            markersize=10,
            markerfacecolor=color_x,
            markeredgecolor="white",
            markeredgewidth=2.0,
            label="CB (Canonical Base) vs X"
        ),
        Line2D(
            [0], [0],
            color=color_y,
            linewidth=FIG2B_LINEWIDTH,
            marker="o",
            markersize=10,
            markerfacecolor=color_y,
            markeredgecolor="white",
            markeredgewidth=2.0,
            label="CB (Canonical Base) vs Y"
        ),
    ]
    ax.legend(
        handles=legend_handles,
        loc="best",
        frameon=False,
        prop={"size": FIG2B_SUBPANEL_FONTSIZE, "weight": "bold"},
        handlelength=2.2,
    )

    fig.subplots_adjust(left=0.10, right=0.985, bottom=0.27, top=0.97)

    stem = "Figure2B_litemamba_XY_overall_performance_combined"
    png_path = out_dir / f"{stem}.png"
    pdf_path = out_dir / f"{stem}.pdf"
    svg_path = out_dir / f"{stem}.svg"

    fig.savefig(png_path, dpi=600, bbox_inches="tight")
    fig.savefig(pdf_path, bbox_inches="tight")
    fig.savefig(svg_path, bbox_inches="tight")
    plt.close(fig)

    print(f"  ✓ Figure 2B combined 总体性能图 PNG: {png_path}")
    print(f"  ✓ Figure 2B combined 总体性能图 PDF: {pdf_path}")
    print(f"  ✓ Figure 2B combined 总体性能图 SVG: {svg_path}")

    return {
        "png": str(png_path),
        "pdf": str(pdf_path),
        "svg": str(svg_path),
    }

def _generate_figure2b_roc_pr_curves(out_dir: Path, device: torch.device, args) -> Dict[str, Dict[str, str]]:
    out_dir = Path(out_dir)
    curve_dir = out_dir / "roc_pr_curves"
    curve_dir.mkdir(parents=True, exist_ok=True)

    results: Dict[str, Dict[str, str]] = {}
    cached: Dict[str, Dict[str, Any]] = {}
    specs = [
        ("POS_X", Path(args.pos_csv), Path(args.pos_model), "X", "CB (Canonical Base) vs X", FIG2F_COLORS["X"]),
        ("NEG_Y", Path(args.neg_csv), Path(args.neg_model), "Y", "CB (Canonical Base) vs Y", FIG2F_COLORS["Y"]),
    ]

    for group_key, csv_path, model_path, modified_label_name, title_prefix, line_color in specs:
        try:
            model = load_litemamba_model(
                model_path=model_path,
                device=device,
                raw_signal_len=args.raw_signal_len,
                num_classes=2,
                seq_len=args.mer_len,
            )
            y_true_all, y_prob_all = predict_one_file(
                model=model,
                csv_path=csv_path,
                device=device,
                args=args,
            )
            label_names = get_label_names(
                csv_path,
                modified_label_name=modified_label_name,
                mer_len=args.mer_len,
                natural_label_name=args.natural_label_name,
                use_kmer_base_label=False,
            )
            cached[group_key] = {
                "y_true": y_true_all,
                "y_prob": y_prob_all,
                "label_names": label_names,
                "modified_label_name": modified_label_name,
                "title_prefix": title_prefix,
                "line_color": line_color,
            }

            task_suffix = "X" if group_key == "POS_X" else "Y"
            out_roc = curve_dir / f"Figure2B_litemamba_{task_suffix}_ROC.png"
            out_pr = curve_dir / f"Figure2B_litemamba_{task_suffix}_PR.png"
            plot_single_roc_pr(
                y_true=y_true_all,
                y_prob=y_prob_all,
                out_roc=out_roc,
                out_pr=out_pr,
                label_names=label_names,
                title_prefix=title_prefix,
                roc_xlim=(args.roc_xmin, args.roc_xmax),
                roc_ylim=(args.roc_ymin, args.roc_ymax),
                pr_xlim=(args.pr_xmin, args.pr_xmax),
                pr_ylim=(args.pr_ymin, args.pr_ymax),
            )

            out_roc_pdf = curve_dir / f"Figure2B_litemamba_{task_suffix}_ROC.pdf"
            out_pr_pdf = curve_dir / f"Figure2B_litemamba_{task_suffix}_PR.pdf"
            out_roc_svg = curve_dir / f"Figure2B_litemamba_{task_suffix}_ROC.svg"
            out_pr_svg = curve_dir / f"Figure2B_litemamba_{task_suffix}_PR.svg"

            fig, ax = plt.subplots(figsize=(8, 8))
            if len(np.unique(y_true_all)) == 2:
                fpr, tpr, _ = roc_curve(y_true_all, y_prob_all[:, 1])
                auc = roc_auc_score(y_true_all, y_prob_all[:, 1])
                ax.plot(fpr, tpr, linewidth=3, color=line_color, label=f"{modified_label_name} AUROC={auc:.3f}")
            else:
                ax.text(0.5, 0.5, "Only one class retained\nROC is undefined", ha="center", va="center", fontsize=18)
            ax.plot([0, 1], [0, 1], linestyle="--", linewidth=1.5, color="gray", alpha=0.8)
            ax.set_xlabel("False Positive Rate", fontsize=CURVE_AXIS_LABEL_FONTSIZE)
            ax.set_ylabel("True Positive Rate", fontsize=CURVE_AXIS_LABEL_FONTSIZE)
            ax.set_title(f"{title_prefix} ROC", fontsize=CURVE_TITLE_FONTSIZE, fontweight="bold", pad=12)
            force_ticks_01(ax, xlim=(args.roc_xmin, args.roc_xmax), ylim=(args.roc_ymin, args.roc_ymax), step=0.1)
            try:
                ax.set_box_aspect(1)
            except Exception:
                ax.set_aspect("equal", adjustable="box")
            ax.tick_params(direction="out", length=6, width=3, labelsize=CURVE_TICK_FONTSIZE)
            for spine in ax.spines.values():
                spine.set_linewidth(3)
            fix_corner_ticklabel_overlap(ax)
            ax.legend(loc="lower right", prop={"size": CURVE_LEGEND_FONTSIZE}, frameon=False)
            plt.tight_layout()
            fig.savefig(out_roc_pdf, bbox_inches="tight")
            fig.savefig(out_roc_svg, bbox_inches="tight")
            plt.close(fig)

            fig, ax = plt.subplots(figsize=(8, 8))
            if len(np.unique(y_true_all)) == 2:
                precision, recall, _ = precision_recall_curve(y_true_all, y_prob_all[:, 1])
                ap = average_precision_score(y_true_all, y_prob_all[:, 1])
                ax.plot(recall, precision, linewidth=3, color=line_color, label=f"{modified_label_name} AUPRC={ap:.3f}")
            else:
                ax.text(0.5, 0.5, "Only one class retained\nPR is undefined", ha="center", va="center", fontsize=18)
            ax.set_xlabel("Recall", fontsize=CURVE_AXIS_LABEL_FONTSIZE)
            ax.set_ylabel("Precision", fontsize=CURVE_AXIS_LABEL_FONTSIZE)
            ax.set_title(f"{title_prefix} PR", fontsize=CURVE_TITLE_FONTSIZE, fontweight="bold", pad=12)
            force_ticks_01(ax, xlim=(args.pr_xmin, args.pr_xmax), ylim=(args.pr_ymin, args.pr_ymax), step=0.1)
            try:
                ax.set_box_aspect(1)
            except Exception:
                ax.set_aspect("equal", adjustable="box")
            ax.tick_params(direction="out", length=6, width=3, labelsize=CURVE_TICK_FONTSIZE)
            for spine in ax.spines.values():
                spine.set_linewidth(3)
            fix_corner_ticklabel_overlap(ax)
            ax.legend(loc="lower left", prop={"size": CURVE_LEGEND_FONTSIZE}, frameon=False)
            plt.tight_layout()
            fig.savefig(out_pr_pdf, bbox_inches="tight")
            fig.savefig(out_pr_svg, bbox_inches="tight")
            plt.close(fig)

            results[group_key] = {
                "roc_png": str(out_roc),
                "roc_pdf": str(out_roc_pdf),
                "roc_svg": str(out_roc_svg),
                "pr_png": str(out_pr),
                "pr_pdf": str(out_pr_pdf),
                "pr_svg": str(out_pr_svg),
            }
            del model
            gc.collect()
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
            print(f"  ✓ Figure 2B {title_prefix} ROC/PR 已输出到: {curve_dir}")
        except Exception as e:
            print(f"  ⚠ Figure 2B {title_prefix} ROC/PR 绘制失败: {type(e).__name__}: {e}")

    # combined ROC/PR
    if len(cached) >= 2:
        combined_roc_png = curve_dir / "Figure2B_litemamba_XY_combined_ROC.png"
        combined_roc_pdf = curve_dir / "Figure2B_litemamba_XY_combined_ROC.pdf"
        combined_roc_svg = curve_dir / "Figure2B_litemamba_XY_combined_ROC.svg"
        fig, ax = plt.subplots(figsize=(8, 8))
        plotted = 0
        for group_key in ["POS_X", "NEG_Y"]:
            info = cached.get(group_key)
            if info is None:
                continue
            y_true_all = info["y_true"]
            y_prob_all = info["y_prob"]
            modified_label_name = info["modified_label_name"]
            line_color = info["line_color"]
            if len(np.unique(y_true_all)) == 2:
                fpr, tpr, _ = roc_curve(y_true_all, y_prob_all[:, 1])
                auc = roc_auc_score(y_true_all, y_prob_all[:, 1])
                ax.plot(fpr, tpr, linewidth=3, color=line_color, label=f"{modified_label_name} AUROC={auc:.3f}")
                plotted += 1
        ax.plot([0, 1], [0, 1], linestyle="--", linewidth=1.5, color="gray", alpha=0.8)
        if plotted == 0:
            ax.text(0.5, 0.5, "No valid ROC curve", ha="center", va="center", fontsize=18)
        ax.set_xlabel("False Positive Rate", fontsize=CURVE_AXIS_LABEL_FONTSIZE)
        ax.set_ylabel("True Positive Rate", fontsize=CURVE_AXIS_LABEL_FONTSIZE)
        ax.set_title("CB (Canonical Base) vs X/Y ROC", fontsize=CURVE_TITLE_FONTSIZE, fontweight="bold", pad=12)
        force_ticks_01(ax, xlim=(args.roc_xmin, args.roc_xmax), ylim=(args.roc_ymin, args.roc_ymax), step=0.1)
        try:
            ax.set_box_aspect(1)
        except Exception:
            ax.set_aspect("equal", adjustable="box")
        ax.tick_params(direction="out", length=6, width=3, labelsize=CURVE_TICK_FONTSIZE)
        for spine in ax.spines.values():
            spine.set_linewidth(3)
        fix_corner_ticklabel_overlap(ax)
        ax.legend(loc="lower right", prop={"size": CURVE_LEGEND_FONTSIZE}, frameon=False)
        plt.tight_layout()
        fig.savefig(combined_roc_png, dpi=600, bbox_inches="tight")
        fig.savefig(combined_roc_pdf, bbox_inches="tight")
        fig.savefig(combined_roc_svg, bbox_inches="tight")
        plt.close(fig)

        combined_pr_png = curve_dir / "Figure2B_litemamba_XY_combined_PR.png"
        combined_pr_pdf = curve_dir / "Figure2B_litemamba_XY_combined_PR.pdf"
        combined_pr_svg = curve_dir / "Figure2B_litemamba_XY_combined_PR.svg"
        fig, ax = plt.subplots(figsize=(8, 8))
        plotted = 0
        for group_key in ["POS_X", "NEG_Y"]:
            info = cached.get(group_key)
            if info is None:
                continue
            y_true_all = info["y_true"]
            y_prob_all = info["y_prob"]
            modified_label_name = info["modified_label_name"]
            line_color = info["line_color"]
            if len(np.unique(y_true_all)) == 2:
                precision, recall, _ = precision_recall_curve(y_true_all, y_prob_all[:, 1])
                ap = average_precision_score(y_true_all, y_prob_all[:, 1])
                ax.plot(recall, precision, linewidth=3, color=line_color, label=f"{modified_label_name} AUPRC={ap:.3f}")
                plotted += 1
        if plotted == 0:
            ax.text(0.5, 0.5, "No valid PR curve", ha="center", va="center", fontsize=18)
        ax.set_xlabel("Recall", fontsize=CURVE_AXIS_LABEL_FONTSIZE)
        ax.set_ylabel("Precision", fontsize=CURVE_AXIS_LABEL_FONTSIZE)
        ax.set_title("CB (Canonical Base) vs X/Y PR", fontsize=CURVE_TITLE_FONTSIZE, fontweight="bold", pad=12)
        force_ticks_01(ax, xlim=(args.pr_xmin, args.pr_xmax), ylim=(args.pr_ymin, args.pr_ymax), step=0.1)
        try:
            ax.set_box_aspect(1)
        except Exception:
            ax.set_aspect("equal", adjustable="box")
        ax.tick_params(direction="out", length=6, width=3, labelsize=CURVE_TICK_FONTSIZE)
        for spine in ax.spines.values():
            spine.set_linewidth(3)
        fix_corner_ticklabel_overlap(ax)
        ax.legend(loc="lower left", prop={"size": CURVE_LEGEND_FONTSIZE}, frameon=False)
        plt.tight_layout()
        fig.savefig(combined_pr_png, dpi=600, bbox_inches="tight")
        fig.savefig(combined_pr_pdf, bbox_inches="tight")
        fig.savefig(combined_pr_svg, bbox_inches="tight")
        plt.close(fig)

        results["combined_xy"] = {
            "roc_png": str(combined_roc_png),
            "roc_pdf": str(combined_roc_pdf),
            "roc_svg": str(combined_roc_svg),
            "pr_png": str(combined_pr_png),
            "pr_pdf": str(combined_pr_pdf),
            "pr_svg": str(combined_pr_svg),
        }
        print(f"  ✓ Figure 2B X+Y 合并 ROC/PR 已输出到: {curve_dir}")

    return results

def generate_figure2b(global_rows: List[Dict[str, Any]], args, device: torch.device) -> Dict[str, Any]:
    out_dir = Path(args.figure2b_out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    rows = _collect_figure2b_rows(global_rows, condition=args.figure2b_condition)
    if not rows:
        print("  ⚠ Figure 2B 未绘制：没有找到指定条件下的有效总体性能结果。")
        return {}

    values_csv = out_dir / "Figure2B_litemamba_overall_performance_values.csv"
    write_figure2b_values_csv(rows, values_csv)

    metric_items = _figure2b_metric_items()
    results: Dict[str, Any] = {"values_csv": str(values_csv)}

    task_order = [
        ("POS_X", "CB (Canonical Base) vs X"),
        ("NEG_Y", "CB (Canonical Base) vs Y"),
    ]
    for group_key, task_label in task_order:
        row = next((r for r in rows if r.get("group") == group_key), None)
        if row is None:
            print(f"  ⚠ Figure 2B {task_label} 未绘制：未找到对应结果。")
            continue
        paths = _plot_single_figure2b_task(
            row=row,
            group_key=group_key,
            task_label=task_label,
            out_dir=out_dir,
            metric_items=metric_items,
        )
        if paths:
            results[group_key] = paths

    combined_paths = _plot_combined_figure2b_tasks(
        rows=rows,
        out_dir=out_dir,
        metric_items=metric_items,
    )
    if combined_paths:
        results["combined_xy_overall_performance"] = combined_paths

    curve_paths = _generate_figure2b_roc_pr_curves(out_dir=out_dir, device=device, args=args)
    if curve_paths:
        results["roc_pr_curves"] = curve_paths

    return results

# =========================
# 11) 文件写入
# =========================
def json_safe(obj: Any) -> Any:
    if isinstance(obj, dict):
        return {str(k): json_safe(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple)):
        return [json_safe(v) for v in obj]
    if isinstance(obj, np.ndarray):
        return obj.tolist()
    if isinstance(obj, (np.integer,)):
        return int(obj)
    if isinstance(obj, (np.floating,)):
        v = float(obj)
        if np.isnan(v) or np.isinf(v):
            return None
        return v
    if isinstance(obj, float):
        if np.isnan(obj) or np.isinf(obj):
            return None
        return obj
    return obj

def write_metrics_csv(metrics_rows: List[Dict[str, Any]], out_path: Path):
    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fields = [
        "group", "csv_file", "kmer", "label0_name", "label1_name",
        "condition", "condition_label", "threshold", "status",
        "total_original", "kept_samples", "rejected_samples", "coverage",
        "n_true_0", "n_true_1", "n_pred_0", "n_pred_1",
        "TN", "FP", "FN", "TP",
        "accuracy", "balanced_accuracy",
        "macro_precision", "macro_recall", "macro_f1",
        "weighted_precision", "weighted_recall", "weighted_f1",
        "precision_pos_label1", "recall_sensitivity_TPR_pos_label1", "f1_pos_label1",
        "precision_label0", "recall_specificity_TNR_label0", "f1_label0",
        "specificity_TNR", "NPV", "FPR", "FNR", "FDR", "FOR",
        "mcc", "cohen_kappa",
        "roc_auc_pos", "roc_auc_neg", "roc_auc_macro",
        "average_precision_pos", "average_precision_neg", "average_precision_macro",
        "log_loss", "brier_score_pos", "mean_max_prob_kept", "median_max_prob_kept", "ece_15bins",
        "confusion_matrix_png", "roc_png", "pr_png",
    ]
    with open(out_path, "w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fields)
        writer.writeheader()
        for row in metrics_rows:
            writer.writerow({k: row.get(k, "") for k in fields})

def write_classification_report(y_true: np.ndarray,
                                y_pred: np.ndarray,
                                label_names: Dict[int, str],
                                out_path: Path):
    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    report = classification_report(
        y_true,
        y_pred,
        labels=[0, 1],
        target_names=[label_names[0], label_names[1]],
        digits=4,
        zero_division=0,
    )
    with open(out_path, "w", encoding="utf-8") as f:
        f.write(report)

def save_predictions_csv(y_true: np.ndarray,
                         y_prob: np.ndarray,
                         out_path: Path):
    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    y_pred = np.argmax(y_prob, axis=1)
    max_prob = np.max(y_prob, axis=1)
    with open(out_path, "w", encoding="utf-8", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["row_index", "y_true", "y_pred", "prob_0", "prob_1", "max_prob"])
        for i in range(y_true.shape[0]):
            writer.writerow([
                i,
                int(y_true[i]),
                int(y_pred[i]),
                float(y_prob[i, 0]),
                float(y_prob[i, 1]),
                float(max_prob[i]),
            ])

# =========================
# 12) 单个文件/文件夹处理
# =========================
def evaluate_one_csv(group_name: str,
                     csv_path: Path,
                     output_root: Path,
                     model: FusionLiteMamba,
                     modified_label_name: str,
                     device: torch.device,
                     args,
                     dataset_name: Optional[str] = None) -> List[Dict[str, Any]]:
    st = time.time()
    csv_path = Path(csv_path)
    output_root = Path(output_root)

    if dataset_name is None:
        kmer = get_kmer_from_file(csv_path, mer_len=args.mer_len)
    else:
        kmer = dataset_name

    label_names = get_label_names(
        csv_path,
        modified_label_name=modified_label_name,
        mer_len=args.mer_len,
        natural_label_name=args.natural_label_name,
        use_kmer_base_label=args.use_kmer_base_label,
    )

    file_out_dir = output_root / csv_path.stem
    cm_dir = file_out_dir / "confusion_matrices"
    curve_dir = file_out_dir / "roc_pr_curves"
    report_dir = file_out_dir / "classification_reports"
    values_dir = file_out_dir / "values"
    for d in [cm_dir, curve_dir, report_dir, values_dir]:
        d.mkdir(parents=True, exist_ok=True)

    print(f"  kmer={kmer}, label0={label_names[0]}, label1={label_names[1]}")
    y_true_all, y_prob_all = predict_one_file(
        model=model,
        csv_path=csv_path,
        device=device,
        args=args,
    )
    total = int(y_true_all.size)
    print(f"  推理完成: N={total}, true0={int(np.sum(y_true_all == 0))}, true1={int(np.sum(y_true_all == 1))}")

    if args.save_predictions:
        save_predictions_csv(y_true_all, y_prob_all, values_dir / f"{csv_path.stem}_predictions.csv")

    thresholds: List[Optional[float]] = [None] + list(args.thresholds)
    metrics_rows: List[Dict[str, Any]] = []
    metrics_full: Dict[str, Any] = {
        "group": group_name,
        "csv_path": str(csv_path),
        "csv_file": csv_path.name,
        "kmer": kmer,
        "label_names": label_names,
        "thresholds": [None] + [float(x) for x in args.thresholds],
        "conditions": {},
    }

    for thr in thresholds:
        cond = condition_name(thr)
        cond_lab = condition_label(thr)
        print(f"    评估条件: {cond_lab}")

        metric, y_true, y_pred, y_prob = compute_metrics_for_mask(
            y_true_all=y_true_all,
            y_prob_all=y_prob_all,
            threshold=thr,
            label_names=label_names,
            total_original=total,
        )

        title_prefix = f"{kmer} | {cond_lab}"
        cm_path = cm_dir / f"{csv_path.stem}_{cond}_confusion_matrix_norm.png"
        roc_path = curve_dir / f"{csv_path.stem}_{cond}_roc.png"
        pr_path = curve_dir / f"{csv_path.stem}_{cond}_pr.png"
        rep_path = report_dir / f"{csv_path.stem}_{cond}_classification_report.txt"

        if metric.get("status") == "OK" and y_true.size > 0 and not args.no_plots:
            plot_confusion_matrix_norm(
                y_true=y_true,
                y_pred=y_pred,
                out_path=cm_path,
                label_names=label_names,
            )
            write_classification_report(y_true, y_pred, label_names, rep_path)
            plot_single_roc_pr(
                y_true=y_true,
                y_prob=y_prob,
                out_roc=roc_path,
                out_pr=pr_path,
                label_names=label_names,
                title_prefix=title_prefix,
                roc_xlim=(args.roc_xmin, args.roc_xmax),
                roc_ylim=(args.roc_ymin, args.roc_ymax),
                pr_xlim=(args.pr_xmin, args.pr_xmax),
                pr_ylim=(args.pr_ymin, args.pr_ymax),
            )
            metric["confusion_matrix_png"] = str(cm_path)
            metric["roc_png"] = str(roc_path)
            metric["pr_png"] = str(pr_path)
            metric["classification_report_txt"] = str(rep_path)
        else:
            metric["confusion_matrix_png"] = ""
            metric["roc_png"] = ""
            metric["pr_png"] = ""
            metric["classification_report_txt"] = ""

        metric.update({
            "group": group_name,
            "csv_path": str(csv_path),
            "csv_file": csv_path.name,
            "kmer": kmer,
        })
        metrics_rows.append(metric)
        metrics_full["conditions"][cond] = metric

    multi_roc_path = curve_dir / f"{csv_path.stem}_ALL_thresholds_roc.png"
    multi_pr_path = curve_dir / f"{csv_path.stem}_ALL_thresholds_pr.png"
    if not args.no_plots:
        plot_multi_threshold_roc_pr(
            y_true_all=y_true_all,
            y_prob_all=y_prob_all,
            thresholds=thresholds,
            out_roc=multi_roc_path,
            out_pr=multi_pr_path,
            label_names=label_names,
            title_prefix=kmer,
            min_kept_for_curve=args.min_kept_for_curve,
            roc_xlim=(args.roc_xmin, args.roc_xmax),
            roc_ylim=(args.roc_ymin, args.roc_ymax),
            pr_xlim=(args.pr_xmin, args.pr_xmax),
            pr_ylim=(args.pr_ymin, args.pr_ymax),
        )
        metrics_full["multi_threshold_roc_png"] = str(multi_roc_path)
        metrics_full["multi_threshold_pr_png"] = str(multi_pr_path)
    else:
        metrics_full["multi_threshold_roc_png"] = ""
        metrics_full["multi_threshold_pr_png"] = ""

    per_file_metrics_csv = file_out_dir / f"{csv_path.stem}_metrics_all_thresholds.csv"
    per_file_metrics_json = file_out_dir / f"{csv_path.stem}_metrics_all_thresholds.json"
    write_metrics_csv(metrics_rows, per_file_metrics_csv)
    with open(per_file_metrics_json, "w", encoding="utf-8") as f:
        json.dump(json_safe(metrics_full), f, ensure_ascii=False, indent=2)

    print(f"  ✓ 完成: {csv_path.name}, 用时 {time.time() - st:.1f}s")
    print(f"    输出目录: {file_out_dir}")
    return metrics_rows

def process_one_csv_dataset(group_name: str,
                            input_csv: Path,
                            output_dir: Path,
                            model: FusionLiteMamba,
                            modified_label_name: str,
                            device: torch.device,
                            args) -> List[Dict[str, Any]]:
    input_csv = Path(input_csv)
    output_dir = Path(output_dir)

    if not input_csv.exists():
        raise FileNotFoundError(f"输入 CSV 不存在: {input_csv}")
    if not input_csv.is_file():
        raise FileNotFoundError(f"输入路径不是文件: {input_csv}")

    dataset_name = f"{group_name}_ALL_KMERS"

    print("\n" + "=" * 100)
    print(f"开始处理 {group_name}")
    print(f"输入 CSV  : {input_csv}")
    print(f"输出文件夹: {output_dir}")
    print(f"数据集名称: {dataset_name}")
    print("=" * 100)

    all_rows: List[Dict[str, Any]] = []
    try:
        rows = evaluate_one_csv(
            group_name=group_name,
            csv_path=input_csv,
            output_root=output_dir,
            model=model,
            modified_label_name=modified_label_name,
            device=device,
            args=args,
            dataset_name=dataset_name,
        )
        all_rows.extend(rows)
    except Exception as e:
        print(f"  ❌ 失败: {type(e).__name__}: {e}")
        label_names = get_label_names(
            input_csv,
            modified_label_name=modified_label_name,
            mer_len=args.mer_len,
            natural_label_name=args.natural_label_name,
            use_kmer_base_label=args.use_kmer_base_label,
        )
        all_rows.append({
            "group": group_name,
            "csv_path": str(input_csv),
            "csv_file": input_csv.name,
            "kmer": dataset_name,
            "label0_name": label_names[0],
            "label1_name": label_names[1],
            "condition": "ERROR",
            "condition_label": "ERROR",
            "threshold": "",
            "status": f"FAIL: {type(e).__name__}: {e}",
        })

    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()

    return all_rows

# =========================
# 13) 参数与主函数
# =========================
def parse_threshold_list(values: Optional[List[float]]) -> List[float]:
    if values is None or len(values) == 0:
        return [0.5, 0.6, 0.7, 0.8, 0.9]
    cleaned = sorted(set(float(x) for x in values))
    for x in cleaned:
        if x < 0.0 or x > 1.0:
            raise ValueError(f"threshold 必须在 [0,1] 范围内: {x}")
    return cleaned

def parse_args():
    parser = argparse.ArgumentParser(
        description="Evaluate binary CB-vs-X and CB-vs-Y LiteMamba checkpoints on 6-mer test CSV files."
    )

    parser.add_argument("--pos_csv", type=str, required=True,
                        help="Balanced binary test CSV for CB-vs-X.")
    parser.add_argument("--neg_csv", type=str, required=True,
                        help="Balanced binary test CSV for CB-vs-Y.")
    parser.add_argument("--pos_model", type=str, required=True,
                        help="LiteMamba checkpoint for CB-vs-X.")
    parser.add_argument("--neg_model", type=str, required=True,
                        help="LiteMamba checkpoint for CB-vs-Y.")
    parser.add_argument("--output_root", type=str, required=True,
                        help="Root directory for all evaluation outputs.")

    parser.add_argument("--device", type=str, default="auto")
    parser.add_argument("--batch_size", type=int, default=1024)
    parser.add_argument("--workers", type=int, default=0)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--amp", action="store_true")

    parser.add_argument("--mer_len", type=int, default=6)
    parser.add_argument("--raw_signal_len", type=int, default=30)
    parser.add_argument("--dwell_divisor", type=float, default=60.0)
    parser.add_argument("--quality_divisor", type=float, default=50.0)

    parser.add_argument("--thresholds", type=float, nargs="*", default=None)
    parser.add_argument("--natural_label_name", type=str, default="CB")
    parser.add_argument("--use_kmer_base_label", action="store_true")
    parser.add_argument("--min_kept_for_curve", type=int, default=20)

    parser.add_argument("--figure2b_condition", type=str, default="argmax_all")

    parser.add_argument(
        "--figure2f_metric", type=str, choices=["accuracy", "macro_f1", "both"],
        default="accuracy", help=argparse.SUPPRESS,
    )
    parser.add_argument("--figure2f_ymin", type=float, default=0.97)
    parser.add_argument("--figure2f_ymax", type=float, default=0.99)

    parser.add_argument("--save_predictions", action="store_true")
    parser.add_argument("--no_plots", action="store_true",
                        help="Skip all plotting and write metrics/tables only.")

    parser.add_argument("--roc_xmin", type=float, default=0.0)
    parser.add_argument("--roc_xmax", type=float, default=0.4)
    parser.add_argument("--roc_ymin", type=float, default=0.6)
    parser.add_argument("--roc_ymax", type=float, default=1.0)
    parser.add_argument("--pr_xmin", type=float, default=0.6)
    parser.add_argument("--pr_xmax", type=float, default=1.0)
    parser.add_argument("--pr_ymin", type=float, default=0.6)
    parser.add_argument("--pr_ymax", type=float, default=1.0)

    args = parser.parse_args()
    output_root = Path(args.output_root).resolve()
    args.pos_out_dir = str(output_root / "CB_vs_X")
    args.neg_out_dir = str(output_root / "CB_vs_Y")
    args.global_summary_csv = str(output_root / "global_summary.csv")
    args.figure2b_out_dir = str(output_root / "figure2b")
    args.figure2f_out_dir = str(output_root / "figure2f")

    for label, path_value in [
        ("--pos_csv", args.pos_csv),
        ("--neg_csv", args.neg_csv),
        ("--pos_model", args.pos_model),
        ("--neg_model", args.neg_model),
    ]:
        if not Path(path_value).is_file():
            parser.error(f"{label} file does not exist: {path_value}")

    if args.batch_size <= 0:
        parser.error("--batch_size must be > 0")
    if args.workers < 0:
        parser.error("--workers must be >= 0")

    args.thresholds = parse_threshold_list(args.thresholds)
    if args.dwell_divisor <= 0:
        raise ValueError("--dwell_divisor 必须 > 0")
    if args.quality_divisor <= 0:
        raise ValueError("--quality_divisor 必须 > 0")
    if args.figure2f_ymin is not None and not (0.0 <= args.figure2f_ymin < 1.0):
        raise ValueError("--figure2f_ymin 必须在 [0,1) 范围内")
    if not (0.0 < args.figure2f_ymax <= 1.0):
        raise ValueError("--figure2f_ymax 必须在 (0,1] 范围内")
    if args.figure2f_ymin is not None and args.figure2f_ymin >= args.figure2f_ymax:
        raise ValueError("--figure2f_ymin 必须小于 --figure2f_ymax")
    return args

def main():
    args = parse_args()
    set_seed(args.seed)
    enable_speedup()
    device = get_device(args.device)
    Path(args.output_root).mkdir(parents=True, exist_ok=True)

    print("=" * 100)
    print("Batch evaluation for XNA/CB 6-mer all-kmer test CSV files")
    print(f"device              : {device}")
    print(f"pos_csv             : {args.pos_csv}")
    print(f"neg_csv             : {args.neg_csv}")
    print(f"pos_model           : {args.pos_model}")
    print(f"neg_model           : {args.neg_model}")
    print(f"raw_signal_len      : {args.raw_signal_len}")
    print(f"dwell_divisor       : {args.dwell_divisor}")
    print(f"quality_divisor     : {args.quality_divisor}")
    print(f"thresholds          : {args.thresholds}")
    print(f"global_summary_csv  : {args.global_summary_csv}")
    print(f"figure2b_out_dir    : {args.figure2b_out_dir}")
    print(f"figure2b_condition  : {args.figure2b_condition}")
    print(f"figure2f_out_dir    : {args.figure2f_out_dir}")
    print("figure2f_layout     : split X/Y; x=threshold; left=accuracy; right=remaining rate")
    print("=" * 100)

    global_rows: List[Dict[str, Any]] = []

    pos_model = load_litemamba_model(
        model_path=Path(args.pos_model),
        device=device,
        raw_signal_len=args.raw_signal_len,
        num_classes=2,
        seq_len=args.mer_len,
    )
    pos_rows = process_one_csv_dataset(
        group_name="POS_X",
        input_csv=Path(args.pos_csv),
        output_dir=Path(args.pos_out_dir),
        model=pos_model,
        modified_label_name="X",
        device=device,
        args=args,
    )
    global_rows.extend(pos_rows)

    del pos_model
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()

    neg_model = load_litemamba_model(
        model_path=Path(args.neg_model),
        device=device,
        raw_signal_len=args.raw_signal_len,
        num_classes=2,
        seq_len=args.mer_len,
    )
    neg_rows = process_one_csv_dataset(
        group_name="NEG_Y",
        input_csv=Path(args.neg_csv),
        output_dir=Path(args.neg_out_dir),
        model=neg_model,
        modified_label_name="Y",
        device=device,
        args=args,
    )
    global_rows.extend(neg_rows)

    if args.no_plots:
        figure2b_paths = {}
        figure2f_paths = {}
    else:
        figure2b_paths = generate_figure2b(global_rows, args, device)
        figure2f_paths = generate_figure2f(global_rows, args)

    write_metrics_csv(global_rows, Path(args.global_summary_csv))

    n_ok = sum(1 for r in global_rows if r.get("status") == "OK")
    n_fail = sum(1 for r in global_rows if str(r.get("status", "")).startswith("FAIL") or r.get("condition") == "ERROR")

    print("\n" + "=" * 100)
    print("全部处理完成")
    print(f"OK rows   : {n_ok}")
    print(f"FAIL rows : {n_fail}")
    print(f"POS 输出  : {args.pos_out_dir}")
    print(f"NEG 输出  : {args.neg_out_dir}")
    print(f"汇总 CSV  : {args.global_summary_csv}")
    print(f"Figure 2B : {args.figure2b_out_dir}")
    print(f"Figure 2F : {args.figure2f_out_dir}")
    print("=" * 100)

    _ = figure2b_paths, figure2f_paths

if __name__ == "__main__":
    main()