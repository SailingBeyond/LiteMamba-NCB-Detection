#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Plot Figure 3D using UMAP embeddings from a trained LiteMamba model.

The input CSV must use the same 11-column format as train_all.py. The script
loads the public model definition from scripts/06_training/model_benchmark_all.py,
extracts either the 192-dimensional backbone vector or the 128-dimensional
classifier hidden vector, applies per-class reservoir sampling, and runs UMAP.
"""

import os
import csv
import gc
import time
import random
import argparse
import sys
from pathlib import Path
from typing import Iterator, Optional, List, Tuple, Dict

import numpy as np
import torch
from torch.utils.data import IterableDataset, DataLoader, get_worker_info

import matplotlib as mpl
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler

REPO_ROOT = Path(__file__).resolve().parents[2]
TRAINING_DIR = REPO_ROOT / "scripts" / "06_training"
if str(TRAINING_DIR) not in sys.path:
    sys.path.insert(0, str(TRAINING_DIR))

try:
    from model_benchmark_all import build_model
except ImportError as exc:
    raise ImportError(
        f"Could not import {TRAINING_DIR / 'model_benchmark_all.py'}. "
        "Place this script inside scripts/08_figures of the repository."
    ) from exc

try:
    import umap
except Exception as e:
    raise ImportError("Could not import umap. Install it with: pip install umap-learn") from e


# =========================
# 0) 画图风格：沿用你刚才聚类图脚本的方式
# =========================
mpl.rcParams["font.family"] = "sans-serif"
mpl.rcParams["font.sans-serif"] = ["Arial", "Liberation Sans", "DejaVu Sans"]
mpl.rcParams["pdf.fonttype"] = 42
mpl.rcParams["ps.fonttype"] = 42
mpl.rcParams["axes.unicode_minus"] = False

AXIS_LABEL_FONTSIZE = 30
TICK_FONTSIZE = 30
LEGEND_FONTSIZE = 25
LEGEND_TITLE_FONTSIZE = 28
TITLE_FONTSIZE = 18
LEGEND_TITLE = ""

# label 从 0 到 5 分别为 A T C G X Y
PLOT_ORDER = [0, 1, 2, 3, 4, 5]
LABEL_NAMES = {
    0: "A",
    1: "T",
    2: "C",
    3: "G",
    4: "X",
    5: "Y",
}

# 保持论文图常用柔和配色，顺序对应 A/T/C/G/X/Y
LABEL_COLORS = {
    0: "#377DB8",  # A
    1: "#D98484",  # T
    2: "#4EA648",  # C
    3: "#D6B16A",  # G
    4: "#B8A2CF",  # X
    5: "#8E4B98",  # Y
}


# =========================
# 1) Model and data conventions
# =========================

# =========================
# 2) Model definition
# =========================
# The architecture is imported from scripts/06_training/model_benchmark_all.py
# so UMAP extraction always uses the same public model definition as training.

# =========================
# 3) IterableDataset：逐行读取大 CSV，不一次性读入内存
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
        worker = get_worker_info()
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
# 4) 数据解析：严格对齐 train_all.py 的 11 列格式
# =========================
_MAP = {"A": 0, "C": 1, "T": 2, "G": 3, "N": 4}
BASE_DIM = 5
MERGED_FEATURE_DIM = BASE_DIM + 8


def encode_seq_block(seq: str, mer_len: int) -> Optional[np.ndarray]:
    """ATCGN -> (mer_len, 5) one-hot; non-ATCGN returns None."""
    seq = seq.strip().upper()
    if len(seq) != mer_len:
        return None

    out = np.zeros((mer_len, BASE_DIM), dtype=np.float32)
    for i, ch in enumerate(seq):
        v = _MAP.get(ch, None)
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
    """
    与 train_all.py 保持一致：
    - 每个碱基一个 signal segment，用 | 分隔；
    - segment 内电流值用 * 分隔；
    - 长度不足时两侧补 0；
    - 长度超出时 random.sample 采样并按原始顺序排列。
    """
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
    """Merge 5-dim base one-hot with 8 scalar features -> 13 dims per base."""
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

        # 跳过表头
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

        # 只保留 0~5 六类，避免脏标签影响绘图
        if label not in LABEL_NAMES:
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
            torch.empty(0, mer_len, BASE_DIM, dtype=torch.float32),
            torch.empty(0, mer_len, MERGED_FEATURE_DIM, dtype=torch.float32),
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
def set_seed(seed: int):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def get_device(device_arg: str) -> torch.device:
    if device_arg.lower() == "auto":
        return torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    return torch.device(device_arg)


def strip_state_dict_prefix(state: Dict[str, torch.Tensor]) -> Dict[str, torch.Tensor]:
    """兼容 DataParallel / torch.compile 可能产生的前缀。"""
    if not state:
        return state

    new_state = state
    prefixes = ["module.", "_orig_mod."]

    changed = True
    while changed:
        changed = False
        keys = list(new_state.keys())
        for prefix in prefixes:
            if all(k.startswith(prefix) for k in keys):
                new_state = {k[len(prefix):]: v for k, v in new_state.items()}
                changed = True
                break

    return new_state


def load_litemamba_model(
    model_path: Path,
    device: torch.device,
    raw_signal_len: int,
    num_classes: int = 6,
    seq_len: int = 6,
) -> torch.nn.Module:
    model_path = Path(model_path)
    if not model_path.exists():
        raise FileNotFoundError(f"Model checkpoint not found: {model_path}")

    model = build_model(
        model_type="litemamba",
        num_classes=num_classes,
        seq_len=seq_len,
        raw_signal_len=raw_signal_len,
    )
    checkpoint = torch.load(str(model_path), map_location=device)
    if isinstance(checkpoint, dict) and "state_dict" in checkpoint:
        state = checkpoint["state_dict"]
    elif isinstance(checkpoint, dict) and "model_state_dict" in checkpoint:
        state = checkpoint["model_state_dict"]
    else:
        state = checkpoint
    if not isinstance(state, dict):
        raise TypeError(f"Unsupported checkpoint object: {type(state)}")

    state = strip_state_dict_prefix(state)
    model.load_state_dict(state, strict=True)
    model.to(device)
    model.eval()
    print(f"Loaded LiteMamba checkpoint: {model_path}")
    return model


def get_loader(csv_path: Path,
               batch_size: int,
               num_workers: int,
               mer_len: int,
               raw_signal_len: int,
               dwell_divisor: float,
               quality_divisor: float) -> DataLoader:
    ds = LineIterableDataset(str(csv_path))
    return DataLoader(
        ds,
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
    )


class PerLabelFeatureBuffer:
    """
    每类最多保留 max_points_per_label 个点，避免单个测试集太大导致 UMAP 跑不动。
    - max_points_per_label <= 0：保留全部点；
    - max_points_per_label > 0：对每个 label 做 reservoir sampling。
    """

    def __init__(self, max_points_per_label: int, seed: int):
        self.max_points_per_label = int(max_points_per_label)
        self.rng = np.random.default_rng(seed)
        self.features = {i: [] for i in PLOT_ORDER}
        self.labels = {i: [] for i in PLOT_ORDER}
        self.seen = {i: 0 for i in PLOT_ORDER}

    def add(self, feat_np: np.ndarray, label_np: np.ndarray):
        for label in PLOT_ORDER:
            idx = np.where(label_np == label)[0]
            if idx.size == 0:
                continue

            for row_idx in idx:
                self.seen[label] += 1
                if self.max_points_per_label <= 0:
                    self.features[label].append(feat_np[row_idx])
                    self.labels[label].append(label)
                else:
                    current_len = len(self.features[label])
                    if current_len < self.max_points_per_label:
                        self.features[label].append(feat_np[row_idx])
                        self.labels[label].append(label)
                    else:
                        j = int(self.rng.integers(0, self.seen[label]))
                        if j < self.max_points_per_label:
                            self.features[label][j] = feat_np[row_idx]
                            self.labels[label][j] = label

    def concat(self) -> Tuple[np.ndarray, np.ndarray]:
        feat_parts = []
        label_parts = []
        for label in PLOT_ORDER:
            if len(self.features[label]) == 0:
                continue
            feat_parts.append(np.stack(self.features[label], axis=0).astype(np.float32, copy=False))
            label_parts.append(np.asarray(self.labels[label], dtype=np.int64))

        if not feat_parts:
            raise ValueError("没有任何可用于 UMAP 的特征。")

        features = np.concatenate(feat_parts, axis=0)
        labels = np.concatenate(label_parts, axis=0)
        return features, labels


@torch.no_grad()
def extract_features_for_umap(model: torch.nn.Module,
                              dataloader: DataLoader,
                              device: torch.device,
                              feature_source: str,
                              max_points_per_label: int,
                              seed: int) -> Tuple[np.ndarray, np.ndarray, Dict[int, int], Dict[int, int]]:
    print("提取模型中间层特征...")
    buffer = PerLabelFeatureBuffer(max_points_per_label=max_points_per_label, seed=seed)

    parsed_total = 0
    batch_count = 0

    for d_cpu, l_cpu in dataloader:
        if l_cpu.numel() == 0:
            continue

        batch_count += 1
        parsed_total += int(l_cpu.numel())
        if batch_count % 20 == 0:
            print(f"  已处理 batch={batch_count}, parsed_total={parsed_total}")

        d = tuple(t.to(device, non_blocking=True) for t in d_cpu)
        label_np = l_cpu.detach().cpu().numpy()

        _, feat = model(d)  # feat: [B, 192]

        if feature_source == "backbone":
            layer_features = feat
        elif feature_source == "classifier_hidden":
            # cls: Linear(192->128) + GELU + Dropout + Linear(128->6)
            layer_features = model.cls[1](model.cls[0](feat))
        else:
            raise ValueError("feature_source 只能是 backbone 或 classifier_hidden")

        feat_np = layer_features.detach().cpu().numpy().astype(np.float32, copy=False)
        buffer.add(feat_np, label_np)

    if parsed_total == 0:
        raise ValueError("没有解析到有效样本，请检查 CSV 是否为 train_all.py 对应的 11 列格式。")

    features, labels = buffer.concat()

    kept_counts = {i: int(np.sum(labels == i)) for i in PLOT_ORDER}
    seen_counts = {i: int(buffer.seen[i]) for i in PLOT_ORDER}

    print("特征提取完成:")
    print(f"  原始解析样本数: {parsed_total}")
    print(f"  UMAP使用样本数: {features.shape[0]}")
    for label in PLOT_ORDER:
        print(f"    label {label} ({LABEL_NAMES[label]}): seen={seen_counts[label]}, kept={kept_counts[label]}")

    return features, labels, seen_counts, kept_counts


def apply_umap(features: np.ndarray,
               n_neighbors: int,
               min_dist: float,
               spread: float,
               repulsion_strength: float,
               random_state: int) -> np.ndarray:
    n_samples = features.shape[0]
    if n_samples < 3:
        raise ValueError(f"样本数太少，无法稳定绘制 UMAP: n_samples={n_samples}")

    n_neighbors_eff = max(2, min(n_neighbors, n_samples - 1))
    if n_neighbors_eff != n_neighbors:
        print(f"  [WARN] 样本数较少，n_neighbors 从 {n_neighbors} 自动调整为 {n_neighbors_eff}")

    print("应用 UMAP 降维...")
    print(
        f"  n_samples={n_samples}, n_neighbors={n_neighbors_eff}, "
        f"min_dist={min_dist}, spread={spread}, repulsion_strength={repulsion_strength}"
    )

    scaler = StandardScaler()
    features_scaled = scaler.fit_transform(features)

    reducer = umap.UMAP(
        n_neighbors=n_neighbors_eff,
        min_dist=min_dist,
        n_components=2,
        spread=spread,
        repulsion_strength=repulsion_strength,
        random_state=random_state,
        verbose=False,
    )
    embedding = reducer.fit_transform(features_scaled)
    print(f"UMAP 完成: {embedding.shape}")
    return embedding


def plot_umap(embedding: np.ndarray,
              labels: np.ndarray,
              output_png: Path,
              output_pdf: Optional[Path],
              figsize: Tuple[float, float],
              point_size: float,
              alpha: float,
              fixed_ylim: Optional[Tuple[float, float]],
              add_title: bool,
              title: str):
    print("绘制 UMAP 聚类图...")
    output_png = Path(output_png)
    output_png.parent.mkdir(parents=True, exist_ok=True)

    fig, ax = plt.subplots(figsize=figsize)

    handle_map = {}
    name_map = {}

    for label in PLOT_ORDER:
        mask = labels == label
        if not np.any(mask):
            continue

        h = ax.scatter(
            embedding[mask, 0],
            embedding[mask, 1],
            s=point_size,
            alpha=alpha,
            color=LABEL_COLORS[label],
            edgecolors="none",
        )
        handle_map[label] = h
        name_map[label] = LABEL_NAMES[label]

    ax.set_xlabel("UMAP Dimension 1", fontsize=AXIS_LABEL_FONTSIZE)
    ax.set_ylabel("UMAP Dimension 2", fontsize=AXIS_LABEL_FONTSIZE)
    ax.tick_params(
        axis="both",
        which="major",
        labelsize=TICK_FONTSIZE,
        direction="out",
        length=6,
        width=3,
        colors="black",
    )

    for side in ["left", "right", "top", "bottom"]:
        ax.spines[side].set_linewidth(3)
        ax.spines[side].set_color("black")

    if fixed_ylim is not None:
        ax.set_ylim(fixed_ylim[0], fixed_ylim[1])

    if add_title:
        ax.set_title(title, fontsize=TITLE_FONTSIZE)

    ordered = [label for label in PLOT_ORDER if label in handle_map]
    handles = [handle_map[label] for label in ordered]
    legend_labels = [name_map[label] for label in ordered]

    leg = ax.legend(
        handles,
        legend_labels,
        loc="center left",
        bbox_to_anchor=(1.02, 0.5),
        ncol=1,
        fontsize=LEGEND_FONTSIZE,
        frameon=False,
        borderaxespad=0.0,
        handletextpad=0.4,
        labelspacing=0.6,
        title=LEGEND_TITLE,
    )

    if leg is not None and leg.get_title() is not None:
        leg.get_title().set_fontsize(LEGEND_TITLE_FONTSIZE)
        leg.get_title().set_multialignment("center")
        leg.get_title().set_ha("center")

    fig.tight_layout(rect=[0.0, 0.0, 0.80, 1.0])
    fig.savefig(output_png, dpi=600, bbox_inches="tight")
    print(f"PNG 已保存: {output_png}")

    if output_pdf is not None:
        output_pdf = Path(output_pdf)
        output_pdf.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(output_pdf, bbox_inches="tight")
        print(f"PDF 已保存: {output_pdf}")

    plt.close(fig)


def write_summary(summary_csv: Path,
                  csv_path: Path,
                  model_path: Path,
                  output_png: Path,
                  output_pdf: Optional[Path],
                  seen_counts: Dict[int, int],
                  kept_counts: Dict[int, int],
                  args):
    summary_csv = Path(summary_csv)
    summary_csv.parent.mkdir(parents=True, exist_ok=True)

    fields = [
        "csv_path",
        "model_path",
        "output_png",
        "output_pdf",
        "raw_signal_len",
        "dwell_divisor",
        "quality_divisor",
        "max_points_per_label",
        "feature_source",
        "n_neighbors",
        "min_dist",
        "spread",
        "repulsion_strength",
        "label",
        "label_name",
        "seen_count",
        "kept_count",
    ]

    with open(summary_csv, "w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fields)
        writer.writeheader()
        for label in PLOT_ORDER:
            writer.writerow({
                "csv_path": str(csv_path),
                "model_path": str(model_path),
                "output_png": str(output_png),
                "output_pdf": "" if output_pdf is None else str(output_pdf),
                "raw_signal_len": args.raw_signal_len,
                "dwell_divisor": args.dwell_divisor,
                "quality_divisor": args.quality_divisor,
                "max_points_per_label": args.max_points_per_label,
                "feature_source": args.feature_source,
                "n_neighbors": args.n_neighbors,
                "min_dist": args.min_dist,
                "spread": args.spread,
                "repulsion_strength": args.repulsion_strength,
                "label": label,
                "label_name": LABEL_NAMES[label],
                "seen_count": seen_counts.get(label, 0),
                "kept_count": kept_counts.get(label, 0),
            })

    print(f"summary 已保存: {summary_csv}")


def write_embedding_csv(path: Path, embedding: np.ndarray, labels: np.ndarray) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8", newline="") as handle:
        writer = csv.writer(handle)
        writer.writerow(["umap_1", "umap_2", "label", "label_name"])
        for point, label in zip(embedding, labels):
            writer.writerow([
                f"{float(point[0]):.8f}",
                f"{float(point[1]):.8f}",
                int(label),
                LABEL_NAMES[int(label)],
            ])


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Plot Figure 3D UMAP for the six-class LiteMamba model."
    )
    parser.add_argument("--csv_path", required=True, help="Six-class test CSV in the 11-column format.")
    parser.add_argument("--model_path", required=True, help="LiteMamba six-class checkpoint.")
    parser.add_argument("--output_dir", required=True, help="Output directory.")
    parser.add_argument("--output_name", default="Figure3D_LiteMamba_UMAP")
    parser.add_argument("--device", default="auto", help="auto, cpu, cuda:0, ...")
    parser.add_argument("--batch_size", type=int, default=1024)
    parser.add_argument("--workers", type=int, default=0)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--umap_seed", type=int, default=42)
    parser.add_argument("--mer_len", type=int, default=6)
    parser.add_argument("--raw_signal_len", type=int, default=30)
    parser.add_argument("--dwell_divisor", type=float, default=60.0)
    parser.add_argument("--quality_divisor", type=float, default=50.0)
    parser.add_argument(
        "--feature_source",
        choices=["classifier_hidden", "backbone"],
        default="classifier_hidden",
    )
    parser.add_argument("--max_points_per_label", type=int, default=5000)
    parser.add_argument("--n_neighbors", type=int, default=10)
    parser.add_argument("--min_dist", type=float, default=0.3)
    parser.add_argument("--spread", type=float, default=1.5)
    parser.add_argument("--repulsion_strength", type=float, default=1.5)
    parser.add_argument("--fig_width", type=float, default=14.0)
    parser.add_argument("--fig_height", type=float, default=8.0)
    parser.add_argument("--point_size", type=float, default=50.0)
    parser.add_argument("--alpha", type=float, default=0.50)
    parser.add_argument("--no_fixed_ylim", action="store_true")
    parser.add_argument("--ymin", type=float, default=-10.0)
    parser.add_argument("--ymax", type=float, default=20.0)
    parser.add_argument("--add_title", action="store_true")
    parser.add_argument("--title", default="LiteMamba six-class UMAP")
    parser.add_argument("--no_pdf", action="store_true")
    args = parser.parse_args()

    if args.batch_size <= 0 or args.raw_signal_len <= 0 or args.mer_len <= 0:
        parser.error("batch_size, raw_signal_len and mer_len must be greater than zero")
    if args.dwell_divisor <= 0 or args.quality_divisor <= 0:
        parser.error("dwell_divisor and quality_divisor must be greater than zero")
    if args.max_points_per_label < 0:
        parser.error("max_points_per_label must be non-negative")
    if args.n_neighbors < 2:
        parser.error("n_neighbors must be at least 2")
    if not (0.0 <= args.alpha <= 1.0):
        parser.error("alpha must be between 0 and 1")
    return args


def main() -> None:
    args = parse_args()
    set_seed(args.seed)
    device = get_device(args.device)
    csv_path = Path(args.csv_path)
    model_path = Path(args.model_path)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    if not csv_path.exists():
        raise FileNotFoundError(f"Test CSV not found: {csv_path}")
    if not model_path.exists():
        raise FileNotFoundError(f"Checkpoint not found: {model_path}")

    output_png = output_dir / f"{args.output_name}.png"
    output_pdf = None if args.no_pdf else output_dir / f"{args.output_name}.pdf"
    summary_csv = output_dir / f"{args.output_name}_summary.csv"
    embedding_csv = output_dir / f"{args.output_name}_embedding.csv"

    start = time.time()
    model = load_litemamba_model(
        model_path=model_path,
        device=device,
        raw_signal_len=args.raw_signal_len,
        num_classes=6,
        seq_len=args.mer_len,
    )
    dataloader = get_loader(
        csv_path=csv_path,
        batch_size=args.batch_size,
        num_workers=args.workers,
        mer_len=args.mer_len,
        raw_signal_len=args.raw_signal_len,
        dwell_divisor=args.dwell_divisor,
        quality_divisor=args.quality_divisor,
    )
    features, labels, seen_counts, kept_counts = extract_features_for_umap(
        model=model,
        dataloader=dataloader,
        device=device,
        feature_source=args.feature_source,
        max_points_per_label=args.max_points_per_label,
        seed=args.seed,
    )
    embedding = apply_umap(
        features=features,
        n_neighbors=args.n_neighbors,
        min_dist=args.min_dist,
        spread=args.spread,
        repulsion_strength=args.repulsion_strength,
        random_state=args.umap_seed,
    )
    write_embedding_csv(embedding_csv, embedding, labels)

    fixed_ylim = None if args.no_fixed_ylim else (args.ymin, args.ymax)
    plot_umap(
        embedding=embedding,
        labels=labels,
        output_png=output_png,
        output_pdf=output_pdf,
        figsize=(args.fig_width, args.fig_height),
        point_size=args.point_size,
        alpha=args.alpha,
        fixed_ylim=fixed_ylim,
        add_title=args.add_title,
        title=args.title,
    )
    write_summary(
        summary_csv=summary_csv,
        csv_path=csv_path,
        model_path=model_path,
        output_png=output_png,
        output_pdf=output_pdf,
        seen_counts=seen_counts,
        kept_counts=kept_counts,
        args=args,
    )

    del model
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()

    print("=" * 90)
    print("Figure 3D UMAP completed")
    print(f"Elapsed seconds : {time.time() - start:.1f}")
    print(f"Figure          : {output_png}")
    print(f"Embedding CSV   : {embedding_csv}")
    print(f"Summary CSV     : {summary_csv}")
    print("=" * 90)


if __name__ == "__main__":
    main()
