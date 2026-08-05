#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
evaluate_external_templates.py

功能
----
统一评估 Figure 4 使用的全部外部测试集，并导出可复现的指标与绘图源数据：

1) XNA1–12 / DNA1–12
   - 6-class: ATCGXY_6class
   - 5X:      ATCGX_5class
   - 5Y:      ATCGY_5class

2) XNA13–16 / DNA13–16
   - 6-class
   - 5X
   - 5Y

3) XNA17–20（以及若存在则对应 DNA/PC17–20）
   - 6-class
   - 5X
   - 5Y

输出
----
1. raw_evaluation/
   各数据集 × 各模型的逐文件预测与 summary_metrics.csv
2. analysis_data/
   后续重新绘图所需的全部 CSV，无需再次跑模型

说明
----
- 本代码尽量兼容你之前几版脚本中的不同文件命名方式。
- XNA17–20 会自动将 84Ds4-AA/AB/AC/AD 映射到 XNA17/18/19/20。
- Figure 4B 会自动汇总所有已发现的 DNA/PC 对照模板；如果 17–20 没有 control，只会使用 1–16。
- 本公开版只负责模型推理、指标汇总和 Figure 4 绘图源数据导出。
- 最终 Figure 4 绘图脚本单独放入 scripts/08_figures，避免评价逻辑与版式调整耦合。
"""

from __future__ import annotations

import argparse
import csv
import gzip
import hashlib
import json
import math
import os
import platform
import random
import re
import sys
import time
from contextlib import nullcontext
from dataclasses import dataclass, asdict
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Dataset

TRAINING_DIR = Path(__file__).resolve().parents[1] / "06_training"
if str(TRAINING_DIR) not in sys.path:
    sys.path.insert(0, str(TRAINING_DIR))

from model_benchmark_all import build_model


# =============================================================================
# Model/task constants
# =============================================================================

MODEL_ORDER = ["litemamba", "transformer", "bigru", "convmixer"]
MODEL_DISPLAY = {
    "litemamba": "LiteMamba",
    "transformer": "Transformer",
    "bigru": "BiGRU",
    "convmixer": "ConvMixer",
}
MODEL_COLORS = {
    "litemamba": "#1B9E77",
    "transformer": "#D95F02",
    "bigru": "#7570B3",
    "convmixer": "#666666",
}
TASK_ORDER = ["6-class", "5X", "5Y"]
TASK_LABELS = {"6-class": "ATCGXY 6-Class", "5X": "ATCGX 5-Class", "5Y": "ATCGY 5-Class"}

BASE_MAP = {"A": 0, "C": 1, "T": 2, "G": 3, "N": 4}
BASE_DIM = 5
MERGED_FEATURE_DIM = 13
CLASS_NAMES_6 = ["A", "T", "C", "G", "X", "Y"]
CLASS_NAMES_5X = ["A", "T", "C", "G", "X"]
CLASS_NAMES_5Y = ["A", "T", "C", "G", "Y"]

REFERENCE_ALIAS_TO_TEMPLATE = {
    # XNA1-12 / DNA1-12 (PC)
    **{f"XNA{i:02d}": f"XNA{i}" for i in range(1, 13)},
    **{f"PC{i:02d}": f"XNA{i}" for i in range(1, 13)},
    # XNA13-16 / DNA13-16 (PC)
    **{f"XNA{i}": f"XNA{i}" for i in range(13, 17)},
    **{f"PC{i}": f"XNA{i}" for i in range(13, 17)},
    # 17-20 aliases
    "84DS4-AA": "XNA17",
    "84DS4-AB": "XNA18",
    "84DS4-AC": "XNA19",
    "84DS4-AD": "XNA20",
    **{f"XNA{i}": f"XNA{i}" for i in range(17, 21)},
    **{f"PC{i}": f"XNA{i}" for i in range(17, 21)},
    **{f"DNA{i}": f"XNA{i}" for i in range(17, 21)},
}

TARGET_REFERENCE_ALIASES = {"84DS4-AA", "84DS4-AB", "84DS4-AC", "84DS4-AD"}
TARGET_REFERENCE_ALIASES.update({f"XNA{i:02d}" for i in range(1, 13)})
TARGET_REFERENCE_ALIASES.update({f"XNA{i}" for i in range(13, 21)})


@dataclass
class DatasetSpec:
    key: str
    cohort: str
    task: str
    test_dir: str
    filename_style: str
    num_classes: int
    model_files: Dict[str, str]
    expected_templates: List[str]
    group_label: str


def load_model_manifest(path: Path) -> Dict[str, Dict[str, str]]:
    """Load checkpoint filenames/paths for all tasks and model types.

    Expected JSON structure::

        {
          "6-class": {"bigru": "...pt", "transformer": "...pt",
                      "convmixer": "...pt", "litemamba": "...pt"},
          "5X": {...},
          "5Y": {...}
        }
    """
    path = Path(path)
    if not path.is_file():
        raise FileNotFoundError(f"Model manifest does not exist: {path}")
    with open(path, "r", encoding="utf-8") as handle:
        data = json.load(handle)

    required_tasks = {"6-class", "5X", "5Y"}
    missing_tasks = required_tasks - set(data)
    if missing_tasks:
        raise ValueError(f"Model manifest missing tasks: {sorted(missing_tasks)}")

    normalized: Dict[str, Dict[str, str]] = {}
    for task in sorted(required_tasks):
        task_map = data[task]
        if not isinstance(task_map, dict):
            raise TypeError(f"Manifest entry for {task} must be an object")
        missing_models = set(MODEL_ORDER) - set(task_map)
        if missing_models:
            raise ValueError(
                f"Model manifest task {task} missing models: {sorted(missing_models)}"
            )
        normalized[task] = {model: str(task_map[model]) for model in MODEL_ORDER}
    return normalized


# =============================================================================
# 通用工具
# =============================================================================

def ensure_dir(path: Path) -> Path:
    path.mkdir(parents=True, exist_ok=True)
    return path


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def set_csv_field_size_limit() -> None:
    max_int = sys.maxsize
    while True:
        try:
            csv.field_size_limit(max_int)
            return
        except OverflowError:
            max_int //= 10


def safe_div(num: float, den: float) -> float:
    return float(num / den) if den > 0 else float("nan")


def natural_key(path_or_name) -> List[object]:
    s = str(path_or_name)
    return [int(x) if x.isdigit() else x.lower() for x in re.split(r"(\d+)", s)]


def normalize_ref_name(ref_name: str) -> str:
    return str(ref_name).strip().upper()


def template_number(template: str) -> int:
    m = re.search(r"(\d+)$", str(template))
    return int(m.group(1)) if m else 10 ** 9


def infer_ref_metadata(ref_name: str) -> Tuple[str, str]:
    ref_key = normalize_ref_name(ref_name)
    if ref_key in REFERENCE_ALIAS_TO_TEMPLATE:
        template = REFERENCE_ALIAS_TO_TEMPLATE[ref_key]
        role = "target" if ref_key in TARGET_REFERENCE_ALIASES or ref_key.startswith("XNA") or ref_key.startswith("84DS4") else "control"
        return template, role

    m = re.search(r"(\d+)$", ref_key)
    if not m:
        return "UNKNOWN", "unknown"
    num = int(m.group(1))
    template = f"XNA{num}"
    role = "target" if ref_key.startswith("XNA") else "control"
    return template, role


def strand_description(strand: str) -> str:
    if strand == "pos":
        return "positive/X/5X"
    if strand == "neg":
        return "negative/Y/5Y"
    return "unknown"


def class_names_for_task(task: str) -> List[str]:
    if task == "6-class":
        return CLASS_NAMES_6
    if task == "5X":
        return CLASS_NAMES_5X
    return CLASS_NAMES_5Y


def target_labels_for(task: str, strand: Optional[str], args: argparse.Namespace) -> List[int]:
    if task == "6-class":
        if strand == "pos":
            return [args.six_x_label]
        if strand == "neg":
            return [args.six_y_label]
        return [args.six_x_label, args.six_y_label]
    return [args.five_target_label]


def make_autocast(device: str, enabled: bool):
    if not enabled or not device.startswith("cuda"):
        return nullcontext()
    try:
        return torch.amp.autocast(device_type="cuda", enabled=True)
    except Exception:
        return torch.cuda.amp.autocast(enabled=True)


def extract_logits(model_output):
    if isinstance(model_output, (tuple, list)):
        return model_output[0]
    if isinstance(model_output, dict):
        for key in ("logits", "output", "pred"):
            if key in model_output:
                return model_output[key]
    return model_output


def save_json(data, path: Path) -> None:
    ensure_dir(path.parent)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(data, f, indent=2, ensure_ascii=False, default=str)


def open_text_output(path: Path, compression: str):
    ensure_dir(path.parent)
    if compression == "gzip":
        gz_path = path if path.suffix == ".gz" else path.with_suffix(path.suffix + ".gz")
        return gzip.open(gz_path, "wt", newline="", encoding="utf-8"), gz_path
    return open(path, "w", newline="", encoding="utf-8", buffering=1024 * 1024 * 8), path


def stable_hash(parts: Sequence[str]) -> str:
    payload = "\x1f".join(parts).encode("utf-8", errors="replace")
    return hashlib.blake2b(payload, digest_size=12).hexdigest()


# =============================================================================
# 数据集规格
# =============================================================================

def build_specs(args: argparse.Namespace, model_manifest: Dict[str, Dict[str, str]]) -> List[DatasetSpec]:
    root_01_12 = Path(args.dir_01_12_root)
    return [
        DatasetSpec("xna01_12_6class", "XNA1-12", "6-class", str(root_01_12 / "ATCGXY_6class"), "01_12", 6, model_manifest["6-class"], [f"XNA{i}" for i in range(1, 13)], "Single-NCB templates"),
        DatasetSpec("xna01_12_5X", "XNA1-12", "5X", str(root_01_12 / "ATCGX_5class"), "01_12", 5, model_manifest["5X"], [f"XNA{i}" for i in range(1, 13)], "Single-NCB templates"),
        DatasetSpec("xna01_12_5Y", "XNA1-12", "5Y", str(root_01_12 / "ATCGY_5class"), "01_12", 5, model_manifest["5Y"], [f"XNA{i}" for i in range(1, 13)], "Single-NCB templates"),

        DatasetSpec("xna13_16_6class", "XNA13-16", "6-class", args.dir_13_16_6, "13_16", 6, model_manifest["6-class"], [f"XNA{i}" for i in range(13, 17)], "Multiple-NCB templates"),
        DatasetSpec("xna13_16_5X", "XNA13-16", "5X", args.dir_13_16_5x, "13_16", 5, model_manifest["5X"], [f"XNA{i}" for i in range(13, 17)], "Multiple-NCB templates"),
        DatasetSpec("xna13_16_5Y", "XNA13-16", "5Y", args.dir_13_16_5y, "13_16", 5, model_manifest["5Y"], [f"XNA{i}" for i in range(13, 17)], "Multiple-NCB templates"),

        DatasetSpec("xna17_20_6class", "XNA17-20", "6-class", args.dir_17_20_6, "17_20", 6, model_manifest["6-class"], [f"XNA{i}" for i in range(17, 21)], "Four equidistant NCB templates"),
        DatasetSpec("xna17_20_5X", "XNA17-20", "5X", args.dir_17_20_5x, "17_20", 5, model_manifest["5X"], [f"XNA{i}" for i in range(17, 21)], "Four equidistant NCB templates"),
        DatasetSpec("xna17_20_5Y", "XNA17-20", "5Y", args.dir_17_20_5y, "17_20", 5, model_manifest["5Y"], [f"XNA{i}" for i in range(17, 21)], "Four equidistant NCB templates"),
    ]


# =============================================================================
# 文件名解析与发现
# =============================================================================

def parse_split_filename(csv_path: Path, style: str) -> Tuple[str, str]:
    stem = csv_path.stem

    if style == "01_12":
        m = re.match(r"^.+__(?P<ref>[^_]+)_(?P<strand>pos|neg)$", stem, flags=re.IGNORECASE)
        if m:
            return m.group("strand").lower(), m.group("ref")
        m = re.match(r"^.+_(?P<strand>pos|neg)_(?P<ref>[^_]+)$", stem, flags=re.IGNORECASE)
        if m:
            return m.group("strand").lower(), m.group("ref")

    elif style == "13_16":
        m = re.match(r"^.+_(?P<strand>pos|neg)_(?P<ref>[^_]+)$", stem, flags=re.IGNORECASE)
        if m:
            return m.group("strand").lower(), m.group("ref")

    elif style == "17_20":
        m = re.match(r"^xna_6mer_features_17_20_(?P<ref>.+)_(?P<strand>pos|neg)$", stem, flags=re.IGNORECASE)
        if m:
            return m.group("strand").lower(), m.group("ref")
        m = re.match(r"^.+_(?P<ref>.+)_(?P<strand>pos|neg)$", stem, flags=re.IGNORECASE)
        if m:
            return m.group("strand").lower(), m.group("ref")

    m = re.match(r"^.+_(?P<ref>.+)_(?P<strand>pos|neg)$", stem, flags=re.IGNORECASE)
    if m:
        return m.group("strand").lower(), m.group("ref")

    m = re.match(r"^.+_(?P<strand>pos|neg)_(?P<ref>[^_]+)$", stem, flags=re.IGNORECASE)
    if m:
        return m.group("strand").lower(), m.group("ref")

    return "UNKNOWN", "UNKNOWN"


def is_target_test_csv(path: Path, spec: DatasetSpec) -> bool:
    if path.suffix.lower() != ".csv":
        return False

    lower_name = path.name.lower()
    bad_keywords = [
        "prediction", "predictions", "summary", "confusion", "classification", "metrics",
        "label_count", "combined", "file_summary", "overall", "analysis_data", "figure4", "eval_",
    ]
    if any(k in lower_name for k in bad_keywords):
        return False

    strand, ref_name = parse_split_filename(path, spec.filename_style)
    if strand not in ("pos", "neg"):
        return False
    template, role = infer_ref_metadata(ref_name)
    if template not in spec.expected_templates:
        return False
    if role not in ("target", "control"):
        return False

    return True


def find_test_files(spec: DatasetSpec) -> Tuple[List[Path], List[Dict[str, str]]]:
    root = Path(spec.test_dir)
    if not root.exists():
        raise FileNotFoundError(f"测试集目录不存在: {root}")

    all_csv = sorted(root.glob("*.csv"), key=lambda p: natural_key(p.name))
    files = [p for p in all_csv if is_target_test_csv(p, spec)]

    diagnostics: List[Dict[str, str]] = []
    for p in all_csv:
        strand, ref_name = parse_split_filename(p, spec.filename_style)
        template, role = infer_ref_metadata(ref_name)
        diagnostics.append({
            "dataset_key": spec.key,
            "cohort": spec.cohort,
            "task": spec.task,
            "source_file": p.name,
            "parsed_strand": strand,
            "parsed_reference": ref_name,
            "resolved_template": template,
            "reference_role": role,
            "accepted": str(bool(is_target_test_csv(p, spec))),
        })

    if not files:
        print(f"\n[ERROR] 未找到符合规则的 CSV: {root}", file=sys.stderr)
        print("目录解析结果：", file=sys.stderr)
        for row in diagnostics:
            print(f"  {row['source_file']} -> strand={row['parsed_strand']} ref={row['parsed_reference']} template={row['resolved_template']} role={row['reference_role']} accepted={row['accepted']}", file=sys.stderr)
        raise RuntimeError(f"没有找到符合规则的测试 CSV: {root}")

    print(f"[INFO] {spec.key}: 找到 {len(files)} 个测试文件")
    for p in files:
        strand, ref_name = parse_split_filename(p, spec.filename_style)
        template, role = infer_ref_metadata(ref_name)
        print(f"       {p.name} | ref={ref_name} -> {template} | strand={strand} ({strand_description(strand)}) | role={role}")

    return files, diagnostics


def write_reference_mapping(path: Path) -> None:
    rows = []
    for ref_name in ["84Ds4-AA", "84Ds4-AB", "84Ds4-AC", "84Ds4-AD"]:
        template, role = infer_ref_metadata(ref_name)
        for strand in ("pos", "neg"):
            rows.append({
                "reference_name": ref_name,
                "template": template,
                "reference_role": role,
                "strand_suffix": strand,
                "strand_interpretation": "positive strand / X / 5X" if strand == "pos" else "negative strand / Y / 5Y",
                "example_filename": f"xna_6mer_features_17_20_{ref_name}_{strand}.csv",
            })
    pd.DataFrame(rows).to_csv(path, index=False)


# =============================================================================
# 数据解析与 DataLoader
# =============================================================================

def encode_seq_block(seq: str, mer_len: int) -> Optional[np.ndarray]:
    seq = seq.strip().upper()
    if len(seq) != mer_len:
        return None
    out = np.zeros((mer_len, BASE_DIM), dtype=np.float32)
    for i, ch in enumerate(seq):
        idx = BASE_MAP.get(ch)
        if idx is None:
            return None
        out[i, idx] = 1.0
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
            pad = signals_len - arr.size
            left = pad // 2
            right = pad - left
            arr = np.pad(arr, (left, right), mode="constant", constant_values=0.0)
        elif arr.size > signals_len:
            idx = sorted(random.sample(range(arr.size), signals_len))
            arr = arr[idx]
        signals_rect[i] = arr.astype(np.float32, copy=False)
    return signals_rect


def build_merged_features(seq_onehot_np, mean_np, std_np, median_np, dwell_np, quality_np, mismatch_np, insertion_np, deletion_np, dwell_divisor: float, quality_divisor: float):
    dwell_np = (dwell_np / dwell_divisor).astype(np.float32, copy=False)
    quality_np = (quality_np / quality_divisor).astype(np.float32, copy=False)
    scalar_part = np.stack([mean_np, std_np, median_np, dwell_np, quality_np, mismatch_np, insertion_np, deletion_np], axis=1).astype(np.float32, copy=False)
    return np.concatenate([seq_onehot_np.astype(np.float32, copy=False), scalar_part], axis=1).astype(np.float32, copy=False)


def parse_one_csv_row(row: Sequence[str], row_id: int, mer_len: int, raw_signal_len_per_base: int, dwell_divisor: float, quality_divisor: float):
    if len(row) != 11:
        return None
    if row[0].strip().lower() == "kmer" or row[-1].strip().lower() == "label":
        return None

    seq_s, mean_s, std_s, median_s, dwell_s, quality_s, mismatch_s, insertion_s, deletion_s, signal_s, label_s = row
    seq_onehot_np = encode_seq_block(seq_s, mer_len)
    if seq_onehot_np is None:
        return None

    arrays = [
        fast_array_from_pipe(mean_s, mer_len),
        fast_array_from_pipe(std_s, mer_len),
        fast_array_from_pipe(median_s, mer_len),
        fast_array_from_pipe(dwell_s, mer_len),
        fast_array_from_pipe(quality_s, mer_len),
        fast_array_from_pipe(mismatch_s, mer_len),
        fast_array_from_pipe(insertion_s, mer_len),
        fast_array_from_pipe(deletion_s, mer_len),
    ]
    raw_rect_np = get_signals_rect_from_str(signal_s, mer_len, raw_signal_len_per_base)
    if any(x is None for x in arrays) or raw_rect_np is None:
        return None

    try:
        label = int(float(label_s))
    except Exception:
        return None

    merged_feat = build_merged_features(
        seq_onehot_np,
        arrays[0], arrays[1], arrays[2], arrays[3], arrays[4], arrays[5], arrays[6], arrays[7],
        dwell_divisor=dwell_divisor,
        quality_divisor=quality_divisor,
    )
    rec = {"row_id": row_id, "kmer": seq_s, "sample_hash": stable_hash(row[:-1])}
    return rec, (seq_onehot_np, merged_feat, raw_rect_np), label


class EvalCSVDataset(Dataset):
    def __init__(self, csv_path: str, mer_len: int, raw_signal_len_per_base: int, dwell_divisor: float, quality_divisor: float):
        self.csv_path = str(csv_path)
        self.rows = []
        self.total_lines = 0
        self.skipped_lines = 0
        with open(self.csv_path, "r", encoding="utf-8", newline="", buffering=1024 * 1024 * 16) as f:
            reader = csv.reader(f)
            for line_id, row in enumerate(reader):
                self.total_lines += 1
                if line_id == 0 and row and row[0].strip().lower() == "kmer":
                    continue
                parsed = parse_one_csv_row(row, line_id, mer_len, raw_signal_len_per_base, dwell_divisor, quality_divisor)
                if parsed is None:
                    self.skipped_lines += 1
                    continue
                self.rows.append(parsed)

    def __len__(self) -> int:
        return len(self.rows)

    def __getitem__(self, idx: int):
        return self.rows[idx]


def make_collate_eval_batch(mer_len: int, raw_signal_len: int):
    def collate_eval_batch(batch):
        batch = [item for item in batch if item is not None]
        if not batch:
            empty_x = (
                torch.empty(0, mer_len, BASE_DIM, dtype=torch.float32),
                torch.empty(0, mer_len, MERGED_FEATURE_DIM, dtype=torch.float32),
                torch.empty(0, mer_len, raw_signal_len, dtype=torch.float32),
            )
            return [], empty_x, torch.empty(0, dtype=torch.long)

        records, seq_list, merged_list, raw_list, labels = [], [], [], [], []
        for rec, x, label in batch:
            seq_np, merged_np, raw_np = x
            records.append(rec)
            seq_list.append(seq_np)
            merged_list.append(merged_np)
            raw_list.append(raw_np)
            labels.append(label)

        x_tensor = (
            torch.from_numpy(np.stack(seq_list).astype(np.float32, copy=False)),
            torch.from_numpy(np.stack(merged_list).astype(np.float32, copy=False)),
            torch.from_numpy(np.stack(raw_list).astype(np.float32, copy=False)),
        )
        y_tensor = torch.from_numpy(np.asarray(labels, dtype=np.int64))
        return records, x_tensor, y_tensor
    return collate_eval_batch


# =============================================================================
# 模型加载与指标
# =============================================================================

def strip_module_prefix(state_dict: Dict[str, torch.Tensor]) -> Dict[str, torch.Tensor]:
    new_sd = {}
    for key, value in state_dict.items():
        new_key = key
        if new_key.startswith("module."):
            new_key = new_key[len("module."):]
        if new_key.startswith("_orig_mod."):
            new_key = new_key[len("_orig_mod."):]
        new_sd[new_key] = value
    return new_sd


def extract_state_dict(obj):
    if isinstance(obj, dict):
        for key in ("state_dict", "model_state_dict", "model"):
            if key in obj and isinstance(obj[key], dict):
                return obj[key]
    return obj


def infer_num_classes_from_state_dict(state_dict: Dict[str, torch.Tensor]) -> int:
    for key, value in state_dict.items():
        if torch.is_tensor(value) and value.ndim == 2 and key.endswith("cls.3.weight"):
            return int(value.shape[0])
    candidates = []
    for key, value in state_dict.items():
        if torch.is_tensor(value) and value.ndim == 2 and 2 <= int(value.shape[0]) <= 20:
            candidates.append((key, int(value.shape[0])))
    if candidates:
        return candidates[-1][1]
    raise RuntimeError("无法从 checkpoint 推断 num_classes")


def load_model_from_checkpoint(model_type: str, ckpt_path: Path, device: str, mer_len: int, raw_signal_len: int, expected_num_classes: int):
    if not ckpt_path.exists():
        raise FileNotFoundError(f"模型文件不存在: {ckpt_path}")
    obj = torch.load(str(ckpt_path), map_location=device)
    state_dict = strip_module_prefix(extract_state_dict(obj))
    inferred = infer_num_classes_from_state_dict(state_dict)
    if inferred != expected_num_classes:
        raise RuntimeError(f"checkpoint 类别数不匹配: {ckpt_path.name}, 推断={inferred}, 任务要求={expected_num_classes}")
    model = build_model(model_type=model_type, num_classes=expected_num_classes, seq_len=mer_len, raw_signal_len=raw_signal_len).to(device)
    model.load_state_dict(state_dict, strict=True)
    model.eval()
    return model


def init_confusion_matrix(num_classes: int) -> np.ndarray:
    return np.zeros((num_classes, num_classes), dtype=np.int64)


def update_confusion_matrix(cm: np.ndarray, y_true: np.ndarray, y_pred: np.ndarray) -> None:
    valid = (y_true >= 0) & (y_true < cm.shape[0]) & (y_pred >= 0) & (y_pred < cm.shape[1])
    if valid.any():
        np.add.at(cm, (y_true[valid].astype(int), y_pred[valid].astype(int)), 1)


def wilson_interval(success: int, total: int) -> Tuple[float, float]:
    if total <= 0:
        return float("nan"), float("nan")
    z = 1.959963984540054
    p = success / total
    denom = 1 + z * z / total
    center = (p + z * z / (2 * total)) / denom
    half = z * math.sqrt(p * (1 - p) / total + z * z / (4 * total * total)) / denom
    return max(0.0, center - half), min(1.0, center + half)


def compute_metrics_from_cm(cm: np.ndarray, target_labels: Sequence[int]) -> Tuple[Dict[str, float], List[Dict[str, float]]]:
    cm = np.asarray(cm, dtype=np.int64)
    total = int(cm.sum())
    diag = np.diag(cm).astype(float)
    supports = cm.sum(axis=1).astype(float)
    pred_counts = cm.sum(axis=0).astype(float)

    recalls = np.divide(diag, supports, out=np.zeros_like(diag), where=supports > 0)
    precisions = np.divide(diag, pred_counts, out=np.zeros_like(diag), where=pred_counts > 0)
    f1s = np.divide(2 * precisions * recalls, precisions + recalls, out=np.zeros_like(diag), where=(precisions + recalls) > 0)
    recalls[supports <= 0] = np.nan
    supported = supports > 0

    accuracy = safe_div(diag.sum(), total)
    balanced_accuracy = float(np.nanmean(recalls[supported])) if supported.any() else float("nan")
    macro_precision = float(np.nanmean(precisions[supported])) if supported.any() else float("nan")
    macro_f1 = float(np.nanmean(f1s[supported])) if supported.any() else float("nan")
    weighted_f1 = safe_div(np.nansum(f1s[supported] * supports[supported]), np.nansum(supports[supported]))

    valid_targets = [lab for lab in target_labels if 0 <= lab < cm.shape[0] and supports[lab] > 0]
    if valid_targets:
        target_recall_macro = float(np.mean([recalls[lab] for lab in valid_targets]))
        target_precision_macro = float(np.mean([precisions[lab] for lab in valid_targets]))
        target_f1_macro = float(np.mean([f1s[lab] for lab in valid_targets]))
        target_tp = int(sum(cm[lab, lab] for lab in valid_targets))
        target_support = int(sum(cm[lab, :].sum() for lab in valid_targets))
        target_recall_micro = safe_div(target_tp, target_support)
        ci_low, ci_high = wilson_interval(target_tp, target_support)
    else:
        target_recall_macro = float("nan")
        target_precision_macro = float("nan")
        target_f1_macro = float("nan")
        target_tp = 0
        target_support = 0
        target_recall_micro = float("nan")
        ci_low = float("nan")
        ci_high = float("nan")

    metrics = {
        "total_samples": total,
        "correct_samples": int(diag.sum()),
        "accuracy": accuracy,
        "balanced_accuracy": balanced_accuracy,
        "macro_precision": macro_precision,
        "macro_recall": balanced_accuracy,
        "macro_f1": macro_f1,
        "weighted_f1": weighted_f1,
        "target_recall": target_recall_macro,
        "target_recall_macro": target_recall_macro,
        "target_recall_micro": target_recall_micro,
        "target_precision": target_precision_macro,
        "target_f1": target_f1_macro,
        "target_tp": target_tp,
        "target_support": target_support,
        "target_recall_micro_ci_low": ci_low,
        "target_recall_micro_ci_high": ci_high,
    }

    class_rows = []
    for lab in range(cm.shape[0]):
        tp = int(cm[lab, lab])
        support = int(cm[lab, :].sum())
        fp = int(cm[:, lab].sum() - tp)
        fn = int(cm[lab, :].sum() - tp)
        tn = int(total - tp - fp - fn)
        rec_ci_low, rec_ci_high = wilson_interval(tp, support)
        class_rows.append({
            "label": lab,
            "support": support,
            "tp": tp,
            "fp": fp,
            "fn": fn,
            "tn": tn,
            "precision": precisions[lab],
            "recall": recalls[lab],
            "recall_ci_low": rec_ci_low,
            "recall_ci_high": rec_ci_high,
            "f1": f1s[lab],
            "specificity": safe_div(tn, tn + fp),
        })
    return metrics, class_rows


def compute_false_noncanonical_metrics(cm: np.ndarray, task: str, args: argparse.Namespace) -> Dict[str, float]:
    total = int(np.asarray(cm).sum())
    if total <= 0:
        return {
            "false_noncanonical_call_rate": float("nan"),
            "false_x_call_rate": float("nan"),
            "false_y_call_rate": float("nan"),
            "canonical_accuracy": float("nan"),
        }
    if task == "6-class":
        false_x = float(cm[:, args.six_x_label].sum() / total)
        false_y = float(cm[:, args.six_y_label].sum() / total)
        false_nc = false_x + false_y
    elif task == "5X":
        false_x = float(cm[:, args.five_target_label].sum() / total)
        false_y = float("nan")
        false_nc = false_x
    else:
        false_x = float("nan")
        false_y = float(cm[:, args.five_target_label].sum() / total)
        false_nc = false_y
    return {
        "false_noncanonical_call_rate": false_nc,
        "false_x_call_rate": false_x,
        "false_y_call_rate": false_y,
        "canonical_accuracy": float(np.trace(cm) / total),
    }


def write_confusion_matrix_csv(cm: np.ndarray, path: Path) -> None:
    ensure_dir(path.parent)
    with open(path, "w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(["true\\pred"] + [f"pred_{i}" for i in range(cm.shape[1])] + ["support"])
        for i in range(cm.shape[0]):
            writer.writerow([f"true_{i}"] + cm[i].tolist() + [int(cm[i].sum())])


def aggregate_cms(file_cm_records: List[Dict[str, object]], group_keys: Sequence[str]) -> Dict[Tuple, np.ndarray]:
    grouped: Dict[Tuple, np.ndarray] = {}
    for rec in file_cm_records:
        key = tuple(rec[k] for k in group_keys)
        cm = rec["cm"]
        if key not in grouped:
            grouped[key] = np.zeros_like(cm)
        grouped[key] += cm
    return grouped


# =============================================================================
# 评估单文件
# =============================================================================

def time_forward(model, x, device: str, use_amp: bool, should_time: bool):
    if device.startswith("cuda") and should_time:
        start_event = torch.cuda.Event(enable_timing=True)
        end_event = torch.cuda.Event(enable_timing=True)
        start_event.record()
        with make_autocast(device, use_amp):
            output = model(x)
        end_event.record()
        torch.cuda.synchronize(device)
        elapsed = start_event.elapsed_time(end_event) / 1000.0
        return extract_logits(output), elapsed
    if should_time:
        start = time.perf_counter()
        with make_autocast(device, use_amp):
            output = model(x)
        if device.startswith("cuda"):
            torch.cuda.synchronize(device)
        return extract_logits(output), time.perf_counter() - start
    with make_autocast(device, use_amp):
        output = model(x)
    return extract_logits(output), 0.0


def evaluate_one_file(model, model_type: str, csv_path: Path, spec: DatasetSpec, out_dir: Path, args: argparse.Namespace, benchmark_state: Dict[str, float]) -> Tuple[Dict[str, object], np.ndarray]:
    strand, ref_name = parse_split_filename(csv_path, spec.filename_style)
    template, ref_role = infer_ref_metadata(ref_name)
    file_target_labels = target_labels_for(spec.task, strand, args)

    total_wall_start = time.perf_counter()
    parse_start = time.perf_counter()
    dataset = EvalCSVDataset(str(csv_path), args.mer_len, args.raw_signal_len, args.dwell_divisor, args.quality_divisor)
    parse_sec = time.perf_counter() - parse_start

    loader = DataLoader(
        dataset,
        batch_size=args.batch_size,
        shuffle=False,
        drop_last=False,
        num_workers=args.workers,
        pin_memory=args.device.startswith("cuda"),
        collate_fn=make_collate_eval_batch(args.mer_len, args.raw_signal_len),
    )

    cm = init_confusion_matrix(spec.num_classes)
    total = 0
    loss_sum = 0.0
    invalid_label_counter = {}
    label_counter = {}
    pred_counter = {}
    loss_fn = nn.CrossEntropyLoss(reduction="sum").to(args.device)

    pred_writer = None
    pred_fp = None
    pred_path = None
    if not args.no_predictions:
        pred_base = out_dir / f"predictions_{csv_path.stem}.csv"
        pred_fp, pred_path = open_text_output(pred_base, args.prediction_compression)
        pred_writer = csv.writer(pred_fp)
        pred_writer.writerow([
            "source_file", "cohort", "task", "model", "template", "strand", "reference_name", "reference_role",
            "row_id", "kmer", "sample_hash", "true_label", "pred_label", "correct",
            *[f"prob_{i}" for i in range(spec.num_classes)],
        ])

    try:
        with torch.no_grad():
            for records, x_cpu, y_cpu in loader:
                if y_cpu.numel() == 0:
                    continue

                valid_mask = (y_cpu >= 0) & (y_cpu < spec.num_classes)
                if not bool(valid_mask.all()):
                    bad_vals = y_cpu[~valid_mask].tolist()
                    for val in bad_vals:
                        invalid_label_counter[int(val)] = invalid_label_counter.get(int(val), 0) + 1
                    if int(valid_mask.sum()) == 0:
                        continue
                    keep_idx = valid_mask.nonzero(as_tuple=False).view(-1).tolist()
                    records = [records[i] for i in keep_idx]
                    x_cpu = tuple(t[valid_mask] for t in x_cpu)
                    y_cpu = y_cpu[valid_mask]

                x = tuple(t.to(args.device, non_blocking=True) for t in x_cpu)
                y = y_cpu.to(args.device, non_blocking=True)

                if benchmark_state["warmup_remaining"] > 0:
                    should_time = False
                    benchmark_state["warmup_remaining"] -= 1
                else:
                    should_time = True

                logits, forward_sec = time_forward(model, x, args.device, not args.no_amp, should_time)
                if should_time:
                    benchmark_state["pure_forward_sec"] += forward_sec
                    benchmark_state["timed_samples"] += int(y.numel())
                    benchmark_state["timed_batches"] += 1

                loss = loss_fn(logits, y)
                probs = torch.softmax(logits, dim=1)
                pred = torch.argmax(logits, dim=1)

                y_np = y.detach().cpu().numpy()
                pred_np = pred.detach().cpu().numpy()
                probs_np = probs.detach().cpu().numpy()

                total += int(y_np.size)
                loss_sum += float(loss.item())
                update_confusion_matrix(cm, y_np, pred_np)
                for v in y_np:
                    label_counter[int(v)] = label_counter.get(int(v), 0) + 1
                for v in pred_np:
                    pred_counter[int(v)] = pred_counter.get(int(v), 0) + 1

                if pred_writer is not None:
                    for rec, true_lab, pred_lab, prob_vec in zip(records, y_np, pred_np, probs_np):
                        pred_writer.writerow([
                            csv_path.name, spec.cohort, spec.task, model_type, template, strand, ref_name, ref_role,
                            rec["row_id"], rec["kmer"], rec["sample_hash"], int(true_lab), int(pred_lab), int(true_lab == pred_lab),
                            *[f"{float(v):.8f}" for v in prob_vec],
                        ])
    finally:
        if pred_fp is not None:
            pred_fp.close()

    total_wall_sec = time.perf_counter() - total_wall_start
    metrics, _ = compute_metrics_from_cm(cm, file_target_labels)
    result = {
        **metrics,
        "model": model_type,
        "model_display": MODEL_DISPLAY[model_type],
        "cohort": spec.cohort,
        "task": spec.task,
        "dataset_key": spec.key,
        "group_label": spec.group_label,
        "source_file": csv_path.name,
        "strand": strand,
        "strand_description": strand_description(strand),
        "reference_name": ref_name,
        "reference_role": ref_role,
        "template": template,
        "template_num": template_number(template),
        "total_lines": dataset.total_lines,
        "valid_samples": len(dataset),
        "skipped_lines": dataset.skipped_lines,
        "eval_samples": total,
        "loss": safe_div(loss_sum, total),
        "parse_sec": parse_sec,
        "total_wall_sec": total_wall_sec,
        "invalid_labels": ";".join(f"{k}:{v}" for k, v in sorted(invalid_label_counter.items())),
        "labels_true": ";".join(f"{k}:{v}" for k, v in sorted(label_counter.items())),
        "labels_pred": ";".join(f"{k}:{v}" for k, v in sorted(pred_counter.items())),
        "prediction_csv": str(pred_path) if pred_path is not None else "",
    }
    if ref_role == "control":
        result.update(compute_false_noncanonical_metrics(cm, spec.task, args))
    return result, cm


# =============================================================================
# 聚合与专用数据表
# =============================================================================

def cohort_to_group_label(cohort: str) -> str:
    if cohort == "XNA1-12":
        return "Single-NCB"
    if cohort == "XNA13-16":
        return "Multi-NCB"
    if cohort == "XNA17-20":
        return "4-NCB equal spacing"
    return cohort


def aggregate_role_metrics(file_cm_records: List[Dict[str, object]], args: argparse.Namespace, reference_role: str, level: str) -> pd.DataFrame:
    if level not in ("template", "template_strand"):
        raise ValueError("level must be template or template_strand")
    filtered = [rec for rec in file_cm_records if rec["reference_role"] == reference_role]
    group_keys = ["cohort", "task", "model", "template"] if level == "template" else ["cohort", "task", "model", "template", "strand"]
    grouped = aggregate_cms(filtered, group_keys)
    rows = []
    for key, cm in grouped.items():
        meta = dict(zip(group_keys, key))
        target_labels = target_labels_for(meta["task"], meta.get("strand") if level == "template_strand" else None, args)
        metrics, _ = compute_metrics_from_cm(cm, target_labels)
        row = {
            "reference_role": reference_role,
            "aggregation_level": level,
            "cohort": meta["cohort"],
            "group_label": cohort_to_group_label(meta["cohort"]),
            "task": meta["task"],
            "model": meta["model"],
            "model_display": MODEL_DISPLAY[meta["model"]],
            "template": meta["template"],
            "template_num": template_number(meta["template"]),
            "strand": meta.get("strand", "combined"),
            **metrics,
        }
        if reference_role == "control":
            row.update(compute_false_noncanonical_metrics(cm, meta["task"], args))
            row["display_template"] = f"DNA{template_number(meta['template'])}"
        else:
            row["display_template"] = meta["template"]
        rows.append(row)
    df = pd.DataFrame(rows)
    if not df.empty:
        df = df.sort_values(["cohort", "task", "model", "template_num", "strand"], kind="mergesort").reset_index(drop=True)
    return df


def run_evaluation(specs: List[DatasetSpec], args: argparse.Namespace, out_root: Path) -> None:
    raw_root = ensure_dir(out_root / "raw_evaluation")
    data_root = ensure_dir(out_root / "analysis_data")
    write_reference_mapping(data_root / "reference_name_mapping.csv")

    all_file_metrics = []
    file_cm_records = []
    efficiency_rows = []
    discovery_rows = []

    for spec in specs:
        print("\n" + "=" * 110)
        print(f"[DATASET] {spec.key} | {spec.cohort} | {spec.task}")
        print(f"[DIR]     {spec.test_dir}")
        print("=" * 110)

        test_files, diagnostics = find_test_files(spec)
        discovery_rows.extend(diagnostics)

        for model_type in args.models:
            ckpt_name = spec.model_files[model_type]
            ckpt_path = Path(args.model_dir) / ckpt_name
            model_out = ensure_dir(raw_root / spec.key / f"eval_{model_type}")
            print("\n" + "-" * 110)
            print(f"[MODEL] {MODEL_DISPLAY[model_type]} | {spec.key}")
            print(f"[CKPT]  {ckpt_path}")
            print("-" * 110)

            model = load_model_from_checkpoint(model_type, ckpt_path, args.device, args.mer_len, args.raw_signal_len, spec.num_classes)
            benchmark_state = {"warmup_remaining": float(args.benchmark_warmup_batches), "pure_forward_sec": 0.0, "timed_samples": 0.0, "timed_batches": 0.0}
            model_rows = []
            model_overall_cm = init_confusion_matrix(spec.num_classes)

            for i, csv_path in enumerate(test_files, start=1):
                print(f"[{i:02d}/{len(test_files):02d}] {csv_path.name}")
                result, cm = evaluate_one_file(model, model_type, csv_path, spec, model_out, args, benchmark_state)
                model_rows.append(result)
                all_file_metrics.append(result)
                model_overall_cm += cm
                file_cm_records.append({
                    "cohort": spec.cohort,
                    "task": spec.task,
                    "dataset_key": spec.key,
                    "group_label": spec.group_label,
                    "model": model_type,
                    "template": result["template"],
                    "strand": result["strand"],
                    "reference_name": result["reference_name"],
                    "reference_role": result["reference_role"],
                    "source_file": result["source_file"],
                    "cm": cm,
                })
                extra = ""
                if result["reference_role"] == "control":
                    extra = f" | false_noncanonical={result.get('false_noncanonical_call_rate', float('nan')):.4f}"
                print(
                    f"    ref={result['reference_name']} -> {result['template']} | role={result['reference_role']} | "
                    f"n={result['eval_samples']} | acc={result['accuracy']:.4f} | bal_acc={result['balanced_accuracy']:.4f} | "
                    f"target_recall={result['target_recall']:.4f}{extra}"
                )

            pd.DataFrame(model_rows).to_csv(model_out / "summary_metrics.csv", index=False)
            write_confusion_matrix_csv(model_overall_cm, model_out / "overall_confusion_matrix.csv")
            overall_metrics, _ = compute_metrics_from_cm(model_overall_cm, target_labels_for(spec.task, None, args))
            timed_samples = int(benchmark_state["timed_samples"])
            pure_forward_sec = float(benchmark_state["pure_forward_sec"])
            efficiency_rows.append({
                "cohort": spec.cohort,
                "task": spec.task,
                "dataset_key": spec.key,
                "model": model_type,
                "model_display": MODEL_DISPLAY[model_type],
                "checkpoint": str(ckpt_path),
                "batch_size": args.batch_size,
                "precision": "FP32" if args.no_amp or not args.device.startswith("cuda") else "AMP",
                "device": args.device,
                "warmup_batches": args.benchmark_warmup_batches,
                "timed_batches": int(benchmark_state["timed_batches"]),
                "timed_samples": timed_samples,
                "pure_forward_sec": pure_forward_sec,
                "samples_per_sec": safe_div(timed_samples, pure_forward_sec),
                "sec_per_million_samples": safe_div(pure_forward_sec * 1_000_000, timed_samples),
                **overall_metrics,
            })

            del model
            if args.device.startswith("cuda"):
                torch.cuda.empty_cache()

    pd.DataFrame(discovery_rows).to_csv(data_root / "discovered_test_files.csv", index=False)
    pd.DataFrame(all_file_metrics).to_csv(data_root / "file_metrics_long.csv", index=False)
    pd.DataFrame(efficiency_rows).to_csv(data_root / "inference_efficiency_long.csv", index=False)

    target_template_df = aggregate_role_metrics(file_cm_records, args, "target", "template")
    target_strand_df = aggregate_role_metrics(file_cm_records, args, "target", "template_strand")
    control_template_df = aggregate_role_metrics(file_cm_records, args, "control", "template")
    control_strand_df = aggregate_role_metrics(file_cm_records, args, "control", "template_strand")

    target_template_df.to_csv(data_root / "target_template_metrics_long.csv", index=False)
    target_strand_df.to_csv(data_root / "target_template_strand_metrics_long.csv", index=False)
    control_template_df.to_csv(data_root / "control_template_metrics_long.csv", index=False)
    control_strand_df.to_csv(data_root / "control_template_strand_metrics_long.csv", index=False)
    # 兼容保留 target-only 输出
    target_template_df.to_csv(data_root / "template_metrics_long.csv", index=False)
    target_strand_df.to_csv(data_root / "template_strand_metrics_long.csv", index=False)

    write_figure_specific_tables(data_root, target_template_df, control_template_df)

    manifest = {
        "generated_at": datetime.now().isoformat(timespec="seconds"),
        "script": Path(__file__).name,
        "output_root": str(out_root),
        "datasets": [asdict(spec) for spec in specs],
        "xna17_20_reference_mapping": {"84Ds4-AA": "XNA17", "84Ds4-AB": "XNA18", "84Ds4-AC": "XNA19", "84Ds4-AD": "XNA20"},
        "model_dir": args.model_dir,
        "device": args.device,
        "device_name": torch.cuda.get_device_name(torch.device(args.device)) if args.device.startswith("cuda") and torch.cuda.is_available() else (platform.processor() or "CPU"),
        "torch_version": torch.__version__,
        "batch_size": args.batch_size,
        "workers": args.workers,
        "amp_enabled": not args.no_amp,
        "raw_signal_len": args.raw_signal_len,
        "mer_len": args.mer_len,
        "dwell_divisor": args.dwell_divisor,
        "quality_divisor": args.quality_divisor,
        "seed": args.seed,
        "six_x_label": args.six_x_label,
        "six_y_label": args.six_y_label,
        "five_target_label": args.five_target_label,
    }
    save_json(manifest, out_root / "run_manifest.json")
    print(f"\n[INFO] 评估数据已保存到: {data_root}")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Evaluate held-out XNA1-20/DNA control templates with the four benchmark "
            "models and export Figure 4 analysis tables."
        )
    )
    parser.add_argument("--out_root", required=True,
                        help="Root directory for raw evaluation and analysis tables.")
    parser.add_argument("--model_dir", required=True,
                        help="Directory containing model checkpoint files.")
    parser.add_argument("--model_manifest", required=True,
                        help="JSON mapping each task/model to its checkpoint filename or path.")

    parser.add_argument("--dir_01_12_root", required=True,
                        help="Root containing ATCGXY_6class, ATCGX_5class, and ATCGY_5class for XNA1-12.")
    parser.add_argument("--dir_13_16_6", required=True)
    parser.add_argument("--dir_13_16_5x", required=True)
    parser.add_argument("--dir_13_16_5y", required=True)
    parser.add_argument("--dir_17_20_6", required=True)
    parser.add_argument("--dir_17_20_5x", required=True)
    parser.add_argument("--dir_17_20_5y", required=True)

    parser.add_argument("--models", nargs="+", choices=MODEL_ORDER, default=MODEL_ORDER,
                        help="Models to evaluate. Default: all four models.")
    parser.add_argument("--batch_size", type=int, default=256)
    parser.add_argument("--workers", type=int, default=0)
    parser.add_argument("--raw_signal_len", type=int, default=30)
    parser.add_argument("--mer_len", type=int, default=6)
    parser.add_argument("--dwell_divisor", type=float, default=60.0)
    parser.add_argument("--quality_divisor", type=float, default=50.0)
    parser.add_argument("--device", default="cuda:0" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--no_amp", action="store_true")
    parser.add_argument("--no_predictions", action="store_true")
    parser.add_argument("--prediction_compression", choices=["gzip", "none"], default="gzip")

    parser.add_argument("--six_x_label", type=int, default=4)
    parser.add_argument("--six_y_label", type=int, default=5)
    parser.add_argument("--five_target_label", type=int, default=4)
    parser.add_argument("--benchmark_warmup_batches", type=int, default=5)

    args = parser.parse_args()
    if args.batch_size <= 0:
        parser.error("--batch_size must be > 0")
    if args.workers < 0:
        parser.error("--workers must be >= 0")
    if args.raw_signal_len <= 0 or args.mer_len <= 0:
        parser.error("--raw_signal_len and --mer_len must be > 0")
    if args.dwell_divisor <= 0 or args.quality_divisor <= 0:
        parser.error("--dwell_divisor and --quality_divisor must be > 0")
    if args.benchmark_warmup_batches < 0:
        parser.error("--benchmark_warmup_batches must be >= 0")

    required_dirs = [
        args.dir_01_12_root, args.dir_13_16_6, args.dir_13_16_5x,
        args.dir_13_16_5y, args.dir_17_20_6, args.dir_17_20_5x,
        args.dir_17_20_5y, args.model_dir,
    ]
    missing = [path for path in required_dirs if not Path(path).is_dir()]
    if missing:
        parser.error("Required directories do not exist:\n" + "\n".join(missing))
    if not Path(args.model_manifest).is_file():
        parser.error(f"--model_manifest does not exist: {args.model_manifest}")
    return args


# =============================================================================
# New Figure 4 narrative plotting overrides
# B–E only show LiteMamba; F compares all four models with a Pareto/EGS design.
# =============================================================================

def write_figure_specific_tables(data_root: Path, target_template_df: pd.DataFrame, control_template_df: pd.DataFrame) -> None:
    """Export all data needed to redraw the revised Figure 4 without rerunning inference."""
    design_rows = []
    for cohort, purpose, level in [
        ("XNA1-12", "Single-NCB templates", 1),
        ("XNA13-16", "Multiple-NCB templates with variable spacing", 2),
        ("XNA17-20", "Four equidistant NCB templates", 3),
    ]:
        sub_t = target_template_df[target_template_df["cohort"] == cohort]
        sub_c = control_template_df[control_template_df["cohort"] == cohort]
        design_rows.append({
            "level": level,
            "cohort": cohort,
            "design_label": purpose,
            "num_xna_templates_detected": int(sub_t["template"].nunique()) if not sub_t.empty else 0,
            "num_dna_controls_detected": int(sub_c["template"].nunique()) if not sub_c.empty else 0,
            "dna_controls_note": "false non-canonical call test",
            "xna_templates_note": "target X/Y recovery test",
            "heldout_note": "held out from training and model selection",
        })
    pd.DataFrame(design_rows).to_csv(data_root / "figure4a_external_test_design_data.csv", index=False)

    dna_controls = control_template_df.copy()
    dna_controls.to_csv(data_root / "figure4b_dna_controls_false_noncanonical_data.csv", index=False)

    target_template_df[target_template_df["cohort"] == "XNA1-12"].to_csv(data_root / "figure4c_xna1_12_single_ncb_data.csv", index=False)
    target_template_df[target_template_df["cohort"] == "XNA13-16"].to_csv(data_root / "figure4d_xna13_16_multi_ncb_data.csv", index=False)
    target_template_df[target_template_df["cohort"] == "XNA17-20"].to_csv(data_root / "figure4e_xna17_20_four_equidistant_data.csv", index=False)

    # Figure 4F: transparent, pre-defined external generalization score.
    pooled_dna = (
        dna_controls.groupby(["task", "model", "model_display"], as_index=False)["false_noncanonical_call_rate"]
        .mean()
        .rename(columns={"false_noncanonical_call_rate": "pooled_dna_false_noncanonical_call_rate"})
    )
    pooled_dna["dna_specificity"] = 1.0 - pooled_dna["pooled_dna_false_noncanonical_call_rate"]

    xna_summary = (
        target_template_df.groupby(["task", "model", "model_display"], as_index=False)
        .agg(
            mean_xna_target_recall=("target_recall", "mean"),
            worst_template_target_recall=("target_recall", "min"),
            sd_template_target_recall=("target_recall", "std"),
            n_xna_templates=("template", "nunique"),
            mean_target_support=("target_support", "mean"),
        )
    )
    xna_summary["template_stability"] = 1.0 - xna_summary["sd_template_target_recall"].fillna(0.0)
    xna_summary["template_stability"] = xna_summary["template_stability"].clip(lower=0.0, upper=1.0)

    # Group-level means are exported for the supplementary table and for interpretation.
    group_means = (
        target_template_df.groupby(["cohort", "group_label", "task", "model", "model_display"], as_index=False)["target_recall"]
        .mean()
        .rename(columns={"target_recall": "mean_group_target_recall"})
    )
    group_means.to_csv(data_root / "figure4f_group_level_recall_data.csv", index=False)

    egs = pooled_dna.merge(xna_summary, on=["task", "model", "model_display"], how="outer")

    efficiency_path = data_root / "inference_efficiency_long.csv"
    if efficiency_path.exists():
        eff = pd.read_csv(efficiency_path)
        if "samples_per_sec" in eff.columns:
            eff_summary = (
                eff.groupby(["task", "model", "model_display"], as_index=False)["samples_per_sec"]
                .mean()
                .rename(columns={"samples_per_sec": "mean_samples_per_sec"})
            )
            egs = egs.merge(eff_summary, on=["task", "model", "model_display"], how="left")
        else:
            egs["mean_samples_per_sec"] = np.nan
    else:
        egs["mean_samples_per_sec"] = np.nan

    # Normalize efficiency within each task so that EGS remains in [0,1].
    norm_values = []
    for task, sub in egs.groupby("task", sort=False):
        vals = sub["mean_samples_per_sec"].astype(float)
        if vals.notna().sum() <= 1 or float(vals.max()) == float(vals.min()):
            norm = pd.Series([1.0 if pd.notna(v) else 0.5 for v in vals], index=sub.index)
        else:
            norm = (vals - vals.min()) / (vals.max() - vals.min())
            norm = norm.fillna(0.5)
        norm_values.append(norm)
    if norm_values:
        egs["efficiency_norm"] = pd.concat(norm_values).sort_index()
    else:
        egs["efficiency_norm"] = np.nan

    # EGS weights are explicitly exported so the score is transparent.
    # Components: DNA specificity, mean XNA recall, worst-template recall, template stability, and relative throughput.
    weights = {
        "weight_dna_specificity": 0.30,
        "weight_mean_xna_target_recall": 0.35,
        "weight_worst_template_target_recall": 0.20,
        "weight_template_stability": 0.10,
        "weight_efficiency_norm": 0.05,
    }
    egs["external_generalization_score"] = (
        weights["weight_dna_specificity"] * egs["dna_specificity"]
        + weights["weight_mean_xna_target_recall"] * egs["mean_xna_target_recall"]
        + weights["weight_worst_template_target_recall"] * egs["worst_template_target_recall"]
        + weights["weight_template_stability"] * egs["template_stability"]
        + weights["weight_efficiency_norm"] * egs["efficiency_norm"]
    )
    for key, value in weights.items():
        egs[key] = value

    egs = egs.sort_values(["task", "external_generalization_score"], ascending=[True, False], kind="mergesort")
    egs.to_csv(data_root / "figure4f_external_generalization_score_data.csv", index=False)
    egs.to_csv(data_root / "figure4f_pareto_summary_data.csv", index=False)


# =============================================================================
# main
# =============================================================================

def main() -> None:
    set_csv_field_size_limit()
    args = parse_args()
    set_seed(args.seed)
    out_root = ensure_dir(Path(args.out_root))
    model_manifest = load_model_manifest(Path(args.model_manifest))
    specs = build_specs(args, model_manifest)

    print("=" * 110)
    print("External template evaluation for Figure 4")
    print(f"out_root   : {out_root}")
    print(f"device     : {args.device}")
    print(f"batch_size : {args.batch_size}")
    print(f"models     : {args.models}")
    print(f"AMP        : {not args.no_amp}")
    print("XNA17-20 reference mapping: 84Ds4-AA/AB/AC/AD -> XNA17/18/19/20")
    print("strand mapping: pos -> X/5X; neg -> Y/5Y")
    print("=" * 110)

    run_evaluation(specs, args, out_root)

    print("\n" + "=" * 110)
    print("External evaluation complete.")
    print(f"Raw evaluation: {out_root / 'raw_evaluation'}")
    print(f"Analysis data : {out_root / 'analysis_data'}")
    print("=" * 110)


if __name__ == "__main__":
    main()
