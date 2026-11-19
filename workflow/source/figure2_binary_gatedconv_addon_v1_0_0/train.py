#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Read 隔离二分类训练与冻结模型评价。每轮仅使用 train/val，test 不参与选择。

建议队列训练传入 --defer_test；所有架构、seed 和训练设置冻结后，再运行
--mode evaluate --output_dir RUN_DIR。充分保存模型、逐样本预测和中间表示，
后续统计与绘图可直接重新运行 analyze_results.py，无需重新训练。
默认每轮保存精简 validation logits/labels/sample_index 的 NPZ，可结合固定
metadata manifest 重算各参考/motif的逐轮曲线；--no_epoch_logits 可关闭。
"""
import argparse
import contextlib
import csv
import gzip
import hashlib
import importlib.metadata
import json
import logging
import math
import os
import platform
import random
import resource
import shutil
import subprocess
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader

from config_19 import DEFAULT_DATA_DIR, ModelConfig
from IterDataset import build_data_bundle
from model_benchmark import build_model
from analyze_results import binary_metrics, analyze_predictions

VERSION = "2.0.0"
ACTIVE_RUN_DIR = None
ACTIVE_MODE = None
SCHEMES = ("full", "raw_only", "context_only", "responses_only", "statistics_only",
           "no_raw", "no_context", "no_responses", "no_statistics")
PREDICTION_FIELDS = ["sample_index", "sample_id", "read_id", "chrom", "pair_id", "template_group",
                     "bam_strand", "motif", "original_kmer", "label", "logit_0", "logit_1",
                     "prob_0", "prob_1", "calibrated_prob_0", "calibrated_prob_1", "prediction",
                     "confidence", "calibrated_confidence", "run_id", "source_dataset", "source_file",
                     "source_row", "model_type", "seed", "split", "feature_scheme", "trainer_version"]
FEATURE_DEFINITION = {"sequence": "ATCGN one-hot, target index 3 masked to N",
    "merged": {"context": [0, 5], "statistics": [5, 9], "responses": [9, 13]},
    "statistics": ["mean", "std", "median", "dwell"],
    "responses": ["quality", "mismatch", "insertion", "deletion"],
    "raw": "6 base-level segments, fixed deterministic sampling per base",
    "target_readout": "zero-based target index 3 plus global attention",
    "ssm": "pure PyTorch selective recurrence on 6 base-level tokens; no official CUDA Mamba kernel"}


def parser():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--mode", choices=("train", "evaluate"), default="train")
    p.add_argument("--data_dir", default=DEFAULT_DATA_DIR)
    p.add_argument("--strand", choices=("pos", "neg"), default="pos")
    for split in ("train", "val", "test"):
        p.add_argument("--" + split + "_data")
        p.add_argument("--" + split + "_meta")
    p.add_argument("--output_dir")
    p.add_argument("--cache_dir")
    p.add_argument("--model_type", choices=("litemamba", "bigru", "transformer", "convmixer"), default="litemamba")
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--epochs", type=int, default=200)
    p.add_argument("--patience", type=int, default=30)
    p.add_argument("--lr", type=float, default=1e-3)
    p.add_argument("--batch_size", type=int, default=256)
    p.add_argument("--workers", type=int, default=4)
    p.add_argument("--raw_signal_len", type=int, default=30)
    p.add_argument("--dwell_divisor", type=float, default=60)
    p.add_argument("--quality_divisor", type=float, default=50)
    p.add_argument("--accum_steps", type=int, default=1)
    p.add_argument("--weight_decay", type=float, default=1e-4)
    p.add_argument("--label_smoothing", type=float, default=0.0)
    p.add_argument("--clip_grad", type=float, default=1.0)
    p.add_argument("--device", default="auto")
    p.add_argument("--amp", dest="amp", action="store_true")
    p.add_argument("--no_amp", dest="amp", action="store_false")
    p.set_defaults(amp=None)
    p.add_argument("--cpu_threads", type=int, default=4, help="CPU intra-op 线程数；0 保留 torch 默认")
    p.add_argument("--positive_name", help="标签1语义；默认正链 X，反链 Y")
    p.add_argument("--defer_test", action="store_true")
    p.add_argument("--checkpoint")
    p.add_argument("--resume", nargs="?", const="auto")
    p.add_argument("--checkpoint_every", type=int, default=10)
    p.add_argument("--monitor", choices=("val_loss", "val_macro_f1"), default="val_loss")
    p.add_argument("--save_epoch_predictions", action="store_true")
    p.add_argument("--no_epoch_logits", action="store_true",
                   help="关闭默认每轮 val float32 logits/labels/sample_index 的压缩 NPZ 留档")
    p.add_argument("--no_embeddings", action="store_true")
    p.add_argument("--embedding_dtype", choices=("float32", "float16"), default="float32")
    p.add_argument("--no_attention", action="store_true")
    p.add_argument("--bootstrap", type=int, default=200)
    p.add_argument("--no_plots", action="store_true")
    p.add_argument("--external_manifest")
    p.add_argument("--force_evaluate", action="store_true", help="显式重新计算已有冻结模型评价；默认复用完整结果")
    for key, default in (("model_dim", 96), ("num_blocks", 4), ("state_dim", 16),
                         ("target_index", 3), ("nhead", 4), ("num_layers", 2), ("gru_hidden", 64)):
        p.add_argument("--" + key, type=int, default=default)
    p.add_argument("--dropout", type=float, default=0.1)
    p.add_argument("--classifier_dropout", type=float, default=0.2)
    p.add_argument("--ssm_bidirectional", type=int, choices=(0, 1), default=1)
    p.add_argument("--litemamba_variant", choices=("selective_ssm", "legacy_gatedconv"), default="selective_ssm")
    p.add_argument("--readout", choices=("target_attention", "legacy_last_attention"), default="target_attention")
    p.add_argument("--feature_scheme", choices=SCHEMES, default="full")
    return p


def json_safe(value):
    if isinstance(value, dict):
        return {str(k): json_safe(v) for k, v in value.items()}
    if isinstance(value, (tuple, list)):
        return [json_safe(v) for v in value]
    if isinstance(value, np.ndarray):
        return json_safe(value.tolist())
    if isinstance(value, np.generic):
        return json_safe(value.item())
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, float) and not math.isfinite(value):
        return None
    return value


def write_json(path, value):
    path = Path(path)
    temporary = path.with_name(path.name + ".tmp")
    temporary.write_text(json.dumps(json_safe(value), ensure_ascii=False, indent=2, allow_nan=False) + "\n", encoding="utf-8")
    temporary.replace(path)


def load_json(path):
    return json.loads(Path(path).read_text(encoding="utf-8"))


def sha256(path):
    digest = hashlib.sha256()
    with open(path, "rb") as stream:
        for chunk in iter(lambda: stream.read(8 * 1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def timestamp():
    return datetime.now(timezone.utc).isoformat()


def setup_logging(run_dir):
    logger = logging.getLogger("binary_training")
    logger.setLevel(logging.INFO)
    logger.handlers.clear()
    formatter = logging.Formatter("%(asctime)s %(levelname)s %(message)s")
    for handler in (logging.StreamHandler(sys.stdout), logging.FileHandler(run_dir / "train.log", encoding="utf-8")):
        handler.setFormatter(formatter)
        logger.addHandler(handler)
    return logger


def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True


def get_rng():
    return {"python": random.getstate(), "numpy": np.random.get_state(), "torch": torch.get_rng_state(),
            "cuda": torch.cuda.get_rng_state_all() if torch.cuda.is_available() else []}


def restore_rng(state):
    random.setstate(state["python"])
    np.random.set_state(state["numpy"])
    torch.set_rng_state(state["torch"].cpu())
    if torch.cuda.is_available() and state.get("cuda"):
        torch.cuda.set_rng_state_all([x.cpu() for x in state["cuda"]])


def seed_worker(worker_id):
    seed = torch.initial_seed() % (2 ** 32)
    np.random.seed(seed)
    random.seed(seed)


def loader(dataset, args, train=False, epoch=0):
    generator = torch.Generator().manual_seed(args.seed + epoch * 100003 + (1 if train else 2))
    return DataLoader(dataset, batch_size=args.batch_size, shuffle=train, num_workers=args.workers,
                      drop_last=False, pin_memory=str(args.device).startswith("cuda"),
                      worker_init_fn=seed_worker, generator=generator, persistent_workers=False)


def batch_to_device(batch, device):
    return tuple(x.to(device, non_blocking=True) for x in batch["inputs"]), batch["label"].to(device).long()


def model_options(args):
    keys = ModelConfig.model_kwargs().keys()
    return {key: bool(getattr(args, key)) if key == "ssm_bidirectional" else getattr(args, key) for key in keys}


def configure_paths(args):
    args.data_dir = str(Path(args.data_dir).expanduser().resolve())
    for split in ("train", "val", "test"):
        path = getattr(args, split + "_data")
        if path is None:
            path = str(Path(args.data_dir) / ("xna_pc_6mer_features_{}_balanced_{}.csv".format(args.strand, split)))
        path = str(Path(path).expanduser().resolve())
        setattr(args, split + "_data", path)
        meta = getattr(args, split + "_meta")
        if meta is None:
            candidates = [Path(path).with_name(Path(path).stem + "_meta.tsv"), Path(path).with_suffix(".meta.tsv")]
            meta = next((str(p) for p in candidates if p.is_file()), None)
        if meta is not None:
            meta = str(Path(meta).expanduser().resolve())
        setattr(args, split + "_meta", meta)


def select_device(args):
    if args.device == "auto":
        args.device = "cuda:0" if torch.cuda.is_available() else "cpu"
    device = torch.device(args.device)
    if device.type == "cuda" and not torch.cuda.is_available():
        raise ValueError("指定 CUDA，但当前 torch 未发现 CUDA。")
    args.amp = device.type == "cuda" if args.amp is None else args.amp
    if device.type != "cuda" and args.amp:
        raise ValueError("本实现 AMP 仅支持 CUDA；CPU 请使用 --no_amp。")
    if args.cpu_threads:
        torch.set_num_threads(args.cpu_threads)
    return device


def autocast_context(args, device):
    return torch.autocast(device_type=device.type, dtype=torch.float16, enabled=args.amp)


def environment(device):
    versions = {}
    for package in ("numpy", "scipy", "scikit-learn", "matplotlib", "torch"):
        try:
            versions[package] = importlib.metadata.version(package)
        except importlib.metadata.PackageNotFoundError:
            versions[package] = None
    return {"created_at": timestamp(), "python": sys.version, "platform": platform.platform(),
            "packages": versions, "torch": torch.__version__, "cuda_runtime": torch.version.cuda,
            "cudnn": torch.backends.cudnn.version(), "device": str(device),
            "gpu_name": torch.cuda.get_device_name(device) if device.type == "cuda" else None,
            "torch_threads": torch.get_num_threads(),
            "torch_interop_threads": torch.get_num_interop_threads(),
            "deterministic_cudnn": torch.backends.cudnn.deterministic}


def source_snapshot(run_dir):
    root = Path(__file__).resolve().parent
    target = run_dir / "source_snapshot"
    target.mkdir(exist_ok=True)
    hashes = {}
    for path in sorted(root.glob("*.py")):
        hashes[path.name] = sha256(path)
        shutil.copy2(path, target / path.name)
    try:
        git = subprocess.run(["git", "rev-parse", "HEAD"], cwd=str(root), capture_output=True, text=True)
        revision = git.stdout.strip() if git.returncode == 0 else None
    except OSError:
        revision = None
    info = {"files_sha256": hashes, "git_revision": revision}
    write_json(target / "manifest.json", info)
    return info


def current_source_hashes():
    return {path.name: sha256(path) for path in sorted(Path(__file__).resolve().parent.glob("*.py"))}


def host_memory():
    raw = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
    factor = 1 if sys.platform == "darwin" else 1024
    result = {"host_peak_rss_bytes": int(raw * factor), "host_peak_rss_source": "resource.ru_maxrss (Darwin bytes; Linux KiB)"}
    try:
        import psutil
        result["host_current_rss_bytes"] = psutil.Process().memory_info().rss
    except ImportError:
        result["host_current_rss_bytes"] = None
    return result


def gpu_memory(device):
    if device.type != "cuda":
        return {"gpu_peak_allocated_bytes": None, "gpu_peak_reserved_bytes": None}
    return {"gpu_peak_allocated_bytes": torch.cuda.max_memory_allocated(device),
            "gpu_peak_reserved_bytes": torch.cuda.max_memory_reserved(device)}


def torch_load(path, device="cpu"):
    try:
        return torch.load(path, map_location=device, weights_only=False)
    except TypeError:
        return torch.load(path, map_location=device)


def save_checkpoint(path, checkpoint):
    temporary = path.with_name(path.name + ".tmp")
    torch.save(checkpoint, temporary)
    temporary.replace(path)


def metric_pass(model, dataset, args, device, optimizer=None, scaler=None, epoch=0):
    """一个 forward 同时计算 loss 与所有分类指标；累积梯度按真实样本数归一。"""
    training = optimizer is not None
    model.train(training)
    data_loader = loader(dataset, args, train=training, epoch=epoch)
    if device.type == "cuda":
        torch.cuda.reset_peak_memory_stats(device)
        torch.cuda.synchronize(device)
    started = time.perf_counter()
    probabilities, labels, logits_all = [], [], []
    loss_total, samples, steps, window_samples = 0.0, 0, 0, 0
    grad_norms = []
    if training:
        optimizer.zero_grad(set_to_none=True)
    grad_context = contextlib.nullcontext() if training else torch.no_grad()
    with grad_context:
        for batch_number, batch in enumerate(data_loader, start=1):
            inputs, target = batch_to_device(batch, device)
            with autocast_context(args, device):
                logits, _ = model(inputs)
                loss_sum = F.cross_entropy(logits, target, reduction="sum", label_smoothing=args.label_smoothing)
            if not torch.isfinite(loss_sum):
                raise RuntimeError("训练/验证 loss 非有限值；停止而不跳过批次。")
            n = target.numel()
            if training:
                scaler.scale(loss_sum).backward()
                window_samples += n
                if batch_number % args.accum_steps == 0 or batch_number == len(data_loader):
                    scaler.unscale_(optimizer)
                    for parameter in model.parameters():
                        if parameter.grad is not None:
                            parameter.grad.div_(window_samples)
                    norm = torch.nn.utils.clip_grad_norm_(model.parameters(), args.clip_grad)
                    grad_norms.append(float(norm.detach().cpu()))
                    scaler.step(optimizer)
                    scaler.update()
                    optimizer.zero_grad(set_to_none=True)
                    window_samples = 0
                    steps += 1
            logits_cpu = logits.detach().float().cpu().numpy()
            logits_all.append(logits_cpu)
            probabilities.append(torch.softmax(logits.detach().float(), dim=-1).cpu().numpy())
            labels.append(target.detach().cpu().numpy())
            loss_total += float(loss_sum.detach().cpu())
            samples += n
    if device.type == "cuda":
        torch.cuda.synchronize(device)
    elapsed = time.perf_counter() - started
    y, p, z = np.concatenate(labels), np.concatenate(probabilities), np.concatenate(logits_all)
    metrics = binary_metrics(y, p)
    metrics.update(loss=loss_total / samples, processed_samples=samples, batches=len(data_loader),
                   optimizer_steps=steps, seconds=elapsed, samples_per_second=samples / elapsed,
                   mean_grad_norm=float(np.mean(grad_norms)) if grad_norms else None,
                   max_grad_norm=max(grad_norms) if grad_norms else None, **host_memory(), **gpu_memory(device))
    if samples != len(dataset):
        raise RuntimeError("训练/验证实际处理数与 dataset 长度不一致。")
    return json_safe(metrics), y, z


def write_history(run_dir, history):
    fields = ["epoch", "lr", "train_loss", "val_loss", "train_accuracy", "val_accuracy",
              "train_macro_f1", "val_macro_f1", "train_precision", "val_precision", "train_recall",
              "val_recall", "val_specificity", "val_fpr", "train_seconds", "val_seconds",
              "train_samples_per_second", "val_samples_per_second", "mean_grad_norm", "max_grad_norm"]
    with open(run_dir / "metrics.csv", "w", encoding="utf-8", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=fields)
        writer.writeheader()
        for item in history:
            row = {"epoch": item["epoch"], "lr": item["lr"]}
            for split in ("train", "val"):
                for key in ("loss", "accuracy", "macro_f1", "precision", "recall", "specificity", "fpr", "seconds", "samples_per_second"):
                    if split + "_" + key in fields:
                        row[split + "_" + key] = item[split].get(key)
            for key in ("mean_grad_norm", "max_grad_norm"):
                row[key] = item["train"].get(key)
            writer.writerow(row)
    with open(run_dir / "epoch_metrics.jsonl", "w", encoding="utf-8") as stream:
        for item in history:
            stream.write(json.dumps(json_safe(item), ensure_ascii=False, allow_nan=False) + "\n")


def probability(logits, temperature=1.0):
    z = np.asarray(logits, dtype=np.float64) / temperature
    z -= z.max(axis=1, keepdims=True)
    e = np.exp(z)
    return e / e.sum(axis=1, keepdims=True)


def fit_temperature(logits, labels):
    """仅在 validation 上优化单一温度，黄金分割搜索 log(T)，不依赖 scipy。"""
    def nll(log_temperature):
        total = 0.0
        t = math.exp(log_temperature)
        for start in range(0, len(labels), 65536):
            stop = min(start + 65536, len(labels))
            diff = (np.asarray(logits[start:stop, 1], dtype=np.float64)
                    - np.asarray(logits[start:stop, 0], dtype=np.float64)) / t
            total += float(np.sum(np.logaddexp(0, diff) - np.asarray(labels[start:stop]) * diff))
        return total / len(labels)
    left, right = math.log(0.05), math.log(10.0)
    ratio = (math.sqrt(5) - 1) / 2
    c, d = right - ratio * (right - left), left + ratio * (right - left)
    fc, fd = nll(c), nll(d)
    for _ in range(60):
        if fc < fd:
            right, d, fd = d, c, fc
            c = right - ratio * (right - left)
            fc = nll(c)
        else:
            left, c, fc = c, d, fd
            d = left + ratio * (right - left)
            fd = nll(d)
    optimum = (left + right) / 2
    raw = nll(0.0)
    if nll(optimum) > raw:
        optimum = 0.0
    return {"temperature": math.exp(optimum), "fitted_on": "val", "n_samples": len(labels),
            "method": "bounded golden-section search on log_temperature", "temperature_bounds": [0.05, 10.0],
            "raw_val_nll": raw, "calibrated_val_nll": nll(optimum)}


def metadata_for(dataset, index):
    value = dataset.get_metadata(int(index))
    if not isinstance(value, dict):
        raise TypeError("dataset.get_metadata(index) 必须返回 dict。")
    return value


def write_predictions(path, dataset, labels, logits, temperature, args, split):
    with gzip.open(path, "wt", encoding="utf-8", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=PREDICTION_FIELDS)
        writer.writeheader()
        for start in range(0, len(dataset), 65536):
            stop = min(start + 65536, len(dataset))
            raw = probability(logits[start:stop])
            calibrated = probability(logits[start:stop], temperature)
            for offset in range(stop - start):
                index = start + offset
                meta = metadata_for(dataset, index)
                row = {key: meta.get(key, "") for key in PREDICTION_FIELDS}
                row.update(sample_index=index, sample_id=meta.get("sample_id") or split + ":" + str(index),
                           motif=meta.get("masked_motif") or meta.get("motif") or meta.get("kmer", ""),
                           label=int(labels[index]), logit_0=float(logits[index, 0]), logit_1=float(logits[index, 1]),
                           prob_0=float(raw[offset, 0]), prob_1=float(raw[offset, 1]),
                           calibrated_prob_0=float(calibrated[offset, 0]), calibrated_prob_1=float(calibrated[offset, 1]),
                           prediction=int(raw[offset, 1] >= 0.5), confidence=float(raw[offset].max()),
                           calibrated_confidence=float(calibrated[offset].max()),
                           source_dataset=meta.get("source_dataset") or "unknown",
                           run_id=meta.get("run_id") or "unknown", model_type=args.model_type,
                           seed=args.seed, split=split, feature_scheme=args.feature_scheme, trainer_version=VERSION)
                writer.writerow(row)


def infer_arrays(model, dataset, args, device, out_dir):
    """按本地 sample_index 写 memmap，保留 float32 logits 与默认 float32 embedding。"""
    n = len(dataset)
    out_dir.mkdir(parents=True, exist_ok=True)
    z = np.lib.format.open_memmap(out_dir / "logits.npy", mode="w+", dtype="float32", shape=(n, 2))
    y = np.lib.format.open_memmap(out_dir / "labels.npy", mode="w+", dtype="int64", shape=(n,))
    embedding = attention = None
    seen = np.zeros(n, dtype=bool)
    model.eval()
    started = time.perf_counter()
    with torch.no_grad():
        for batch in loader(dataset, args):
            inputs, labels = batch_to_device(batch, device)
            with autocast_context(args, device):
                logits, representation, aux = model(inputs, return_aux=True)
            indices = np.asarray(batch["index"], dtype=np.int64)
            if indices.ndim != 1 or np.any(indices < 0) or np.any(indices >= n) or np.any(seen[indices]):
                raise RuntimeError("推理 sample_index 越界或重复。")
            seen[indices] = True
            z[indices] = logits.float().cpu().numpy()
            y[indices] = labels.cpu().numpy()
            if not args.no_embeddings:
                if embedding is None:
                    embedding = np.lib.format.open_memmap(out_dir / "embeddings.npy", mode="w+",
                        dtype=args.embedding_dtype, shape=(n, representation.shape[-1]))
                embedding[indices] = representation.float().cpu().numpy()
            if not args.no_attention and "attention" in aux:
                if attention is None:
                    attention = np.lib.format.open_memmap(out_dir / "attention.npy", mode="w+",
                        dtype="float32", shape=(n, aux["attention"].shape[-1]))
                attention[indices] = aux["attention"].float().cpu().numpy()
    if not seen.all() or not np.isfinite(z).all():
        raise RuntimeError("推理缺失样本或出现非有限 logits。")
    for array in (z, y, embedding, attention):
        if array is not None:
            array.flush()
    return y, z, time.perf_counter() - started


def finish_export(dataset, args, split, out_dir, labels, logits, temperature, checkpoint_hash, inference_seconds):
    started = time.perf_counter()
    predictions = out_dir / "predictions.csv.gz"
    write_predictions(predictions, dataset, labels, logits, temperature, args, split)
    metrics = {"raw": binary_metrics(labels, probability(logits)),
               "calibrated": binary_metrics(labels, probability(logits, temperature))}
    write_json(out_dir / "prediction_metrics.json", metrics)
    analysis = analyze_predictions(str(predictions), str(out_dir / "analysis"), bootstrap=args.bootstrap,
                                   seed=args.seed, plots=not args.no_plots,
                                   run_metadata={key: getattr(args, key) for key in (
                                       "model_type", "seed", "strand", "feature_scheme", "litemamba_variant", "readout")})
    info = {"split": split, "n_samples": len(dataset), "checkpoint_sha256": checkpoint_hash,
            "dataset_identity": export_dataset_identity(dataset),
            "temperature": temperature, "inference_seconds": inference_seconds,
            "export_and_analysis_seconds": time.perf_counter() - started,
            "predictions": str(predictions), "metrics": metrics,
            "embeddings": str(out_dir / "embeddings.npy") if (out_dir / "embeddings.npy").exists() else None,
            "attention": str(out_dir / "attention.npy") if (out_dir / "attention.npy").exists() else None,
            "analysis": str(out_dir / "analysis"), "completed_at": timestamp()}
    write_json(out_dir / "export_complete.json", info)
    return info


def export_split(model, dataset, args, device, split, run_dir, temperature, checkpoint_hash):
    out_dir = run_dir / split
    complete = out_dir / "export_complete.json"
    if complete.exists():
        prior = load_json(complete)
        if not args.force_evaluate:
            if prior.get("dataset_identity") != export_dataset_identity(dataset):
                raise ValueError("已有预测的输入内容 SHA256 或预处理不同，不能复用：{}；"
                                 "请换 external name 或显式 --force_evaluate（会归档旧输出）。".format(out_dir))
            if prior["checkpoint_sha256"] != checkpoint_hash or prior["temperature"] != temperature:
                raise ValueError("已有预测属于另一 checkpoint/temperature，拒绝混用：" + str(out_dir))
            return prior
    if args.force_evaluate and out_dir.exists():
        archive = run_dir / "previous_exports" / (datetime.now().strftime("%Y%m%d_%H%M%S_%f") + "_forced_" + split)
        archive.parent.mkdir(exist_ok=True)
        shutil.move(str(out_dir), str(archive))
        write_json(archive / "archived_from.json", {"original_path": str(out_dir), "reason": "explicit force_evaluate",
                                                   "archived_at": timestamp()})
    labels, logits, seconds = infer_arrays(model, dataset, args, device, out_dir)
    return finish_export(dataset, args, split, out_dir, labels, logits, temperature, checkpoint_hash, seconds)


def benchmark_model(model, dataset, args, device):
    batch = next(iter(loader(dataset, args)))
    inputs, target = batch_to_device(batch, device)
    model.eval()
    def sync():
        if device.type == "cuda":
            torch.cuda.synchronize(device)
    with torch.no_grad():
        for _ in range(10):
            with autocast_context(args, device):
                model(inputs)
        sync()
        if device.type == "cuda":
            torch.cuda.reset_peak_memory_stats(device)
        durations = []
        for _ in range(30):
            sync()
            start = time.perf_counter()
            with autocast_context(args, device):
                model(inputs)
            sync()
            durations.append(time.perf_counter() - start)
    mean = float(np.mean(durations))
    return {"total_parameters": sum(p.numel() for p in model.parameters()),
            "trainable_parameters": sum(p.numel() for p in model.parameters() if p.requires_grad),
            "batch_size": target.numel(), "warmup": 10, "repeats": 30,
            "forward_mean_batch_ms": mean * 1000, "forward_p50_batch_ms": float(np.percentile(durations, 50)) * 1000,
            "forward_p95_batch_ms": float(np.percentile(durations, 95)) * 1000,
            "amortized_sample_ms": mean * 1000 / target.numel(), "samples_per_second": target.numel() / mean,
            "device": str(device), "amp": args.amp, "scope": "pure forward; excludes data loading, host-to-device copy and export",
            **host_memory(), **gpu_memory(device)}


def checkpoint_payload(model, optimizer, scheduler, scaler, epoch, best_epoch, best_value, patience, history, config):
    return {"trainer_version": VERSION, "model_state_dict": model.state_dict(),
            "optimizer_state_dict": optimizer.state_dict(), "scheduler_state_dict": scheduler.state_dict(),
            "scaler_state_dict": scaler.state_dict(), "epoch": epoch, "best_epoch": best_epoch,
            "best_value": best_value, "patience_counter": patience, "history": history,
            "rng_state": get_rng(), "resolved_config": config, "created_at": timestamp()}


def train_model(model, datasets, args, device, run_dir, config, log):
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    monitor_mode = "min" if args.monitor == "val_loss" else "max"
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode=monitor_mode, factor=0.5, patience=8, min_lr=1e-6)
    try:
        scaler = torch.amp.GradScaler("cuda", enabled=args.amp)
    except AttributeError:
        scaler = torch.cuda.amp.GradScaler(enabled=args.amp)
    history, best_epoch, best_value, patience_counter, start_epoch = [], 0, math.inf if monitor_mode == "min" else -math.inf, 0, 1
    if args.resume:
        path = Path(args.checkpoint or (args.resume if args.resume != "auto" else run_dir / "last_checkpoint.pt"))
        checkpoint = torch_load(path, device)
        old = checkpoint["resolved_config"]
        for key in ("source_snapshot", "model_kwargs", "seed", "monitor", "dwell_divisor", "quality_divisor", "raw_signal_len"):
            if old.get(key) != config.get(key):
                raise ValueError("Resume 的代码、数据或设置指纹不一致：" + key)
        if content_fingerprints(old["input_fingerprints"]) != content_fingerprints(config["input_fingerprints"]):
            raise ValueError("Resume 数据内容 SHA256 不一致。")
        model.load_state_dict(checkpoint["model_state_dict"])
        optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
        scheduler.load_state_dict(checkpoint["scheduler_state_dict"])
        scaler.load_state_dict(checkpoint["scaler_state_dict"])
        history = checkpoint["history"]
        best_epoch, best_value = checkpoint["best_epoch"], checkpoint["best_value"]
        patience_counter, start_epoch = checkpoint["patience_counter"], checkpoint["epoch"] + 1
        restore_rng(checkpoint["rng_state"])
        log.info("Resume %s，下一 epoch=%s", path, start_epoch)
    for epoch in range(start_epoch, args.epochs + 1):
        lr = optimizer.param_groups[0]["lr"]
        train_metrics, _, _ = metric_pass(model, datasets["train"], args, device, optimizer, scaler, epoch)
        val_metrics, y_val, logits_val = metric_pass(model, datasets["val"], args, device)
        value = val_metrics["loss" if args.monitor == "val_loss" else "macro_f1"]
        if value is None or not math.isfinite(value):
            raise RuntimeError("模型选择指标不可用：" + args.monitor)
        scheduler.step(value)
        improved = value < best_value if monitor_mode == "min" else value > best_value
        if improved:
            best_value, best_epoch, patience_counter = value, epoch, 0
        else:
            patience_counter += 1
        history.append({"epoch": epoch, "lr": lr, "next_lr": optimizer.param_groups[0]["lr"],
                        "train": train_metrics, "val": val_metrics, "monitor": args.monitor,
                        "monitor_value": value, "improved": improved, "patience_counter": patience_counter})
        write_history(run_dir, history)
        payload = checkpoint_payload(model, optimizer, scheduler, scaler, epoch, best_epoch, best_value,
                                     patience_counter, history, config)
        save_checkpoint(run_dir / "last_checkpoint.pt", payload)
        if improved:
            save_checkpoint(run_dir / "best_checkpoint.pt", payload)
            save_checkpoint(run_dir / "best_state_dict.pt", model.state_dict())
        if args.checkpoint_every > 0 and epoch % args.checkpoint_every == 0:
            save_checkpoint(run_dir / ("checkpoint_epoch_{:04d}.pt".format(epoch)), payload)
        if args.save_epoch_predictions:
            directory = run_dir / "epoch_predictions"
            directory.mkdir(exist_ok=True)
            write_predictions(directory / ("val_epoch_{:04d}.csv.gz".format(epoch)), datasets["val"],
                              y_val, logits_val, 1.0, args, "val")
        if not args.no_epoch_logits:
            directory = run_dir / "epoch_predictions"
            directory.mkdir(exist_ok=True)
            np.savez_compressed(directory / ("val_epoch_{:04d}.npz".format(epoch)),
                                logits=np.asarray(logits_val, dtype=np.float32),
                                labels=np.asarray(y_val, dtype=np.int64),
                                sample_index=np.arange(len(y_val), dtype=np.int64))
        log.info("epoch=%d train_loss=%.6f val_loss=%.6f val_macro_f1=%s lr=%.3g best=%d patience=%d",
                 epoch, train_metrics["loss"], val_metrics["loss"], val_metrics.get("macro_f1"), lr, best_epoch, patience_counter)
        if patience_counter >= args.patience:
            log.info("按 validation %s 提前停止；未计算 test 指标。", args.monitor)
            break
    if not (run_dir / "best_checkpoint.pt").exists():
        raise RuntimeError("尚无最佳 checkpoint，不能导出最终结果。")
    best = torch_load(run_dir / "best_checkpoint.pt", device)
    model.load_state_dict(best["model_state_dict"])
    return best


def external_datasets(manifest, bundle, args, run_dir):
    from IterDataset import prepare_external_dataset, audit_external_read_overlap
    base = Path(manifest).expanduser().resolve().parent
    with open(manifest, "r", encoding="utf-8-sig", newline="") as stream:
        reader = csv.DictReader(stream)
        if not {"name", "features", "metadata"}.issubset(reader.fieldnames or []):
            raise ValueError("external manifest 需要 name,features,metadata 列。")
        rows = list(reader)
    names = set()
    result = []
    for row in rows:
        requested_strand = row.get("strand", "").strip().lower()
        if requested_strand not in ("", "pos", "neg"):
            raise ValueError("external manifest 的可选 strand 列仅接受 pos、neg 或留空。")
        if requested_strand and requested_strand != args.strand:
            logging.getLogger("binary_training").info("跳过外部集 %s：manifest strand=%s，当前模型=%s",
                                                      row["name"], requested_strand, args.strand)
            continue
        name = row["name"]
        if not name or name in names or not all(c.isalnum() or c in "_-" for c in name):
            raise ValueError("external name 必须唯一且仅含字母、数字、下划线和短横线。")
        names.add(name)
        def resolve(value):
            path = Path(value).expanduser()
            return str(path.resolve() if path.is_absolute() else (base / path).resolve())
        def boolean(key):
            value = row.get(key, "").strip().lower()
            if value not in ("", "0", "false", "no", "1", "true", "yes"):
                raise ValueError("external " + key + " 必须是布尔值。")
            return value in ("1", "true", "yes")
        if not row["metadata"].strip():
            raise ValueError("外部集必须提供可追溯 metadata，以核查 read/template/run 独立性。")
        prepared = prepare_external_dataset(
            resolve(row["features"]), resolve(row["metadata"]), cache_dir=args.cache_dir,
            seq_len=6, raw_signal_len=args.raw_signal_len, dwell_divisor=args.dwell_divisor,
            quality_divisor=args.quality_divisor)
        dataset = prepared["dataset"]
        expected_strand = "+" if args.strand == "pos" else "-"
        observed_strands = {}
        for index in range(len(dataset)):
            observed = str(metadata_for(dataset, index).get("bam_strand", "")).strip()
            observed_strands[observed] = observed_strands.get(observed, 0) + 1
        if set(observed_strands) != {expected_strand}:
            raise ValueError("外部集 {} 的链方向与模型不相容：model={} 需要 bam_strand={!r}，"
                             "metadata 实际计数={}；未知或混合方向不能作为相容证据。".format(
                                 name, args.strand, expected_strand, observed_strands))
        audit = audit_external_read_overlap(dataset, bundle, reference_splits=("train", "val"))
        directory = run_dir / ("external_" + name)
        directory.mkdir(exist_ok=True)
        audit.update(input_fingerprints=prepared["input_fingerprints"], manifest_path=prepared["manifest_path"],
                     model_strand=args.strand, expected_bam_strand=expected_strand,
                     observed_bam_strand_counts=observed_strands, manifest_strand=requested_strand or "unspecified")
        attempt_path = directory / ("audit_attempt_" + datetime.now().strftime("%Y%m%d_%H%M%S_%f") + ".json")
        if not audit["is_read_held_out_from_reference_splits"]:
            write_json(attempt_path, audit)
            raise ValueError("外部集 {} 与 train/val 有 read 重叠：{}".format(name, audit["read_overlap_counts"]))
        for key, requirement in (("template_group", "require_template_disjoint"), ("run_id", "require_run_disjoint")):
            known_external, unknown_external = identity_set(dataset, key)
            known_reference, unknown_reference = set(), 0
            for split in ("train", "val"):
                known, unknown = identity_set(bundle["datasets"][split], key)
                known_reference.update(known)
                unknown_reference += unknown
            overlap = sorted(known_external & known_reference)
            evidence_complete = unknown_external == 0 and unknown_reference == 0
            audit[key] = {"known_external": sorted(known_external), "known_train_val": sorted(known_reference),
                          "unknown_external_rows": unknown_external, "unknown_train_val_rows": unknown_reference,
                          "overlap": overlap, "is_disjoint": not overlap if evidence_complete else None,
                          "required": boolean(requirement)}
            if boolean(requirement) and (not evidence_complete or overlap):
                write_json(attempt_path, audit)
                raise ValueError("外部集 {} 的 {} 独立性{}：{}".format(
                    name, key, "证据不足" if not evidence_complete else "不满足", audit[key]))
        write_json(attempt_path, audit)
        result.append(("external_" + name, dataset, audit))
    return result


def identity_set(dataset, key):
    known, unknown = set(), 0
    for index in range(len(dataset)):
        value = str(metadata_for(dataset, index).get(key, "")).strip()
        if value.lower() in ("", "unknown", "na", "n/a", "none", "null"):
            unknown += 1
        else:
            known.add(value)
    return known, unknown


def content_fingerprints(value):
    """允许文件迁移；相同内容的 SHA256 必须一致，路径和mtime不决定样本身份。"""
    if isinstance(value, dict):
        if "sha256" in value:
            return {"sha256": value["sha256"], "size": value.get("size")}
        return {key: content_fingerprints(item) for key, item in value.items()}
    if isinstance(value, (tuple, list)):
        return [content_fingerprints(item) for item in value]
    return value


def export_dataset_identity(dataset):
    return {"input_fingerprints": content_fingerprints(dataset.input_fingerprints),
            "preprocessing": json_safe(dataset.audit["preprocessing"])}


def resolve_existing(args, provided):
    """评价/恢复使用已保存结构与设置，只允许显式的路径或运行参数迁移。"""
    if not (args.mode == "evaluate" or args.resume):
        return args, None
    if args.output_dir:
        directory = Path(args.output_dir).expanduser().resolve()
    elif args.checkpoint:
        directory = Path(args.checkpoint).expanduser().resolve().parent
    elif args.resume and args.resume != "auto":
        directory = Path(args.resume).expanduser().resolve().parent
    else:
        raise ValueError("evaluate/resume 需要 --output_dir 或 checkpoint 路径。")
    config_path = directory / "resolved_config.json"
    saved = load_json(config_path)
    merged = dict(saved)
    allowed = {"mode", "output_dir", "checkpoint", "resume", "device", "amp", "workers", "batch_size",
               "cache_dir", "bootstrap", "no_plots", "no_embeddings", "embedding_dtype", "no_attention",
               "external_manifest", "force_evaluate", "cpu_threads", "epochs", "defer_test"}
    allowed.update({"no_epoch_logits", "save_epoch_predictions"})
    if args.mode == "evaluate":
        allowed.update({"data_dir", "train_data", "val_data", "test_data", "train_meta", "val_meta", "test_meta"})
    for key, value in vars(args).items():
        if key in provided or key in ("mode", "resume"):
            if key not in allowed and saved.get(key) != value:
                raise ValueError("恢复/评价不能更改已冻结设置：" + key)
            merged[key] = value
    if args.mode == "evaluate" and "data_dir" in provided:
        for split in ("train", "val", "test"):
            if split + "_data" not in provided:
                merged[split + "_data"] = None
            if split + "_meta" not in provided:
                merged[split + "_meta"] = None
    merged["output_dir"] = str(directory)
    return argparse.Namespace(**{key: merged.get(key, value) for key, value in vars(args).items()}), saved


def main():
    global ACTIVE_RUN_DIR, ACTIVE_MODE
    provided = {value.split("=", 1)[0][2:] for value in sys.argv[1:] if value.startswith("--")}
    args, prior_config = resolve_existing(parser().parse_args(), provided)
    if min(args.epochs, args.patience, args.batch_size, args.accum_steps, args.raw_signal_len) < 1 or args.workers < 0:
        raise ValueError("epoch/patience/batch/accum_steps/raw_signal_len 必须正数，workers 非负。")
    if args.lr <= 0 or args.dwell_divisor <= 0 or args.quality_divisor <= 0 or not 0 <= args.label_smoothing < 1:
        raise ValueError("学习率及缩放除数必须正数，label_smoothing 在 [0,1)。")
    if args.target_index != 3 or args.clip_grad <= 0:
        raise ValueError("当前固定目标位点为第4位，target_index 必须3；clip_grad 必须正数。")
    configure_paths(args)
    if args.external_manifest and args.mode != "evaluate":
        raise ValueError("外部测试请在 --mode evaluate 的冻结模型阶段执行。")
    if args.output_dir:
        run_dir = Path(args.output_dir).expanduser().resolve()
    else:
        run_dir = Path("binary_training_runs") / (datetime.now().strftime("%Y%m%d_%H%M%S_%f") + "_" + args.model_type + "_" + args.strand + "_seed" + str(args.seed))
        run_dir = run_dir.resolve()
    if args.mode == "train" and not args.resume and run_dir.exists():
        allowed = {"queue_requested_config.json", "runner_stdout.log", "runner.log", "stdout_train.log", "stdout_evaluate.log"}
        if any(path.name not in allowed for path in run_dir.iterdir()):
            raise ValueError("运行目录已有训练产物，请换目录或使用 --resume；不会覆盖：" + str(run_dir))
    run_dir.mkdir(parents=True, exist_ok=True)
    old_test_started = (run_dir / "test").exists() and any((run_dir / "test").iterdir())
    if args.resume and (old_test_started or ((run_dir / "results.json").exists()
            and load_json(run_dir / "results.json").get("test_evaluated"))):
        raise ValueError("该 run 已完成冻结 test 评价，不允许继续训练并反复使用同一 test；请使用训练前的独立方案。")
    ACTIVE_RUN_DIR, ACTIVE_MODE = run_dir, args.mode
    args.output_dir = str(run_dir)
    log = setup_logging(run_dir)
    device = select_device(args)
    set_seed(args.seed)
    started = time.perf_counter()
    log.info("trainer=%s mode=%s model=%s seed=%d device=%s", VERSION, args.mode, args.model_type, args.seed, device)
    audit_directory = run_dir if args.mode == "train" else run_dir / "evaluation_audit"
    bundle = build_data_bundle(args.train_data, args.val_data, args.test_data,
        train_meta=args.train_meta, val_meta=args.val_meta, test_meta=args.test_meta, cache_dir=args.cache_dir,
        seq_len=6, raw_signal_len=args.raw_signal_len, dwell_divisor=args.dwell_divisor,
        quality_divisor=args.quality_divisor, seed=args.seed, output_dir=str(audit_directory))
    datasets = bundle["datasets"]
    expected_bam_strand = "+" if args.strand == "pos" else "-"
    for split, dataset in datasets.items():
        observed = dataset.audit.get("strand_counts", {})
        if set(observed) != {expected_bam_strand}:
            raise ValueError("内部 {} 的 metadata 方向与 --strand={} 不相容：需要全 {!r}，实际计数={}。".format(
                split, args.strand, expected_bam_strand, observed))
    config = {**vars(args), "trainer_version": VERSION, "feature_definition": FEATURE_DEFINITION,
              "decision_threshold": 0.5,
              "class_names": {"0": "canonical", "1": args.positive_name or ("X" if args.strand == "pos" else "Y")},
              "model_kwargs": model_options(args), "input_fingerprints": json_safe(bundle.get("input_fingerprints")),
              "data_manifest_paths": json_safe(bundle.get("manifest_paths")), "created_at": timestamp()}
    if args.mode == "train" and not args.resume:
        config["source_snapshot"] = source_snapshot(run_dir)
        write_json(run_dir / "environment.json", environment(device))
    else:
        config["source_snapshot"] = prior_config["source_snapshot"]
        if args.resume and current_source_hashes() != prior_config["source_snapshot"]["files_sha256"]:
            raise ValueError("Resume 发现源码已改变，拒绝假称完全恢复。")
        if content_fingerprints(prior_config["input_fingerprints"]) != content_fingerprints(config["input_fingerprints"]):
            raise ValueError("恢复/评价的 train/val/test 内容 SHA256 与冻结数据不同；新测试数据请用 external_manifest。")
        if args.mode == "evaluate":
            original = prior_config["source_snapshot"]["files_sha256"]
            for filename in ("model_benchmark.py", "IterDataset.py", "config_19.py"):
                if sha256(Path(__file__).resolve().parent / filename) != original.get(filename):
                    raise ValueError("冻结评价的模型/数据源码与训练快照不一致：{}；请恢复原快照后评价。".format(filename))
    model = build_model(args.model_type, num_classes=2, seq_len=6, raw_signal_len=args.raw_signal_len,
                        **config["model_kwargs"]).to(device)
    write_json(audit_directory / "data_audit.json", bundle.get("audit"))
    if args.mode == "train":
        write_json(run_dir / "resolved_config.json", config)
        write_json(run_dir / "model_config.json", model.model_config)
        (run_dir / "architecture.txt").write_text(str(model) + "\n", encoding="utf-8")
        best = train_model(model, datasets, args, device, run_dir, config, log)
        checkpoint_path = run_dir / "best_checkpoint.pt"
        checkpoint_hash = sha256(checkpoint_path)
        for split in ("train", "val", "test"):
            old_dir = run_dir / split
            marker = old_dir / "export_complete.json"
            if marker.exists() and load_json(marker)["checkpoint_sha256"] != checkpoint_hash:
                old_hash = load_json(marker)["checkpoint_sha256"][:16]
                archive = run_dir / "previous_exports" / old_hash
                archive.mkdir(parents=True, exist_ok=True)
                if (archive / split).exists():
                    raise ValueError("旧预测归档路径已存在，拒绝覆盖：" + str(archive / split))
                shutil.move(str(old_dir), str(archive / split))
        val_dir = run_dir / "val"
        labels, logits, seconds = infer_arrays(model, datasets["val"], args, device, val_dir)
        calibration = fit_temperature(logits, labels)
        calibration.update(checkpoint_sha256=checkpoint_hash, fitted_at=timestamp())
        write_json(run_dir / "calibration.json", calibration)
        val_info = finish_export(datasets["val"], args, "val", val_dir, labels, logits,
                                 calibration["temperature"], checkpoint_hash, seconds)
        infos = {"val": val_info, "train": export_split(model, datasets["train"], args, device, "train", run_dir,
                                                        calibration["temperature"], checkpoint_hash)}
        result = {"status": "train_complete", "model_type": args.model_type, "seed": args.seed,
                  "strand": args.strand, "feature_scheme": args.feature_scheme, "best_epoch": best["best_epoch"],
                  "litemamba_variant": args.litemamba_variant, "readout": args.readout,
                  "class_names": config["class_names"],
                  "best_val_metrics": best["history"][best["best_epoch"] - 1]["val"],
                  "checkpoints": {"best": str(checkpoint_path), "last": str(run_dir / "last_checkpoint.pt"),
                                  "state_dict": str(run_dir / "best_state_dict.pt")},
                  "metrics_by_split": {key: value["metrics"] for key, value in infos.items()},
                  "exports_by_split": infos, "created_at": timestamp(), "test_evaluated": False,
                  "provenance": {"trainer_version": VERSION, "input_fingerprints": config["input_fingerprints"],
                                 "source_snapshot": config["source_snapshot"], "checkpoint_sha256": checkpoint_hash},
                  "runtime": {"training_and_train_val_export_seconds": time.perf_counter() - started}}
        write_json(run_dir / "benchmark.json", benchmark_model(model, datasets["val"], args, device))
        write_json(run_dir / "results.json", result)
    else:
        checkpoint_path = Path(args.checkpoint or run_dir / "best_checkpoint.pt").expanduser().resolve()
        best = torch_load(checkpoint_path, device)
        model.load_state_dict(best["model_state_dict"])
        checkpoint_hash = sha256(checkpoint_path)
        calibration = load_json(run_dir / "calibration.json")
        if calibration["checkpoint_sha256"] != checkpoint_hash:
            raise ValueError("calibration 与 checkpoint 不一致，不能在 test 上重新拟合温度。")
        result = load_json(run_dir / "results.json")
        if result["provenance"]["checkpoint_sha256"] != checkpoint_hash:
            raise ValueError("results 与 checkpoint 不一致。")
    if args.mode == "evaluate" or not args.defer_test:
        test_info = export_split(model, datasets["test"], args, device, "test", run_dir,
                                 calibration["temperature"], checkpoint_hash)
        result["metrics_by_split"]["test"] = test_info["metrics"]
        result["exports_by_split"]["test"] = test_info
        result["status"], result["test_evaluated"] = "complete", True
        result["runtime"]["test_inference_seconds"] = test_info["inference_seconds"]
        result["runtime"]["test_export_and_analysis_seconds"] = test_info["export_and_analysis_seconds"]
    if args.external_manifest:
        if args.mode != "evaluate":
            raise ValueError("外部测试请在 --mode evaluate 的冻结模型阶段执行。")
        for name, dataset, audit in external_datasets(args.external_manifest, bundle, args, run_dir):
            info = export_split(model, dataset, args, device, name, run_dir, calibration["temperature"], checkpoint_hash)
            write_json(run_dir / name / "external_audit.json", audit)
            result["metrics_by_split"][name], result["exports_by_split"][name] = info["metrics"], info
    result["updated_at"] = timestamp()
    result["runtime"]["last_command_seconds"] = time.perf_counter() - started
    write_json(run_dir / "results.json", result)
    log.info("完成 status=%s；结果 %s", result["status"], run_dir / "results.json")


if __name__ == "__main__":
    try:
        main()
    except Exception as error:
        logging.getLogger("binary_training").exception("运行失败；保留现有 checkpoint 和日志用于审计。")
        if ACTIVE_RUN_DIR is not None:
            path = ACTIVE_RUN_DIR / "results.json"
            result = load_json(path) if path.exists() else {}
            if ACTIVE_MODE == "evaluate" and result.get("status") in ("train_complete", "complete"):
                result["last_evaluation_error"] = {"message": str(error), "at": timestamp()}
            else:
                result.update(status="failed", error=str(error), updated_at=timestamp())
            write_json(path, result)
        raise
