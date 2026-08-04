#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Train one benchmark model on the 11-column XNA/PC 6-mer CSV format.

Expected columns
----------------
kmer, mean, std, median, dwell, quality, mismatch, insertion, deletion,
signal, label

Per-base scalar fields use ``|`` as the separator. The raw-signal field uses
``|`` between bases and ``*`` between measurements within one base.

The script is self-contained apart from ``model_benchmark_all.py`` and does not
require the original private ``config_*.py`` or ``IterDataset.py`` files.
"""

from __future__ import annotations

import argparse
import csv
import gc
import logging
import os
import random
import shutil
import subprocess
import sys
import time
from contextlib import nullcontext
from pathlib import Path
from typing import Iterator, List, Optional, Sequence, Tuple

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.optim.lr_scheduler import ReduceLROnPlateau
from torch.utils.data import DataLoader, IterableDataset, get_worker_info

from model_benchmark_all import build_model

os.environ.setdefault("PYTORCH_CUDA_ALLOC_CONF", "max_split_size_mb:128")

BASE_MAP = {"A": 0, "C": 1, "T": 2, "G": 3, "N": 4}
BASE_DIM = 5
SCALAR_FEATURE_DIM = 8
MERGED_FEATURE_DIM = BASE_DIM + SCALAR_FEATURE_DIM
EXPECTED_COLUMNS = 11
MODEL_NAMES = ("bigru", "transformer", "convmixer", "litemamba")


class LineIterableDataset(IterableDataset):
    """Stream a large text/CSV file without loading it fully into memory."""

    def __init__(self, file_path: Path) -> None:
        super().__init__()
        self.file_path = Path(file_path)

    @staticmethod
    def _iter_binary_range(
        handle,
        start: int,
        end: Optional[int],
    ) -> Iterator[str]:
        handle.seek(start)
        if start != 0:
            handle.readline()  # discard the partial line at this byte offset

        while True:
            position = handle.tell()
            if end is not None and position >= end:
                break
            raw_line = handle.readline()
            if not raw_line:
                break
            yield raw_line.decode("utf-8").rstrip("\r\n")

    def __iter__(self) -> Iterator[str]:
        file_size = self.file_path.stat().st_size
        worker = get_worker_info()

        with self.file_path.open("rb", buffering=1 << 20) as handle:
            if worker is None:
                yield from self._iter_binary_range(handle, 0, None)
                return

            start = (file_size * worker.id) // worker.num_workers
            end = (
                (file_size * (worker.id + 1)) // worker.num_workers
                if worker.id + 1 < worker.num_workers
                else None
            )
            yield from self._iter_binary_range(handle, start, end)


def parse_device(value: str) -> torch.device:
    """Resolve ``auto``, CPU or CUDA device strings."""

    if value == "auto":
        return torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    device = torch.device(value)
    if device.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError(
            f"Requested device {value!r}, but CUDA is not available"
        )
    return device


def configure_runtime(device: torch.device, deterministic: bool) -> None:
    """Configure optional CUDA acceleration and deterministic execution."""

    if device.type != "cuda":
        return

    torch.backends.cuda.matmul.allow_tf32 = not deterministic
    torch.backends.cudnn.allow_tf32 = not deterministic
    torch.backends.cudnn.benchmark = not deterministic
    torch.backends.cudnn.deterministic = deterministic

    try:
        torch.set_float32_matmul_precision("highest" if deterministic else "high")
    except Exception:
        pass


def set_random_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)


def create_logger(log_path: Path) -> logging.Logger:
    log_path.parent.mkdir(parents=True, exist_ok=True)
    logger = logging.getLogger(f"xna_trainer.{log_path.resolve()}")
    logger.setLevel(logging.INFO)
    logger.propagate = False

    if logger.handlers:
        return logger

    formatter = logging.Formatter(
        "[%(asctime)s] %(message)s",
        "%Y-%m-%d %H:%M:%S",
    )
    stream_handler = logging.StreamHandler(sys.stdout)
    file_handler = logging.FileHandler(log_path, encoding="utf-8")
    stream_handler.setFormatter(formatter)
    file_handler.setFormatter(formatter)
    logger.addHandler(stream_handler)
    logger.addHandler(file_handler)
    return logger


def encode_seq_block(seq: str, mer_len: int) -> Optional[np.ndarray]:
    """Convert an ATCGN sequence to a ``[mer_len, 5]`` one-hot matrix."""

    sequence = seq.strip().upper()
    if len(sequence) != mer_len:
        return None

    output = np.zeros((mer_len, BASE_DIM), dtype=np.float32)
    for index, base in enumerate(sequence):
        encoded = BASE_MAP.get(base)
        if encoded is None:
            return None
        output[index, encoded] = 1.0
    return output


def parse_pipe_array(value: str, mer_len: int) -> Optional[np.ndarray]:
    array = np.fromstring(value, sep="|", dtype=np.float32)
    if array.size != mer_len or not np.all(np.isfinite(array)):
        return None
    return array


def rectify_signal_segment(
    segment: str,
    target_length: int,
) -> np.ndarray:
    values = (
        np.fromstring(segment, sep="*", dtype=np.float32)
        if segment
        else np.empty(0, dtype=np.float32)
    )
    values = np.around(values, decimals=4)

    if values.size < target_length:
        total_padding = target_length - values.size
        left_padding = total_padding // 2
        right_padding = total_padding - left_padding
        values = np.pad(
            values,
            (left_padding, right_padding),
            mode="constant",
            constant_values=0.0,
        )
    elif values.size > target_length:
        selected_indices = sorted(
            random.sample(range(values.size), target_length)
        )
        values = values[selected_indices]

    return values.astype(np.float32, copy=False)


def parse_raw_signal(
    signal_value: str,
    mer_len: int,
    target_length: int,
) -> Optional[np.ndarray]:
    segments = signal_value.split("|")
    if len(segments) != mer_len:
        return None

    output = np.empty((mer_len, target_length), dtype=np.float32)
    for index, segment in enumerate(segments):
        output[index] = rectify_signal_segment(segment, target_length)

    if not np.all(np.isfinite(output)):
        return None
    return output


def build_merged_features(
    seq_onehot: np.ndarray,
    mean: np.ndarray,
    std: np.ndarray,
    median: np.ndarray,
    dwell: np.ndarray,
    quality: np.ndarray,
    mismatch: np.ndarray,
    insertion: np.ndarray,
    deletion: np.ndarray,
    dwell_divisor: float,
    quality_divisor: float,
) -> np.ndarray:
    scalar_features = np.stack(
        [
            mean,
            std,
            median,
            dwell / dwell_divisor,
            quality / quality_divisor,
            mismatch,
            insertion,
            deletion,
        ],
        axis=1,
    ).astype(np.float32, copy=False)

    return np.concatenate(
        [seq_onehot.astype(np.float32, copy=False), scalar_features],
        axis=1,
    ).astype(np.float32, copy=False)


def parse_batch_lines(
    lines: Sequence[str],
    mer_len: int,
    raw_signal_len: int,
    dwell_divisor: float,
    quality_divisor: float,
    num_classes: int,
) -> Tuple[Tuple[torch.Tensor, torch.Tensor, torch.Tensor], torch.Tensor]:
    seq_onehot_list: List[np.ndarray] = []
    merged_feature_list: List[np.ndarray] = []
    raw_signal_list: List[np.ndarray] = []
    labels: List[int] = []

    for line in lines:
        if not line:
            continue

        parts = line.split(",")
        if len(parts) != EXPECTED_COLUMNS:
            continue
        if parts[0].strip().lower() == "kmer":
            continue

        (
            seq_value,
            mean_value,
            std_value,
            median_value,
            dwell_value,
            quality_value,
            mismatch_value,
            insertion_value,
            deletion_value,
            signal_value,
            label_value,
        ) = parts

        seq_onehot = encode_seq_block(seq_value, mer_len)
        scalar_arrays = [
            parse_pipe_array(value, mer_len)
            for value in (
                mean_value,
                std_value,
                median_value,
                dwell_value,
                quality_value,
                mismatch_value,
                insertion_value,
                deletion_value,
            )
        ]
        raw_signal = parse_raw_signal(
            signal_value,
            mer_len,
            raw_signal_len,
        )

        if seq_onehot is None or raw_signal is None:
            continue
        if any(array is None for array in scalar_arrays):
            continue

        try:
            label = int(float(label_value))
        except (TypeError, ValueError):
            continue
        if label < 0 or label >= num_classes:
            continue

        merged_features = build_merged_features(
            seq_onehot,
            *scalar_arrays,
            dwell_divisor=dwell_divisor,
            quality_divisor=quality_divisor,
        )
        seq_onehot_list.append(seq_onehot)
        merged_feature_list.append(merged_features)
        raw_signal_list.append(raw_signal)
        labels.append(label)

    if not labels:
        empty_inputs = (
            torch.empty(0, mer_len, BASE_DIM, dtype=torch.float32),
            torch.empty(0, mer_len, MERGED_FEATURE_DIM, dtype=torch.float32),
            torch.empty(0, mer_len, raw_signal_len, dtype=torch.float32),
        )
        return empty_inputs, torch.empty(0, dtype=torch.long)

    inputs = (
        torch.from_numpy(np.stack(seq_onehot_list).astype(np.float32, copy=False)),
        torch.from_numpy(np.stack(merged_feature_list).astype(np.float32, copy=False)),
        torch.from_numpy(np.stack(raw_signal_list).astype(np.float32, copy=False)),
    )
    targets = torch.from_numpy(np.asarray(labels, dtype=np.int64))
    return inputs, targets


class BatchCollator:
    """Picklable collator for Linux/macOS/Windows DataLoader workers."""

    def __init__(
        self,
        mer_len: int,
        raw_signal_len: int,
        dwell_divisor: float,
        quality_divisor: float,
        num_classes: int,
    ) -> None:
        self.mer_len = mer_len
        self.raw_signal_len = raw_signal_len
        self.dwell_divisor = dwell_divisor
        self.quality_divisor = quality_divisor
        self.num_classes = num_classes

    def __call__(self, lines: Sequence[str]):
        return parse_batch_lines(
            lines,
            mer_len=self.mer_len,
            raw_signal_len=self.raw_signal_len,
            dwell_divisor=self.dwell_divisor,
            quality_divisor=self.quality_divisor,
            num_classes=self.num_classes,
        )


def chunk_shuffle(
    input_path: Path,
    output_path: Path,
    chunk_size: int,
    seed: int,
    logger: logging.Logger,
) -> None:
    """Shuffle lines inside large chunks when GNU ``shuf`` is unavailable."""

    start_time = time.time()
    rng = random.Random(seed)
    logger.info("Python chunk shuffle started: chunk_size=%d", chunk_size)

    with input_path.open("r", encoding="utf-8", buffering=1 << 20) as source:
        first_line = source.readline()
        has_header = first_line.lower().startswith("kmer,")
        if not has_header:
            source.seek(0)

        with output_path.open(
            "w",
            encoding="utf-8",
            buffering=1 << 20,
        ) as destination:
            if has_header:
                destination.write(first_line)

            buffer: List[str] = []
            for line in source:
                buffer.append(line)
                if len(buffer) >= chunk_size:
                    rng.shuffle(buffer)
                    destination.writelines(buffer)
                    buffer.clear()

            if buffer:
                rng.shuffle(buffer)
                destination.writelines(buffer)

    logger.info(
        "Python chunk shuffle completed in %.1f seconds",
        time.time() - start_time,
    )


def prepare_shuffled_file(
    input_path: Path,
    disable_shuffle: bool,
    chunk_size: int,
    seed: int,
    logger: logging.Logger,
) -> Path:
    if disable_shuffle:
        logger.info("Training-file shuffle disabled")
        return input_path

    output_path = input_path.with_name(input_path.name + ".shuf")
    if (
        output_path.exists()
        and output_path.stat().st_mtime >= input_path.stat().st_mtime
    ):
        logger.info("Reusing shuffled file: %s", output_path)
        return output_path

    shuf_executable = shutil.which("shuf")
    if shuf_executable is None:
        logger.warning(
            "GNU shuf is unavailable; falling back to within-chunk shuffling"
        )
        chunk_shuffle(
            input_path,
            output_path,
            chunk_size,
            seed,
            logger,
        )
        return output_path

    first_line = ""
    with input_path.open("r", encoding="utf-8", buffering=1 << 20) as source:
        first_line = source.readline()
    has_header = first_line.lower().startswith("kmer,")

    logger.info("GNU shuf started: %s -> %s", input_path, output_path)
    temporary_body = output_path.with_name(output_path.name + ".body.tmp")
    temporary_shuffled = output_path.with_name(output_path.name + ".shuffled.tmp")

    try:
        if has_header:
            with input_path.open(
                "r",
                encoding="utf-8",
                buffering=1 << 20,
            ) as source, temporary_body.open(
                "w",
                encoding="utf-8",
                buffering=1 << 20,
            ) as body:
                source.readline()
                shutil.copyfileobj(source, body, length=1 << 20)
            subprocess.run(
                [shuf_executable, str(temporary_body), "-o", str(temporary_shuffled)],
                check=True,
            )
            with output_path.open("w", encoding="utf-8", buffering=1 << 20) as destination:
                destination.write(first_line)
                with temporary_shuffled.open(
                    "r",
                    encoding="utf-8",
                    buffering=1 << 20,
                ) as shuffled_body:
                    shutil.copyfileobj(shuffled_body, destination, length=1 << 20)
        else:
            subprocess.run(
                [shuf_executable, str(input_path), "-o", str(output_path)],
                check=True,
            )
    except subprocess.CalledProcessError as error:
        logger.warning(
            "GNU shuf failed with code %s; using Python chunk shuffle",
            error.returncode,
        )
        chunk_shuffle(
            input_path,
            output_path,
            chunk_size,
            seed,
            logger,
        )
    finally:
        temporary_body.unlink(missing_ok=True)
        temporary_shuffled.unlink(missing_ok=True)

    return output_path


def seed_worker(worker_id: int) -> None:
    worker_seed = torch.initial_seed() % (2**32)
    np.random.seed(worker_seed + worker_id)
    random.seed(worker_seed + worker_id)


def create_loader(
    path: Path,
    *,
    train: bool,
    batch_size: int,
    workers: int,
    prefetch: int,
    pin_memory: bool,
    persistent_workers: bool,
    disable_shuffle: bool,
    chunk_size: int,
    shuffle_seed: int,
    drop_last: bool,
    collate_fn,
    logger: logging.Logger,
) -> Tuple[DataLoader, Path]:
    data_path = (
        prepare_shuffled_file(
            path,
            disable_shuffle,
            chunk_size,
            shuffle_seed,
            logger,
        )
        if train
        else path
    )

    parameters = {
        "batch_size": batch_size,
        "shuffle": False,
        "drop_last": drop_last if train else False,
        "collate_fn": collate_fn,
        "num_workers": workers,
        "pin_memory": pin_memory,
        "persistent_workers": persistent_workers and workers > 0,
        "worker_init_fn": seed_worker if workers > 0 else None,
    }
    if workers > 0:
        parameters["prefetch_factor"] = max(1, prefetch)

    return DataLoader(LineIterableDataset(data_path), **parameters), data_path


def autocast_context(device: torch.device, enabled: bool):
    if enabled and device.type == "cuda":
        return torch.cuda.amp.autocast()
    return nullcontext()


def move_inputs_to_device(
    inputs: Tuple[torch.Tensor, torch.Tensor, torch.Tensor],
    device: torch.device,
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    return tuple(
        tensor.to(device, non_blocking=True)
        for tensor in inputs
    )


@torch.no_grad()
def evaluate(
    model: nn.Module,
    loader: DataLoader,
    loss_function: nn.Module,
    device: torch.device,
    amp_enabled: bool,
) -> Tuple[float, float, int]:
    model.eval()
    total_loss = 0.0
    total_correct = 0
    total_examples = 0
    batch_count = 0

    for inputs_cpu, targets_cpu in loader:
        if targets_cpu.numel() == 0:
            continue

        inputs = move_inputs_to_device(inputs_cpu, device)
        targets = targets_cpu.to(device, non_blocking=True)
        with autocast_context(device, amp_enabled):
            logits, _ = model(inputs)
            loss = loss_function(logits, targets)

        total_loss += float(loss.item())
        total_correct += int((logits.argmax(dim=1) == targets).sum().item())
        total_examples += int(targets.numel())
        batch_count += 1

    if batch_count == 0 or total_examples == 0:
        raise RuntimeError(
            "Validation produced no valid samples. Check the CSV format, "
            "sequence length and num_classes."
        )

    return (
        total_loss / batch_count,
        total_correct / total_examples,
        total_examples,
    )


def write_metrics_header(metrics_path: Path, append: bool) -> None:
    metrics_path.parent.mkdir(parents=True, exist_ok=True)
    if append and metrics_path.exists() and metrics_path.stat().st_size > 0:
        return

    with metrics_path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.writer(handle)
        writer.writerow(
            [
                "epoch",
                "model_type",
                "train_loss",
                "train_acc",
                "val_loss",
                "val_acc",
                "lr",
                "train_samples",
                "val_samples",
                "epoch_time_sec",
                "timestamp",
            ]
        )


def append_metrics(metrics_path: Path, row: Sequence[object]) -> None:
    with metrics_path.open("a", newline="", encoding="utf-8") as handle:
        writer = csv.writer(handle)
        writer.writerow(row)
        handle.flush()
        try:
            os.fsync(handle.fileno())
        except OSError:
            pass


def unwrap_model(model: nn.Module) -> nn.Module:
    original = getattr(model, "_orig_mod", None)
    return original if isinstance(original, nn.Module) else model


def clean_state_dict_keys(state_dict):
    cleaned = {}
    for key, value in state_dict.items():
        new_key = key
        changed = True
        while changed:
            changed = False
            for prefix in ("module.", "_orig_mod."):
                if new_key.startswith(prefix):
                    new_key = new_key[len(prefix):]
                    changed = True
        cleaned[new_key] = value
    return cleaned


def load_pretrained_weights(
    model: nn.Module,
    checkpoint_path: Path,
    device: torch.device,
) -> None:
    checkpoint = torch.load(checkpoint_path, map_location=device)
    if isinstance(checkpoint, dict):
        for key in ("state_dict", "model_state_dict", "model"):
            nested = checkpoint.get(key)
            if isinstance(nested, dict):
                checkpoint = nested
                break
    if not isinstance(checkpoint, dict):
        raise RuntimeError("The pretrained checkpoint does not contain a state dict")
    model.load_state_dict(clean_state_dict_keys(checkpoint), strict=True)


def train_model(args: argparse.Namespace) -> int:
    train_path = args.train_data.resolve()
    test_path = args.test_data.resolve()
    output_dir = args.output_dir.resolve()

    for path, label in ((train_path, "training"), (test_path, "test")):
        if not path.is_file():
            raise FileNotFoundError(f"The {label} CSV does not exist: {path}")

    output_dir.mkdir(parents=True, exist_ok=True)
    timestamp = time.strftime("%Y%m%d_%H%M%S")
    save_path = (args.save or output_dir / f"{args.model_type}_best.pt").resolve()
    metrics_path = (
        args.metrics_out or output_dir / f"{args.model_type}_metrics.csv"
    ).resolve()
    loss_curve_path = (
        args.loss_curve or output_dir / f"{args.model_type}_loss.png"
    ).resolve()
    log_path = (args.log_path or output_dir / f"{args.model_type}_{timestamp}.log").resolve()

    for path in (save_path, metrics_path, loss_curve_path, log_path):
        path.parent.mkdir(parents=True, exist_ok=True)

    logger = create_logger(log_path)
    device = parse_device(args.device)
    configure_runtime(device, args.deterministic)
    set_random_seed(args.seed)

    if args.num_classes < 2:
        raise ValueError("--num_classes must be at least 2")
    if args.seq_len <= 0 or args.raw_signal_len <= 0:
        raise ValueError("--seq_len and --raw_signal_len must be greater than 0")
    if args.dwell_divisor <= 0 or args.quality_divisor <= 0:
        raise ValueError("Feature scaling divisors must be greater than 0")
    if args.accum_steps <= 0:
        raise ValueError("--accum_steps must be greater than 0")

    logger.info("Model: %s", args.model_type)
    logger.info("Device: %s", device)
    logger.info("Training CSV: %s", train_path)
    logger.info("Test CSV: %s", test_path)
    logger.info("Output directory: %s", output_dir)
    logger.info("Number of classes: %d", args.num_classes)
    logger.info("Sequence length: %d", args.seq_len)
    logger.info("Raw signal length per base: %d", args.raw_signal_len)
    logger.info(
        "Feature scaling: dwell / %.6g, quality / %.6g",
        args.dwell_divisor,
        args.quality_divisor,
    )
    logger.info("Random seed: %d", args.seed)

    collate_fn = BatchCollator(
        mer_len=args.seq_len,
        raw_signal_len=args.raw_signal_len,
        dwell_divisor=args.dwell_divisor,
        quality_divisor=args.quality_divisor,
        num_classes=args.num_classes,
    )

    pin_memory = device.type == "cuda"
    train_loader, actual_train_path = create_loader(
        train_path,
        train=True,
        batch_size=args.batch_size,
        workers=args.workers,
        prefetch=args.prefetch,
        pin_memory=pin_memory,
        persistent_workers=args.persistent_workers,
        disable_shuffle=args.no_shuffle,
        chunk_size=args.chunk_size,
        shuffle_seed=args.seed,
        drop_last=args.drop_last,
        collate_fn=collate_fn,
        logger=logger,
    )
    validation_workers = (
        args.val_workers
        if args.val_workers is not None
        else max(0, args.workers // 2)
    )
    validation_loader, _ = create_loader(
        test_path,
        train=False,
        batch_size=min(args.validation_batch_size, args.batch_size),
        workers=validation_workers,
        prefetch=max(1, args.prefetch // 2),
        pin_memory=pin_memory,
        persistent_workers=args.persistent_workers,
        disable_shuffle=True,
        chunk_size=args.chunk_size,
        shuffle_seed=args.seed,
        drop_last=False,
        collate_fn=collate_fn,
        logger=logger,
    )

    base_model = build_model(
        model_type=args.model_type,
        num_classes=args.num_classes,
        seq_len=args.seq_len,
        raw_signal_len=args.raw_signal_len,
    ).to(device)

    if args.pre_train is not None:
        pretrained_path = args.pre_train.resolve()
        if not pretrained_path.is_file():
            raise FileNotFoundError(
                f"Pretrained checkpoint does not exist: {pretrained_path}"
            )
        logger.info("Loading pretrained weights: %s", pretrained_path)
        load_pretrained_weights(base_model, pretrained_path, device)

    model: nn.Module = base_model
    if args.compile:
        if not hasattr(torch, "compile"):
            raise RuntimeError("--compile requires PyTorch 2.0 or newer")
        logger.info("torch.compile enabled")
        model = torch.compile(base_model, mode=args.compile_mode)

    optimizer = optim.AdamW(
        model.parameters(),
        lr=args.lr,
        weight_decay=args.weight_decay,
    )
    scheduler = ReduceLROnPlateau(
        optimizer,
        mode="min",
        factor=args.lr_factor,
        patience=args.lr_patience,
    )
    loss_function = nn.CrossEntropyLoss(
        label_smoothing=args.label_smoothing
    ).to(device)

    amp_enabled = device.type == "cuda" and not args.no_amp
    try:
        scaler = torch.amp.GradScaler("cuda", enabled=amp_enabled)
    except (AttributeError, TypeError):
        scaler = torch.cuda.amp.GradScaler(enabled=amp_enabled)
    write_metrics_header(metrics_path, append=args.append_metrics)

    best_validation_accuracy = -1.0
    epochs_without_improvement = 0
    training_losses: List[float] = []

    for epoch in range(1, args.epochs + 1):
        epoch_start = time.time()
        model.train()
        optimizer.zero_grad(set_to_none=True)

        running_loss = 0.0
        batch_count = 0
        correct = 0
        sample_count = 0
        pending_accumulation = 0

        for batch_index, (inputs_cpu, targets_cpu) in enumerate(train_loader, 1):
            if targets_cpu.numel() == 0:
                continue

            inputs = move_inputs_to_device(inputs_cpu, device)
            targets = targets_cpu.to(device, non_blocking=True)

            with autocast_context(device, amp_enabled):
                logits, _ = model(inputs)
                unscaled_loss = loss_function(logits, targets)
                loss = unscaled_loss / args.accum_steps

            scaler.scale(loss).backward()
            pending_accumulation += 1

            if pending_accumulation == args.accum_steps:
                scaler.step(optimizer)
                scaler.update()
                optimizer.zero_grad(set_to_none=True)
                pending_accumulation = 0

            running_loss += float(unscaled_loss.item())
            batch_count += 1
            correct += int((logits.detach().argmax(dim=1) == targets).sum().item())
            sample_count += int(targets.numel())

            if batch_index % args.log_every == 0:
                logger.info(
                    "Epoch %d | batch %d | loss %.6f | accuracy %.6f",
                    epoch,
                    batch_index,
                    running_loss / max(batch_count, 1),
                    correct / max(sample_count, 1),
                )

        if pending_accumulation > 0:
            scaler.step(optimizer)
            scaler.update()
            optimizer.zero_grad(set_to_none=True)

        if batch_count == 0 or sample_count == 0:
            raise RuntimeError(
                "Training produced no valid batches. Check the CSV format, "
                "batch size, --drop_last, sequence length and num_classes."
            )

        train_loss = running_loss / batch_count
        train_accuracy = correct / sample_count
        validation_loss, validation_accuracy, validation_samples = evaluate(
            model,
            validation_loader,
            loss_function,
            device,
            amp_enabled,
        )
        scheduler.step(validation_loss)
        current_lr = float(optimizer.param_groups[0]["lr"])
        epoch_duration = time.time() - epoch_start
        training_losses.append(train_loss)

        logger.info(
            "Epoch %03d | train_loss %.6f | train_acc %.6f | "
            "val_loss %.6f | val_acc %.6f | lr %.3e | %.1fs",
            epoch,
            train_loss,
            train_accuracy,
            validation_loss,
            validation_accuracy,
            current_lr,
            epoch_duration,
        )

        append_metrics(
            metrics_path,
            [
                epoch,
                args.model_type,
                f"{train_loss:.8f}",
                f"{train_accuracy:.8f}",
                f"{validation_loss:.8f}",
                f"{validation_accuracy:.8f}",
                f"{current_lr:.10e}",
                sample_count,
                validation_samples,
                f"{epoch_duration:.3f}",
                time.strftime("%Y-%m-%d %H:%M:%S"),
            ],
        )

        if validation_accuracy > best_validation_accuracy:
            best_validation_accuracy = validation_accuracy
            epochs_without_improvement = 0
            torch.save(unwrap_model(model).state_dict(), save_path)
            logger.info(
                "New best validation accuracy %.6f; saved %s",
                best_validation_accuracy,
                save_path,
            )
        else:
            epochs_without_improvement += 1
            logger.info(
                "No validation-accuracy improvement: %d/%d",
                epochs_without_improvement,
                args.patience,
            )
            if epochs_without_improvement >= args.patience:
                logger.info("Early stopping")
                break

        gc.collect()
        if device.type == "cuda":
            torch.cuda.empty_cache()

    plt.figure(figsize=(7, 5))
    plt.plot(range(1, len(training_losses) + 1), training_losses)
    plt.xlabel("Epoch")
    plt.ylabel("Training loss")
    plt.tight_layout()
    plt.savefig(loss_curve_path, dpi=200)
    plt.close()

    logger.info("Training completed")
    logger.info("Best validation accuracy: %.6f", best_validation_accuracy)
    logger.info("Best model: %s", save_path)
    logger.info("Metrics: %s", metrics_path)
    logger.info("Loss curve: %s", loss_curve_path)

    if args.delete_shuffled and actual_train_path != train_path:
        actual_train_path.unlink(missing_ok=True)
        logger.info("Deleted shuffled training file: %s", actual_train_path)

    return 0


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Train one XNA/PC 6-mer benchmark model"
    )
    parser.add_argument("--train_data", type=Path, required=True)
    parser.add_argument("--test_data", type=Path, required=True)
    parser.add_argument("--output_dir", type=Path, required=True)
    parser.add_argument(
        "--model_type",
        choices=MODEL_NAMES,
        default="litemamba",
    )
    parser.add_argument("--num_classes", type=int, required=True)
    parser.add_argument("--seq_len", type=int, default=6)
    parser.add_argument("--raw_signal_len", type=int, default=30)

    parser.add_argument("--save", type=Path)
    parser.add_argument("--metrics_out", type=Path)
    parser.add_argument("--loss_curve", type=Path)
    parser.add_argument("--log_path", type=Path)
    parser.add_argument("--pre_train", type=Path)

    parser.add_argument("--epochs", type=int, default=10000)
    parser.add_argument("--batch_size", type=int, default=256)
    parser.add_argument("--validation_batch_size", type=int, default=256)
    parser.add_argument("--workers", type=int, default=8)
    parser.add_argument("--val_workers", type=int)
    parser.add_argument("--prefetch", type=int, default=2)
    persistent_group = parser.add_mutually_exclusive_group()
    persistent_group.add_argument(
        "--persistent_workers",
        "--persistent-workers",
        dest="persistent_workers",
        action="store_true",
    )
    persistent_group.add_argument(
        "--no_persistent_workers",
        "--no-persistent-workers",
        dest="persistent_workers",
        action="store_false",
    )
    parser.set_defaults(persistent_workers=True)

    drop_last_group = parser.add_mutually_exclusive_group()
    drop_last_group.add_argument(
        "--drop_last",
        "--drop-last",
        dest="drop_last",
        action="store_true",
    )
    drop_last_group.add_argument(
        "--no_drop_last",
        "--no-drop-last",
        dest="drop_last",
        action="store_false",
    )
    parser.set_defaults(drop_last=True)
    parser.add_argument("--accum_steps", type=int, default=1)
    parser.add_argument("--log_every", type=int, default=100)

    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--weight_decay", type=float, default=1e-4)
    parser.add_argument("--label_smoothing", type=float, default=0.0)
    parser.add_argument("--lr_factor", type=float, default=0.5)
    parser.add_argument("--lr_patience", type=int, default=5)
    parser.add_argument("--patience", type=int, default=50)

    parser.add_argument("--dwell_divisor", type=float, default=60.0)
    parser.add_argument("--quality_divisor", type=float, default=50.0)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--device", default="auto")
    parser.add_argument("--deterministic", action="store_true")
    parser.add_argument("--no_amp", action="store_true")

    parser.add_argument("--compile", action="store_true")
    parser.add_argument(
        "--compile_mode",
        choices=("default", "reduce-overhead", "max-autotune"),
        default="default",
    )
    parser.add_argument("--no_shuffle", action="store_true")
    parser.add_argument("--chunk_size", type=int, default=1_000_000)
    parser.add_argument("--delete_shuffled", action="store_true")
    parser.add_argument("--append_metrics", action="store_true")
    return parser


def main() -> int:
    parser = build_parser()
    args = parser.parse_args()
    return train_model(args)


if __name__ == "__main__":
    raise SystemExit(main())
