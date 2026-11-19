#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Strict, reproducible input preparation for the Figure 2 binary benchmark.

The original 11-column feature CSV is read together with its row-aligned TSV.
Signals are parsed once into disk-backed float32 arrays, never accumulated in
RAM. The fourth reference base is masked before merging the input channels.
All splits use the same deterministic signal rectification. Randomness is
restricted to the training sample order through EpochShuffleSampler.
"""

from __future__ import annotations

import argparse
import contextlib
import csv
import hashlib
import json
import math
import os
from pathlib import Path
import re
import shutil
import sqlite3
import tempfile
import time
from collections import Counter, defaultdict
from decimal import Decimal, InvalidOperation
from itertools import zip_longest
from typing import Any, Iterator, Mapping, Optional, Sequence

import numpy as np
import torch
from torch.utils.data import DataLoader, Dataset, Sampler


CACHE_VERSION = "binary-read-held-out-v1.5"
FEATURE_COLUMNS = (
    "kmer", "mean", "std", "median", "dwell", "quality", "mismatch",
    "insertion", "deletion", "signal", "label",
)
REQUIRED_METADATA = {"read_id", "chrom", "bam_strand", "kmer", "label"}
BASES = {"A": 0, "C": 1, "T": 2, "G": 3, "N": 4}
UNKNOWN = "unknown"
UNKNOWN_VALUES = {"", "unknown", "na", "n/a", "none", "null", "nan"}


class DataValidationError(ValueError):
    """A malformed row or incompatible split prevents a valid experiment."""


def _write_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as stream:
        json.dump(value, stream, ensure_ascii=False, indent=2, sort_keys=True)
        stream.write("\n")


def _fingerprint(path: Path) -> dict[str, Any]:
    """Hash bytes, counting physical lines without loading complete rows."""
    before = path.stat()
    digest, n_newlines, last_byte = hashlib.sha256(), 0, b""
    with path.open("rb") as stream:
        while True:
            block = stream.read(8 * 1024 * 1024)
            if not block:
                break
            digest.update(block)
            n_newlines += block.count(b"\n")
            last_byte = block[-1:]
    after = path.stat()
    if (before.st_size, before.st_mtime_ns) != (after.st_size, after.st_mtime_ns):
        raise DataValidationError(f"Input changed while hashing: {path}")
    return {
        "path": str(path), "sha256": digest.hexdigest(), "size": after.st_size,
        "mtime_ns": after.st_mtime_ns,
        "physical_lines": n_newlines + int(bool(last_byte) and last_byte != b"\n"),
    }


def _resolve_metadata(feature_csv: Path, metadata_tsv: Optional[str]) -> Path:
    if metadata_tsv:
        candidate = Path(metadata_tsv).expanduser().resolve()
        if not candidate.is_file():
            raise FileNotFoundError(f"Metadata file does not exist: {candidate}")
        return candidate
    candidates = [
        feature_csv.with_name(feature_csv.stem + "_meta.tsv"),
        Path(str(feature_csv) + ".meta.tsv"),
        feature_csv.with_suffix(".meta.tsv"),
    ]
    existing = list(dict.fromkeys(p for p in candidates if p.is_file()))
    if len(existing) == 1:
        return existing[0].resolve()
    if len(existing) > 1:
        raise DataValidationError(
            f"Multiple metadata files found for {feature_csv}; specify one explicitly: {existing}"
        )
    raise FileNotFoundError(
        f"Row-aligned metadata is required for {feature_csv}. Expected "
        f"{candidates[0]}; pass the metadata path explicitly if named differently."
    )


def _label(value: Any, where: str, allow_legacy: bool = False) -> int:
    try:
        parsed = Decimal(str(value).strip())
    except (InvalidOperation, ValueError, TypeError) as exc:
        raise DataValidationError(f"{where}: invalid binary label {value!r}") from exc
    allowed = (0, 1, 2) if allow_legacy else (0, 1)
    if not parsed.is_finite() or parsed not in allowed:
        raise DataValidationError(f"{where}: label must be one of {allowed}, got {value!r}")
    return int(parsed)


def _external_binary_label(label: int, chrom: str, strand: str, where: str) -> int:
    """Only the documented old XNA negative-strand label 2 is remapped."""
    family = re.match(r"^(PC|DNA|XNA)(?:[_-]?\d|$)", chrom, re.I)
    family = family.group(1).upper() if family else None
    if family in ("PC", "DNA"):
        if label != 0:
            raise DataValidationError(f"{where}: PC/DNA controls must carry label 0")
        return 0
    if family == "XNA":
        if label == 1 or (label == 2 and strand == "-"):
            return 1
        raise DataValidationError(f"{where}: XNA requires label 1, or documented negative-strand legacy label 2")
    if label == 2:
        raise DataValidationError(f"{where}: label 2 is only supported for XNA references on strand -")
    return label


def _sequence(value: Any, seq_len: int, where: str) -> str:
    seq = str(value).strip().upper()
    if len(seq) != seq_len or any(
        base not in BASES and not (position == 3 and base in "XY")
        for position, base in enumerate(seq)
    ):
        raise DataValidationError(
            f"{where}: expected {seq_len} ATCGN bases, allowing X/Y only at target index 3; got {value!r}"
        )
    return seq


def normalize_template_pair(chrom: str, pair_id: Optional[str] = None) -> str:
    """PC01, DNA1 and XNA1 share pair_1; no run identity is inferred."""
    explicit = str(pair_id or "").strip()
    if explicit.lower() in UNKNOWN_VALUES:
        explicit = ""
    candidate = explicit or str(chrom).strip()
    match = re.match(r"^(?:PC|DNA|XNA)[_-]?0*(\d+)(?:$|[^0-9])", candidate, re.I)
    if match:
        return f"pair_{int(match.group(1))}"
    if explicit:
        numeric = re.fullmatch(r"(?:pair[_-]?)?0*(\d+)", explicit, re.I)
        return f"pair_{int(numeric.group(1))}" if numeric else explicit
    return UNKNOWN


def _scalar_values(value: str, seq_len: int, where: str) -> np.ndarray:
    tokens = value.split("|")
    if len(tokens) != seq_len or any(not token.strip() for token in tokens):
        raise DataValidationError(f"{where}: expected {seq_len} nonempty scalar values")
    try:
        out = np.asarray([float(token) for token in tokens], dtype=np.float32)
    except (ValueError, OverflowError) as exc:
        raise DataValidationError(f"{where}: invalid numeric scalar") from exc
    if not np.isfinite(out).all():
        raise DataValidationError(f"{where}: all scalar features must be finite float32 values")
    return out


def rectify_signal(signal: str, seq_len: int, raw_signal_len: int, where: str = "signal") -> np.ndarray:
    """Symmetric zero padding or ordered, deterministic uniform subsampling."""
    segments = signal.split("|")
    if len(segments) != seq_len:
        raise DataValidationError(f"{where}: expected {seq_len} signal segments")
    output = np.zeros((seq_len, raw_signal_len), dtype=np.float32)
    for base_index, segment in enumerate(segments):
        tokens = segment.split("*")
        if any(not token.strip() for token in tokens):
            raise DataValidationError(f"{where}: base {base_index + 1} has an empty signal segment/value")
        try:
            values = np.asarray([float(token) for token in tokens], dtype=np.float32)
        except (ValueError, OverflowError) as exc:
            raise DataValidationError(f"{where}: invalid signal at base {base_index + 1}") from exc
        if not np.isfinite(values).all():
            raise DataValidationError(f"{where}: all signal values must be finite float32 values")
        values = np.around(values.astype(np.float64), decimals=4).astype(np.float32)
        length = len(values)
        if length <= raw_signal_len:
            left = (raw_signal_len - length) // 2
            output[base_index, left:left + length] = values
        else:
            # Indices are distinct because the source is longer than the target.
            indices = np.floor(np.linspace(0, length - 1, raw_signal_len)).astype(np.int64)
            output[base_index] = values[indices]
    return output


def _feature_rows(path: Path) -> Iterator[tuple[int, list[str]]]:
    with path.open("r", newline="", encoding="utf-8-sig") as stream:
        reader = csv.reader(stream)
        for ordinal, row in enumerate(reader, 1):
            if ordinal == 1 and row and row[0].strip().lower() == "kmer":
                if tuple(item.strip().lower() for item in row) != FEATURE_COLUMNS:
                    raise DataValidationError(f"{path}: unexpected 11-column header {row!r}")
                continue
            if len(row) != len(FEATURE_COLUMNS):
                raise DataValidationError(f"{path}:{reader.line_num}: expected 11 columns, got {len(row)}")
            if any("\n" in field or "\r" in field for field in row):
                raise DataValidationError(f"{path}:{reader.line_num}: multiline feature fields are unsupported")
            yield reader.line_num, row


def _metadata_rows(path: Path) -> Iterator[tuple[int, dict[str, str]]]:
    with path.open("r", newline="", encoding="utf-8-sig") as stream:
        reader = csv.DictReader(stream, delimiter="\t")
        if not reader.fieldnames:
            raise DataValidationError(f"{path}: metadata header is missing")
        reader.fieldnames = [name.strip() for name in reader.fieldnames]
        if len(set(reader.fieldnames)) != len(reader.fieldnames):
            raise DataValidationError(f"{path}: metadata contains duplicate column names")
        missing = REQUIRED_METADATA - set(reader.fieldnames)
        if missing:
            raise DataValidationError(f"{path}: required metadata columns missing: {sorted(missing)}")
        for row in reader:
            if None in row or any(value is None for value in row.values()):
                raise DataValidationError(f"{path}:{reader.line_num}: malformed metadata row")
            if any("\n" in value or "\r" in value for value in row.values()):
                raise DataValidationError(f"{path}:{reader.line_num}: multiline metadata fields are unsupported")
            yield reader.line_num, row


@contextlib.contextmanager
def _cache_lock(lock_path: Path):
    """Serialize first-time preprocessing when several models start together."""
    lock_path.parent.mkdir(parents=True, exist_ok=True)
    with lock_path.open("a+") as lock:
        try:
            import fcntl
        except ImportError:
            # The training server is Linux. Atomic directory publication below
            # still prevents a partially written cache on other platforms.
            yield
        else:
            fcntl.flock(lock.fileno(), fcntl.LOCK_EX)
            try:
                yield
            finally:
                fcntl.flock(lock.fileno(), fcntl.LOCK_UN)


class _MetadataView:
    def __init__(self, dataset: "CachedFeatureDataset"):
        self.dataset = dataset

    def __len__(self) -> int:
        return len(self.dataset)

    def __getitem__(self, index: int) -> dict[str, Any]:
        return self.dataset.get_metadata(index)

    def __iter__(self):
        return self.dataset.iter_metadata()


class CachedFeatureDataset(Dataset):
    """Map-style dataset with lazily opened disk arrays and indexed metadata."""

    def __init__(self, cache_path: str, global_index_offset: int = 0):
        self.cache_path = Path(cache_path)
        with (self.cache_path / "audit.json").open(encoding="utf-8") as stream:
            self.audit = json.load(stream)
        self.size = int(self.audit["sample_count"])
        self.seq_len = int(self.audit["preprocessing"]["seq_len"])
        self.raw_signal_len = int(self.audit["preprocessing"]["raw_signal_len"])
        self.global_index_offset = int(global_index_offset)
        self.manifest_path = str(self.cache_path / "manifest.jsonl")
        self.input_fingerprints = self.audit["input_fingerprints"]
        self._merged = self._raw = self._labels = self._offsets = self._metadata_handle = None
        self._owner_pid = os.getpid()

    @property
    def metadata(self) -> _MetadataView:
        return _MetadataView(self)

    def __len__(self) -> int:
        return self.size

    def _open_arrays(self) -> None:
        self._ensure_process_local()
        if self._merged is None:
            self._merged = np.load(self.cache_path / "merged.npy", mmap_mode="r", allow_pickle=False)
            self._raw = np.load(self.cache_path / "raw.npy", mmap_mode="r", allow_pickle=False)
            self._labels = np.load(self.cache_path / "labels.npy", mmap_mode="r", allow_pickle=False)

    def _ensure_process_local(self) -> None:
        if self._owner_pid != os.getpid():
            # Forked workers must not share the metadata seek pointer inherited
            # from a parent that has already inspected an example row.
            self.close()
            self._owner_pid = os.getpid()

    def __getitem__(self, index: int) -> dict[str, Any]:
        index = int(index)
        if index < 0:
            index += self.size
        if not 0 <= index < self.size:
            raise IndexError(index)
        self._open_arrays()
        # Copy only one sample so tensors are writable and DataLoader workers
        # never mutate a shared read-only memory map.
        merged = torch.from_numpy(np.array(self._merged[index], copy=True))
        raw = torch.from_numpy(np.array(self._raw[index], copy=True))
        return {
            "inputs": (merged[:, :5], merged, raw),
            "label": torch.tensor(int(self._labels[index]), dtype=torch.long),
            "index": index, "global_index": self.global_index_offset + index,
        }

    def get_metadata(self, index: int) -> dict[str, Any]:
        self._ensure_process_local()
        index = int(index)
        if index < 0:
            index += self.size
        if not 0 <= index < self.size:
            raise IndexError(index)
        if self._offsets is None:
            self._offsets = np.load(self.cache_path / "metadata_offsets.npy", mmap_mode="r", allow_pickle=False)
        if self._metadata_handle is None:
            self._metadata_handle = open(self.manifest_path, "rb")
        self._metadata_handle.seek(int(self._offsets[index]))
        value = json.loads(self._metadata_handle.readline())
        value["global_index"] = self.global_index_offset + index
        return value

    def iter_metadata(self) -> Iterator[dict[str, Any]]:
        with open(self.manifest_path, "r", encoding="utf-8") as stream:
            for index, line in enumerate(stream):
                value = json.loads(line)
                value["global_index"] = self.global_index_offset + index
                yield value

    def __getstate__(self) -> dict[str, Any]:
        state = self.__dict__.copy()
        for name in ("_merged", "_raw", "_labels", "_offsets", "_metadata_handle"):
            state[name] = None
        state["_owner_pid"] = None
        return state

    def close(self) -> None:
        if self._metadata_handle is not None:
            self._metadata_handle.close()
            self._metadata_handle = None
        for name in ("_merged", "_raw", "_labels", "_offsets"):
            value = getattr(self, name)
            if value is not None and hasattr(value, "_mmap"):
                value._mmap.close()
            setattr(self, name, None)

    def __del__(self):
        try:
            self.close()
        except Exception:
            pass


def _cache_valid(path: Path, config: dict[str, Any], fingerprints: dict[str, Any]) -> bool:
    try:
        with (path / "audit.json").open(encoding="utf-8") as stream:
            audit = json.load(stream)
        if audit["preprocessing"] != config or audit["input_fingerprints"] != fingerprints:
            return False
        n = int(audit["sample_count"])
        shapes = {
            "merged.npy": ((n, config["seq_len"], 13), np.dtype("float32")),
            "raw.npy": ((n, config["seq_len"], config["raw_signal_len"]), np.dtype("float32")),
            "labels.npy": ((n,), np.dtype("int64")),
            "metadata_offsets.npy": ((n,), np.dtype("int64")),
        }
        for name, (shape, dtype) in shapes.items():
            arr = np.load(path / name, mmap_mode="r", allow_pickle=False)
            valid = arr.shape == shape and arr.dtype == dtype
            arr._mmap.close()
            if not valid:
                return False
        return (path / "manifest.jsonl").is_file()
    except (OSError, ValueError, KeyError, TypeError):
        return False


def prepare_dataset(
    feature_csv: str,
    metadata_tsv: Optional[str] = None,
    *,
    cache_dir: Optional[str] = None,
    seq_len: int = 6,
    raw_signal_len: int = 30,
    dwell_divisor: float = 60.0,
    quality_divisor: float = 50.0,
    mask_index: int = 3,
    split_name: str = "external",
    global_index_offset: int = 0,
    log_every: int = 50000,
    allow_legacy_strand_labels: bool = False,
    **unused: Any,
) -> dict[str, Any]:
    """Validate a CSV/TSV pair, prepare/reuse a cache, and return its dataset."""
    if unused:
        raise TypeError(f"Unknown prepare_dataset options: {sorted(unused)}")
    if seq_len != 6 or mask_index != 3:
        raise ValueError("This revision defines a 6mer target at the fourth base (mask_index=3).")
    if raw_signal_len < 1:
        raise ValueError("raw_signal_len must be positive")
    for name, value in (("dwell_divisor", dwell_divisor), ("quality_divisor", quality_divisor)):
        if not math.isfinite(value) or value <= 0:
            raise ValueError(f"{name} must be a finite positive constant")
    feature_path = Path(feature_csv).expanduser().resolve()
    if not feature_path.is_file():
        raise FileNotFoundError(feature_path)
    metadata_path = _resolve_metadata(feature_path, metadata_tsv)
    if feature_path == metadata_path:
        raise DataValidationError("Feature CSV and metadata TSV must be different files")
    print(f"[data] SHA256 and validation source: {split_name}: {feature_path}", flush=True)
    fingerprints = {"feature": _fingerprint(feature_path), "metadata": _fingerprint(metadata_path)}
    config = {
        "cache_version": CACHE_VERSION, "seq_len": seq_len,
        "raw_signal_len": raw_signal_len, "base_order": list(BASES),
        "dwell_divisor": float(dwell_divisor), "quality_divisor": float(quality_divisor),
        "mask_index": mask_index, "mask_base": "N", "signal_round_decimals": 4,
        "signal_rectification": "symmetric_zero_pad_or_floor_linspace_indices",
        "label_values": [0, 1], "feature_columns": list(FEATURE_COLUMNS),
        "input_label_policy": "binary_or_documented_XNA_negative_strand_2" if allow_legacy_strand_labels else "strict_binary_0_1",
    }
    identity = {"preprocessing": config, "input_fingerprints": fingerprints}
    key = hashlib.sha256(json.dumps(identity, sort_keys=True).encode()).hexdigest()
    root = Path(cache_dir or "binary_feature_cache").expanduser().resolve()
    target = root / key
    root.mkdir(parents=True, exist_ok=True)
    with _cache_lock(root / (key + ".lock")):
        if not _cache_valid(target, config, fingerprints):
            tmp = Path(tempfile.mkdtemp(prefix=key + ".building-", dir=root))
            try:
                _build_cache(tmp, feature_path, metadata_path, config, fingerprints, log_every)
                if target.exists():
                    # An incompatible cache is retained for diagnostics instead
                    # of modifying a source or trusting stale cache metadata.
                    backup = root / (key + ".invalid-" + str(time.time_ns()))
                    target.rename(backup)
                tmp.rename(target)
            except BaseException:
                shutil.rmtree(tmp, ignore_errors=True)
                raise
        else:
            print(f"[data] Reusing validated cache: {target.name[:16]}", flush=True)
    dataset = CachedFeatureDataset(str(target), global_index_offset)
    return {
        "dataset": dataset, "audit": dataset.audit,
        "manifest_path": dataset.manifest_path,
        "input_fingerprints": fingerprints, "metadata": dataset.metadata,
        "cache_path": str(target),
    }


def _build_cache(
    target: Path, feature_path: Path, metadata_path: Path,
    config: dict[str, Any], fingerprints: dict[str, Any], log_every: int,
) -> None:
    with feature_path.open("r", newline="", encoding="utf-8-sig") as stream:
        first = next(csv.reader(stream), [])
    has_header = bool(first and first[0].strip().lower() == "kmer")
    n = fingerprints["feature"]["physical_lines"] - int(has_header)
    meta_n = fingerprints["metadata"]["physical_lines"] - 1
    if n < 1 or meta_n != n:
        raise DataValidationError(
            f"CSV/TSV row count mismatch or empty dataset: features={n}, metadata={meta_n}; "
            "files must be from the same run and remain row aligned"
        )
    seq_len, raw_len = config["seq_len"], config["raw_signal_len"]
    merged = np.lib.format.open_memmap(target / "merged.npy", mode="w+", dtype=np.float32, shape=(n, seq_len, 13))
    raw = np.lib.format.open_memmap(target / "raw.npy", mode="w+", dtype=np.float32, shape=(n, seq_len, raw_len))
    labels = np.lib.format.open_memmap(target / "labels.npy", mode="w+", dtype=np.int64, shape=(n,))
    offsets = np.lib.format.open_memmap(target / "metadata_offsets.npy", mode="w+", dtype=np.int64, shape=(n,))
    class_counts, input_label_counts, strands, mask_replacements = Counter(), Counter(), Counter(), 0
    chroms, pairs, runs = defaultdict(set), defaultdict(set), defaultdict(set)
    unknown_runs, original_available = 0, 0
    sentinel = object()
    count = 0
    try:
        with (target / "manifest.jsonl").open("wb") as manifest:
            iterator = zip_longest(_feature_rows(feature_path), _metadata_rows(metadata_path), fillvalue=sentinel)
            for index, (feature_item, meta_item) in enumerate(iterator):
                if feature_item is sentinel or meta_item is sentinel:
                    raise DataValidationError(f"CSV/metadata row counts differ at sample {index}")
                feature_line, row = feature_item
                meta_line, metadata = meta_item
                where = f"{feature_path}:{feature_line} (metadata line {meta_line})"
                if index >= n:
                    raise DataValidationError(f"{where}: unexpected multiline/extra row")
                seq = _sequence(row[0], seq_len, where)
                meta_seq = _sequence(metadata["kmer"], seq_len, where + " metadata.kmer")
                allow_legacy = config["input_label_policy"] != "strict_binary_0_1"
                input_label = _label(row[-1], where, allow_legacy)
                input_meta_label = _label(metadata["label"], where + " metadata.label", allow_legacy)
                if seq != meta_seq or input_label != input_meta_label:
                    raise DataValidationError(f"{where}: feature/metadata kmer or label mismatch; row alignment is unreliable")
                read_id = metadata["read_id"].strip()
                chrom, strand = metadata["chrom"].strip(), metadata["bam_strand"].strip()
                if not read_id or not chrom or strand not in ("+", "-"):
                    raise DataValidationError(f"{where}: nonempty read_id/chrom and bam_strand +/- are required")
                # Known reference names must agree with supervised identity;
                # arbitrary reference names retain valid binary labels.
                label = _external_binary_label(input_label, chrom, strand, where)
                masked = seq[:3] + "N" + seq[4:]
                mask_replacements += int(seq[3] != "N")
                onehot = np.zeros((seq_len, 5), dtype=np.float32)
                onehot[np.arange(seq_len), [BASES[base] for base in masked]] = 1.0
                scalars = np.stack([
                    _scalar_values(row[col], seq_len, where + " " + FEATURE_COLUMNS[col])
                    for col in range(1, 9)
                ], axis=1)
                scalars[:, 3] /= config["dwell_divisor"]
                scalars[:, 4] /= config["quality_divisor"]
                if not np.isfinite(scalars).all():
                    raise DataValidationError(f"{where}: fixed feature scaling overflowed float32")
                merged[index, :, :5], merged[index, :, 5:] = onehot, scalars
                raw[index] = rectify_signal(row[9], seq_len, raw_len, where + " signal")
                labels[index] = label
                pair = normalize_template_pair(chrom, metadata.get("pair_id") or metadata.get("template_group"))
                run_id = (metadata.get("run_id") or UNKNOWN).strip()
                source_dataset = (metadata.get("source_dataset") or UNKNOWN).strip()
                if run_id.lower() in UNKNOWN_VALUES:
                    run_id = UNKNOWN
                if source_dataset.lower() in UNKNOWN_VALUES:
                    source_dataset = UNKNOWN
                original = metadata.get("original_kmer") or seq
                original = _sequence(original, seq_len, where + " original_kmer")
                original_known = original[3] != "N"
                original_available += int(original_known)
                record = dict(metadata)
                record.update({
                    "sample_index": index,
                    "sample_id": hashlib.sha256((fingerprints["feature"]["sha256"] + ":" + str(index)).encode()).hexdigest()[:24],
                    "feature_line": feature_line, "metadata_line": meta_line,
                    "read_id": read_id, "chrom": chrom, "pair_id": pair,
                    "template_group": pair, "bam_strand": strand,
                    "kmer": masked, "masked_motif": masked, "original_kmer": original,
                    "original_kmer_unmasked_available": original_known,
                    "original_label": metadata.get("original_label") if metadata.get("original_label", "") != "" else input_label,
                    "input_label": input_label, "input_meta_label": input_meta_label,
                    "label": label, "run_id": run_id, "source_dataset": source_dataset,
                    "source_file": metadata.get("source_file") or str(feature_path),
                    "source_row": metadata.get("source_row") or feature_line,
                })
                offsets[index] = manifest.tell()
                manifest.write((json.dumps(record, ensure_ascii=False, separators=(",", ":")) + "\n").encode("utf-8"))
                class_counts[label] += 1
                input_label_counts[input_label] += 1
                strands[strand] += 1
                chroms[label].add(chrom)
                if pair != UNKNOWN:
                    pairs[label].add(pair)
                if run_id.lower() not in UNKNOWN_VALUES:
                    runs[label].add(run_id)
                else:
                    unknown_runs += 1
                count += 1
                if log_every > 0 and count % log_every == 0:
                    print(f"[data] Parsed {count:,}/{n:,} rows: {feature_path.name}", flush=True)
        if count != n:
            raise DataValidationError(f"CSV/metadata row count mismatch: expected {n}, parsed {count}")
        for arr in (merged, raw, labels, offsets):
            arr.flush()
        # Detect a source being modified during preprocessing without another
        # full multigigabyte read. Every fresh invocation hashes both sources.
        for name, path in (("feature", feature_path), ("metadata", metadata_path)):
            stat = path.stat()
            if (stat.st_size, stat.st_mtime_ns) != (fingerprints[name]["size"], fingerprints[name]["mtime_ns"]):
                raise DataValidationError(f"Input changed during preprocessing: {path}")
        class_coverage = {}
        for label in (0, 1):
            class_coverage[str(label)] = {
                "samples": class_counts[label], "independent_chrom_count": len(chroms[label]),
                "chroms": sorted(chroms[label]), "independent_template_pair_count": len(pairs[label]),
                "template_pairs": sorted(pairs[label]), "known_run_count": len(runs[label]),
                "known_run_ids": sorted(runs[label]),
            }
        _write_json(target / "audit.json", {
            "cache_version": CACHE_VERSION, "sample_count": count,
            "preprocessing": config, "input_fingerprints": fingerprints,
            "class_counts": {str(k): class_counts[k] for k in (0, 1)},
            "input_label_counts": {str(k): v for k, v in input_label_counts.items()},
            "class_coverage": class_coverage, "strand_counts": dict(strands),
            "fourth_position_replacements": mask_replacements,
            "all_model_input_fourth_bases_masked": True,
            "unmasked_original_kmer_available_rows": original_available,
            "unknown_run_rows": unknown_runs,
            "run_independence_claim": "not_established; file names are never treated as run identifiers",
            "feature_scaling": "fixed constants; no fitted statistics use validation or test",
            "evaluation_preprocessing": "deterministic; no random sampling or skipped rows",
        })
        print(f"[data] Cache ready: {count:,} samples, classes={dict(class_counts)}", flush=True)
    finally:
        for arr in (merged, raw, labels, offsets):
            if hasattr(arr, "_mmap"):
                arr._mmap.close()


def _audit_splits(datasets: Mapping[str, CachedFeatureDataset], directory: Path) -> dict[str, Any]:
    directory.mkdir(parents=True, exist_ok=True)
    database = directory / "read_membership.sqlite"
    connection = sqlite3.connect(str(database))
    connection.execute("PRAGMA journal_mode=WAL")
    connection.execute("DROP TABLE IF EXISTS read_membership")
    connection.execute("CREATE TABLE read_membership (read_id TEXT NOT NULL, split TEXT NOT NULL, windows INTEGER NOT NULL, PRIMARY KEY(read_id, split))")
    template_sets, run_sets = {}, {}
    split_summary = {}
    try:
        for split, dataset in datasets.items():
            templates, runs, batch = set(), set(), []
            for metadata in dataset.iter_metadata():
                batch.append((metadata["read_id"], split, 1))
                if metadata["pair_id"] != UNKNOWN:
                    templates.add(metadata["pair_id"])
                run = metadata["run_id"]
                if run.lower() not in UNKNOWN_VALUES:
                    runs.add(run)
                if len(batch) >= 5000:
                    connection.executemany("INSERT INTO read_membership VALUES (?,?,?) ON CONFLICT(read_id,split) DO UPDATE SET windows=windows+1", batch)
                    batch.clear()
            if batch:
                connection.executemany("INSERT INTO read_membership VALUES (?,?,?) ON CONFLICT(read_id,split) DO UPDATE SET windows=windows+1", batch)
            connection.commit()
            template_sets[split], run_sets[split] = templates, runs
            read_stats = connection.execute(
                "SELECT COUNT(*), SUM(CASE WHEN windows>1 THEN 1 ELSE 0 END), COALESCE(MAX(windows),0) FROM read_membership WHERE split=?", (split,),
            ).fetchone()
            split_summary[split] = {
                "sample_count": len(dataset), "unique_read_count": read_stats[0],
                "reads_with_multiple_windows": read_stats[1] or 0,
                "max_windows_per_read": read_stats[2],
                "class_counts": dataset.audit["class_counts"],
                "class_coverage": dataset.audit["class_coverage"],
                "unknown_run_rows": dataset.audit["unknown_run_rows"],
            }
        overlap, template_overlap, known_run_overlap = {}, {}, {}
        split_names = list(datasets)
        for left_index, left in enumerate(split_names):
            for right in split_names[left_index + 1:]:
                key = f"{left}__{right}"
                overlap[key] = connection.execute(
                    "SELECT COUNT(*) FROM read_membership a JOIN read_membership b USING(read_id) WHERE a.split=? AND b.split=?", (left, right),
                ).fetchone()[0]
                template_overlap[key] = sorted(template_sets[left] & template_sets[right])
                known_run_overlap[key] = sorted(run_sets[left] & run_sets[right])
        preview = connection.execute(
            "SELECT read_id, GROUP_CONCAT(split) FROM read_membership GROUP BY read_id HAVING COUNT(*)>1 LIMIT 20"
        ).fetchall()
        missing_classes = [
            split for split, dataset in datasets.items()
            if any(dataset.audit["class_counts"][str(label)] == 0 for label in (0, 1))
        ]
        audit = {
            "status": "READ_OVERLAP_REJECTED" if any(overlap.values()) else "MISSING_CLASS_REJECTED" if missing_classes else "PASS",
            "evaluation_scope": "read-held-out; template-held-out and run-held-out are not implied",
            "split_summary": split_summary, "read_overlap_counts": overlap,
            "read_overlap_examples": [dict(read_id=row[0], splits=row[1]) for row in preview],
            "template_pair_overlap": template_overlap, "known_run_overlap": known_run_overlap,
            "run_independence_established": False,
            "splits_missing_a_binary_class": missing_classes,
            "run_identity_note": "Missing run_id is unknown. File names/source file paths do not establish sequencing run identity.",
            "read_membership_database": str(database),
        }
        _write_json(directory / "data_audit.json", audit)
        if any(overlap.values()):
            raise DataValidationError(
                f"Same read_id occurs across splits: {overlap}. Training rejected. "
                f"Details: {directory / 'data_audit.json'}"
            )
        if missing_classes:
            raise DataValidationError(f"Both binary classes are required in each split; missing classes in {missing_classes}")
        return audit
    finally:
        connection.close()


def build_data_bundle(
    train_csv: str, val_csv: str, test_csv: str,
    train_meta: Optional[str] = None, val_meta: Optional[str] = None,
    test_meta: Optional[str] = None, cache_dir: Optional[str] = None,
    seq_len: int = 6, raw_signal_len: int = 30,
    dwell_divisor: float = 60.0, quality_divisor: float = 50.0,
    seed: int = 42, output_dir: Optional[str] = None,
    **kwargs: Any,
) -> dict[str, Any]:
    """Prepare three fixed datasets and reject read overlap before training.

    Test is read here only for input validation, fingerprints and preprocessing;
    this function computes no model score and performs no parameter selection.
    """
    prepared, datasets = {}, {}
    offset = 0
    for split, feature, meta in (
        ("train", train_csv, train_meta), ("val", val_csv, val_meta), ("test", test_csv, test_meta),
    ):
        prepared[split] = prepare_dataset(
            feature, meta, cache_dir=cache_dir, seq_len=seq_len,
            raw_signal_len=raw_signal_len, dwell_divisor=dwell_divisor,
            quality_divisor=quality_divisor, split_name=split,
            global_index_offset=offset, **kwargs,
        )
        datasets[split] = prepared[split]["dataset"]
        offset += len(datasets[split])
    bundle_identity = {name: prepared[name]["cache_path"] for name in prepared}
    key = hashlib.sha256(json.dumps(bundle_identity, sort_keys=True).encode()).hexdigest()[:24]
    directory = Path(output_dir).expanduser().resolve() if output_dir else Path(cache_dir or "binary_feature_cache").resolve() / "audits" / key
    with _cache_lock(directory / "data_audit.lock"):
        audit = _audit_splits(datasets, directory)
    manifests = {split: prepared[split]["manifest_path"] for split in prepared}
    fingerprints = {split: prepared[split]["input_fingerprints"] for split in prepared}
    data_manifest = {
        "cache_version": CACHE_VERSION, "seed_for_training_order": int(seed),
        "manifest_paths": manifests, "input_fingerprints": fingerprints,
        "global_index_offsets": {name: dataset.global_index_offset for name, dataset in datasets.items()},
        "preprocessing": datasets["train"].audit["preprocessing"],
        "audit_path": str(directory / "data_audit.json"),
        "evaluation_scope": audit["evaluation_scope"],
    }
    _write_json(directory / "data_manifest.json", data_manifest)
    return {
        "datasets": datasets, "audit": audit, "manifest_paths": manifests,
        "input_fingerprints": fingerprints,
        "metadata": {split: prepared[split]["metadata"] for split in prepared},
        "audit_path": str(directory / "data_audit.json"),
        "data_manifest_path": str(directory / "data_manifest.json"),
        "read_membership_database": audit["read_membership_database"],
        "seed": int(seed),
    }


def prepare_external_dataset(
    feature_csv: str, metadata_tsv: Optional[str] = None, *,
    cache_dir: Optional[str] = None, seq_len: int = 6,
    raw_signal_len: int = 30, dwell_divisor: float = 60.0,
    quality_divisor: float = 50.0, **kwargs: Any,
) -> dict[str, Any]:
    """Apply exactly the same deterministic preprocessing to external data."""
    kwargs.setdefault("allow_legacy_strand_labels", True)
    return prepare_dataset(
        feature_csv, metadata_tsv, cache_dir=cache_dir, seq_len=seq_len,
        raw_signal_len=raw_signal_len, dwell_divisor=dwell_divisor,
        quality_divisor=quality_divisor, split_name="external", **kwargs,
    )


def audit_external_read_overlap(
    external_dataset: CachedFeatureDataset,
    reference_bundle: Mapping[str, Any],
    reference_splits: Sequence[str] = ("train", "val"),
) -> dict[str, Any]:
    """Report overlap without silently presenting external data as held out."""
    database = reference_bundle.get("read_membership_database")
    if not database:
        raise ValueError("Reference bundle does not contain a read_membership_database")
    connection = sqlite3.connect(str(database))
    overlap = {split: set() for split in reference_splits}
    external_reads = set()
    try:
        for metadata in external_dataset.iter_metadata():
            read_id = metadata["read_id"]
            if read_id in external_reads:
                continue
            external_reads.add(read_id)
            matches = connection.execute("SELECT split FROM read_membership WHERE read_id=?", (read_id,)).fetchall()
            for (split,) in matches:
                if split in overlap:
                    overlap[split].add(read_id)
    finally:
        connection.close()
    return {
        "external_unique_read_count": len(external_reads),
        "read_overlap_counts": {split: len(values) for split, values in overlap.items()},
        "read_overlap_examples": {split: sorted(values)[:20] for split, values in overlap.items()},
        "is_read_held_out_from_reference_splits": not any(overlap.values()),
        "reference_splits": list(reference_splits),
    }


class EpochShuffleSampler(Sampler[int]):
    """Independent of global RNG and DataLoader worker count."""

    def __init__(self, data_source: Dataset, seed: int = 42, epoch: int = 0):
        self.data_source, self.seed, self.epoch = data_source, int(seed), int(epoch)

    def set_epoch(self, epoch: int) -> None:
        self.epoch = int(epoch)

    def __len__(self) -> int:
        return len(self.data_source)

    def __iter__(self):
        generator = torch.Generator()
        generator.manual_seed((self.seed + self.epoch) % (2**63 - 1))
        yield from torch.randperm(len(self.data_source), generator=generator).tolist()


def seed_worker(worker_id: int) -> None:
    # Preprocessing is deterministic; this also seeds any explicit future
    # train-only augmentation without relying on the process launch method.
    import random
    worker_seed = torch.initial_seed() % (2**32)
    random.seed(worker_seed)
    np.random.seed(worker_seed)


def make_data_loader(
    dataset: CachedFeatureDataset, *, batch_size: int = 256,
    num_workers: int = 0, train: bool = False, seed: int = 42,
    epoch: int = 0, pin_memory: bool = False,
    persistent_workers: bool = False, prefetch_factor: int = 2,
    drop_last: bool = False,
) -> DataLoader:
    """Sequential eval; seeded training permutation; no byte-boundary sharding."""
    if not train and drop_last:
        raise ValueError("Evaluation must not drop samples: drop_last must be False")
    sampler = EpochShuffleSampler(dataset, seed, epoch) if train else None
    generator = torch.Generator().manual_seed(int(seed))
    options = dict(
        dataset=dataset, batch_size=batch_size, sampler=sampler, shuffle=False,
        num_workers=num_workers, pin_memory=pin_memory,
        persistent_workers=bool(persistent_workers and num_workers > 0),
        worker_init_fn=seed_worker, generator=generator, drop_last=drop_last,
    )
    if num_workers > 0:
        options["prefetch_factor"] = prefetch_factor
    return DataLoader(**options)


def _main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--train_csv", required=True)
    parser.add_argument("--val_csv", required=True)
    parser.add_argument("--test_csv", required=True)
    parser.add_argument("--train_meta")
    parser.add_argument("--val_meta")
    parser.add_argument("--test_meta")
    parser.add_argument("--cache_dir", default="binary_feature_cache")
    parser.add_argument("--output_dir", default="data_audit")
    parser.add_argument("--raw_signal_len", type=int, default=30)
    parser.add_argument("--dwell_divisor", type=float, default=60.0)
    parser.add_argument("--quality_divisor", type=float, default=50.0)
    args = parser.parse_args()
    bundle = build_data_bundle(**vars(args))
    print(json.dumps(bundle["audit"], ensure_ascii=False, indent=2))


if __name__ == "__main__":
    _main()
