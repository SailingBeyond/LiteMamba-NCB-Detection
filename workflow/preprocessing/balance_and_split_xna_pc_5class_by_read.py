#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""从原始六类合并特征生成 A/T/C/G/X 与 A/T/C/G/Y 两套五分类数据。

固定原始 read 分区后精确下采样，第四位 N；原标签5在Y任务中映射为4。
仅使用Python标准库；不依赖原六分类脚本或sixclass_io.py。
"""
from __future__ import annotations

import csv

import hashlib

import itertools

import math

from pathlib import Path

import re

import sys

from decimal import Decimal, InvalidOperation

_csv_limit = sys.maxsize

while True:
    try:
        csv.field_size_limit(_csv_limit)
        break
    except OverflowError:
        _csv_limit //= 10

VERSION = "2.1.0"

LABEL_NAMES = {0: "A", 1: "T", 2: "C", 3: "G", 4: "X", 5: "Y"}

FEATURE_HEADER = ["kmer", "mean", "std", "median", "dwell", "quality", "mismatch",
                  "insertion", "deletion", "signal", "label"]

HEADER_ALIASES = [
    {"kmer", "sequence", "seq", "motif"},
    {"mean", "means", "currentmean", "meancurrent", "signalmean"},
    {"std", "stds", "stdev", "currentstd", "signalstd"},
    {"median", "medians", "currentmedian", "signalmedian"},
    {"dwell", "dwells", "dwelltime", "dwelltimes"},
    {"quality", "qualities", "qual", "quals", "qscore", "basequality"},
    {"mismatch", "mismatches", "mismatchrate"},
    {"insertion", "insertions", "insertionrate"},
    {"deletion", "deletions", "deletionrate"},
    {"signal", "signals", "rawsignal", "rawsignals"},
    {"label", "labels", "class", "target"},
]

SPLIT_NAMES = ("train", "validation", "test")

SPLIT_METHOD = "blake2b64-read-split-v1"

BASE_META_HEADER = ["read_id", "chrom", "bam_strand", "kmer", "label", "split",
                    "split_seed", "split_ratios", "split_method", "source_dataset",
                    "run_id", "source_file", "source_row", "original_kmer",
                    "original_label", "merged_input_file", "merged_input_row",
                    "merge_input_role", "read_id_kind", "read_identity_basis",
                    "original_read_id", "reference_identity_known",
                    "positive_source_sha256", "positive_data_record"]

def normalized_header(value):
    return re.sub(r"[^a-z0-9]", "", value.strip().lower())

def exact_label(value, context="label"):
    try:
        number = Decimal(str(value).strip())
    except (ValueError, InvalidOperation) as exc:
        raise ValueError("{}: label不是有效整数: {!r}".format(context, value)) from exc
    if not number.is_finite() or number != number.to_integral_value() or not 0 <= number <= 5:
        raise ValueError("{}: label必须是精确整数0..5，实际={!r}".format(context, value))
    return int(number)

def mask_kmer(value, context="kmer"):
    sequence = str(value).strip().upper()
    if len(sequence) != 6 or any(base not in "ATCGN" for i, base in enumerate(sequence) if i != 3) or sequence[3:4] not in "ATCGNXY":
        raise ValueError("{}: 必须是6mer；侧翼仅ATCGN，第四位允许ATCGNXY，实际={!r}".format(context, value))
    return sequence[:3] + "N" + sequence[4:]

def parse_split_ratios(text):
    try:
        values = tuple(float(part.strip()) for part in str(text).split(","))
    except (ValueError, TypeError) as exc:
        raise ValueError("split_ratios必须为三个正数，例如7,2,1") from exc
    if len(values) != 3 or any(not math.isfinite(x) or x <= 0 for x in values):
        raise ValueError("split_ratios必须为三个有限正数")
    largest = max(values)
    scaled = tuple(x / largest for x in values)
    total = sum(scaled)
    result = tuple(x / total for x in scaled)
    if min(result) <= 0:
        raise ValueError("split_ratios数值差异过大")
    return result

def assign_read_split(read_id, split_seed=20260429, split_ratios=None):
    if not isinstance(read_id, str) or not read_id or read_id != read_id.strip():
        raise ValueError("read_id必须为无首尾空白的非空字符串")
    split_ratios = split_ratios or parse_split_ratios("7,2,1")
    if isinstance(split_ratios, str):
        split_ratios = parse_split_ratios(split_ratios)
    payload = str(int(split_seed)).encode("ascii") + b"\x00" + read_id.encode("utf-8")
    value = int.from_bytes(hashlib.blake2b(payload, digest_size=8).digest(), "big")
    scale = 1 << 64
    if value < int(split_ratios[0] * scale):
        return "train"
    if value < int((split_ratios[0] + split_ratios[1]) * scale):
        return "validation"
    return "test"

def canonical_split(value):
    value = str(value).strip().lower()
    return {"val": "validation", "valid": "validation"}.get(value, value)

def resolve_metadata(feature_path, explicit=None):
    feature = Path(feature_path).expanduser().resolve()
    if explicit:
        path = Path(explicit).expanduser().resolve()
        if not path.is_file():
            raise ValueError("metadata不存在: {}".format(path))
        return path
    candidates = [feature.with_name(feature.stem + "_meta.tsv"), Path(str(feature) + ".meta.tsv"),
                  feature.with_suffix(".meta.tsv")]
    existing = list(dict.fromkeys(path.resolve() for path in candidates if path.is_file()))
    if len(existing) != 1:
        raise ValueError("{}: metadata自动匹配到{}个文件{}。已查找: {}。请显式提供同次提取、逐行对应的metadata路径；阴性用--negative_meta，阳性默认CSV行代理模式不调用此查找。".format(
            feature, len(existing), ", ".join(map(str, existing)), ", ".join(map(str, candidates))))
    return existing[0]

def metadata_header(path):
    with open(path, "r", encoding="utf-8-sig", newline="") as stream:
        header = next(csv.reader(stream, delimiter="\t", strict=True), None)
    if not header:
        raise ValueError("metadata为空: {}".format(path))
    names = [value.strip() for value in header]
    if len(names) != len(set(names)) or any(not name for name in names):
        raise ValueError("metadata表头为空或重复: {}".format(path))
    required = {"read_id", "chrom", "label"}
    if not required.issubset(names) or not {"kmer", "output_kmer", "6mer", "masked_6mer", "masked_kmer"}.intersection(names):
        raise ValueError("{}: metadata必须含read_id, chrom, label及kmer/output_kmer/masked_6mer等序列列；实际={}".format(path, names))
    return names

def _feature_rows(path):
    with open(path, "r", encoding="utf-8-sig", newline="", buffering=8 * 1024 * 1024) as stream:
        reader = csv.reader(stream, strict=True)
        first = next(reader, None)
        if first is None:
            raise ValueError("特征文件为空: {}".format(path))
        first_cell = normalized_header(first[0]) if first else ""
        has_header = first_cell in HEADER_ALIASES[0]
        if has_header:
            if len(first) != 11 or any(normalized_header(value) not in choices for value, choices in zip(first, HEADER_ALIASES)):
                raise ValueError("{}: 11列表头顺序不兼容，实际={}".format(path, first))
        # csv.reader.line_num counts physical source lines, including any quoted
        # embedded newline. Keep that location distinct from the record ordinal.
        previous_end = reader.line_num
        if not has_header:
            if len(first) != 11:
                raise ValueError("{}:1: 特征须恰好11列，实际{}列；空行也不允许跳过".format(path, len(first)))
            yield 1, first
        for row in reader:
            line_no = previous_end + 1
            previous_end = reader.line_num
            if len(row) != 11:
                raise ValueError("{}:{}: 特征须恰好11列，实际{}列；空行也不允许跳过".format(path, line_no, len(row)))
            yield line_no, row

def _validate_features(row, context):
    masked = mask_kmer(row[0], context)
    label = exact_label(row[10], context)
    for idx in range(1, 9):
        tokens = row[idx].split("|")
        if len(tokens) != 6 or any(not token.strip() for token in tokens):
            raise ValueError("{}: {}必须含6个数值".format(context, FEATURE_HEADER[idx]))
        try:
            if any(not math.isfinite(float(token)) for token in tokens):
                raise ValueError("非有限值")
        except ValueError as exc:
            raise ValueError("{}: {}含无效数值".format(context, FEATURE_HEADER[idx])) from exc
    segments = row[9].split("|")
    if len(segments) != 6 or any(not segment.strip() for segment in segments):
        raise ValueError("{}: signal必须为6个非空信号段".format(context))
    for segment in segments:
        # 实际提取器每段以*分隔，外层六段才以|分隔；兼容逗号/空格。
        # 不改变数值文本，训练端仍接收到原始payload。
        if "*" in segment or "," in segment:
            tokens = re.split(r"[*,]", segment.strip())
        else:
            tokens = segment.split()
        try:
            if not tokens or any(not token.strip() or not math.isfinite(float(token)) for token in tokens):
                raise ValueError("无效信号")
        except ValueError as exc:
            raise ValueError("{}: signal段含非数值或非有限值".format(context)) from exc
    return masked, label

def iter_features(feature_path, allowed_labels=None):
    """Yield physical CSV line, normalized row and original-record provenance.

    No metadata lookup occurs. The 1-based data record ordinal excludes an
    optional header and remains separate from the physical CSV source line.
    Numeric feature field strings are preserved without recomputation.
    """
    for ordinal, (line_no, row) in enumerate(_feature_rows(feature_path), 1):
        context = "{}:{} (数据记录{})".format(feature_path, line_no, ordinal)
        masked, label = _validate_features(row, context)
        if allowed_labels is not None and label not in allowed_labels:
            raise ValueError("{}: 该来源仅允许label={}，实际={}".format(context, sorted(allowed_labels), label))
        original = {"original_kmer": row[0], "original_label": row[10],
                    "record_ordinal": ordinal}
        row[0], row[10] = masked, str(label)
        yield line_no, row, original

def is_positive_row_proxy(meta, label=None):
    """Recognize only the canonical, fully disclosed CSV-row proxy format.

    This validates the recorded proxy format, not the experimental assumption or
    a real read UUID. It never turns an unknown reference into a known one.
    """
    try:
        label = exact_label(meta.get("label")) if label is None else int(label)
    except (ValueError, TypeError):
        return False
    digest = meta.get("positive_source_sha256", "")
    ordinal = meta.get("positive_data_record", "")
    return (label in (4, 5)
            and meta.get("merge_input_role") == "positive"
            and meta.get("read_id_kind") == "positive_row_proxy"
            and meta.get("read_identity_basis") == "assumed_one_csv_row_per_positive_read"
            and meta.get("reference_identity_known") == "false"
            and meta.get("original_read_id") == "unknown"
            and meta.get("source_dataset") == "XNA_replenishment"
            and normalize_known(meta.get("chrom")) == "unknown"
            and normalize_known(meta.get("bam_strand")) == "unknown"
            and normalize_known(meta.get("run_id")) == "unknown"
            and bool(re.fullmatch(r"[0-9a-f]{64}", digest))
            and bool(re.fullmatch(r"[1-9][0-9]*", str(ordinal)))
            and meta.get("read_id") == "positive-row:{}:{}".format(digest, ordinal))

def iter_feature_meta(feature_path, meta_path, allowed_labels=None):
    """Yield (CSV record number, canonical feature list, standardized metadata).

    This validates row counts and available sequence/label identity. Only genuine,
    correctly paired extraction metadata can establish the true read provenance.
    Numeric payloads remain byte-for-byte field values; only kmer/label are
    standardized. Sequence aliases and all extra metadata columns are retained.
    """
    names = metadata_header(meta_path)
    with open(meta_path, "r", encoding="utf-8-sig", newline="", buffering=8 * 1024 * 1024) as stream:
        metadata = csv.reader(stream, delimiter="\t", strict=True)
        next(metadata)
        sentinel = object()
        for index, pair in enumerate(itertools.zip_longest(_feature_rows(feature_path), metadata, fillvalue=sentinel), 1):
            feature, values = pair
            if feature is sentinel or values is sentinel:
                raise ValueError("{} 与 {} 行数不一致，首个不配对的数据记录={}".format(feature_path, meta_path, index))
            line_no, row = feature
            context = "{}:{} (metadata记录{})".format(feature_path, line_no, index)
            if len(values) != len(names):
                raise ValueError("{}: metadata列数与表头不符".format(context))
            meta = dict(zip(names, values))
            masked, label = _validate_features(row, context)
            if allowed_labels is not None and label not in allowed_labels:
                raise ValueError("{}: 该来源仅允许label={}，实际={}".format(context, sorted(allowed_labels), label))
            if exact_label(meta["label"], context + " metadata") != label:
                raise ValueError("{}: CSV与metadata标签不一致".format(context))
            for key in ("kmer", "output_kmer", "6mer", "masked_6mer", "masked_kmer", "original_kmer", "raw_oriented_kmer", "raw_oriented_6mer"):
                if meta.get(key, "").strip() and mask_kmer(meta[key], context + " " + key) != masked:
                    raise ValueError("{}: CSV与metadata.{}的侧翼不一致".format(context, key))
            if meta.get("original_label", "").strip() and exact_label(meta["original_label"], context + " original_label") != label:
                raise ValueError("{}: original_label与当前六类标签不一致；本脚本不接受已做二分类映射的数据".format(context))
            read_id = meta["read_id"]
            if normalize_known(read_id) == "unknown" or read_id != read_id.strip():
                raise ValueError("{}: read_id缺失、为unknown/NaN或含首尾空白".format(context))
            proxy = is_positive_row_proxy(meta, label)
            if meta.get("read_id_kind") == "positive_row_proxy" and not proxy:
                raise ValueError("{}: 阳性行代理ID的格式或来源flags不完整/冲突".format(context))
            if normalize_known(meta["chrom"]) == "unknown" and not proxy:
                raise ValueError("{}: chrom缺失或为unknown/NaN".format(context))
            if normalize_known(meta.get("bam_strand")) not in ("+", "-", "unknown"):
                raise ValueError("{}: bam_strand应为+/-/unknown/.".format(context))
            for key in ("label_base", "true_base"):
                if meta.get(key, "").strip() and meta[key].strip().upper() != LABEL_NAMES[label]:
                    raise ValueError("{}: metadata.{}={!r}与label={}不符".format(context, key, meta[key], label))
            if label <= 3:
                for key in ("original_kmer", "raw_oriented_kmer", "raw_oriented_6mer"):
                    sequence = meta.get(key, "").strip().upper()
                    if sequence and sequence[3] in "ATCG" and sequence[3] != LABEL_NAMES[label]:
                        raise ValueError("{}: {}第四位与天然碱基label={}冲突".format(context, key, label))
                if row[0][3].upper() in "ATCG" and row[0][3].upper() != LABEL_NAMES[label]:
                    raise ValueError("{}: CSV第四位与天然碱基label={}冲突".format(context, label))
            original_kmer, original_label = row[0], row[10]
            meta["original_kmer"] = (meta.get("original_kmer") or meta.get("raw_oriented_kmer")
                                     or meta.get("raw_oriented_6mer") or original_kmer)
            meta["original_label"] = meta.get("original_label") or original_label
            meta["kmer"], meta["label"] = masked, str(label)
            row[0], row[10] = masked, str(label)
            yield line_no, row, meta

def parse_label(value):
    return exact_label(value)

def normalize_known(value):
    """Treat missing provenance as unknown; never infer a sequencing run."""
    value = "" if value is None else str(value).strip()
    return "unknown" if value.lower() in ("", "unknown", "na", "n/a", "nan", "none", "null", ".") else value

def validate_or_assign_split(meta, read_id, seed=20260429, ratio_spec="7,2,1"):
    ratios = parse_split_ratios(ratio_spec) if isinstance(ratio_spec, str) else tuple(ratio_spec)
    ratio_text = ratio_spec if isinstance(ratio_spec, str) else ",".join(str(x) for x in ratio_spec)
    expected = assign_read_split(read_id, seed, ratios)
    if meta.get("split", "").strip() and canonical_split(meta["split"]) != expected:
        raise ValueError("read_id={}已有split={!r}与固定read哈希分区{}冲突；不能重新划分已分区数据".format(read_id, meta["split"], expected))
    if meta.get("split_seed", "").strip():
        try:
            prior_seed = Decimal(meta["split_seed"])
        except (InvalidOperation, ValueError) as exc:
            raise ValueError("read_id={}的split_seed无效".format(read_id)) from exc
        if prior_seed != Decimal(seed):
            raise ValueError("read_id={}的已有split_seed与配置不一致".format(read_id))
    if meta.get("split_ratios", "").strip():
        prior = parse_split_ratios(meta["split_ratios"])
        if any(abs(a - b) > 1e-14 for a, b in zip(prior, ratios)):
            raise ValueError("read_id={}的已有split_ratios与配置不一致".format(read_id))
    if meta.get("split_method", "").strip() and meta["split_method"] != SPLIT_METHOD:
        raise ValueError("read_id={}的split_method不是{}".format(read_id, SPLIT_METHOD))
    meta.update(split=expected, split_seed=str(seed), split_ratios=ratio_text, split_method=SPLIT_METHOD)
    return expected

import argparse
import json
import sqlite3
import tempfile
import os
import shutil
import traceback
import time
from collections import Counter, defaultdict
from contextlib import ExitStack
from datetime import datetime, timezone
from fractions import Fraction

VERSION = "1.0.0"
BASE = Path("/path/to/revision-inputs/CPLX-XNA-replenishment/CPLX-XNA-replenishment/fast5_single_guppy")
DEFAULT_INPUT = BASE / "xna_pc_6mer_features_all_replenishment_6_overlap.csv"
SOURCE_LABEL_NAMES = dict(LABEL_NAMES)
TASK_SOURCE_LABELS = {"X": (0,1,2,3,4), "Y": (0,1,2,3,5)}
EXTRA_META = ["kmer", "original_kmer", "original_label", "motif", "input_feature_file", "input_feature_row",
              "input_metadata_file", "input_metadata_row", "selected_index", "split", "split_seed",
              "split_ratios", "split_method", "source_label", "output_label", "source_class_name",
              "task", "label_mapping_id"]

# 以下分区、输入审计、整数配额及抽样函数沿用六分类脚本的实现；
# 配额仅在当前任务选定的五种原始标签上计算。sample priority保持原算法。


def sha256(path):
    digest = hashlib.sha256()
    with open(path, "rb") as handle:
        for chunk in iter(lambda: handle.read(8 * 1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def fingerprint(path):
    path = Path(path).resolve()
    before = path.stat()
    digest = sha256(path)
    after = path.stat()
    if (before.st_size, before.st_mtime_ns) != (after.st_size, after.st_mtime_ns):
        raise ValueError("输入在计算指纹时发生变化: " + str(path))
    return {"path": str(path), "size_bytes": after.st_size, "mtime_ns": after.st_mtime_ns, "sha256": digest}


def integer_weights(text):
    try:
        weights = [Fraction(item.strip()) for item in str(text).split(",")]
    except (ValueError, ZeroDivisionError):
        raise ValueError("split_ratios 需要三个正数，例如 7,2,1")
    if len(weights) != 3 or any(item <= 0 for item in weights):
        raise ValueError("split_ratios 需要三个正数，例如 7,2,1")
    denominator = math.lcm(*(item.denominator for item in weights))
    integers = [int(item * denominator) for item in weights]
    divisor = math.gcd(*integers)
    integers = tuple(item // divisor for item in integers)
    if max(integers) > 10**9:
        raise ValueError("split_ratios 的最简整数权重过大，无法进行有意义的精确均衡")
    return integers


def normalize_split(text):
    text = str(text).strip().lower()
    if text in ("val", "valid"):
        return "validation"
    return text


def read_metadata_header(path):
    return metadata_header(path)


def iter_input(features, metadata):
    """Validated, canonicalized paired rows; ordinal excludes the header."""
    for ordinal, (_line_no, feature, meta) in enumerate(iter_feature_meta(features, metadata), 1):
        yield ordinal, feature, meta, feature[0]


def validate_assignment(meta, args, ordinal):
    expected = assign_read_split(meta["read_id"], args.split_seed, args.normalized_ratios)
    existing = normalize_split(meta.get("split", ""))
    if existing and existing != expected:
        raise ValueError("已有 split 与固定 read 哈希分区冲突，不允许重分: 数据行 {0}, read={1}, metadata={2}, expected={3}".format(ordinal, meta["read_id"], existing, expected))
    if existing and existing not in SPLIT_NAMES:
        raise ValueError("未知 split: " + existing)
    # Newly extracted canonical windows must carry their original split contract.
    if int(meta["label"]) < 4 and not existing:
        raise ValueError("阴性数据必须保留 extractor 生成的 split 信息；数据行 " + str(ordinal))
    if existing:
        for field in ("split_seed", "split_ratios", "split_method"):
            if not meta.get(field):
                raise ValueError("已有 split 却缺少 " + field + "，无法确认分区来源；数据行 " + str(ordinal))
    if meta.get("split_seed"):
        try:
            seed = int(meta["split_seed"])
        except ValueError:
            raise ValueError("metadata split_seed 必须为整数，数据行 " + str(ordinal))
        if seed != args.split_seed:
            raise ValueError("metadata split_seed 与当前参数不一致，数据行 " + str(ordinal))
    if meta.get("split_ratios"):
        original = parse_split_ratios(meta["split_ratios"])
        # Floating boundaries, not only integer proportions, must match the extractor.
        boundaries = lambda values: (int(values[0] * (1 << 64)), int((values[0] + values[1]) * (1 << 64)))
        if boundaries(original) != boundaries(args.normalized_ratios):
            raise ValueError("metadata split_ratios 的 read 哈希边界与当前参数不一致，数据行 " + str(ordinal))
    if meta.get("split_method") and meta["split_method"] != SPLIT_METHOD:
        raise ValueError("metadata split_method 与 extractor 不一致，数据行 " + str(ordinal))
    return expected


def json_write(path, value):
    with open(path, "w", encoding="utf-8") as handle:
        json.dump(value, handle, ensure_ascii=False, indent=2, allow_nan=False)
        handle.write("\n")


def csv_write(path, header, rows, delimiter=","):
    with open(path, "w", encoding="utf-8", newline="") as handle:
        writer = csv.writer(handle, delimiter=delimiter)
        writer.writerow(header)
        writer.writerows(rows)


def open_database(path):
    con = sqlite3.connect(path)
    con.execute("PRAGMA journal_mode=DELETE")
    con.execute("PRAGMA synchronous=NORMAL")
    con.execute("PRAGMA temp_store=FILE")
    con.execute("CREATE TABLE windows (ordinal INTEGER PRIMARY KEY, read_id TEXT NOT NULL, motif TEXT NOT NULL, label INTEGER NOT NULL, chrom TEXT NOT NULL, split TEXT NOT NULL, source TEXT NOT NULL, strand TEXT NOT NULL, run_id TEXT NOT NULL, read_id_kind TEXT NOT NULL, reference_identity_known INTEGER NOT NULL, priority BLOB NOT NULL, physical_key TEXT, selected INTEGER NOT NULL DEFAULT 0)")
    con.execute("CREATE UNIQUE INDEX unique_physical_window ON windows(physical_key) WHERE physical_key IS NOT NULL")
    con.execute("CREATE UNIQUE INDEX unique_positive_read ON windows(read_id) WHERE label>=4")
    con.execute("CREATE TABLE reads (read_id TEXT PRIMARY KEY, split TEXT NOT NULL, input_windows INTEGER NOT NULL DEFAULT 0, selected_windows INTEGER NOT NULL DEFAULT 0, role INTEGER NOT NULL CHECK(role IN (1,2)), read_id_kind TEXT NOT NULL)")
    return con


def scan_input(args, stage, con):
    counts = Counter()
    source_counts = Counter()
    rows = []
    read_batch = Counter()
    last_log = time.monotonic()
    total = 0

    def flush():
        try:
            con.executemany("INSERT INTO windows (ordinal,read_id,motif,label,chrom,split,source,strand,run_id,read_id_kind,reference_identity_known,priority,physical_key) VALUES (?,?,?,?,?,?,?,?,?,?,?,?,?)", rows)
        except sqlite3.IntegrityError as error:
            if "windows.read_id" in str(error):
                raise ValueError("阳性 label4/5 每个 read 只能有一个 window；发现重复 positive read_id。" + str(error)) from error
            raise ValueError("输入中出现相同物理 window：read_id/chrom/bam_strand/kmer_ref_start/kmer_ref_end 重复；不能把重复提取的同一窗口计入可用样本。" + str(error)) from error
        try:
            con.executemany("INSERT INTO reads (read_id,split,input_windows,role,read_id_kind) VALUES (?,?,?,?,?) ON CONFLICT(read_id) DO UPDATE SET input_windows=input_windows+excluded.input_windows, role=role|excluded.role", ((read, split, count, role, kind) for (read, split, role, kind), count in read_batch.items()))
        except sqlite3.IntegrityError as error:
            raise ValueError("同一 read_id 同时出现在 canonical 阴性 label0..3 与 X/Y 阳性 label4/5；来源身份冲突，不能继续。") from error
        con.commit()
        rows.clear()
        read_batch.clear()

    for ordinal, feature, meta, motif in iter_input(args.input, args.meta):
        split = validate_assignment(meta, args, ordinal)
        label = int(feature[-1])
        source = normalize_known(meta.get("source_dataset") or meta.get("source_type"))
        strand = normalize_known(meta.get("bam_strand"))
        run_id = normalize_known(meta.get("run_id"))
        read_id_kind = "positive_row_proxy" if is_positive_row_proxy(meta, label) else "provided_real_read_id"
        reference_known = normalize_known(meta.get("chrom")) != "unknown"
        chrom = normalize_known(meta.get("chrom"))
        priority = hashlib.blake2b((str(args.sample_seed) + "\x00" + str(ordinal)).encode("utf-8"), digest_size=16, person=b"sixclass-sample").digest()
        physical_key = None
        if meta.get("chrom") != meta.get("chrom", "").strip():
            raise ValueError("chrom 首尾不能含空白，数据行 " + str(ordinal))
        coordinate_values = [meta.get(name, "").strip() for name in ("kmer_ref_start", "kmer_ref_end")]
        if all(coordinate_values) and strand in ("+", "-") and reference_known and read_id_kind != "positive_row_proxy":
            try:
                coordinates = [Decimal(value) for value in coordinate_values]
                if any(not value.is_finite() or value != value.to_integral_value() for value in coordinates):
                    raise ValueError("坐标不是有限整数")
                start, end = (int(value) for value in coordinates)
                if start > end:
                    raise ValueError("kmer_ref_start 大于 kmer_ref_end")
            except (ValueError, InvalidOperation) as error:
                raise ValueError("metadata window 坐标无效，数据行 " + str(ordinal)) from error
            physical_key = json.dumps([meta["read_id"], meta["chrom"], strand, start, end], ensure_ascii=False, separators=(",", ":"))
        rows.append((ordinal, meta["read_id"], motif, label, chrom, split, source, strand, run_id, read_id_kind, int(reference_known), priority, physical_key))
        read_batch[(meta["read_id"], split, 1 if label < 4 else 2, read_id_kind)] += 1
        counts[(motif, label, chrom, split)] += 1
        source_counts[("input", source, strand, run_id, chrom, label, split)] += 1
        total = ordinal
        if len(rows) >= 10000:
            flush()
        now = time.monotonic()
        if (args.log_every and ordinal % args.log_every == 0) or now - last_log >= args.log_seconds:
            print("[审计] 已对齐 {:,} 行".format(ordinal), file=sys.stderr, flush=True)
            last_log = now
    if rows:
        flush()
    if total == 0:
        raise ValueError("输入无数据行")
    con.execute("CREATE INDEX windows_strata ON windows (motif,label,chrom,split,priority,ordinal)")
    con.execute("CREATE INDEX windows_class_split ON windows (motif,label,split,priority,ordinal)")
    con.execute("CREATE INDEX windows_reads ON windows (read_id)")
    con.commit()
    conflicting = con.execute("SELECT read_id FROM windows GROUP BY read_id HAVING count(DISTINCT split)>1 LIMIT 1").fetchone()
    if conflicting:
        raise RuntimeError("同一 read 被分配到多个分区: " + conflicting[0])
    conflicting_kind = con.execute("SELECT read_id FROM windows GROUP BY read_id HAVING count(DISTINCT read_id_kind)>1 LIMIT 1").fetchone()
    if conflicting_kind:
        raise ValueError("同一标识符的 read_id_kind 不一致: " + conflicting_kind[0])
    return counts, source_counts, total


def make_quotas(args, counts):
    references = defaultdict(set)
    totals = Counter()
    motifs = set()
    for (motif, label, chrom, split), count in counts.items():
        motifs.add(motif)
        references[(motif, label)].add(chrom)
        totals[(motif, label, split)] += count
    quotas = []
    motif_report = []
    reference_report = []
    retained = []
    for motif in sorted(motifs):
        ks = [len(references[(motif, label)]) for label in args.source_labels]
        absent = [str(label) for label, k in zip(args.source_labels, ks) if not k]
        lattice = math.lcm(*(k for k in ks if k)) if all(ks) else 0
        constraints = []
        if absent:
            t_max = t = 0
            reason = "missing_labels:" + ",".join(absent)
        elif args.reference_balance == "equal":
            for label in args.source_labels:
                k = ks[args.source_labels.index(label)]
                for chrom in sorted(references[(motif, label)]):
                    for split, weight in zip(SPLIT_NAMES, args.integer_weights):
                        capacity = counts[(motif, label, chrom, split)] * k // weight
                        constraints.append((capacity, label, chrom, split))
            t_max = min(value[0] for value in constraints)
            t = t_max // lattice * lattice
            reason = "retained" if t else "insufficient_read_partition_or_reference_capacity"
        else:
            for label in args.source_labels:
                for split, weight in zip(SPLIT_NAMES, args.integer_weights):
                    constraints.append((totals[(motif, label, split)] // weight, label, "all", split))
            t_max = min(value[0] for value in constraints)
            t = t_max
            lattice = 1
            reason = "retained" if t else "insufficient_read_partition_capacity"
        limiting = [list(value) for value in constraints if value[0] == t_max]
        retained.append(motif) if t else None
        for label in args.source_labels:
            known_reference_count = sum(normalize_known(chrom) != "unknown" for chrom in references[(motif, label)])
            for split, weight in zip(SPLIT_NAMES, args.integer_weights):
                available = totals[(motif, label, split)]
                target = weight * t
                motif_report.append((motif, label, LABEL_NAMES[label], split, available, target, available-target, known_reference_count, lattice, t_max, t, "retained" if t else "dropped", reason, json.dumps(limiting, ensure_ascii=False), ks[args.source_labels.index(label)]))
                if args.reference_balance == "none":
                    if target:
                        quotas.append((motif, label, None, split, target))
                for chrom in sorted(references[(motif, label)]):
                    ref_target = target // ks[args.source_labels.index(label)] if args.reference_balance == "equal" else None
                    known = normalize_known(chrom) != "unknown"
                    status = ("equal_within_motif_and_class" if args.reference_balance == "equal" else "not_requested") if known else "not_performed_unknown_reference_identity"
                    reference_report.append([motif, label, LABEL_NAMES[label], chrom, split, counts[(motif, label, chrom, split)], ref_target, None, args.reference_balance, reason, known, status, known and args.reference_balance == "equal"])
                    if args.reference_balance == "equal" and ref_target:
                        quotas.append((motif, label, chrom, split, ref_target))
    return quotas, motif_report, reference_report, retained


def select_windows(args, con, quotas):
    for motif, label, chrom, split, target in quotas:
        if chrom is None:
            con.execute("UPDATE windows SET selected=1 WHERE ordinal IN (SELECT ordinal FROM windows WHERE motif=? AND label=? AND split=? ORDER BY priority,ordinal LIMIT ?)", (motif, label, split, target))
        else:
            con.execute("UPDATE windows SET selected=1 WHERE ordinal IN (SELECT ordinal FROM windows WHERE motif=? AND label=? AND chrom=? AND split=? ORDER BY priority,ordinal LIMIT ?)", (motif, label, chrom, split, target))
    con.execute("UPDATE reads SET selected_windows=(SELECT count(*) FROM windows WHERE windows.read_id=reads.read_id AND windows.selected=1)")
    con.commit()
    actual = {(motif, label, chrom, split): count for motif, label, chrom, split, count in con.execute("SELECT motif,label,chrom,split,count(*) FROM windows WHERE selected=1 GROUP BY motif,label,chrom,split")}
    actual_combined = Counter()
    for (motif, label, chrom, split), count in actual.items():
        actual_combined[(motif, label, split)] += count
    for motif, label, chrom, split, target in quotas:
        got = actual_combined[(motif, label, split)] if chrom is None else actual.get((motif, label, chrom, split), 0)
        if got != target:
            raise RuntimeError("抽样数量与 quota 不一致: " + str((motif, label, chrom, split, target, got)))
    return actual


def label_mapping(case):
    sources = TASK_SOURCE_LABELS[case]
    return {
        "task": "fiveclass_" + case, "mapping_id": "fiveclass-{}-v1".format(case),
        "num_classes": 5, "source_labels": list(sources),
        "source_to_output": {str(label): label if label < 4 else 4 for label in sources},
        "output_label_names": {str(i): name for i, name in enumerate(("A", "T", "C", "G", case))},
        "excluded_source_labels": [label for label in range(6) if label not in sources],
        "provenance": "original_label/source_label retain the original six-class label; label/output_label are training labels",
        "sequence_target_index": 3, "sequence_target_character": "N",
    }


def output_label(source):
    return source if source < 4 else 4


def source_condition(labels):
    # Labels are internal constants, never SQL supplied by the user.
    return "label IN ({})".format(",".join(str(int(label)) for label in labels))


def write_task_reports(stage, motif_report, reference_report, selected=None):
    def mapped_prefix(row):
        source = int(row[1])
        return [row[0], output_label(source), row[2], source] + list(row[3:])
    csv_write(stage / "motif_balance_report.csv",
        ["motif", "label", "class_name", "source_label", "split", "available", "selected", "discarded",
         "references_in_class", "lcm_reference_counts", "maximum_t_before_rounding", "selected_t", "status",
         "reason", "limiting_constraints_json", "sampling_reference_buckets"],
        (mapped_prefix(row) for row in motif_report))
    rows = []
    for record in reference_report:
        row = list(record)
        row[7] = (selected or {}).get((row[0], row[1], row[3], row[4]), 0)
        rows.append(mapped_prefix(row))
    csv_write(stage / "reference_balance_report.csv",
        ["motif", "label", "class_name", "source_label", "chrom", "split", "available", "sampling_bucket_target",
         "selected", "reference_balance_requested", "reason", "reference_identity_known", "reference_balance_status",
         "equal_reference_balance_applied"], rows)


def export_task_audits(args, stage, con, source_counts):
    scope = source_condition(args.source_labels)
    read_rows = con.execute("SELECT read_id,split,count(*),sum(selected),read_id_kind FROM windows WHERE " + scope +
                           " GROUP BY read_id ORDER BY read_id")
    csv_write(stage / "read_splits.tsv",
        ["read_id", "split", "input_windows", "selected_windows", "read_id_kind", "split_seed", "split_ratios", "split_method"],
        (tuple(row) + (args.split_seed, args.split_ratios, SPLIT_METHOD) for row in read_rows), delimiter="\t")
    rows = []
    for key, count in sorted(source_counts.items()):
        phase, source, strand, run_id, chrom, label, split = key
        if label in args.source_labels:
            rows.append((phase, source, strand, run_id, chrom, output_label(label), split, count, label, LABEL_NAMES[label]))
    selected = con.execute("SELECT source,strand,run_id,chrom,label,split,count(*) FROM windows WHERE selected=1 GROUP BY source,strand,run_id,chrom,label,split")
    rows.extend(("selected", source, strand, run_id, chrom, output_label(label), split, count, label, LABEL_NAMES[label])
                for source, strand, run_id, chrom, label, split, count in selected)
    csv_write(stage / "source_strand_run_template_counts.csv",
        ["stage", "source_dataset", "bam_strand", "run_id", "chrom", "label", "split", "windows", "source_label", "class_name"], rows)
    for category, column in (("source", "source"), ("strand", "strand"), ("run", "run_id"), ("template", "chrom")):
        query = ("SELECT CASE selected WHEN 1 THEN 'selected' ELSE 'not_selected' END," + column +
                 ",label,split,count(*),count(DISTINCT read_id) FROM windows WHERE " + scope +
                 " GROUP BY selected," + column + ",label,split ORDER BY selected," + column + ",label,split")
        csv_write(stage / (category + "_class_counts.csv"),
            ["selection", category, "label", "split", "windows", "unique_reads", "source_label", "class_name"],
            ((selection, value, output_label(label), split, n, reads, label, LABEL_NAMES[label])
             for selection, value, label, split, n, reads in con.execute(query)))
    provenance = con.execute("SELECT ordinal,read_id,motif,label,chrom,split,source,strand,run_id,read_id_kind,reference_identity_known FROM windows WHERE selected=1 ORDER BY ordinal")
    csv_write(stage / "selected_provenance.tsv",
        ["selected_index", "input_feature_row", "read_id", "motif", "label", "chrom", "split", "source_dataset", "bam_strand", "run_id", "read_id_kind", "reference_identity_known", "source_label", "class_name"],
        ((index, ordinal, read, motif, output_label(label), chrom, split, source, strand, run_id, kind, known, label, LABEL_NAMES[label])
         for index, (ordinal, read, motif, label, chrom, split, source, strand, run_id, kind, known) in enumerate(provenance, 1)), delimiter="\t")


def window_distribution(con, labels, positive, selected=False):
    condition = source_condition(labels) + (" AND label>=4" if positive else " AND label<4")
    if selected:
        condition += " AND selected=1"
    query = "SELECT count(*) AS n FROM windows WHERE " + condition + " GROUP BY read_id"
    n = con.execute("SELECT count(*) FROM (" + query + ")").fetchone()[0]
    if not n:
        return {"reads": 0, "min": None, "p50": None, "p95": None, "max": None}
    def at(q):
        return con.execute("SELECT n FROM (" + query + ") ORDER BY n LIMIT 1 OFFSET ?", (math.floor(q*(n-1)),)).fetchone()[0]
    return {"reads": n, "min": at(0), "p50": at(.5), "p95": at(.95), "max": at(1),
            "quantile_method": "lower order statistic at floor(q*(n-1))"}


def task_summary(args, con, fingerprints, input_total, counts, retained, output_counts):
    scope = source_condition(args.source_labels)
    count = lambda condition: con.execute("SELECT count(*) FROM windows WHERE " + condition).fetchone()[0]
    read_count = lambda condition: con.execute("SELECT count(DISTINCT read_id) FROM windows WHERE " + condition).fetchone()[0]
    splits = {}
    for split in SPLIT_NAMES:
        per_class = {str(output_label(label)): output_counts[(split, label)] for label in args.source_labels}
        if len(per_class) != 5 or len(set(per_class.values())) != 1 or not min(per_class.values()):
            raise RuntimeError("分区内五类数量不相等或缺类：" + split)
        splits[split] = {"windows": sum(per_class.values()), "per_class": per_class,
            "unique_reads": read_count("selected=1 AND split='" + split + "'")}
    for label in args.source_labels:
        units = [output_counts[(split, label)] // weight for split, weight in zip(SPLIT_NAMES, args.integer_weights)]
        if len(set(units)) != 1 or any(output_counts[(split, label)] % weight for split, weight in zip(SPLIT_NAMES, args.integer_weights)):
            raise RuntimeError("输出窗口比例不满足精确整数配额")
    overlap = con.execute("SELECT count(*) FROM (SELECT read_id FROM windows WHERE selected=1 GROUP BY read_id HAVING count(DISTINCT split)>1)").fetchone()[0]
    if overlap:
        raise RuntimeError("输出中出现跨分区 read")
    positive_scope = scope + " AND label>=4"
    proxy_condition = positive_scope + " AND read_id_kind='positive_row_proxy'"
    proxies = read_count(proxy_condition)
    real = read_count(positive_scope + " AND read_id_kind!='positive_row_proxy'")
    positive = {
        "real_read_id_count": real, "row_proxy_count": proxies,
        "selected_real_read_id_count": read_count(positive_scope + " AND selected=1 AND read_id_kind!='positive_row_proxy'"),
        "selected_row_proxy_count": read_count(proxy_condition + " AND selected=1"),
        "positive_one_window_per_read_assumed": bool(proxies),
        "positive_one_window_per_real_read_verified": not bool(proxies),
        "provided_real_read_ids_one_window_each": True,
        "actual_rawpositiveoriginalread_overlap_check": "unavailable" if proxies else "verified_for_provided_real_read_ids",
        "basis": "assumed_one_csv_row_per_positive_read" if proxies else "provided_row_aligned_real_read_ids",
        "limitation": "Row proxies identify input records and do not verify original positive read UUIDs or templates." if proxies else "Read identity alone does not establish biological independence.",
    }
    reference_audit = {}
    for role, condition in (("negative", "label<4"), ("positive", "label>=4")):
        known = count("selected=1 AND " + condition + " AND reference_identity_known=1")
        unknown = count("selected=1 AND " + condition + " AND reference_identity_known=0")
        status = ("not_performed_unknown_positive_reference_identity" if role == "positive" and not known
                  else "partially_unavailable_unknown_reference_identity") if unknown else (
                  "equal_within_motif_and_class" if args.reference_balance == "equal" else "not_requested")
        reference_audit[role] = {"status": status, "known_reference_rows": known, "unknown_reference_rows": unknown,
            "limitation": "Unknown is a sampling bucket, not a verified reference; its underlying references cannot be balanced." if unknown else None}
    unit = "provided_real_read_ids_or_positive_csv_row_proxies" if proxies else "provided_real_read_ids"
    config = {name: str(value) if isinstance(value, Path) else value for name, value in vars(args).items()}
    mapping = label_mapping(args.case)
    return {
        "status": "complete", "script_version": VERSION, "task": mapping["task"],
        "created_utc": datetime.now(timezone.utc).isoformat(), "config": config,
        "num_classes": 5, "label_names": mapping["output_label_names"], "label_mapping": mapping,
        "input_fingerprints": fingerprints, "code_sha256": {"script": sha256(Path(__file__))},
        "input_windows": sum(counts.values()), "input_windows_in_full_sixclass_file": input_total,
        "input_unique_reads": read_count(scope), "selected_windows": sum(s["windows"] for s in splits.values()),
        "selected_unique_reads": read_count("selected=1"), "unique_read_count_unit": unit,
        "retained_motif_count": len(retained), "retained_motifs": retained,
        "split_counts": splits, "read_ids_in_multiple_splits": overlap,
        "positive": positive, "all_original_positive_read_ids_verified": not bool(proxies),
        "read_id_overlap_check_scope": unit,
        "read_window_audit": {
            "negative_input": window_distribution(con, args.source_labels, False),
            "negative_selected": window_distribution(con, args.source_labels, False, True),
            "positive_input": window_distribution(con, args.source_labels, True),
            "positive_selected": window_distribution(con, args.source_labels, True, True),
            "positive_one_window_per_read": None if proxies else True,
            "positive_one_window_per_read_assumed": bool(proxies),
            "cross_positive_negative_provided_identifier_collision_count": 0,
            "cross_positive_negative_read_id_count": None if proxies else 0,
        },
        "physical_window_duplicate_audit": {
            "input_windows_with_complete_identity": count(scope + " AND physical_key IS NOT NULL"),
            "input_windows_without_complete_identity": count(scope + " AND physical_key IS NULL"),
            "duplicate_complete_identities": 0,
            "key": ["read_id", "chrom", "bam_strand", "kmer_ref_start", "kmer_ref_end"],
            "limitation": "Missing coordinates/strand or row proxies cannot prove physically distinct windows or original read identity.",
        },
        "templates_per_class": {str(output_label(label)): con.execute("SELECT count(DISTINCT chrom) FROM windows WHERE selected=1 AND label=? AND reference_identity_known=1", (label,)).fetchone()[0] for label in args.source_labels},
        "known_reference_count_per_class": {str(output_label(label)): con.execute("SELECT count(DISTINCT chrom) FROM windows WHERE selected=1 AND label=? AND reference_identity_known=1", (label,)).fetchone()[0] for label in args.source_labels},
        "known_runs_per_class": {str(output_label(label)): con.execute("SELECT count(DISTINCT run_id) FROM windows WHERE selected=1 AND label=? AND run_id!='unknown'", (label,)).fetchone()[0] for label in args.source_labels},
        "selected_windows_with_unknown_reference_by_class": {str(output_label(label)): count("selected=1 AND label=" + str(label) + " AND reference_identity_known=0") for label in args.source_labels},
        "selected_windows_with_unknown_run": count("selected=1 AND run_id='unknown'"),
        "split_contract": {
            "unit": unit, "method": SPLIT_METHOD, "seed": args.split_seed, "ratio_spec": args.split_ratios,
            "hash_boundaries_uint64": [int(args.normalized_ratios[0]*(1<<64)), int((args.normalized_ratios[0]+args.normalized_ratios[1])*(1<<64))],
            "window_ratio": ":".join(map(str,args.integer_weights)), "read_ratios_are_expected_not_exact": True,
            "read_assignments_shared_across_tasks": True,
        },
        "sampling": {
            "without_replacement": True, "reference_balance": args.reference_balance,
            "priority": "BLAKE2b-128(sample_seed + NUL + original_1_based_row), person=sixclass-sample; ascending bytes then ordinal",
            "order": "original input record order; numerical feature payload is unchanged",
            "reference_balance_audit": reference_audit,
        },
        "scope_and_limits": [
            "Two tasks are independently downsampled. Shared original reads have identical fixed partitions, but their selected windows can differ between tasks.",
            "Exact 7:2:1 (or requested integer ratio) applies to selected windows, not exact unique-read counts. Overlapping windows are grouped by provided read ID.",
            "Positive CSV row proxies follow the one-row-per-read experimental assumption; original positive read overlap cannot be independently verified.",
            "Unknown positive reference, strand and run remain unknown. Unknown buckets do not establish true-reference balance.",
            "Read-disjoint evaluation does not establish template/run-held-out or biological-replicate generalization.",
            "Original six-class source labels are retained; Y source label5 maps to output label4, whose biological meaning is Y only in the Y task.",
            "Fourth sequence position is N; signal/statistics/basecalling response fields are retained without recomputation.",
        ],
        "outputs": {split: {"features": split+".csv", "metadata":split+"_meta.tsv"} for split in SPLIT_NAMES},
    }


def export_all_tasks(args, stage, con, input_header):
    """Second full input pass: write both tasks while retaining original ordinals."""
    output_header = list(input_header) + [field for field in EXTRA_META if field not in input_header]
    cursor = iter(con.execute("SELECT ordinal,case_name,split FROM task_selections ORDER BY ordinal,case_name"))
    chosen = next(cursor, None)
    counts, selected_indices = Counter(), Counter()
    with ExitStack() as stack:
        writers = {}
        for case in args.cases:
            folder = stage / ("fiveclass_" + case)
            for split in SPLIT_NAMES:
                features = stack.enter_context(open(folder/(split+".csv"),"w",encoding="utf-8",newline=""))
                metadata = stack.enter_context(open(folder/(split+"_meta.tsv"),"w",encoding="utf-8",newline=""))
                fw = csv.writer(features)
                mw = csv.DictWriter(metadata,fieldnames=output_header,delimiter="\t",extrasaction="raise")
                fw.writerow(FEATURE_HEADER); mw.writeheader()
                writers[(case,split)] = fw,mw
        for ordinal, feature, meta, motif in iter_input(args.input,args.meta):
            split = validate_assignment(meta,args,ordinal)
            while chosen is not None and chosen[0] == ordinal:
                _,case,selected_split = chosen
                source = int(feature[-1])
                if split != selected_split or source not in TASK_SOURCE_LABELS[case]:
                    raise RuntimeError("第二遍读取的分区/类别与抽样记录不一致")
                out_feature,out_meta = list(feature),dict(meta)
                mapped = output_label(source)
                out_feature[0],out_feature[-1] = motif,str(mapped)
                selected_indices[case] += 1
                for alias in ("kmer","output_kmer","6mer","masked_6mer","masked_kmer"):
                    if alias in out_meta:
                        out_meta[alias] = motif
                out_meta.update(kmer=motif,label=str(mapped),original_label=meta["original_label"],
                    original_kmer=meta["original_kmer"],motif=motif,source_label=str(source),output_label=str(mapped),
                    source_class_name=LABEL_NAMES[source],task="fiveclass_"+case,label_mapping_id=label_mapping(case)["mapping_id"],
                    input_feature_file=str(args.input),input_feature_row=str(ordinal),input_metadata_file=str(args.meta),
                    input_metadata_row=str(ordinal),selected_index=str(selected_indices[case]),split=split,
                    split_seed=str(args.split_seed),split_ratios=args.split_ratios,split_method=SPLIT_METHOD)
                fw,mw = writers[(case,split)]
                fw.writerow(out_feature);mw.writerow(out_meta)
                counts[(case,split,source)] += 1
                chosen = next(cursor,None)
    if chosen is not None:
        raise RuntimeError("第二遍读取未找到全部选中记录")
    return counts


def run(args):
    args.input = Path(args.input).expanduser().resolve()
    args.meta = Path(args.meta).expanduser().resolve()
    args.output_dir = Path(args.output_dir).expanduser().resolve()
    args.normalized_ratios = parse_split_ratios(args.split_ratios)
    args.integer_weights = integer_weights(args.split_ratios)
    if args.log_every < 0 or not math.isfinite(args.log_seconds) or args.log_seconds<=0:
        raise ValueError("log_every 需要非负，log_seconds 需要有限正数")
    if not args.cases or len(set(args.cases))!=len(args.cases):
        raise ValueError("cases 不能为空或重复")
    if args.input==args.meta or not args.input.is_file() or not args.meta.is_file():
        raise ValueError("需要原六分类合并 CSV 及其不同文件、逐行配对的 metadata；不会寻找 positive_meta")
    if args.output_dir.exists():
        raise FileExistsError("输出目录已存在，不会覆盖："+str(args.output_dir))
    if args.output_dir in (args.input,args.meta,args.input.parent,args.meta.parent) or args.output_dir in args.input.parents or args.output_dir in args.meta.parents:
        raise ValueError("输出目录不能覆盖输入或其父目录")
    args.output_dir.parent.mkdir(parents=True,exist_ok=True)
    input_header = read_metadata_header(args.meta)
    stage = Path(tempfile.mkdtemp(prefix="."+args.output_dir.name+".partial-",dir=args.output_dir.parent))
    con = None
    try:
        fingerprints = {"features":fingerprint(args.input),"metadata":fingerprint(args.meta)}
        json_write(stage/"requested_config.json",{k:str(v) if isinstance(v,Path) else v for k,v in vars(args).items()})
        con = open_database(stage/"selection.sqlite")
        counts,source_counts,total = scan_input(args,stage,con)
        con.execute("CREATE TABLE task_selections (case_name TEXT NOT NULL, ordinal INTEGER NOT NULL, split TEXT NOT NULL, PRIMARY KEY(case_name,ordinal))")
        con.execute("CREATE INDEX task_selections_ordinal ON task_selections(ordinal,case_name)")
        summaries,expected_counts,failed_cases = {},{},[]
        for case in args.cases:
            args.case,args.source_labels = case,TASK_SOURCE_LABELS[case]
            folder = stage/("fiveclass_"+case);folder.mkdir()
            mapping = label_mapping(case);json_write(folder/"label_mapping.json",mapping)
            selected_counts = Counter({key:n for key,n in counts.items() if key[1] in args.source_labels})
            quotas,motif_report,reference_report,retained = make_quotas(args,selected_counts)
            write_task_reports(folder,motif_report,reference_report)
            if not retained:
                failed_cases.append(case)
                continue
            con.execute("UPDATE windows SET selected=0")
            selected = select_windows(args,con,quotas)
            con.execute("INSERT INTO task_selections SELECT ?,ordinal,split FROM windows WHERE selected=1",(case,))
            con.commit()
            write_task_reports(folder,motif_report,reference_report,selected)
            export_task_audits(args,folder,con,source_counts)
            per_class = Counter({(split,label):n for label,split,n in con.execute("SELECT label,split,count(*) FROM windows WHERE selected=1 GROUP BY label,split")})
            expected_counts[case] = per_class
            summaries[case] = task_summary(args,con,fingerprints,total,selected_counts,retained,per_class)
            print("[均衡 {}] 保留 {} / {} motif；选中 {:,} 窗口".format(case,len(retained),len({key[0] for key in selected_counts}),sum(per_class.values())),file=sys.stderr,flush=True)
        if failed_cases:
            raise ValueError("任务 {} 没有 motif 能满足五类、固定 read 分区、参考均衡及精确比例；诊断报告已保留。可检查瓶颈或显式只运行 --cases X/--cases Y，不能挪动 read 凑配额。".format(",".join(failed_cases)))
        observed = export_all_tasks(args,stage,con,input_header)
        for case,expected in expected_counts.items():
            if expected != Counter({(split,label):n for (task,split,label),n in observed.items() if task==case}):
                raise RuntimeError("输出数量与已验证配额不一致："+case)
        for name,path in (("features",args.input),("metadata",args.meta)):
            if fingerprint(path)!=fingerprints[name]:
                raise RuntimeError("两遍读取之间输入发生变化："+str(path))
        cross = con.execute("SELECT count(*) FROM (SELECT w.read_id FROM task_selections s JOIN windows w ON s.ordinal=w.ordinal GROUP BY w.read_id HAVING count(DISTINCT s.split)>1)").fetchone()[0]
        if cross:
            raise RuntimeError("不同任务中的同一 read 出现分区冲突")
        # The per-task table is authoritative; selected flags now mean union of all tasks.
        con.execute("UPDATE windows SET selected=EXISTS(SELECT 1 FROM task_selections s WHERE s.ordinal=windows.ordinal)")
        con.execute("UPDATE reads SET selected_windows=(SELECT count(*) FROM windows w WHERE w.read_id=reads.read_id AND w.selected=1)")
        con.commit();con.close();con=None
        for case,summary in summaries.items():
            folder=stage/("fiveclass_"+case)
            summary["output_fingerprints"]={p.name:{"size_bytes":p.stat().st_size,"sha256":sha256(p)} for p in sorted(folder.iterdir()) if p.is_file()}
            summary["selection_database"]={"file":"../selection.sqlite","task_selections_case_name":case,"windows_selected_flag":"union of selected windows across all tasks"}
            json_write(folder/"summary.json",summary)
        root_summary={"status":"complete","script_version":VERSION,"created_utc":datetime.now(timezone.utc).isoformat(),
            "input_fingerprints":fingerprints,"input_windows":total,"input_class_counts":{str(label):sum(n for key,n in counts.items() if key[1]==label) for label in range(6)},
            "code_sha256":sha256(Path(__file__)),"tasks":{case:{"directory":"fiveclass_"+case,"label_mapping":label_mapping(case),"retained_motif_count":summary["retained_motif_count"],"split_counts":summary["split_counts"]} for case,summary in summaries.items()},
            "same_read_in_different_splits_across_tasks":cross,"all_sixclass_input_rows_validated_before_filtering":True,
            "split_assignments_reused_without_task_specific_hashing":True,
            "positive_identity_note":"Provided row proxies retain their original IDs and disclosed experimental assumption; they do not establish original positive-read independence.",
            "selection_database":"selection.sqlite; task_selections(case_name,ordinal,split) records both tasks; windows.selected is their union",}
        root_summary["output_fingerprints"]={p.relative_to(stage).as_posix():{"size_bytes":p.stat().st_size,"sha256":sha256(p)} for p in sorted(stage.rglob("*")) if p.is_file()}
        json_write(stage/"summary.json",root_summary)
        if args.output_dir.exists():
            raise FileExistsError("输出目录在处理中被创建，拒绝覆盖")
        os.rename(stage,args.output_dir)
        print("[完成] "+str(args.output_dir),file=sys.stderr,flush=True)
        for case,summary in summaries.items():
            print("  {}：0=A,1=T,2=C,3=G,4={}".format("fiveclass_"+case,case),file=sys.stderr)
            for split,info in summary["split_counts"].items():
                print("    {}：{:,} 窗口，五类各 {:,}".format(split,info["windows"],info["per_class"]["0"]),file=sys.stderr)
        return root_summary
    except BaseException as error:
        if con is not None:con.close()
        json_write(stage/"error.json",{"status":"failed","type":type(error).__name__,"error":str(error),"traceback":traceback.format_exc()})
        failed=args.output_dir.with_name(args.output_dir.name+".failed-"+datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S")+"-"+stage.name.rsplit("-",1)[-1])
        os.rename(stage,failed)
        print("[失败诊断] "+str(failed),file=sys.stderr,flush=True)
        raise


def parser():
    p=argparse.ArgumentParser(description=__doc__,formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    p.add_argument("--input",default=str(DEFAULT_INPUT),help="原六分类合并 CSV，不是已下采样的 train/validation/test")
    p.add_argument("--meta",default=None,help="与合并 CSV 同次生成、逐行对应的 _meta.tsv；不会寻找 positive_meta")
    p.add_argument("--output_dir",default=None,help="默认 input 父目录/fiveclass_overlap_read_split_721；内部创建 fiveclass_X/Y")
    p.add_argument("--cases",nargs="+",choices=("X","Y"),default=["X","Y"])
    p.add_argument("--split_seed",type=int,default=20260429)
    p.add_argument("--sample_seed",type=int,default=20260429)
    p.add_argument("--split_ratios",default="7,2,1")
    p.add_argument("--reference_balance",choices=("equal","none"),default="equal")
    p.add_argument("--log_every",type=int,default=100000)
    p.add_argument("--log_seconds",type=float,default=30.0)
    p.add_argument("--version",action="version",version="%(prog)s "+VERSION)
    return p


def main(argv=None):
    args=parser().parse_args(argv)
    args.meta=args.meta or str(Path(args.input).with_name(Path(args.input).stem+"_meta.tsv"))
    args.output_dir=args.output_dir or str(Path(args.input).parent/"fiveclass_overlap_read_split_721")
    try:
        run(args)
    except (Exception,KeyboardInterrupt) as error:
        print("[ERROR] "+str(error),file=sys.stderr,flush=True)
        return 130 if isinstance(error,KeyboardInterrupt) else 1
    return 0


if __name__=="__main__":
    sys.exit(main())

