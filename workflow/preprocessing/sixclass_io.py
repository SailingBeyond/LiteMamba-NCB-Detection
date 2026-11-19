#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Streaming validation shared by the six-class merge and split scripts.

Negative windows retain genuine extraction read IDs. Positive CSV-only inputs
can use explicit row proxy IDs under the stated one-row-per-read assumption;
these IDs and their unknown reference identity are never presented as observed.
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

# Native csv defaults to a 128 KiB cell limit; long nanopore signal fields
# legitimately exceed it. Raising this parser limit does not read any files.
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
