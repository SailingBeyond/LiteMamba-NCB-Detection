#!/usr/bin/env python3
"""六类固定分区与精确均衡；阴性按真实 read，缺少 metadata 的阳性按已声明行代理。"""
from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
import os
import shutil
import sqlite3
import sys
import tempfile
import time
import traceback
from collections import Counter, defaultdict
from contextlib import ExitStack
from datetime import datetime, timezone
from fractions import Fraction
from decimal import Decimal, InvalidOperation
from pathlib import Path

from sixclass_io import (FEATURE_HEADER, SPLIT_METHOD, SPLIT_NAMES, assign_read_split,
                         parse_split_ratios, iter_feature_meta, metadata_header, normalize_known,
                         is_positive_row_proxy)

VERSION = "1.1.0"
BASE = Path("/path/to/revision-inputs/CPLX-XNA-replenishment/CPLX-XNA-replenishment/fast5_single_guppy")
DEFAULT_INPUT = BASE / "xna_pc_6mer_features_all_replenishment_6_overlap.csv"
LABEL_NAMES = {0: "A", 1: "T", 2: "C", 3: "G", 4: "X", 5: "Y"}
EXTRA_META = ["kmer", "original_kmer", "original_label", "motif", "input_feature_file", "input_feature_row", "input_metadata_file", "input_metadata_row", "selected_index", "split", "split_seed", "split_ratios", "split_method"]


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
        ks = [len(references[(motif, label)]) for label in range(6)]
        absent = [str(label) for label, k in enumerate(ks) if not k]
        lattice = math.lcm(*(k for k in ks if k)) if all(ks) else 0
        constraints = []
        if absent:
            t_max = t = 0
            reason = "missing_labels:" + ",".join(absent)
        elif args.reference_balance == "equal":
            for label in range(6):
                k = ks[label]
                for chrom in sorted(references[(motif, label)]):
                    for split, weight in zip(SPLIT_NAMES, args.integer_weights):
                        capacity = counts[(motif, label, chrom, split)] * k // weight
                        constraints.append((capacity, label, chrom, split))
            t_max = min(value[0] for value in constraints)
            t = t_max // lattice * lattice
            reason = "retained" if t else "insufficient_read_partition_or_reference_capacity"
        else:
            for label in range(6):
                for split, weight in zip(SPLIT_NAMES, args.integer_weights):
                    constraints.append((totals[(motif, label, split)] // weight, label, "all", split))
            t_max = min(value[0] for value in constraints)
            t = t_max
            lattice = 1
            reason = "retained" if t else "insufficient_read_partition_capacity"
        limiting = [list(value) for value in constraints if value[0] == t_max]
        retained.append(motif) if t else None
        for label in range(6):
            known_reference_count = sum(normalize_known(chrom) != "unknown" for chrom in references[(motif, label)])
            for split, weight in zip(SPLIT_NAMES, args.integer_weights):
                available = totals[(motif, label, split)]
                target = weight * t
                motif_report.append((motif, label, LABEL_NAMES[label], split, available, target, available-target, known_reference_count, lattice, t_max, t, "retained" if t else "dropped", reason, json.dumps(limiting, ensure_ascii=False), ks[label]))
                if args.reference_balance == "none":
                    if target:
                        quotas.append((motif, label, None, split, target))
                for chrom in sorted(references[(motif, label)]):
                    ref_target = target // ks[label] if args.reference_balance == "equal" else None
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


def write_reports(stage, motif_report, reference_report, selected=None):
    csv_write(stage / "motif_balance_report.csv", ["motif", "label", "class_name", "split", "available", "selected", "discarded", "references_in_class", "lcm_reference_counts", "maximum_t_before_rounding", "selected_t", "status", "reason", "limiting_constraints_json", "sampling_reference_buckets"], motif_report)
    for row in reference_report:
        row[7] = (selected or {}).get((row[0], row[1], row[3], row[4]), 0)
    csv_write(stage / "reference_balance_report.csv", ["motif", "label", "class_name", "chrom", "split", "available", "sampling_bucket_target", "selected", "reference_balance_requested", "reason", "reference_identity_known", "reference_balance_status", "equal_reference_balance_applied"], reference_report)


def export_selected(args, stage, con, input_header):
    output_header = list(input_header) + [name for name in EXTRA_META if name not in input_header]
    output_counts = Counter()
    selected_cursor = iter(con.execute("SELECT ordinal,split FROM windows WHERE selected=1 ORDER BY ordinal"))
    next_selected = next(selected_cursor, None)
    selected_index = 0
    with ExitStack() as stack:
        writers = {}
        for split in SPLIT_NAMES:
            feature_handle = stack.enter_context(open(stage / (split + ".csv"), "w", encoding="utf-8", newline=""))
            meta_handle = stack.enter_context(open(stage / (split + "_meta.tsv"), "w", encoding="utf-8", newline=""))
            feature_writer = csv.writer(feature_handle)
            meta_writer = csv.DictWriter(meta_handle, fieldnames=output_header, delimiter="\t", extrasaction="raise")
            feature_writer.writerow(FEATURE_HEADER)
            meta_writer.writeheader()
            writers[split] = feature_writer, meta_writer
        for ordinal, feature, meta, motif in iter_input(args.input, args.meta):
            # Audit *every* row again, even those not selected. No early stop at last selected row.
            split = validate_assignment(meta, args, ordinal)
            if next_selected is None or ordinal != next_selected[0]:
                continue
            if split != next_selected[1]:
                raise RuntimeError("两遍读取之间的分区不一致")
            original = meta.get("original_kmer") or feature[0]
            feature[0] = motif
            for field in ("kmer", "output_kmer"):
                if field in meta:
                    meta[field] = motif
            selected_index += 1
            meta.update({"original_kmer": original, "motif": motif, "input_feature_file": str(args.input), "input_feature_row": str(ordinal), "input_metadata_file": str(args.meta), "input_metadata_row": str(ordinal), "selected_index": str(selected_index), "split": split, "split_seed": str(args.split_seed), "split_ratios": args.split_ratios, "split_method": SPLIT_METHOD})
            feature_writer, meta_writer = writers[split]
            feature_writer.writerow(feature)
            meta_writer.writerow(meta)
            output_counts[(split, int(feature[-1]))] += 1
            next_selected = next(selected_cursor, None)
    if next_selected is not None:
        raise RuntimeError("第二遍读取未能找到所有已选中窗口")
    return output_counts


def export_audits(args, stage, con, source_counts):
    csv_write(stage / "read_splits.tsv", ["read_id", "split", "input_windows", "selected_windows", "read_id_kind", "split_seed", "split_ratios", "split_method"], (tuple(row) + (args.split_seed, args.split_ratios, SPLIT_METHOD) for row in con.execute("SELECT read_id,split,input_windows,selected_windows,read_id_kind FROM reads ORDER BY read_id")), delimiter="\t")
    selected_source = con.execute("SELECT source,strand,run_id,chrom,label,split,count(*) FROM windows WHERE selected=1 GROUP BY source,strand,run_id,chrom,label,split")
    rows = [tuple(key) + (count,) for key, count in sorted(source_counts.items())]
    rows.extend(("selected",) + tuple(row) for row in selected_source)
    csv_write(stage / "source_strand_run_template_counts.csv", ["stage", "source_dataset", "bam_strand", "run_id", "chrom", "label", "split", "windows"], rows)
    for category, column in (("source", "source"), ("strand", "strand"), ("run", "run_id"), ("template", "chrom")):
        rows = con.execute("SELECT CASE selected WHEN 1 THEN 'selected' ELSE 'not_selected' END," + column + ",label,split,count(*),count(DISTINCT read_id) FROM windows GROUP BY selected," + column + ",label,split ORDER BY selected," + column + ",label,split")
        csv_write(stage / (category + "_class_counts.csv"), ["selection", category, "label", "split", "windows", "unique_reads"], rows)
    csv_write(stage / "selected_provenance.tsv", ["selected_index", "input_feature_row", "read_id", "motif", "label", "chrom", "split", "source_dataset", "bam_strand", "run_id", "read_id_kind", "reference_identity_known"], ((index,) + tuple(row) for index, row in enumerate(con.execute("SELECT ordinal,read_id,motif,label,chrom,split,source,strand,run_id,read_id_kind,reference_identity_known FROM windows WHERE selected=1 ORDER BY ordinal"), 1)), delimiter="\t")


def read_window_distribution(con, role, selected=False):
    column = "selected_windows" if selected else "input_windows"
    where = "role=?" + (" AND selected_windows>0" if selected else "")
    n = con.execute("SELECT count(*) FROM reads WHERE " + where, (role,)).fetchone()[0]
    if not n:
        return {"reads": 0, "min": None, "p50": None, "p95": None, "max": None}
    at = lambda q: con.execute("SELECT " + column + " FROM reads WHERE " + where + " ORDER BY " + column + " LIMIT 1 OFFSET ?", (role, math.floor(q * (n - 1)))).fetchone()[0]
    return {"reads": n, "min": at(0), "p50": at(0.5), "p95": at(0.95), "max": at(1), "quantile_method": "lower order statistic at floor(q*(n-1))"}


def positive_identity_audit(con):
    """Keep genuine input IDs separate from identifiers based only on CSV rows."""
    count = lambda extra="": con.execute("SELECT count(*) FROM reads WHERE role=2 " + extra).fetchone()[0]
    real = count("AND read_id_kind!='positive_row_proxy'")
    proxy = count("AND read_id_kind='positive_row_proxy'")
    return {
        "real_read_id_count": real,
        "row_proxy_count": proxy,
        "selected_real_read_id_count": count("AND selected_windows>0 AND read_id_kind!='positive_row_proxy'"),
        "selected_row_proxy_count": count("AND selected_windows>0 AND read_id_kind='positive_row_proxy'"),
        "positive_one_window_per_read_assumed": bool(proxy),
        "positive_one_window_per_real_read_verified": not bool(proxy),
        "provided_real_read_ids_one_window_each": True,
        "actual_rawpositiveoriginalread_overlap_check": "unavailable" if proxy else "verified_for_provided_real_read_ids",
        "basis": "assumed_one_csv_row_per_positive_read" if proxy else "provided_row_aligned_real_read_ids",
        "limitation": ("Positive CSV row proxies identify input records, not original sequencing reads. One positive row per read is the user's experimental-design assumption; original positive read overlap cannot be independently checked." if proxy else "Checks concern the provided extraction read IDs; biological independence is not established by read IDs alone."),
    }


def create_summary(args, stage, con, input_fingerprints, total, retained, output_counts):
    splits = {}
    for split in SPLIT_NAMES:
        per_class = {str(label): output_counts[(split, label)] for label in range(6)}
        if len(set(per_class.values())) != 1:
            raise RuntimeError("分区内六类数量不相等: " + split)
        splits[split] = {"windows": sum(per_class.values()), "per_class": per_class, "unique_reads": con.execute("SELECT count(*) FROM reads WHERE split=? AND selected_windows>0", (split,)).fetchone()[0]}
    overlap = con.execute("SELECT count(*) FROM (SELECT read_id FROM windows WHERE selected=1 GROUP BY read_id HAVING count(DISTINCT split)>1)").fetchone()[0]
    if overlap:
        raise RuntimeError("输出中发现跨分区 read")
    unique_templates = {str(label): con.execute("SELECT count(DISTINCT chrom) FROM windows WHERE selected=1 AND label=? AND reference_identity_known=1", (label,)).fetchone()[0] for label in range(6)}
    unique_runs = {str(label): con.execute("SELECT count(DISTINCT run_id) FROM windows WHERE selected=1 AND label=? AND run_id!='unknown'", (label,)).fetchone()[0] for label in range(6)}
    unknown_runs = con.execute("SELECT count(*) FROM windows WHERE selected=1 AND run_id='unknown'").fetchone()[0]
    config = {name: str(value) if isinstance(value, Path) else value for name, value in vars(args).items()}
    summary = {"status": "complete", "script_version": VERSION, "created_utc": datetime.now(timezone.utc).isoformat(), "config": config, "label_names": LABEL_NAMES, "input_fingerprints": input_fingerprints, "code_sha256": {"script": sha256(Path(__file__)), "sixclass_io": sha256(Path(__file__).with_name("sixclass_io.py"))}, "input_windows": total, "selected_windows": sum(value["windows"] for value in splits.values()), "input_unique_reads": con.execute("SELECT count(*) FROM reads").fetchone()[0], "selected_unique_reads": con.execute("SELECT count(*) FROM reads WHERE selected_windows>0").fetchone()[0], "retained_motif_count": len(retained), "retained_motifs": retained, "split_counts": splits, "read_ids_in_multiple_splits": overlap, "read_window_audit": {"positive_input": read_window_distribution(con, 2), "positive_selected": read_window_distribution(con, 2, True), "negative_input": read_window_distribution(con, 1), "negative_selected": read_window_distribution(con, 1, True), "positive_one_window_per_read": True, "cross_positive_negative_read_id_count": 0}, "physical_window_duplicate_audit": {"input_windows_with_complete_identity": con.execute("SELECT count(*) FROM windows WHERE physical_key IS NOT NULL").fetchone()[0], "input_windows_without_complete_identity": con.execute("SELECT count(*) FROM windows WHERE physical_key IS NULL").fetchone()[0], "duplicate_complete_identities": 0, "key": ["read_id", "chrom", "bam_strand", "kmer_ref_start", "kmer_ref_end"], "limitation": "Rows lacking complete window coordinates/strand can only be guaranteed sampled once by input row, not proven physically distinct."}, "known_runs_per_class": unique_runs, "templates_per_class": unique_templates, "selected_windows_with_unknown_run": unknown_runs, "split_contract": {"unit": "read_id", "method": SPLIT_METHOD, "seed": args.split_seed, "ratio_spec": args.split_ratios, "hash_boundaries_uint64": [int(args.normalized_ratios[0] * (1 << 64)), int((args.normalized_ratios[0] + args.normalized_ratios[1]) * (1 << 64))], "window_ratio": ":".join(map(str, args.integer_weights)), "read_ratios_are_expected_not_exact": True}, "sampling": {"without_replacement": True, "priority": "BLAKE2b-128(sample_seed + NUL + original_1_based_row), person=sixclass-sample; ascending bytes then ordinal", "order": "original input row order, no signal shuffling; shuffle training samples using the DataLoader", "reference_balance": args.reference_balance}, "scope_and_limits": ["Read-disjoint evaluation; overlapping windows within a read remain in its one assigned partition.", "Exact window quotas do not imply exact unique-read ratios or independent windows.", "Read separation does not prove held-out-template, held-out-run or biological-replicate generalization.", "Source/strand/run/template by class tables expose confounding; unknown run IDs are not inferred from filenames.", "Fourth position is N for every output feature row; original kmer and original metadata are retained.", "Canonical labels0..3/motifs may share a read in one split. Positive labels4/5 must have exactly one window per read, and may not share read IDs with canonical data."], "outputs": {split: {"features": split + ".csv", "metadata": split + "_meta.tsv"} for split in SPLIT_NAMES}}


    positive = positive_identity_audit(con)
    unknown_refs = {str(label): con.execute("SELECT count(*) FROM windows WHERE selected=1 AND label=? AND reference_identity_known=0", (label,)).fetchone()[0] for label in range(6)}
    reference_audit = {}
    for role, condition in (("positive", "label>=4"), ("negative", "label<4")):
        known, unknown = con.execute("SELECT coalesce(sum(reference_identity_known),0),coalesce(sum(1-reference_identity_known),0) FROM windows WHERE selected=1 AND " + condition).fetchone()
        if unknown:
            status = "not_performed_unknown_positive_reference_identity" if role == "positive" and not known else "partially_unavailable_unknown_reference_identity"
        else:
            status = "equal_within_motif_and_class" if args.reference_balance == "equal" else "not_requested"
        reference_audit[role] = {"status": status, "known_reference_rows": known, "unknown_reference_rows": unknown,
                                 "limitation": "Unknown reference labels are sampling buckets only and are excluded from true template counts; their underlying references cannot be balanced or audited." if unknown else None}
    summary["positive"] = positive
    summary["read_id_overlap_check_scope"] = "provided_real_read_ids_and_positive_row_proxies" if positive["row_proxy_count"] else "provided_real_read_ids"
    summary["all_original_positive_read_ids_verified"] = not bool(positive["row_proxy_count"])
    summary["unique_read_count_unit"] = "provided_real_read_ids_or_positive_csv_row_proxies" if positive["row_proxy_count"] else "provided_real_read_ids"
    summary["read_window_audit"].update({
        "positive_one_window_per_read": None if positive["row_proxy_count"] else True,
        "positive_one_window_per_read_assumed": positive["positive_one_window_per_read_assumed"],
        "positive_one_window_per_real_read_verified": positive["positive_one_window_per_real_read_verified"],
        "actual_rawpositiveoriginalread_overlap_check": positive["actual_rawpositiveoriginalread_overlap_check"],
        "cross_positive_negative_read_id_count": None if positive["row_proxy_count"] else 0,
        "cross_positive_negative_provided_identifier_collision_count": 0,
        "distribution_count_unit": summary["unique_read_count_unit"],
    })
    summary["selected_windows_with_unknown_reference_by_class"] = unknown_refs
    summary["selected_windows_with_unknown_reference"] = sum(unknown_refs.values())
    summary["sampling"]["reference_balance_audit"] = reference_audit
    summary["split_contract"]["unit"] = summary["unique_read_count_unit"]
    summary["scope_and_limits"] = [
        "Canonical windows are separated by provided original read IDs; overlapping windows from each canonical read remain in one fixed partition.",
        "Positive rows without metadata are separated by explicit CSV row proxies under the user's one-row-per-positive-read assumption. These proxies are not original read IDs and cannot verify the original positive read overlap.",
        "Exact window quotas do not imply exact unique-read ratios or independent windows.",
        "Read or row-proxy separation does not establish held-out-template, held-out-run or biological-replicate generalization.",
        "Reference balancing is performed only where true reference identities are supplied. An unknown positive-reference bucket is not a reference and does not prove equal positive template representation.",
        "Source/strand/run/template by class tables expose available provenance. Missing positive template, strand and run identities remain unknown; they are never inferred from sequences or filenames.",
        "Fourth position is N for every output feature row; original kmer and original metadata are retained.",
        "Canonical labels0..3 may share a read within one split. Genuine positive read IDs, when supplied, must have one window each and may not collide with canonical read IDs. Positive row proxies are each sampled at most once, which checks input-row reuse only.",
    ]
    if positive["row_proxy_count"]:
        summary["physical_window_duplicate_audit"]["limitation"] += " Positive row proxies cannot establish physical window identity or original-read distinctness."
    return summary


def run(args):
    args.input = Path(args.input).resolve()
    args.meta = Path(args.meta).resolve()
    args.output_dir = Path(args.output_dir).resolve()
    args.normalized_ratios = parse_split_ratios(args.split_ratios)
    args.integer_weights = integer_weights(args.split_ratios)
    if args.log_every < 0 or not math.isfinite(args.log_seconds) or args.log_seconds <= 0:
        raise ValueError("log_every 需要非负，log_seconds 需要有限正数")
    if args.input == args.meta:
        raise ValueError("CSV 和 metadata 不能为同一文件")
    if not args.input.is_file() or not args.meta.is_file():
        raise ValueError("特征 CSV 或配对 metadata 不存在；必须先运行新版 combine_csv_header.py")
    if args.output_dir.exists():
        raise FileExistsError("输出目录已存在，不会覆盖: " + str(args.output_dir))
    if args.output_dir == args.input.parent or args.output_dir in (args.input, args.meta):
        raise ValueError("输出目录不能覆盖输入")
    args.output_dir.parent.mkdir(parents=True, exist_ok=True)
    header = read_metadata_header(args.meta)
    stage = Path(tempfile.mkdtemp(prefix="." + args.output_dir.name + ".partial-", dir=args.output_dir.parent))
    con = None
    try:
        input_fingerprints = {"features": fingerprint(args.input), "metadata": fingerprint(args.meta)}
        json_write(stage / "requested_config.json", {key: str(value) if isinstance(value, Path) else value for key, value in vars(args).items()})
        con = open_database(stage / "selection.sqlite")
        counts, source_counts, total = scan_input(args, stage, con)
        quotas, motif_report, reference_report, retained = make_quotas(args, counts)
        write_reports(stage, motif_report, reference_report)
        if not retained:
            raise ValueError("没有 motif 能同时满足六类、固定 read 分区和精确比例约束。已保存各 motif/ref 的瓶颈报告；不能把 read 重新划分来凑配额。")
        print("[均衡] 输入 {:,} 行，保留 {} / {} motif，抽样 strata {}".format(total, len(retained), len({key[0] for key in counts}), len(quotas)), file=sys.stderr, flush=True)
        selected = select_windows(args, con, quotas)
        write_reports(stage, motif_report, reference_report, selected)
        output_counts = export_selected(args, stage, con, header)
        export_audits(args, stage, con, source_counts)
        for name, path in (("features", args.input), ("metadata", args.meta)):
            current = fingerprint(path)
            if current != input_fingerprints[name]:
                raise RuntimeError("两遍读取之间输入文件发生变化: " + str(path))
        summary = create_summary(args, stage, con, input_fingerprints, total, retained, output_counts)
        con.close()
        con = None
        outputs = {}
        for path in sorted(stage.iterdir()):
            if path.is_file():
                outputs[path.name] = {"size_bytes": path.stat().st_size, "sha256": sha256(path)}
        summary["output_fingerprints"] = outputs
        json_write(stage / "summary.json", summary)
        if args.output_dir.exists():
            raise FileExistsError("输出目录在处理中被创建，拒绝覆盖: " + str(args.output_dir))
        os.rename(stage, args.output_dir)
        print("[完成] " + str(args.output_dir), file=sys.stderr)
        if summary["positive"]["row_proxy_count"]:
            print("[阳性假设] 每条阳性 CSV 记录视为一个 read；行代理不是原始 read_id，阳性原始 read 去重/跨集合重叠及模板均衡不可核验，已在报告中标记。", file=sys.stderr)
        for split, values in summary["split_counts"].items():
            unit = "read/行代理标识组" if summary["positive"]["row_proxy_count"] else "reads"
            print("  {}: {:,} 窗口，{:,} {}，六类各 {:,}".format(split, values["windows"], values["unique_reads"], unit, values["per_class"]["0"]), file=sys.stderr)
        return summary
    except Exception as error:
        if con is not None:
            con.close()
        json_write(stage / "error.json", {"status": "failed", "error": str(error), "type": type(error).__name__, "traceback": traceback.format_exc()})
        failed = args.output_dir.with_name(args.output_dir.name + ".failed-" + datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S") + "-" + stage.name.rsplit("-", 1)[-1])
        os.rename(stage, failed)
        print("[失败诊断] " + str(failed), file=sys.stderr)
        raise


def parser():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--input", default=str(DEFAULT_INPUT))
    p.add_argument("--meta", default=None, help="默认为 input 同名 _meta.tsv；使用新版合并脚本生成的配对 metadata，阳性可携带明确标注的 CSV 行代理")
    p.add_argument("--output_dir", default=None, help="默认 input 父目录/sixclass_overlap_read_split_721；已有目录不覆盖")
    p.add_argument("--split_seed", type=int, default=20260429)
    p.add_argument("--split_ratios", default="7,2,1")
    p.add_argument("--sample_seed", type=int, default=20260429)
    p.add_argument("--reference_balance", choices=("equal", "none"), default="equal")
    p.add_argument("--log_every", type=int, default=100000)
    p.add_argument("--log_seconds", type=float, default=30.0)
    return p


def main(argv=None):
    args = parser().parse_args(argv)
    args.meta = args.meta or str(Path(args.input).with_name(Path(args.input).stem + "_meta.tsv"))
    args.output_dir = args.output_dir or str(Path(args.input).parent / "sixclass_overlap_read_split_721")
    try:
        run(args)
    except Exception as error:
        print("[ERROR] " + str(error), file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    sys.exit(main())
