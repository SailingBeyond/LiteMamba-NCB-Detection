#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Merge six-class XNA positives and overlapping E. coli negatives.

Positives default to CSV-only mode, assuming one row per positive read as
specified by the experimental design. Stable, explicitly marked row proxies
support reproducible splitting without claiming an observed read UUID, template
or run. Negatives still require extraction metadata and preserve read groups.
The fourth base is N; A/T/C/G/X/Y labels remain 0/1/2/3/4/5.
"""
from __future__ import annotations

import argparse
import csv
import hashlib
import json
import os
from pathlib import Path
import sqlite3
import sys
import tempfile
from collections import Counter

VERSION = "2.1.0"
DEFAULT_BASE = Path("/path/to/revision-inputs/CPLX-XNA-replenishment/CPLX-XNA-replenishment/fast5_single_guppy")
DEFAULT_POSITIVE = DEFAULT_BASE / "xna_6mer_features_all_replenishment.csv"
DEFAULT_NEGATIVE = Path("/path/to/revision-results/ecoli_er2925_balanced_by_xna_6mer_features_overlap.csv")
DEFAULT_OUTPUT = DEFAULT_BASE / "xna_pc_6mer_features_all_replenishment_6_overlap.csv"

from sixclass_io import (FEATURE_HEADER, SPLIT_NAMES, SPLIT_METHOD, BASE_META_HEADER,
                         exact_label, mask_kmer, parse_split_ratios, assign_read_split,
                         canonical_split, resolve_metadata, metadata_header,
                         iter_feature_meta, iter_features, validate_or_assign_split, normalize_known)

def file_fingerprint(path):
    before = Path(path).stat()
    digest = hashlib.sha256()
    with open(path, "rb") as stream:
        for chunk in iter(lambda: stream.read(8 * 1024 * 1024), b""):
            digest.update(chunk)
    stat = Path(path).stat()
    if (before.st_size, before.st_mtime_ns, before.st_ino) != (stat.st_size, stat.st_mtime_ns, stat.st_ino):
        raise ValueError("SHA256计算期间输入发生变化: {}".format(path))
    return {"path": str(Path(path).resolve()), "size_bytes": stat.st_size,
            "mtime_ns": stat.st_mtime_ns, "sha256": digest.hexdigest()}


def _atomic_staging(outputs, inputs, overwrite=False):
    resolved = [Path(path).expanduser().resolve() for path in outputs]
    if len(set(resolved)) != len(resolved):
        raise ValueError("CSV、metadata和summary输出路径不能相同")
    input_set = {Path(path).resolve() for path in inputs}
    for path in resolved:
        if path in input_set:
            raise ValueError("输出不得覆盖输入: {}".format(path))
        if path.exists() and not overwrite:
            raise ValueError("输出已存在: {}；请使用新名称或显式--overwrite".format(path))
        path.parent.mkdir(parents=True, exist_ok=True)
    staging = []
    for path in resolved:
        fd, staged = tempfile.mkstemp(prefix="." + path.name + ".", suffix=".tmp", dir=str(path.parent))
        os.close(fd)
        staging.append(Path(staged))
    return resolved, staging


def merge_csv_files_stream(csv_files, output_csv, buffer_size=16 * 1024 * 1024,
                           meta_files=None, output_meta=None, summary_out=None,
                           split_seed=20260429, split_ratios="7,2,1", overwrite=False):
    """Merge [positive, negative], with optional explicit positive metadata.

    With no positive metadata argument, no positive metadata discovery occurs.
    Its rows receive disclosed SHA256/record-ordinal proxy identities. Genuine
    negative extraction metadata remains mandatory and is auto-discovered when
    not given. The legacy buffer_size argument remains accepted.
    """
    del buffer_size
    if len(csv_files) != 2:
        raise ValueError("必须提供两个输入，顺序为阳性XNA、阴性E. coli")
    features = [Path(path).expanduser().resolve() for path in csv_files]
    for path in features:
        if not path.is_file() or path.stat().st_size == 0:
            raise ValueError("输入不存在或为空（不会跳过）: {}".format(path))
    if len(set(features)) != 2:
        raise ValueError("阳性和阴性输入不能为同一文件")
    meta_files = meta_files or (None, None)
    if len(meta_files) != 2:
        raise ValueError("meta_files必须按阳性、阴性提供两个路径")
    metas = [resolve_metadata(features[0], meta_files[0]) if meta_files[0] else None,
             resolve_metadata(features[1], meta_files[1])]
    positive_proxy_mode = metas[0] is None
    present_metas = [path for path in metas if path is not None]
    output_csv = Path(output_csv).expanduser().resolve()
    output_meta = Path(output_meta) if output_meta else output_csv.with_name(output_csv.stem + "_meta.tsv")
    summary_out = Path(summary_out) if summary_out else output_csv.with_name(output_csv.stem + "_merge_summary.json")
    ratios = parse_split_ratios(split_ratios)
    header = list(BASE_META_HEADER)
    for path in present_metas:
        for key in metadata_header(path):
            if key not in header:
                header.append(key)
    paths, staged = _atomic_staging([output_csv, output_meta, summary_out], features + present_metas, overwrite)
    counts = Counter()
    source_counts = Counter()
    split_counts = Counter()
    masked_count = 0
    total = 0
    read_database = None
    read_database_path = None
    try:
        # Genuine positive IDs are indexed only in explicit metadata mode.
        # Row proxies are reproducible identities, not verified read UUIDs.
        descriptor, database_name = tempfile.mkstemp(prefix=".sixclass_positive_reads.", suffix=".sqlite", dir=str(paths[0].parent))
        os.close(descriptor)
        read_database_path = Path(database_name)
        read_database = sqlite3.connect(database_name)
        read_database.execute("PRAGMA journal_mode=OFF")
        read_database.execute("PRAGMA synchronous=OFF")
        read_database.execute("CREATE TABLE positive_reads (read_id TEXT PRIMARY KEY, source_row INTEGER NOT NULL, chrom TEXT NOT NULL) WITHOUT ROWID")
        fingerprints = []
        for path in features + present_metas:
            print("[SHA256/合并前] {}".format(path), flush=True)
            fingerprints.append(file_fingerprint(path))
        positive_digest = fingerprints[0]["sha256"]
        with open(staged[0], "w", encoding="utf-8", newline="", buffering=8 * 1024 * 1024) as fout, \
             open(staged[1], "w", encoding="utf-8", newline="", buffering=8 * 1024 * 1024) as mout:
            writer = csv.writer(fout)
            meta_writer = csv.DictWriter(mout, fieldnames=header, delimiter="\t", extrasaction="raise")
            writer.writerow(FEATURE_HEADER)
            meta_writer.writeheader()
            specs = [("positive", "XNA_replenishment", {4, 5}), ("negative", "Ecoli_ER2925", {0, 1, 2, 3})]
            for feature, meta_path, (role, dataset_name, labels) in zip(features, metas, specs):
                if role == "positive" and positive_proxy_mode:
                    print("[合并] positive: {}（不查找metadata；按每条数据记录对应一条阳性read的实验假设分区）".format(feature), flush=True)
                    def proxy_rows():
                        for physical_row, normalized_row, original in iter_features(feature, labels):
                            ordinal = original["record_ordinal"]
                            proxy_meta = {"read_id": "positive-row:{}:{}".format(positive_digest, ordinal),
                                          "read_id_kind": "positive_row_proxy",
                                          "read_identity_basis": "assumed_one_csv_row_per_positive_read",
                                          "original_read_id": "unknown", "reference_identity_known": "false",
                                          "positive_source_sha256": positive_digest,
                                          "positive_data_record": str(ordinal),
                                          "chrom": "unknown", "bam_strand": "unknown", "run_id": "unknown",
                                          "source_dataset": "XNA_replenishment", "merge_input_role": "positive",
                                          "source_file": str(feature), "source_row": str(physical_row),
                                          "original_kmer": original["original_kmer"],
                                          "original_label": original["original_label"]}
                            yield physical_row, normalized_row, proxy_meta
                    rows = proxy_rows()
                else:
                    print("[合并] {}: {} + {}".format(role, feature, meta_path), flush=True)
                    rows = iter_feature_meta(feature, meta_path, labels)
                for line_no, row, meta in rows:
                    read_id = meta["read_id"]
                    if role == "positive" and not positive_proxy_mode:
                        if meta.get("read_id_kind") == "positive_row_proxy":
                            raise ValueError("{}:{}: 显式--positive_meta模式要求真实提取read身份，不能把已生成的行代理ID当作真实read核验".format(feature, line_no))
                        try:
                            read_database.execute("INSERT INTO positive_reads VALUES (?,?,?)", (read_id, line_no, meta["chrom"]))
                        except sqlite3.IntegrityError as exc:
                            prior = read_database.execute("SELECT source_row, chrom FROM positive_reads WHERE read_id=?", (read_id,)).fetchone()
                            raise ValueError("{}:{}: 阳性read_id重复={!r}，先前CSV行{}/chrom={}，当前chrom={}。按每条阳性read仅提取一个6mer的约定，拒绝继续；不会静默去重。".format(
                                feature, line_no, read_id, prior[0], prior[1], meta["chrom"])) from exc
                    elif role == "negative" and not positive_proxy_mode:
                        prior = read_database.execute("SELECT source_row, chrom FROM positive_reads WHERE read_id=?", (read_id,)).fetchone()
                        if prior:
                            raise ValueError("{}:{}: 阴阳性来源出现相同read_id={!r}；阳性CSV行{}/chrom={}，阴性chrom={}。请核实原始read身份；不能添加来源前缀伪造独立read。".format(
                                feature, line_no, read_id, prior[0], prior[1], meta["chrom"]))
                    label = exact_label(row[10])
                    motif = mask_kmer(row[0])
                    original_kmer, original_label = row[0], row[10]
                    try:
                        expected_split = validate_or_assign_split(meta, meta["read_id"], split_seed, split_ratios)
                    except ValueError as exc:
                        raise ValueError("{}:{}: {}".format(feature, line_no, exc)) from exc
                    meta["original_kmer"] = meta.get("original_kmer") or meta.get("raw_oriented_kmer") or original_kmer
                    meta["original_label"] = meta.get("original_label") or original_label
                    meta["source_file"] = meta.get("source_file") or str(feature)
                    meta["source_row"] = meta.get("source_row") or str(line_no)
                    existing_dataset = normalize_known(meta.get("source_dataset"))
                    meta["source_dataset"] = dataset_name if existing_dataset == "unknown" else existing_dataset
                    meta["run_id"] = normalize_known(meta.get("run_id"))
                    meta["bam_strand"] = normalize_known(meta.get("bam_strand"))
                    if not (role == "positive" and positive_proxy_mode):
                        meta["read_id_kind"] = meta.get("read_id_kind") or "extracted_read_id"
                        meta["read_identity_basis"] = meta.get("read_identity_basis") or "paired_extraction_metadata"
                        meta["original_read_id"] = meta.get("original_read_id") or read_id
                        meta["reference_identity_known"] = "true"
                    meta.update(kmer=motif, label=str(label), split=expected_split,
                                split_seed=str(split_seed), split_ratios=str(split_ratios), split_method=SPLIT_METHOD,
                                merged_input_file=str(feature), merged_input_row=str(line_no), merge_input_role=role)
                    row[0], row[10] = motif, str(label)
                    writer.writerow(row)
                    meta_writer.writerow(meta)
                    total += 1
                    masked_count += meta["original_kmer"].upper() != motif
                    counts[label] += 1
                    source_counts[role] += 1
                    split_counts[(expected_split, label)] += 1
                    if total % 100000 == 0:
                        print("[进度] {:,}条".format(total), flush=True)
                read_database.commit()
            fout.flush()
            mout.flush()
            os.fsync(fout.fileno())
            os.fsync(mout.fileno())
        if not source_counts["positive"] or not source_counts["negative"]:
            raise ValueError("阳性和阴性都必须至少含一条特征数据；不会发布不完整合并文件")
        for fingerprint in fingerprints:
            print("[SHA256/合并后] {}".format(fingerprint["path"]), flush=True)
            after = file_fingerprint(fingerprint["path"])
            if any(after[key] != fingerprint[key] for key in ("size_bytes", "mtime_ns", "sha256")):
                raise ValueError("合并期间输入被修改: {}".format(fingerprint["path"]))
        summary = {"status": "complete", "script_version": VERSION, "total_rows": total,
                   "label_mapping": {"0": "A", "1": "T", "2": "C", "3": "G", "4": "X", "5": "Y"},
                   "label_counts": {str(key): counts[key] for key in range(6)},
                   "source_role_counts": dict(source_counts), "masked_rows": masked_count,
                   "positive_identity_mode": "positive_row_proxy" if positive_proxy_mode else "extraction_metadata",
                   "positive_unique_read_count": None if positive_proxy_mode else read_database.execute("SELECT COUNT(*) FROM positive_reads").fetchone()[0],
                   "positive_proxy_identity_count": source_counts["positive"] if positive_proxy_mode else 0,
                   "positive_read_duplicate_count": None if positive_proxy_mode else 0,
                   "cross_source_read_collision_count": None if positive_proxy_mode else 0,
                   "positive_one_row_one_read_assumption_verified": False if positive_proxy_mode else True,
                   "positive_identity_audit": {
                       "one_csv_row_per_positive_read": "assumed_from_experimental_design" if positive_proxy_mode else "checked_unique_extracted_read_id",
                       "real_positive_read_duplicate_check": "not_observable_without_positive_metadata" if positive_proxy_mode else "passed",
                       "cross_source_real_read_identity_check": "not_observable_without_positive_metadata" if positive_proxy_mode else "passed",
                       "proxy_id_rule": "positive-row:<positive_csv_full_sha256>:<one_based_data_record_ordinal>" if positive_proxy_mode else None,
                       "positive_reference_identity": "unknown" if positive_proxy_mode else "from_extraction_metadata"},
                   "split_seed": split_seed, "split_ratios": str(split_ratios), "split_method": SPLIT_METHOD,
                   "split_label_counts": {split: {str(label): split_counts[(split, label)] for label in range(6)} for split in SPLIT_NAMES},
                   "input_fingerprints": fingerprints,
                   "output_feature_csv": str(paths[0]), "output_metadata_tsv": str(paths[1]),
                   "metadata_columns": header,
                   "notes": ["仅合并，不进行均衡；下游在固定read分区内下采样。",
                             "read哈希的7:2:1为期望比例，单motif/类别的实际样本比例不保证恰好相同。",
                             "run_id缺失保持unknown，source_dataset不是测序run。",
                             "默认阳性不查找metadata；每数据记录一条read依据实验设计假定，代理ID不是已核验的真实read UUID。" if positive_proxy_mode else "显式阳性metadata模式已核验read_id唯一性和阴阳性ID碰撞。",
                             "阳性无metadata时chrom、strand和run保持unknown，不能由label或motif推断，也不能核验阳性真实read重复及跨来源身份。" if positive_proxy_mode else "CSV/meta行数、kmer与label核验不能替代同次提取的真实read溯源。",
                             "summary最后提交，作为本组三个输出完成标记。"]}
        with open(staged[2], "w", encoding="utf-8") as stream:
            json.dump(summary, stream, ensure_ascii=False, indent=2, allow_nan=False)
            stream.write("\n")
            stream.flush()
            os.fsync(stream.fileno())
        # Recheck before commit to avoid overwriting files created during processing.
        for path in paths:
            if path.exists() and not overwrite:
                raise ValueError("运行期间输出已出现，拒绝覆盖: {}".format(path))
        for source, destination in zip(staged, paths):
            os.replace(source, destination)
        print("[完成] {:,}条；CSV={}；metadata={}".format(total, paths[0], paths[1]), flush=True)
        return summary
    finally:
        if read_database is not None:
            read_database.close()
        if read_database_path is not None:
            try:
                read_database_path.unlink()
            except FileNotFoundError:
                pass
        for path in staged:
            try:
                path.unlink()
            except FileNotFoundError:
                pass


def build_parser():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--positive", "--positive_csv", default=str(DEFAULT_POSITIVE), help="阳性六类原始CSV，仅label4/5")
    parser.add_argument("--negative", "--negative_csv", default=str(DEFAULT_NEGATIVE), help="新的overlap阴性CSV，仅label0..3")
    parser.add_argument("--positive_meta", default=None, help="可选：显式提供同次阳性提取逐行metadata；默认不查找，按每行一条阳性read的设计生成已标注行代理ID")
    parser.add_argument("--negative_meta", default=None, help="同次阴性提取的逐行metadata TSV；默认唯一自动匹配")
    parser.add_argument("--output", "--output_csv", default=str(DEFAULT_OUTPUT), help="新的合并CSV")
    parser.add_argument("--meta_out", "--output_meta", default=None, help="默认<output_stem>_meta.tsv")
    parser.add_argument("--summary_out", default=None, help="默认<output_stem>_merge_summary.json")
    parser.add_argument("--split_seed", type=int, default=20260429, help="须与阴性提取一致，默认20260429")
    parser.add_argument("--split_ratios", default="7,2,1", help="须与阴性提取一致，默认7,2,1")
    parser.add_argument("--overwrite", action="store_true", help="显式允许替换已有输出，永不覆盖输入")
    parser.add_argument("--version", action="version", version=VERSION)
    return parser


def main(argv=None):
    args = build_parser().parse_args(argv)
    try:
        merge_csv_files_stream([args.positive, args.negative], args.output,
                               meta_files=[args.positive_meta, args.negative_meta],
                               output_meta=args.meta_out, summary_out=args.summary_out,
                               split_seed=args.split_seed, split_ratios=args.split_ratios, overwrite=args.overwrite)
    except (ValueError, OSError, csv.Error) as exc:
        print("[ERROR] {}".format(exc), file=sys.stderr, flush=True)
        return 2
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
