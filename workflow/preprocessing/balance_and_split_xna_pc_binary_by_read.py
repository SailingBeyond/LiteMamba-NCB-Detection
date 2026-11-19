#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""按 read ID 对 XNA/PC 二分类特征均衡、下采样，并按 7:2:1 划分。

输入是已提取的 11 列 CSV 和逐行配套的 metadata TSV，而不是原始测序 TSV。
metadata 必须与特征来自同一次提取且未单独重排；字段匹配不能独立证明 read 身份。
pos/neg 始终指测序正链/反链；输出标签 0=PC/DNA，1=XNA。
兼容已核实的旧输入标签：PC/DNA=0，XNA 正链=1，XNA 反链=1 或 2。
旧反链标签 2 仅在 XNA 且 bam_strand='-' 时转换为 1，其他非二分类标签拒绝。
先将第 4 位统一遮蔽为 N，再于每个链方向、遮蔽背景内保留配对的
PCxx/DNAxx 与 XNAxx；各参考贡献相同数量的独立 read，严格 7:2:1。
序列第 4 位改为 N，标签归一为 0/1；8 列统计量和 signal 共 9 列原样保留。
metadata 另存 original_kmer、original_label、original_meta_label 以便审计。
本脚本针对固定一个目标窗口的二分类数据；同一 read 的冲突记录会报错。
信号按记录处理，SQLite 只保存身份、小型 metadata 和特征文件字节位置。
默认按输入顺序导出以减少硬盘寻道；--shuffle_output 可选择随机读取导出。
"""
import argparse
import csv
import hashlib
import itertools
import json
import os
import re
import sqlite3
import sys
import time
from collections import Counter, defaultdict
from contextlib import ExitStack
from decimal import Decimal, InvalidOperation
from pathlib import Path

VERSION = "1.1.0"
FEATURE_HEADER = ["kmer", "mean", "std", "median", "dwell", "quality",
                  "mismatch", "insertion", "deletion", "signal", "label"]
META_HEADER = ["read_id", "chrom", "bam_strand", "row_start", "row_end",
               "kmer_ref_start", "kmer_ref_end", "barcode_overlap", "kmer", "label"]
EXTRA_META = ["source_file", "source_row", "original_kmer", "original_label",
              "original_meta_label", "motif", "pair_id", "split"]
SPLITS = ("train", "val", "test")
STRAND_NAME = {"+": "pos", "-": "neg"}
REF_RE = re.compile(r"^(XNA|PC|DNA)([0-9]+)$")
KMER_RE = re.compile(r"^[ATCGNXY]{6}$")


def arguments():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--input_dir", default="/path/to/revision-inputs/POC-XNA_DNA-01_16/POC-XNA_DNA-01_16/fast5_single_guppy")
    p.add_argument("--output_dir", help="默认是 input_dir/binary_read_split_721")
    p.add_argument("--features_pos", help="正链特征 CSV；pos 不代表阳性")
    p.add_argument("--meta_pos", help="与正链 CSV 逐行对应的 TSV")
    p.add_argument("--features_neg", help="反链特征 CSV；neg 不代表阴性")
    p.add_argument("--meta_neg", help="与反链 CSV 逐行对应的 TSV")
    p.add_argument("--features_all", help="正反链混合的原始 11 列 CSV")
    p.add_argument("--meta_all", help="与 all CSV 对应的 TSV；也可按方向过滤后对应 pos/neg CSV")
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--max_per_reference", type=int, default=0,
                   help="每个链/motif/reference 最多保留多少 read；0 不限，最终向下取 10 的倍数")
    p.add_argument("--unpaired_policy", choices=("error", "drop"), default="error",
                   help="同一链/motif 缺失配对参考时：error 报错；drop 剔除该编号并报告")
    p.add_argument("--log_every", type=int, default=100000)
    p.add_argument("--log_seconds", type=float, default=30)
    p.add_argument("--shuffle_output", action="store_true",
                   help="按固定 seed 打乱输出；大硬盘会产生大量随机寻道，默认按原输入顺序导出")
    p.add_argument("--overwrite", action="store_true", help="允许替换此脚本的已有结果文件，不会替换输入")
    return p.parse_args()


def stable_key(seed, namespace, read_id):
    # 不使用 Python hash()，保证跨进程、输入顺序改变后仍能复现 read 身份。
    return hashlib.sha256((str(seed) + "\0" + namespace + "\0" + read_id).encode("utf-8")).hexdigest()


def metadata_candidate(folder, name):
    names = ["xna_pc_6mer_features_meta_" + name + ".tsv",
             "xna_pc_6mer_features_" + name + "_meta.tsv",
             "xna_pc_6mer_features_" + name + ".meta.tsv"]
    return next((folder / n for n in names if (folder / n).is_file()), None)


def resolve_inputs(args):
    """优先显式输入，其次完整的分链配对，最后回退到完整 all 配对。"""
    folder = Path(args.input_dir).expanduser().resolve()
    args.output_dir = str(Path(args.output_dir).expanduser().resolve() if args.output_dir
                          else folder / "binary_read_split_721")
    explicit_features = any((args.features_pos, args.features_neg, args.features_all))
    specs = []
    if args.features_all:
        if any((args.features_pos, args.features_neg, args.meta_pos, args.meta_neg)):
            raise ValueError("--features_all 不能与分链输入混用。")
        meta = args.meta_all or metadata_candidate(Path(args.features_all).resolve().parent, "all")
        if not meta:
            raise ValueError("--features_all 必须配套 --meta_all，无法凭特征猜 read ID。"
                             "如旧版没有保存 metadata，请从原始 TSV 重新提取并指定 --meta_out_pos/--meta_out_neg。")
        specs.append((args.features_all, meta, None, None))
    elif explicit_features:
        for name, strand in (("pos", "+"), ("neg", "-")):
            feature = getattr(args, "features_" + name)
            meta = getattr(args, "meta_" + name)
            if not feature:
                if meta:
                    raise ValueError("--meta_" + name + " 缺少对应 --features_" + name)
                continue
            filtered = None
            if not meta and args.meta_all:
                meta, filtered = args.meta_all, strand
            if not meta:
                meta = metadata_candidate(Path(feature).resolve().parent, name)
            if not meta:
                raise ValueError("缺少 " + name + " 的 metadata，请指定 --meta_" + name + " 或 --meta_all。"
                                 "如从未保存 metadata，请从原始 TSV 重新提取并指定 --meta_out_pos/--meta_out_neg。")
            specs.append((feature, meta, strand, filtered))
    else:
        if any((args.meta_all, args.meta_pos, args.meta_neg)):
            raise ValueError("显式 metadata 参数必须同时指定对应的 features 参数。")
        pairs = [(folder / ("xna_pc_6mer_features_" + n + ".csv"), metadata_candidate(folder, n), s)
                 for n, s in (("pos", "+"), ("neg", "-"))]
        if all(f.is_file() and m for f, m, _ in pairs):
            specs = [(f, m, s, None) for f, m, s in pairs]
        elif (folder / "xna_pc_6mer_features_all.csv").is_file() and metadata_candidate(folder, "all"):
            specs = [(folder / "xna_pc_6mer_features_all.csv", metadata_candidate(folder, "all"), None, None)]
        else:
            needed = "\n".join("  {} + {}".format(f, m or "xna_pc_6mer_features_meta_" + STRAND_NAME[s] + ".tsv")
                               for f, m, s in pairs)
            raise ValueError("未找到完整的特征/metadata 对：\n" + needed +
                             "\n或者提供 xna_pc_6mer_features_all.csv + xna_pc_6mer_features_meta_all.tsv。"
                             "\n如只处理一个方向，请显式指定该方向的 --features_pos/neg 和 --meta_pos/neg。"
                             "\n旧输出若没有 metadata，请从原始 TSV 重新提取并指定 --meta_out_pos/--meta_out_neg。")
    result = []
    for feature, meta, strand, filtered in specs:
        f, m = Path(feature).expanduser().resolve(), Path(meta).expanduser().resolve()
        if not f.is_file() or not m.is_file():
            raise ValueError("输入文件不存在：{} / {}".format(f, m))
        result.append({"feature": str(f), "meta": str(m), "expected_strand": strand,
                       "filter_meta_strand": filtered})
    if len({x["feature"] for x in result}) != len(result):
        raise ValueError("同一特征文件不能重复指定为多个输入方向。")
    return result


def meta_rows(path, strand_filter, headers):
    """header 可选；共享 meta_all 时，按链方向过滤后逐行匹配特征。"""
    with open(path, "r", encoding="utf-8-sig", newline="") as stream:
        reader = csv.reader(stream, delimiter="\t")
        first = next(reader, None)
        if first is None:
            return
        if first and first[0].strip() == "read_id":
            names = [v.strip() for v in first]
            rows = reader
        else:
            names = META_HEADER
            rows = itertools.chain([first], reader)
        if len(names) != len(set(names)) or not set(META_HEADER).issubset(names):
            raise ValueError(path + " 的 metadata 表头缺失必要字段或有重复字段。"
                             "实际字段=" + str(names) + "；必要字段=" + str(META_HEADER))
        if set(EXTRA_META) & set(names):
            raise ValueError(path + " 包含脚本保留的输出字段：" + str(set(EXTRA_META) & set(names)))
        for name in names:
            if name not in headers:
                headers.append(name)
        for line_no, row in enumerate(rows, start=2 if first[0].strip() == "read_id" else 1):
            if len(row) != len(names):
                raise ValueError("{}: metadata 第 {} 行列数不符。".format(path, line_no))
            item = dict(zip(names, row))
            if strand_filter is None or item["bam_strand"] == strand_filter:
                yield item


def feature_row(raw, path, line_no):
    """原提取器的 CSV 每条记录占一行；不接受内嵌换行以免位置索引失配。"""
    try:
        row = next(csv.reader([raw.decode("utf-8-sig")], strict=True))
    except (UnicodeError, csv.Error) as exc:
        raise ValueError("{}:{} CSV 解析失败：{}".format(path, line_no, exc))
    return row


def record_fingerprint(row, meta):
    """原始特征和原始标准 metadata 均参与校验，包括归一化前的标签文本。"""
    return hashlib.sha256(json.dumps(
        [row, [meta[name] for name in META_HEADER]], ensure_ascii=False, separators=(",", ":")
    ).encode("utf-8")).hexdigest()


def exact_label(value, context):
    """用 Decimal 精确解析，不把 1.0000000000000001 等错误舍入为整数。"""
    try:
        number = Decimal(value)
    except (InvalidOperation, ValueError):
        raise ValueError(context + " 标签不是有效数字。")
    if not number.is_finite() or number not in (Decimal(0), Decimal(1), Decimal(2)):
        raise ValueError(context + " 输入标签只接受精确数值 0、1、2；2 还须通过 XNA 反链检查。")
    return int(number)


def validate_record(row, meta, expected_strand, context):
    if len(row) != 11:
        raise ValueError(context + " 特征必须恰好为 11 列，当前为 " + str(len(row)))
    kmer, original_label = row[0], row[10]
    details = ("kmer={!r}, kmer_len={}, feature_label={!r}, meta_label={!r}, "
               "chrom={!r}, bam_strand={!r}, read_id={!r}").format(
                   kmer, len(kmer), original_label, meta["label"], meta["chrom"],
                   meta["bam_strand"], meta["read_id"])
    if not KMER_RE.fullmatch(kmer):
        raise ValueError(context + " kmer 必须为大写 ATCGNXY 构成的 6mer；" + details)
    if meta["kmer"] != kmer:
        raise ValueError(context + " 特征与 metadata 的 kmer 不一致；meta_kmer={!r}；{}".format(
            meta["kmer"], details))
    feature_label = exact_label(original_label, context + " 特征 label；" + details)
    meta_label = exact_label(meta["label"], context + " metadata label；" + details)
    if feature_label != meta_label:
        raise ValueError(context + " 特征与 metadata 的原始 label 数值不一致；" + details)
    if not meta["read_id"].strip() or meta["read_id"] != meta["read_id"].strip():
        raise ValueError(context + " read_id 为空或含首尾空白。")
    strand, chrom = meta["bam_strand"], meta["chrom"]
    if strand not in STRAND_NAME or (expected_strand and strand != expected_strand):
        raise ValueError(context + " metadata 链方向与输入方向不符。")
    match = REF_RE.fullmatch(chrom)
    if not match or int(match.group(2)) < 1:
        raise ValueError(context + " 参考名应为 PCxx/DNAxx/XNAxx，实际为 " + chrom)
    is_xna = match.group(1) == "XNA"
    allowed = (1, 2) if is_xna and strand == "-" else (1,) if is_xna else (0,)
    if feature_label not in allowed:
        raise ValueError(context + " 参考/链方向与标签冲突：PC/DNA 仅 0，XNA+ 仅 1，"
                         "XNA- 可为 1 或已核实的旧标签 2；" + details)
    label = 1 if is_xna else 0
    try:
        start, end = int(meta["kmer_ref_start"]), int(meta["kmer_ref_end"])
        row_start, row_end = int(meta["row_start"]), int(meta["row_end"])
        if start < 1 or end - start != 5 or row_start > start or row_end < end:
            raise ValueError("窗口不在 read 参考区间内或长度不为 6")
        if int(meta["barcode_overlap"]) < 0:
            raise ValueError("barcode_overlap < 0")
    except ValueError as exc:
        raise ValueError(context + " 坐标字段无效：" + str(exc))
    # 8 个碱基级统计向量及 6 个信号段只检查结构，不重新计算或改变数值。
    for index in range(1, 9):
        tokens = row[index].split("|")
        if len(tokens) != 6 or any(not x.strip() for x in tokens):
            raise ValueError(context + " " + FEATURE_HEADER[index] + " 必须含 6 项。")
        try:
            for token in tokens:
                float(token)
        except ValueError:
            raise ValueError(context + " " + FEATURE_HEADER[index] + " 含非数值项。")
    if len(row[9].split("|")) != 6 or any(not x.strip() for x in row[9].split("|")):
        raise ValueError(context + " signal 必须含 6 个非空信号段。")
    pair_id = str(int(match.group(2))).zfill(2)
    fingerprint = record_fingerprint(row, meta)
    masked_motif = kmer[:3] + "N" + kmer[4:]
    return strand, masked_motif, chrom, pair_id, label, fingerprint


def file_stat(path):
    value = os.stat(path)
    return {"path": path, "size_bytes": value.st_size, "mtime_ns": value.st_mtime_ns}


def init_database(path):
    db = sqlite3.connect(str(path))
    db.execute("PRAGMA journal_mode=OFF")
    db.execute("PRAGMA synchronous=OFF")
    db.execute("PRAGMA temp_store=FILE")
    db.execute("PRAGMA cache_size=-65536")
    db.execute("""CREATE TABLE reads (
        read_id TEXT PRIMARY KEY, strand TEXT, motif TEXT, ref TEXT, pair_id TEXT,
        label INTEGER, file_id INTEGER, offset INTEGER, source_row INTEGER,
        meta_json TEXT, fingerprint TEXT, sample_key TEXT, split_key TEXT,
        shuffle_key TEXT, split TEXT)""")
    return db


def index_inputs(db, specs, args, meta_header):
    stats = Counter()
    label_audit = {
        "input_original_label_counts": Counter(),
        "input_original_feature_label_strings": Counter(),
        "input_original_meta_label_strings": Counter(),
        "input_label_mapping_counts": Counter(),
        "input_legacy_reverse_label2_rows": 0,
        "input_numeric_labels_remapped_rows": 0,
        "input_feature_label_strings_normalized_rows": 0,
        "input_metadata_label_strings_normalized_rows": 0,
        "output_original_label_counts": Counter(),
        "output_original_feature_label_strings": Counter(),
        "output_original_meta_label_strings": Counter(),
        "output_label_mapping_counts": Counter(),
        "output_legacy_reverse_label2_rows": 0,
        "output_numeric_labels_remapped_rows": 0,
    }
    insert = "INSERT INTO reads VALUES (?,?,?,?,?,?,?,?,?,?,?,?,?,?,NULL)"
    last_log = time.monotonic()
    for file_id, spec in enumerate(specs):
        print("[输入] {}\n       {}".format(spec["feature"], spec["meta"]), flush=True)
        iterator = meta_rows(spec["meta"], spec["filter_meta_strand"], meta_header)
        data_row, physical_row = 0, 0
        with open(spec["feature"], "rb", buffering=8 * 1024 * 1024) as stream:
            while True:
                offset = stream.tell()
                raw = stream.readline()
                if not raw:
                    break
                physical_row += 1
                row = feature_row(raw, spec["feature"], physical_row)
                if physical_row == 1 and row and row[0].strip() == "kmer":
                    if [x.strip() for x in row] != FEATURE_HEADER:
                        raise ValueError(spec["feature"] + " 的 CSV 表头与 11 列格式不符。")
                    stats["feature_headers"] += 1
                    continue
                data_row += 1
                meta = next(iterator, None)
                context = "{} 数据行 {}：".format(spec["feature"], data_row)
                if meta is None:
                    raise ValueError(context + "metadata 行数少于特征；请检查是否来自同一次运行。")
                strand, motif, ref, pair_id, label, fp = validate_record(
                    row, meta, spec["expected_strand"], context)
                original_number = int(Decimal(row[10]))
                label_audit["input_original_label_counts"][str(original_number)] += 1
                label_audit["input_original_feature_label_strings"][row[10]] += 1
                label_audit["input_original_meta_label_strings"][meta["label"]] += 1
                mapping = "{}:{}->{}".format(strand, original_number, label)
                label_audit["input_label_mapping_counts"][mapping] += 1
                label_audit["input_legacy_reverse_label2_rows"] += int(original_number == 2)
                label_audit["input_numeric_labels_remapped_rows"] += int(original_number != label)
                label_audit["input_feature_label_strings_normalized_rows"] += int(row[10] != str(label))
                label_audit["input_metadata_label_strings_normalized_rows"] += int(meta["label"] != str(label))
                read_id = meta["read_id"]
                values = (read_id, strand, motif, ref, pair_id, label, file_id, offset, data_row,
                          json.dumps(meta, ensure_ascii=False, separators=(",", ":")), fp,
                          stable_key(args.seed, "sample", read_id),
                          stable_key(args.seed, "split", read_id),
                          stable_key(args.seed, "output", read_id))
                try:
                    db.execute(insert, values)
                    stats["unique_reads"] += 1
                except sqlite3.IntegrityError:
                    old = db.execute("SELECT fingerprint,ref,strand,source_row,file_id FROM reads WHERE read_id=?",
                                     (read_id,)).fetchone()
                    if not old or old[0] != fp:
                        raise ValueError(context + "同一 read_id 有冲突记录：{}；旧来源 ref={} strand={} file={} row={}。"
                                         "本脚本面向每 read 固定一个目标窗口，请先核查重复比对/重复输入。".format(
                                             read_id, old[1] if old else "?", old[2] if old else "?",
                                             old[4] if old else "?", old[3] if old else "?"))
                    stats["identical_duplicates_removed"] += 1
                stats["input_rows"] += 1
                stats["input_rows_" + STRAND_NAME[strand]] += 1
                if data_row % 10000 == 0:
                    db.commit()
                now = time.monotonic()
                if ((args.log_every > 0 and stats["input_rows"] % args.log_every == 0)
                        or (args.log_seconds > 0 and now - last_log >= args.log_seconds)):
                    print("[索引] rows={:,} unique_reads={:,} duplicates={:,}".format(
                        stats["input_rows"], stats["unique_reads"], stats["identical_duplicates_removed"]), flush=True)
                    last_log = now
        if next(iterator, None) is not None:
            raise ValueError(spec["feature"] + " 的 metadata 行数多于特征，拒绝按位置猜测。")
        if not data_row:
            raise ValueError(spec["feature"] + " 没有特征数据。")
        spec["data_rows"] = data_row
        db.commit()
    db.execute("CREATE INDEX sample_idx ON reads(strand,motif,ref,sample_key,read_id)")
    db.commit()
    return stats, label_audit


def balance_and_assign(db, args):
    counts = defaultdict(dict)
    for strand, motif, ref, pair_id, label, count in db.execute(
            "SELECT strand,motif,ref,pair_id,label,COUNT(*) FROM reads GROUP BY strand,motif,ref,pair_id,label"):
        counts[(strand, motif)][ref] = {"pair_id": pair_id, "label": label, "available": count}
    balance, sources = [], []
    unpaired_messages = []
    for (strand, motif), references in sorted(counts.items()):
        pairs = defaultdict(lambda: defaultdict(list))
        for ref, details in references.items():
            pairs[details["pair_id"]][details["label"]].append(ref)
        usable, dropped = [], []
        for pair_id, labels in sorted(pairs.items()):
            if any(len(v) > 1 for v in labels.values()):
                raise ValueError("{} {} 编号 {} 存在多个同类参考别名：{}。请核查 PC/DNA 重复参考。".format(
                    strand, motif, pair_id, dict(labels)))
            if set(labels) != {0, 1}:
                dropped.extend(ref for refs in labels.values() for ref in refs)
                unpaired_messages.append("{} {} pair {} 缺少配对参考".format(strand, motif, pair_id))
            else:
                usable.extend([labels[0][0], labels[1][0]])
        n = min((references[r]["available"] for r in usable), default=0)
        if args.max_per_reference > 0:
            n = min(n, args.max_per_reference)
        n = (n // 10) * 10
        status = "kept" if n else ("no_complete_pair" if not usable else "fewer_than_10")
        balance.append({"strand": strand, "motif": motif, "complete_pairs": len(usable) // 2,
                        "dropped_unpaired_refs": ";".join(dropped), "reads_per_reference": n,
                        "train_per_reference": n * 7 // 10, "val_per_reference": n * 2 // 10,
                        "test_per_reference": n // 10, "selected_total": n * len(usable), "status": status})
        for ref, detail in sorted(references.items()):
            kept = n if ref in usable else 0
            reason = "unpaired_reference" if ref in dropped else status
            sources.append({"strand": strand, "motif": motif, "reference": ref,
                            "pair_id": detail["pair_id"], "label": detail["label"],
                            "available_unique_reads": detail["available"], "selected_reads": kept,
                            "train": kept * 7 // 10, "val": kept * 2 // 10, "test": kept // 10,
                            "removed_reads": detail["available"] - kept, "status": reason})
        if n:
            for ref in usable:
                # 先按 sample key 选 n 条，再按独立 split key 分 7:2:1。
                cursor = db.execute("""SELECT read_id FROM (
                    SELECT read_id,split_key FROM reads WHERE strand=? AND motif=? AND ref=?
                    ORDER BY sample_key,read_id LIMIT ?) ORDER BY split_key,read_id""",
                                    (strand, motif, ref, n))
                seen, batch = 0, []
                for (read_id,) in cursor:
                    split = "train" if seen < n * 7 // 10 else "val" if seen < n * 9 // 10 else "test"
                    batch.append((split, read_id))
                    seen += 1
                    if len(batch) >= 10000:
                        db.executemany("UPDATE reads SET split=? WHERE read_id=?", batch)
                        batch = []
                if batch:
                    db.executemany("UPDATE reads SET split=? WHERE read_id=?", batch)
                if seen != n:
                    raise RuntimeError("内部采样数量不一致。")
        db.commit()
    if unpaired_messages and args.unpaired_policy == "error":
        preview = "\n".join(unpaired_messages[:20])
        raise ValueError("发现 {} 个链/motif/编号缺少对照：\n{}\n"
                         "默认停止，确保未静默更改参考组成。确需剔除时使用 --unpaired_policy drop。".format(
                             len(unpaired_messages), preview))
    for (strand,) in db.execute("SELECT DISTINCT strand FROM reads"):
        found = db.execute("SELECT COUNT(*) FROM reads WHERE strand=? AND split IS NOT NULL", (strand,)).fetchone()[0]
        if not found:
            raise ValueError("方向 {} 没有可输出的平衡样本（完整配对每参考至少需 10 条独立 read）。".format(strand))
    return balance, sources, unpaired_messages


def write_table(path, rows, fields, delimiter=","):
    with open(path, "w", encoding="utf-8", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=fields, delimiter=delimiter)
        writer.writeheader()
        writer.writerows(rows)


def write_outputs(db, specs, args, meta_header, output_dir, stats, label_audit):
    output_counts = Counter()
    outputs = []
    manifest_fields = ["read_id", "strand", "motif", "pair_id", "chrom", "label",
                       "split", "source_file", "source_row"]
    manifest_path = output_dir / "read_split_manifest.tsv"
    last_log = time.monotonic()
    with ExitStack() as stack:
        input_handles = [stack.enter_context(open(x["feature"], "rb", buffering=1024 * 1024)) for x in specs]
        manifest = csv.DictWriter(stack.enter_context(open(manifest_path, "w", encoding="utf-8", newline="")),
                                  fieldnames=manifest_fields, delimiter="\t")
        manifest.writeheader()
        writers = {}
        active_strands = [x[0] for x in db.execute("SELECT DISTINCT strand FROM reads ORDER BY strand")]
        for strand in active_strands:
            for split in SPLITS:
                stem = "xna_pc_6mer_features_{}_balanced_{}".format(STRAND_NAME[strand], split)
                feature_path, meta_path = output_dir / (stem + ".csv"), output_dir / (stem + "_meta.tsv")
                features = csv.writer(stack.enter_context(open(feature_path, "w", encoding="utf-8", newline="", buffering=1024 * 1024)))
                metadata = csv.DictWriter(stack.enter_context(open(meta_path, "w", encoding="utf-8", newline="", buffering=1024 * 1024)),
                                          fieldnames=meta_header + EXTRA_META, delimiter="\t")
                features.writerow(FEATURE_HEADER)
                metadata.writeheader()
                writers[(strand, split)] = (features, metadata)
                outputs.extend([str(feature_path), str(meta_path)])
        # 默认按输入字节位置导出，避免 HDD 随机寻道；两种顺序的样本集合完全相同。
        ordering = "shuffle_key,read_id" if args.shuffle_output else "file_id,offset,read_id"
        cursor = db.execute("""SELECT read_id,strand,motif,ref,pair_id,label,file_id,offset,source_row,meta_json,split
                             ,fingerprint FROM reads WHERE split IS NOT NULL ORDER BY """ + ordering)
        for read_id, strand, motif, ref, pair_id, label, file_id, offset, source_row, meta_json, split, fp in cursor:
            handle = input_handles[file_id]
            handle.seek(offset)
            row = feature_row(handle.readline(), specs[file_id]["feature"], source_row)
            metadata = json.loads(meta_json)
            # 使用原始字段 fingerprint 复核；旧反链 label=2 不应与归一后的 1 比较。
            if (len(row) != 11 or row[0] != metadata["kmer"]
                    or record_fingerprint(row, metadata) != fp):
                raise RuntimeError("输入在索引后改变，字节位置重新读取校验失败。")
            original_kmer, original_label, original_meta_label = row[0], row[10], metadata["label"]
            original_number = int(Decimal(original_label))
            label_audit["output_original_label_counts"][str(original_number)] += 1
            label_audit["output_original_feature_label_strings"][original_label] += 1
            label_audit["output_original_meta_label_strings"][original_meta_label] += 1
            label_audit["output_label_mapping_counts"]["{}:{}->{}".format(strand, original_number, label)] += 1
            label_audit["output_legacy_reverse_label2_rows"] += int(original_number == 2)
            label_audit["output_numeric_labels_remapped_rows"] += int(original_number != label)
            row[0] = motif
            row[10] = str(label)
            metadata["kmer"] = motif
            metadata["label"] = str(label)
            metadata.update({"source_file": specs[file_id]["feature"], "source_row": source_row,
                             "original_kmer": original_kmer, "original_label": original_label,
                             "original_meta_label": original_meta_label, "motif": motif,
                             "pair_id": pair_id, "split": split})
            feature_writer, metadata_writer = writers[(strand, split)]
            feature_writer.writerow(row)
            metadata_writer.writerow(metadata)
            manifest.writerow({"read_id": read_id, "strand": strand, "motif": motif, "pair_id": pair_id,
                               "chrom": ref, "label": label, "split": split,
                               "source_file": specs[file_id]["feature"], "source_row": source_row})
            output_counts[strand + "/" + split + "/label_" + str(label)] += 1
            stats["output_rows"] += 1
            now = time.monotonic()
            if ((args.log_every > 0 and stats["output_rows"] % args.log_every == 0)
                    or (args.log_seconds > 0 and now - last_log >= args.log_seconds)):
                print("[写入] selected_rows={:,}".format(stats["output_rows"]), flush=True)
                last_log = now
    outputs.append(str(manifest_path))
    return dict(output_counts), outputs


def verify_split(db):
    # 显式复核全局 read 交集，以及每个来源的 7:2:1 和同 motif 等贡献。
    overlap = {}
    for left, right in (("train", "val"), ("train", "test"), ("val", "test")):
        value = db.execute("""SELECT COUNT(*) FROM reads a JOIN reads b USING(read_id)
                              WHERE a.split=? AND b.split=?""", (left, right)).fetchone()[0]
        overlap[left + "_" + right] = value
    if any(overlap.values()):
        raise RuntimeError("全局 read 隔离检查失败。")
    by_motif = defaultdict(list)
    for strand, motif, ref, total, train, val, test in db.execute("""SELECT strand,motif,ref,COUNT(*),
            SUM(split='train'),SUM(split='val'),SUM(split='test') FROM reads
            WHERE split IS NOT NULL GROUP BY strand,motif,ref"""):
        if total % 10 or (train, val, test) != (total * 7 // 10, total * 2 // 10, total // 10):
            raise RuntimeError("参考 {} 的比例检查失败。".format(ref))
        by_motif[(strand, motif)].append((total, train, val, test))
    if any(len(set(values)) != 1 for values in by_motif.values()):
        raise RuntimeError("同 motif 多参考等贡献检查失败。")
    for strand, motif, split, zeros, ones in db.execute("""SELECT strand,motif,split,
            SUM(label=0),SUM(label=1) FROM reads WHERE split IS NOT NULL GROUP BY strand,motif,split"""):
        if zeros != ones:
            raise RuntimeError("{} {} {} 两类未均衡。".format(strand, motif, split))
    return overlap


def main():
    args = arguments()
    print("[版本] balance_and_split_xna_pc_binary_by_read.py {}".format(VERSION), flush=True)
    if args.max_per_reference < 0 or (0 < args.max_per_reference < 10):
        raise ValueError("max_per_reference 应为 0 或不小于 10。")
    if args.log_every < 0 or args.log_seconds < 0:
        raise ValueError("日志间隔不能为负数。")
    # 大 raw signal 字段超过 csv 默认 128 KiB 时仍能读取。
    limit = sys.maxsize
    while True:
        try:
            csv.field_size_limit(limit)
            break
        except OverflowError:
            limit //= 10
    specs = resolve_inputs(args)
    output_dir = Path(args.output_dir)
    planned_names = ["source_reference_counts.csv", "motif_balance_report.csv", "read_split_manifest.tsv",
                     "summary.json", "summary.txt", "read_index.sqlite", "RUNNING", "FAILED.txt"]
    for name in ("pos", "neg"):
        for split in SPLITS:
            stem = "xna_pc_6mer_features_{}_balanced_{}".format(name, split)
            planned_names.extend([stem + ".csv", stem + "_meta.tsv"])
    input_paths = {Path(spec[k]).resolve() for spec in specs for k in ("feature", "meta")}
    collisions = [str(output_dir / name) for name in planned_names if (output_dir / name).resolve() in input_paths]
    if collisions:
        raise ValueError("输出与输入路径相同，任何参数都不允许覆盖输入：" + str(collisions))
    output_dir.mkdir(parents=True, exist_ok=True)
    existing = [output_dir / name for name in planned_names if (output_dir / name).exists()]
    if existing and not args.overwrite:
        raise ValueError("结果目录已有输出，请换目录或显式 --overwrite：" + str(existing[:5]))
    for path in existing:
        if not path.is_file():
            raise ValueError("预期输出路径不是普通文件：" + str(path))
        path.unlink()
    running = output_dir / "RUNNING"
    running.write_text("运行尚未完成，只有 summary.json 完成后才可使用结果。\n", encoding="utf-8")
    start = time.monotonic()
    initial_stats = {path: file_stat(path) for path in sorted(str(p) for p in input_paths)}
    db = None
    try:
        db = init_database(output_dir / "read_index.sqlite")
        meta_header = []
        stats, label_audit = index_inputs(db, specs, args, meta_header)
        print("[标签兼容] PC/DNA 0->0；XNA+ 1->1；XNA- 1/2->1；输入旧反链 label=2：{:,} 行。".format(
            label_audit["input_legacy_reverse_label2_rows"]), flush=True)
        print("[均衡] 每个链/motif 内所有完整配对参考等量；每参考向下取 10 的倍数。", flush=True)
        balance, sources, unpaired = balance_and_assign(db, args)
        overlap = verify_split(db)
        write_table(output_dir / "source_reference_counts.csv", sources, list(sources[0]))
        write_table(output_dir / "motif_balance_report.csv", balance, list(balance[0]))
        output_counts, outputs = write_outputs(db, specs, args, meta_header, output_dir, stats, label_audit)
        for path, before in initial_stats.items():
            if file_stat(path) != before:
                raise RuntimeError("输入文件在运行中改变：" + path)
        total = db.execute("SELECT COUNT(*) FROM reads WHERE split IS NOT NULL").fetchone()[0]
        if total != stats["output_rows"]:
            raise RuntimeError("输出行数与采样清单不一致。")
        split_counts = {split: db.execute("SELECT COUNT(*) FROM reads WHERE split=?", (split,)).fetchone()[0]
                        for split in SPLITS}
        summary = {"status": "SUCCESS", "algorithm_version": VERSION, "seed": args.seed,
                   "split_ratio": {"train": 0.7, "val": 0.2, "test": 0.1},
                   "group_unit": "global read_id", "motif_definition": "strand + 6mer with position 4 masked as N",
                   "balance_unit": "strand/motif/reference; complete PC-or-DNA/XNA reference pairs",
                   "target_masking": "position 4 becomes N; 8 statistics columns and signal preserved",
                   "label_mapping": {"PC/DNA": "0->0", "XNA+": "1->1", "XNA-": "1/2->1"},
                   "label_conversion": {key: dict(value) if isinstance(value, Counter) else value
                                        for key, value in label_audit.items()},
                   "output_order": "seeded shuffle" if args.shuffle_output else "input file and byte offset",
                   "max_per_reference": args.max_per_reference, "unpaired_policy": args.unpaired_policy,
                   "input_file_stats": list(initial_stats.values()), "inputs": specs,
                   "statistics": dict(stats), "split_counts": split_counts, "output_counts": output_counts,
                   "read_intersections": overlap, "unpaired_groups": unpaired,
                   "skipped_motif_groups": sum(row["status"] != "kept" for row in balance),
                   "kept_motif_groups": sum(row["status"] == "kept" for row in balance),
                   "elapsed_seconds": round(time.monotonic() - start, 2), "output_files": outputs,
                   "notes": ["每个 read 只允许一个固定目标窗口；11 列特征与 10 个标准 metadata 字段完全相同的重复项已去重。",
                             "同遮蔽 motif 各参考在 train/val/test 内均等量，两类也等量。",
                             "标签按 Decimal 精确解析；只接受 PC/DNA=0、XNA+=1、XNA-=1/2，输出统一二分类 0/1。",
                             "原标签字符串分别保存为 metadata.original_label 和 original_meta_label，禁止任意非零转阳性。",
                             "输入 metadata 必须来自同次提取且未单独重排；行数/kmer/label 校验不能独立证明 read ID 归属。",
                             "metadata 的 source_row 是特征数据行号，从 1 开始，不包含 CSV 表头。",
                             "同 read 隔离不能替代独立模板或独立测序 run 的泛化评价。",
                             "训练仅用 train，模型选择、调参、早停只用 val；test 留作固定方案的最终评价。"]}
        lines = ["运行成功", "算法版本：{}；seed：{}".format(VERSION, args.seed),
                 "输入数据行：{:,}；唯一 read：{:,}；完全重复移除：{:,}".format(
                     stats["input_rows"], stats["unique_reads"], stats["identical_duplicates_removed"]),
                 "输出：{:,}；train={:,}，val={:,}，test={:,}".format(total, *(split_counts[s] for s in SPLITS)),
                 "read ID 交集：" + json.dumps(overlap),
                 "保留 motif 分组：{}；跳过分组：{}；缺配对编号：{}".format(
                     summary["kept_motif_groups"], summary["skipped_motif_groups"], len(unpaired)),
                 "pos=正链，neg=反链；输出0=PC/DNA，1=XNA。",
                 "标签映射：PC/DNA 0->0；XNA+ 1->1；XNA- 1/2->1。",
                 "旧 XNA 反链 label 2->1：输入={:,} 行，实际输出={:,} 行。".format(
                     label_audit["input_legacy_reverse_label2_rows"], label_audit["output_legacy_reverse_label2_rows"]),
                 "同一链方向/第4位遮蔽6mer 内，每个保留参考独立 read 数相等，每参考严格 7:2:1。",
                 "序列第 4 位统一为 N，标签统一0/1；8列统计量及signal共9列保留原值。",
                 "原序列及标签文本保存在 metadata 的 original_kmer/original_label/original_meta_label。"] + summary["notes"]
        (output_dir / "summary.txt").write_text("\n".join(lines) + "\n", encoding="utf-8")
        with open(output_dir / "summary.json", "w", encoding="utf-8") as stream:
            json.dump(summary, stream, ensure_ascii=False, indent=2)
            stream.write("\n")
        running.unlink()
        print("\n" + "\n".join(lines[:6]) + "\n结果目录：" + str(output_dir), flush=True)
    except Exception as exc:
        (output_dir / "FAILED.txt").write_text(str(exc) + "\n本次结果不完整，请勿用于训练。\n", encoding="utf-8")
        raise
    finally:
        if db is not None:
            db.close()


if __name__ == "__main__":
    try:
        main()
    except (ValueError, RuntimeError, OSError, csv.Error, sqlite3.Error) as error:
        print("[ERROR] " + str(error), file=sys.stderr, flush=True)
        sys.exit(1)
