#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
split_features_by_reference_and_strand.py

功能
----
根据 xna_pc_6mer_features_meta_all.tsv 中的 chrom 和 bam_strand，
把 xna_pc_6mer_features_all.csv 拆分成适用于 3 类模型评估的测试集：

1. ATCGXY_6class
   label:
       A -> 0
       T -> 1
       C -> 2
       G -> 3
       X -> 4
       Y -> 5

2. ATCGX_5class
   label:
       A -> 0
       T -> 1
       C -> 2
       G -> 3
       X -> 4
   只保留 XNA 正链作为 X，跳过 XNA 负链 Y。

3. ATCGY_5class
   label:
       A -> 0
       T -> 1
       C -> 2
       G -> 3
       Y -> 4
   只保留 XNA 负链作为 Y，跳过 XNA 正链 X。

重要规则
--------
1. 每个输出文件仍然按照 chrom + bam_strand 拆分。
   例如：
       xna_pc_6mer_features_all__PC01_pos.csv
       xna_pc_6mer_features_all__PC01_neg.csv
       xna_pc_6mer_features_all__XNA01_pos.csv
       xna_pc_6mer_features_all__XNA01_neg.csv

2. 输出文件中第一列 6mer motif 的第 4 个碱基统一改成 N。
   例如：
       ACGXTA -> ACGNTA
       TCGACA -> TCGNCA

3. label 根据“改成 N 之前”的原始第 4 位或者 chrom + strand 决定。

4. features_all.csv 本身没有 chrom 和 bam_strand，
   所以必须依赖 meta_all.tsv，并要求二者数据行顺序一一对应。

输入与输出
----------
必须通过命令行显式指定特征 CSV、metadata TSV 和输出目录。
特征 CSV 与 metadata TSV 的数据行必须一一对应；脚本会在行数不一致时立即报错。
"""

import argparse
import csv
import os
import sys
from collections import Counter, defaultdict
from itertools import chain, zip_longest


DEFAULT_FEATURE_HEADER = [
    "kmer", "mean", "std", "median", "dwell",
    "quality", "mismatch", "insertion", "deletion",
    "signal", "label"
]


TASKS = {
    "ATCGXY_6class": {
        "folder": "ATCGXY_6class",
        "description": "6-class model: A/T/C/G/X/Y",
    },
    "ATCGX_5class": {
        "folder": "ATCGX_5class",
        "description": "5-class X model: A/T/C/G/X",
    },
    "ATCGY_5class": {
        "folder": "ATCGY_5class",
        "description": "5-class Y model: A/T/C/G/Y",
    },
}


CANONICAL_LABEL = {
    "A": 0,
    "T": 1,
    "C": 2,
    "G": 3,
}


def parse_args():
    parser = argparse.ArgumentParser(
        description=(
            "Split xna_pc_6mer_features_all.csv into test sets for "
            "ATCGXY 6-class, ATCGX 5-class, and ATCGY 5-class models."
        )
    )

    parser.add_argument(
        "--input",
        required=True,
        help="Input feature CSV produced by the feature-extraction step."
    )
    parser.add_argument(
        "--meta",
        required=True,
        help="Metadata TSV whose data rows correspond one-to-one with the feature CSV."
    )
    parser.add_argument(
        "--outroot",
        required=True,
        help="Output root directory for the three task-specific split datasets."
    )
    parser.add_argument(
        "--prefix",
        default=None,
        help=(
            "Output file prefix. Default: input basename without .csv, "
            "for example xna_pc_6mer_features_all"
        )
    )
    parser.add_argument(
        "--header",
        choices=["auto", "always", "never"],
        default="always",
        help=(
            "Whether to write CSV header to split files. "
            "auto = preserve input header status. Default: always"
        )
    )
    parser.add_argument(
        "--label_check",
        choices=["warn", "error", "off"],
        default="warn",
        help=(
            "Check whether old feature label matches meta chrom/strand. "
            "Old expected labels: PC=0, XNA plus=1, XNA minus=2. Default: warn"
        )
    )
    parser.add_argument(
        "--xna_pos_is",
        choices=["X", "Y"],
        default="X",
        help=(
            "Which non-canonical base is represented by XNA positive strand. "
            "Default: X. If set to Y, XNA positive strand is treated as Y "
            "and XNA negative strand is treated as X."
        )
    )
    parser.add_argument(
        "--log_every",
        type=int,
        default=200000,
        help="Print progress every N rows. Default: 200000"
    )
    parser.add_argument(
        "--summary_name",
        default="split_summary.tsv",
        help="Summary TSV filename under outroot. Default: split_summary.tsv"
    )

    args = parser.parse_args()
    if args.log_every < 0:
        parser.error("--log_every must be >= 0")
    return args


def is_feature_header(row):
    if not row:
        return False
    first = row[0].strip().lower()
    last = row[-1].strip().lower()
    return first == "kmer" and last == "label"


def safe_name(text):
    keep = []
    for ch in str(text):
        if ch.isalnum() or ch in ("_", "-", "."):
            keep.append(ch)
        else:
            keep.append("_")
    return "".join(keep)


def strand_to_name(strand):
    if strand == "+":
        return "pos"
    if strand == "-":
        return "neg"
    return safe_name(strand)


def expected_old_label_from_meta(chrom, strand):
    """
    检查原始二/三分类提取文件中的旧 label 是否和 meta 对得上。

    原始特征提取代码中：
        PC  -> 0
        XNA + -> 1
        XNA - -> 2
    """
    if chrom.startswith("PC"):
        return "0"
    if chrom.startswith("XNA"):
        if strand == "+":
            return "1"
        if strand == "-":
            return "2"
    return None


def get_xna_base_from_strand(strand, xna_pos_is):
    """
    根据链方向判断当前 XNA 数据是 X 还是 Y。

    默认：
        XNA + -> X
        XNA - -> Y

    如果 --xna_pos_is Y：
        XNA + -> Y
        XNA - -> X
    """
    if xna_pos_is == "X":
        if strand == "+":
            return "X"
        if strand == "-":
            return "Y"
    else:
        if strand == "+":
            return "Y"
        if strand == "-":
            return "X"

    raise ValueError(f"Unsupported strand: {strand}")


def mask_kmer_center_to_n(kmer):
    """
    把 6mer 的第 4 个碱基改成 N。

    注意：
        Python 下标从 0 开始，所以第 4 位是 index 3。
    """
    if len(kmer) < 4:
        raise ValueError(f"kmer length < 4: {kmer}")

    return kmer[:3] + "N" + kmer[4:]


def infer_new_label(task_name, chrom, strand, original_kmer, xna_pos_is):
    """
    根据任务类型、参考序列和原始 motif 推断新 label。

    返回：
        int label 或 None

    None 表示该行应该在该任务中跳过。
    """
    if len(original_kmer) < 4:
        raise ValueError(f"kmer length < 4: {original_kmer}")

    center_base = original_kmer[3].upper()

    if chrom.startswith("PC"):
        if center_base not in CANONICAL_LABEL:
            raise ValueError(
                f"PC row has non-canonical center base: chrom={chrom}, "
                f"strand={strand}, kmer={original_kmer}, center_base={center_base}"
            )
        return CANONICAL_LABEL[center_base]

    if chrom.startswith("XNA"):
        xna_base = get_xna_base_from_strand(strand, xna_pos_is=xna_pos_is)

        if task_name == "ATCGXY_6class":
            if xna_base == "X":
                return 4
            if xna_base == "Y":
                return 5

        if task_name == "ATCGX_5class":
            if xna_base == "X":
                return 4
            return None

        if task_name == "ATCGY_5class":
            if xna_base == "Y":
                return 4
            return None

    raise ValueError(
        f"Cannot infer label: task={task_name}, chrom={chrom}, "
        f"strand={strand}, kmer={original_kmer}"
    )


def open_writer_for_group(
    task_name,
    chrom,
    strand,
    outroot,
    prefix,
    feature_header,
    write_header,
    writers,
    file_handles,
    output_paths,
):
    key = (task_name, chrom, strand)
    if key in writers:
        return writers[key]

    task_folder = TASKS[task_name]["folder"]
    outdir = os.path.join(outroot, task_folder)
    os.makedirs(outdir, exist_ok=True)

    strand_name = strand_to_name(strand)
    out_name = f"{prefix}__{safe_name(chrom)}_{strand_name}.csv"
    out_path = os.path.join(outdir, out_name)

    fp = open(out_path, "w", newline="", encoding="utf-8", buffering=1024 * 1024 * 8)
    writer = csv.writer(fp)

    if write_header:
        writer.writerow(feature_header)

    file_handles[key] = fp
    writers[key] = writer
    output_paths[key] = out_path

    return writer


def main():
    args = parse_args()

    if not os.path.exists(args.input):
        raise FileNotFoundError(f"Feature CSV not found: {args.input}")

    if not os.path.exists(args.meta):
        raise FileNotFoundError(f"Metadata TSV not found: {args.meta}")

    os.makedirs(args.outroot, exist_ok=True)

    for task_name in TASKS:
        os.makedirs(os.path.join(args.outroot, TASKS[task_name]["folder"]), exist_ok=True)

    if args.prefix is None:
        base = os.path.basename(args.input)
        if base.lower().endswith(".csv"):
            prefix = base[:-4]
        else:
            prefix = base
    else:
        prefix = args.prefix

    stats = Counter()
    task_counts = defaultdict(int)
    group_counts = defaultdict(int)

    writers = {}
    file_handles = {}
    output_paths = {}

    with open(args.input, "r", newline="", encoding="utf-8", buffering=1024 * 1024 * 16) as f_feat, \
         open(args.meta, "r", newline="", encoding="utf-8", buffering=1024 * 1024 * 16) as f_meta:

        feature_reader = csv.reader(f_feat)
        meta_reader = csv.DictReader(f_meta, delimiter="\t")

        try:
            first_feature_row = next(feature_reader)
        except StopIteration:
            raise RuntimeError(f"Feature CSV is empty: {args.input}")

        input_has_header = is_feature_header(first_feature_row)

        if input_has_header:
            feature_header = first_feature_row
            feature_iter = feature_reader
        else:
            feature_header = DEFAULT_FEATURE_HEADER
            feature_iter = chain([first_feature_row], feature_reader)

        if args.header == "always":
            write_header = True
        elif args.header == "never":
            write_header = False
        else:
            write_header = input_has_header

        required_meta_cols = {"read_id", "chrom", "bam_strand"}
        if meta_reader.fieldnames is None:
            raise RuntimeError(f"Metadata TSV has no header: {args.meta}")

        missing_cols = required_meta_cols - set(meta_reader.fieldnames)
        if missing_cols:
            raise RuntimeError(
                f"Metadata TSV missing required columns: {sorted(missing_cols)}\n"
                f"Found columns: {meta_reader.fieldnames}"
            )

        try:
            for row_idx, pair in enumerate(zip_longest(feature_iter, meta_reader), start=1):
                feature_row, meta_row = pair

                if feature_row is None:
                    stats["error_meta_has_extra_rows"] += 1
                    raise RuntimeError(
                        f"Metadata TSV has more data rows than feature CSV. "
                        f"First extra meta row index: {row_idx}"
                    )

                if meta_row is None:
                    stats["error_feature_has_extra_rows"] += 1
                    raise RuntimeError(
                        f"Feature CSV has more data rows than metadata TSV. "
                        f"First extra feature row index: {row_idx}"
                    )

                stats["rows_total"] += 1

                if args.log_every > 0 and stats["rows_total"] % args.log_every == 0:
                    print(
                        f"[progress] rows={stats['rows_total']:,} "
                        f"written_6class={task_counts['ATCGXY_6class']:,} "
                        f"written_5x={task_counts['ATCGX_5class']:,} "
                        f"written_5y={task_counts['ATCGY_5class']:,}",
                        file=sys.stderr,
                        flush=True,
                    )

                if not feature_row:
                    stats["skip_empty_feature_row"] += 1
                    continue

                if len(feature_row) < 2:
                    stats["bad_feature_row_too_short"] += 1
                    continue

                chrom = meta_row.get("chrom", "").strip()
                strand = meta_row.get("bam_strand", "").strip()

                if not chrom:
                    stats["bad_meta_empty_chrom"] += 1
                    continue

                if strand not in ("+", "-"):
                    stats["bad_meta_strand"] += 1
                    continue

                original_kmer = feature_row[0].strip()
                if len(original_kmer) != 6:
                    stats["bad_kmer_length_not_6"] += 1
                    continue

                if args.label_check != "off":
                    expected_old_label = expected_old_label_from_meta(chrom, strand)
                    observed_old_label = feature_row[-1].strip()

                    if expected_old_label is not None and observed_old_label != expected_old_label:
                        stats["old_label_mismatch"] += 1
                        msg = (
                            f"Old label mismatch at data row {row_idx}: "
                            f"chrom={chrom}, strand={strand}, "
                            f"expected_old_label={expected_old_label}, "
                            f"observed_old_label={observed_old_label}, "
                            f"kmer={original_kmer}"
                        )
                        if args.label_check == "error":
                            raise RuntimeError(msg)
                        elif stats["old_label_mismatch"] <= 20:
                            print(f"[warning] {msg}", file=sys.stderr)

                try:
                    masked_kmer = mask_kmer_center_to_n(original_kmer)
                except Exception as e:
                    stats["bad_mask_kmer"] += 1
                    if stats["bad_mask_kmer"] <= 20:
                        print(f"[warning] {e}", file=sys.stderr)
                    continue

                for task_name in TASKS:
                    try:
                        new_label = infer_new_label(
                            task_name=task_name,
                            chrom=chrom,
                            strand=strand,
                            original_kmer=original_kmer,
                            xna_pos_is=args.xna_pos_is,
                        )
                    except Exception as e:
                        stats[f"bad_label_infer_{task_name}"] += 1
                        if stats[f"bad_label_infer_{task_name}"] <= 20:
                            print(f"[warning] {e}", file=sys.stderr)
                        continue

                    if new_label is None:
                        stats[f"skip_{task_name}"] += 1
                        continue

                    out_row = list(feature_row)
                    out_row[0] = masked_kmer
                    out_row[-1] = str(new_label)

                    writer = open_writer_for_group(
                        task_name=task_name,
                        chrom=chrom,
                        strand=strand,
                        outroot=args.outroot,
                        prefix=prefix,
                        feature_header=feature_header,
                        write_header=write_header,
                        writers=writers,
                        file_handles=file_handles,
                        output_paths=output_paths,
                    )

                    writer.writerow(out_row)

                    task_counts[task_name] += 1
                    group_counts[(task_name, chrom, strand)] += 1
                    stats["rows_written_total"] += 1

                    if chrom.startswith("PC"):
                        stats[f"{task_name}_PC_rows"] += 1
                    elif chrom.startswith("XNA"):
                        xna_base = get_xna_base_from_strand(
                            strand=strand,
                            xna_pos_is=args.xna_pos_is,
                        )
                        stats[f"{task_name}_XNA_{xna_base}_rows"] += 1

        finally:
            for fp in file_handles.values():
                fp.close()

    summary_path = os.path.join(args.outroot, args.summary_name)
    with open(summary_path, "w", newline="", encoding="utf-8") as f_sum:
        summary_writer = csv.writer(f_sum, delimiter="\t")
        summary_writer.writerow([
            "task",
            "chrom",
            "strand",
            "strand_name",
            "rows",
            "output_csv",
        ])

        for (task_name, chrom, strand), count in sorted(group_counts.items()):
            summary_writer.writerow([
                task_name,
                chrom,
                strand,
                strand_to_name(strand),
                count,
                output_paths.get((task_name, chrom, strand), ""),
            ])

    task_summary_path = os.path.join(args.outroot, "task_summary.tsv")
    with open(task_summary_path, "w", newline="", encoding="utf-8") as f_task:
        task_writer = csv.writer(f_task, delimiter="\t")
        task_writer.writerow(["task", "rows"])
        for task_name in TASKS:
            task_writer.writerow([task_name, task_counts[task_name]])

    print("\n[Summary]", file=sys.stderr)
    print(f"input_feature_csv: {args.input}", file=sys.stderr)
    print(f"input_meta_tsv    : {args.meta}", file=sys.stderr)
    print(f"output_root       : {args.outroot}", file=sys.stderr)
    print(f"summary_tsv       : {summary_path}", file=sys.stderr)
    print(f"task_summary_tsv  : {task_summary_path}", file=sys.stderr)
    print(f"input_has_header  : {input_has_header}", file=sys.stderr)
    print(f"write_header      : {write_header}", file=sys.stderr)
    print(f"xna_pos_is        : {args.xna_pos_is}", file=sys.stderr)

    print("\n[Task counts]", file=sys.stderr)
    for task_name in TASKS:
        print(f"{task_name}: {task_counts[task_name]}", file=sys.stderr)

    print("\n[Stats]", file=sys.stderr)
    for key in sorted(stats):
        print(f"{key}: {stats[key]}", file=sys.stderr)

    print("\n[Groups]", file=sys.stderr)
    for (task_name, chrom, strand), count in sorted(group_counts.items()):
        print(
            f"{task_name}\t{chrom}\t{strand}\t{strand_to_name(strand)}\t"
            f"{count}\t{output_paths.get((task_name, chrom, strand), '')}",
            file=sys.stderr,
        )


if __name__ == "__main__":
    main()