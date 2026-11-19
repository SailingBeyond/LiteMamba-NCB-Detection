#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""E. coli 阴性特征提取：允许窗口重叠，默认按各组实际可用数量完成提取。

每个 read_id 采用第一条通过 QC 的 TSV 记录，避免重复/替代比对重复取样。
在该记录中用 seed+read_id 确定随机相位，按固定步长形成候选窗口网格，
默认窗口起点间距 1 nt，检查每个起点，没有每 read 窗口数上限。
按 (kmer, A/T/C/G label) 的 max(X,Y) 原配额不放回抽样。
默认 --quota_mode available：实际目标=min(原配额, 可用窗口数)，不足组全部
保留真实窗口并记录缺口，继续输出。可用 --quota_mode strict 恢复原配额
不足即退出的行为。成功检查始终以本次实际目标为准，不复制窗口补数。

主 CSV 仍是原有 11 列；metadata 和抽样报告默认自动输出。
间距按 TSV 的参考序列位置计算，不能据此宣称信号/分子位置独立。
使用只依赖 read_id 和 split_seed 的固定哈希，按预期 7:2:1 分配 read。
metadata 的 split 列及 read 分区表是后续唯一分区依据，不能再按行重划分。
可用 --split_out_dir 同步输出三个实际分区的 11 列特征及配对 metadata。
X/Y 数据必须复用同一哈希规则和分区配置；本脚本只处理 E. coli 天然碱基。
训练时只用 validation 调参/早停/选 checkpoint，需由后续训练代码落实。
metadata 中 read_candidate_count 指该 read 在间距过滤后的合格候选数。
运行要求：Python >= 3.6 且已安装 NumPy。建议显式使用 python3，或激活
之前使用的 Conda 环境。旧解释器会给出明确提示后退出，不会开始提取。
"""

from __future__ import print_function

import sys

# Keep the whole file parseable by older interpreters so this check can run.
# The extraction itself intentionally requires Python >= 3.6.
if sys.version_info < (3, 6):
    sys.stderr.write(
        "ERROR: This script requires Python >= 3.6 and NumPy.\n"
        "Current Python: {0}\nCurrent executable: {1}\n"
        "Use: python3 extract_ecoli_6mer_features_spaced_by_read.py [the same arguments]\n"
        "Or activate the Conda environment previously used for extraction,\n"
        "then check: python --version\n"
        "No input data have been processed and no output files have been opened.\n"
        .format(sys.version.split()[0], sys.executable)
    )
    sys.exit(2)

import argparse
import csv
import os
import random
import hashlib
import time
import math
import copy
import tempfile
from contextlib import ExitStack
from collections import Counter, defaultdict

try:
    import numpy as np
except ImportError:
    sys.stderr.write(
        "ERROR: NumPy could not be imported by this Python interpreter.\n"
        "Current executable: {0}\n"
        "Activate your existing Python 3 environment containing NumPy.\n"
        "Check it with: python3 -c \"import sys, numpy; print(sys.executable); print(numpy.__version__)\"\n"
        "If you choose to install into this interpreter, use: {0} -m pip install numpy\n"
        "No input data have been processed and no output files have been opened.\n"
        .format(sys.executable)
    )
    sys.exit(2)


COMP_TABLE_XY = str.maketrans({
    "A": "T", "C": "G", "G": "C", "T": "A", "N": "N", "X": "Y", "Y": "X",
    "a": "t", "c": "g", "g": "c", "t": "a", "n": "n", "x": "y", "y": "x",
})

LABEL_MAP = {
    "A": 0,
    "T": 1,
    "C": 2,
    "G": 3,
}


def revcomp_xy(seq):
    return seq.translate(COMP_TABLE_XY)[::-1]


def robust_mad_scale(arr, use_unique=True, eps=1e-8):
    if arr.size == 0:
        return None, None

    base = np.unique(arr) if use_unique else arr
    if base.size == 0:
        return None, None

    center = float(np.median(base))
    mad = float(np.median(np.abs(base - center)) * 1.4826)

    if (not np.isfinite(mad)) or mad < eps:
        std = float(np.std(base))
        if np.isfinite(std) and std >= eps:
            mad = std
        else:
            mad = 1.0

    return center, mad


def fmt_num(x, ndigits=6):
    s = '{0:.{1}f}'.format(float(x), ndigits)
    s = s.rstrip("0").rstrip(".")
    return s if s else "0"


def mask_center_base(seq, masked_base="N", center_index_0based=3) :
    if not (0 <= center_index_0based < len(seq)):
        raise ValueError('Bad center_index_0based={0}, seq_len={1}'.format(center_index_0based, len(seq)))
    return seq[:center_index_0based] + masked_base + seq[center_index_0based + 1:]


def parse_int_like(x):
    if x is None:
        return 0
    x = str(x).strip()
    if x == "":
        return 0
    return int(float(x))


def find_column(fieldnames, candidates):
    lower_map = {c.lower(): c for c in fieldnames}
    for cand in candidates:
        if cand in fieldnames:
            return cand
        if cand.lower() in lower_map:
            return lower_map[cand.lower()]
    return None


def parse_label_list(s):
    labels = []
    for x in s.split(","):
        x = x.strip()
        if not x:
            continue
        labels.append(int(x))
    return labels


def load_positive_target_counts(count_csv, kmer_col=None, label4_col=None, label5_col=None):
    """
    读取正样本统计表。

    期望表中至少包含：
        kmer
        label_4
        label_5

    但为了兼容不同列名，也支持：
        4 / 5
        count_4 / count_5
        label4 / label5
    """

    with open(count_csv, "r", encoding="utf-8-sig", newline="") as f:
        reader = csv.DictReader(f)
        if reader.fieldnames is None:
            raise ValueError('Empty CSV or missing header: {0}'.format(count_csv))

        fieldnames = reader.fieldnames

        if kmer_col is not None:
            if kmer_col not in fieldnames:
                raise ValueError('--kmer_col={0} not found in CSV columns: {1}'.format(kmer_col, fieldnames))
            real_kmer_col = kmer_col
        else:
            real_kmer_col = find_column(fieldnames, ["kmer", "KMER", "motif", "sequence", "seq"])
            if real_kmer_col is None:
                real_kmer_col = fieldnames[0]

        if label4_col is not None:
            if label4_col not in fieldnames:
                raise ValueError('--label4_col={0} not found in CSV columns: {1}'.format(label4_col, fieldnames))
            real_label4_col = label4_col
        else:
            real_label4_col = find_column(fieldnames, ["label_4", "4", "count_4", "label4", "Label_4"])

        if label5_col is not None:
            if label5_col not in fieldnames:
                raise ValueError('--label5_col={0} not found in CSV columns: {1}'.format(label5_col, fieldnames))
            real_label5_col = label5_col
        else:
            real_label5_col = find_column(fieldnames, ["label_5", "5", "count_5", "label5", "Label_5"])

        if real_label4_col is None or real_label5_col is None:
            raise ValueError(
                'Could not auto-detect label_4 / label_5 columns.\nCSV columns are: {0}\nPlease specify --label4_col and --label5_col manually.'.format(fieldnames)
            )

        target_by_kmer = {}
        label4_sum = 0
        label5_sum = 0

        for row in reader:
            kmer = row[real_kmer_col].strip().upper()
            if not kmer:
                continue

            c4 = parse_int_like(row.get(real_label4_col, 0))
            c5 = parse_int_like(row.get(real_label5_col, 0))
            if c4 < 0 or c5 < 0:
                raise ValueError('Negative positive-sample count for kmer {0}.'.format(kmer))
            if kmer in target_by_kmer:
                raise ValueError('Duplicate kmer in positive_count_csv: {0}'.format(kmer))

            target = max(c4, c5)
            target_by_kmer[kmer] = target

            label4_sum += c4
            label5_sum += c5

    if not target_by_kmer:
        raise ValueError('No target kmer loaded from {0}'.format(count_csv))

    return target_by_kmer, real_kmer_col, real_label4_col, real_label5_col, label4_sum, label5_sum


def prepare_row_from_tsv_line(line, args, stats, need_features):
    """
    从原始 TSV 的一行中解析出后续滑窗所需内容。

    need_features=False:
        第一遍扫描，验证 read 并提供选窗所需序列，不生成完整输出特征。

    need_features=True:
        第二遍仅对最终入选的记录生成 mean/std/median/dwell/signal 等字段。
    """

    parts = line.rstrip("\n").split("\t")
    if len(parts) < 12:
        stats["bad_rows"] += 1
        stats["bad_col_count"] += 1
        return None

    read_id = parts[0].strip()
    chrom = parts[1]

    if not read_id:
        stats["bad_rows"] += 1
        stats["empty_read_id"] += 1
        return None

    try:
        row_start = int(parts[2])
    except Exception:
        stats["bad_rows"] += 1
        stats["bad_start"] += 1
        return None

    ref_seq_full = parts[3]
    if not ref_seq_full:
        stats["bad_rows"] += 1
        stats["empty_ref_seq"] += 1
        return None

    qual_s = parts[4]
    bam_strand = parts[5]
    tombo_strand = parts[6]
    strand_match = parts[7]
    signal_s = parts[8]
    mismatch_s = parts[9]
    insertion_s = parts[10]
    deletion_s = parts[11]

    if args.require_strand_match and strand_match != "1":
        stats["rows_filtered"] += 1
        stats["skip_strand_mismatch"] += 1
        return None

    if bam_strand not in ("+", "-"):
        stats["bad_rows"] += 1
        stats["bad_bam_strand"] += 1
        return None

    full_len = len(ref_seq_full)
    if full_len < args.kmer_size:
        stats["rows_filtered"] += 1
        stats["skip_short_row"] += 1
        return None

    qual_list = qual_s.split("|")
    mismatch_list = mismatch_s.split("|")
    insertion_list = insertion_s.split("|")
    deletion_list = deletion_s.split("|")
    signal_list = signal_s.split("|")

    if not (
        len(qual_list) == full_len and
        len(mismatch_list) == full_len and
        len(insertion_list) == full_len and
        len(deletion_list) == full_len and
        len(signal_list) == full_len
    ):
        stats["bad_rows"] += 1
        stats["bad_field_length_inconsistent"] += 1
        return None

    # 和旧代码保持一致：使用整条 read 的所有 signal 做 robust scaling
    flat_signal = np.fromstring(signal_s.replace("|", "*"), sep="*", dtype=np.float32)
    if flat_signal.size == 0:
        stats["bad_rows"] += 1
        stats["empty_signal"] += 1
        return None

    center, scale = robust_mad_scale(flat_signal, use_unique=bool(args.mad_use_unique))
    if center is None or scale is None:
        stats["bad_rows"] += 1
        stats["bad_mad_scale"] += 1
        return None

    # 第一遍也要检查这些字段是否真的能正常转成数值，
    # 否则第一遍统计数量和第二遍真实输出数量可能对不上。
    try:
        qual_float = [float(x) for x in qual_list]
        mismatch_float = [float(x) for x in mismatch_list]
        insertion_float = [float(x) for x in insertion_list]
        deletion_float = [float(x) for x in deletion_list]
    except Exception:
        stats["bad_rows"] += 1
        stats["bad_numeric_field"] += 1
        return None

    # 旧代码里只要有一个 signal segment 为空，就整行跳过。
    # 这里也保持一致。
    if need_features:
        means_all = []
        stds_all = []
        medians_all = []
        dwells_all = []
        raw_norm_segments_all = []

        for seg_str in signal_list:
            seg = np.fromstring(seg_str, sep="*", dtype=np.float32)
            if seg.size == 0:
                stats["bad_rows"] += 1
                stats["bad_signal_segment"] += 1
                return None

            seg = (seg - center) / scale

            means_all.append(fmt_num(np.round(np.mean(seg), args.stat_round), args.stat_round))
            stds_all.append(fmt_num(np.round(np.std(seg), args.stat_round), args.stat_round))
            medians_all.append(fmt_num(np.round(np.median(seg), args.stat_round), args.stat_round))
            dwells_all.append(fmt_num(np.round(seg.size / args.dwell_scale, args.stat_round), args.stat_round))
            raw_norm_segments_all.append("*".join(fmt_num(v, args.signal_round) for v in seg))

        qual_all = [
            fmt_num(np.round(x / args.quality_scale, args.stat_round), args.stat_round)
            for x in qual_float
        ]
        mismatch_all = [
            fmt_num(np.round(x, args.stat_round), args.stat_round)
            for x in mismatch_float
        ]
        insertion_all = [
            fmt_num(np.round(x, args.stat_round), args.stat_round)
            for x in insertion_float
        ]
        deletion_all = [
            fmt_num(np.round(x, args.stat_round), args.stat_round)
            for x in deletion_float
        ]

    else:
        for seg_str in signal_list:
            seg = np.fromstring(seg_str, sep="*", dtype=np.float32)
            if seg.size == 0:
                stats["bad_rows"] += 1
                stats["bad_signal_segment"] += 1
                return None

        means_all = None
        stds_all = None
        medians_all = None
        dwells_all = None
        raw_norm_segments_all = None
        qual_all = None
        mismatch_all = None
        insertion_all = None
        deletion_all = None

    row_end = row_start + full_len - 1

    stats["rows_usable"] += 1

    return {
        "read_id": read_id,
        "chrom": chrom,
        "row_start": row_start,
        "row_end": row_end,
        "ref_seq_full": ref_seq_full,
        "full_len": full_len,
        "bam_strand": bam_strand,
        "tombo_strand": tombo_strand,
        "strand_match": strand_match,

        "qual_all": qual_all,
        "mismatch_all": mismatch_all,
        "insertion_all": insertion_all,
        "deletion_all": deletion_all,
        "means_all": means_all,
        "stds_all": stds_all,
        "medians_all": medians_all,
        "dwells_all": dwells_all,
        "signal_all": raw_norm_segments_all,
    }


def iter_kmer_candidates(row, args, target_kmers, target_labels, need_features,
                         window_index=None, window_indices=None):
    """
    从一条 TSV 记录中滑窗生成候选 kmer。

    输出 kmer 默认会把第 target_index 位 mask 成 N，
    但 label 仍然由原始输出方向下的第 target_index 位 A/T/C/G 决定。
    """

    ref_seq_full = row["ref_seq_full"]
    full_len = row["full_len"]
    row_start = row["row_start"]
    row_end = row["row_end"]
    bam_strand = row["bam_strand"]

    target_idx0 = args.target_index - 1
    max_start = full_len - args.kmer_size

    # 可传入等间距网格或第二遍实际入选的索引，避免扫描所有相邻窗口。
    if window_indices is not None:
        window_starts = window_indices
    elif window_index is None:
        window_starts = range(max_start + 1)
    elif 0 <= window_index <= max_start:
        window_starts = (window_index,)
    else:
        raise ValueError('Window index {0} is outside read {1}.'.format(window_index, row['read_id']))

    for i in window_starts:
        kmer_ref_start = row_start + i
        kmer_ref_end = kmer_ref_start + args.kmer_size - 1

        ref_seq_win = ref_seq_full[i:i + args.kmer_size]

        if need_features:
            qual_win = row["qual_all"][i:i + args.kmer_size]
            mismatch_win = row["mismatch_all"][i:i + args.kmer_size]
            insertion_win = row["insertion_all"][i:i + args.kmer_size]
            deletion_win = row["deletion_all"][i:i + args.kmer_size]
            means_win = row["means_all"][i:i + args.kmer_size]
            stds_win = row["stds_all"][i:i + args.kmer_size]
            medians_win = row["medians_all"][i:i + args.kmer_size]
            dwells_win = row["dwells_all"][i:i + args.kmer_size]
            signal_win = row["signal_all"][i:i + args.kmer_size]

        # 负链统一转成 5'->3' 输出，并翻转对应位置特征顺序
        if bam_strand == "-":
            ref_seq_win = revcomp_xy(ref_seq_win)

            if need_features:
                qual_win = qual_win[::-1]
                mismatch_win = mismatch_win[::-1]
                insertion_win = insertion_win[::-1]
                deletion_win = deletion_win[::-1]
                means_win = means_win[::-1]
                stds_win = stds_win[::-1]
                medians_win = medians_win[::-1]
                dwells_win = dwells_win[::-1]
                signal_win = signal_win[::-1]

        ref_seq_win = ref_seq_win.upper()

        label_base = ref_seq_win[target_idx0].upper()
        if label_base not in LABEL_MAP:
            continue

        label = LABEL_MAP[label_base]
        if label not in target_labels:
            continue

        if args.mask_center:
            out_kmer = mask_center_base(
                ref_seq_win,
                masked_base=args.mask_base,
                center_index_0based=target_idx0,
            ).upper()
        else:
            out_kmer = ref_seq_win.upper()

        if out_kmer not in target_kmers:
            continue

        if not need_features:
            yield {
                "kmer": out_kmer,
                "label": label,
                "window_index": i,
            }
        else:
            feature_row = [
                out_kmer,
                "|".join(means_win),
                "|".join(stds_win),
                "|".join(medians_win),
                "|".join(dwells_win),
                "|".join(qual_win),
                "|".join(mismatch_win),
                "|".join(insertion_win),
                "|".join(deletion_win),
                "|".join(signal_win),
                label,
            ]

            meta_row = [
                row["read_id"],
                row["chrom"],
                bam_strand,
                row["row_start"],
                row_end,
                kmer_ref_start,
                kmer_ref_end,
                ref_seq_win,
                out_kmer,
                label,
                label_base,
            ]

            yield {
                "kmer": out_kmer,
                "label": label,
                "window_index": i,
                "feature_row": feature_row,
                "meta_row": meta_row,
            }


SPLIT_NAMES = ('train', 'validation', 'test')
SPLIT_METHOD = 'blake2b64-read-split-v1'


def parse_split_ratios(text):
    """Accept three finite, strictly positive weights; return normalized weights."""
    try:
        values = tuple(float(item.strip()) for item in str(text).split(','))
    except (TypeError, ValueError):
        raise ValueError('split_ratios must contain three numeric weights, e.g. 7,2,1')
    if len(values) != 3 or any(not math.isfinite(x) or x <= 0 for x in values):
        raise ValueError('split_ratios requires exactly three finite positive weights')
    # Scale first so even large, finite input weights cannot overflow their sum.
    largest = max(values)
    scaled = tuple(x / largest for x in values)
    total = sum(scaled)
    normalized = tuple(x / total for x in scaled)
    if any(x <= 0 for x in normalized):
        raise ValueError('split_ratios weights differ too much to normalize safely')
    return normalized


def assign_read_split(read_id, split_seed, split_ratios):
    """Stable global read assignment, independent of files, contexts and labels.

    Hash the UTF-8 bytes of ``decimal_seed + NUL + read_id`` with BLAKE2b,
    digest_size=8. Interpret the digest as an unsigned big-endian integer.
    Compare it against cumulative normalized ratio boundaries on [0, 2**64).
    Identical read IDs and seeds always have identical assignments. The ratios
    are expected proportions across reads, not exact per-context sample quotas.
    Reuse the same seed, ratios and this method across canonical and XNA files.
    """
    if not isinstance(read_id, str) or not read_id:
        raise ValueError('read_id must be a nonempty string')
    payload = str(int(split_seed)).encode('ascii') + b'\x00' + read_id.encode('utf-8')
    digest = hashlib.blake2b(payload, digest_size=8).digest()
    value = int.from_bytes(digest, byteorder='big', signed=False)
    scale = 1 << 64
    if value < int(split_ratios[0] * scale):
        return SPLIT_NAMES[0]
    if value < int((split_ratios[0] + split_ratios[1]) * scale):
        return SPLIT_NAMES[1]
    return SPLIT_NAMES[2]


def get_split_output_paths(directory):
    if directory is None:
        return {}
    directory = os.path.abspath(directory)
    return dict((name, {
        'features': os.path.join(directory, name + '.csv'),
        'meta': os.path.join(directory, name + '_meta.tsv')
    }) for name in SPLIT_NAMES)


def write_read_manifest(path, args, read_plan, read_splits, selected_counts):
    """Record every QC-passing planned read, including reads with zero output."""
    ratios = args.split_ratio_spec
    with open(path, 'w', newline='', encoding='utf-8') as handle:
        writer = csv.writer(handle, delimiter='\t', lineterminator='\n')
        writer.writerow([
            'read_id', 'split', 'selected_windows', 'source_tsv',
            'input_line_no', 'candidate_windows', 'split_seed',
            'split_ratios', 'split_method'
        ])
        for read_id in sorted(read_plan):
            line_no, phase, candidate_count, num_starts = read_plan[read_id]
            writer.writerow([
                read_id, read_splits[read_id], selected_counts.get(read_id, 0),
                os.path.abspath(args.input), line_no, candidate_count,
                args.split_seed, ratios, SPLIT_METHOD
            ])


def prepare_staged_outputs(args):
    """Stage every output beside its final path, without changing existing files.

    Return a shallow namespace copy with staged output paths, plus (temp, final)
    pairs. Publication replaces files individually; the complete output set is
    not an atomic filesystem transaction.
    """
    staged = copy.copy(args)
    pairs = []
    used = set()

    def stage(final):
        final = os.path.abspath(final)
        if final in used:
            raise ValueError('Output paths must be distinct: ' + final)
        used.add(final)
        parent = os.path.dirname(final)
        os.makedirs(parent, exist_ok=True)
        descriptor, temporary = tempfile.mkstemp(
            prefix='.' + os.path.basename(final) + '.', suffix='.tmp', dir=parent)
        pairs.append((temporary, final))
        os.close(descriptor)
        return temporary

    try:
        for attribute in ('output', 'meta_out', 'report_out', 'read_split_out'):
            setattr(staged, attribute, stage(getattr(args, attribute)))
        staged.split_paths = {}
        for name in SPLIT_NAMES:
            if name in args.split_paths:
                staged.split_paths[name] = {}
                for kind in ('features', 'meta'):
                    staged.split_paths[name][kind] = stage(args.split_paths[name][kind])
    except BaseException:
        cleanup_staged_outputs(pairs)
        raise
    return staged, pairs


def cleanup_staged_outputs(pairs):
    """Remove leftover staged files; already published final files are untouched."""
    for temporary, final in pairs:
        try:
            os.remove(temporary)
        except FileNotFoundError:
            pass


def publish_staged_outputs(pairs):
    """Replace outputs one at a time; this is not an all-files transaction."""
    for temporary, final in pairs:
        os.replace(temporary, final)


def read_window_phase(read_id, seed, spacing, num_window_starts):
    """Stable per-read phase, with at least one grid position for a short record."""
    if spacing == 1:
        return 0
    digest = hashlib.blake2b(
        '{0}:spaced_windows:{1}'.format(seed, read_id).encode("utf-8"), digest_size=8,
    ).digest()
    return int.from_bytes(digest, "big") % min(spacing, num_window_starts)


def progress_due(args, rows, last_log):
    now = time.monotonic()
    by_rows = args.log_every > 0 and rows % args.log_every == 0
    by_time = args.log_seconds > 0 and now - last_log >= args.log_seconds
    return by_rows or by_time, now


def scan_available_counts(args, target_kmers, target_labels):
    """Count spaced candidates from the first QC-passing record of each read.

    read_plan[read_id] = (physical_line_number, phase, eligible_grid_windows,
                          number_of_all_possible_window_starts).
    Memory scales with read count and 4,096 groups, not candidate window count.
    """
    available = Counter()
    available_reads = Counter()
    stats = Counter()
    read_plan = {}
    last_log = time.monotonic()

    with open(args.input, "r", encoding="utf-8-sig", buffering=1024 * 1024 * 16) as fin:
        for line_no, line in enumerate(fin, start=1):
            if line_no == 1 and line.startswith("read_id\t"):
                stats["header_skipped"] += 1
                continue
            stats["rows_total"] += 1
            due, now = progress_due(args, stats["rows_total"], last_log)
            if due:
                print(
                    '[pass1 count] rows={0:,} unique_usable_reads={1:,} spaced_candidate_windows={2:,} bad_rows={3:,} repeated_records_skipped={4:,}'.format(stats['rows_total'], len(read_plan), stats['candidate_windows'], stats['bad_rows'], stats['repeated_read_records_skipped']),
                    file=sys.stderr, flush=True,
                )
                last_log = now

            read_id = line.split("\t", 1)[0].strip()
            if read_id in read_plan:
                stats["repeated_read_records_skipped"] += 1
                continue
            row = prepare_row_from_tsv_line(line, args, stats, need_features=False)
            if row is None:
                continue
            num_starts = row["full_len"] - args.kmer_size + 1
            phase = read_window_phase(row["read_id"], args.seed, args.min_window_spacing, num_starts)
            starts = range(phase, num_starts, args.min_window_spacing)
            candidate_count = 0
            groups_for_read = set()
            for cand in iter_kmer_candidates(
                row, args, target_kmers, target_labels, need_features=False,
                window_indices=starts,
            ):
                group = (cand["kmer"], cand["label"])
                available[group] += 1
                groups_for_read.add(group)
                candidate_count += 1
            available_reads.update(groups_for_read)
            read_plan[row["read_id"]] = (line_no, phase, candidate_count, num_starts)
            stats["window_positions_before_spacing"] += num_starts
            stats["grid_positions_examined"] += len(starts)
            stats["candidate_windows"] += candidate_count
            stats["read_ids_with_candidate" if candidate_count else "read_ids_without_candidate"] += 1

    stats["unique_usable_read_ids"] = len(read_plan)
    return available, stats, read_plan, available_reads


FEATURE_HEADER = [
    "kmer", "mean", "std", "median", "dwell", "quality", "mismatch",
    "insertion", "deletion", "signal", "label",
]
META_HEADER = [
    "read_id", "chrom", "bam_strand", "row_start", "row_end", "kmer_ref_start",
    "kmer_ref_end", "raw_oriented_kmer", "output_kmer", "label", "label_base",
    "source_tsv", "input_line_no", "read_candidate_count", "window_index",
    "min_window_spacing", "window_phase", "read_record_policy", "split",
    "split_seed", "split_ratios", "split_method",
]
META_SPLIT_INDEX = META_HEADER.index("split")


def write_pair(writer, meta_writer, feature_row, meta_row, split_writers):
    writer.writerow(feature_row)
    if meta_writer is not None:
        meta_writer.writerow(meta_row)
    if split_writers:
        split_writer, split_meta_writer = split_writers[meta_row[META_SPLIT_INDEX]]
        split_writer.writerow(feature_row)
        split_meta_writer.writerow(meta_row)


def write_or_buffer(writer, meta_writer, feature_row, meta_row, buffer, rng,
                    shuffle_buffer_size, split_writers=None):
    """
    为了让输出顺序也不至于完全按原 TSV 顺序，可以设置 --shuffle_buffer_size。
    如果 shuffle_buffer_size <= 0，则直接写出。
    """

    if shuffle_buffer_size <= 0:
        write_pair(writer, meta_writer, feature_row, meta_row, split_writers)
        return

    buffer.append((feature_row, meta_row))

    if len(buffer) >= shuffle_buffer_size:
        idx = rng.randrange(len(buffer))
        buffer[idx], buffer[-1] = buffer[-1], buffer[idx]
        frow, mrow = buffer.pop()
        write_pair(writer, meta_writer, frow, mrow, split_writers)


def flush_buffer(writer, meta_writer, buffer, rng, split_writers=None):
    if not buffer:
        return

    rng.shuffle(buffer)
    for frow, mrow in buffer:
        write_pair(writer, meta_writer, frow, mrow, split_writers)

    buffer.clear()


def sample_and_write(args, target_kmers, target_labels, available, effective_target,
                     read_plan, read_splits):
    """Replay the same grid and sample each group exactly without replacement.

    Only rows with selected windows have their full normalized features rebuilt.
    Keeping a subset of grid windows cannot reduce their minimum separation.
    """
    rng = random.Random('{0}:group_sampling'.format(args.seed))
    shuffle_rng = random.Random('{0}:output_shuffle'.format(args.seed))
    available_left = Counter(available)
    needed_left = Counter(effective_target)
    selected = Counter()
    selected_reads = Counter()
    split_counts = Counter()
    split_reads = Counter()
    stats = Counter()
    selected_per_read = Counter()
    split_read_sets = {name: set() for name in SPLIT_NAMES}
    observed_min_distance = None
    selected_total = 0
    last_log = time.monotonic()
    for output_path in (args.output, args.meta_out):
        os.makedirs(os.path.dirname(os.path.abspath(output_path)), exist_ok=True)
    buffer = []

    with open(args.input, "r", encoding="utf-8-sig", buffering=1024 * 1024 * 16) as fin, \
         open(args.output, "w", newline="", encoding="utf-8", buffering=1024 * 1024 * 16) as fout, \
         open(args.meta_out, "w", newline="", encoding="utf-8", buffering=1024 * 1024 * 8) as meta_fp, \
         ExitStack() as split_stack:
        writer = csv.writer(fout)
        meta_writer = csv.writer(meta_fp, delimiter="\t")
        meta_writer.writerow(META_HEADER)
        if args.with_header:
            writer.writerow(FEATURE_HEADER)
        split_writers = {}
        for split_name, paths in args.split_paths.items():
            feature_fp = split_stack.enter_context(open(
                paths["features"], "w", newline="", encoding="utf-8", buffering=1024 * 1024 * 4,
            ))
            split_meta_fp = split_stack.enter_context(open(
                paths["meta"], "w", newline="", encoding="utf-8", buffering=1024 * 1024 * 2,
            ))
            split_writer = csv.writer(feature_fp)
            split_meta_writer = csv.writer(split_meta_fp, delimiter="\t")
            if args.with_header:
                split_writer.writerow(FEATURE_HEADER)
            split_meta_writer.writerow(META_HEADER)
            split_writers[split_name] = (split_writer, split_meta_writer)
        split_ratio_text = args.split_ratio_spec
        for line_no, line in enumerate(fin, start=1):
            if line_no == 1 and line.startswith("read_id\t"):
                stats["header_skipped"] += 1
                continue
            stats["rows_total"] += 1
            due, now = progress_due(args, stats["rows_total"], last_log)
            if due:
                print(
                    '[pass2 sample] rows={0:,} feature_records_built={1:,} selected={2:,} remaining_need={3:,}'.format(stats['rows_total'], stats['rows_usable'], selected_total, sum(needed_left.values())),
                    file=sys.stderr, flush=True,
                )
                last_log = now
            parts = line.rstrip("\n").split("\t", 6)
            read_id = parts[0].strip()
            plan = read_plan.get(read_id)
            if plan is None or plan[0] != line_no or plan[2] == 0:
                stats["rows_without_spaced_candidates_or_not_representative"] += 1
                continue
            _, phase, expected_count, num_starts = plan
            # Pass 1 has validated the entire record. This lightweight view is
            # sufficient for sampling; full feature calculation happens below.
            row_start = int(parts[2])
            ref_seq = parts[3]
            row_view = {
                "read_id": read_id, "chrom": parts[1], "row_start": row_start,
                "row_end": row_start + len(ref_seq) - 1, "ref_seq_full": ref_seq,
                "full_len": len(ref_seq), "bam_strand": parts[5],
            }
            if len(ref_seq) - args.kmer_size + 1 != num_starts:
                raise RuntimeError('Input record changed for read {0}.'.format(read_id))
            chosen = []
            scanned = 0
            for cand in iter_kmer_candidates(
                row_view, args, target_kmers, target_labels, need_features=False,
                window_indices=range(phase, num_starts, args.min_window_spacing),
            ):
                group = (cand["kmer"], cand["label"])
                avail = available_left[group]
                need = needed_left[group]
                if avail <= 0:
                    raise RuntimeError('Spaced candidate count exhausted unexpectedly: {0}'.format(group))
                take = need > 0 and (need >= avail or rng.random() < need / avail)
                available_left[group] -= 1
                scanned += 1
                if take:
                    chosen.append((cand["window_index"], group))
                    selected[group] += 1
                    needed_left[group] -= 1
            stats["spaced_candidates_scanned"] += scanned
            if scanned != expected_count:
                raise RuntimeError('Pass 1/2 candidate count mismatch for read {0}.'.format(read_id))
            if not chosen:
                continue
            if read_id in selected_per_read:
                raise RuntimeError('Repeated representative record for read {0}.'.format(read_id))
            split_name = read_splits[read_id]
            if split_name != assign_read_split(read_id, args.split_seed, args.split_ratios):
                raise RuntimeError("Read split assignment changed unexpectedly: " + read_id)
            differences = [right[0] - left[0] for left, right in zip(chosen, chosen[1:])]
            if differences:
                row_min = min(differences)
                if row_min < args.min_window_spacing:
                    raise RuntimeError('Spacing violation for read {0}.'.format(read_id))
                observed_min_distance = row_min if observed_min_distance is None else min(observed_min_distance, row_min)
                stats["overlapping_adjacent_output_window_pairs"] += sum(
                    distance < args.kmer_size for distance in differences
                )
            row = prepare_row_from_tsv_line(line, args, stats, need_features=True)
            if row is None:
                raise RuntimeError('Cannot rebuild features for read {0} at line {1}.'.format(read_id, line_no))
            rebuilt = 0
            for cand, (window_index, group) in zip(
                iter_kmer_candidates(
                    row, args, target_kmers, target_labels, need_features=True,
                    window_indices=[entry[0] for entry in chosen],
                ), chosen,
            ):
                if cand["window_index"] != window_index or (cand["kmer"], cand["label"]) != group:
                    raise RuntimeError('Pass 1/2 window identity mismatch for read {0}.'.format(read_id))
                meta_row = cand["meta_row"] + [
                    os.path.abspath(args.input), line_no, expected_count, window_index,
                    args.min_window_spacing, phase, "first_valid_record_per_read",
                    split_name, args.split_seed, split_ratio_text, SPLIT_METHOD,
                ]
                write_or_buffer(
                    writer, meta_writer, cand["feature_row"], meta_row,
                    buffer, shuffle_rng, args.shuffle_buffer_size, split_writers,
                )
                rebuilt += 1
            if rebuilt != len(chosen):
                raise RuntimeError('Feature reconstruction lost windows for read {0}.'.format(read_id))
            selected_per_read[read_id] = rebuilt
            split_read_sets[split_name].add(read_id)
            stats["split_" + split_name + "_windows"] += rebuilt
            split_counts.update((split_name, group) for _, group in chosen)
            split_reads.update((split_name, group) for group in {group for _, group in chosen})
            selected_reads.update({group for _, group in chosen})
            selected_total += rebuilt
        flush_buffer(writer, meta_writer, buffer, shuffle_rng, split_writers)

    stats["output_samples"] = selected_total
    stats["selected_unique_read_ids"] = len(selected_per_read)
    stats["reads_with_multiple_output_windows"] = sum(n > 1 for n in selected_per_read.values())
    stats["max_samples_per_read"] = max(selected_per_read.values(), default=0)
    stats["minimum_observed_window_start_distance"] = observed_min_distance
    stats["spacing_violations"] = 0
    stats["duplicate_windows_in_output"] = 0  # strictly increasing starts in one record/read
    for i, left in enumerate(SPLIT_NAMES):
        stats["split_" + left + "_windows"] += 0
        stats["split_" + left + "_unique_reads"] = len(split_read_sets[left])
        for right in SPLIT_NAMES[i + 1:]:
            overlap = len(split_read_sets[left].intersection(split_read_sets[right]))
            stats["read_overlap_" + left + "_" + right] = overlap
            if overlap:
                raise RuntimeError("A read appears in multiple partitions.")
    if sum(stats["split_" + name + "_windows"] for name in SPLIT_NAMES) != selected_total:
        raise RuntimeError("Partition counts do not sum to the pooled output count.")
    counts = np.asarray(list(selected_per_read.values()), dtype=np.int64)
    if counts.size:
        for name, value in zip(("p25", "median", "p75", "p95"), np.percentile(counts, (25, 50, 75, 95))):
            stats['windows_per_selected_read_{0}'.format(name)] = float(value)
        stats["windows_per_selected_read_mean"] = float(np.mean(counts))
    if selected_total != sum(selected.values()):
        raise RuntimeError("Written sample count differs from selected group counts.")
    if sum(available_left.values()) != 0 or any(needed_left.values()):
        raise RuntimeError("Pass 1/2 candidate counts differ; input must remain unchanged.")
    return selected, stats, needed_left, selected_reads, selected_per_read, split_counts, split_reads


def write_report(args, target_by_group, available, effective_target, selected,
                 available_reads, selected_reads, status="complete", split_counts=None, split_reads=None):
    os.makedirs(os.path.dirname(os.path.abspath(args.report_out)), exist_ok=True)
    split_counts = split_counts or Counter()
    split_reads = split_reads or Counter()
    with open(args.report_out, "w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow([
            "kmer", "label", "positive_target", "negative_available",
            "effective_target", "selected", "shortage", "ok", "sampling_unit",
            "min_window_spacing", "available_unique_read_ids", "target_fraction_met",
            "read_record_policy", "selected_unique_read_ids", "status", "strict_full_quota",
            "train_selected", "train_unique_read_ids", "validation_selected",
            "validation_unique_read_ids", "test_selected", "test_unique_read_ids",
            "quota_mode", "original_quota_met", "remaining_original_shortage",
        ])
        for group in sorted(target_by_group):
            target = target_by_group[group]
            avail = available.get(group, 0)
            eff = effective_target.get(group, 0)
            sel = selected.get(group, 0)
            writer.writerow([
                group[0], group[1], target, avail, eff, sel,
                max(0, target - avail), int(status.startswith("complete") and sel == eff),
                "overlapping_windows_within_read" if args.min_window_spacing < args.kmer_size
                else "spaced_windows_within_read",
                args.min_window_spacing, available_reads.get(group, 0),
                fmt_num(sel / target) if target else "NA", "first_valid_record_per_read",
                selected_reads.get(group, 0), status, int(args.quota_mode == "strict"),
                split_counts[("train", group)], split_reads[("train", group)],
                split_counts[("validation", group)], split_reads[("validation", group)],
                split_counts[("test", group)], split_reads[("test", group)],
                args.quota_mode, int(status.startswith("complete") and sel == target),
                max(0, target - sel),
            ])


def parse_args():
    p = argparse.ArgumentParser(
        description=(
            "Extract overlapping 6mer windows per read (default start spacing: 1 nt). "
            "Use the first QC-passing TSV record per read, a stable seed/read-specific "
            "random grid phase, then exact kmer+label quota sampling. No per-read window cap. "
            "For each kmer/canonical label, target=max(label_4,label_5). "
            "By default, output min(original target, available windows) per group and report shortages. "
            "Use --quota_mode strict to require every original target. Never duplicate windows. "
            "Save deterministic read-level split assignments; optionally write partition files."
        )
    )

    p.add_argument("--input", required=True,
                   help="Input TSV from extract_signal_from_fast5_*.py")
    p.add_argument("--output", required=True,
                   help="Output sampled feature CSV")
    p.add_argument("--positive_count_csv", required=True,
                   help="CSV pivot containing kmer, label_4 and label_5 counts")
    p.add_argument("--quota_mode", choices=["available", "strict"], default="available",
                   help="available (default): sample min(original quota, available) in each group, "
                        "retaining every available window in shortage groups and continuing. "
                        "strict: abort before feature writing if any group is below its original quota.")
    p.add_argument("--meta_out", default=None,
                   help="Metadata TSV. Default: <output>.meta.tsv (always saved)")
    p.add_argument("--report_out", default=None,
                   help="Sampling report CSV. Default: <output>.sampling_report.csv")

    p.add_argument("--kmer_col", default=None,
                   help="Column name of kmer in positive_count_csv. Default: auto-detect")
    p.add_argument("--label4_col", default=None,
                   help="Column name of label_4 count in positive_count_csv. Default: auto-detect")
    p.add_argument("--label5_col", default=None,
                   help="Column name of label_5 count in positive_count_csv. Default: auto-detect")

    p.add_argument("--target_labels", default="0,1,2,3",
                   help="Negative labels to extract. Default: 0,1,2,3")

    p.add_argument("--kmer_size", type=int, default=6,
                   help="K-mer size. Default: 6")
    p.add_argument("--target_index", type=int, default=4,
                   help="Which base in the final output kmer is used for labeling, 1-based. Default: 4")

    p.add_argument("--mask_center", type=int, choices=[0, 1], default=1,
                   help="Whether to mask target_index base in output kmer. Default: 1")
    p.add_argument("--mask_base", default="N",
                   help="Mask base used when --mask_center 1. Default: N")

    p.add_argument("--dwell_scale", type=float, default=1.0,
                   help="Output dwell as len(segment)/dwell_scale. Default: 1.0")
    p.add_argument("--quality_scale", type=float, default=1.0,
                   help="Output quality as Q/quality_scale. Default: 1.0")
    p.add_argument("--mad_use_unique", type=int, choices=[0, 1], default=1,
                   help="Use unique full-read signal values for median-MAD scaling. Default: 1")
    p.add_argument("--require_strand_match", type=int, choices=[0, 1], default=1,
                   help="Skip rows whose bam_strand != tombo_strand. Default: 1")

    p.add_argument("--with_header", action="store_true",
                   help="Write CSV header")
    p.add_argument("--signal_round", type=int, default=6,
                   help="Decimal places for normalized raw signal. Default: 6")
    p.add_argument("--stat_round", type=int, default=6,
                   help="Decimal places for summary statistics. Default: 6")

    p.add_argument("--seed", type=int, default=20260429,
                   help="Seed for stable read grid phases and independent quota/output-shuffle RNGs. Default: 20260429")
    p.add_argument("--shuffle_buffer_size", type=int, default=0,
                   help=(
                       "Optional local shuffle buffer for output order. "
                       "0 means no output-order shuffle, but sampling is still random. "
                       "For example 50000. Default: 0"
                   ))

    p.add_argument("--min_window_spacing", type=int, default=1,
                   help="Minimum 6mer START-to-START spacing in TSV reference bases. "
                        "Default: 1 (all overlapping windows); minimum: 1. No per-read window cap.")
    p.add_argument("--read_split_out", default=None,
                   help="Read partition manifest TSV. Default: <output>.read_splits.tsv")
    p.add_argument("--split_out_dir", default=None,
                   help="Optional directory for additional train/validation/test CSVs and paired metadata. "
                        "The pooled output is also retained, requiring extra disk space.")
    p.add_argument("--split_seed", type=int, default=20260429,
                   help="Independent seed for global read assignment; reuse across X/Y and canonical data.")
    p.add_argument("--split_ratios", default="7,2,1",
                   help="Expected train,validation,test READ weights. Default: 7,2,1; "
                        "window counts and individual contexts need not have exact ratios.")
    p.add_argument("--log_seconds", type=float, default=30.0,
                   help="Also log at row boundaries every N seconds; 0 disables. Default: 30")
    p.add_argument("--log_every", type=int, default=100000,
                   help="Log progress every N input rows. Default: 100000")

    args = p.parse_args()

    if args.kmer_size != 6:
        raise ValueError("This script is currently designed for 6mer extraction. Please keep --kmer_size 6.")

    if args.min_window_spacing < 1:
        raise ValueError("--min_window_spacing must be >= 1.")
    if not np.isfinite(args.log_seconds) or args.log_seconds < 0:
        raise ValueError("--log_seconds must be a nonnegative finite number.")
    if args.shuffle_buffer_size < 0:
        raise ValueError("--shuffle_buffer_size must be >= 0.")

    if not (1 <= args.target_index <= args.kmer_size):
        raise ValueError("--target_index must be within [1, kmer_size].")

    args.target_labels = parse_label_list(args.target_labels)
    if not args.target_labels or len(args.target_labels) != len(set(args.target_labels)):
        raise ValueError("--target_labels must contain distinct labels and cannot be empty.")

    for lab in args.target_labels:
        if lab not in (0, 1, 2, 3):
            raise ValueError('This script is for negative labels 0-3 only, but got label={0}'.format(lab))

    args.mask_center = bool(args.mask_center)
    if len(args.mask_base) != 1:
        raise ValueError("--mask_base must contain exactly one character.")
    if not np.isfinite(args.dwell_scale) or args.dwell_scale <= 0:
        raise ValueError("--dwell_scale must be a positive finite number.")
    if not np.isfinite(args.quality_scale) or args.quality_scale <= 0:
        raise ValueError("--quality_scale must be a positive finite number.")

    # 即使旧命令没有传 --meta_out，也保留 read 身份，防止后续无法审计。
    args.meta_out = args.meta_out or (args.output + ".meta.tsv")
    args.report_out = args.report_out or (args.output + ".sampling_report.csv")

    args.read_split_out = args.read_split_out or (args.output + ".read_splits.tsv")
    args.split_ratio_spec = args.split_ratios
    args.split_ratios = parse_split_ratios(args.split_ratios)
    args.split_paths = get_split_output_paths(args.split_out_dir)
    input_paths = {os.path.realpath(args.input), os.path.realpath(args.positive_count_csv)}
    all_outputs = [args.output, args.meta_out, args.report_out, args.read_split_out]
    for paths in args.split_paths.values():
        all_outputs.extend((paths["features"], paths["meta"]))
    output_paths = [os.path.realpath(p) for p in all_outputs]
    if len(set(output_paths)) != len(output_paths) or input_paths.intersection(output_paths):
        raise ValueError("All output files must be distinct and cannot overwrite either input.")

    return args


def main():
    args = parse_args()

    print("[Step 0] Loading positive kmer target counts...", file=sys.stderr, flush=True)

    target_by_kmer, kmer_col, label4_col, label5_col, label4_sum, label5_sum = load_positive_target_counts(
        count_csv=args.positive_count_csv,
        kmer_col=args.kmer_col,
        label4_col=args.label4_col,
        label5_col=args.label5_col,
    )

    # 配额为 0 的组不参加候选抽样。
    target_kmers = {kmer for kmer, target in target_by_kmer.items() if target > 0}
    target_labels = set(args.target_labels)

    print('[Target CSV] kmer_col={0}, label4_col={1}, label5_col={2}'.format(kmer_col, label4_col, label5_col), file=sys.stderr)
    print('[Target CSV] target_kmers={0:,}'.format(len(target_kmers)), file=sys.stderr)
    print('[Target CSV] sum(label_4)={0:,}, sum(label_5)={1:,}'.format(label4_sum, label5_sum), file=sys.stderr)
    print('[Target CSV] sum(max(label_4,label_5) per kmer)={0:,}'.format(sum(target_by_kmer.values())), file=sys.stderr)
    print('[Target] negative labels={0}'.format(sorted(target_labels)), file=sys.stderr)
    print('[Target] mask_center={0}, mask_base={1}'.format(args.mask_center, args.mask_base), file=sys.stderr)
    print('[Sampling] multiple windows per read; min START spacing={0} nt; overlap allowed={1}'.format(
        args.min_window_spacing, args.min_window_spacing < args.kmer_size), file=sys.stderr)
    if args.quota_mode == "strict":
        print("[Quota] STRICT: each group must reach max(label_4,label_5); insufficient data aborts before feature writing", file=sys.stderr)
    else:
        print("[Quota] AVAILABLE: each group outputs min(max(label_4,label_5), available); shortages do not stop extraction", file=sys.stderr)
    print("[Sampling] first QC-passing record per read_id; stable random phase; no per-read window cap", file=sys.stderr)
    print('[Read split] expected READ ratios={0}; split_seed={1}; method={2}'.format(
        args.split_ratio_spec, args.split_seed, SPLIT_METHOD), file=sys.stderr)
    print("[Split reminder] Reuse this exact read split rule/configuration for X/Y. Never split rows again.", file=sys.stderr)
    print("[Training reminder] Learning-rate scheduling, early stopping and checkpoint choice must use validation only.", file=sys.stderr)
    print("[Labels] A=0, T=1, C=2, G=3; kmer is context, not a new class label", file=sys.stderr)
    print('[Metadata] {0}'.format(args.meta_out), file=sys.stderr)
    print('[Report] {0}'.format(args.report_out), file=sys.stderr)
    print('[Read manifest] {0}'.format(args.read_split_out), file=sys.stderr)
    if args.split_out_dir:
        print('[Partition files] {0}'.format(os.path.abspath(args.split_out_dir)), file=sys.stderr)
    else:
        print("[Partition files] Not requested; pooled CSV is NOT a training partition. Route rows using metadata split.", file=sys.stderr)

    target_by_group = {}
    for kmer, target in target_by_kmer.items():
        for label in target_labels:
            target_by_group[(kmer, label)] = target

    print("\n[Step 1] First pass: count eligible spaced windows per read...", file=sys.stderr, flush=True)

    input_stat = os.stat(args.input)
    input_signature = (input_stat.st_size, input_stat.st_mtime_ns)
    available, stats1, read_plan, available_reads = scan_available_counts(
        args=args,
        target_kmers=target_kmers,
        target_labels=target_labels,
    )

    effective_target = {}
    shortage_groups = 0
    shortage_total = 0

    for group, target in target_by_group.items():
        avail = available.get(group, 0)
        eff = min(target, avail)
        effective_target[group] = eff

        if avail < target:
            shortage_groups += 1
            shortage_total += (target - avail)

    print("\n[Pass 1 Summary]", file=sys.stderr)
    for k in sorted(stats1):
        print('{0}: {1}'.format(k, stats1[k]), file=sys.stderr)

    print("\n[Availability Summary]", file=sys.stderr)
    print('target_groups: {0:,}'.format(len(target_by_group)), file=sys.stderr)
    print('groups_with_available_candidates: {0:,}'.format(len(available)), file=sys.stderr)
    print('positive_target_total_for_all_negative_labels: {0:,}'.format(sum(target_by_group.values())), file=sys.stderr)
    print('available_spaced_candidate_windows: {0:,}'.format(sum(available.values())), file=sys.stderr)
    print('unique_reads_with_spaced_candidates: {0:,}'.format(stats1['read_ids_with_candidate']), file=sys.stderr)
    print('groups_with_zero_candidates: {0:,}'.format(len(target_by_group) - len(available)), file=sys.stderr)
    print('overall_target_fraction_available: {0:.4%}'.format(sum(effective_target.values()) / max(1, sum(target_by_group.values()))), file=sys.stderr)
    print('effective_output_total: {0:,}'.format(sum(effective_target.values())), file=sys.stderr)
    print('shortage_groups: {0:,}'.format(shortage_groups), file=sys.stderr)
    print('shortage_total: {0:,}'.format(shortage_total), file=sys.stderr)

    current_stat = os.stat(args.input)
    if (current_stat.st_size, current_stat.st_mtime_ns) != input_signature:
        raise RuntimeError("Input TSV changed during pass 1; restart with a fixed input file.")

    if shortage_groups > 0 and args.quota_mode == "strict":
        # Report every group, but leave existing feature/metadata/partition files untouched.
        report_dir = os.path.dirname(os.path.abspath(args.report_out))
        os.makedirs(report_dir, exist_ok=True)
        descriptor, temp_report = tempfile.mkstemp(prefix=".quota-shortage-", suffix=".csv", dir=report_dir)
        os.close(descriptor)
        report_args = copy.copy(args)
        report_args.report_out = temp_report
        try:
            write_report(
                report_args, target_by_group, available, effective_target, Counter(),
                available_reads, Counter(), status="insufficient_candidates_no_features_written",
            )
            os.replace(temp_report, args.report_out)
        finally:
            if os.path.exists(temp_report):
                os.remove(temp_report)
        print("\n[ERROR] Strict original quotas cannot be met from the available unique windows.", file=sys.stderr)
        missing = [(group, target - available.get(group, 0))
                   for group, target in target_by_group.items() if target > available.get(group, 0)]
        for (kmer, label), shortage in sorted(missing, key=lambda item: item[1], reverse=True)[:20]:
            print('  kmer={0} label={1} target={2:,} available={3:,} shortage={4:,}'.format(
                kmer, label, target_by_group[(kmer, label)], available.get((kmer, label), 0), shortage),
                file=sys.stderr)
        print('[Shortage report] {0}'.format(args.report_out), file=sys.stderr)
        if args.min_window_spacing > 1:
            print("Use --min_window_spacing 1 to examine every overlapping window.", file=sys.stderr)
        else:
            print("Every start position was examined. Additional real reads are required for these groups.", file=sys.stderr)
        print("[NOT COMPLETE] No new feature, metadata, read manifest or partition files were generated.", file=sys.stderr)
        print("Existing files with those names, if any, were NOT updated. No windows were copied to fill quotas.", file=sys.stderr)
        return 2

    if shortage_groups > 0:
        print("\n[CONTINUING WITH SHORTAGES] Keeping all available real windows in shortage groups; other groups retain their original quota.", file=sys.stderr)
        missing = [(group, target - available.get(group, 0))
                   for group, target in target_by_group.items() if target > available.get(group, 0)]
        for (kmer, label), shortage in sorted(missing, key=lambda item: item[1], reverse=True)[:20]:
            print('  kmer={0} label={1} target={2:,} available={3:,} shortage={4:,}'.format(
                kmer, label, target_by_group[(kmer, label)], available.get((kmer, label), 0), shortage),
                file=sys.stderr)
        print("[Report] Original targets, effective targets and remaining shortages will be preserved in the final report.", file=sys.stderr)

    read_splits = {read_id: assign_read_split(read_id, args.split_seed, args.split_ratios)
                   for read_id in read_plan}
    staged_args, staged_pairs = prepare_staged_outputs(args)
    publishing = False
    try:
        print("\n[Step 2] Sample exact effective targets and write staged paired outputs...", file=sys.stderr, flush=True)
        selected, stats2, needed_left, selected_reads, selected_per_read, split_counts, split_reads = sample_and_write(
            args=staged_args, target_kmers=target_kmers, target_labels=target_labels,
            available=available, effective_target=effective_target,
            read_plan=read_plan, read_splits=read_splits,
        )
        current_stat = os.stat(args.input)
        if (current_stat.st_size, current_stat.st_mtime_ns) != input_signature:
            raise RuntimeError("Input TSV changed during pass 2; staged outputs will be discarded.")

        print("\n[Pass 2 Summary]", file=sys.stderr)
        for key in sorted(stats2):
            print('{0}: {1}'.format(key, stats2[key]), file=sys.stderr)
        mismatch_groups = sum(selected.get(group, 0) != target for group, target in target_by_group.items())
        mismatch_effective = sum(selected.get(group, 0) != target for group, target in effective_target.items())
        print("\n[Final Check]", file=sys.stderr)
        print('selected_total: {0:,}'.format(sum(selected.values())), file=sys.stderr)
        print('original_quota_total: {0:,}'.format(sum(target_by_group.values())), file=sys.stderr)
        print('effective_target_total: {0:,}'.format(sum(effective_target.values())), file=sys.stderr)
        print('mismatch_groups_selected_vs_original_quota: {0:,}'.format(mismatch_groups), file=sys.stderr)
        print('mismatch_groups_selected_vs_effective_target: {0:,}'.format(mismatch_effective), file=sys.stderr)
        print('remaining_original_shortage_total: {0:,}'.format(sum(
            max(0, target - selected.get(group, 0)) for group, target in target_by_group.items())), file=sys.stderr)
        print('selected_unique_read_ids: {0:,}'.format(stats2['selected_unique_read_ids']), file=sys.stderr)
        print('duplicate_windows_in_output: {0}'.format(stats2['duplicate_windows_in_output']), file=sys.stderr)
        for left, right in (("train", "validation"), ("train", "test"), ("validation", "test")):
            print('read_overlap_{0}_{1}: {2}'.format(left, right, stats2['read_overlap_' + left + '_' + right]), file=sys.stderr)
        if mismatch_effective or any(needed_left.values()):
            raise RuntimeError("Effective targets not met. Staged feature files will not be published.")
        if args.quota_mode == "strict" and mismatch_groups:
            raise RuntimeError("Strict original quotas not met. Staged feature files will not be published.")
        label_selected = Counter()
        for (kmer, label), count in selected.items():
            label_selected[label] += count
        print("\n[Selected Count By Label]", file=sys.stderr)
        for label in sorted(label_selected):
            print('label_{0}: {1:,}'.format(label, label_selected[label]), file=sys.stderr)

        write_read_manifest(staged_args.read_split_out, args, read_plan, read_splits, selected_per_read)
        write_report(
            staged_args, target_by_group, available, effective_target, selected,
            available_reads, selected_reads,
            status="complete_with_shortages" if shortage_groups else "complete",
            split_counts=split_counts, split_reads=split_reads,
        )
        publishing = True
        publish_staged_outputs(staged_pairs)
    except BaseException:
        if publishing:
            print("[ERROR] Output publication failed; some final files may have been replaced. Rerun before using them.", file=sys.stderr)
        else:
            print("[ERROR] Run failed before publication; staged files discarded and existing final data files unchanged.", file=sys.stderr)
        raise
    finally:
        cleanup_staged_outputs(staged_pairs)

    print("\n[SUCCESS] Every effective kmer+label target was met without replacement; all read partitions are disjoint.", file=sys.stderr)
    if shortage_groups:
        print('[COMPLETED WITH SHORTAGES] {0} groups remain below their original quota by {1:,} windows in total; see report.'.format(
            shortage_groups, shortage_total), file=sys.stderr)
    else:
        print("[Original quotas] All original quotas were also met.", file=sys.stderr)
    print('[Done] pooled features: {0}'.format(args.output), file=sys.stderr)
    print('[Done] paired metadata: {0}'.format(args.meta_out), file=sys.stderr)
    print('[Done] read manifest: {0}'.format(args.read_split_out), file=sys.stderr)
    print('[Done] report: {0}'.format(args.report_out), file=sys.stderr)
    if args.split_out_dir:
        print('[Done] E. coli partition files: {0}'.format(os.path.abspath(args.split_out_dir)), file=sys.stderr)
    return 0


if __name__ == "__main__":
    sys.exit(main())
