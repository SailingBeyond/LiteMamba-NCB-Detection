#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Create a per-reference-position signal table from Tombo-resquiggled single-read FAST5 files.

The script joins three synchronized sources by read ID:
1. Tombo events and raw current stored in single-read FAST5 files;
2. alignment, base quality, and CIGAR-derived error features from a filtered BAM;
3. the canonicalized reference FASTA used for both resquiggle and alignment.

It supports processing only a numeric immediate-subfolder range so large FAST5
collections can be handled in reproducible batches. The output is the 12-column
TSV consumed by scripts/04_feature_extraction/extract_xna_6mer_features_all.py.
"""

from __future__ import absolute_import
import argparse
import os
import re
import h5py
import multiprocessing
import subprocess
import tempfile
import shutil
from collections import defaultdict

import numpy as np
from tqdm import tqdm

CIGAR_RE = re.compile(r'(\d+)([MIDNSHP=X])')
COMP_TABLE = str.maketrans("ACGTNacgtn", "TGCANtgcan")


def revcomp(seq):
    return seq.translate(COMP_TABLE)[::-1]


def decode_if_bytes(x):
    if isinstance(x, bytes):
        return x.decode()
    return x


def base_to_mismatch_value(base):
    base = base.upper()
    if base == "A":
        return 0.25
    elif base == "T":
        return 0.5
    elif base == "C":
        return 0.75
    elif base == "G":
        return 1.0
    return 0.0


def load_reference_fasta(reference_path):
    reference_dict = {}
    contig = None
    with open(reference_path) as f:
        for line in f:
            line = line.rstrip()
            if not line:
                continue
            if line.startswith(">"):
                contig = line.split()[0][1:]
                reference_dict[contig] = []
            else:
                reference_dict[contig].append(line)
    return {k: "".join(v) for k, v in reference_dict.items()}


def build_fast5_bucket_index(fast5_root, folder_start=None, folder_end=None):
    """
    扫描 FAST5 文件并建立 read_id -> bucket_idx 索引。

    新增功能：
    - 如果同时指定 folder_start 和 folder_end：
      只扫描 fast5_root 下一级目录中，名称为 folder_start...folder_end 的数字文件夹。
      例如：--folder_start 0 --folder_end 499
      只扫描 fast5_root/0, fast5_root/1, ..., fast5_root/499。
    - 如果不指定 folder_start / folder_end：
      默认扫描 fast5_root 下所有含 .fast5 的文件夹，兼容旧逻辑。

    注意：
    - 这里仍然沿用原始代码逻辑：直接使用 FAST5 文件名 stem 作为 read_id。
    - 每个被扫描的文件夹作为一个 bucket。
    """
    if (folder_start is None) ^ (folder_end is None):
        raise ValueError("folder_start 和 folder_end 必须同时指定，或者同时不指定。")

    if folder_start is not None:
        folder_start = int(folder_start)
        folder_end = int(folder_end)
        if folder_start > folder_end:
            raise ValueError(f"folder_start ({folder_start}) 不能大于 folder_end ({folder_end})。")

    bucket_dirs = []
    read_to_bucket = {}
    total_fast5_files = 0
    duplicate_read_ids = 0
    missing_selected_folders = []
    empty_selected_folders = []

    def add_one_folder(dirpath, folder_label=None):
        nonlocal total_fast5_files, duplicate_read_ids

        try:
            fast5_names = sorted(fn for fn in os.listdir(dirpath) if fn.endswith(".fast5"))
        except Exception:
            if folder_label is not None:
                missing_selected_folders.append(str(folder_label))
            return

        if not fast5_names:
            if folder_label is not None:
                empty_selected_folders.append(str(folder_label))
            return

        bucket_idx = len(bucket_dirs)
        bucket_dirs.append(dirpath)

        for fn in fast5_names:
            total_fast5_files += 1
            read_id = os.path.splitext(fn)[0]
            if read_id in read_to_bucket:
                duplicate_read_ids += 1
                continue
            read_to_bucket[read_id] = bucket_idx

    if folder_start is not None:
        # 新模式：只扫描指定数字范围内的一级文件夹
        for folder_id in range(folder_start, folder_end + 1):
            dirpath = os.path.join(fast5_root, str(folder_id))
            if not os.path.isdir(dirpath):
                missing_selected_folders.append(str(folder_id))
                continue
            add_one_folder(dirpath, folder_label=folder_id)
    else:
        # 旧模式：扫描所有含 fast5 的文件夹
        # 优先按 fast5_root 下一级数字文件夹的自然顺序扫描；
        # 如果不存在这种结构，则退回 os.walk 兼容任意层级目录。
        numeric_dirs = []
        try:
            with os.scandir(fast5_root) as it:
                for entry in it:
                    if not entry.is_dir():
                        continue
                    try:
                        folder_id = int(entry.name)
                    except ValueError:
                        continue
                    numeric_dirs.append((folder_id, entry.path))
        except Exception:
            numeric_dirs = []

        if numeric_dirs:
            for folder_id, dirpath in sorted(numeric_dirs, key=lambda x: x[0]):
                add_one_folder(dirpath, folder_label=folder_id)
        else:
            for dirpath, _, filenames in os.walk(fast5_root):
                fast5_names = [fn for fn in filenames if fn.endswith(".fast5")]
                if not fast5_names:
                    continue

                bucket_idx = len(bucket_dirs)
                bucket_dirs.append(dirpath)

                for fn in sorted(fast5_names):
                    total_fast5_files += 1
                    read_id = os.path.splitext(fn)[0]
                    if read_id in read_to_bucket:
                        duplicate_read_ids += 1
                        continue
                    read_to_bucket[read_id] = bucket_idx

    return (
        bucket_dirs,
        read_to_bucket,
        total_fast5_files,
        duplicate_read_ids,
        missing_selected_folders,
        empty_selected_folders,
    )


def cigar_to_ref_len(cigar):
    ref_len = 0
    for length_str, op in CIGAR_RE.findall(cigar):
        length = int(length_str)
        if op in ("M", "=", "X", "D", "N"):
            ref_len += length
    return ref_len


def build_features_from_alignment(seq, qual, cigar, ref_seq):
    """
    直接从 BAM + reference 生成：
    - qualities
    - mismatch
    - insertion
    - deletion
    长度全部按参考位点对齐
    """
    if qual == "*" or qual is None:
        base_quality_list = [0] * len(seq)
    else:
        base_quality_list = [ord(ch) - 33 for ch in qual]

    mapped_qualities = []
    mismatch_features = []
    insertion_features = []
    deletion_features = []

    query_index = 0
    ref_index = 0
    pending_insertion = 0

    for length_str, op in CIGAR_RE.findall(cigar):
        length = int(length_str)

        if op in ("M", "=", "X"):
            for _ in range(length):
                read_base = seq[query_index].upper() if query_index < len(seq) else "N"
                ref_base = ref_seq[ref_index].upper() if ref_index < len(ref_seq) else "N"
                read_qual = base_quality_list[query_index] if query_index < len(base_quality_list) else 0

                mapped_qualities.append(read_qual)

                if pending_insertion > 0:
                    insertion_features.append(min(1.0, pending_insertion / 10.0))
                    pending_insertion = 0
                else:
                    insertion_features.append(0.0)

                deletion_features.append(0.0)

                if op == "=":
                    mismatch_features.append(0.0)
                elif op == "X":
                    mismatch_features.append(base_to_mismatch_value(read_base))
                else:  # M
                    if read_base != ref_base:
                        mismatch_features.append(base_to_mismatch_value(read_base))
                    else:
                        mismatch_features.append(0.0)

                query_index += 1
                ref_index += 1

        elif op == "I":
            query_index += length
            pending_insertion += length

        elif op in ("D", "N"):
            for _ in range(length):
                mapped_qualities.append(0)

                if pending_insertion > 0:
                    insertion_features.append(min(1.0, pending_insertion / 10.0))
                    pending_insertion = 0
                else:
                    insertion_features.append(0.0)

                deletion_features.append(1.0)
                mismatch_features.append(0.0)
                ref_index += 1

        elif op == "S":
            query_index += length

        elif op in ("H", "P"):
            pass

        else:
            raise ValueError(f"Unsupported CIGAR op: {op}")

    return mapped_qualities, mismatch_features, insertion_features, deletion_features


def samtools_supports_name_filter(samtools_bin):
    try:
        result = subprocess.run(
            [samtools_bin, "view", "--help"],
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            text=True,
            check=False,
        )
        return " -N " in result.stdout or "\n  -N " in result.stdout or "--qname-file" in result.stdout
    except Exception:
        return False


def iter_bam_records_with_samtools(bam_path, samtools_bin="samtools", threads=1):
    cmd = [samtools_bin, "view", "-@", str(threads), bam_path]
    proc = subprocess.Popen(cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True, bufsize=1)

    try:
        for line in proc.stdout:
            if not line:
                continue
            yield line.rstrip("\n")
    finally:
        proc.stdout.close()
        stderr = proc.stderr.read()
        ret = proc.wait()
        if ret != 0:
            raise RuntimeError(f"samtools view failed (exit={ret}). stderr:\n{stderr}")


def serialize_int_list(vals):
    return "|".join(str(x) for x in vals)


def serialize_float_list(vals):
    return "|".join(str(x) for x in vals)


def deserialize_int_list(s):
    if not s:
        return []
    return [int(x) for x in s.split("|")]


def deserialize_float_list(s):
    if not s:
        return []
    return [float(x) for x in s.split("|")]


def stream_bam_to_bucket_files(reference_path, bam_path, read_to_bucket, cache_dir):
    """
    只流式扫一遍 BAM。
    如果 read_id 出现在 fast5 文件名索引中，就计算特征并写入对应 bucket 临时文件。
    """
    reference_dict = load_reference_fasta(reference_path)
    bucket_fp_dict = {}
    stats = defaultdict(int)

    try:
        iterator = iter_bam_records_with_samtools(
            bam_path=bam_path,
            samtools_bin=args.samtools,
            threads=int(args.bam_threads),
        )

        for line in tqdm(iterator, desc="Streaming BAM -> bucket cache", unit="read"):
            stats["bam_total"] += 1

            items = line.split("\t")
            if len(items) < 11:
                stats["bam_malformed"] += 1
                continue

            read_id = items[0]
            bucket_idx = read_to_bucket.get(read_id, None)
            if bucket_idx is None:
                stats["bam_not_in_fast5_index"] += 1
                continue

            flag = int(items[1])
            chrom = items[2]
            pos1 = int(items[3])
            cigar = items[5]
            seq = items[9]
            qual = items[10]

            if chrom == "*" or cigar == "*" or seq == "*":
                stats["bam_unmapped_or_empty"] += 1
                continue
            if flag & int(args.skip_flags):
                stats["bam_skipped_by_flag"] += 1
                continue
            if chrom not in reference_dict:
                stats["bam_missing_ref"] += 1
                continue

            strand = "-" if (flag & 16) else "+"
            ref_len = cigar_to_ref_len(cigar)
            if ref_len <= 0:
                stats["bam_zero_ref_len"] += 1
                continue

            end1 = pos1 + ref_len - 1
            ref_seq = reference_dict[chrom][pos1 - 1:end1]
            if len(ref_seq) == 0:
                stats["bam_empty_ref_seq"] += 1
                continue

            mapped_qualities, mismatch, insertion, deletion = build_features_from_alignment(
                seq=seq,
                qual=qual,
                cigar=cigar,
                ref_seq=ref_seq,
            )

            min_len = min(
                len(ref_seq),
                len(mapped_qualities),
                len(mismatch),
                len(insertion),
                len(deletion),
            )
            if min_len <= 0:
                stats["bam_empty_after_feature_build"] += 1
                continue

            ref_seq = ref_seq[:min_len]
            mapped_qualities = mapped_qualities[:min_len]
            mismatch = mismatch[:min_len]
            insertion = insertion[:min_len]
            deletion = deletion[:min_len]
            end1 = pos1 + min_len - 1

            bucket_path = os.path.join(cache_dir, f"bucket_{bucket_idx:05d}.tsv")
            if bucket_idx not in bucket_fp_dict:
                bucket_fp_dict[bucket_idx] = open(bucket_path, "w")

            fp = bucket_fp_dict[bucket_idx]
            fp.write(
                f"{read_id}\t{chrom}\t{pos1}\t{end1}\t{strand}\t{ref_seq}\t"
                f"{serialize_int_list(mapped_qualities)}\t"
                f"{serialize_float_list(mismatch)}\t"
                f"{serialize_float_list(insertion)}\t"
                f"{serialize_float_list(deletion)}\n"
            )
            stats["bam_written_to_bucket"] += 1

    finally:
        for fp in bucket_fp_dict.values():
            fp.close()

    return stats


def load_bucket_records(bucket_path):
    """
    读取单个 bucket 文件（通常对应一个 fast5 文件夹，约 4000 条）。
    """
    bam_dict = {}
    if not os.path.exists(bucket_path):
        return bam_dict

    with open(bucket_path) as f:
        for line in f:
            parts = line.rstrip("\n").split("\t")
            if len(parts) != 10:
                continue

            read_id, chrom, start1, end1, strand, ref_seq, quals_s, mismatch_s, insertion_s, deletion_s = parts
            bam_dict[read_id] = {
                "chrom": chrom,
                "start1": int(start1),
                "end1": int(end1),
                "strand": strand,
                "ref_seq": ref_seq,
                "qualities": deserialize_int_list(quals_s),
                "mismatch": deserialize_float_list(mismatch_s),
                "insertion": deserialize_float_list(insertion_s),
                "deletion": deserialize_float_list(deletion_s),
            }

    return bam_dict


def read_fast5_signal_once(task):
    """
    task = (fast5_path, read_id_from_filename, basecall_group, basecall_subgroup)

    只打开一次 HDF5：
    - 读 raw signal
    - 读 scaling / offset
    - 读 Tombo events
    - 读 Tombo alignment attrs
    """
    fast5_path, read_id, basecall_group, basecall_subgroup = task

    try:
        with h5py.File(fast5_path, "r") as fast5_data:
            # raw signal
            raw_read_group = list(fast5_data["/Raw/Reads/"].values())[0]
            raw_data = raw_read_group["Signal"][()]

            # scaling
            channel_info = fast5_data["/UniqueGlobalKey/channel_id"].attrs
            scaling = channel_info["range"] / channel_info["digitisation"]
            offset = channel_info["offset"]

            # corrected events
            corr_data = fast5_data[f"/Analyses/{basecall_group}/{basecall_subgroup}/Events"]
            corr_attrs = dict(corr_data.attrs.items())
            corr_data = corr_data[()]

            # alignment attrs
            alignment_data = fast5_data[f"/Analyses/{basecall_group}/{basecall_subgroup}/Alignment"]
            mapped_start0 = int(alignment_data.attrs["mapped_start"])   # 0-based inclusive
            mapped_end0 = int(alignment_data.attrs["mapped_end"])       # 0-based exclusive / open interval end
            mapped_strand = decode_if_bytes(alignment_data.attrs.get("mapped_strand", "+"))
            mapped_chrom = decode_if_bytes(alignment_data.attrs.get("mapped_chrom", ""))

            corr_start_rel_to_raw = int(corr_attrs["read_start_rel_to_raw"])

            if len(raw_data) > 99999999:
                return None
            if any(len(vals) <= 1 for vals in (corr_data, raw_data)):
                return None

            event_starts = corr_data["start"] + corr_start_rel_to_raw
            event_lengths = corr_data["length"]
            event_bases = corr_data["base"]

            # rescale
            raw_data = np.round(scaling * (raw_data + offset), 3).tolist()
            sequence = "".join([decode_if_bytes(x) for x in event_bases])

            signal_list = []
            for i in range(len(event_lengths)):
                seg = raw_data[event_starts[i]: event_starts[i] + event_lengths[i]]
                signal_list.append("*".join(str(x) for x in seg))

        return {
            "read_id": read_id,
            "sequence": sequence,
            "signal_list": signal_list,
            "fast5_start1": mapped_start0 + 1,
            "fast5_end1": mapped_end0,
            "fast5_strand": mapped_strand,
            "fast5_chrom": mapped_chrom,
        }

    except Exception:
        return None


def slice_fast5_to_reference_orientation(signal_info, overlap_start1, overlap_end1):
    fast5_start1 = signal_info["fast5_start1"]
    fast5_end1 = signal_info["fast5_end1"]
    fast5_strand = signal_info["fast5_strand"]

    if overlap_start1 > overlap_end1:
        return None, None

    if fast5_strand == "+":
        left = overlap_start1 - fast5_start1
        right = overlap_end1 - fast5_start1 + 1
        seq_slice = signal_info["sequence"][left:right]
        sig_slice = signal_info["signal_list"][left:right]
        return seq_slice, sig_slice

    left = fast5_end1 - overlap_end1
    right = fast5_end1 - overlap_start1 + 1
    seq_slice = signal_info["sequence"][left:right]
    sig_slice = signal_info["signal_list"][left:right]

    seq_slice = revcomp(seq_slice)
    sig_slice = sig_slice[::-1]
    return seq_slice, sig_slice


def reconcile_lengths(ref_seq, seq_from_fast5, signal_list, quals, mismatch, insertion, deletion,
                      max_len_gap=10, always_take_intersection=False):
    lengths = {
        "ref_seq": len(ref_seq),
        "seq_from_fast5": len(seq_from_fast5),
        "signal": len(signal_list),
        "quals": len(quals),
        "mismatch": len(mismatch),
        "insertion": len(insertion),
        "deletion": len(deletion),
    }

    min_len = min(lengths.values())
    max_len = max(lengths.values())
    gap = max_len - min_len

    if min_len <= 0:
        return None, {
            "status": "empty_after_intersection",
            "lengths": lengths,
            "gap": gap,
        }

    if gap == 0:
        return {
            "ref_seq": ref_seq,
            "seq_from_fast5": seq_from_fast5,
            "signal": signal_list,
            "quals": quals,
            "mismatch": mismatch,
            "insertion": insertion,
            "deletion": deletion,
            "common_len": min_len,
            "gap": gap,
            "reconciled": False,
        }, None

    if (gap <= max_len_gap) or always_take_intersection:
        return {
            "ref_seq": ref_seq[:min_len],
            "seq_from_fast5": seq_from_fast5[:min_len],
            "signal": signal_list[:min_len],
            "quals": quals[:min_len],
            "mismatch": mismatch[:min_len],
            "insertion": insertion[:min_len],
            "deletion": deletion[:min_len],
            "common_len": min_len,
            "gap": gap,
            "reconciled": True,
        }, None

    return None, {
        "status": "length_gap_too_large",
        "lengths": lengths,
        "gap": gap,
    }


def open_optional_file(path, mode="w"):
    if path is None:
        return None
    return open(path, mode)


def iter_folder_fast5_tasks(folder_path, valid_read_ids, basecall_group, basecall_subgroup):
    tasks = []
    with os.scandir(folder_path) as it:
        for entry in it:
            if entry.is_file() and entry.name.endswith(".fast5"):
                read_id = os.path.splitext(entry.name)[0]
                if read_id in valid_read_ids:
                    tasks.append((entry.path, read_id, basecall_group, basecall_subgroup))
    return tasks


def process_signal_info(signal_info, bam_dict, output_fp, mismatch_fp, stats):
    if signal_info is None:
        stats["fast5_failed"] += 1
        return

    read_id = signal_info["read_id"]
    bam_info = bam_dict.get(read_id)
    if bam_info is None:
        stats["not_in_bam_bucket"] += 1
        return

    tombo_chrom = signal_info["fast5_chrom"]
    tombo_strand = signal_info["fast5_strand"]
    bam_chrom = bam_info["chrom"]
    bam_strand = bam_info["strand"]

    if tombo_chrom != bam_chrom:
        stats["chrom_mismatch"] += 1
        if mismatch_fp is not None:
            mismatch_fp.write(
                f"{read_id}\t{bam_chrom}\t{bam_info['start1']}\t{bam_info['end1']}\t{bam_strand}\t"
                f"{tombo_chrom}\t{signal_info['fast5_start1']}\t{signal_info['fast5_end1']}\t{tombo_strand}\t"
                f"chrom_mismatch\t.\n"
            )
        return
    stats["chrom_match"] += 1

    strand_match = (bam_strand == tombo_strand)
    if strand_match:
        stats["strand_match"] += 1
    else:
        stats["strand_mismatch"] += 1
        if mismatch_fp is not None:
            mismatch_fp.write(
                f"{read_id}\t{bam_chrom}\t{bam_info['start1']}\t{bam_info['end1']}\t{bam_strand}\t"
                f"{tombo_chrom}\t{signal_info['fast5_start1']}\t{signal_info['fast5_end1']}\t{tombo_strand}\t"
                f"strand_mismatch\t.\n"
            )
        if args.require_strand_match:
            return

    overlap_start1 = max(signal_info["fast5_start1"], bam_info["start1"])
    overlap_end1 = min(signal_info["fast5_end1"], bam_info["end1"])
    if overlap_start1 > overlap_end1:
        stats["no_overlap"] += 1
        return

    seq_from_fast5, intersected_signal = slice_fast5_to_reference_orientation(
        signal_info, overlap_start1, overlap_end1
    )
    if intersected_signal is None:
        stats["slice_failed"] += 1
        return

    bam_left = overlap_start1 - bam_info["start1"]
    bam_right = overlap_end1 - bam_info["start1"] + 1

    ref_seq = bam_info["ref_seq"][bam_left:bam_right]
    quals = bam_info["qualities"][bam_left:bam_right]
    mismatch = bam_info["mismatch"][bam_left:bam_right]
    insertion = bam_info["insertion"][bam_left:bam_right]
    deletion = bam_info["deletion"][bam_left:bam_right]

    reconciled, err = reconcile_lengths(
        ref_seq=ref_seq,
        seq_from_fast5=seq_from_fast5,
        signal_list=intersected_signal,
        quals=quals,
        mismatch=mismatch,
        insertion=insertion,
        deletion=deletion,
        max_len_gap=args.max_len_gap,
        always_take_intersection=args.always_take_intersection,
    )

    if reconciled is None:
        stats[err["status"]] += 1
        if mismatch_fp is not None:
            mismatch_fp.write(
                f"{read_id}\t{bam_chrom}\t{bam_info['start1']}\t{bam_info['end1']}\t{bam_strand}\t"
                f"{tombo_chrom}\t{signal_info['fast5_start1']}\t{signal_info['fast5_end1']}\t{tombo_strand}\t"
                f"{err['status']}\t{err['lengths']}\n"
            )
        return

    if reconciled["reconciled"]:
        stats["length_reconciled"] += 1
    else:
        stats["length_exact_match"] += 1

    ref_seq = reconciled["ref_seq"]
    seq_from_fast5 = reconciled["seq_from_fast5"]
    intersected_signal = reconciled["signal"]
    quals = reconciled["quals"]
    mismatch = reconciled["mismatch"]
    insertion = reconciled["insertion"]
    deletion = reconciled["deletion"]

    seq_match_after_norm = (seq_from_fast5 == ref_seq)
    if seq_match_after_norm:
        stats["seq_match_after_norm"] += 1
    else:
        stats["seq_mismatch_after_norm"] += 1
        if args.require_seq_match:
            if mismatch_fp is not None:
                mismatch_fp.write(
                    f"{read_id}\t{bam_chrom}\t{bam_info['start1']}\t{bam_info['end1']}\t{bam_strand}\t"
                    f"{tombo_chrom}\t{signal_info['fast5_start1']}\t{signal_info['fast5_end1']}\t{tombo_strand}\t"
                    f"seq_mismatch_after_norm\t.\n"
                )
            return

    line = "%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\n" % (
        read_id,
        bam_chrom,
        overlap_start1,
        ref_seq,
        "|".join([str(x) for x in quals]),
        bam_strand,
        tombo_strand,
        "1" if strand_match else "0",
        "|".join(intersected_signal),
        "|".join([str(x) for x in mismatch]),
        "|".join([str(x) for x in insertion]),
        "|".join([str(x) for x in deletion]),
    )
    output_fp.write(line)
    stats["written"] += 1


def process_all_folders(bucket_dirs, cache_dir, output_path):
    stats = defaultdict(int)

    mismatch_fp = open_optional_file(args.strand_mismatch_output, "w")
    if mismatch_fp is not None:
        mismatch_fp.write(
            "read_id\tbam_chrom\tbam_start\tbam_end\tbam_strand\t"
            "tombo_chrom\ttombo_start\ttombo_end\ttombo_strand\treason\textra\n"
        )

    output_fp = open(output_path, "w")
    if args.with_header:
        output_fp.write(
            "read_id\tchrom\tstart\tref_seq\tbase_qualities\tbam_strand\t"
            "tombo_strand\tstrand_match\tsignal\tmismatch\tinsertion\tdeletion\n"
        )

    pool = None
    if int(args.process) > 1:
        pool = multiprocessing.Pool(processes=int(args.process))

    try:
        for bucket_idx, folder_path in enumerate(bucket_dirs):
            bucket_path = os.path.join(cache_dir, f"bucket_{bucket_idx:05d}.tsv")
            bam_dict = load_bucket_records(bucket_path)
            if not bam_dict:
                stats["empty_bucket"] += 1
                continue

            valid_read_ids = set(bam_dict.keys())
            tasks = iter_folder_fast5_tasks(
                folder_path=folder_path,
                valid_read_ids=valid_read_ids,
                basecall_group=args.basecall_group,
                basecall_subgroup=args.basecall_subgroup,
            )

            if not tasks:
                stats["empty_folder_tasks"] += 1
                continue

            desc = f"Folder {bucket_idx + 1}/{len(bucket_dirs)}"
            stats["folder_count_processed"] += 1
            stats["folder_fast5_tasks"] += len(tasks)

            if pool is not None:
                iterator = pool.imap_unordered(read_fast5_signal_once, tasks, chunksize=args.fast5_chunksize)
                for signal_info in tqdm(iterator, total=len(tasks), desc=desc, unit="fast5"):
                    stats["fast5_total"] += 1
                    process_signal_info(signal_info, bam_dict, output_fp, mismatch_fp, stats)
            else:
                for task in tqdm(tasks, total=len(tasks), desc=desc, unit="fast5"):
                    stats["fast5_total"] += 1
                    signal_info = read_fast5_signal_once(task)
                    process_signal_info(signal_info, bam_dict, output_fp, mismatch_fp, stats)

    finally:
        if pool is not None:
            pool.close()
            pool.join()
        output_fp.close()
        if mismatch_fp is not None:
            mismatch_fp.close()

    return stats


def main():
    if not os.path.isdir(args.fast5):
        raise FileNotFoundError(f"FAST5 root directory not found: {args.fast5}")
    if not os.path.isfile(args.reference):
        raise FileNotFoundError(f"Reference FASTA not found: {args.reference}")
    if not os.path.isfile(args.bam):
        raise FileNotFoundError(f"Filtered BAM not found: {args.bam}")
    if args.process <= 0 or args.bam_threads <= 0 or args.fast5_chunksize <= 0:
        raise ValueError("--process, --bam_threads, and --fast5_chunksize must be > 0")
    if args.max_len_gap < 0:
        raise ValueError("--max_len_gap must be >= 0")

    output_parent = os.path.dirname(os.path.abspath(args.output))
    os.makedirs(output_parent, exist_ok=True)
    if args.strand_mismatch_output:
        mismatch_parent = os.path.dirname(os.path.abspath(args.strand_mismatch_output))
        os.makedirs(mismatch_parent, exist_ok=True)
    if args.cache_dir:
        os.makedirs(args.cache_dir, exist_ok=True)

    print("[1/4] Scanning FAST5 filenames only (no HDF5 open) ...")
    (
        bucket_dirs,
        read_to_bucket,
        total_fast5_files,
        duplicate_read_ids,
        missing_selected_folders,
        empty_selected_folders,
    ) = build_fast5_bucket_index(
        args.fast5,
        folder_start=args.folder_start,
        folder_end=args.folder_end,
    )

    if args.folder_start is not None:
        print(f"Requested folder range: {args.folder_start} to {args.folder_end} (inclusive)")

    print(f"FAST5 files found in selected folders: {total_fast5_files}")
    print(f"FAST5 folders selected with FAST5 files: {len(bucket_dirs)}")
    print(f"Unique read IDs from FAST5 filenames: {len(read_to_bucket)}")

    if duplicate_read_ids:
        print(f"Duplicate FAST5 filename stems skipped: {duplicate_read_ids}")

    if missing_selected_folders:
        preview = ", ".join(missing_selected_folders[:20])
        suffix = " ..." if len(missing_selected_folders) > 20 else ""
        print(f"[Warning] Missing selected folders: {len(missing_selected_folders)} ({preview}{suffix})")

    if empty_selected_folders:
        preview = ", ".join(empty_selected_folders[:20])
        suffix = " ..." if len(empty_selected_folders) > 20 else ""
        print(f"[Warning] Selected folders without FAST5 files: {len(empty_selected_folders)} ({preview}{suffix})")

    if not bucket_dirs:
        raise RuntimeError("No FAST5 folders were selected. Please check --fast5 / --folder_start / --folder_end.")

    print("[2/4] Creating temporary BAM bucket cache ...")
    cache_dir = args.cache_dir
    created_temp_dir = False
    if cache_dir is None:
        cache_dir = tempfile.mkdtemp(prefix="fast5_bam_bucket_")
        created_temp_dir = True
    else:
        os.makedirs(cache_dir, exist_ok=True)

    bam_stats = stream_bam_to_bucket_files(
        reference_path=args.reference,
        bam_path=args.bam,
        read_to_bucket=read_to_bucket,
        cache_dir=cache_dir,
    )

    print("\n[BAM bucket summary]")
    for k in sorted(bam_stats.keys()):
        print(f"{k}: {bam_stats[k]}")

    print("[3/4] Reading FAST5 folder by folder and writing final output ...")
    final_stats = process_all_folders(
        bucket_dirs=bucket_dirs,
        cache_dir=cache_dir,
        output_path=args.output,
    )

    print("\n[Final Summary]")
    for k in sorted(final_stats.keys()):
        print(f"{k}: {final_stats[k]}")

    if final_stats.get("strand_mismatch", 0) > 0:
        print("\n[Note]")
        print("Tombo mapped_strand and BAM strand are not fully identical.")
        print("You should inspect the mismatch file and check whether the FAST5 files, resquiggle result,")
        print("and BAM all come from the same reference / same read set / same preprocessing round.")

    print("[4/4] Cleaning temporary cache ...")
    if created_temp_dir and (not args.keep_cache):
        shutil.rmtree(cache_dir, ignore_errors=True)
        print(f"Temporary cache removed: {cache_dir}")
    else:
        print(f"Temporary cache kept at: {cache_dir}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Fast per-base signal extraction from single-read FAST5 using BAM + reference only."
    )
    parser.add_argument("-o", "--output", required=True, help="Output TSV file.")
    parser.add_argument("--basecall_group", default="RawGenomeCorrected_000",
                        help="Tombo basecall group.")
    parser.add_argument("--basecall_subgroup", default="BaseCalled_template",
                        help="Tombo basecall subgroup.")
    parser.add_argument("-p", "--process", default=1, type=int,
                        help="FAST5 reading worker count.")
    parser.add_argument("--fast5_chunksize", default=32, type=int,
                        help="Multiprocessing chunksize for FAST5 reading. Default: 32.")
    parser.add_argument("--fast5", required=True,
                        help="Root directory containing single-read FAST5 files.")

    parser.add_argument("--folder_start", default=None, type=int,
                        help="Only scan numeric immediate subfolders whose names are >= this value. "
                             "Use with --folder_end. Example: --folder_start 0 --folder_end 499.")
    parser.add_argument("--folder_end", default=None, type=int,
                        help="Only scan numeric immediate subfolders whose names are <= this value. "
                             "Use with --folder_start. Example: --folder_start 0 --folder_end 499.")
    parser.add_argument("-r", "--reference", required=True,
                        help="Reference fasta file.")
    parser.add_argument("--bam", required=True,
                        help="Filtered BAM file.")
    parser.add_argument("--samtools", default="samtools",
                        help="Path to samtools executable.")
    parser.add_argument("--bam_threads", default=8, type=int,
                        help="samtools view decompression threads.")
    parser.add_argument("--skip_flags", default=3844, type=int,
                        help="Skip BAM records with these SAM flag bits set. Default: 3844.")
    parser.add_argument("--require_strand_match", action="store_true",
                        help="Only keep reads whose Tombo mapped_strand equals BAM strand.")
    parser.add_argument("--require_seq_match", action="store_true",
                        help="Only keep reads whose normalized FAST5 sequence equals BAM reference overlap.")
    parser.add_argument("--strand_mismatch_output", default=None,
                        help="Optional TSV file to record chrom/strand mismatch reads.")
    parser.add_argument("--with_header", action="store_true",
                        help="Write header line to output TSV.")
    parser.add_argument("--max_len_gap", default=10, type=int,
                        help="If lengths differ by no more than this many bases, truncate to common intersection. Default: 10.")
    parser.add_argument("--always_take_intersection", action="store_true",
                        help="Always truncate to the common shortest length, regardless of length gap.")
    parser.add_argument("--cache_dir", default=None,
                        help="Directory to store temporary BAM bucket cache. Default: auto temp dir.")
    parser.add_argument("--keep_cache", action="store_true",
                        help="Keep temporary BAM bucket cache after finishing.")
    args = parser.parse_args()

    main()