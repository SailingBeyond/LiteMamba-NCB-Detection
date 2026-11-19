#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import argparse
import csv
import os
import sys
from collections import Counter

import numpy as np

COMP_TABLE_XY = str.maketrans({
    "A": "T", "C": "G", "G": "C", "T": "A", "N": "N", "X": "Y", "Y": "X",
    "a": "t", "c": "g", "g": "c", "t": "a", "n": "n", "x": "y", "y": "x",
})


def revcomp_xy(seq: str) -> str:
    return seq.translate(COMP_TABLE_XY)[::-1]


def overlap_len(start1_a: int, end1_a: int, start1_b: int, end1_b: int) -> int:
    left = max(start1_a, start1_b)
    right = min(end1_a, end1_b)
    return max(0, right - left + 1)


def robust_mad_scale(arr: np.ndarray, use_unique: bool = True, eps: float = 1e-8):
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
    s = f"{float(x):.{ndigits}f}"
    s = s.rstrip("0").rstrip(".")
    return s if s else "0"


def build_target_pos_map_13_16():
    """
    1-based reference coordinates for all expected modified sites in XNA13-XNA16,
    and the corresponding natural sites in PC13-PC16.

    These positions are inferred from the original design table:
        XNA13 UB Kmer: CACXGTAAXGGG       -> X positions: 1291, 1296
        XNA14 UB Kmer: CACXGTACAAXGGG     -> X positions: 1291, 1298
        XNA15 UB Kmer: CACXGTATTACAAXGGG  -> X positions: 1291, 1301
        XNA16 UB Kmer: CACXGTATCGTTACAAXGGG -> X positions: 1291, 1304

    In the uploaded refdb.fasta, PC15 is one base shorter before the UB region
    than XNA15, so the corresponding PC15 sites are 1290 and 1300.
    """
    return {
        "XNA13": [1291, 1296],
        "XNA14": [1291, 1298],
        "XNA15": [1291, 1301],
        "XNA16": [1291, 1304],
        "PC13":  [1291, 1296],
        "PC14":  [1291, 1298],
        "PC15":  [1290, 1300],
        "PC16":  [1291, 1304],
    }


def compute_kmer_ref_window(target_pos: int, strand: str):
    """
    Return the 6mer reference-coordinate window in 1-based inclusive coordinates.

    + strand: output target base as the 4th base:
        [target-3, target+2]

    - strand: after reverse-complement, output target base as the 4th base:
        [target-2, target+3]
    """
    if strand == "+":
        return target_pos - 3, target_pos + 2
    if strand == "-":
        return target_pos - 2, target_pos + 3
    raise ValueError(f"Unsupported strand: {strand}")



def parse_args():
    p = argparse.ArgumentParser(
        description=(
            "Streamingly extract strand-aware 6mer features for all expected modified "
            "sites in XNA13-XNA16 and corresponding natural sites in PC13-PC16. "
            "The output kmer keeps the natural bases from the uploaded FASTA, "
            "without restoring X/Y markers, and outputs are split into + and - strand CSVs."
        )
    )
    p.add_argument("--input", required=True, help="Input TSV from extract_signal_from_fast5_XY.py")
    p.add_argument("--output_pos", required=True, help="Output feature CSV for + strand only")
    p.add_argument("--output_neg", required=True, help="Output feature CSV for - strand only")
    p.add_argument("--meta_out_pos", default=None, help="Optional metadata TSV for + strand output")
    p.add_argument("--meta_out_neg", default=None, help="Optional metadata TSV for - strand output")
    p.add_argument("--barcode_start", type=int, default=1240, help="Barcode region start (1-based). Default: 1240")
    p.add_argument("--barcode_end", type=int, default=1263, help="Barcode region end (1-based). Default: 1263")
    p.add_argument("--barcode_min_cover", type=int, default=15,
                   help="Minimum required barcode overlap length. Default: 15")
    p.add_argument("--dwell_scale", type=float, default=1,
                   help="Output dwell as len(segment)/dwell_scale. Default: 1")
    p.add_argument("--quality_scale", type=float, default=1,
                   help="Output quality as Q/quality_scale. Default: 1")
    p.add_argument("--mad_use_unique", type=int, choices=[0, 1], default=1,
                   help="Use unique full-read signal values for median-MAD scaling. Default: 1")
    p.add_argument("--require_strand_match", type=int, choices=[0, 1], default=1,
                   help="Skip rows whose bam_strand != tombo_strand. Default: 1")
    p.add_argument("--with_header", action="store_true", help="Write CSV header to both output files")
    p.add_argument("--signal_round", type=int, default=6, help="Decimal places for normalized raw signal. Default: 6")
    p.add_argument("--stat_round", type=int, default=6, help="Decimal places for summary statistics. Default: 6")
    p.add_argument("--log_every", type=int, default=100000, help="Log progress every N input rows. Default: 100000")
    return p.parse_args()


def main():
    args = parse_args()
    target_pos_map = build_target_pos_map_13_16()
    allowed_refs = set(target_pos_map)

    stats = Counter()

    for out_path in (args.output_pos, args.output_neg):
        out_dir = os.path.dirname(os.path.abspath(out_path))
        if out_dir:
            os.makedirs(out_dir, exist_ok=True)

    for meta_path in (args.meta_out_pos, args.meta_out_neg):
        if meta_path:
            meta_dir = os.path.dirname(os.path.abspath(meta_path))
            if meta_dir:
                os.makedirs(meta_dir, exist_ok=True)

    with open(args.input, "r", encoding="utf-8", buffering=1024 * 1024 * 16) as fin, \
         open(args.output_pos, "w", newline="", encoding="utf-8", buffering=1024 * 1024 * 16) as fout_pos, \
         open(args.output_neg, "w", newline="", encoding="utf-8", buffering=1024 * 1024 * 16) as fout_neg:

        writer_pos = csv.writer(fout_pos)
        writer_neg = csv.writer(fout_neg)

        meta_fp_pos = None
        meta_fp_neg = None
        meta_writer_pos = None
        meta_writer_neg = None

        if args.meta_out_pos:
            meta_fp_pos = open(args.meta_out_pos, "w", encoding="utf-8", buffering=1024 * 1024 * 8)
            meta_writer_pos = csv.writer(meta_fp_pos, delimiter='\t')
            meta_writer_pos.writerow([
                "read_id", "chrom", "bam_strand", "row_start", "row_end",
                "target_pos", "kmer_ref_start", "kmer_ref_end",
                "barcode_overlap", "kmer", "label"
            ])
        if args.meta_out_neg:
            meta_fp_neg = open(args.meta_out_neg, "w", encoding="utf-8", buffering=1024 * 1024 * 8)
            meta_writer_neg = csv.writer(meta_fp_neg, delimiter='\t')
            meta_writer_neg.writerow([
                "read_id", "chrom", "bam_strand", "row_start", "row_end",
                "target_pos", "kmer_ref_start", "kmer_ref_end",
                "barcode_overlap", "kmer", "label"
            ])

        try:
            if args.with_header:
                header = [
                    "kmer", "mean", "std", "median", "dwell",
                    "quality", "mismatch", "insertion", "deletion",
                    "signal", "label"
                ]
                writer_pos.writerow(header)
                writer_neg.writerow(header)

            for line_no, line in enumerate(fin, start=1):
                if line_no == 1 and line.startswith("read_id\t"):
                    stats["header_skipped"] += 1
                    continue

                stats["rows_total"] += 1
                if args.log_every > 0 and stats["rows_total"] % args.log_every == 0:
                    print(
                        f"[progress] rows={stats['rows_total']:,} windows_kept={stats['windows_written']:,} "
                        f"kept_pos={stats['rows_written_pos']:,} kept_neg={stats['rows_written_neg']:,} "
                        f"bad={stats['bad_rows']:,} filtered={stats['rows_filtered']:,}",
                        file=sys.stderr,
                        flush=True,
                    )

                parts = line.rstrip("\n").split("\t")
                if len(parts) < 12:
                    stats["bad_rows"] += 1
                    continue

                read_id = parts[0]
                chrom = parts[1]
                if chrom not in allowed_refs:
                    stats["rows_filtered"] += 1
                    stats["skip_ref_not_target"] += 1
                    continue

                try:
                    row_start = int(parts[2])
                except Exception:
                    stats["bad_rows"] += 1
                    stats["bad_start"] += 1
                    continue

                ref_seq_full = parts[3]
                if not ref_seq_full:
                    stats["bad_rows"] += 1
                    stats["empty_ref_seq"] += 1
                    continue

                bam_strand = parts[5]
                strand_match = parts[7]
                if args.require_strand_match and strand_match != "1":
                    stats["rows_filtered"] += 1
                    stats["skip_strand_mismatch"] += 1
                    continue
                if bam_strand not in ("+", "-"):
                    stats["bad_rows"] += 1
                    stats["bad_bam_strand"] += 1
                    continue

                row_end = row_start + len(ref_seq_full) - 1

                barcode_cov = overlap_len(row_start, row_end, args.barcode_start, args.barcode_end)
                if barcode_cov < args.barcode_min_cover:
                    stats["rows_filtered"] += 1
                    stats["skip_barcode_cover"] += 1
                    continue

                qual_s = parts[4]
                signal_s = parts[8]
                mismatch_s = parts[9]
                insertion_s = parts[10]
                deletion_s = parts[11]

                qual_list = qual_s.split("|")
                mismatch_list = mismatch_s.split("|")
                insertion_list = insertion_s.split("|")
                deletion_list = deletion_s.split("|")
                signal_list = signal_s.split("|")

                full_len = len(ref_seq_full)
                if not (
                    len(qual_list) == full_len and
                    len(mismatch_list) == full_len and
                    len(insertion_list) == full_len and
                    len(deletion_list) == full_len and
                    len(signal_list) == full_len
                ):
                    stats["bad_rows"] += 1
                    stats["bad_field_length_inconsistent"] += 1
                    continue

                flat_signal = np.fromstring(signal_s.replace("|", "*"), sep="*", dtype=np.float32)
                if flat_signal.size == 0:
                    stats["bad_rows"] += 1
                    stats["empty_signal"] += 1
                    continue

                center, scale = robust_mad_scale(flat_signal, use_unique=bool(args.mad_use_unique))
                if center is None or scale is None:
                    stats["bad_rows"] += 1
                    stats["bad_mmad"] += 1
                    continue

                if chrom.startswith("PC"):
                    label = 0
                elif chrom.startswith("XNA"):
                    label = 1
                else:
                    stats["bad_rows"] += 1
                    stats["bad_label_ref"] += 1
                    continue

                row_kept_this_input = 0

                for target_pos in target_pos_map[chrom]:
                    stats["candidate_windows"] += 1
                    try:
                        kmer_ref_start, kmer_ref_end = compute_kmer_ref_window(target_pos, bam_strand)
                    except Exception:
                        stats["bad_rows"] += 1
                        stats["bad_target_window"] += 1
                        continue

                    if not (row_start <= kmer_ref_start and row_end >= kmer_ref_end):
                        stats["skip_kmer_not_fully_covered"] += 1
                        continue

                    left = kmer_ref_start - row_start
                    right = kmer_ref_end - row_start + 1
                    if left < 0 or right > len(ref_seq_full) or (right - left) != 6:
                        stats["bad_rows"] += 1
                        stats["bad_window_indices"] += 1
                        continue

                    ref_seq_win = ref_seq_full[left:right]
                    qual_win = qual_list[left:right]
                    mismatch_win = mismatch_list[left:right]
                    insertion_win = insertion_list[left:right]
                    deletion_win = deletion_list[left:right]
                    signal_win = signal_list[left:right]

                    if bam_strand == "-":
                        ref_seq_win = revcomp_xy(ref_seq_win)
                        qual_win = qual_win[::-1]
                        mismatch_win = mismatch_win[::-1]
                        insertion_win = insertion_win[::-1]
                        deletion_win = deletion_win[::-1]
                        signal_win = signal_win[::-1]

                    # Important for model training:
                    # keep the kmer exactly as it appears in the uploaded FASTA.
                    # Do NOT restore X/Y markers for XNA rows, otherwise the kmer
                    # field would leak the label information into the model input.

                    means = []
                    stds = []
                    medians = []
                    dwells = []
                    raw_norm_segments = []

                    ok = True
                    for seg_str in signal_win:
                        seg = np.fromstring(seg_str, sep="*", dtype=np.float32)
                        if seg.size == 0:
                            ok = False
                            break
                        seg = (seg - center) / scale
                        means.append(fmt_num(np.round(np.mean(seg), args.stat_round), args.stat_round))
                        stds.append(fmt_num(np.round(np.std(seg), args.stat_round), args.stat_round))
                        medians.append(fmt_num(np.round(np.median(seg), args.stat_round), args.stat_round))
                        dwells.append(fmt_num(np.round(seg.size / args.dwell_scale, args.stat_round), args.stat_round))
                        raw_norm_segments.append("*".join(fmt_num(v, args.signal_round) for v in seg))

                    if not ok or len(raw_norm_segments) != 6:
                        stats["bad_rows"] += 1
                        stats["bad_signal_segment"] += 1
                        continue

                    try:
                        qual_out = [fmt_num(np.round(float(x) / args.quality_scale, args.stat_round), args.stat_round)
                                    for x in qual_win]
                        mismatch_out = [fmt_num(np.round(float(x), args.stat_round), args.stat_round) for x in mismatch_win]
                        insertion_out = [fmt_num(np.round(float(x), args.stat_round), args.stat_round) for x in insertion_win]
                        deletion_out = [fmt_num(np.round(float(x), args.stat_round), args.stat_round) for x in deletion_win]
                    except Exception:
                        stats["bad_rows"] += 1
                        stats["bad_numeric_field"] += 1
                        continue

                    row_out = [
                        ref_seq_win,
                        "|".join(means),
                        "|".join(stds),
                        "|".join(medians),
                        "|".join(dwells),
                        "|".join(qual_out),
                        "|".join(mismatch_out),
                        "|".join(insertion_out),
                        "|".join(deletion_out),
                        "|".join(raw_norm_segments),
                        label,
                    ]

                    if bam_strand == "+":
                        writer_pos.writerow(row_out)
                        stats["rows_written_pos"] += 1
                        if meta_writer_pos is not None:
                            meta_writer_pos.writerow([
                                read_id, chrom, bam_strand, row_start, row_end,
                                target_pos, kmer_ref_start, kmer_ref_end,
                                barcode_cov, ref_seq_win, label
                            ])
                    else:
                        writer_neg.writerow(row_out)
                        stats["rows_written_neg"] += 1
                        if meta_writer_neg is not None:
                            meta_writer_neg.writerow([
                                read_id, chrom, bam_strand, row_start, row_end,
                                target_pos, kmer_ref_start, kmer_ref_end,
                                barcode_cov, ref_seq_win, label
                            ])

                    row_kept_this_input += 1
                    stats["windows_written"] += 1
                    stats["rows_written"] += 1
                    stats[f"label_{label}"] += 1
                    stats[f"strand_{bam_strand}"] += 1
                    stats[f"ref_{chrom}"] += 1
                    stats[f"target_{chrom}_{target_pos}"] += 1
                    stats[f"ref_prefix_{'XNA' if chrom.startswith('XNA') else 'PC'}"] += 1

                if row_kept_this_input == 0:
                    stats["rows_filtered"] += 1
                    stats["skip_no_target_window_kept"] += 1

        finally:
            if meta_fp_pos is not None:
                meta_fp_pos.close()
            if meta_fp_neg is not None:
                meta_fp_neg.close()

    print("\n[Target positions]", file=sys.stderr)
    for chrom in sorted(target_pos_map):
        print(f"{chrom}: {','.join(map(str, target_pos_map[chrom]))}", file=sys.stderr)

    print("\n[Summary]", file=sys.stderr)
    for k in sorted(stats):
        print(f"{k}: {stats[k]}", file=sys.stderr)


if __name__ == "__main__":
    main()
