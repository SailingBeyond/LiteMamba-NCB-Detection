#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Extract strand-aware 6-mer features from an upstream nanopore signal TSV.

The input TSV is expected to contain at least these 12 tab-separated columns:
    read_id, chrom, row_start, ref_seq, quality, bam_strand,
    tombo_strand, strand_match, signal, mismatch, insertion, deletion

The output CSV contains the 11 columns consumed by the training pipeline:
    kmer, mean, std, median, dwell, quality,
    mismatch, insertion, deletion, signal, label

Default label scheme:
    PC reference, either strand -> 0
    XNA reference, positive strand -> 1
    XNA reference, negative strand -> 2

The output 6-mer is always oriented 5'->3', and the target site is placed at
the fourth position. By default, the output includes a header.
"""

from __future__ import annotations

import argparse
import csv
import sys
from collections import Counter
from pathlib import Path

import numpy as np


OUTPUT_COLUMNS = [
    "kmer",
    "mean",
    "std",
    "median",
    "dwell",
    "quality",
    "mismatch",
    "insertion",
    "deletion",
    "signal",
    "label",
]

COMP_TABLE_XY = str.maketrans(
    {
        "A": "T",
        "C": "G",
        "G": "C",
        "T": "A",
        "N": "N",
        "X": "Y",
        "Y": "X",
        "a": "t",
        "c": "g",
        "g": "c",
        "t": "a",
        "n": "n",
        "x": "y",
        "y": "x",
    }
)


def revcomp_xy(seq: str) -> str:
    """Return reverse complement while treating X and Y as complements."""
    return seq.translate(COMP_TABLE_XY)[::-1]


def overlap_len(start_a: int, end_a: int, start_b: int, end_b: int) -> int:
    """Return overlap length for two 1-based inclusive intervals."""
    left = max(start_a, start_b)
    right = min(end_a, end_b)
    return max(0, right - left + 1)


def robust_mad_scale(
    arr: np.ndarray,
    *,
    use_unique: bool = True,
    eps: float = 1e-8,
) -> tuple[float | None, float | None]:
    """Return median center and robust MAD scale."""
    if arr.size == 0 or not np.all(np.isfinite(arr)):
        return None, None

    base = np.unique(arr) if use_unique else arr
    if base.size == 0:
        return None, None

    center = float(np.median(base))
    scale = float(np.median(np.abs(base - center)) * 1.4826)

    if not np.isfinite(scale) or scale < eps:
        std = float(np.std(base))
        scale = std if np.isfinite(std) and std >= eps else 1.0

    return center, scale


def fmt_num(value: float, ndigits: int = 6) -> str:
    """Format a number compactly while retaining reproducible precision."""
    text = f"{float(value):.{ndigits}f}".rstrip("0").rstrip(".")
    return text if text else "0"


def build_allowed_refs(max_idx: int) -> set[str]:
    """Build XNA01..XNAxx and PC01..PCxx reference names."""
    refs: set[str] = set()
    for idx in range(1, max_idx + 1):
        refs.add(f"XNA{idx:02d}")
        refs.add(f"PC{idx:02d}")
    return refs


def compute_kmer_ref_window(target_pos: int, strand: str) -> tuple[int, int]:
    """
    Return the reference-coordinate 6-mer window, 1-based and inclusive.

    Positive strand:
        [target - 3, target + 2]

    Negative strand:
        [target - 2, target + 3], followed by reverse complementation.

    In both cases, the target site is the fourth base of the final 5'->3' 6-mer.
    """
    if strand == "+":
        return target_pos - 3, target_pos + 2
    if strand == "-":
        return target_pos - 2, target_pos + 3
    raise ValueError(f"Unsupported strand: {strand}")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Extract strand-aware 6-mer nanopore features."
    )
    parser.add_argument("--input", type=Path, required=True, help="Input TSV.")
    parser.add_argument("--output", type=Path, required=True, help="Output feature CSV.")
    parser.add_argument(
        "--meta_out",
        type=Path,
        default=None,
        help="Optional metadata TSV for read-level traceability.",
    )

    parser.add_argument(
        "--target_pos",
        type=int,
        default=1291,
        help="Target reference position, 1-based. Default: 1291.",
    )
    parser.add_argument(
        "--barcode_start",
        type=int,
        default=1240,
        help="Barcode interval start, 1-based. Default: 1240.",
    )
    parser.add_argument(
        "--barcode_end",
        type=int,
        default=1263,
        help="Barcode interval end, 1-based. Default: 1263.",
    )
    parser.add_argument(
        "--barcode_min_cover",
        type=int,
        default=15,
        help="Minimum barcode overlap length. Default: 15.",
    )
    parser.add_argument(
        "--max_ref_index",
        type=int,
        default=12,
        help=(
            "Include XNA01..XNAxx and PC01..PCxx. Default: 12. "
            "Keep 12 for the training templates used in the held-out-template design; "
            "set 16 only when inclusion of templates 13-16 is intentional."
        ),
    )

    parser.add_argument(
        "--dwell_scale",
        type=float,
        default=1.0,
        help="Write dwell as segment_length / dwell_scale. Default: 1.0.",
    )
    parser.add_argument(
        "--quality_scale",
        type=float,
        default=1.0,
        help="Write quality as Q / quality_scale. Default: 1.0.",
    )
    parser.add_argument(
        "--mad_use_unique",
        type=int,
        choices=(0, 1),
        default=1,
        help="Use unique full-read signal values for median-MAD scaling. Default: 1.",
    )
    parser.add_argument(
        "--require_strand_match",
        type=int,
        choices=(0, 1),
        default=1,
        help="Require the upstream strand_match field to equal 1. Default: 1.",
    )

    header_group = parser.add_mutually_exclusive_group()
    header_group.add_argument(
        "--with_header",
        dest="write_header",
        action="store_true",
        help="Write the output CSV header.",
    )
    header_group.add_argument(
        "--no_header",
        dest="write_header",
        action="store_false",
        help="Do not write the output CSV header.",
    )
    parser.set_defaults(write_header=True)

    parser.add_argument(
        "--signal_round",
        type=int,
        default=6,
        help="Decimal places for normalized raw signal. Default: 6.",
    )
    parser.add_argument(
        "--stat_round",
        type=int,
        default=6,
        help="Decimal places for summary statistics. Default: 6.",
    )
    parser.add_argument(
        "--log_every",
        type=int,
        default=100_000,
        help="Report progress every N input rows. Set 0 to disable.",
    )

    args = parser.parse_args()

    if not args.input.is_file():
        parser.error(f"Input TSV does not exist: {args.input}")
    if args.target_pos <= 0:
        parser.error("--target_pos must be positive.")
    if args.barcode_start <= 0 or args.barcode_end < args.barcode_start:
        parser.error("Invalid barcode interval.")
    if args.barcode_min_cover < 0:
        parser.error("--barcode_min_cover must be >= 0.")
    if args.max_ref_index <= 0:
        parser.error("--max_ref_index must be positive.")
    if args.dwell_scale <= 0:
        parser.error("--dwell_scale must be > 0.")
    if args.quality_scale <= 0:
        parser.error("--quality_scale must be > 0.")
    if args.signal_round < 0 or args.stat_round < 0:
        parser.error("Rounding precision must be >= 0.")

    return args


def parse_numeric_vector(
    text: str,
    expected_len: int,
) -> np.ndarray | None:
    arr = np.fromstring(text, sep="|", dtype=np.float32)
    if arr.size != expected_len or not np.all(np.isfinite(arr)):
        return None
    return arr


def main() -> None:
    args = parse_args()
    allowed_refs = build_allowed_refs(args.max_ref_index)
    stats: Counter[str] = Counter()

    args.output.parent.mkdir(parents=True, exist_ok=True)
    if args.meta_out is not None:
        args.meta_out.parent.mkdir(parents=True, exist_ok=True)

    meta_fp = None

    with args.input.open(
        "r",
        encoding="utf-8",
        buffering=16 * 1024 * 1024,
    ) as fin, args.output.open(
        "w",
        newline="",
        encoding="utf-8",
        buffering=16 * 1024 * 1024,
    ) as fout:
        writer = csv.writer(fout)

        try:
            meta_writer = None
            if args.meta_out is not None:
                meta_fp = args.meta_out.open(
                    "w",
                    newline="",
                    encoding="utf-8",
                    buffering=8 * 1024 * 1024,
                )
                meta_writer = csv.writer(meta_fp, delimiter="\t")
                meta_writer.writerow(
                    [
                        "read_id",
                        "chrom",
                        "bam_strand",
                        "row_start",
                        "row_end",
                        "kmer_ref_start",
                        "kmer_ref_end",
                        "barcode_overlap",
                        "kmer",
                        "label",
                    ]
                )

            if args.write_header:
                writer.writerow(OUTPUT_COLUMNS)

            for line_no, line in enumerate(fin, start=1):
                if line_no == 1 and line.startswith("read_id\t"):
                    stats["header_skipped"] += 1
                    continue

                stats["rows_total"] += 1

                if args.log_every > 0 and stats["rows_total"] % args.log_every == 0:
                    print(
                        f"[progress] rows={stats['rows_total']:,} "
                        f"kept={stats['rows_written']:,} "
                        f"bad={stats['bad_rows']:,} "
                        f"filtered={stats['rows_filtered']:,}",
                        file=sys.stderr,
                        flush=True,
                    )

                parts = line.rstrip("\r\n").split("\t")
                if len(parts) < 12:
                    stats["bad_rows"] += 1
                    stats["bad_tsv_field_count"] += 1
                    continue

                read_id = parts[0]
                chrom = parts[1]
                if chrom not in allowed_refs:
                    stats["rows_filtered"] += 1
                    stats["skip_ref_not_target"] += 1
                    continue

                try:
                    row_start = int(parts[2])
                except ValueError:
                    stats["bad_rows"] += 1
                    stats["bad_start"] += 1
                    continue

                ref_seq_full = parts[3]
                if not ref_seq_full:
                    stats["bad_rows"] += 1
                    stats["empty_ref_seq"] += 1
                    continue

                quality_text = parts[4]
                bam_strand = parts[5]
                strand_match = parts[7]
                signal_text = parts[8]
                mismatch_text = parts[9]
                insertion_text = parts[10]
                deletion_text = parts[11]

                if args.require_strand_match and strand_match != "1":
                    stats["rows_filtered"] += 1
                    stats["skip_strand_mismatch"] += 1
                    continue
                if bam_strand not in {"+", "-"}:
                    stats["bad_rows"] += 1
                    stats["bad_bam_strand"] += 1
                    continue

                row_end = row_start + len(ref_seq_full) - 1
                barcode_overlap = overlap_len(
                    row_start,
                    row_end,
                    args.barcode_start,
                    args.barcode_end,
                )
                if barcode_overlap < args.barcode_min_cover:
                    stats["rows_filtered"] += 1
                    stats["skip_barcode_cover"] += 1
                    continue

                kmer_ref_start, kmer_ref_end = compute_kmer_ref_window(
                    args.target_pos,
                    bam_strand,
                )

                if not (
                    row_start <= kmer_ref_start
                    and row_end >= kmer_ref_end
                ):
                    stats["rows_filtered"] += 1
                    stats["skip_kmer_not_fully_covered"] += 1
                    continue

                left = kmer_ref_start - row_start
                right = kmer_ref_end - row_start + 1

                if left < 0 or right > len(ref_seq_full) or right - left != 6:
                    stats["bad_rows"] += 1
                    stats["bad_window_indices"] += 1
                    continue

                ref_seq_win = ref_seq_full[left:right]
                quality_list = quality_text.split("|")
                mismatch_list = mismatch_text.split("|")
                insertion_list = insertion_text.split("|")
                deletion_list = deletion_text.split("|")
                signal_list = signal_text.split("|")

                full_len = len(ref_seq_full)
                if not all(
                    len(values) == full_len
                    for values in (
                        quality_list,
                        mismatch_list,
                        insertion_list,
                        deletion_list,
                        signal_list,
                    )
                ):
                    stats["bad_rows"] += 1
                    stats["bad_field_length_inconsistent"] += 1
                    continue

                quality_win = quality_list[left:right]
                mismatch_win = mismatch_list[left:right]
                insertion_win = insertion_list[left:right]
                deletion_win = deletion_list[left:right]
                signal_win = signal_list[left:right]

                if bam_strand == "-":
                    ref_seq_win = revcomp_xy(ref_seq_win)
                    quality_win.reverse()
                    mismatch_win.reverse()
                    insertion_win.reverse()
                    deletion_win.reverse()
                    signal_win.reverse()

                flat_signal = np.fromstring(
                    signal_text.replace("|", "*"),
                    sep="*",
                    dtype=np.float32,
                )
                center, scale = robust_mad_scale(
                    flat_signal,
                    use_unique=bool(args.mad_use_unique),
                )
                if center is None or scale is None:
                    stats["bad_rows"] += 1
                    stats["bad_mad_scale"] += 1
                    continue

                means: list[str] = []
                stds: list[str] = []
                medians: list[str] = []
                dwells: list[str] = []
                normalized_segments: list[str] = []

                signal_ok = True
                for segment_text in signal_win:
                    segment = np.fromstring(
                        segment_text,
                        sep="*",
                        dtype=np.float32,
                    )
                    if segment.size == 0 or not np.all(np.isfinite(segment)):
                        signal_ok = False
                        break

                    segment = (segment - center) / scale
                    means.append(fmt_num(np.mean(segment), args.stat_round))
                    stds.append(fmt_num(np.std(segment), args.stat_round))
                    medians.append(fmt_num(np.median(segment), args.stat_round))
                    dwells.append(
                        fmt_num(segment.size / args.dwell_scale, args.stat_round)
                    )
                    normalized_segments.append(
                        "*".join(
                            fmt_num(value, args.signal_round)
                            for value in segment
                        )
                    )

                if not signal_ok or len(normalized_segments) != 6:
                    stats["bad_rows"] += 1
                    stats["bad_signal_segment"] += 1
                    continue

                try:
                    quality_out = [
                        fmt_num(float(value) / args.quality_scale, args.stat_round)
                        for value in quality_win
                    ]
                    mismatch_out = [
                        fmt_num(float(value), args.stat_round)
                        for value in mismatch_win
                    ]
                    insertion_out = [
                        fmt_num(float(value), args.stat_round)
                        for value in insertion_win
                    ]
                    deletion_out = [
                        fmt_num(float(value), args.stat_round)
                        for value in deletion_win
                    ]
                except ValueError:
                    stats["bad_rows"] += 1
                    stats["bad_numeric_field"] += 1
                    continue

                if chrom.startswith("PC"):
                    label = 0
                elif chrom.startswith("XNA"):
                    label = 1 if bam_strand == "+" else 2
                else:
                    stats["bad_rows"] += 1
                    stats["bad_label_reference"] += 1
                    continue

                writer.writerow(
                    [
                        ref_seq_win.upper(),
                        "|".join(means),
                        "|".join(stds),
                        "|".join(medians),
                        "|".join(dwells),
                        "|".join(quality_out),
                        "|".join(mismatch_out),
                        "|".join(insertion_out),
                        "|".join(deletion_out),
                        "|".join(normalized_segments),
                        label,
                    ]
                )

                stats["rows_written"] += 1
                stats[f"label_{label}"] += 1
                stats[f"strand_{bam_strand}"] += 1

                if meta_writer is not None:
                    meta_writer.writerow(
                        [
                            read_id,
                            chrom,
                            bam_strand,
                            row_start,
                            row_end,
                            kmer_ref_start,
                            kmer_ref_end,
                            barcode_overlap,
                            ref_seq_win.upper(),
                            label,
                        ]
                    )
        finally:
            if meta_fp is not None:
                meta_fp.close()

    print("\n[Summary]", file=sys.stderr)
    for key in sorted(stats):
        print(f"{key}: {stats[key]}", file=sys.stderr)


if __name__ == "__main__":
    main()
