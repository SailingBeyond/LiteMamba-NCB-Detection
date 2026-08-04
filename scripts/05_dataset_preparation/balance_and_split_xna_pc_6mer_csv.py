#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Balance a feature CSV within every kmer x label group and create train/test sets.

For each retained k-mer, the script finds the smallest count among the selected
labels and samples exactly that many rows from every label. The selected rows
are then split within each kmer x label group according to --train_ratio.

Selected original labels are remapped, in the order supplied, to continuous
model labels 0..N-1.

The script can work in either of two modes:

1. Self-contained two-pass mode:
       omit --count_pivot_csv
   The first pass counts kmer x label groups, and the second pass samples rows.

2. Precomputed-count mode:
       provide --count_pivot_csv
   The supplied pivot must contain columns such as:
       kmer,label_0,label_1,...,total

The input is expected to use kmer as its first field and label as its last
field. This matches the feature files produced by the repository's extraction
scripts.
"""

from __future__ import annotations

import argparse
import csv
import random
from collections import Counter, defaultdict
from pathlib import Path
from typing import BinaryIO


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Balance kmer feature data and create exact train/test splits."
    )

    parser.add_argument("--input", type=Path, required=True, help="Input feature CSV.")
    parser.add_argument(
        "--count_pivot_csv",
        type=Path,
        default=None,
        help=(
            "Optional precomputed kmer x label count pivot. "
            "If omitted, counts are generated from the input in a first pass."
        ),
    )
    parser.add_argument(
        "--count_pivot_out",
        type=Path,
        default=None,
        help="Optional path for saving internally generated counts.",
    )

    parser.add_argument(
        "--balanced_out",
        type=Path,
        required=True,
        help="Balanced complete dataset.",
    )
    parser.add_argument("--train_out", type=Path, required=True, help="Training CSV.")
    parser.add_argument("--test_out", type=Path, required=True, help="Test CSV.")
    parser.add_argument(
        "--summary_out",
        type=Path,
        default=None,
        help="Optional detailed balance summary CSV.",
    )

    parser.add_argument(
        "--num_labels",
        type=int,
        default=6,
        help="Use original labels 0..N-1 when --labels is omitted. Default: 6.",
    )
    parser.add_argument(
        "--labels",
        default=None,
        help=(
            "Explicit original labels to retain, in remapping order. "
            "Example: --labels 0,1,2,3,5 maps original label 5 to output label 4."
        ),
    )
    parser.add_argument(
        "--train_ratio",
        type=float,
        default=0.8,
        help="Training fraction within each kmer x label group. Default: 0.8.",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=12345,
        help="Random seed for exact sequential sampling. Default: 12345.",
    )

    parser.add_argument(
        "--has_header",
        choices=("auto", "0", "1"),
        default="auto",
        help="Input header mode: auto, 0, or 1. Default: auto.",
    )

    output_header_group = parser.add_mutually_exclusive_group()
    output_header_group.add_argument(
        "--write_header",
        dest="write_header",
        action="store_true",
        help="Write a header to balanced/train/test outputs.",
    )
    output_header_group.add_argument(
        "--no_header",
        dest="write_header",
        action="store_false",
        help="Do not write output headers.",
    )
    parser.set_defaults(write_header=True)

    parser.add_argument(
        "--delimiter",
        default=",",
        help="Single-byte CSV delimiter. Default: comma.",
    )
    parser.add_argument(
        "--kmer_col",
        default="kmer",
        help="K-mer column name in a precomputed pivot. Default: kmer.",
    )
    parser.add_argument(
        "--label_prefix",
        default="label_",
        help="Label column prefix in a precomputed pivot. Default: label_.",
    )
    parser.add_argument(
        "--log_every",
        type=int,
        default=1_000_000,
        help="Report progress every N input rows. Set 0 to disable.",
    )

    args = parser.parse_args()

    if not args.input.is_file():
        parser.error(f"Input CSV does not exist: {args.input}")
    if args.count_pivot_csv is not None and not args.count_pivot_csv.is_file():
        parser.error(f"Count pivot does not exist: {args.count_pivot_csv}")
    if not 0 < args.train_ratio < 1:
        parser.error("--train_ratio must be between 0 and 1.")
    if len(args.delimiter.encode("utf-8")) != 1:
        parser.error("--delimiter must be a single-byte character.")
    if args.num_labels <= 0:
        parser.error("--num_labels must be positive.")
    if args.log_every < 0:
        parser.error("--log_every must be >= 0.")

    return args


def parse_labels(args: argparse.Namespace) -> tuple[str, ...]:
    if args.labels is not None:
        labels = tuple(item.strip() for item in args.labels.split(",") if item.strip())
        if not labels:
            raise ValueError("--labels was supplied but contains no labels.")
        if len(labels) != len(set(labels)):
            raise ValueError(f"Duplicate labels in --labels: {labels}")
        return labels

    return tuple(str(index) for index in range(args.num_labels))


def build_label_mapping(original_labels: tuple[str, ...]) -> dict[str, str]:
    return {
        original_label: str(new_label)
        for new_label, original_label in enumerate(original_labels)
    }


def ensure_parent(path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)


def strip_newline(line: bytes) -> tuple[bytes, bytes]:
    if line.endswith(b"\r\n"):
        return line[:-2], b"\r\n"
    if line.endswith(b"\n"):
        return line[:-1], b"\n"
    if line.endswith(b"\r"):
        return line[:-1], b"\r"
    return line, b""


def parse_first_last_fields(
    line: bytes,
    delimiter: bytes,
) -> tuple[str, str] | None:
    raw, _ = strip_newline(line)
    if not raw:
        return None

    first = raw.find(delimiter)
    last = raw.rfind(delimiter)

    if first <= 0 or last <= first:
        return None

    try:
        kmer = raw[:first].decode("utf-8").strip()
        label = raw[last + len(delimiter) :].decode("utf-8").strip()
    except UnicodeDecodeError:
        return None

    if not kmer or not label:
        return None

    return kmer, label


def looks_like_header(line: bytes, delimiter: bytes) -> bool:
    fields = parse_first_last_fields(line, delimiter)
    if fields is None:
        return False
    first, last = fields
    return first.lower() == "kmer" and last.lower() == "label"


def resolve_header_mode(
    input_path: Path,
    delimiter: bytes,
    mode: str,
) -> tuple[bool, bytes | None]:
    with input_path.open("rb") as handle:
        first_line = handle.readline()

    if not first_line:
        raise ValueError(f"Input CSV is empty: {input_path}")

    if mode == "1":
        return True, first_line
    if mode == "0":
        return False, None

    detected = looks_like_header(first_line, delimiter)
    return detected, first_line if detected else None


def replace_last_csv_field(
    line: bytes,
    delimiter: bytes,
    new_last_field: bytes,
) -> bytes:
    raw, newline = strip_newline(line)
    last = raw.rfind(delimiter)
    if last < 0:
        raise ValueError("Cannot find delimiter when replacing the last CSV field.")
    return raw[: last + len(delimiter)] + new_last_field + newline


def parse_int(value: object) -> int:
    if value is None:
        return 0
    text = str(value).strip()
    if not text:
        return 0
    return int(float(text))


def count_groups_from_input(
    args: argparse.Namespace,
    original_labels: tuple[str, ...],
    *,
    input_has_header: bool,
    delimiter: bytes,
) -> dict[str, dict[str, int]]:
    selected_labels = set(original_labels)
    counts: dict[str, Counter[str]] = defaultdict(Counter)
    scanned = 0
    skipped_bad = 0

    with args.input.open("rb", buffering=16 * 1024 * 1024) as handle:
        if input_has_header:
            handle.readline()

        for line in handle:
            scanned += 1
            parsed = parse_first_last_fields(line, delimiter)
            if parsed is None:
                skipped_bad += 1
                continue

            kmer, label = parsed
            if label in selected_labels:
                counts[kmer][label] += 1

            if args.log_every > 0 and scanned % args.log_every == 0:
                print(
                    f"[count] rows={scanned:,}, kmers={len(counts):,}, "
                    f"bad_rows={skipped_bad:,}",
                    flush=True,
                )

    normalized: dict[str, dict[str, int]] = {}
    for kmer, label_counts in counts.items():
        normalized[kmer] = {
            label: int(label_counts.get(label, 0))
            for label in original_labels
        }

    if not normalized:
        raise ValueError(
            "No selected kmer x label groups were found in the input CSV."
        )

    return normalized


def write_count_pivot(
    output_path: Path,
    counts: dict[str, dict[str, int]],
    original_labels: tuple[str, ...],
    *,
    kmer_col: str,
    label_prefix: str,
) -> None:
    ensure_parent(output_path)

    with output_path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.writer(handle)
        writer.writerow(
            [kmer_col]
            + [f"{label_prefix}{label}" for label in original_labels]
            + ["total"]
        )

        for kmer in sorted(counts):
            values = [counts[kmer].get(label, 0) for label in original_labels]
            writer.writerow([kmer] + values + [sum(values)])


def load_counts_from_pivot(
    args: argparse.Namespace,
    original_labels: tuple[str, ...],
) -> dict[str, dict[str, int]]:
    assert args.count_pivot_csv is not None

    label_columns = {
        label: f"{args.label_prefix}{label}"
        for label in original_labels
    }
    counts: dict[str, dict[str, int]] = {}

    with args.count_pivot_csv.open(
        "r",
        encoding="utf-8-sig",
        newline="",
    ) as handle:
        reader = csv.DictReader(handle)

        if reader.fieldnames is None:
            raise ValueError(f"Empty count pivot: {args.count_pivot_csv}")

        available = set(reader.fieldnames)
        if args.kmer_col not in available:
            raise ValueError(
                f"K-mer column '{args.kmer_col}' is missing from the count pivot. "
                f"Available columns: {reader.fieldnames}"
            )

        missing = [
            column
            for column in label_columns.values()
            if column not in available
        ]
        if missing:
            raise ValueError(
                f"Count pivot is missing label columns: {missing}. "
                f"Available columns: {reader.fieldnames}"
            )

        for row in reader:
            kmer = str(row.get(args.kmer_col, "")).strip()
            if not kmer:
                continue
            counts[kmer] = {
                label: parse_int(row.get(label_columns[label]))
                for label in original_labels
            }

    if not counts:
        raise ValueError(f"No k-mer counts were loaded from {args.count_pivot_csv}")

    return counts


def build_balance_plan(
    counts: dict[str, dict[str, int]],
    original_labels: tuple[str, ...],
    train_ratio: float,
) -> tuple[dict[str, dict[str, object]], dict[str, dict[str, int]]]:
    plan: dict[str, dict[str, object]] = {}
    skipped: dict[str, dict[str, int]] = {}

    for kmer, group_counts in counts.items():
        normalized_counts = {
            label: int(group_counts.get(label, 0))
            for label in original_labels
        }
        minimum = min(normalized_counts.values())

        if minimum <= 0:
            skipped[kmer] = normalized_counts
            continue

        train_count = int(round(minimum * train_ratio))
        train_count = max(0, min(train_count, minimum))
        test_count = minimum - train_count

        plan[kmer] = {
            "counts": normalized_counts,
            "target_each_label": minimum,
            "train_each_label": train_count,
            "test_each_label": test_count,
        }

    if not plan:
        raise ValueError(
            "No k-mer contains at least one row for every selected label."
        )

    return plan, skipped


def initialize_remaining(
    plan: dict[str, dict[str, object]],
    original_labels: tuple[str, ...],
) -> tuple[
    dict[tuple[str, str], int],
    dict[tuple[str, str], int],
    dict[tuple[str, str], int],
]:
    remaining_available: dict[tuple[str, str], int] = {}
    remaining_train: dict[tuple[str, str], int] = {}
    remaining_test: dict[tuple[str, str], int] = {}

    for kmer, info in plan.items():
        counts = info["counts"]
        assert isinstance(counts, dict)

        for label in original_labels:
            group = (kmer, label)
            remaining_available[group] = int(counts[label])
            remaining_train[group] = int(info["train_each_label"])
            remaining_test[group] = int(info["test_each_label"])

    return remaining_available, remaining_train, remaining_test


def open_outputs(
    args: argparse.Namespace,
) -> tuple[BinaryIO, BinaryIO, BinaryIO]:
    for path in (args.balanced_out, args.train_out, args.test_out):
        ensure_parent(path)

    return (
        args.balanced_out.open("wb", buffering=16 * 1024 * 1024),
        args.train_out.open("wb", buffering=16 * 1024 * 1024),
        args.test_out.open("wb", buffering=16 * 1024 * 1024),
    )


def exact_stream_sample(
    args: argparse.Namespace,
    original_labels: tuple[str, ...],
    label_map: dict[str, str],
    plan: dict[str, dict[str, object]],
    *,
    input_has_header: bool,
    input_header: bytes | None,
    delimiter: bytes,
) -> dict[str, object]:
    """
    Sample exactly the planned train and test sizes in one sequential pass.

    For the current row in a group:
        A = remaining available rows
        T = remaining rows required for train
        E = remaining rows required for test

    The row is assigned to train with probability T/A, to test with probability
    E/A, and otherwise skipped. This gives exact final counts when the supplied
    group counts match the input.
    """
    rng = random.Random(args.seed)
    selected_labels = set(original_labels)
    target_kmers = set(plan)

    (
        remaining_available,
        remaining_train,
        remaining_test,
    ) = initialize_remaining(plan, original_labels)

    stats: Counter[str] = Counter()
    selected_train: Counter[tuple[str, str]] = Counter()
    selected_test: Counter[tuple[str, str]] = Counter()
    selected_total: Counter[tuple[str, str]] = Counter()
    output_label_balanced: Counter[str] = Counter()
    output_label_train: Counter[str] = Counter()
    output_label_test: Counter[str] = Counter()

    balanced_handle, train_handle, test_handle = open_outputs(args)

    try:
        if args.write_header:
            header = input_header
            if header is None:
                header = (
                    b"kmer,mean,std,median,dwell,quality,mismatch,"
                    b"insertion,deletion,signal,label\n"
                )
            balanced_handle.write(header)
            train_handle.write(header)
            test_handle.write(header)

        with args.input.open("rb", buffering=16 * 1024 * 1024) as input_handle:
            if input_has_header:
                input_handle.readline()

            for line in input_handle:
                stats["data_rows_scanned"] += 1
                parsed = parse_first_last_fields(line, delimiter)

                if parsed is None:
                    stats["skipped_bad_format"] += 1
                    continue

                kmer, old_label = parsed

                if old_label not in selected_labels:
                    stats["skipped_label_outside_target"] += 1
                    continue
                if kmer not in target_kmers:
                    stats["skipped_kmer_not_in_plan"] += 1
                    continue

                group = (kmer, old_label)
                available = remaining_available.get(group, 0)
                need_train = remaining_train.get(group, 0)
                need_test = remaining_test.get(group, 0)

                if available <= 0:
                    stats["extra_rows_beyond_count_plan"] += 1
                    continue

                total_needed = need_train + need_test

                if total_needed <= 0:
                    remaining_available[group] = available - 1
                    stats["skipped_group_already_full"] += 1
                    continue

                draw = rng.random()
                train_probability = need_train / available
                selected_probability = total_needed / available

                new_label = label_map[old_label]
                output_line = replace_last_csv_field(
                    line,
                    delimiter,
                    new_label.encode("utf-8"),
                )

                if draw < train_probability:
                    balanced_handle.write(output_line)
                    train_handle.write(output_line)

                    remaining_train[group] = need_train - 1
                    selected_train[group] += 1
                    selected_total[group] += 1
                    output_label_balanced[new_label] += 1
                    output_label_train[new_label] += 1
                    stats["written_balanced"] += 1
                    stats["written_train"] += 1

                elif draw < selected_probability:
                    balanced_handle.write(output_line)
                    test_handle.write(output_line)

                    remaining_test[group] = need_test - 1
                    selected_test[group] += 1
                    selected_total[group] += 1
                    output_label_balanced[new_label] += 1
                    output_label_test[new_label] += 1
                    stats["written_balanced"] += 1
                    stats["written_test"] += 1

                else:
                    stats["skipped_by_sampling"] += 1

                remaining_available[group] = available - 1

                if (
                    args.log_every > 0
                    and stats["data_rows_scanned"] % args.log_every == 0
                ):
                    remaining_needed = (
                        sum(remaining_train.values())
                        + sum(remaining_test.values())
                    )
                    print(
                        f"[sample] rows={stats['data_rows_scanned']:,}, "
                        f"balanced={stats['written_balanced']:,}, "
                        f"train={stats['written_train']:,}, "
                        f"test={stats['written_test']:,}, "
                        f"remaining_need={remaining_needed:,}",
                        flush=True,
                    )
    finally:
        balanced_handle.close()
        train_handle.close()
        test_handle.close()

    return {
        "stats": stats,
        "remaining_available": remaining_available,
        "remaining_train": remaining_train,
        "remaining_test": remaining_test,
        "selected_train": selected_train,
        "selected_test": selected_test,
        "selected_total": selected_total,
        "output_label_balanced": output_label_balanced,
        "output_label_train": output_label_train,
        "output_label_test": output_label_test,
    }


def write_summary(
    args: argparse.Namespace,
    original_labels: tuple[str, ...],
    label_map: dict[str, str],
    plan: dict[str, dict[str, object]],
    skipped: dict[str, dict[str, int]],
    result: dict[str, object],
) -> None:
    if args.summary_out is None:
        return

    ensure_parent(args.summary_out)

    selected_train = result["selected_train"]
    selected_test = result["selected_test"]
    selected_total = result["selected_total"]
    remaining_available = result["remaining_available"]
    remaining_train = result["remaining_train"]
    remaining_test = result["remaining_test"]

    assert isinstance(selected_train, Counter)
    assert isinstance(selected_test, Counter)
    assert isinstance(selected_total, Counter)
    assert isinstance(remaining_available, dict)
    assert isinstance(remaining_train, dict)
    assert isinstance(remaining_test, dict)

    with args.summary_out.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.writer(handle)

        writer.writerow(
            ["kmer", "status"]
            + [f"count_original_label_{label}" for label in original_labels]
            + [
                "target_each_label",
                "train_each_label",
                "test_each_label",
            ]
            + [
                f"selected_total_original_label_{label}"
                for label in original_labels
            ]
            + [
                f"selected_train_original_label_{label}"
                for label in original_labels
            ]
            + [
                f"selected_test_original_label_{label}"
                for label in original_labels
            ]
            + [
                f"remaining_available_original_label_{label}"
                for label in original_labels
            ]
            + [
                f"remaining_train_need_original_label_{label}"
                for label in original_labels
            ]
            + [
                f"remaining_test_need_original_label_{label}"
                for label in original_labels
            ]
        )

        for kmer in sorted(plan):
            info = plan[kmer]
            counts = info["counts"]
            assert isinstance(counts, dict)

            total_values = [
                selected_total.get((kmer, label), 0)
                for label in original_labels
            ]
            train_values = [
                selected_train.get((kmer, label), 0)
                for label in original_labels
            ]
            test_values = [
                selected_test.get((kmer, label), 0)
                for label in original_labels
            ]

            expected_total = int(info["target_each_label"])
            expected_train = int(info["train_each_label"])
            expected_test = int(info["test_each_label"])

            ok = (
                all(value == expected_total for value in total_values)
                and all(value == expected_train for value in train_values)
                and all(value == expected_test for value in test_values)
            )

            writer.writerow(
                [kmer, "included_ok" if ok else "included_mismatch"]
                + [counts[label] for label in original_labels]
                + [expected_total, expected_train, expected_test]
                + total_values
                + train_values
                + test_values
                + [
                    remaining_available.get((kmer, label), 0)
                    for label in original_labels
                ]
                + [
                    remaining_train.get((kmer, label), 0)
                    for label in original_labels
                ]
                + [
                    remaining_test.get((kmer, label), 0)
                    for label in original_labels
                ]
            )

        for kmer in sorted(skipped):
            writer.writerow(
                [kmer, "skipped_missing_label"]
                + [skipped[kmer].get(label, 0) for label in original_labels]
                + ["", "", ""]
                + [""] * (6 * len(original_labels))
            )

        writer.writerow([])
        writer.writerow(["LABEL_MAPPING"])
        writer.writerow(["original_label", "output_label"])
        for original_label in original_labels:
            writer.writerow([original_label, label_map[original_label]])


def print_plan_summary(
    original_labels: tuple[str, ...],
    label_map: dict[str, str],
    plan: dict[str, dict[str, object]],
    skipped: dict[str, dict[str, int]],
) -> None:
    print("\n========== Balance Plan ==========")
    print(f"Original labels: {','.join(original_labels)}")
    print(
        "Output labels:   "
        + ",".join(label_map[label] for label in original_labels)
    )
    print(f"Included k-mers: {len(plan):,}")
    print(f"Skipped k-mers:  {len(skipped):,}")

    for label in original_labels:
        original_count = 0
        balanced_count = 0
        train_count = 0
        test_count = 0

        for info in plan.values():
            counts = info["counts"]
            assert isinstance(counts, dict)
            original_count += int(counts[label])
            balanced_count += int(info["target_each_label"])
            train_count += int(info["train_each_label"])
            test_count += int(info["test_each_label"])

        print(
            f"original {label} -> output {label_map[label]}: "
            f"original={original_count:,}, "
            f"balanced={balanced_count:,}, "
            f"train={train_count:,}, "
            f"test={test_count:,}"
        )

    print("==================================\n")


def print_final_check(
    original_labels: tuple[str, ...],
    label_map: dict[str, str],
    plan: dict[str, dict[str, object]],
    result: dict[str, object],
) -> int:
    stats = result["stats"]
    selected_train = result["selected_train"]
    selected_test = result["selected_test"]
    selected_total = result["selected_total"]
    remaining_train = result["remaining_train"]
    remaining_test = result["remaining_test"]
    output_balanced = result["output_label_balanced"]
    output_train = result["output_label_train"]
    output_test = result["output_label_test"]

    assert isinstance(stats, Counter)
    assert isinstance(selected_train, Counter)
    assert isinstance(selected_test, Counter)
    assert isinstance(selected_total, Counter)
    assert isinstance(remaining_train, dict)
    assert isinstance(remaining_test, dict)
    assert isinstance(output_balanced, Counter)
    assert isinstance(output_train, Counter)
    assert isinstance(output_test, Counter)

    mismatches = 0

    for kmer, info in plan.items():
        expected_total = int(info["target_each_label"])
        expected_train = int(info["train_each_label"])
        expected_test = int(info["test_each_label"])

        for label in original_labels:
            group = (kmer, label)
            if (
                selected_total.get(group, 0) != expected_total
                or selected_train.get(group, 0) != expected_train
                or selected_test.get(group, 0) != expected_test
            ):
                mismatches += 1

    print("\n========== Final Check ==========")
    print(f"Rows scanned:     {stats['data_rows_scanned']:,}")
    print(f"Balanced written: {stats['written_balanced']:,}")
    print(f"Train written:    {stats['written_train']:,}")
    print(f"Test written:     {stats['written_test']:,}")
    print(f"Mismatch groups:  {mismatches:,}")
    print(f"Remaining train need: {sum(remaining_train.values()):,}")
    print(f"Remaining test need:  {sum(remaining_test.values()):,}")

    print("\nOutput label distribution:")
    for original_label in original_labels:
        new_label = label_map[original_label]
        print(
            f"output {new_label} (from original {original_label}): "
            f"balanced={output_balanced[new_label]:,}, "
            f"train={output_train[new_label]:,}, "
            f"test={output_test[new_label]:,}"
        )

    if stats["extra_rows_beyond_count_plan"] > 0:
        print(
            "\nWARNING: input contains more rows than the supplied count plan for "
            f"{stats['extra_rows_beyond_count_plan']:,} rows."
        )

    print("=================================\n")
    return mismatches


def main() -> None:
    args = parse_args()
    original_labels = parse_labels(args)
    label_map = build_label_mapping(original_labels)
    delimiter = args.delimiter.encode("utf-8")

    input_has_header, input_header = resolve_header_mode(
        args.input,
        delimiter,
        args.has_header,
    )

    print("Selected label mapping:")
    for original_label in original_labels:
        print(f"  {original_label} -> {label_map[original_label]}")

    if args.count_pivot_csv is None:
        print("Counting kmer x label groups from the input CSV...")
        counts = count_groups_from_input(
            args,
            original_labels,
            input_has_header=input_has_header,
            delimiter=delimiter,
        )

        if args.count_pivot_out is not None:
            write_count_pivot(
                args.count_pivot_out,
                counts,
                original_labels,
                kmer_col=args.kmer_col,
                label_prefix=args.label_prefix,
            )
            print(f"Generated count pivot: {args.count_pivot_out}")
    else:
        print(f"Loading count pivot: {args.count_pivot_csv}")
        counts = load_counts_from_pivot(args, original_labels)

    plan, skipped = build_balance_plan(
        counts,
        original_labels,
        args.train_ratio,
    )
    print_plan_summary(original_labels, label_map, plan, skipped)

    result = exact_stream_sample(
        args,
        original_labels,
        label_map,
        plan,
        input_has_header=input_has_header,
        input_header=input_header,
        delimiter=delimiter,
    )

    mismatches = print_final_check(
        original_labels,
        label_map,
        plan,
        result,
    )
    write_summary(
        args,
        original_labels,
        label_map,
        plan,
        skipped,
        result,
    )

    print(f"Balanced dataset: {args.balanced_out}")
    print(f"Training dataset: {args.train_out}")
    print(f"Test dataset:     {args.test_out}")
    if args.summary_out is not None:
        print(f"Summary:          {args.summary_out}")

    if mismatches:
        raise SystemExit(
            "Sampling did not reach the planned counts. "
            "Check whether the input and count pivot describe the same file."
        )


if __name__ == "__main__":
    main()
