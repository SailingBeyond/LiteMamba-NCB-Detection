#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Merge text-table batches while retaining exactly one header line."""

from __future__ import annotations

import argparse
import glob
import re
from pathlib import Path
from typing import Iterable


def natural_key(path: Path) -> list[object]:
    return [int(part) if part.isdigit() else part.lower() for part in re.split(r"(\d+)", path.name)]


def resolve_inputs(patterns: Iterable[str], output: Path) -> list[Path]:
    found: dict[Path, None] = {}
    output_abs = output.resolve()
    for pattern in patterns:
        matches = [Path(item) for item in glob.glob(pattern)]
        if not matches:
            candidate = Path(pattern)
            if candidate.is_file():
                matches = [candidate]
        for path in matches:
            if path.is_file() and path.resolve() != output_abs:
                found[path.resolve()] = None
    return sorted(found, key=natural_key)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Merge CSV/TSV batch files with one validated header.")
    parser.add_argument("--inputs", nargs="+", required=True, help="Input paths or glob patterns.")
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument(
        "--header_mode",
        choices=("one", "none"),
        default="one",
        help="one: validate and keep one header; none: concatenate all lines as data.",
    )
    parser.add_argument("--allow_empty_files", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    inputs = resolve_inputs(args.inputs, args.output)
    if not inputs:
        raise FileNotFoundError("No batch files matched the supplied inputs.")

    args.output.parent.mkdir(parents=True, exist_ok=True)
    expected_header: str | None = None
    total_data_lines = 0
    used_files = 0

    with args.output.open("w", encoding="utf-8", newline="") as out_handle:
        for input_path in inputs:
            with input_path.open("r", encoding="utf-8-sig", newline="") as in_handle:
                first_line = in_handle.readline()
                if first_line == "":
                    if args.allow_empty_files:
                        continue
                    raise ValueError(f"Empty batch file: {input_path}")

                if args.header_mode == "one":
                    header = first_line.rstrip("\r\n")
                    if expected_header is None:
                        expected_header = header
                        out_handle.write(header + "\n")
                    elif header != expected_header:
                        raise ValueError(
                            f"Header mismatch in {input_path}\n"
                            f"Expected: {expected_header}\n"
                            f"Observed: {header}"
                        )
                else:
                    out_handle.write(first_line)
                    total_data_lines += 1

                for line in in_handle:
                    out_handle.write(line)
                    total_data_lines += 1
                used_files += 1

    print(f"Merged files     : {used_files}")
    print(f"Data lines       : {total_data_lines}")
    print(f"Output           : {args.output}")


if __name__ == "__main__":
    main()
