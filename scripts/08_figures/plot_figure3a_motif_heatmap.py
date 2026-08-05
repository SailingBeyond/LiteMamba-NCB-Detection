#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""Plot Figure 3A: a 16 x 64 per-motif accuracy heatmap.

The input table must contain at least two columns:
    kmer, accuracy

Only motifs matching XXXNYY are used. The three bases to the left of N
form the 64 columns, and the two bases to the right of N form the 16 rows.
"""

from __future__ import annotations

import argparse
from itertools import product
from pathlib import Path
from typing import Sequence

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

BASES = ["A", "C", "G", "T"]


def configure_style() -> None:
    plt.rcParams.update({
        "font.family": "sans-serif",
        "font.sans-serif": ["Arial", "Liberation Sans", "DejaVu Sans"],
        "font.size": 26,
        "axes.labelsize": 34,
        "axes.titlesize": 40,
        "xtick.labelsize": 22,
        "ytick.labelsize": 28,
        "axes.linewidth": 1.8,
        "xtick.major.width": 1.6,
        "ytick.major.width": 1.6,
        "xtick.major.size": 7,
        "ytick.major.size": 7,
        "pdf.fonttype": 42,
        "ps.fonttype": 42,
        "svg.fonttype": "none",
        "axes.unicode_minus": False,
    })


def normalize_kmer(value: object) -> str:
    text = str(value).strip().upper()
    if text.startswith("KMER_"):
        text = text[5:]
    return text


def parse_motif(kmer: str):
    kmer = normalize_kmer(kmer)
    if len(kmer) != 6 or kmer[3] != "N":
        return None
    left3, right2 = kmer[:3], kmer[4:]
    if any(base not in BASES for base in left3 + right2):
        return None
    return left3, right2, kmer


def load_accuracy_table(path: Path) -> pd.DataFrame:
    if not path.exists():
        raise FileNotFoundError(f"Input CSV not found: {path}")
    df = pd.read_csv(path)
    missing = {"kmer", "accuracy"} - set(df.columns)
    if missing:
        raise ValueError(
            f"Input CSV is missing columns {sorted(missing)}. "
            f"Available columns: {list(df.columns)}"
        )
    out = df[["kmer", "accuracy"]].copy()
    out["kmer"] = out["kmer"].map(normalize_kmer)
    out["accuracy"] = pd.to_numeric(out["accuracy"], errors="coerce")
    return out.dropna(subset=["kmer", "accuracy"])


def build_matrix(df: pd.DataFrame):
    col_labels = ["".join(x) for x in product(BASES, repeat=3)]
    row_labels = ["".join(x) for x in product(BASES, repeat=2)]
    col_index = {value: index for index, value in enumerate(col_labels)}
    row_index = {value: index for index, value in enumerate(row_labels)}
    matrix = np.full((16, 64), np.nan, dtype=float)

    used = skipped = duplicates = 0
    seen: set[str] = set()
    for _, row in df.iterrows():
        parsed = parse_motif(row["kmer"])
        if parsed is None:
            skipped += 1
            continue
        left3, right2, motif = parsed
        if motif in seen:
            duplicates += 1
        seen.add(motif)
        matrix[row_index[right2], col_index[left3]] = float(row["accuracy"])
        used += 1

    matrix_df = pd.DataFrame(matrix, index=row_labels, columns=col_labels)
    return matrix, matrix_df, row_labels, col_labels, used, skipped, duplicates


def save_figure(fig: plt.Figure, output_stem: Path, formats: Sequence[str], dpi: int) -> None:
    output_stem.parent.mkdir(parents=True, exist_ok=True)
    for fmt in formats:
        kwargs = {"bbox_inches": "tight"}
        if fmt == "png":
            kwargs["dpi"] = dpi
        fig.savefig(output_stem.with_suffix(f".{fmt}"), **kwargs)
    plt.close(fig)


def plot_heatmap(
    matrix: np.ndarray,
    row_labels: list[str],
    col_labels: list[str],
    output_stem: Path,
    formats: Sequence[str],
    dpi: int,
    vmin: float,
    vmax: float,
    figure_width: float,
    figure_height: float,
    show_cell_text: bool,
) -> None:
    cmap = plt.cm.viridis.copy()
    cmap.set_bad(color="#D9D9D9")
    fig, ax = plt.subplots(figsize=(figure_width, figure_height))
    image = ax.imshow(
        matrix,
        aspect="auto",
        cmap=cmap,
        vmin=vmin,
        vmax=vmax,
        interpolation="nearest",
    )

    ax.set_xticks(np.arange(len(col_labels)))
    ax.set_yticks(np.arange(len(row_labels)))
    ax.set_xticklabels(col_labels, rotation=45, ha="right", rotation_mode="anchor")
    ax.set_yticklabels(row_labels)
    ax.set_xlabel("Three bases on the left side of N", labelpad=24)
    ax.set_ylabel("Two bases on the right side of N", labelpad=24)

    ax.set_xticks(np.arange(-0.5, len(col_labels), 1), minor=True)
    ax.set_yticks(np.arange(-0.5, len(row_labels), 1), minor=True)
    ax.grid(which="minor", color="white", linestyle="-", linewidth=0.45)
    ax.tick_params(which="minor", bottom=False, left=False)
    ax.tick_params(axis="x", pad=8)
    ax.tick_params(axis="y", pad=8)

    if show_cell_text:
        for i in range(matrix.shape[0]):
            for j in range(matrix.shape[1]):
                value = matrix[i, j]
                if np.isfinite(value):
                    ax.text(j, i, f"{value:.2f}", ha="center", va="center", fontsize=8)

    colorbar = fig.colorbar(image, ax=ax, fraction=0.028, pad=0.02)
    colorbar.set_label("Accuracy", labelpad=18)
    colorbar.ax.tick_params(width=1.6, length=7)
    colorbar.set_ticks(np.linspace(vmin, vmax, 5))

    fig.subplots_adjust(left=0.07, right=0.965, top=0.96, bottom=0.23)
    save_figure(fig, output_stem, formats, dpi)


def parse_formats(text: str) -> list[str]:
    formats = [item.strip().lower() for item in text.split(",") if item.strip()]
    allowed = {"png", "pdf", "svg"}
    if not formats or any(fmt not in allowed for fmt in formats):
        raise ValueError("--formats must contain only png,pdf,svg")
    return formats


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Plot Figure 3A motif accuracy heatmap.")
    parser.add_argument("--input_csv", required=True, help="Per-motif metrics CSV with kmer and accuracy columns.")
    parser.add_argument("--output_dir", required=True, help="Directory for the figure and matrix CSV.")
    parser.add_argument("--output_name", default="Figure3A_motif_accuracy_heatmap_16x64")
    parser.add_argument("--matrix_name", default="Figure3A_motif_accuracy_matrix_16x64.csv")
    parser.add_argument("--vmin", type=float, default=0.80)
    parser.add_argument("--vmax", type=float, default=1.00)
    parser.add_argument("--figure_width", type=float, default=42.0)
    parser.add_argument("--figure_height", type=float, default=14.0)
    parser.add_argument("--dpi", type=int, default=600)
    parser.add_argument("--formats", default="png,pdf,svg")
    parser.add_argument("--show_cell_text", action="store_true")
    args = parser.parse_args()
    if not (0 <= args.vmin < args.vmax <= 1.0):
        parser.error("Require 0 <= vmin < vmax <= 1")
    if args.dpi < 72:
        parser.error("--dpi must be at least 72")
    return args


def main() -> None:
    configure_style()
    args = parse_args()
    input_csv = Path(args.input_csv)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    formats = parse_formats(args.formats)

    df = load_accuracy_table(input_csv)
    matrix, matrix_df, rows, cols, used, skipped, duplicates = build_matrix(df)
    matrix_path = output_dir / args.matrix_name
    matrix_df.to_csv(matrix_path, encoding="utf-8-sig")

    plot_heatmap(
        matrix=matrix,
        row_labels=rows,
        col_labels=cols,
        output_stem=output_dir / args.output_name,
        formats=formats,
        dpi=args.dpi,
        vmin=args.vmin,
        vmax=args.vmax,
        figure_width=args.figure_width,
        figure_height=args.figure_height,
        show_cell_text=args.show_cell_text,
    )

    filled = int(np.isfinite(matrix).sum())
    print("=" * 90)
    print("Figure 3A motif heatmap completed")
    print(f"Input rows       : {len(df)}")
    print(f"Used rows        : {used}")
    print(f"Skipped rows     : {skipped}")
    print(f"Duplicate motifs : {duplicates}")
    print(f"Filled cells     : {filled}/1024")
    print(f"Matrix CSV       : {matrix_path}")
    print(f"Figure stem      : {output_dir / args.output_name}")
    print("=" * 90)


if __name__ == "__main__":
    main()
