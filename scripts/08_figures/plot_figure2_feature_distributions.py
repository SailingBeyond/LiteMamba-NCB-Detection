#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Plot Figure 2 feature distributions for one canonical/non-canonical base pair.

This script replaces the two nearly identical X- and Y-specific plotting scripts.
It reads an 11-column feature CSV, filters one 6-mer motif and two labels, and
produces:

1. A compact 1 x 8 target-site panel for mean, standard deviation, median,
   dwell, base quality, mismatch, insertion, and deletion.
2. Optional six-position panels for each of the eight features.

Examples
--------
X versus A::

    python plot_figure2_feature_distributions.py \
        --input xna_pc_6mer_features_pos.csv \
        --output_dir results/figure2/features_x \
        --target_motif ATAAGA \
        --natural_name A \
        --modified_name X

Y versus C::

    python plot_figure2_feature_distributions.py \
        --input xna_pc_6mer_features_neg.csv \
        --output_dir results/figure2/features_y \
        --target_motif ATACGA \
        --natural_name C \
        --modified_name Y
"""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import Dict, Iterable, List, Sequence, Tuple

import matplotlib
matplotlib.use("Agg")
import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


EXPECTED_COLUMNS = [
    "kmer", "mean", "std", "median", "dwell",
    "quality", "mismatch", "insertion", "deletion",
    "signal", "label",
]

CONTINUOUS_FEATURES = ["mean", "std", "median", "dwell", "quality"]
ERROR_FEATURES = ["mismatch", "insertion", "deletion"]
PANEL_FEATURES = CONTINUOUS_FEATURES + ERROR_FEATURES

FEATURE_TITLES = {
    "mean": "Mean",
    "std": "Std",
    "median": "Median",
    "dwell": "Dwell",
    "quality": "Quality",
    "mismatch": "Mismatch",
    "insertion": "Insertion",
    "deletion": "Deletion",
}

FEATURE_YLABELS = {
    "mean": "Mean",
    "std": "Standard deviation",
    "median": "Median",
    "dwell": "Dwell",
    "quality": "Base quality",
    "mismatch": "Proportion of mismatches",
    "insertion": "Proportion of insertions",
    "deletion": "Proportion of deletions",
}

FEATURE_YLIMS: Dict[str, Tuple[float, float]] = {
    "mean": (-1.5, 1.5),
    "std": (0.0, 0.5),
    "median": (-1.5, 1.5),
    "dwell": (0.0, 60.0),
    "quality": (0.0, 50.0),
    "mismatch": (0.0, 0.5),
    "insertion": (0.0, 0.5),
    "deletion": (0.0, 0.5),
}

COMPACT_YTICKS: Dict[str, Sequence[float]] = {
    "mean": (-1.5, 0.0, 1.5),
    "std": (0.0, 0.25, 0.5),
    "median": (-1.5, 0.0, 1.5),
    "dwell": (0.0, 30.0, 60.0),
    "quality": (0.0, 25.0, 50.0),
    "mismatch": (0.0, 0.25, 0.5),
    "insertion": (0.0, 0.25, 0.5),
    "deletion": (0.0, 0.25, 0.5),
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Plot Figure 2 feature distributions for one motif and base pair."
    )
    parser.add_argument("--input", required=True, help="Input 11-column feature CSV.")
    parser.add_argument("--output_dir", required=True, help="Output directory.")
    parser.add_argument("--target_motif", required=True, help="Target 6-mer motif, e.g. ATAAGA.")
    parser.add_argument("--natural_label", type=int, default=0, help="Canonical-base label. Default: 0")
    parser.add_argument("--modified_label", type=int, default=1, help="Non-canonical-base label. Default: 1")
    parser.add_argument("--natural_name", required=True, help="Canonical-base display name, e.g. A or C.")
    parser.add_argument("--modified_name", required=True, help="Non-canonical-base display name, e.g. X or Y.")
    parser.add_argument("--target_index", type=int, default=3, help="0-based target position within the 6-mer. Default: 3")
    parser.add_argument("--natural_color", default="#EB716B", help="Canonical-base color.")
    parser.add_argument("--modified_color", default="#6CB3DA", help="Non-canonical-base color.")
    parser.add_argument(
        "--formats",
        default="png,pdf,svg",
        help="Comma-separated output formats chosen from png,pdf,svg. Default: png,pdf,svg",
    )
    parser.add_argument("--dpi", type=int, default=600, help="PNG resolution. Default: 600")
    parser.add_argument(
        "--compact_only",
        action="store_true",
        help="Only draw the compact target-site 1 x 8 panel.",
    )
    args = parser.parse_args()

    motif = args.target_motif.strip().upper()
    if len(motif) != 6:
        parser.error("--target_motif must contain exactly 6 characters.")
    if not (0 <= args.target_index < 6):
        parser.error("--target_index must be between 0 and 5.")
    if args.natural_label == args.modified_label:
        parser.error("--natural_label and --modified_label must differ.")
    if args.dpi < 72:
        parser.error("--dpi must be at least 72.")

    allowed = {"png", "pdf", "svg"}
    requested = {x.strip().lower() for x in args.formats.split(",") if x.strip()}
    if not requested or not requested.issubset(allowed):
        parser.error("--formats may only contain png,pdf,svg.")

    args.target_motif = motif
    args.formats = tuple(x.strip().lower() for x in args.formats.split(",") if x.strip())
    return args


def configure_matplotlib() -> None:
    mpl.rcParams["font.family"] = "Arial"
    mpl.rcParams["font.sans-serif"] = ["Arial", "DejaVu Sans"]
    mpl.rcParams["pdf.fonttype"] = 42
    mpl.rcParams["ps.fonttype"] = 42
    mpl.rcParams["axes.unicode_minus"] = False


def load_feature_csv(path: Path) -> pd.DataFrame:
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(f"Input CSV not found: {path}")

    data = pd.read_csv(path)
    if set(EXPECTED_COLUMNS).issubset(data.columns):
        return data.loc[:, EXPECTED_COLUMNS].copy()

    data = pd.read_csv(path, header=None)
    if data.shape[1] < len(EXPECTED_COLUMNS):
        raise ValueError(
            f"Input CSV must contain at least {len(EXPECTED_COLUMNS)} columns; "
            f"found {data.shape[1]}."
        )
    data = data.iloc[:, : len(EXPECTED_COLUMNS)].copy()
    data.columns = EXPECTED_COLUMNS
    return data


def split_six_values(series: pd.Series, feature_name: str) -> pd.DataFrame:
    expanded = series.astype(str).str.split("|", expand=True)
    if expanded.shape[1] != 6:
        raise ValueError(
            f"Feature '{feature_name}' must contain six pipe-separated values per row; "
            f"found {expanded.shape[1]} columns."
        )
    return expanded.apply(pd.to_numeric, errors="coerce")


def position_labels(motif: str, target_index: int) -> List[str]:
    labels = []
    for index, base in enumerate(motif):
        relative = index - target_index
        labels.append(f"{relative}\n({base})")
    return labels


def save_figure(fig: plt.Figure, output_dir: Path, stem: str, formats: Iterable[str], dpi: int) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    for fmt in formats:
        out_path = output_dir / f"{stem}.{fmt}"
        kwargs = {"bbox_inches": "tight"}
        if fmt == "png":
            kwargs["dpi"] = dpi
        fig.savefig(out_path, **kwargs)
        print(f"[ok] Saved: {out_path}")


def style_small_axis(ax: plt.Axes) -> None:
    ax.set_facecolor("white")
    ax.grid(axis="y", linestyle="-", linewidth=0.22, alpha=0.25)
    for spine in ax.spines.values():
        spine.set_visible(True)
        spine.set_color("black")
        spine.set_linewidth(0.45)
        spine.set_linestyle("solid")
    ax.tick_params(axis="x", labelsize=6.5, colors="black", pad=0.4, length=1.5)
    ax.tick_params(axis="y", labelsize=5.2, colors="black", pad=0.3, length=1.5)


def target_site_values(df: pd.DataFrame, feature: str, target_index: int) -> pd.Series:
    expanded = split_six_values(df[feature], feature)
    return expanded.iloc[:, target_index]


def draw_compact_panel(
    df: pd.DataFrame,
    output_dir: Path,
    motif: str,
    target_index: int,
    label_names: Sequence[str],
    colors: Sequence[str],
    formats: Iterable[str],
    dpi: int,
) -> None:
    x_positions = np.arange(2)
    fig, axes = plt.subplots(1, 8, figsize=(4.0, 1.2), dpi=300)

    for ax, feature in zip(axes, PANEL_FEATURES):
        values = target_site_values(df, feature, target_index)
        local = pd.DataFrame({"label_text": df["label_text"].astype(str), "value": values}).dropna()

        if feature in CONTINUOUS_FEATURES:
            values_by_label = [
                local.loc[local["label_text"] == name, "value"].to_numpy()
                for name in label_names
            ]
            boxes = ax.boxplot(
                values_by_label,
                positions=x_positions,
                widths=0.28,
                patch_artist=True,
                showfliers=False,
                medianprops={"color": "black", "linewidth": 0.7},
                boxprops={"color": "black", "linewidth": 0.45},
                whiskerprops={"color": "black", "linewidth": 0.45},
                capprops={"color": "black", "linewidth": 0.45},
            )
            for patch, color in zip(boxes["boxes"], colors):
                patch.set_facecolor(color)
                patch.set_alpha(0.9)
        else:
            proportions = []
            for name in label_names:
                arr = local.loc[local["label_text"] == name, "value"].to_numpy()
                proportions.append(float(np.mean(arr != 0)) if arr.size else np.nan)
            bars = ax.bar(x_positions, proportions, width=0.28, edgecolor="black", linewidth=0.45)
            for bar, color in zip(bars, colors):
                bar.set_facecolor(color)
                bar.set_alpha(0.9)

        ax.set_title(FEATURE_TITLES[feature], fontsize=7.0, fontfamily="Arial", pad=2.0)
        ax.set_xticks(x_positions)
        ax.set_xticklabels(label_names, fontsize=6.5, fontfamily="Arial")
        ax.set_xlim(-0.28, 1.28)
        ax.set_ylim(FEATURE_YLIMS[feature])
        ax.set_yticks(COMPACT_YTICKS[feature])
        ax.set_ylabel("")
        style_small_axis(ax)

    fig.subplots_adjust(left=0.07, right=0.995, bottom=0.17, top=0.90, wspace=0.6)
    stem = f"Figure2_{motif}_{label_names[0]}_vs_{label_names[1]}_target_site_8features"
    save_figure(fig, output_dir, stem, formats, dpi)
    plt.close(fig)


def draw_position_panel(
    df: pd.DataFrame,
    feature: str,
    output_dir: Path,
    motif: str,
    target_index: int,
    label_names: Sequence[str],
    colors: Sequence[str],
    formats: Iterable[str],
    dpi: int,
) -> None:
    expanded = split_six_values(df[feature], feature)
    pos_labels = position_labels(motif, target_index)
    fig, axes = plt.subplots(1, 6, figsize=(8.0, 4.0), sharey=True)

    for position, ax in enumerate(axes):
        local = pd.DataFrame(
            {
                "label_text": df["label_text"].astype(str),
                "value": expanded.iloc[:, position],
            }
        ).dropna()

        if feature in CONTINUOUS_FEATURES:
            values_by_label = [
                local.loc[local["label_text"] == name, "value"].to_numpy()
                for name in label_names
            ]
            boxes = ax.boxplot(
                values_by_label,
                positions=[0, 1],
                widths=0.5,
                patch_artist=True,
                showfliers=False,
                medianprops={"color": "black", "linewidth": 1.0},
                boxprops={"color": "black", "linewidth": 0.8},
                whiskerprops={"color": "black", "linewidth": 0.8},
                capprops={"color": "black", "linewidth": 0.8},
            )
            for patch, color in zip(boxes["boxes"], colors):
                patch.set_facecolor(color)
                patch.set_alpha(0.9)
        else:
            proportions = []
            for name in label_names:
                arr = local.loc[local["label_text"] == name, "value"].to_numpy()
                proportions.append(float(np.mean(arr != 0)) if arr.size else np.nan)
            bars = ax.bar([0, 1], proportions, width=0.5, edgecolor="black", linewidth=0.8)
            for bar, color in zip(bars, colors):
                bar.set_facecolor(color)
                bar.set_alpha(0.9)

        ax.set_title(pos_labels[position], fontsize=12)
        ax.set_xticks([0, 1])
        ax.set_xticklabels(label_names, fontsize=11)
        ax.set_ylim(FEATURE_YLIMS[feature])
        ax.grid(axis="y", linestyle="-", linewidth=0.5, alpha=0.25)
        for spine in ax.spines.values():
            spine.set_linewidth(0.8)

    axes[0].set_ylabel(FEATURE_YLABELS[feature], fontsize=13)
    fig.subplots_adjust(left=0.09, right=0.995, bottom=0.16, top=0.88, wspace=0.12)
    stem = f"Figure2_{motif}_{label_names[0]}_vs_{label_names[1]}_{feature}_six_positions"
    save_figure(fig, output_dir, stem, formats, dpi)
    plt.close(fig)


def main() -> None:
    args = parse_args()
    configure_matplotlib()

    input_path = Path(args.input)
    output_dir = Path(args.output_dir)
    data = load_feature_csv(input_path)
    data["kmer"] = data["kmer"].astype(str).str.upper()
    data["label"] = pd.to_numeric(data["label"], errors="coerce")

    selected = data[
        (data["kmer"] == args.target_motif)
        & data["label"].isin([args.natural_label, args.modified_label])
    ].copy()
    if selected.empty:
        raise ValueError(
            f"No rows found for motif={args.target_motif} and labels="
            f"[{args.natural_label}, {args.modified_label}]."
        )

    mapping = {
        args.natural_label: args.natural_name,
        args.modified_label: args.modified_name,
    }
    label_names = [args.natural_name, args.modified_name]
    selected["label_text"] = pd.Categorical(
        selected["label"].map(mapping), categories=label_names, ordered=True
    )
    selected = selected.dropna(subset=["label_text"]).copy()

    print(f"[info] Input: {input_path}")
    print(f"[info] Motif: {args.target_motif}")
    print(f"[info] Selected rows: {len(selected):,}")
    print("[info] Label counts:")
    print(selected["label_text"].value_counts(sort=False))

    colors = [args.natural_color, args.modified_color]
    draw_compact_panel(
        selected,
        output_dir,
        args.target_motif,
        args.target_index,
        label_names,
        colors,
        args.formats,
        args.dpi,
    )

    if not args.compact_only:
        for feature in PANEL_FEATURES:
            draw_position_panel(
                selected,
                feature,
                output_dir,
                args.target_motif,
                args.target_index,
                label_names,
                colors,
                args.formats,
                args.dpi,
            )

    print("[done] Figure 2 feature plots generated successfully.")


if __name__ == "__main__":
    main()
