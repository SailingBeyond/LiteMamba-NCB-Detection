#!/usr/bin/env python3
"""Replot Figure 3D–F from the archived coordinate and paired-motif tables."""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import re

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib import font_manager
from matplotlib.colors import TwoSlopeNorm
from matplotlib.ticker import ScalarFormatter
import numpy as np
import pandas as pd

PACKAGE_DIR = Path(__file__).resolve().parent

SEEDS = (42, 2024, 2025)
TASK_NAMES = {"6-class": "ATCGXY 6-Class", "5X": "ATCGX 5-Class", "5Y": "ATCGY 5-Class"}
TASKS = ("6-class", "5X", "5Y")
TASK_LABELS = {"6-class": ["A", "T", "C", "G", "X", "Y"], "5X": ["A", "T", "C", "G", "X"], "5Y": ["A", "T", "C", "G", "Y"]}
LABEL_COLORS = {"A": "#377DB8", "T": "#D98484", "C": "#4EA648", "G": "#D6B16A", "X": "#B8A2CF", "Y": "#8E4B98"}
UMAP_FIGSIZE = (14, 8)
UMAP_LAYOUT_RECT = (0.0, 0.0, .80, 1.0)

class MissingInput(FileNotFoundError):
    """A requested panel has no scientifically admissible prepared input yet."""


def configure_style():
    available = {f.name for f in font_manager.fontManager.ttflist}
    family = "Arial" if "Arial" in available else "DejaVu Sans"
    plt.rcParams.update({"font.family": family, "font.size": 16, "axes.labelsize": 18,
                         "axes.titlesize": 18, "xtick.labelsize": 15, "ytick.labelsize": 15,
                         "legend.fontsize": 13, "axes.linewidth": 1.5,
                         "xtick.major.width": 1.4, "ytick.major.width": 1.4,
                         "xtick.major.size": 5.5, "ytick.major.size": 5.5,
                         "pdf.fonttype": 42, "ps.fonttype": 42, "svg.fonttype": "none",
                         "figure.facecolor": "white", "axes.facecolor": "white"})
    return family


def _save(fig, stem, dpi):
    stem = Path(stem)
    stem.parent.mkdir(parents=True, exist_ok=True)
    paths = []
    try:
        for ext in ("png", "pdf", "svg"):
            path = stem.with_suffix("." + ext)
            fig.savefig(path, dpi=dpi, bbox_inches="tight", facecolor="white")
            paths.append(str(path))
    finally:
        plt.close(fig)
    return paths


def _csv(frame, path):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    frame.to_csv(path, index=False)
    return str(path)


def _numeric(frame, columns, probability=False, nonnegative=False):
    for column in columns:
        frame[column] = pd.to_numeric(frame[column], errors="raise")
        if not np.isfinite(frame[column]).all():
            raise ValueError(f"{column} contains nonfinite values")
        if probability and not frame[column].between(0, 1).all():
            raise ValueError(f"{column} must lie in [0, 1]")
        if nonnegative and (frame[column] < 0).any():
            raise ValueError(f"{column} must be nonnegative")


def normalize_motif(value):
    text = str(value).strip().upper()
    if text.startswith("KMER_"):
        text = text[5:]
    if not re.fullmatch(r"[ACGT]{3}N[ACGT]{2}", text):
        raise ValueError(f"Not a fourth-position-masked 6mer: {value!r}")
    return text


def _unique(frame, keys, name):
    if frame.duplicated(keys).any():
        example = frame.loc[frame.duplicated(keys, keep=False), keys].head(3).to_dict("records")
        raise ValueError(f"{name}: duplicate keys {keys}: {example}; raw/cal or repeated seeds must be resolved upstream")


def _require_seeds(frame, group_keys, name):
    for key, group in frame.groupby(group_keys, observed=True, dropna=False):
        if set(group.seed) != set(SEEDS):
            raise ValueError(f"{name}: {key} has seeds {sorted(set(group.seed))}, expected {list(SEEDS)}")


def _axis(ax):
    ax.tick_params(direction="out")
    for spine in ax.spines.values():
        spine.set_linewidth(1.5)


def load_paired(root):
    path = Path(root) / "data" / "paired_motif_metrics.csv"
    if not path.is_file():
        raise MissingInput("E/F 需要同一批合格测试窗口上的 paired_motif_metrics.csv；不能用各任务自身的背景准确率代替。请完成配对窗口审计与评估。")
    df = pd.read_csv(path)
    required = {"comparison", "seed", "motif", "accuracy_six", "accuracy_five", "n_samples"}
    if not required.issubset(df.columns) or df.empty:
        raise MissingInput("配对背景表缺少完整的同窗口评估记录。")
    numeric_seed = pd.to_numeric(df.seed, errors="raise")
    if not np.isfinite(numeric_seed).all() or not np.equal(numeric_seed, np.floor(numeric_seed)).all():
        raise ValueError("Paired-table seed must be a finite integer")
    df["seed"] = numeric_seed.astype(int)
    df = df[df.seed.isin(SEEDS)].copy()
    if set(df.comparison) != {"5X", "5Y"}:
        raise ValueError("Paired table must contain 5X and 5Y comparisons")
    df["motif"] = df.motif.map(normalize_motif)
    _unique(df, ["comparison", "seed", "motif"], "paired_motif_metrics")
    _require_seeds(df, ["comparison", "motif"], "paired_motif_metrics")
    _numeric(df, ["accuracy_six", "accuracy_five"], probability=True)
    _numeric(df, ["n_samples"], nonnegative=True)
    if (df.n_samples <= 0).any():
        raise ValueError("Paired-window metrics require positive support")
    if not np.equal(df.n_samples, np.floor(df.n_samples)).all():
        raise ValueError("Paired-window support must be an integer")
    return df


def aggregate_paired(df):
    z = df.groupby(["comparison", "motif"], observed=True).agg(
        accuracy_six=("accuracy_six", "mean"), accuracy_five=("accuracy_five", "mean"),
        accuracy_six_sd=("accuracy_six", "std"), accuracy_five_sd=("accuracy_five", "std"),
        n_samples_min=("n_samples", "min"), n_samples_max=("n_samples", "max"),
        n_seeds=("seed", "nunique")).reset_index()
    z["difference_five_minus_six"] = z.accuracy_five - z.accuracy_six
    return z


def draw_paired(ax, frame, comparison, limits, color_limit):
    z = frame[frame.comparison.eq(comparison)]
    norm = TwoSlopeNorm(vmin=-color_limit, vcenter=0, vmax=color_limit)
    scatter = ax.scatter(z.accuracy_six, z.accuracy_five, c=z.difference_five_minus_six,
                         cmap="coolwarm", norm=norm, s=25, alpha=.8, linewidths=0, rasterized=True)
    ax.plot(limits, limits, "--", color="0.35", linewidth=1.2)
    ax.set_xlim(*limits)
    ax.set_ylim(*limits)
    ax.set_aspect("equal", adjustable="box")
    ax.set_xlabel("ATCGXY 6-Class per-motif accuracy")
    ax.set_ylabel(TASK_NAMES[comparison] + " per-motif accuracy")
    ax.grid(alpha=.2)
    _axis(ax)
    return scatter


def plot_paired(root, output, dpi=600):
    df = load_paired(root)
    averaged = aggregate_paired(df)
    paths = [_csv(averaged, Path(output) / "plot_data" / "Figure3EF_paired_motif_mean_sd.csv")]
    lower = max(0, min(.85, float(averaged[["accuracy_six", "accuracy_five"]].min().min()) - .02))
    limits = (lower, 1.005)
    color_limit = max(.01, float(np.abs(averaged.difference_five_minus_six).max()))
    stats = []
    for task, letter in (("5X", "E"), ("5Y", "F")):
        z = averaged[averaged.comparison.eq(task)]
        stats.append({"comparison": task, "n_motifs": len(z), "mean_difference": float(z.difference_five_minus_six.mean()),
                      "median_difference": float(z.difference_five_minus_six.median()),
                      "fraction_five_higher": float((z.difference_five_minus_six > 0).mean())})
        fig, ax = plt.subplots(figsize=(7.2, 6))
        sc = draw_paired(ax, averaged, task, limits, color_limit)
        cbar = fig.colorbar(sc, ax=ax, fraction=.046, pad=.025)
        cbar.set_label("Accuracy difference")
        fig.tight_layout()
        paths.extend(_save(fig, Path(output) / "figures" / ("Figure3" + letter + "_paired_GatedConv_" + task), dpi))
    paths.append(_csv(pd.DataFrame(stats), Path(output) / "plot_data" / "Figure3EF_comparison_summary.csv"))
    fig, axes = plt.subplots(1, 2, figsize=(15.6, 6.5))
    for ax, task in zip(axes, ("5X", "5Y")):
        sc = draw_paired(ax, averaged, task, limits, color_limit)
        cbar = fig.colorbar(sc, ax=ax, fraction=.046, pad=.025)
        cbar.set_label("Accuracy difference")
    fig.tight_layout()
    paths.extend(_save(fig, Path(output) / "figures" / "Figure3EF_combined_paired_GatedConv", dpi))
    notes = ("Each point is a masked 6mer, using the same admitted test windows for both models. Values are equal-weight means across seeds 42/2024/2025. "
             "The six-class model retains six-way argmax: predicting the excluded class remains an error. "
             "Different original training-set sizes remain; this is a comparison of the trained systems, not a causal class-count-only ablation.\n")
    path = Path(output) / "plot_data" / "Figure3EF_caption_notes.txt"
    path.write_text(notes, encoding="utf-8")
    paths.append(str(path))
    return paths


class _UMAPScalarFormatter(ScalarFormatter):
    """Keep the original ASCII minus after the local rendering context exits."""

    @staticmethod
    def fix_minus(text):
        return text


def draw_umap_coordinates(embedding, labels, task):
    """Render coordinates without recomputing features, scaling or UMAP."""
    embedding = np.asarray(embedding)
    labels = np.asarray(labels)
    if embedding.ndim != 2 or embedding.shape[1] != 2 or len(embedding) != len(labels):
        raise ValueError("UMAP coordinates and labels must align")
    if not len(labels) or not np.isfinite(embedding).all():
        raise ValueError("UMAP coordinates must be nonempty and finite")
    if not np.isfinite(labels).all() or not np.equal(labels, np.floor(labels)).all():
        raise ValueError("UMAP labels must be finite integers")
    labels = labels.astype(np.int64)
    if set(labels) != set(range(len(TASK_LABELS[task]))):
        raise ValueError("UMAP coordinates must retain every local biological class")
    # The original UMAP scripts use the default 10pt base font, unlike the
    # 16pt shared statistics style. tight_layout padding depends on that base
    # font; restoring only figsize/rect did not restore the exported geometry.
    available = {f.name for f in font_manager.fontManager.ttflist}
    family = "Arial" if "Arial" in available else "DejaVu Sans"
    with plt.rc_context({"font.family": family, "font.size": 10,
                         "font.weight": "normal", "axes.labelweight": "normal",
                         "axes.unicode_minus": False}):
        fig, ax = plt.subplots(figsize=UMAP_FIGSIZE)
        for label, name in enumerate(TASK_LABELS[task]):
            mask = labels == label
            ax.scatter(embedding[mask, 0], embedding[mask, 1], color=LABEL_COLORS[name],
                       s=50, alpha=.50, edgecolors="none", label=name, rasterized=True)
        ax.set_xlabel("UMAP Dimension 1", fontsize=30)
        ax.set_ylabel("UMAP Dimension 2", fontsize=30)
        # Tick labels are formatted again on savefig, after rc_context exits.
        # Persist this one original-script setting on these two axes only.
        ax.xaxis.set_major_formatter(_UMAPScalarFormatter())
        ax.yaxis.set_major_formatter(_UMAPScalarFormatter())
        ax.tick_params(axis="both", which="major", labelsize=30,
                       direction="out", length=6, width=3, colors="black")
        for side in ("left", "right", "top", "bottom"):
            ax.spines[side].set_linewidth(3)
            ax.spines[side].set_color("black")
        legend = ax.legend(frameon=False, loc="center left", bbox_to_anchor=(1.02, .5),
                           ncol=1, borderaxespad=0, handletextpad=.4, labelspacing=.6,
                           markerscale=1.0, fontsize=25, title="")
        legend.get_title().set_fontsize(28)
        legend.get_title().set_multialignment("center")
        legend.get_title().set_ha("center")
        # Retain all new-data coordinates. Original fixed y-limits were tuned
        # to the historical maps and can clip points in newly fitted maps.
        fig.tight_layout(rect=UMAP_LAYOUT_RECT)
    return fig


def replot_saved_umaps(root, output, tasks=TASKS, dpi=600):
    """Style-only replay; no torch/UMAP import, no writes to coordinate files."""
    paths, missing = [], []
    for task in tasks:
        prefix = "Figure3D" if task == "6-class" else "Supplementary_UMAP_" + task
        npz = Path(root) / "plot_data" / (prefix + "_coordinates.npz")
        csv = npz.with_suffix(".csv")
        if npz.is_file():
            with np.load(npz, allow_pickle=False) as data:
                embedding, labels = data["embedding"], data["labels"]
        elif csv.is_file():
            data = pd.read_csv(csv)
            required = {"umap_1", "umap_2", "label"}
            if not required.issubset(data.columns):
                raise ValueError(f"{csv}: missing coordinate columns")
            embedding = data[["umap_1", "umap_2"]].to_numpy()
            labels = data["label"].to_numpy()
        else:
            missing.append({"task": task, "reason": "未找到已保存的UMAP坐标；本次不重提特征、不重新拟合UMAP。",
                            "expected": [str(npz), str(csv)]})
            continue
        fig = draw_umap_coordinates(embedding, labels, task)
        paths.extend(_save(fig, Path(output) / "figures" / (prefix + "_GatedConv_seed42"), dpi))
    return {"produced": paths, "missing": missing}

def verify_digest(path, expected):
    actual = hashlib.sha256(path.read_bytes()).hexdigest()
    if actual != expected:
        raise ValueError(f"SHA-256 mismatch for {path.name}")


def verify_embedding(prefix):
    input_file = PACKAGE_DIR / "plot_data" / (prefix + "_coordinates.npz")
    manifest_file = PACKAGE_DIR / "plot_data" / (prefix + "_manifest.json")
    manifest = json.loads(manifest_file.read_text(encoding="utf-8"))
    verify_digest(input_file, manifest["coordinates_sha256"])
    with np.load(input_file, allow_pickle=False) as data:
        labels = data["labels"]
    counts = {name: int(np.count_nonzero(labels == index))
              for index, name in enumerate(TASK_LABELS[manifest["task"]])}
    if len(labels) != manifest["n_points"] or counts != manifest["points_by_class"]:
        raise ValueError(f"Class counts do not match {manifest_file.name}")


def parse_options():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", type=Path, default=PACKAGE_DIR / "outputs")
    parser.add_argument("--dpi", type=int, default=600)
    args = parser.parse_args()
    if not 72 <= args.dpi <= 1200:
        parser.error("--dpi must be between 72 and 1200")
    return args


def main():
    args = parse_options()
    audit = json.loads((PACKAGE_DIR / "data/paired_audit.json").read_text(encoding="utf-8"))
    verify_digest(PACKAGE_DIR / "data/paired_motif_metrics.csv", audit["paired_motif_csv"]["sha256"])
    verify_embedding("Figure3D")
    configure_style()
    report = replot_saved_umaps(PACKAGE_DIR, args.output_dir, tasks=("6-class",), dpi=args.dpi)
    if report["missing"]:
        raise MissingInput(report["missing"])
    plot_paired(PACKAGE_DIR, args.output_dir, dpi=args.dpi)
    print(f"Figure 3D–F written to {args.output_dir.resolve()}")


if __name__ == "__main__":
    main()
