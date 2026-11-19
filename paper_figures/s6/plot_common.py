#!/usr/bin/env python3
"""Figure 3 and supplementary plots from frozen, audited current results.

This module never loads a neural network or trains a model. It deliberately
accepts only the canonical tables produced by collect_results.py. In particular,
task-specific accuracies are not silently substituted for paired-window data.
"""
from __future__ import annotations

import argparse
from collections import Counter
from fractions import Fraction
from itertools import product
import json
import math
import os
from pathlib import Path
import re
import sys
from typing import Callable

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib import font_manager
from matplotlib.colors import Normalize, TwoSlopeNorm
from matplotlib.ticker import MaxNLocator, PercentFormatter, ScalarFormatter
import numpy as np
import pandas as pd

DEFAULT_RESULT_ROOT = Path(__file__).resolve().parent / "results"
SEEDS = (42, 2024, 2025)
TASKS = ("6-class", "5X", "5Y")
TASK_NAMES = {"6-class": "ATCGXY 6-Class", "5X": "ATCGX 5-Class", "5Y": "ATCGY 5-Class"}
TASK_LABELS = {"6-class": ["A", "T", "C", "G", "X", "Y"], "5X": ["A", "T", "C", "G", "X"], "5Y": ["A", "T", "C", "G", "Y"]}
MODELS = ("BiGRU", "Transformer", "ConvMixer", "GatedConv")
MODEL_COLORS = {"BiGRU": "#4C72B0", "Transformer": "#DD8452", "ConvMixer": "#55A868", "GatedConv": "#C44E52"}
TASK_COLORS = {"6-class": "#4C72B0", "5X": "#DD8452", "5Y": "#55A868"}
LABEL_COLORS = {"A": "#377DB8", "T": "#D98484", "C": "#4EA648", "G": "#D6B16A", "X": "#B8A2CF", "Y": "#8E4B98"}
METRIC_NAMES = {"accuracy": "Accuracy", "balanced_accuracy": "Balanced accuracy", "f1_macro": "Macro-F1", "mcc": "MCC"}
BASES = ("A", "C", "G", "T")
LEFT_LABELS = ["".join(v) for v in product(BASES, repeat=3)]
RIGHT_LABELS = ["".join(v) for v in product(BASES, repeat=2)]
MOTIFS = [left + "N" + right for right in RIGHT_LABELS for left in LEFT_LABELS]

# Exact canvas/layout settings from the original uploaded plotting scripts.
# bbox_inches="tight" is retained, just as in those scripts.
HEATMAP_FIGSIZE = (42, 14)
HEATMAP_MARGINS = dict(left=.07, right=.965, top=.88, bottom=.23)
OVERALL_SINGLE_FIGSIZE = (5, 5.5)
OVERALL_SUMMARY_FIGSIZE = (10.6, 10.2)
ECDF_FIGSIZE = (5, 5.5)
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


def _write_json(path, obj):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(obj, ensure_ascii=False, indent=2, allow_nan=False) + "\n", encoding="utf-8")


def _csv(frame, path):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    frame.to_csv(path, index=False)
    return str(path)


def _read(root, name, required):
    path = Path(root) / "data" / name
    if not path.is_file():
        raise MissingInput(f"缺少 {path}；请先运行结果整理入口。")
    frame = pd.read_csv(path)
    absent = sorted(set(required) - set(frame.columns))
    if absent:
        raise ValueError(f"{name} 缺少字段：{absent}")
    if frame.empty:
        raise MissingInput(f"{name} 没有可用记录。")
    if "task" in frame and not set(frame.task.astype(str)).issubset(TASKS):
        raise ValueError(f"{name} contains an unsupported task")
    if "model" in frame and not set(frame.model.astype(str)).issubset(MODELS):
        raise ValueError(f"{name} contains an unsupported model; legacy LiteMamba scores cannot be renamed")
    if "seed" in frame:
        numeric = pd.to_numeric(frame.seed, errors="raise")
        if not np.isfinite(numeric).all() or not np.equal(numeric, np.floor(numeric)).all():
            raise ValueError(f"{name}: seed must be a finite integer")
        frame["seed"] = numeric.astype(int)
        frame = frame[frame.seed.isin(SEEDS)].copy()
    return frame


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


def load_overall(root):
    cols = ["task", "model", "seed", *METRIC_NAMES, "n_samples"]
    df = _read(root, "overall_metrics.csv", cols)
    _unique(df, ["task", "model", "seed"], "overall_metrics")
    _numeric(df, ["accuracy", "balanced_accuracy", "f1_macro"], probability=True)
    _numeric(df, ["mcc"])
    if not df.mcc.between(-1, 1).all():
        raise ValueError("MCC outside [-1,1]")
    _numeric(df, ["n_samples"], nonnegative=True)
    if (df.n_samples <= 0).any() or not np.equal(df.n_samples, np.floor(df.n_samples)).all():
        raise ValueError("Overall test support must be a positive integer")
    if set(zip(df.task, df.model)) != {(t, m) for t in TASKS for m in MODELS}:
        raise ValueError("Overall results must contain all 12 architecture/task combinations")
    _require_seeds(df, ["task", "model"], "overall_metrics")
    return df


def load_motifs(root):
    df = _read(root, "per_motif_metrics.csv", ["task", "model", "seed", "motif", "accuracy", "n_samples"])
    df = df[df.model.eq("GatedConv")].copy()
    if df.empty:
        raise MissingInput("No final GatedConv per-motif records")
    df["motif"] = df.motif.map(normalize_motif)
    _unique(df, ["task", "model", "seed", "motif"], "per_motif_metrics")
    _numeric(df, ["accuracy"], probability=True)
    _numeric(df, ["n_samples"], nonnegative=True)
    if set(df.task) != set(TASKS):
        raise ValueError("Per-motif results must contain the three final GatedConv tasks")
    _require_seeds(df, ["task"], "per_motif_metrics")
    if (df.n_samples <= 0).any():
        raise ValueError("Per-motif accuracy requires positive support; missing backgrounds must be absent, not zero-filled")
    if not np.equal(df.n_samples, np.floor(df.n_samples)).all():
        raise ValueError("Per-motif support must be an integer")
    return df


def aggregate_motifs(frame, *, retain_seed_counts=True):
    """Equal seed weighting; never pool counts or silently keep the last seed."""
    out = frame.groupby(["task", "motif"], observed=True).agg(
        accuracy_mean=("accuracy", "mean"), accuracy_sd=("accuracy", "std"),
        n_seeds=("seed", "nunique"), n_samples_min=("n_samples", "min"), n_samples_max=("n_samples", "max")).reset_index()
    out["complete_seeds"] = out.n_seeds.eq(len(SEEDS))
    out.loc[~out.complete_seeds, ["accuracy_mean", "accuracy_sd"]] = np.nan
    if retain_seed_counts:
        # The ECDF needs the original per-run denominators to classify equality
        # exactly, including when different runs have different window counts.
        correct = frame["n_correct"] if "n_correct" in frame else [None] * len(frame)
        observations = list(zip(frame.accuracy, frame.n_samples, correct))
        by_motif = frame.assign(seed_counts=observations).groupby(["task", "motif"], observed=True)
        out = out.join(by_motif.seed_counts.agg(tuple), on=["task", "motif"])
    return out


def build_heatmap_matrix(aggregated, task):
    values = aggregated[aggregated.task.eq(task)].set_index("motif").accuracy_mean.to_dict()
    matrix = np.array([[values.get(left + "N" + right, np.nan) for left in LEFT_LABELS] for right in RIGHT_LABELS], dtype=float)
    return matrix


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


def _axis(ax):
    ax.tick_params(direction="out")
    for spine in ax.spines.values():
        spine.set_linewidth(1.5)


def plot_heatmaps(root, tasks=TASKS, dpi=600, heatmap_vmin=None, heatmap_vmax=1.0):
    df = load_motifs(root)
    aggregated = aggregate_motifs(df, retain_seed_counts=False)
    paths = [_csv(aggregated, Path(root) / "plot_data" / "motif_mean_sd.csv")]
    from heatmap_audit import write_heatmap_audit
    audit = write_heatmap_audit(root, df, aggregated)
    paths.extend(audit["paths"])
    finite = aggregated.accuracy_mean.dropna()
    if finite.empty:
        raise ValueError("No backgrounds have the complete three-seed accuracy")
    global_minimum = float(finite.min())
    # Cover every observed individual seed as well as the displayed seed means.
    # This is a range choice only: each cell still shows the three-seed mean.
    raw_minimum = float(df.accuracy.min())
    automatic_vmin = max(0.0, math.floor((raw_minimum + 1e-12) / .05) * .05)
    if automatic_vmin >= heatmap_vmax:
        automatic_vmin = max(0.0, heatmap_vmax - .05)
    vmin = automatic_vmin if heatmap_vmin is None else float(heatmap_vmin)
    vmax = float(heatmap_vmax)
    if not np.isfinite([vmin, vmax]).all() or not 0 <= vmin < vmax <= 1:
        raise ValueError("Heatmap bounds must satisfy 0 <= vmin < vmax <= 1")
    # Never silently saturate values by accepting a manually narrowed range.
    if global_minimum < vmin - 1e-12 or float(finite.max()) > vmax + 1e-12:
        raise ValueError("Colorbar bounds exclude an observed heatmap mean; expand the range")
    color_manifest = Path(root) / "plot_data" / "motif_heatmap_color_scale.json"
    _write_json(color_manifest, {"vmin": vmin, "vmax": vmax,
                                 "complete_seed_mean_minimum_all_three_tasks": global_minimum,
                                 "individual_seed_minimum_all_three_tasks": raw_minimum,
                                 "automatic_vmin": automatic_vmin,
                                 "bound_source": "automatic" if heatmap_vmin is None else "user-specified",
                                 "rule": "Default common lower bound: floor(global individual-seed minimum / 0.05)*0.05; cells remain equal-weight three-seed means; all three tasks determine the common bounds even when plotting only one task",
                                 "task_minima": audit["summary"],
                                 "missing_color": "#D9D9D9", "statistic": "equal-weight seed mean"})
    paths.append(str(color_manifest))
    print(f"[HEATMAP COLORBAR] common range {vmin:.2f}–{vmax:.2f}; cells = three-seed mean", flush=True)
    for task in tasks:
        matrix = build_heatmap_matrix(aggregated, task)
        fig, ax = plt.subplots(figsize=HEATMAP_FIGSIZE)
        cmap = plt.get_cmap("viridis").copy()
        cmap.set_bad("#D9D9D9")
        image = ax.imshow(np.ma.masked_invalid(matrix), aspect="auto", cmap=cmap, vmin=vmin, vmax=vmax, interpolation="nearest")
        ax.set_xticks(np.arange(64), LEFT_LABELS, rotation=45, ha="right", rotation_mode="anchor", fontsize=22)
        ax.set_yticks(np.arange(16), RIGHT_LABELS, fontsize=28)
        ax.set_xlabel("Three bases on the left side of N", fontsize=34, labelpad=24)
        ax.set_ylabel("Two bases on the right side of N", fontsize=34, labelpad=24)
        ax.set_title(TASK_NAMES[task], fontsize=42, fontweight="bold", pad=28)
        ax.tick_params(axis="x", pad=8)
        ax.tick_params(axis="y", pad=8)
        cbar = fig.colorbar(image, ax=ax, fraction=0.028, pad=0.02)
        cbar.set_label("Mean accuracy (3 seeds)", fontsize=34, labelpad=18)
        cbar.ax.tick_params(labelsize=26)
        tick_start = math.ceil((vmin - 1e-12) / .05) * .05
        ticks = np.concatenate(([vmin], np.arange(tick_start, vmax + 1e-12, .05), [vmax]))
        cbar.set_ticks(np.unique(np.round(ticks, 10)))
        _axis(ax)
        fig.subplots_adjust(**HEATMAP_MARGINS)
        prefix = "Figure3A" if task == "6-class" else "Supplementary_motif_heatmap_" + task
        paths.extend(_save(fig, Path(root) / "figures" / (prefix + "_GatedConv"), dpi))
        matrix_df = pd.DataFrame(matrix, index=RIGHT_LABELS, columns=LEFT_LABELS)
        matrix_df.index.name = "right_two_bases"
        path = Path(root) / "plot_data" / (prefix + "_matrix.csv")
        path.parent.mkdir(parents=True, exist_ok=True)
        matrix_df.to_csv(path)
        paths.append(str(path))
    return paths


def summarize_overall(df):
    pieces = []
    for metric in METRIC_NAMES:
        z = df.groupby(["task", "model"], observed=True)[metric].agg(["mean", "std", "count"]).reset_index()
        z["metric"] = metric
        pieces.append(z)
    return pd.concat(pieces, ignore_index=True)


def draw_overall(ax, df, summary, metric, with_legend=True):
    x = np.arange(3, dtype=float)
    offsets = np.linspace(-0.24, 0.24, len(MODELS))
    for mi, model in enumerate(MODELS):
        group = summary[summary.model.eq(model) & summary.metric.eq(metric)].set_index("task").loc[list(TASKS)]
        xpos = x + offsets[mi]
        ax.errorbar(xpos, group["mean"], yerr=group["std"], fmt="o", markersize=6.5,
                    capsize=4, linewidth=1.7, color=MODEL_COLORS[model], label=model, zorder=4)
        for si, seed in enumerate(SEEDS):
            sub = df[df.model.eq(model) & df.seed.eq(seed)].set_index("task").loc[list(TASKS)]
            ax.scatter(xpos + (si - 1) * 0.026, sub[metric], s=20, color=MODEL_COLORS[model], alpha=0.45, zorder=3)
    metric_summary = summary[summary.metric.eq(metric)]
    vals = np.concatenate([df[metric].to_numpy(float),
                           (metric_summary["mean"] - metric_summary["std"]).to_numpy(float),
                           (metric_summary["mean"] + metric_summary["std"]).to_numpy(float)])
    pad = max(0.005, float(np.ptp(vals)) * 0.18)
    # Mean ± SD can extend slightly outside a metric's bounded value range.
    # Show the entire whisker rather than clipping it at an old fixed axis.
    ax.set_ylim(float(vals.min()) - pad, float(vals.max()) + pad)
    ax.set_xticks(x, [TASK_NAMES[t].replace(" ", "\n", 1) for t in TASKS], fontsize=13)
    ax.set_xlim(-0.55, 2.55)
    ax.set_xlabel("Classification task")
    ax.set_ylabel(METRIC_NAMES[metric])
    ax.yaxis.set_major_locator(MaxNLocator(6))
    ax.grid(axis="y", alpha=0.22, linewidth=0.8)
    if with_legend:
        ax.legend(frameon=False, fontsize=11, loc="best", ncol=2)
    _axis(ax)


def plot_overall(root, dpi=600):
    df = load_overall(root)
    summary = summarize_overall(df)
    paths = [_csv(df, Path(root) / "plot_data" / "Figure3B_per_seed.csv"),
             _csv(summary, Path(root) / "plot_data" / "Figure3B_mean_sd.csv")]
    for metric in METRIC_NAMES:
        fig, ax = plt.subplots(figsize=OVERALL_SINGLE_FIGSIZE)
        draw_overall(ax, df, summary, metric)
        fig.tight_layout()
        paths.extend(_save(fig, Path(root) / "figures" / ("Figure3B_" + metric), dpi))
    fig, axes = plt.subplots(2, 2, figsize=OVERALL_SUMMARY_FIGSIZE)
    for ax, metric in zip(axes.flat, METRIC_NAMES):
        draw_overall(ax, df, summary, metric, with_legend=False)
    handles, names = axes.flat[0].get_legend_handles_labels()
    fig.legend(handles, names, loc="lower center", bbox_to_anchor=(.5, .005),
               ncol=4, frameon=False, fontsize=11)
    fig.tight_layout(rect=(0, .06, 1, 1))
    paths.extend(_save(fig, Path(root) / "figures" / "Figure3B_all_metrics", dpi))
    return paths


def _count_ratio(accuracy, support, correct):
    """Recover v1.0.3's integer numerator or use the explicit v1.0.4 count."""
    n = int(support)
    expected = float(accuracy) * n
    if correct is None:
        correct = round(expected)
    if not math.isfinite(correct) or int(correct) != correct or not 0 <= correct <= n:
        raise ValueError(f"Invalid correct count {correct} for {n} windows")
    if not math.isclose(expected, correct, rel_tol=0, abs_tol=1e-7):
        raise ValueError(f"Accuracy {accuracy} is inconsistent with {correct}/{n} windows")
    return Fraction(int(correct), n)


def _exact_motif_mean(row):
    if hasattr(row, "seed_counts"):
        return sum(_count_ratio(*counts) for counts in row.seed_counts) / row.n_seeds
    # Older exported summaries also determine the exact mean when every seed
    # tested the same number of windows, as in the published dataset.
    if row.n_samples_min == row.n_samples_max:
        return _count_ratio(row.accuracy_mean, row.n_samples_min * row.n_seeds, None)
    raise ValueError("Seed supports differ; use the per-seed counts from per_motif_metrics.csv")


def build_ecdf(aggregated):
    complete = aggregated[aggregated.complete_seeds].copy()
    observed_sets = [set(aggregated.loc[aggregated.task.eq(task), "motif"]) for task in TASKS]
    sets = [set(complete.loc[complete.task.eq(task), "motif"]) for task in TASKS]
    common = set.intersection(*sets)
    if not common:
        raise ValueError("No backgrounds with all three seeds shared by all three tasks")
    records, summaries = [], []
    for task in TASKS:
        z = complete[complete.task.eq(task) & complete.motif.isin(common)].copy()
        exact = [_exact_motif_mean(row) for row in z.itertuples(index=False)]
        threshold = Fraction(19, 20)
        n_ge = sum(value >= threshold for value in exact)
        n_equal = sum(value == threshold for value in exact)
        z["accuracy_mean"] = [float(value) for value in exact]
        z = z.sort_values(["accuracy_mean", "motif"])
        values = z.accuracy_mean.to_numpy(float)
        for rank, row in enumerate(z.itertuples(index=False), 1):
            records.append({"task": task, "motif": row.motif, "accuracy": row.accuracy_mean,
                            "cumulative_motif_fraction": rank / len(z), "n_motifs": len(z)})
        summaries.append({"task": task, "n_motifs": len(z), "median": float(np.median(values)),
                          "minimum": float(values.min()), "q1": float(np.quantile(values, .25)),
                          "q3": float(np.quantile(values, .75)), "n_ge_0.95": n_ge,
                          "n_eq_0.95": n_equal, "proportion_ge_0.95": n_ge / len(z),
                          "n_motifs_observed": len(observed_sets[TASKS.index(task)]),
                          "n_complete_seed_backgrounds": len(sets[TASKS.index(task)]),
                          "excluded_noncommon_or_incomplete_backgrounds": len(observed_sets[TASKS.index(task)] - common)})
    return pd.DataFrame(records), pd.DataFrame(summaries)


def plot_ecdf(root, dpi=600):
    motifs = load_motifs(root)
    ecdf, summaries = build_ecdf(aggregate_motifs(motifs))
    paths = [_csv(ecdf, Path(root) / "plot_data" / "Figure3C_ECDF.csv"),
             _csv(summaries, Path(root) / "plot_data" / "Figure3C_summary.csv")]
    per_seed = []
    common = set(ecdf.motif)
    for (task, seed), group in motifs[motifs.motif.isin(common)].groupby(["task", "seed"], observed=True):
        group = group.sort_values(["accuracy", "motif"]).copy()
        group["cumulative_motif_fraction"] = np.arange(1, len(group) + 1) / len(group)
        per_seed.append(group)
    paths.append(_csv(pd.concat(per_seed, ignore_index=True), Path(root) / "plot_data" / "Figure3C_ECDF_per_seed.csv"))
    fig, ax = plt.subplots(figsize=ECDF_FIGSIZE)
    for task in TASKS:
        z = ecdf[ecdf.task.eq(task)]
        stat = summaries[summaries.task.eq(task)].iloc[0]
        legend = f"{TASK_NAMES[task]}\nmedian={stat['median']:.3f}, ≥0.95={100*stat['proportion_ge_0.95']:.1f}%"
        ax.step(z.accuracy, z.cumulative_motif_fraction, where="post", linewidth=2.0, color=TASK_COLORS[task], label=legend)
    for threshold in (.90, .95, .98):
        ax.axvline(threshold, color="0.65", linestyle="--", linewidth=.8, zorder=0)
    minimum = float(ecdf.accuracy.min())
    ax.set_xlim(max(0, min(.85, minimum - .02)), 1.005)
    ax.set_ylim(0, 1.01)
    ax.set_xlabel("Per-motif accuracy")
    ax.set_ylabel("Cumulative motif proportion")
    ax.legend(frameon=False, loc="upper left", fontsize=10.5)
    ax.grid(alpha=.2)
    _axis(ax)
    fig.tight_layout()
    paths.extend(_save(fig, Path(root) / "figures" / "Figure3C_motif_accuracy_ECDF_GatedConv", dpi))
    return paths


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


def plot_paired(root, dpi=600):
    df = load_paired(root)
    averaged = aggregate_paired(df)
    paths = [_csv(averaged, Path(root) / "plot_data" / "Figure3EF_paired_motif_mean_sd.csv")]
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
        paths.extend(_save(fig, Path(root) / "figures" / ("Figure3" + letter + "_paired_GatedConv_" + task), dpi))
    paths.append(_csv(pd.DataFrame(stats), Path(root) / "plot_data" / "Figure3EF_comparison_summary.csv"))
    fig, axes = plt.subplots(1, 2, figsize=(15.6, 6.5))
    for ax, task in zip(axes, ("5X", "5Y")):
        sc = draw_paired(ax, averaged, task, limits, color_limit)
        cbar = fig.colorbar(sc, ax=ax, fraction=.046, pad=.025)
        cbar.set_label("Accuracy difference")
    fig.tight_layout()
    paths.extend(_save(fig, Path(root) / "figures" / "Figure3EF_combined_paired_GatedConv", dpi))
    notes = ("Each point is a masked 6mer, using the same admitted test windows for both models. Values are equal-weight means across seeds 42/2024/2025. "
             "The six-class model retains six-way argmax: predicting the excluded class remains an error. "
             "Different original training-set sizes remain; this is a comparison of the trained systems, not a causal class-count-only ablation.\n")
    path = Path(root) / "plot_data" / "Figure3EF_caption_notes.txt"
    path.write_text(notes, encoding="utf-8")
    paths.append(str(path))
    return paths


def plot_descriptive_comparisons(root, dpi=600):
    """Retain the original task-specific analysis with an explicit scope label.

    These files never receive the strict paired Figure3E/F output names.
    Matching motif names does not establish matching underlying observations.
    """
    motifs = aggregate_motifs(load_motifs(root), retain_seed_counts=False)
    motifs = motifs[motifs.complete_seeds].copy()
    base = motifs[motifs.task.eq("6-class")][["motif", "accuracy_mean", "n_samples_min", "n_samples_max"]].rename(
        columns={"accuracy_mean": "accuracy_six", "n_samples_min": "six_n_samples_min", "n_samples_max": "six_n_samples_max"})
    comparisons = []
    for task in ("5X", "5Y"):
        other = motifs[motifs.task.eq(task)][["motif", "accuracy_mean", "n_samples_min", "n_samples_max"]].rename(
            columns={"accuracy_mean": "accuracy_five", "n_samples_min": "five_n_samples_min", "n_samples_max": "five_n_samples_max"})
        z = base.merge(other, on="motif", how="inner", validate="one_to_one")
        if z.empty:
            raise ValueError("No common motif names for descriptive task comparison: " + task)
        z["comparison"] = task
        z["difference_five_minus_six"] = z.accuracy_five - z.accuracy_six
        z["scope"] = "task_specific_test_sets_not_window_paired"
        comparisons.append(z)
    combined = pd.concat(comparisons, ignore_index=True)
    paths = [_csv(combined, Path(root) / "plot_data" / "Descriptive_task_specific_background_comparisons.csv")]
    lower = max(0, min(.85, float(combined[["accuracy_six", "accuracy_five"]].min().min()) - .02))
    color_limit = max(.01, float(np.abs(combined.difference_five_minus_six).max()))
    for task in ("5X", "5Y"):
        fig, ax = plt.subplots(figsize=(7.2, 6))
        sc = draw_paired(ax, combined, task, (lower, 1.005), color_limit)
        ax.set_title("Task-specific test sets · descriptive", fontsize=14, pad=12)
        colorbar = fig.colorbar(sc, ax=ax, fraction=.046, pad=.025)
        colorbar.set_label("Accuracy difference")
        fig.tight_layout()
        paths.extend(_save(fig, Path(root) / "figures" / ("Descriptive_task_specific_6class_vs_" + task + "_GatedConv"), dpi))
    notes = ("Original-style descriptive comparisons across task-specific test sets. Each point matches only the masked 6mer name. "
             "Sample counts, class definitions, and training set sizes differ; these are not same-window paired evaluations and must not be called strict paired ablations. "
             "Values are equal-weight means across the three predeclared seeds. No causal class-count-only improvement claim is supported.\n")
    path = Path(root) / "plot_data" / "Descriptive_task_specific_caption_notes.txt"
    path.write_text(notes, encoding="utf-8")
    paths.append(str(path))
    return paths


def load_confusions(root):
    df = _read(root, "confusion_counts.csv", ["task", "model", "seed", "true_label", "pred_label", "count"])
    df = df[df.model.eq("GatedConv")].copy()
    _numeric(df, ["count"], nonnegative=True)
    if not np.equal(df["count"], np.floor(df["count"])).all():
        raise ValueError("Confusion counts must be integers")
    _unique(df, ["task", "model", "seed", "true_label", "pred_label"], "confusion_counts")
    if set(df.task) != set(TASKS):
        raise ValueError("Confusion counts must include all three GatedConv tasks")
    _require_seeds(df, ["task"], "confusion_counts")
    for task, g in df.groupby("task", observed=True):
        allowed = set(TASK_LABELS[task])
        if not set(g.true_label.astype(str)).issubset(allowed) or not set(g.pred_label.astype(str)).issubset(allowed):
            raise ValueError(f"{task}: confusion labels do not match biological label order")
    return df


def confusion_matrix_for(df, task, seed=None):
    labels = TASK_LABELS[task]
    z = df[df.task.eq(task)]
    if seed is not None:
        z = z[z.seed.eq(seed)]
    matrix = np.zeros((len(labels), len(labels)), dtype=np.int64)
    indices = {name: i for i, name in enumerate(labels)}
    for row in z.itertuples(index=False):
        matrix[indices[str(row.true_label)], indices[str(row.pred_label)]] += int(row.count)
    if np.any(matrix.sum(axis=1) == 0):
        raise ValueError(f"{task}: true class has zero support in confusion matrix")
    return matrix


def class_metrics_from_confusion(matrix, labels):
    matrix = np.asarray(matrix, dtype=float)
    tp = np.diag(matrix)
    support, predicted = matrix.sum(axis=1), matrix.sum(axis=0)
    precision = np.divide(tp, predicted, out=np.zeros_like(tp), where=predicted > 0)
    recall = np.divide(tp, support, out=np.zeros_like(tp), where=support > 0)
    f1 = np.divide(2 * precision * recall, precision + recall, out=np.zeros_like(tp), where=precision + recall > 0)
    return pd.DataFrame({"class": labels, "precision": precision, "recall": recall, "f1": f1, "support": support.astype(np.int64)})


def plot_confusions(root, tasks=TASKS, dpi=600):
    df = load_confusions(root)
    paths = []
    figure_numbers = {"6-class": "S8", "5X": "S9", "5Y": "S10"}
    table_numbers = {"6-class": "S3", "5X": "S4", "5Y": "S5"}
    for task in tasks:
        labels = TASK_LABELS[task]
        pooled = confusion_matrix_for(df, task)
        normalized = pooled / pooled.sum(axis=1, keepdims=True) * 100
        all_seed_metrics = []
        for seed in SEEDS:
            cm = confusion_matrix_for(df, task, seed)
            table = class_metrics_from_confusion(cm, labels)
            table["seed"] = seed
            table["task"] = task
            all_seed_metrics.append(table)
        per_seed = pd.concat(all_seed_metrics, ignore_index=True)
        stats = per_seed.groupby("class", sort=False).agg(
            precision_mean=("precision", "mean"), precision_sd=("precision", "std"),
            recall_mean=("recall", "mean"), recall_sd=("recall", "std"),
            f1_mean=("f1", "mean"), f1_sd=("f1", "std"),
            support_per_seed_min=("support", "min"), support_per_seed_max=("support", "max"),
            n_seeds=("seed", "nunique")).reindex(labels).reset_index()
        publication = pd.DataFrame({"Class": stats["class"]})
        for metric, label in (("precision", "Precision"), ("recall", "Recall"), ("f1", "F1")):
            publication[label + " (mean ± SD)"] = [f"{mean:.4f} ± {sd:.4f}" for mean, sd in zip(stats[metric + "_mean"], stats[metric + "_sd"])]
        publication["Support per seed"] = [str(int(low)) if low == high else f"{int(low)}–{int(high)}"
                                                   for low, high in zip(stats.support_per_seed_min, stats.support_per_seed_max)]
        fig, ax = plt.subplots(figsize=(9.2, 8.0))
        image = ax.imshow(normalized, cmap="Blues", vmin=0, vmax=100, interpolation="nearest")
        ax.set_xticks(range(len(labels)), labels, fontsize=25)
        ax.set_yticks(range(len(labels)), labels, fontsize=25)
        ax.set_xlabel("Predicted class", fontsize=28, labelpad=16)
        ax.set_ylabel("True class", fontsize=28, labelpad=16)
        ax.set_xticks(np.arange(-.5, len(labels), 1), minor=True)
        ax.set_yticks(np.arange(-.5, len(labels), 1), minor=True)
        ax.grid(which="minor", color="white", linewidth=1.5)
        ax.tick_params(which="minor", bottom=False, left=False)
        for row in range(len(labels)):
            for column in range(len(labels)):
                value = normalized[row, column]
                ax.text(column, row, f"{value:.2f}", ha="center", va="center", fontsize=18,
                        fontweight="bold", color="white" if value >= 50 else "black")
        cbar = fig.colorbar(image, ax=ax, fraction=.046, pad=.035)
        cbar.set_label("Row-normalized percentage (%)", fontsize=20)
        cbar.ax.tick_params(labelsize=18)
        fig.tight_layout()
        prefix = "Supplementary_Figure_" + figure_numbers[task]
        paths.extend(_save(fig, Path(root) / "figures" / (prefix + "_GatedConv_" + task), dpi))
        table_prefix = "Supplementary_Table_" + table_numbers[task]
        paths.append(_csv(per_seed, Path(root) / "plot_data" / (table_prefix + "_per_seed.csv")))
        paths.append(_csv(stats, Path(root) / "plot_data" / (table_prefix + "_mean_sd.csv")))
        paths.append(_csv(publication, Path(root) / "plot_data" / (table_prefix + "_publication.csv")))
        for kind, values in (("pooled_counts", pooled), ("row_normalized_percent", normalized)):
            path = Path(root) / "plot_data" / (prefix + "_" + kind + ".csv")
            table = pd.DataFrame(values, index=labels, columns=labels)
            table.index.name = "true_label"
            table.to_csv(path)
            paths.append(str(path))
        notes = (f"{prefix}: counts are pooled predictions from the same fixed {task} test set evaluated with three independently trained seeds. "
                 "A test observation can appear in all three evaluations; pooled support is not a count of independent reads. "
                 "The figure is row-normalized after pooling. Tables report per-seed precision/recall/F1 followed by mean ± sample SD (ddof=1), with test support per seed.\n")
        note_path = Path(root) / "plot_data" / (prefix + "_caption_notes.txt")
        note_path.write_text(notes, encoding="utf-8")
        paths.append(str(note_path))
        try:
            xlsx_path = Path(root) / "plot_data" / (table_prefix + ".xlsx")
            with pd.ExcelWriter(xlsx_path, engine="openpyxl") as writer:
                publication.to_excel(writer, sheet_name="Publication", index=False)
                stats.to_excel(writer, sheet_name="Mean_SD", index=False)
                per_seed.to_excel(writer, sheet_name="Per_seed", index=False)
                pd.DataFrame({"Note": [notes]}).to_excel(writer, sheet_name="Read_me", index=False)
                for sheet in writer.book.worksheets:
                    from openpyxl.styles import Alignment, Font
                    sheet.freeze_panes = "B2"
                    sheet.auto_filter.ref = sheet.dimensions
                    for row in sheet.iter_rows():
                        for cell in row:
                            cell.font = Font(name="Times New Roman", size=11, bold=cell.row == 1)
                            cell.alignment = Alignment(horizontal="left" if cell.column == 1 else "center", vertical="center", wrap_text=True)
                    sheet.row_dimensions[1].height = 32
                    for column in sheet.columns:
                        letter = column[0].column_letter
                        sheet.column_dimensions[letter].width = min(50, max(14, max(len(str(c.value or "")) for c in column[:30]) + 2))
            paths.append(str(xlsx_path))
        except ImportError:
            # Complete CSV tables are always written; XLSX is an optional format.
            pass
    return paths


def _embedding_input(root, task):
    path = Path(root) / "data" / "embeddings" / (task + "_seed42.npz")
    if not path.is_file():
        raise MissingInput(f"缺少 {path}；请先从已训练的最佳检查点导出测试表征，无需重训。")
    with np.load(path, allow_pickle=False) as data:
        if not {"features", "labels"}.issubset(data.files):
            raise ValueError(f"{path.name} must contain features and labels")
        features = np.array(data["features"], dtype=np.float32, copy=True)
        labels = np.array(data["labels"], copy=True)
        source_indices = np.array(data["sample_indices"], copy=True) if "sample_indices" in data.files else np.arange(len(labels))
    if features.ndim != 2 or not len(features) or labels.ndim != 1 or len(features) != len(labels):
        raise ValueError(f"{path.name}: invalid features/labels dimensions")
    if not np.isfinite(features).all() or not np.isfinite(labels).all():
        raise ValueError(f"{path.name}: nonfinite embedding inputs")
    if not np.equal(labels, np.floor(labels)).all():
        raise ValueError("Embedding labels must be integers")
    labels = labels.astype(np.int64)
    if set(labels) != set(range(len(TASK_LABELS[task]))):
        raise ValueError(f"{task}: all biological classes must be present using local label order")
    if source_indices.ndim != 1 or len(source_indices) != len(labels):
        raise ValueError("sample_indices do not align to features")
    return path, features, labels, source_indices


def _stratified_indices(labels, max_per_class=5000, seed=42):
    rng = np.random.default_rng(seed)
    selected = []
    for label in sorted(set(labels.tolist())):
        indices = np.flatnonzero(labels == label)
        if max_per_class > 0 and len(indices) > max_per_class:
            indices = np.sort(rng.choice(indices, max_per_class, replace=False))
        selected.extend(indices.tolist())
    return np.asarray(selected, dtype=np.int64)


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


def replot_saved_umaps(root, tasks=TASKS, dpi=600):
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
        paths.extend(_save(fig, Path(root) / "figures" / (prefix + "_GatedConv_seed42"), dpi))
    return {"produced": paths, "missing": missing}


def plot_umaps(root, tasks=TASKS, dpi=600):
    try:
        import umap
        from sklearn.preprocessing import StandardScaler
    except ImportError as exc:
        raise MissingInput("UMAP 需要 umap-learn 和 scikit-learn。请在运行环境安装 requirements.txt 中的依赖后重跑 UMAP 入口；统计图不受影响。") from exc
    paths = []
    for task in tasks:
        input_path, features, labels, indices = _embedding_input(root, task)
        keep = _stratified_indices(labels)
        features, labels, indices = features[keep], labels[keep], indices[keep]
        if len(labels) < 4:
            raise ValueError("Too few observations for UMAP")
        # Display-only unsupervised normalization; never used for classifier fitting.
        scaled = StandardScaler().fit_transform(features)
        reducer = umap.UMAP(n_neighbors=min(10, len(labels) - 1), min_dist=.3, spread=1.5,
                            repulsion_strength=1.5, random_state=42, n_jobs=1)
        embedding = reducer.fit_transform(scaled)  # Never pass class labels to fit.
        if embedding.shape != (len(labels), 2) or not np.isfinite(embedding).all():
            raise ValueError("UMAP produced invalid coordinates")
        prefix = "Figure3D" if task == "6-class" else "Supplementary_UMAP_" + task
        fig = draw_umap_coordinates(embedding, labels, task)
        paths.extend(_save(fig, Path(root) / "figures" / (prefix + "_GatedConv_seed42"), dpi))
        table = pd.DataFrame({"sample_index": indices, "label": labels,
                              "class": [TASK_LABELS[task][v] for v in labels],
                              "umap_1": embedding[:, 0], "umap_2": embedding[:, 1]})
        paths.append(_csv(table, Path(root) / "plot_data" / (prefix + "_coordinates.csv")))
        npz_path = Path(root) / "plot_data" / (prefix + "_coordinates.npz")
        np.savez_compressed(npz_path, embedding=embedding, labels=labels, sample_indices=indices)
        paths.append(str(npz_path))
        summary_path = Path(root) / "plot_data" / (prefix + "_manifest.json")
        _write_json(summary_path, {"task": task, "model": "GatedConv", "checkpoint_seed": 42,
                                   "input": str(input_path), "n_points": len(labels),
                                   "points_by_class": {name: int(np.sum(labels == label)) for label, name in enumerate(TASK_LABELS[task])},
                                   "umap_version": getattr(umap, "__version__", "unknown"),
                                   "parameters": {"random_state": 42, "n_neighbors": min(10, len(labels) - 1), "min_dist": .3,
                                                  "spread": 1.5, "repulsion_strength": 1.5, "n_jobs": 1},
                                   "labels_used_to_fit_umap": False,
                                   "interpretation": "Qualitative visualization of test-set classifier features. Each task has an independently fitted map; axes and intercluster distances cannot be compared across tasks. Display-only scaler is not a classifier preprocessing fit."})
        paths.append(str(summary_path))
    return paths


def _execute(root, jobs, dpi):
    root = Path(root).expanduser().resolve()
    (root / "figures").mkdir(parents=True, exist_ok=True)
    (root / "plot_data").mkdir(parents=True, exist_ok=True)
    font = configure_style()
    result = {"produced": [], "missing": [], "errors": [], "font": font}
    for name, callback in jobs:
        try:
            result["produced"].extend(callback())
        except MissingInput as exc:
            result["missing"].append({"panel": name, "reason": str(exc)})
        except Exception as exc:
            result["errors"].append({"panel": name, "reason": type(exc).__name__ + ": " + str(exc)})
    result["produced"] = list(dict.fromkeys(result["produced"]))
    result["seeds"] = list(SEEDS)
    result["dpi"] = dpi
    result["main_decision_rule"] = "unthresholded argmax; raw classifier output"
    _write_json(root / "plot_data" / "plot_manifest.json", result)
    return result


def run_all_plots(result_root, skip_umap=False, dpi=None, heatmap_vmin=None, heatmap_vmax=1.0):
    """Generate every admitted panel; explicitly report unavailable E/F/UMAP."""
    root = Path(result_root).expanduser().resolve()
    dpi = int(dpi if dpi is not None else os.environ.get("FIGURE3_DPI", 600))
    jobs = [("A and supplementary heatmaps", lambda: plot_heatmaps(root, dpi=dpi, heatmap_vmin=heatmap_vmin, heatmap_vmax=heatmap_vmax)),
            ("B", lambda: plot_overall(root, dpi=dpi)),
            ("C", lambda: plot_ecdf(root, dpi=dpi)),
            ("Descriptive task-specific E/F analysis", lambda: plot_descriptive_comparisons(root, dpi=dpi)),
            ("E/F", lambda: plot_paired(root, dpi=dpi)),
            ("S8/S9/S10 and S3/S4/S5 tables", lambda: plot_confusions(root, dpi=dpi))]
    if not skip_umap:
        for task in TASKS:
            jobs.append(("UMAP " + task, lambda task=task: plot_umaps(root, tasks=[task], dpi=dpi)))
    return _execute(root, jobs, dpi)


def main(group="all", argv=None):
    parser = argparse.ArgumentParser(description="使用当前冻结测试结果绘制 Figure3／补充图；不训练模型。")
    parser.add_argument("--result-root", "--result_root", type=Path, default=DEFAULT_RESULT_ROOT)
    parser.add_argument("--dpi", type=int, default=int(os.environ.get("FIGURE3_DPI", 600)))
    parser.add_argument("--task", choices=["both", "x", "y"], default="both", help="五分类脚本可选择 X/Y")
    parser.add_argument("--skip-umap", action="store_true")
    parser.add_argument("--heatmap-vmin", type=float, default=None,
                        help="Three heatmaps share this lower bound; default rounds the worst individual seed down to 0.05")
    parser.add_argument("--heatmap-vmax", type=float, default=1.0, help="Heatmap upper bound (default: 1.0)")
    args = parser.parse_args(argv)
    if args.dpi < 36:
        parser.error("dpi must be at least 36")
    root, dpi = args.result_root.expanduser().resolve(), args.dpi
    five = ["5X", "5Y"] if args.task == "both" else ["5X" if args.task == "x" else "5Y"]
    # Every individual entry has complete defaults; the bundled result snapshot
    # supplies tables on first run, rather than requiring manual CSV assembly.
    if not (root / "data" / "overall_metrics.csv").is_file():
        from collect_results import collect_results
        collect_results(root)
    if group in ("umap6", "umap5"):
        from export_embeddings import export_embeddings
        specs = json.loads((root / "data" / "run_specs.json").read_text(encoding="utf-8"))
        for task in (["6-class"] if group == "umap6" else five):
            output = root / "data" / "embeddings" / (task + "_seed42.npz")
            spec = next(s for s in specs if s["task"] == task and s["model"] == "GatedConv" and int(s["seed"]) == 42)
            export_embeddings(spec, output)
    if group == "bcef" and not (root / "data" / "paired_motif_metrics.csv").is_file() and not (root / "paired_audit.json").is_file():
        from paired_evaluation import prepare_paired_metrics
        specs = json.loads((root / "data" / "run_specs.json").read_text(encoding="utf-8"))
        prepare_paired_metrics(specs, root)
    groups = {
        "bcef": [("B", lambda: plot_overall(root, dpi)), ("C", lambda: plot_ecdf(root, dpi)),
                 ("Descriptive task-specific E/F analysis", lambda: plot_descriptive_comparisons(root, dpi)),
                 ("E/F", lambda: plot_paired(root, dpi))],
        "heatmaps": [("A and supplementary heatmaps", lambda: plot_heatmaps(root, TASKS, dpi, args.heatmap_vmin, args.heatmap_vmax))],
        "heatmap6": [("A", lambda: plot_heatmaps(root, ["6-class"], dpi, args.heatmap_vmin, args.heatmap_vmax))],
        "heatmap5": [("supplementary five-class heatmaps", lambda: plot_heatmaps(root, five, dpi, args.heatmap_vmin, args.heatmap_vmax))],
        "confusion6": [("S8 and table S3", lambda: plot_confusions(root, ["6-class"], dpi))],
        "confusion5": [("S9/S10 and tables S4/S5", lambda: plot_confusions(root, five, dpi))],
        "umap6": [("D", lambda: plot_umaps(root, ["6-class"], dpi))],
        "umap5": [("UMAP " + task, lambda task=task: plot_umaps(root, [task], dpi)) for task in five],
    }
    result = run_all_plots(root, args.skip_umap, dpi, args.heatmap_vmin, args.heatmap_vmax) if group == "all" else _execute(root, groups[group], dpi)
    print(json.dumps(result, ensure_ascii=False, indent=2))
    return 2 if result["errors"] or result["missing"] else 0


if __name__ == "__main__":
    raise SystemExit(main())
