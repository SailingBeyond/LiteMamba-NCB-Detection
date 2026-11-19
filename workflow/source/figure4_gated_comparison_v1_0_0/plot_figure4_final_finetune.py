#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""从新版本离线统计表重绘 Figure 4，保留全部模型和全部已计划种子。

本脚本只读取 analysis_data 中的 CSV，不加载模型或重新预测。默认绘制
新六分类任务；只有显式选择 --tasks auto 时，才发现其他已有任务表。
图中的误差线表示训练种子间样本标准差，不表示读段 bootstrap 置信区间。
"""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
from typing import Any, Sequence

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.colors import LinearSegmentedColormap
from matplotlib.lines import Line2D
from matplotlib.ticker import PercentFormatter
import numpy as np
import pandas as pd


VERSION = "figure4-offline-plots-v1.0.0"
TASK_CONFIG = {
    "6class": {"false_label": "False X/Y call rate", "recall_label": "Target X/Y recall"},
    "5X": {"false_label": "False X call rate", "recall_label": "Target X recall"},
    "5Y": {"false_label": "False Y call rate", "recall_label": "Target Y recall"},
}
TABLE_PATTERNS = {
    "a": "figure4A_dna01_16_false_call_heatmap_data_{slug}.csv",
    "b": "figure4B_xna01_16_target_recall_lollipop_data_{slug}.csv",
    "c_detail": "figure4C_xna17_20_residual_error_detail_{slug}.csv",
    "c_summary": "figure4C_xna17_20_residual_error_summary_{slug}.csv",
    "d": "figure4D_external_validation_profile_data_{slug}.csv",
}
MODEL_ORDER = ["litemamba", "transformer", "bigru", "convmixer"]
MODEL_DISPLAY = {"litemamba": "LiteMamba", "transformer": "Transformer",
                 "bigru": "BiGRU", "convmixer": "ConvMixer"}
MODEL_COLORS = {"litemamba": "#1B8F74", "transformer": "#676E75",
                "bigru": "#9EA4AA", "convmixer": "#41474D"}
MODEL_MARKERS = {"litemamba": "o", "transformer": "s", "bigru": "^", "convmixer": "D"}
RESIDUAL_ORDER = ["A", "T", "C", "G", "opposite NCB"]
RESIDUAL_COLORS = {"A": "#A6BDD3", "T": "#DCC5A3", "C": "#B6CCB6",
                   "G": "#C4B8D3", "opposite NCB": "#D99B91"}
HEATMAP = LinearSegmentedColormap.from_list(
    "canonical_false_calls", ["#FFFDF6", "#F4E5B5", "#DBAE68", "#B97451", "#874748"]
)
HEATMAP.set_bad("#E6E9EC")
SD_NOTE = "Mean across training seeds; error bars: sample SD, not a confidence interval."


def configure_style() -> None:
    plt.rcParams.update({
        "font.family": "sans-serif", "font.sans-serif": ["Arial", "DejaVu Sans"],
        "font.size": 10, "axes.labelsize": 11, "xtick.labelsize": 9,
        "ytick.labelsize": 9, "legend.fontsize": 9, "pdf.fonttype": 42,
        "ps.fonttype": 42, "axes.linewidth": 0.9,
    })


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def read_table(path: Path, required: Sequence[str]) -> pd.DataFrame:
    try:
        table = pd.read_csv(path)
    except pd.errors.EmptyDataError as error:
        raise ValueError(f"统计表没有表头，无法判断字段：{path}") from error
    missing = sorted(set(required) - set(table.columns))
    if missing:
        raise ValueError(f"{path.name} 缺少字段：{missing}")
    return table


def numeric(table: pd.DataFrame, column: str, *, rate: bool = False) -> np.ndarray:
    values = pd.to_numeric(table[column], errors="raise").to_numpy(dtype=float)
    if np.isinf(values).any():
        raise ValueError(f"{column} 含无限值")
    finite = values[np.isfinite(values)]
    if rate and ((finite < -1e-10).any() or (finite > 1 + 1e-10).any()):
        raise ValueError(f"{column} 必须为 0–1 的比例，不能输入百分数")
    return values


def seed_sd(table: pd.DataFrame, column: str = "seed_sd") -> np.ndarray:
    if column not in table:
        return np.full(len(table), np.nan)
    values = numeric(table, column)
    if (values[np.isfinite(values)] < 0).any():
        raise ValueError(f"{column} 不能为负数")
    if "n_seeds" in table:
        n = pd.to_numeric(table.n_seeds, errors="raise").to_numpy(dtype=float)
        values = values.copy()
        values[n < 2] = np.nan
    return values


def primary_model(table: pd.DataFrame) -> pd.DataFrame:
    result = table.copy()
    if "probability_type" in result:
        result = result[result.probability_type.eq("raw")].copy()
    if "model" in result:
        result = result[result.model.eq("litemamba")].copy()
    return result


def require_unique(table: pd.DataFrame, keys: Sequence[str], name: str) -> None:
    if table.duplicated(list(keys)).any():
        raise ValueError(f"{name} 在 {list(keys)} 上重复；请先按全部计划种子汇总，绘图不会挑选一行")


def template_numbers(table: pd.DataFrame, expected: Sequence[int]) -> pd.DataFrame:
    table = table.copy()
    if len(table):
        numbers = pd.to_numeric(table.template_num, errors="raise")
        if not np.equal(numbers, np.floor(numbers)).all():
            raise ValueError("template_num 必须为整数")
        table["template_num"] = numbers.astype(int)
        if not table.template_num.isin(expected).all():
            raise ValueError(f"模板编号超出本面板范围：{expected}")
    require_unique(table, ["template_num"], "template table")
    return table.set_index("template_num").reindex(expected)


def pct(value: float, decimals: int = 2) -> str:
    return "NA" if not np.isfinite(value) else f"{value * 100:.{decimals}f}%"


def annotate_missing(ax: plt.Axes, x: float, *, y: float = 0.025) -> None:
    ax.text(x, y, "NA", ha="center", va="bottom", color="#777F87", fontsize=8,
            transform=ax.get_xaxis_transform())


def style_axis(ax: plt.Axes) -> None:
    ax.spines[["top", "right"]].set_visible(False)
    ax.grid(axis="y", color="#DBE0E3", linestyle="--", linewidth=0.6, alpha=0.8)
    ax.set_axisbelow(True)


def sd_caption(fig: plt.Figure) -> None:
    fig.text(0.5, 0.012, SD_NOTE, ha="center", va="bottom", fontsize=8, color="#59656E")


def save_figure(fig: plt.Figure, base: Path, formats: Sequence[str], dpi: int,
                *, bottom: float = 0.13) -> list[str]:
    base.parent.mkdir(parents=True, exist_ok=True)
    fig.tight_layout(rect=(0, bottom, 1, 1))
    outputs = []
    try:
        for fmt in formats:
            path = base.with_suffix("." + fmt)
            fig.savefig(path, dpi=dpi if fmt == "png" else None, bbox_inches="tight")
            outputs.append(str(path))
    finally:
        plt.close(fig)
    return outputs


def plot_figure4a(table: pd.DataFrame, slug: str, base: Path,
                  formats: Sequence[str], dpi: int) -> list[str]:
    table = template_numbers(primary_model(table), range(1, 17))
    values = numeric(table, "false_noncanonical_call_rate", rate=True)
    deviations = seed_sd(table)
    matrix = values.reshape(4, 4)
    fig, ax = plt.subplots(figsize=(5.4, 5.0))
    im = ax.imshow(np.ma.masked_invalid(matrix), cmap=HEATMAP, vmin=0, vmax=1, aspect="equal")
    ax.set_xticks([])
    ax.set_yticks([])
    ax.set_xticks(np.arange(-0.5, 4, 1), minor=True)
    ax.set_yticks(np.arange(-0.5, 4, 1), minor=True)
    ax.grid(which="minor", color="white", linewidth=1.8)
    ax.tick_params(which="minor", bottom=False, left=False)
    ax.axhline(2.5, color="#343A40", linewidth=1.6)
    for number, value, deviation in zip(range(1, 17), values, deviations):
        i, j = divmod(number - 1, 4)
        label = f"DNA{number}\n{pct(value)}"
        if np.isfinite(value) and np.isfinite(deviation):
            label = f"DNA{number}\n{value * 100:.2f} ± {deviation * 100:.2f}%"
        ax.text(j, i, label, ha="center", va="center", fontsize=8.7,
                color="white" if np.isfinite(value) and value >= 0.65 else "#26343F")
    cbar = fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    cbar.set_label(TASK_CONFIG[slug]["false_label"])
    cbar.ax.yaxis.set_major_formatter(PercentFormatter(1.0))
    sd_caption(fig)
    return save_figure(fig, base, formats, dpi, bottom=0.075)


def plot_figure4b(table: pd.DataFrame, slug: str, base: Path,
                  formats: Sequence[str], dpi: int) -> list[str]:
    table = template_numbers(primary_model(table), range(1, 17))
    values = numeric(table, "target_recall_micro", rate=True)
    deviations = seed_sd(table)
    x = np.arange(16)
    known = np.isfinite(values)
    fig, ax = plt.subplots(figsize=(10.7, 3.55))
    ax.axvspan(-0.5, 11.5, color="#EFF5F8", zorder=0)
    ax.axvspan(11.5, 15.5, color="#F8F3E9", zorder=0)
    ax.axvline(11.5, color="#707A82", linewidth=1, linestyle="--", zorder=1)
    ax.vlines(x[known], 0, values[known], color=MODEL_COLORS["litemamba"], linewidth=1.7)
    ax.scatter(x[known], values[known], s=47, color=MODEL_COLORS["litemamba"],
               edgecolor="#29423B", linewidth=0.6, zorder=3)
    with_sd = known & np.isfinite(deviations)
    if with_sd.any():
        ax.errorbar(x[with_sd], values[with_sd], yerr=deviations[with_sd], fmt="none",
                    ecolor="#29423B", capsize=2.8, linewidth=1, zorder=4)
    for i in x[~known]:
        annotate_missing(ax, i)
    top = max(1.025, float(np.nanmax(values + np.nan_to_num(deviations))) + 0.03) if known.any() else 1.025
    ax.set(xlim=(-0.7, 15.7), ylim=(0, top), ylabel=TASK_CONFIG[slug]["recall_label"],
           xticks=x, xticklabels=[f"XNA{i}" for i in range(1, 17)])
    ax.tick_params(axis="x", rotation=45)
    for label in ax.get_xticklabels():
        label.set_horizontalalignment("right")
    ax.yaxis.set_major_formatter(PercentFormatter(1.0))
    style_axis(ax)
    sd_caption(fig)
    return save_figure(fig, base, formats, dpi, bottom=0.12)


def plot_figure4c(detail: pd.DataFrame, summary: pd.DataFrame, slug: str,
                  base: Path, formats: Sequence[str], dpi: int) -> list[str]:
    detail, summary = primary_model(detail), primary_model(summary)
    summary = template_numbers(summary, range(17, 21))
    if len(detail):
        detail = detail.copy()
        detail_numbers = pd.to_numeric(detail.template_num, errors="raise")
        if not np.equal(detail_numbers, np.floor(detail_numbers)).all():
            raise ValueError("Figure 4C template_num 必须为整数")
        detail["template_num"] = detail_numbers.astype(int)
        if not detail.template_num.isin(range(17, 21)).all():
            raise ValueError("Figure 4C 明细仅允许 XNA17–20")
        if not detail.error_type.isin(RESIDUAL_ORDER).all():
            raise ValueError("Figure 4C 含未知错误类型，绘图不会将其隐藏")
        require_unique(detail, ["template_num", "error_type"], "Figure 4C detail")
    totals = numeric(summary, "total_error_per_10000")
    recall = numeric(summary, "target_recall_micro", rate=True)
    deviations = seed_sd(summary)
    if (totals[np.isfinite(totals)] < 0).any():
        raise ValueError("残差计数不能为负")
    matrix = np.full((4, len(RESIDUAL_ORDER)), np.nan)
    for i, number in enumerate(range(17, 21)):
        sub = detail[detail.template_num.eq(number)].set_index("error_type")
        if np.isfinite(totals[i]):
            if set(sub.index) != set(RESIDUAL_ORDER):
                raise ValueError(f"XNA{number} 残差缺少分量；合法的零值也应由统计表显式保存")
            matrix[i] = pd.to_numeric(sub.reindex(RESIDUAL_ORDER).error_per_10000, errors="raise")
            if not np.isfinite(matrix[i]).all() or (matrix[i] < 0).any():
                raise ValueError(f"XNA{number} 残差分量无效")
            if not np.isclose(matrix[i].sum(), totals[i], atol=0.02, rtol=1e-6):
                raise ValueError(f"XNA{number} 堆叠分量与总残差不一致")
            if np.isfinite(recall[i]) and not np.isclose((1 - recall[i]) * 10000, totals[i], atol=0.02, rtol=1e-6):
                raise ValueError(f"XNA{number} 总残差与目标召回率不一致")
    known = np.isfinite(totals)
    fig, ax = plt.subplots(figsize=(6.4, 4.2))
    x, bottom = np.arange(4), np.zeros(4)
    for j, error_type in enumerate(RESIDUAL_ORDER):
        values = matrix[:, j]
        ax.bar(x[known], values[known], width=0.6, bottom=bottom[known],
               color=RESIDUAL_COLORS[error_type], edgecolor="white", linewidth=0.6,
               label=error_type)
        bottom[known] += values[known]
    with_sd = known & np.isfinite(deviations)
    if with_sd.any():
        ax.errorbar(x[with_sd], totals[with_sd], yerr=deviations[with_sd], fmt="none",
                    color="#34434D", capsize=3, linewidth=1.1)
    maximum = float(np.nanmax(totals + np.nan_to_num(deviations))) if known.any() else 1.0
    offset = max(maximum * 0.055, 5)
    for i in x:
        if known[i]:
            label = f"Recall = {recall[i] * 100:.2f}%" if np.isfinite(recall[i]) else "Recall = NA"
            ax.text(i, totals[i] + (deviations[i] if np.isfinite(deviations[i]) else 0) + offset,
                    label, ha="center", va="bottom", fontsize=8.7)
        else:
            annotate_missing(ax, i)
    ax.set(xticks=x, xticklabels=[f"XNA{i}" for i in range(17, 21)],
           ylabel="Misclassified calls per 10,000 target NCBs", ylim=(0, max(1.0, maximum + 4 * offset)))
    style_axis(ax)
    ax.legend(frameon=False, ncols=3, loc="upper center", bbox_to_anchor=(0.5, 1.25), fontsize=8.5)
    sd_caption(fig)
    return save_figure(fig, base, formats, dpi, bottom=0.1)


def profile_tables(table: pd.DataFrame, include_overall: bool) -> tuple[pd.DataFrame, pd.DataFrame]:
    table = table.copy()
    table["order"] = pd.to_numeric(table["order"], errors="raise")
    if "probability_type" in table:
        table = table[table.probability_type.eq("raw")].copy()
    if not include_overall:
        table = table[~table.metric_key.eq("overall")].copy()
    if len(table):
        unknown = set(table.model) - set(MODEL_ORDER)
        if unknown:
            raise ValueError(f"profile 含未识别模型，绘图不会忽略：{sorted(unknown)}")
    require_unique(table, ["model", "metric_key"], "Figure 4D profile")
    metrics = table[["metric_key", "metric_label", "order"]].drop_duplicates()
    require_unique(metrics, ["metric_key"], "Figure 4D metric definitions")
    if metrics.order.duplicated().any():
        raise ValueError("Figure 4D 指标 order 重复")
    return table, metrics.sort_values("order")


def plot_profile(table: pd.DataFrame, include_overall: bool, base: Path,
                 formats: Sequence[str], dpi: int) -> list[str]:
    table, metrics = profile_tables(table, include_overall)
    fig, ax = plt.subplots(figsize=(9.6, 4.9))
    keys = metrics.metric_key.tolist()
    if not keys:
        ax.text(0.5, 0.5, "No available profile metrics", transform=ax.transAxes, ha="center")
    for model in MODEL_ORDER:
        selected = table[table.model.eq(model)].set_index("metric_key").reindex(keys)
        values = numeric(selected, "value", rate=True)
        deviations = seed_sd(selected)
        x = np.arange(len(keys))
        color, marker = MODEL_COLORS[model], MODEL_MARKERS[model]
        ax.plot(x, values, marker=marker, color=color,
                linewidth=2.4 if model == "litemamba" else 1.4,
                markersize=6 if model == "litemamba" else 5,
                markerfacecolor=color if model == "litemamba" else "white",
                markeredgewidth=1.1, label=MODEL_DISPLAY[model])
        with_sd = np.isfinite(values) & np.isfinite(deviations)
        if with_sd.any():
            ax.errorbar(x[with_sd], values[with_sd], yerr=deviations[with_sd], fmt="none",
                        color=color, capsize=2.8, linewidth=0.9, alpha=0.85)
    ax.set(xticks=np.arange(len(keys)), xticklabels=metrics.metric_label.tolist(),
           ylim=(0, 1.04), ylabel="Rate", xlim=(-0.25, max(0.75, len(keys) - 0.75)))
    ax.yaxis.set_major_formatter(PercentFormatter(1.0))
    ax.tick_params(axis="x", rotation=12)
    style_axis(ax)
    ax.legend(frameon=False, ncols=2, loc="lower right")
    sd_caption(fig)
    return save_figure(fig, base, formats, dpi, bottom=0.13)


def template_model_plot(table: pd.DataFrame, role: str, metric: str, base: Path,
                         formats: Sequence[str], dpi: int) -> list[str]:
    table = table[table.role.eq(role) & table.metric.eq(metric) & table.probability_type.eq("raw")].copy()
    if "feature_scheme" in table:
        table = table[table.feature_scheme.eq("full")].copy()
    if not len(table):
        return []
    unknown = set(table.model) - set(MODEL_ORDER)
    if unknown:
        raise ValueError(f"逐模板表含未识别模型，绘图不会忽略：{sorted(unknown)}")
    table["template_num"] = pd.to_numeric(table.template.astype(str).str.extract(r"(\d+)$")[0], errors="raise").astype(int)
    require_unique(table, ["model", "template_num"], "per-template seed summary")
    numbers = list(range(1, max(20, int(table.template_num.max())) + 1))
    fig, ax = plt.subplots(figsize=(12, 4.6))
    offsets = np.linspace(-0.2, 0.2, 4)
    for model, offset in zip(MODEL_ORDER, offsets):
        selected = table[table.model.eq(model)].set_index("template_num").reindex(numbers)
        values, deviations = numeric(selected, "mean", rate=True), seed_sd(selected)
        known = np.isfinite(values)
        x = np.arange(len(numbers), dtype=float) + offset
        ax.scatter(x[known], values[known], s=27, marker=MODEL_MARKERS[model], color=MODEL_COLORS[model], label=MODEL_DISPLAY[model])
        with_sd = known & np.isfinite(deviations)
        if with_sd.any():
            ax.errorbar(x[with_sd], values[with_sd], yerr=deviations[with_sd], fmt="none",
                        ecolor=MODEL_COLORS[model], capsize=2, linewidth=0.85)
    present = set(table.template_num)
    for i, number in enumerate(numbers):
        if number not in present:
            annotate_missing(ax, i)
    prefix = "DNA" if role == "control" else "XNA"
    ylabel = "Canonical false X/Y call rate" if role == "control" else "Target X/Y recall"
    ax.set(xticks=np.arange(len(numbers)), xticklabels=[f"{prefix}{n}" for n in numbers],
           ylabel=ylabel, ylim=(0, 1.04), xlim=(-0.7, len(numbers) - 0.3))
    ax.tick_params(axis="x", rotation=45)
    for label in ax.get_xticklabels():
        label.set_horizontalalignment("right")
    ax.yaxis.set_major_formatter(PercentFormatter(1.0))
    style_axis(ax)
    ax.legend(frameon=False, ncols=4, loc="upper center", bbox_to_anchor=(0.5, 1.2))
    sd_caption(fig)
    return save_figure(fig, base, formats, dpi, bottom=0.13)


def load_overall_table(path: Path, required: Sequence[str]) -> pd.DataFrame:
    """分块筛选总体 full 结果，避免把所有 motif 分层表同时载入内存。"""
    selected = []
    columns = None
    try:
        for chunk in pd.read_csv(path, chunksize=100000):
            missing = sorted(set(required) - set(chunk.columns))
            if missing:
                raise ValueError(f"{path.name} 缺少字段：{missing}")
            columns = chunk.columns
            chunk = chunk[chunk.group_level.eq("overall")].copy()
            if "feature_scheme" in chunk:
                chunk = chunk[chunk.feature_scheme.eq("full")].copy()
            if len(chunk):
                selected.append(chunk)
    except pd.errors.EmptyDataError as error:
        raise ValueError(f"统计表没有表头：{path}") from error
    return pd.concat(selected, ignore_index=True) if selected else pd.DataFrame(columns=columns)


def plot_threshold_curves(table: pd.DataFrame, base: Path,
                           formats: Sequence[str], dpi: int) -> list[str]:
    if not len(table):
        return []
    unknown = set(table.model) - set(MODEL_ORDER)
    if unknown:
        raise ValueError(f"阈值表含未识别模型：{sorted(unknown)}")
    require_unique(table, ["model", "seed", "probability_type", "confidence_threshold"], "overall threshold table")
    numeric(table, "confidence_threshold", rate=True)
    metrics = ["coverage", "ncb_precision", "ncb_recall"]
    for metric in metrics:
        numeric(table, metric, rate=True)
    unknown_variants = set(table.probability_type) - {"raw", "calibrated"}
    if unknown_variants:
        raise ValueError(f"阈值表含未识别概率版本：{sorted(unknown_variants)}")
    variants = [name for name in ["raw", "calibrated"] if table.probability_type.eq(name).any()]
    fig, axes = plt.subplots(len(variants), 3, figsize=(12, 3.35 * len(variants)), squeeze=False)
    metric_labels = {"coverage": "Retained fraction", "ncb_precision": "X/Y call precision", "ncb_recall": "X/Y recall"}
    for row, variant in enumerate(variants):
        for col, metric in enumerate(metrics):
            ax = axes[row, col]
            for model in MODEL_ORDER:
                sub = table[table.model.eq(model) & table.probability_type.eq(variant)]
                summary = sub.groupby("confidence_threshold")[metric].agg(["mean", "std"]).sort_index()
                x = summary.index.to_numpy(dtype=float)
                means, deviations = summary["mean"].to_numpy(dtype=float), summary["std"].to_numpy(dtype=float)
                ax.plot(x, means, marker=MODEL_MARKERS[model], markersize=3.5,
                        color=MODEL_COLORS[model], linewidth=2 if model == "litemamba" else 1.15,
                        label=MODEL_DISPLAY[model])
                known = np.isfinite(means) & np.isfinite(deviations)
                if known.any():
                    ax.errorbar(x[known], means[known], yerr=deviations[known], fmt="none",
                                color=MODEL_COLORS[model], capsize=1.5, linewidth=0.7, alpha=0.75)
            prefix = "Raw" if variant == "raw" else "Calibrated"
            ax.set(xlabel="Fixed confidence threshold", ylabel=f"{prefix}: {metric_labels[metric]}",
                   ylim=(0, 1.04), xlim=(-0.02, 1.02), xticks=[0, 0.5, 0.8, 1])
            ax.yaxis.set_major_formatter(PercentFormatter(1.0))
            style_axis(ax)
    handles, labels = axes[0, 0].get_legend_handles_labels()
    fig.legend(handles, labels, ncols=4, loc="upper center", bbox_to_anchor=(0.5, 1.02), frameon=False)
    fig.text(0.5, 0.012, SD_NOTE + "\nRecall counts abstentions as misses; thresholds are descriptive and none is selected here.",
             ha="center", va="bottom", fontsize=8, color="#59656E")
    return save_figure(fig, base, formats, dpi, bottom=0.1)


def plot_calibration_curves(table: pd.DataFrame, base: Path,
                             formats: Sequence[str], dpi: int) -> list[str]:
    if not len(table):
        return []
    unknown = set(table.model) - set(MODEL_ORDER)
    if unknown:
        raise ValueError(f"校准表含未识别模型：{sorted(unknown)}")
    require_unique(table, ["model", "seed", "probability_type", "bin_left", "bin_right"], "overall calibration bins")
    for metric in ["bin_left", "bin_right", "mean_confidence", "accuracy"]:
        numeric(table, metric, rate=True)
    unknown_variants = set(table.probability_type) - {"raw", "calibrated"}
    if unknown_variants:
        raise ValueError(f"校准表含未识别概率版本：{sorted(unknown_variants)}")
    table = table[pd.to_numeric(table.n, errors="raise") > 0].copy()
    variants = [name for name in ["raw", "calibrated"] if table.probability_type.eq(name).any()]
    if not variants:
        return []
    fig, axes = plt.subplots(1, len(variants), figsize=(5.2 * len(variants), 4.9), squeeze=False)
    for col, variant in enumerate(variants):
        ax = axes[0, col]
        ax.plot([0, 1], [0, 1], color="#A6ADB2", linewidth=1, linestyle="--")
        for model in MODEL_ORDER:
            sub = table[table.model.eq(model) & table.probability_type.eq(variant)]
            summary = sub.groupby(["bin_left", "bin_right"]).agg(
                confidence=("mean_confidence", "mean"), accuracy=("accuracy", "mean"),
                deviation=("accuracy", "std"), available_seeds=("seed", "nunique"))
            ax.plot(summary.confidence, summary.accuracy, marker=MODEL_MARKERS[model], markersize=4,
                    color=MODEL_COLORS[model], linewidth=2 if model == "litemamba" else 1.1,
                    label=MODEL_DISPLAY[model])
            known = summary.deviation.notna()
            if known.any():
                ax.errorbar(summary.loc[known, "confidence"], summary.loc[known, "accuracy"],
                            yerr=summary.loc[known, "deviation"], fmt="none", color=MODEL_COLORS[model],
                            capsize=2, linewidth=0.8)
        prefix = "Raw" if variant == "raw" else "Calibrated"
        ax.set(xlabel=f"{prefix}: mean predicted confidence", ylabel="Observed accuracy",
               xlim=(0, 1.02), ylim=(0, 1.04))
        ax.xaxis.set_major_formatter(PercentFormatter(1.0))
        ax.yaxis.set_major_formatter(PercentFormatter(1.0))
        style_axis(ax)
    handles, labels = axes[0, 0].get_legend_handles_labels()
    fig.legend(handles, labels, ncols=4, loc="upper center", bbox_to_anchor=(0.5, 1.02), frameon=False)
    fig.text(0.5, 0.012, "Bin values are averaged over seeds with nonempty bins; bars show seed SD, not confidence intervals.",
             ha="center", va="bottom", fontsize=8, color="#59656E")
    return save_figure(fig, base, formats, dpi, bottom=0.1)


def normalize_tasks(tasks: Sequence[str], data_dir: Path) -> list[str]:
    aliases = {"6-class": "6class", "6_class": "6class", "sixclass": "6class"}
    normalized = [aliases.get(task, task) for task in tasks]
    if normalized == ["auto"]:
        return [slug for slug in TASK_CONFIG if any((data_dir / pattern.format(slug=slug)).is_file() for pattern in TABLE_PATTERNS.values())]
    if "auto" in normalized or any(task not in TASK_CONFIG for task in normalized):
        raise ValueError("--tasks 支持 6class、5X、5Y，或单独使用 auto")
    return list(dict.fromkeys(normalized))


def plot_all(data_dir: Path | str, out_dir: Path | str | None = None,
              formats: Sequence[str] = ("png", "pdf"), dpi: int = 300,
              tasks: Sequence[str] = ("6class",)) -> dict[str, Any]:
    data_dir = Path(data_dir).expanduser().resolve()
    if not data_dir.is_dir():
        raise FileNotFoundError(f"统计目录不存在：{data_dir}")
    out_dir = Path(out_dir).expanduser().resolve() if out_dir else data_dir.parent / "figures"
    if out_dir == data_dir:
        raise ValueError("图片输出目录须与 analysis_data 目录分开")
    formats = list(dict.fromkeys(item.strip().lower() for part in formats for item in part.split(",") if item.strip()))
    if not formats or any(fmt not in {"png", "pdf", "svg"} for fmt in formats):
        raise ValueError("输出格式必须为 png、pdf、svg")
    if dpi < 72:
        raise ValueError("dpi 必须至少为 72")
    tasks = normalize_tasks(tasks, data_dir)
    if not tasks:
        raise FileNotFoundError("没有找到可绘制的新任务统计表")
    configure_style()
    generated, sources, warnings, panels = [], {}, [], []

    def load(key: str, slug: str, required: Sequence[str]):
        path = data_dir / TABLE_PATTERNS[key].format(slug=slug)
        if not path.is_file():
            warnings.append(f"缺少 {path.name}；对应面板未生成")
            return None
        sources[path.name] = {"sha256": sha256_file(path), "bytes": path.stat().st_size}
        return read_table(path, required)

    for slug in tasks:
        a = load("a", slug, ["template_num", "dna_label", "false_noncanonical_call_rate"])
        b = load("b", slug, ["template_num", "xna_label", "target_recall_micro"])
        cd = load("c_detail", slug, ["template_num", "template", "error_type", "error_per_10000"])
        cs = load("c_summary", slug, ["template_num", "template", "total_error_per_10000", "target_recall_micro"])
        d = load("d", slug, ["metric_key", "metric_label", "order", "model", "value"])
        if a is not None:
            generated += plot_figure4a(a, slug, out_dir / f"Figure4A_DNA01_16_false_call_heatmap_{slug}", formats, dpi)
            panels.append({"task": slug, "panel": "A", "primary_model": "litemamba"})
        if b is not None:
            generated += plot_figure4b(b, slug, out_dir / f"Figure4B_XNA01_16_target_recall_lollipop_{slug}", formats, dpi)
            panels.append({"task": slug, "panel": "B", "primary_model": "litemamba"})
        if cd is not None and cs is not None:
            generated += plot_figure4c(cd, cs, slug, out_dir / f"Figure4C_XNA17_20_residual_error_composition_{slug}", formats, dpi)
            panels.append({"task": slug, "panel": "C", "primary_model": "litemamba"})
        if d is not None:
            generated += plot_profile(d, True, out_dir / f"Figure4D_external_validation_profile_with_overall_{slug}", formats, dpi)
            generated += plot_profile(d, False, out_dir / f"Figure4D_external_validation_profile_1to4_only_{slug}", formats, dpi)
            panels.append({"task": slug, "panel": "D", "models": MODEL_ORDER})

    template_path = data_dir / "plot_template_summary.csv"
    if template_path.is_file() and "6class" in tasks:
        summary = read_table(template_path, ["model", "template", "role", "probability_type", "metric", "mean", "seed_sd", "n_seeds"])
        sources[template_path.name] = {"sha256": sha256_file(template_path), "bytes": template_path.stat().st_size}
        generated += template_model_plot(summary, "control", "canonical_false_positive_rate", out_dir / "Figure4_all_models_DNA_false_calls_6class", formats, dpi)
        generated += template_model_plot(summary, "target", "target_recall_micro", out_dir / "Figure4_all_models_XNA_target_recall_6class", formats, dpi)

    if "6class" in tasks:
        threshold_path = data_dir / "confidence_thresholds.csv"
        if threshold_path.is_file():
            table = load_overall_table(threshold_path, ["group_level", "model", "seed", "probability_type", "confidence_threshold", "coverage", "ncb_precision", "ncb_recall"])
            sources[threshold_path.name] = {"sha256": sha256_file(threshold_path), "bytes": threshold_path.stat().st_size}
            generated += plot_threshold_curves(table, out_dir / "Figure4_confidence_coverage_precision_recall_6class", formats, dpi)
        calibration_path = data_dir / "calibration_bins.csv.gz"
        if not calibration_path.is_file():
            calibration_path = data_dir / "calibration_bins.csv"
        if calibration_path.is_file():
            table = load_overall_table(calibration_path, ["group_level", "model", "seed", "probability_type", "bin_left", "bin_right", "n", "mean_confidence", "accuracy"])
            sources[calibration_path.name] = {"sha256": sha256_file(calibration_path), "bytes": calibration_path.stat().st_size}
            generated += plot_calibration_curves(table, out_dir / "Figure4_probability_calibration_6class", formats, dpi)

    if not generated:
        raise FileNotFoundError("所选任务没有可绘制的有效面板，未生成空图作为结果")
    result = {
        "version": VERSION, "status": "complete", "data_dir": str(data_dir), "out_dir": str(out_dir),
        "tasks": tasks, "formats": formats, "dpi": dpi, "panels": panels,
        "statistics": "Saved means across all planned training seeds; error bars are sample SD, not confidence intervals.",
        "missing_policy": "Missing templates remain NA; missing error components are never filled with zero.",
        "model_policy": "A-C retain the manuscript's LiteMamba focus; D and supplementary template comparisons retain every available comparator.",
        "source_tables": sources, "warnings": warnings,
        "outputs": [{"path": str(Path(path).relative_to(out_dir)), "sha256": sha256_file(Path(path)),
                     "bytes": Path(path).stat().st_size} for path in generated],
    }
    out_dir.mkdir(parents=True, exist_ok=True)
    temporary = out_dir / "plot_manifest.json.tmp"
    temporary.write_text(json.dumps(result, ensure_ascii=False, indent=2, allow_nan=False) + "\n", encoding="utf-8")
    temporary.replace(out_dir / "plot_manifest.json")
    print(f"[PLOT COMPLETE] {len(generated)} 个图件：{out_dir}", flush=True)
    for warning in warnings:
        print("[PLOT NOTE] " + warning, flush=True)
    return result


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data_dir", type=Path, required=True, help="新预测流程生成的 analysis_data 目录")
    parser.add_argument("--out_dir", "--output_dir", "--figure_dir", dest="out_dir", type=Path, default=None)
    parser.add_argument("--formats", "--figure_formats", dest="formats", nargs="+", default=["png", "pdf"])
    parser.add_argument("--dpi", "--figure_dpi", dest="dpi", type=int, default=300)
    parser.add_argument("--tasks", nargs="+", default=["6class"], help="默认仅6class；auto显式发现已有任务")
    return parser.parse_args(argv)


def main(argv: Sequence[str] | None = None) -> None:
    args = parse_args(argv)
    plot_all(args.data_dir, args.out_dir, args.formats, args.dpi, args.tasks)


if __name__ == "__main__":
    main()
