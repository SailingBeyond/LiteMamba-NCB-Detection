#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
plot_figure3bcef.py

功能：
    只读取已经完成模型推理后生成的总体指标和逐 motif 指标 CSV，
    不重新加载模型、不重新读取原始测试集、不重新执行推理，直接重绘：

    Figure 3B：3 个任务 × 4 个模型的总体性能点线图
    Figure 3C：3 个任务的 1024 个 motif accuracy ECDF
    Figure 3E/3F：ATCGXY 6-Class 与两个五分类任务的配对 motif accuracy 散点图

本版修改：
    1. 三个任务在 Figure 3B、3C、3E 和 3F 中统一显示为：
           ATCGXY 6-Class
           ATCGX 5-Class
           ATCGY 5-Class
    2. Figure 3B 的所有单图和 2×2 汇总图纵轴均从 0.92 开始。
    3. Figure 3B 和 Figure 3C 的横向长度明显缩小，并使用换行标签/紧凑图例
       避免横向压缩后文字拥挤。
    4. 全部图片统一采用 Arial 字体、白色背景、固定配色、统一坐标轴线宽，
       PDF/PS 使用 Type 42 字体，SVG 保留可编辑文字。
    5. Figure 3E/3F 不再标注单个 motif 名称，避免散点图文字拥挤；
       colorbar 标题简化为 Accuracy difference。
    6. 同时重新输出绘图所需长表、宽表、ECDF 表和配对 motif 表，
       后续继续修改图片时无需重新运行模型。

输入：
    <result_root>/combined_data/figure3_all_tasks_overall_metrics_long.csv
    <result_root>/combined_data/figure3_all_tasks_per_motif_metrics_long.csv

输出：
    <result_root>/figures/
    <result_root>/combined_data/figure3_plot_data/

运行：
    python plot_figure3bcef.py --result_root /path/to/figure3_results

指定 Figure 3C/3E/3F 使用的模型：
    python plot_figure3bcef.py --motif_model Transformer
"""

from __future__ import annotations

import argparse
import math
import re
from pathlib import Path
from typing import Dict, Iterable, List, Tuple

import numpy as np
import pandas as pd

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.colors import Normalize
from matplotlib.ticker import FormatStrFormatter, MultipleLocator


# =============================================================================
# 1. 默认路径和绘图配置
# =============================================================================

DEFAULT_RESULT_ROOT = Path(".")
DEFAULT_COMBINED_DIR = DEFAULT_RESULT_ROOT / "combined_data"
DEFAULT_FIGURE_DIR = DEFAULT_RESULT_ROOT / "figures_replot_updated"
DEFAULT_REPLOT_DATA_DIR = DEFAULT_COMBINED_DIR / "replot_updated"

DEFAULT_OVERALL_CSV = (
    DEFAULT_COMBINED_DIR / "figure3_all_tasks_overall_metrics_long.csv"
)
DEFAULT_MOTIF_CSV = (
    DEFAULT_COMBINED_DIR / "figure3_all_tasks_per_motif_metrics_long.csv"
)

FIGURE_DPI = 600
FIGURE3B_YMIN = 0.94
FIGURE3B_YMAX = 0.97

# -----------------------------------------------------------------------------
# 统一绘图风格
# -----------------------------------------------------------------------------
# 后续只需要修改这里，即可统一调整整套 Figure 3 的字体、字号和线条。
FONT_FAMILY = "Arial"
BASE_FONT_SIZE = 16
AXIS_LABEL_FONT_SIZE = 18
AXIS_TITLE_FONT_SIZE = 18
TICK_LABEL_FONT_SIZE = 15
LEGEND_FONT_SIZE = 13
ANNOTATION_FONT_SIZE = 11

AXIS_LINE_WIDTH = 1.5
TICK_LINE_WIDTH = 1.4
TICK_LENGTH = 5.5
GRID_LINE_WIDTH = 0.8
GRID_ALPHA = 0.25

# 横向压缩版画布尺寸：只缩小宽度，基本保留高度。
FIGURE3B_SINGLE_FIGSIZE = (5, 5.5)
FIGURE3B_SUMMARY_FIGSIZE = (10.6, 10.2)
FIGURE3C_FIGSIZE = (5, 5.5)
FIGURE3EF_FIGSIZE = (7.2, 6)

# 固定颜色，保证不同图片中同一个模型或任务始终使用相同颜色。
MODEL_COLORS = {
    "BiGRU": "#4C72B0",
    "Transformer": "#DD8452",
    "ConvMixer": "#55A868",
    "LiteMamba": "#C44E52",
}

TASK_COLORS = {
    "6-class": "#4C72B0",
    "5X": "#DD8452",
    "5Y": "#55A868",
}

MODEL_ORDER = ["BiGRU", "Transformer", "ConvMixer", "LiteMamba"]

MODEL_SPECS: Dict[str, Dict[str, object]] = {
    "BiGRU": {
        "marker": "o",
        "linestyle": "--",
        "color": MODEL_COLORS["BiGRU"],
    },
    "Transformer": {
        "marker": "s",
        "linestyle": "-.",
        "color": MODEL_COLORS["Transformer"],
    },
    "ConvMixer": {
        "marker": "^",
        "linestyle": ":",
        "color": MODEL_COLORS["ConvMixer"],
    },
    "LiteMamba": {
        "marker": "D",
        "linestyle": "-",
        "color": MODEL_COLORS["LiteMamba"],
    },
}

# 内部统一使用短任务键，图中使用正式任务名称。
TASK_ORDER = ["6-class", "5X", "5Y"]

TASK_DISPLAY_NAMES = {
    "6-class": "ATCGXY 6-Class",
    "5X": "ATCGX 5-Class",
    "5Y": "ATCGY 5-Class",
}

# Figure 3B 横向缩短后，横轴标签改为两行，避免三个长标签互相重叠。
TASK_TICK_NAMES = {
    "6-class": "ATCGXY\n6-Class",
    "5X": "ATCGX\n5-Class",
    "5Y": "ATCGY\n5-Class",
}

TASK_ALIASES = {
    "6-class": "6-class",
    "6_class": "6-class",
    "6 class": "6-class",
    "atcgxy 6-class": "6-class",
    "atcgxy 6 class": "6-class",
    "5x": "5X",
    "5_x": "5X",
    "5-x": "5X",
    "atcgx 5-class": "5X",
    "atcgx 5 class": "5X",
    "5y": "5Y",
    "5_y": "5Y",
    "5-y": "5Y",
    "atcgy 5-class": "5Y",
    "atcgy 5 class": "5Y",
}

METRIC_SPECS = {
    "f1_macro": "Macro-F1",
    "balanced_accuracy": "Balanced accuracy",
    "mcc": "MCC",
    "accuracy": "Accuracy",
}

FIGURE3B_FILENAMES = {
    "f1_macro": "Figure3B_macro_F1",
    "balanced_accuracy": "Figure3B_balanced_accuracy",
    "mcc": "Figure3B_MCC",
    "accuracy": "Figure3B_accuracy",
}


# =============================================================================
# 2. 通用工具函数
# =============================================================================

def configure_matplotlib() -> None:
    """统一设置为论文中常用的 Arial 科研绘图风格。"""
    plt.rcParams.update({
        # 字体：优先 Arial；若当前系统未安装 Arial，再自动使用后备字体。
        "font.family": "sans-serif",
        "font.sans-serif": [
            FONT_FAMILY,
            "Liberation Sans",
            "DejaVu Sans",
        ],
        "font.size": BASE_FONT_SIZE,

        # 坐标轴、标题、刻度和图例字号。
        "axes.labelsize": AXIS_LABEL_FONT_SIZE,
        "axes.titlesize": AXIS_TITLE_FONT_SIZE,
        "axes.titleweight": "normal",
        "axes.labelweight": "normal",
        "xtick.labelsize": TICK_LABEL_FONT_SIZE,
        "ytick.labelsize": TICK_LABEL_FONT_SIZE,
        "legend.fontsize": LEGEND_FONT_SIZE,

        # 线条与刻度。
        "axes.linewidth": AXIS_LINE_WIDTH,
        "xtick.major.width": TICK_LINE_WIDTH,
        "ytick.major.width": TICK_LINE_WIDTH,
        "xtick.major.size": TICK_LENGTH,
        "ytick.major.size": TICK_LENGTH,
        "xtick.direction": "out",
        "ytick.direction": "out",

        # 白色背景及可编辑矢量字体。
        "figure.facecolor": "white",
        "axes.facecolor": "white",
        "savefig.facecolor": "white",
        "savefig.edgecolor": "white",
        "savefig.transparent": False,
        "savefig.dpi": FIGURE_DPI,
        "pdf.fonttype": 42,
        "ps.fonttype": 42,
        "svg.fonttype": "none",

        # 避免负号显示异常。
        "axes.unicode_minus": False,
    })


def apply_common_axis_style(
    ax: plt.Axes,
    grid_axis: str | None = None,
) -> None:
    """给所有子图应用一致的坐标轴、刻度和网格风格。"""
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)

    for spine_name in ["left", "bottom"]:
        ax.spines[spine_name].set_linewidth(AXIS_LINE_WIDTH)
        ax.spines[spine_name].set_color("black")

    ax.tick_params(
        axis="both",
        which="major",
        direction="out",
        width=TICK_LINE_WIDTH,
        length=TICK_LENGTH,
        colors="black",
    )

    if grid_axis is not None:
        ax.grid(
            axis=grid_axis,
            linestyle="--",
            linewidth=GRID_LINE_WIDTH,
            alpha=GRID_ALPHA,
            zorder=0,
        )


def ensure_dir(path: Path | str) -> Path:
    path = Path(path)
    path.mkdir(parents=True, exist_ok=True)
    return path


def safe_name(value: str) -> str:
    return re.sub(r"[^\w.\-]+", "_", str(value))


def save_figure_all_formats(
    fig: plt.Figure,
    output_stem: Path,
    dpi: int = FIGURE_DPI,
) -> None:
    ensure_dir(output_stem.parent)
    save_kwargs = {
        "bbox_inches": "tight",
        "facecolor": "white",
        "edgecolor": "white",
    }
    fig.savefig(
        output_stem.with_suffix(".png"),
        dpi=dpi,
        **save_kwargs,
    )
    fig.savefig(
        output_stem.with_suffix(".pdf"),
        **save_kwargs,
    )
    fig.savefig(
        output_stem.with_suffix(".svg"),
        **save_kwargs,
    )


def normalize_task_name(value: object) -> str:
    raw = str(value).strip()
    key = raw.lower()

    if key in TASK_ALIASES:
        return TASK_ALIASES[key]

    # 保留对标准大小写的兼容。
    if raw in TASK_ORDER:
        return raw

    raise ValueError(
        f"无法识别任务名称：{raw!r}。支持的名称包括："
        "6-class、5X、5Y、ATCGXY 6-Class、ATCGX 5-Class、ATCGY 5-Class。"
    )


def rename_first_existing_column(
    dataframe: pd.DataFrame,
    target_name: str,
    candidates: Iterable[str],
) -> pd.DataFrame:
    if target_name in dataframe.columns:
        return dataframe

    for candidate in candidates:
        if candidate in dataframe.columns:
            return dataframe.rename(columns={candidate: target_name})

    return dataframe


def require_columns(
    dataframe: pd.DataFrame,
    required_columns: Iterable[str],
    dataframe_name: str,
) -> None:
    missing = [column for column in required_columns if column not in dataframe.columns]
    if missing:
        raise KeyError(
            f"{dataframe_name} 缺少必要列：{missing}\n"
            f"当前列为：{list(dataframe.columns)}"
        )


def locate_input_csv(
    explicitly_requested: Path,
    fallback_paths: Iterable[Path],
    description: str,
) -> Path:
    candidates = [explicitly_requested, *fallback_paths]
    checked: List[Path] = []

    for candidate in candidates:
        candidate = Path(candidate)
        if candidate in checked:
            continue
        checked.append(candidate)
        if candidate.exists():
            return candidate

    checked_text = "\n".join(f"  - {path}" for path in checked)
    raise FileNotFoundError(
        f"没有找到{description}。已检查：\n{checked_text}"
    )


# =============================================================================
# 3. 读取并规范化上次运行生成的数据
# =============================================================================

def load_overall_metrics(overall_csv: Path) -> pd.DataFrame:
    dataframe = pd.read_csv(overall_csv)

    dataframe = rename_first_existing_column(
        dataframe,
        "task",
        ["dataset", "task_name"],
    )
    dataframe = rename_first_existing_column(
        dataframe,
        "model",
        ["model_name"],
    )

    required = ["task", "model", *METRIC_SPECS.keys()]
    require_columns(dataframe, required, "总体指标 CSV")

    dataframe = dataframe.copy()
    dataframe["task"] = dataframe["task"].map(normalize_task_name)
    dataframe["task_display"] = dataframe["task"].map(TASK_DISPLAY_NAMES)

    for metric_name in METRIC_SPECS:
        dataframe[metric_name] = pd.to_numeric(
            dataframe[metric_name], errors="coerce"
        )

    if dataframe[list(METRIC_SPECS)].isna().any().any():
        bad_rows = dataframe[
            dataframe[list(METRIC_SPECS)].isna().any(axis=1)
        ]
        raise ValueError(
            "总体指标 CSV 中存在无法转换为数值的指标：\n"
            f"{bad_rows[['task', 'model', *METRIC_SPECS.keys()]].to_string(index=False)}"
        )

    dataframe = dataframe.drop_duplicates(
        subset=["task", "model"],
        keep="last",
    )

    dataframe["task"] = pd.Categorical(
        dataframe["task"], categories=TASK_ORDER, ordered=True
    )
    dataframe["model"] = pd.Categorical(
        dataframe["model"], categories=MODEL_ORDER, ordered=True
    )

    dataframe = dataframe.sort_values(["task", "model"]).reset_index(drop=True)

    present_tasks = set(dataframe["task"].dropna().astype(str))
    missing_tasks = [task for task in TASK_ORDER if task not in present_tasks]
    if missing_tasks:
        raise RuntimeError(f"总体指标 CSV 缺少任务：{missing_tasks}")

    present_models = set(dataframe["model"].dropna().astype(str))
    missing_models = [model for model in MODEL_ORDER if model not in present_models]
    if missing_models:
        raise RuntimeError(f"总体指标 CSV 缺少模型：{missing_models}")

    return dataframe


def load_per_motif_metrics(motif_csv: Path) -> pd.DataFrame:
    dataframe = pd.read_csv(motif_csv)

    dataframe = rename_first_existing_column(
        dataframe,
        "task",
        ["dataset", "task_name"],
    )
    dataframe = rename_first_existing_column(
        dataframe,
        "model",
        ["model_name"],
    )
    dataframe = rename_first_existing_column(
        dataframe,
        "motif",
        ["kmer", "sequence"],
    )
    dataframe = rename_first_existing_column(
        dataframe,
        "n_samples",
        ["eval_samples", "valid_samples", "sample_count"],
    )

    require_columns(
        dataframe,
        ["task", "model", "motif", "accuracy"],
        "逐 motif 指标 CSV",
    )

    dataframe = dataframe.copy()
    dataframe["task"] = dataframe["task"].map(normalize_task_name)
    dataframe["task_display"] = dataframe["task"].map(TASK_DISPLAY_NAMES)
    dataframe["motif"] = dataframe["motif"].astype(str).str.strip()
    dataframe["accuracy"] = pd.to_numeric(dataframe["accuracy"], errors="coerce")

    if "n_samples" not in dataframe.columns:
        dataframe["n_samples"] = np.nan
    else:
        dataframe["n_samples"] = pd.to_numeric(
            dataframe["n_samples"], errors="coerce"
        )

    dataframe = dataframe.dropna(subset=["accuracy"])
    dataframe = dataframe.drop_duplicates(
        subset=["task", "model", "motif"],
        keep="last",
    )

    dataframe["task"] = pd.Categorical(
        dataframe["task"], categories=TASK_ORDER, ordered=True
    )
    dataframe["model"] = pd.Categorical(
        dataframe["model"], categories=MODEL_ORDER, ordered=True
    )

    dataframe = dataframe.sort_values(
        ["task", "model", "motif"]
    ).reset_index(drop=True)

    return dataframe


# =============================================================================
# 4. Figure 3B：任务 × 模型总体性能点线图
# =============================================================================

def prepare_figure3b_data(
    overall_df: pd.DataFrame,
    output_data_dir: Path,
) -> pd.DataFrame:
    records = []

    for _, row in overall_df.iterrows():
        task_name = str(row["task"])
        model_name = str(row["model"])

        for metric_name, metric_display in METRIC_SPECS.items():
            record = {
                "task": task_name,
                "task_display": TASK_DISPLAY_NAMES[task_name],
                "model": model_name,
                "metric": metric_name,
                "metric_display": metric_display,
                "value": float(row[metric_name]),
            }

            for optional_column in [
                "num_classes",
                "total_samples",
                "total_motifs",
                "checkpoint",
            ]:
                if optional_column in overall_df.columns:
                    record[optional_column] = row[optional_column]

            records.append(record)

    plot_df = pd.DataFrame(records)
    plot_df["task"] = pd.Categorical(
        plot_df["task"], categories=TASK_ORDER, ordered=True
    )
    plot_df["model"] = pd.Categorical(
        plot_df["model"], categories=MODEL_ORDER, ordered=True
    )
    plot_df = plot_df.sort_values(
        ["metric", "model", "task"]
    ).reset_index(drop=True)

    long_path = output_data_dir / "figure3B_plot_data_long_updated.csv"
    plot_df.to_csv(long_path, index=False)

    wide_df = plot_df.pivot_table(
        index=["metric", "metric_display", "model"],
        columns="task_display",
        values="value",
        observed=False,
    ).reset_index()

    preferred_columns = [
        "metric",
        "metric_display",
        "model",
        *[TASK_DISPLAY_NAMES[task] for task in TASK_ORDER],
    ]
    wide_df = wide_df[
        [column for column in preferred_columns if column in wide_df.columns]
    ]
    wide_df.to_csv(
        output_data_dir / "figure3B_plot_data_wide_updated.csv",
        index=False,
    )

    return plot_df


def draw_figure3b_metric(
    ax: plt.Axes,
    metric_df: pd.DataFrame,
    ylabel: str,
    show_legend: bool,
) -> None:
    x_positions = np.arange(len(TASK_ORDER))

    for model_index, model_name in enumerate(MODEL_ORDER):
        model_df = metric_df[
            metric_df["model"].astype(str) == model_name
        ].copy()

        value_map = {
            str(task): float(value)
            for task, value in zip(model_df["task"], model_df["value"])
        }
        values = [value_map.get(task, np.nan) for task in TASK_ORDER]

        line_width = 3.0 if model_name == "LiteMamba" else 2.2
        marker_size = 9.5 if model_name == "LiteMamba" else 8.5

        ax.plot(
            x_positions,
            values,
            marker=str(MODEL_SPECS[model_name]["marker"]),
            linestyle=str(MODEL_SPECS[model_name]["linestyle"]),
            linewidth=line_width,
            markersize=marker_size,
            label=model_name,
            color=str(MODEL_SPECS[model_name]["color"]),
            markeredgecolor="white",
            markeredgewidth=0.8,
            zorder=3,
        )

    ax.set_xticks(x_positions)
    ax.set_xticklabels(
        [TASK_TICK_NAMES[task] for task in TASK_ORDER],
        rotation=0,
        ha="center",
        linespacing=1.05,
    )
    ax.set_xlabel("Classification task")
    ax.set_ylabel(ylabel)

    # 按用户要求，Figure 3B 所有纵轴均从 0.92 开始。
    ax.set_ylim(FIGURE3B_YMIN, FIGURE3B_YMAX)
    ax.yaxis.set_major_locator(MultipleLocator(0.01))
    ax.yaxis.set_major_formatter(FormatStrFormatter("%.2f"))

    apply_common_axis_style(ax, grid_axis="y")

    if show_legend:
        ax.legend(
            frameon=False,
            ncol=2,
            loc="upper center",
            bbox_to_anchor=(0.5, 1.15),
            fontsize=LEGEND_FONT_SIZE,
            handlelength=2.2,
            columnspacing=1.0,
            labelspacing=0.40,
            borderaxespad=0.0,
        )


def plot_figure3b(
    plot_df: pd.DataFrame,
    figure_dir: Path,
) -> List[Path]:
    generated: List[Path] = []

    values_below_limit = plot_df[plot_df["value"] < FIGURE3B_YMIN]
    if not values_below_limit.empty:
        print(
            "[WARN] 以下 Figure 3B 数据低于纵轴下限 0.92，"
            "按要求绘图时将位于显示范围之外："
        )
        print(
            values_below_limit[
                ["task_display", "model", "metric_display", "value"]
            ].to_string(index=False)
        )

    for metric_name, metric_display in METRIC_SPECS.items():
        metric_df = plot_df[plot_df["metric"] == metric_name].copy()

        fig, ax = plt.subplots(figsize=FIGURE3B_SINGLE_FIGSIZE)
        draw_figure3b_metric(
            ax=ax,
            metric_df=metric_df,
            ylabel=metric_display,
            show_legend=True,
        )
        fig.tight_layout()

        output_stem = figure_dir / FIGURE3B_FILENAMES[metric_name]
        save_figure_all_formats(fig, output_stem)
        plt.close(fig)
        generated.append(output_stem.with_suffix(".png"))

    fig, axes = plt.subplots(2, 2, figsize=FIGURE3B_SUMMARY_FIGSIZE)

    for ax, (metric_name, metric_display) in zip(
        axes.flat,
        METRIC_SPECS.items(),
    ):
        metric_df = plot_df[plot_df["metric"] == metric_name].copy()
        draw_figure3b_metric(
            ax=ax,
            metric_df=metric_df,
            ylabel=metric_display,
            show_legend=False,
        )
        ax.set_title(metric_display, pad=10)

    handles, labels = axes.flat[0].get_legend_handles_labels()
    fig.legend(
        handles,
        labels,
        loc="lower center",
        ncol=4,
        frameon=False,
        fontsize=LEGEND_FONT_SIZE,
        bbox_to_anchor=(0.5, 0.01),
    )
    fig.tight_layout(rect=(0, 0.06, 1, 1))

    output_stem = figure_dir / "Figure3B_four_metrics"
    save_figure_all_formats(fig, output_stem)
    plt.close(fig)
    generated.append(output_stem.with_suffix(".png"))

    return generated


# =============================================================================
# 5. Figure 3C：逐 motif accuracy ECDF
# =============================================================================

def prepare_figure3c_data(
    motif_df: pd.DataFrame,
    selected_model: str,
    output_data_dir: Path,
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    selected = motif_df[
        motif_df["model"].astype(str) == selected_model
    ].copy()

    if selected.empty:
        raise RuntimeError(
            f"逐 motif 数据中没有找到模型 {selected_model}。"
        )

    ecdf_rows = []
    summary_rows = []

    for task_name in TASK_ORDER:
        task_df = selected[
            selected["task"].astype(str) == task_name
        ].copy()
        task_df = task_df.drop_duplicates(
            subset=["motif"], keep="last"
        ).sort_values("accuracy")

        accuracies = task_df["accuracy"].astype(float).to_numpy()
        motifs = task_df["motif"].astype(str).to_numpy()
        n_motifs = len(accuracies)

        if n_motifs == 0:
            raise RuntimeError(
                f"模型 {selected_model} 的任务 {task_name} 没有逐 motif 数据。"
            )

        cumulative = np.arange(1, n_motifs + 1, dtype=float) / n_motifs

        for rank, (motif, accuracy, cumulative_fraction) in enumerate(
            zip(motifs, accuracies, cumulative),
            start=1,
        ):
            ecdf_rows.append({
                "model": selected_model,
                "task": task_name,
                "task_display": TASK_DISPLAY_NAMES[task_name],
                "motif": motif,
                "accuracy": float(accuracy),
                "ecdf_rank": rank,
                "n_motifs": n_motifs,
                "cumulative_motif_fraction": float(cumulative_fraction),
            })

        summary_rows.append({
            "model": selected_model,
            "task": task_name,
            "task_display": TASK_DISPLAY_NAMES[task_name],
            "n_motifs": n_motifs,
            "minimum": float(np.min(accuracies)),
            "q1": float(np.quantile(accuracies, 0.25)),
            "median": float(np.median(accuracies)),
            "mean": float(np.mean(accuracies)),
            "q3": float(np.quantile(accuracies, 0.75)),
            "maximum": float(np.max(accuracies)),
            "proportion_ge_0.90": float(np.mean(accuracies >= 0.90)),
            "proportion_ge_0.95": float(np.mean(accuracies >= 0.95)),
            "proportion_ge_0.98": float(np.mean(accuracies >= 0.98)),
        })

    ecdf_df = pd.DataFrame(ecdf_rows)
    summary_df = pd.DataFrame(summary_rows)

    ecdf_df.to_csv(
        output_data_dir / "figure3C_ecdf_plot_data_updated.csv",
        index=False,
    )
    summary_df.to_csv(
        output_data_dir / "figure3C_summary_updated.csv",
        index=False,
    )

    return ecdf_df, summary_df


def plot_figure3c(
    ecdf_df: pd.DataFrame,
    summary_df: pd.DataFrame,
    selected_model: str,
    figure_dir: Path,
) -> Path:
    fig, ax = plt.subplots(figsize=FIGURE3C_FIGSIZE)

    for task_index, task_name in enumerate(TASK_ORDER):
        task_df = ecdf_df[
            ecdf_df["task"] == task_name
        ].sort_values("accuracy")

        summary_row = summary_df[
            summary_df["task"] == task_name
        ].iloc[0]

        median = float(summary_row["median"])
        proportion_ge_095 = float(summary_row["proportion_ge_0.95"])
        display_name = TASK_DISPLAY_NAMES[task_name]

        # Figure 3C 横向压缩后，图例改成两行显示，避免超出画布。
        legend_label = (
            f"{display_name}\n"
            f"median={median:.3f}, ≥0.95={proportion_ge_095 * 100:.1f}%"
        )

        x_values = task_df["accuracy"].astype(float).to_numpy()
        y_values = task_df["cumulative_motif_fraction"].astype(float).to_numpy()

        # 从累计比例 0 开始，更符合标准 ECDF 的视觉表达。
        x_plot = np.r_[x_values[0], x_values]
        y_plot = np.r_[0.0, y_values]

        ax.step(
            x_plot,
            y_plot,
            where="post",
            linewidth=2.5,
            label=legend_label,
            color=TASK_COLORS[task_name],
        )

    for threshold in [0.90, 0.95, 0.98]:
        ax.axvline(
            threshold,
            color="0.35",
            linestyle="--",
            linewidth=1.2,
            alpha=0.8,
        )
        ax.text(
            threshold,
            0.025,
            f"{threshold:.2f}",
            rotation=90,
            ha="right",
            va="bottom",
            fontsize=ANNOTATION_FONT_SIZE,
            color="0.25",
        )

    # 为确保 0.90、0.95 和 0.98 三条阈值线都能完整显示，
    # 横轴左端至少扩展到 0.89。
    data_x_min = max(0.0, float(ecdf_df["accuracy"].min()) - 0.015)
    x_min = min(0.89, data_x_min)
    ax.set_xlim(x_min, 1.002)
    ax.set_ylim(0.0, 1.005)
    ax.set_xlabel("Per-motif accuracy")
    ax.set_ylabel("Cumulative motif proportion")
    ax.yaxis.set_major_locator(MultipleLocator(0.2))
    apply_common_axis_style(ax, grid_axis="y")
    ax.legend(
        frameon=False,
        loc="upper left",
        fontsize=LEGEND_FONT_SIZE,
        handlelength=1.9,
        labelspacing=0.55,
        borderaxespad=0.25,
    )

    fig.tight_layout()
    output_stem = (
        figure_dir
        / f"Figure3C_motif_accuracy_ECDF_{safe_name(selected_model)}"
    )
    save_figure_all_formats(fig, output_stem)
    plt.close(fig)

    return output_stem.with_suffix(".png")


# =============================================================================
# 6. Figure 3E/3F：任务间逐 motif accuracy 配对散点图
# =============================================================================

def prepare_figure3ef_data(
    motif_df: pd.DataFrame,
    selected_model: str,
    output_data_dir: Path,
    top_n: int,
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    selected = motif_df[
        motif_df["model"].astype(str) == selected_model
    ][["task", "motif", "accuracy", "n_samples"]].copy()

    if selected.empty:
        raise RuntimeError(
            f"逐 motif 数据中没有找到模型 {selected_model}。"
        )

    accuracy_wide = selected.pivot_table(
        index="motif",
        columns="task",
        values="accuracy",
        aggfunc="first",
        observed=False,
    ).reset_index()

    sample_wide = selected.pivot_table(
        index="motif",
        columns="task",
        values="n_samples",
        aggfunc="first",
        observed=False,
    ).reset_index()

    accuracy_wide.columns = [
        "motif" if str(column) == "motif"
        else f"accuracy_{str(column).replace('-', '_')}"
        for column in accuracy_wide.columns
    ]
    sample_wide.columns = [
        "motif" if str(column) == "motif"
        else f"n_samples_{str(column).replace('-', '_')}"
        for column in sample_wide.columns
    ]

    paired = accuracy_wide.merge(sample_wide, on="motif", how="outer")

    required_accuracy_columns = [
        "accuracy_6_class",
        "accuracy_5X",
        "accuracy_5Y",
    ]
    require_columns(
        paired,
        required_accuracy_columns,
        "Figure 3E/3F 配对数据",
    )

    paired["complete_case"] = paired[
        required_accuracy_columns
    ].notna().all(axis=1)

    paired["diff_ATCGX_5_Class_minus_ATCGXY_6_Class"] = (
        paired["accuracy_5X"] - paired["accuracy_6_class"]
    )
    paired["diff_ATCGY_5_Class_minus_ATCGXY_6_Class"] = (
        paired["accuracy_5Y"] - paired["accuracy_6_class"]
    )

    paired["abs_diff_ATCGX_5_Class_vs_ATCGXY_6_Class"] = paired[
        "diff_ATCGX_5_Class_minus_ATCGXY_6_Class"
    ].abs()
    paired["abs_diff_ATCGY_5_Class_vs_ATCGXY_6_Class"] = paired[
        "diff_ATCGY_5_Class_minus_ATCGXY_6_Class"
    ].abs()

    paired = paired.sort_values("motif").reset_index(drop=True)
    complete = paired[paired["complete_case"]].copy()

    paired.to_csv(
        output_data_dir / "figure3EF_paired_motif_accuracy_all_updated.csv",
        index=False,
    )
    complete.to_csv(
        output_data_dir / "figure3EF_paired_motif_accuracy_complete_updated.csv",
        index=False,
    )

    deviation_rows = []
    comparison_specs = [
        (
            "ATCGXY 6-Class vs ATCGX 5-Class",
            "accuracy_5X",
            "diff_ATCGX_5_Class_minus_ATCGXY_6_Class",
            "abs_diff_ATCGX_5_Class_vs_ATCGXY_6_Class",
        ),
        (
            "ATCGXY 6-Class vs ATCGY 5-Class",
            "accuracy_5Y",
            "diff_ATCGY_5_Class_minus_ATCGXY_6_Class",
            "abs_diff_ATCGY_5_Class_vs_ATCGXY_6_Class",
        ),
    ]

    for comparison, target_accuracy_col, diff_col, abs_diff_col in comparison_specs:
        top_df = complete.nlargest(top_n, abs_diff_col)

        for rank, (_, row) in enumerate(top_df.iterrows(), start=1):
            deviation_rows.append({
                "model": selected_model,
                "comparison": comparison,
                "rank": rank,
                "motif": row["motif"],
                "ATCGXY_6_Class_accuracy": row["accuracy_6_class"],
                "target_task_accuracy": row[target_accuracy_col],
                "signed_difference_target_minus_ATCGXY_6_Class": row[diff_col],
                "absolute_difference": row[abs_diff_col],
            })

    deviation_df = pd.DataFrame(deviation_rows)
    deviation_df.to_csv(
        output_data_dir / "figure3EF_largest_deviations_updated.csv",
        index=False,
    )

    return complete, deviation_df


def get_pair_columns(target_task: str) -> Tuple[str, str, str]:
    if target_task == "5X":
        return (
            "accuracy_5X",
            "diff_ATCGX_5_Class_minus_ATCGXY_6_Class",
            "abs_diff_ATCGX_5_Class_vs_ATCGXY_6_Class",
        )

    if target_task == "5Y":
        return (
            "accuracy_5Y",
            "diff_ATCGY_5_Class_minus_ATCGXY_6_Class",
            "abs_diff_ATCGY_5_Class_vs_ATCGXY_6_Class",
        )

    raise ValueError(f"不支持的目标任务：{target_task}")


def draw_paired_scatter(
    ax: plt.Axes,
    data: pd.DataFrame,
    target_task: str,
    top_n: int,
) -> None:
    # top_n 参数为兼容现有调用方式而保留。
    # Figure 3E/3F 中不再显示任何 motif 名称；top_n 仅用于前面的差异表导出。
    _ = top_n
    y_col, diff_col, abs_diff_col = get_pair_columns(target_task)

    target_display_name = TASK_DISPLAY_NAMES[target_task]
    reference_display_name = TASK_DISPLAY_NAMES["6-class"]

    x_values = data["accuracy_6_class"].astype(float).to_numpy()
    y_values = data[y_col].astype(float).to_numpy()
    diff_values = data[diff_col].astype(float).to_numpy()

    if len(x_values) == 0:
        raise RuntimeError(
            f"{reference_display_name} 与 {target_display_name} 没有完整配对 motif。"
        )

    max_abs_diff = max(float(np.nanmax(np.abs(diff_values))), 1e-6)
    norm = Normalize(vmin=-max_abs_diff, vmax=max_abs_diff)

    scatter = ax.scatter(
        x_values,
        y_values,
        c=diff_values,
        cmap="coolwarm",
        norm=norm,
        s=36,
        alpha=0.62,
        linewidths=0,
        rasterized=True,
        zorder=2,
    )

    all_values = np.concatenate([x_values, y_values])
    axis_min = max(0.0, float(np.nanmin(all_values)) - 0.025)
    axis_max = min(1.005, float(np.nanmax(all_values)) + 0.025)

    if math.isclose(axis_min, axis_max):
        axis_min = max(0.0, axis_min - 0.02)
        axis_max = min(1.005, axis_max + 0.02)

    ax.plot(
        [axis_min, axis_max],
        [axis_min, axis_max],
        linestyle="--",
        linewidth=1.5,
        color="0.25",
        zorder=1,
    )

    ax.set_xlim(axis_min, axis_max)
    ax.set_ylim(axis_min, axis_max)
    ax.set_aspect("equal", adjustable="box")
    ax.set_xlabel(f"{reference_display_name} per-motif accuracy")
    ax.set_ylabel(f"{target_display_name} per-motif accuracy")
    apply_common_axis_style(ax, grid_axis="both")

    colorbar = ax.figure.colorbar(
        scatter,
        ax=ax,
        fraction=0.046,
        pad=0.04,
    )
    colorbar.set_label(
        "Accuracy difference",
        fontsize=AXIS_LABEL_FONT_SIZE,
    )
    colorbar.ax.tick_params(
        labelsize=TICK_LABEL_FONT_SIZE,
        width=TICK_LINE_WIDTH,
        length=TICK_LENGTH,
    )
    colorbar.outline.set_linewidth(AXIS_LINE_WIDTH)


def plot_figure3ef(
    complete_df: pd.DataFrame,
    selected_model: str,
    top_n: int,
    figure_dir: Path,
    plot_combined: bool = False,
) -> List[Path]:
    generated: List[Path] = []

    filename_map = {
        "5X": "Figure3E_ATCGXY_6-Class_vs_ATCGX_5-Class",
        "5Y": "Figure3F_ATCGXY_6-Class_vs_ATCGY_5-Class",
    }

    for target_task in ["5X", "5Y"]:
        fig, ax = plt.subplots(figsize=FIGURE3EF_FIGSIZE)
        draw_paired_scatter(
            ax=ax,
            data=complete_df,
            target_task=target_task,
            top_n=top_n,
        )
        fig.tight_layout()

        output_stem = (
            figure_dir
            / f"{filename_map[target_task]}_{safe_name(selected_model)}"
        )
        save_figure_all_formats(fig, output_stem)
        plt.close(fig)
        generated.append(output_stem.with_suffix(".png"))

    if plot_combined:
        fig, axes = plt.subplots(1, 2, figsize=(15.6, 6.5))
        draw_paired_scatter(
            ax=axes[0], data=complete_df, target_task="5X", top_n=top_n
        )
        axes[0].set_title("ATCGXY 6-Class vs ATCGX 5-Class", pad=10)
        draw_paired_scatter(
            ax=axes[1], data=complete_df, target_task="5Y", top_n=top_n
        )
        axes[1].set_title("ATCGXY 6-Class vs ATCGY 5-Class", pad=10)
        fig.tight_layout()
        output_stem = (
            figure_dir
            / f"Figure3EF_combined_paired_motif_accuracy_{safe_name(selected_model)}"
        )
        save_figure_all_formats(fig, output_stem)
        plt.close(fig)
        generated.append(output_stem.with_suffix(".png"))

    return generated


# =============================================================================
# 7. 图注说明和输出清单
# =============================================================================

def write_caption_notes(
    output_dir: Path,
    selected_model: str,
) -> None:
    text = f"""Figure 3C
Per-motif accuracy distributions for the ATCGXY 6-Class, ATCGX 5-Class, and ATCGY 5-Class tasks evaluated using {selected_model}. The empirical cumulative distribution function summarizes all available 6-mer motifs. Vertical dashed lines indicate accuracies of 0.90, 0.95, and 0.98. The legend reports the median accuracy and the proportion of motifs with accuracy ≥ 0.95.

Figure 3E/3F
Each point represents the same 6-mer motif evaluated using {selected_model}. The dashed diagonal denotes y = x, and point color represents the five-class accuracy minus the ATCGXY 6-Class accuracy. Because the three tasks differ in sample size and class definition, these comparisons are descriptive and should not be interpreted as strictly paired ablation experiments.

中文说明
Figure 3E/3F 中每个点代表同一个 6-mer motif。由于 ATCGXY 6-Class、ATCGX 5-Class 和 ATCGY 5-Class 三个任务的样本量和类别定义不同，该比较仅用于描述任务间 motif 性能差异，不等同于严格的配对消融实验。
"""

    with open(
        output_dir / "Figure3C_3E_3F_caption_notes.txt",
        "w",
        encoding="utf-8",
    ) as file_handle:
        file_handle.write(text)


def write_manifest(
    generated_files: Iterable[Path],
    output_path: Path,
) -> None:
    rows = []

    for png_path in generated_files:
        stem = png_path.with_suffix("")
        rows.append({
            "figure": stem.name,
            "png": str(stem.with_suffix(".png")),
            "pdf": str(stem.with_suffix(".pdf")),
            "svg": str(stem.with_suffix(".svg")),
        })

    pd.DataFrame(rows).to_csv(output_path, index=False)


# =============================================================================
# 8. 参数和主流程
# =============================================================================

def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Read existing Figure 3 evaluation CSV files and redraw "
            "Figure 3B, 3C, 3E and 3F without rerunning model inference."
        )
    )

    parser.add_argument(
        "--result_root",
        required=True,
        help="Evaluation result root containing combined_data/.",
    )
    parser.add_argument(
        "--overall_csv",
        default=None,
        help=(
            "总体指标长表。默认读取 result_root/combined_data/"
            "figure3_all_tasks_overall_metrics_long.csv。"
        ),
    )
    parser.add_argument(
        "--motif_csv",
        default=None,
        help=(
            "逐 motif 指标长表。默认读取 result_root/combined_data/"
            "figure3_all_tasks_per_motif_metrics_long.csv。"
        ),
    )
    parser.add_argument(
        "--figure_dir",
        default=None,
        help=(
            "图片输出目录。默认 result_root/figures。"
        ),
    )
    parser.add_argument(
        "--replot_data_dir",
        default=None,
        help=(
            "本次重新整理的绘图数据输出目录。默认 "
            "result_root/combined_data/figure3_plot_data。"
        ),
    )
    parser.add_argument(
        "--motif_model",
        choices=MODEL_ORDER,
        default="LiteMamba",
        help="Figure 3C、Figure 3E 和 Figure 3F 使用的模型。默认 LiteMamba。",
    )
    parser.add_argument(
        "--top_outliers",
        type=int,
        default=8,
        help="导出 Figure 3E/3F 中偏离对角线最大的 motif 数量；图中不显示 motif 名称。默认 8。",
    )
    parser.add_argument(
        "--plot_combined_ef",
        action="store_true",
        help="另外输出 Figure 3E/3F 的双面板组合图；默认只输出正文单图。",
    )

    args = parser.parse_args()

    if args.top_outliers < 0:
        raise ValueError("--top_outliers 必须 >= 0")

    return args


def main() -> None:
    configure_matplotlib()
    args = parse_args()

    result_root = Path(args.result_root)
    combined_dir = result_root / "combined_data"

    requested_overall_csv = (
        Path(args.overall_csv)
        if args.overall_csv
        else combined_dir / "figure3_all_tasks_overall_metrics_long.csv"
    )
    requested_motif_csv = (
        Path(args.motif_csv)
        if args.motif_csv
        else combined_dir / "figure3_all_tasks_per_motif_metrics_long.csv"
    )

    overall_csv = locate_input_csv(
        explicitly_requested=requested_overall_csv,
        fallback_paths=[
            combined_dir / "available_tasks_overall_metrics_long.csv",
            result_root / "combined_data" / "figure3B_plot_data_long.csv",
        ],
        description="总体指标 CSV",
    )
    motif_csv = locate_input_csv(
        explicitly_requested=requested_motif_csv,
        fallback_paths=[
            combined_dir / "available_tasks_per_motif_metrics_long.csv",
        ],
        description="逐 motif 指标 CSV",
    )

    figure_dir = ensure_dir(
        Path(args.figure_dir)
        if args.figure_dir
        else result_root / "figures"
    )
    output_data_dir = ensure_dir(
        Path(args.replot_data_dir)
        if args.replot_data_dir
        else combined_dir / "figure3_plot_data"
    )

    print("=" * 110)
    print("Redraw Figure 3B, 3C, 3E and 3F from existing CSV files")
    print(f"Overall CSV       : {overall_csv}")
    print(f"Per-motif CSV     : {motif_csv}")
    print(f"Figure directory  : {figure_dir}")
    print(f"Replot data dir   : {output_data_dir}")
    print(f"Figure 3C/3E/3F model: {args.motif_model}")
    print(f"Figure 3B y range : {FIGURE3B_YMIN:.2f}–{FIGURE3B_YMAX:.3f}")
    print(f"Figure 3B single : {FIGURE3B_SINGLE_FIGSIZE}")
    print(f"Figure 3B summary: {FIGURE3B_SUMMARY_FIGSIZE}")
    print(f"Figure 3C size   : {FIGURE3C_FIGSIZE}")
    print(f"Font family      : {FONT_FAMILY}")
    print(
        "Font sizes       : "
        f"base={BASE_FONT_SIZE}, axis={AXIS_LABEL_FONT_SIZE}, "
        f"tick={TICK_LABEL_FONT_SIZE}, legend={LEGEND_FONT_SIZE}"
    )
    print("=" * 110)

    overall_df = load_overall_metrics(overall_csv)
    motif_df = load_per_motif_metrics(motif_csv)

    # 保存一份已经统一正式任务名称的基础数据。
    overall_export = overall_df.copy()
    overall_export["task"] = overall_export["task"].astype(str)
    overall_export["model"] = overall_export["model"].astype(str)
    overall_export.to_csv(
        output_data_dir / "overall_metrics_with_updated_task_names.csv",
        index=False,
    )

    motif_export = motif_df.copy()
    motif_export["task"] = motif_export["task"].astype(str)
    motif_export["model"] = motif_export["model"].astype(str)
    motif_export.to_csv(
        output_data_dir / "per_motif_metrics_with_updated_task_names.csv",
        index=False,
    )

    generated_files: List[Path] = []

    figure3b_df = prepare_figure3b_data(
        overall_df=overall_df,
        output_data_dir=output_data_dir,
    )
    generated_files.extend(
        plot_figure3b(
            plot_df=figure3b_df,
            figure_dir=figure_dir,
        )
    )

    ecdf_df, ecdf_summary_df = prepare_figure3c_data(
        motif_df=motif_df,
        selected_model=args.motif_model,
        output_data_dir=output_data_dir,
    )
    generated_files.append(
        plot_figure3c(
            ecdf_df=ecdf_df,
            summary_df=ecdf_summary_df,
            selected_model=args.motif_model,
            figure_dir=figure_dir,
        )
    )

    paired_complete_df, _ = prepare_figure3ef_data(
        motif_df=motif_df,
        selected_model=args.motif_model,
        output_data_dir=output_data_dir,
        top_n=args.top_outliers,
    )
    generated_files.extend(
        plot_figure3ef(
            complete_df=paired_complete_df,
            selected_model=args.motif_model,
            top_n=args.top_outliers,
            figure_dir=figure_dir,
            plot_combined=args.plot_combined_ef,
        )
    )

    write_caption_notes(
        output_dir=figure_dir,
        selected_model=args.motif_model,
    )
    write_manifest(
        generated_files=generated_files,
        output_path=figure_dir / "figure3_bcef_manifest.csv",
    )

    print("\n" + "=" * 110)
    print("全部图片已重新绘制完成。")
    print(f"图片目录     : {figure_dir}")
    print(f"重绘数据目录 : {output_data_dir}")
    print(f"完整配对 motif 数量: {len(paired_complete_df)}")
    print("输出格式     : PNG / PDF / SVG")
    print("=" * 110)


if __name__ == "__main__":
    main()