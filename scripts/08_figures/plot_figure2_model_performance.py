#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Redraw Figure 2 model-performance panels from saved evaluation tables.

This is a plotting-only script. It does not load a neural-network checkpoint or
rerun inference. The required input is the ``global_summary.csv`` produced by
``scripts/07_evaluation/evaluate_binary_litemamba.py``. Optional prediction CSV
files enable ROC and precision-recall curves.

The script reproduces:

* normalized confusion matrices (optional);
* Figure 2B X, Y, and combined overall-performance panels;
* Figure 2B X, Y, and combined ROC/PR curves;
* Figure 2F separate X and Y accuracy-versus-remaining-rate panels;
* optional per-threshold and multi-threshold ROC/PR curves.
"""

import argparse
import csv
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np

import matplotlib
matplotlib.use("Agg")
import matplotlib as mpl
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from matplotlib.ticker import MultipleLocator, FormatStrFormatter

from sklearn.metrics import (
    average_precision_score,
    precision_recall_curve,
    roc_auc_score,
    roc_curve,
)


# =============================================================================
# 0) 全局绘图风格
# =============================================================================
mpl.rcParams["font.family"] = "Arial"
mpl.rcParams["font.sans-serif"] = ["Arial"]
mpl.rcParams["pdf.fonttype"] = 42
mpl.rcParams["ps.fonttype"] = 42
mpl.rcParams["axes.unicode_minus"] = False


# =============================================================================
# 1) Portable defaults and plotting controls
# =============================================================================
POS_DATASET_STEM = "xna_pc_6mer_features_pos_balanced_test"
NEG_DATASET_STEM = "xna_pc_6mer_features_neg_balanced_test"

PLOT_CONFUSION_MATRICES = False
PLOT_FIGURE2B_OVERALL = True
PLOT_FIGURE2F = True
PLOT_FIGURE2B_ROC_PR = True
PLOT_PER_THRESHOLD_ROC_PR = False
PLOT_MULTI_THRESHOLD_ROC_PR = False

FIGURE2B_CONDITION = "argmax_all"
MIN_KEPT_FOR_CURVE = 20


# =============================================================================
# 3) 与原脚本保持一致的绘图参数
# =============================================================================

# -------------------------
# 混淆矩阵
# -------------------------
CM_FIGSIZE = (8, 7)
CM_AXIS_LABEL_FONTSIZE = 30
CM_TICK_FONTSIZE = 30
CM_TEXT_FONTSIZE = 40
CM_CBAR_FONTSIZE = 30
CM_SPINE_LINEWIDTH = 3
CM_DPI = 600

# -------------------------
# ROC / PR
# -------------------------
CURVE_FIGSIZE = (8, 8)
CURVE_AXIS_LABEL_FONTSIZE = 25
CURVE_TICK_FONTSIZE = 25
CURVE_TITLE_FONTSIZE = 22
CURVE_LEGEND_FONTSIZE = 22
CURVE_LINEWIDTH = 3
CURVE_SPINE_LINEWIDTH = 3
CURVE_DPI = 600

ROC_XMIN = 0.0
ROC_XMAX = 0.4
ROC_YMIN = 0.6
ROC_YMAX = 1.0

PR_XMIN = 0.6
PR_XMAX = 1.0
PR_YMIN = 0.6
PR_YMAX = 1.0

CURVE_TICK_STEP = 0.1

# -------------------------
# Figure 2F
# -------------------------
FIG2F_FIGSIZE = (4.6, 6.0)
FIG2F_AXIS_LABEL_FONTSIZE = 25
FIG2F_TICK_FONTSIZE = 22
FIG2F_LEGEND_FONTSIZE = 19
FIG2F_ANNOTATION_FONTSIZE = 14
FIG2F_LINEWIDTH = 3.5
FIG2F_MARKERSIZE = 10
FIG2F_DPI = 600

FIG2F_COLORS = {
    "X": "#377DB8",
    "Y": "#D98484",
}

FIG2F_ACCURACY_YMIN = 0.97
FIG2F_ACCURACY_YMAX = 0.99

# -------------------------
# Figure 2B
# -------------------------
FIG2B_SINGLE_FIGSIZE = (13.5, 6.8)
FIG2B_COMBINED_FIGSIZE = (13.5, 6.8)

FIG2B_AXIS_LABEL_FONTSIZE = 24
FIG2B_TICK_FONTSIZE = 20
FIG2B_XTICK_FONTSIZE = 22
FIG2B_TEXT_FONTSIZE = 14
FIG2B_SUBPANEL_FONTSIZE = 19
FIG2B_LINEWIDTH = 2.8
FIG2B_POINTSIZE = 180
FIG2B_DPI = 600

FIG2B_COLORS = {
    "POS_X": "#377DB8",
    "NEG_Y": "#D98484",
}

# Combined 图中 X/Y 数值标签的偏移量，单位为 points。
# 只想上下移动时，把第一个数设为 0。
FIG2B_X_LABEL_OFFSET = (-4, 4)
FIG2B_Y_LABEL_OFFSET = (4, -4)

# X/Y 两组棒棒糖在每列中心左右分开的距离。
FIG2B_COMBINED_X_OFFSET = 0.15

LABEL_COLORS_BINARY = {
    0: "#377DB8",
    1: "#D98484",
}


# =============================================================================
# 4) 通用工具函数
# =============================================================================
def safe_float(value: Any) -> float:
    try:
        result = float(value)
    except (TypeError, ValueError):
        return float("nan")

    if not np.isfinite(result):
        return float("nan")
    return result


def safe_int(value: Any, default: int = 0) -> int:
    try:
        if value is None or str(value).strip() == "":
            return default
        return int(float(value))
    except (TypeError, ValueError):
        return default


def safe_div(numerator: float, denominator: float) -> float:
    if denominator == 0:
        return 0.0
    return float(numerator) / float(denominator)


def numeric_threshold(value: Any) -> Optional[float]:
    if value is None:
        return None

    text = str(value).strip().lower()
    if text in {"", "none", "nan"}:
        return None

    try:
        threshold = float(text)
    except ValueError:
        return None

    if not np.isfinite(threshold):
        return None
    return threshold


def condition_name(threshold: Optional[float]) -> str:
    if threshold is None:
        return "argmax_all"
    return f"T{threshold:.2f}".replace(".", "p")


def load_csv_rows(csv_path: Path) -> List[Dict[str, str]]:
    csv_path = Path(csv_path)
    if not csv_path.exists():
        raise FileNotFoundError(f"输入 CSV 不存在: {csv_path}")

    with open(csv_path, "r", encoding="utf-8-sig", newline="") as file:
        return list(csv.DictReader(file))


def style_axes_frame(
    ax,
    spine_lw: float = 3.0,
    tick_lw: float = 3.0,
    tick_len: float = 6.0,
):
    ax.tick_params(
        axis="both",
        direction="out",
        length=tick_len,
        width=tick_lw,
        colors="black",
    )
    for side in ["left", "right", "top", "bottom"]:
        ax.spines[side].set_linewidth(spine_lw)
        ax.spines[side].set_color("black")


def force_ticks(
    ax,
    xlim: Tuple[float, float],
    ylim: Tuple[float, float],
    step: float = 0.1,
):
    ax.set_xlim(*xlim)
    ax.set_ylim(*ylim)

    xticks = np.round(
        np.arange(xlim[0], xlim[1] + 1e-9, step),
        2,
    )
    yticks = np.round(
        np.arange(ylim[0], ylim[1] + 1e-9, step),
        2,
    )

    ax.set_xticks(xticks)
    ax.set_yticks(yticks)
    ax.xaxis.set_major_formatter(FormatStrFormatter("%.1f"))
    ax.yaxis.set_major_formatter(FormatStrFormatter("%.1f"))
    ax.xaxis.set_major_locator(MultipleLocator(step))
    ax.yaxis.set_major_locator(MultipleLocator(step))


def fix_corner_ticklabel_overlap(ax):
    ax.tick_params(axis="x", pad=10)
    ax.tick_params(axis="y", pad=10)

    figure = ax.figure
    figure.canvas.draw()

    xlabels = ax.get_xticklabels()
    ylabels = ax.get_yticklabels()

    if xlabels:
        xlabels[0].set_horizontalalignment("left")
    if ylabels:
        ylabels[0].set_verticalalignment("bottom")


def save_figure_all_formats(
    fig,
    out_dir: Path,
    stem: str,
    dpi: int = 600,
) -> Dict[str, str]:
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    png_path = out_dir / f"{stem}.png"
    pdf_path = out_dir / f"{stem}.pdf"
    svg_path = out_dir / f"{stem}.svg"

    fig.savefig(png_path, dpi=dpi, bbox_inches="tight")
    fig.savefig(pdf_path, bbox_inches="tight")
    fig.savefig(svg_path, bbox_inches="tight")

    return {
        "png": str(png_path),
        "pdf": str(pdf_path),
        "svg": str(svg_path),
    }


# =============================================================================
# 5) 读取 predictions.csv
# =============================================================================
def load_predictions_csv(
    csv_path: Path,
) -> Tuple[np.ndarray, np.ndarray]:
    csv_path = Path(csv_path)
    if not csv_path.exists():
        raise FileNotFoundError(f"predictions.csv 不存在: {csv_path}")

    y_true_list: List[int] = []
    prob_list: List[Tuple[float, float]] = []

    with open(csv_path, "r", encoding="utf-8-sig", newline="") as file:
        reader = csv.DictReader(file)

        required_columns = {"y_true", "prob_0", "prob_1"}
        fieldnames = set(reader.fieldnames or [])
        missing = required_columns - fieldnames
        if missing:
            raise ValueError(
                f"{csv_path} 缺少列: {sorted(missing)}"
            )

        for row in reader:
            try:
                y_true = int(float(row["y_true"]))
                prob_0 = float(row["prob_0"])
                prob_1 = float(row["prob_1"])
            except (TypeError, ValueError):
                continue

            if y_true not in (0, 1):
                continue
            if not np.isfinite(prob_0) or not np.isfinite(prob_1):
                continue

            y_true_list.append(y_true)
            prob_list.append((prob_0, prob_1))

    if not y_true_list:
        raise RuntimeError(f"没有从 predictions.csv 读取到有效数据: {csv_path}")

    y_true_array = np.asarray(y_true_list, dtype=np.int64)
    y_prob_array = np.asarray(prob_list, dtype=np.float64)

    return y_true_array, y_prob_array


# =============================================================================
# 6) 混淆矩阵
# =============================================================================
def plot_confusion_matrix_from_counts(
    tn: int,
    fp: int,
    fn: int,
    tp: int,
    out_dir: Path,
    stem: str,
    label_names: Dict[int, str],
):
    confusion_raw = np.asarray(
        [[tn, fp], [fn, tp]],
        dtype=np.float64,
    )

    row_sum = confusion_raw.sum(axis=1, keepdims=True)
    with np.errstate(divide="ignore", invalid="ignore"):
        confusion_norm = confusion_raw / row_sum
    confusion_norm = np.nan_to_num(confusion_norm)

    fig, ax = plt.subplots(figsize=CM_FIGSIZE)

    image = ax.imshow(
        confusion_norm,
        interpolation="nearest",
        cmap=plt.cm.Blues,
        vmin=0.0,
        vmax=1.0,
    )

    colorbar = fig.colorbar(image, ax=ax)
    colorbar.ax.tick_params(
        labelsize=CM_CBAR_FONTSIZE,
        width=3,
        length=6,
        colors="black",
    )
    colorbar.set_label(
        "Proportion",
        fontsize=CM_CBAR_FONTSIZE,
    )
    try:
        colorbar.outline.set_linewidth(3)
        colorbar.outline.set_edgecolor("black")
    except Exception:
        pass

    for spine in colorbar.ax.spines.values():
        spine.set_linewidth(3)
        spine.set_color("black")

    for row_index in range(2):
        for col_index in range(2):
            value = confusion_norm[row_index, col_index]
            text_color = "white" if value > 0.5 else "black"
            ax.text(
                col_index,
                row_index,
                f"{value:.2f}",
                ha="center",
                va="center",
                fontsize=CM_TEXT_FONTSIZE,
                fontweight="bold",
                color=text_color,
            )

    display_labels = [
        label_names[0],
        label_names[1],
    ]

    ax.set_xticks([0, 1])
    ax.set_xticklabels(
        display_labels,
        rotation=45,
        ha="right",
        fontsize=CM_TICK_FONTSIZE,
    )
    ax.set_yticks([0, 1])
    ax.set_yticklabels(
        display_labels,
        fontsize=CM_TICK_FONTSIZE,
    )

    for index, tick in enumerate(ax.get_xticklabels()):
        tick.set_color(LABEL_COLORS_BINARY[index])
    for index, tick in enumerate(ax.get_yticklabels()):
        tick.set_color(LABEL_COLORS_BINARY[index])

    ax.set_xlabel(
        "Predicted Label",
        fontsize=CM_AXIS_LABEL_FONTSIZE,
    )
    ax.set_ylabel(
        "True Label",
        fontsize=CM_AXIS_LABEL_FONTSIZE,
    )

    style_axes_frame(
        ax,
        spine_lw=CM_SPINE_LINEWIDTH,
        tick_lw=3,
        tick_len=6,
    )

    plt.tight_layout()
    paths = save_figure_all_formats(
        fig=fig,
        out_dir=out_dir,
        stem=stem,
        dpi=CM_DPI,
    )
    plt.close(fig)

    return paths


def generate_confusion_matrices(
    global_rows: List[Dict[str, str]],
    output_root: Path,
):
    output_root = Path(output_root)

    count = 0
    for row in global_rows:
        if row.get("status") != "OK":
            continue

        required = ["TN", "FP", "FN", "TP"]
        if any(str(row.get(key, "")).strip() == "" for key in required):
            continue

        group = str(row.get("group", "")).strip()
        condition = str(row.get("condition", "")).strip()
        csv_file = str(row.get("csv_file", "dataset.csv")).strip()
        dataset_stem = Path(csv_file).stem

        label_names = {
            0: str(row.get("label0_name", "CB")),
            1: str(row.get("label1_name", "X" if group == "POS_X" else "Y")),
        }

        tn = safe_int(row.get("TN"))
        fp = safe_int(row.get("FP"))
        fn = safe_int(row.get("FN"))
        tp = safe_int(row.get("TP"))

        out_dir = (
            output_root
            / group
            / dataset_stem
            / "confusion_matrices"
        )
        stem = f"{dataset_stem}_{condition}_confusion_matrix_norm"

        plot_confusion_matrix_from_counts(
            tn=tn,
            fp=fp,
            fn=fn,
            tp=tp,
            out_dir=out_dir,
            stem=stem,
            label_names=label_names,
        )
        count += 1

    print(f"  ✓ 已重新绘制 {count} 张归一化混淆矩阵")


# =============================================================================
# 7) ROC / PR 绘图
# =============================================================================
def plot_single_roc_pr(
    y_true: np.ndarray,
    y_prob: np.ndarray,
    out_dir: Path,
    stem_prefix: str,
    positive_name: str,
    title_prefix: str,
    line_color: str,
):
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    y_true = np.asarray(y_true, dtype=np.int64)
    y_prob = np.asarray(y_prob, dtype=np.float64)
    y_score = y_prob[:, 1]

    # ROC
    fig, ax = plt.subplots(figsize=CURVE_FIGSIZE)

    if len(np.unique(y_true)) == 2:
        fpr, tpr, _ = roc_curve(y_true, y_score)
        auc_value = roc_auc_score(y_true, y_score)
        ax.plot(
            fpr,
            tpr,
            linewidth=CURVE_LINEWIDTH,
            color=line_color,
            label=f"{positive_name} AUC={auc_value:.3f}",
        )
    else:
        ax.text(
            0.5,
            0.5,
            "Only one class retained\nROC is undefined",
            ha="center",
            va="center",
            fontsize=18,
        )

    ax.plot(
        [0, 1],
        [0, 1],
        linestyle="--",
        linewidth=1.5,
        color="gray",
        alpha=0.8,
    )
    ax.set_xlabel(
        "False Positive Rate",
        fontsize=CURVE_AXIS_LABEL_FONTSIZE,
    )
    ax.set_ylabel(
        "True Positive Rate",
        fontsize=CURVE_AXIS_LABEL_FONTSIZE,
    )
    ax.set_title(
        f"{title_prefix} ROC",
        fontsize=CURVE_TITLE_FONTSIZE,
        fontweight="bold",
        pad=12,
    )

    force_ticks(
        ax,
        xlim=(ROC_XMIN, ROC_XMAX),
        ylim=(ROC_YMIN, ROC_YMAX),
        step=CURVE_TICK_STEP,
    )

    try:
        ax.set_box_aspect(1)
    except Exception:
        ax.set_aspect("equal", adjustable="box")

    ax.tick_params(
        direction="out",
        length=6,
        width=3,
        labelsize=CURVE_TICK_FONTSIZE,
    )

    for spine in ax.spines.values():
        spine.set_linewidth(CURVE_SPINE_LINEWIDTH)

    fix_corner_ticklabel_overlap(ax)
    ax.legend(
        loc="lower right",
        prop={"size": CURVE_LEGEND_FONTSIZE},
        frameon=False,
    )

    plt.tight_layout()
    roc_paths = save_figure_all_formats(
        fig=fig,
        out_dir=out_dir,
        stem=f"{stem_prefix}_ROC",
        dpi=CURVE_DPI,
    )
    plt.close(fig)

    # PR
    fig, ax = plt.subplots(figsize=CURVE_FIGSIZE)

    if len(np.unique(y_true)) == 2:
        precision, recall, _ = precision_recall_curve(
            y_true,
            y_score,
        )
        ap_value = average_precision_score(
            y_true,
            y_score,
        )
        ax.plot(
            recall,
            precision,
            linewidth=CURVE_LINEWIDTH,
            color=line_color,
            label=f"{positive_name} AP={ap_value:.3f}",
        )
    else:
        ax.text(
            0.5,
            0.5,
            "Only one class retained\nPR is undefined",
            ha="center",
            va="center",
            fontsize=18,
        )

    ax.set_xlabel(
        "Recall",
        fontsize=CURVE_AXIS_LABEL_FONTSIZE,
    )
    ax.set_ylabel(
        "Precision",
        fontsize=CURVE_AXIS_LABEL_FONTSIZE,
    )
    ax.set_title(
        f"{title_prefix} PR",
        fontsize=CURVE_TITLE_FONTSIZE,
        fontweight="bold",
        pad=12,
    )

    force_ticks(
        ax,
        xlim=(PR_XMIN, PR_XMAX),
        ylim=(PR_YMIN, PR_YMAX),
        step=CURVE_TICK_STEP,
    )

    try:
        ax.set_box_aspect(1)
    except Exception:
        ax.set_aspect("equal", adjustable="box")

    ax.tick_params(
        direction="out",
        length=6,
        width=3,
        labelsize=CURVE_TICK_FONTSIZE,
    )

    for spine in ax.spines.values():
        spine.set_linewidth(CURVE_SPINE_LINEWIDTH)

    fix_corner_ticklabel_overlap(ax)
    ax.legend(
        loc="lower left",
        prop={"size": CURVE_LEGEND_FONTSIZE},
        frameon=False,
    )

    plt.tight_layout()
    pr_paths = save_figure_all_formats(
        fig=fig,
        out_dir=out_dir,
        stem=f"{stem_prefix}_PR",
        dpi=CURVE_DPI,
    )
    plt.close(fig)

    return {
        "roc": roc_paths,
        "pr": pr_paths,
    }


def collect_thresholds(
    global_rows: List[Dict[str, str]],
    group: str,
) -> List[Optional[float]]:
    thresholds: List[Optional[float]] = []

    for row in global_rows:
        if row.get("group") != group:
            continue
        if row.get("status") != "OK":
            continue

        threshold = numeric_threshold(row.get("threshold"))
        if threshold is None:
            if row.get("condition") == "argmax_all":
                thresholds.append(None)
        else:
            thresholds.append(threshold)

    numeric_values = sorted({
        float(value)
        for value in thresholds
        if value is not None
    })

    result: List[Optional[float]] = [None]
    result.extend(numeric_values)
    return result


def apply_confidence_threshold(
    y_true_all: np.ndarray,
    y_prob_all: np.ndarray,
    threshold: Optional[float],
) -> Tuple[np.ndarray, np.ndarray]:
    if threshold is None:
        mask = np.ones(y_true_all.shape[0], dtype=bool)
    else:
        max_probability = np.max(y_prob_all, axis=1)
        mask = max_probability >= float(threshold)

    return y_true_all[mask], y_prob_all[mask]


def plot_multi_threshold_roc_pr(
    y_true_all: np.ndarray,
    y_prob_all: np.ndarray,
    thresholds: List[Optional[float]],
    out_dir: Path,
    stem_prefix: str,
    positive_name: str,
    title_prefix: str,
):
    total = int(y_true_all.size)

    # ROC
    fig, ax = plt.subplots(figsize=CURVE_FIGSIZE)
    plotted = 0

    for threshold in thresholds:
        y_true, y_prob = apply_confidence_threshold(
            y_true_all,
            y_prob_all,
            threshold,
        )

        kept = int(y_true.size)
        if kept < MIN_KEPT_FOR_CURVE:
            continue
        if len(np.unique(y_true)) < 2:
            continue

        fpr, tpr, _ = roc_curve(
            y_true,
            y_prob[:, 1],
        )
        auc_value = roc_auc_score(
            y_true,
            y_prob[:, 1],
        )
        keep_ratio = safe_div(kept, total) * 100.0

        threshold_label = (
            "All"
            if threshold is None
            else f"T≥{threshold:.2f}"
        )

        ax.plot(
            fpr,
            tpr,
            linewidth=CURVE_LINEWIDTH,
            label=(
                f"{threshold_label} | "
                f"keep {keep_ratio:.1f}% | "
                f"AUC {auc_value:.3f}"
            ),
        )
        plotted += 1

    ax.plot(
        [0, 1],
        [0, 1],
        linestyle="--",
        linewidth=1.5,
        color="gray",
        alpha=0.8,
    )

    if plotted == 0:
        ax.text(
            0.5,
            0.5,
            "No valid ROC curve",
            ha="center",
            va="center",
            fontsize=18,
        )

    ax.set_xlabel(
        "False Positive Rate",
        fontsize=CURVE_AXIS_LABEL_FONTSIZE,
    )
    ax.set_ylabel(
        "True Positive Rate",
        fontsize=CURVE_AXIS_LABEL_FONTSIZE,
    )
    ax.set_title(
        f"{title_prefix} ROC ({positive_name})",
        fontsize=CURVE_TITLE_FONTSIZE,
        fontweight="bold",
        pad=12,
    )

    force_ticks(
        ax,
        xlim=(ROC_XMIN, ROC_XMAX),
        ylim=(ROC_YMIN, ROC_YMAX),
        step=CURVE_TICK_STEP,
    )

    try:
        ax.set_box_aspect(1)
    except Exception:
        ax.set_aspect("equal", adjustable="box")

    ax.tick_params(
        direction="out",
        length=6,
        width=3,
        labelsize=CURVE_TICK_FONTSIZE,
    )
    for spine in ax.spines.values():
        spine.set_linewidth(CURVE_SPINE_LINEWIDTH)

    fix_corner_ticklabel_overlap(ax)
    ax.legend(
        loc="lower right",
        prop={"size": CURVE_LEGEND_FONTSIZE},
        frameon=False,
    )

    plt.tight_layout()
    save_figure_all_formats(
        fig=fig,
        out_dir=out_dir,
        stem=f"{stem_prefix}_ALL_thresholds_ROC",
        dpi=CURVE_DPI,
    )
    plt.close(fig)

    # PR
    fig, ax = plt.subplots(figsize=CURVE_FIGSIZE)
    plotted = 0

    for threshold in thresholds:
        y_true, y_prob = apply_confidence_threshold(
            y_true_all,
            y_prob_all,
            threshold,
        )

        kept = int(y_true.size)
        if kept < MIN_KEPT_FOR_CURVE:
            continue
        if len(np.unique(y_true)) < 2:
            continue

        precision, recall, _ = precision_recall_curve(
            y_true,
            y_prob[:, 1],
        )
        ap_value = average_precision_score(
            y_true,
            y_prob[:, 1],
        )
        keep_ratio = safe_div(kept, total) * 100.0

        threshold_label = (
            "All"
            if threshold is None
            else f"T≥{threshold:.2f}"
        )

        ax.plot(
            recall,
            precision,
            linewidth=CURVE_LINEWIDTH,
            label=(
                f"{threshold_label} | "
                f"keep {keep_ratio:.1f}% | "
                f"AP {ap_value:.3f}"
            ),
        )
        plotted += 1

    if plotted == 0:
        ax.text(
            0.5,
            0.5,
            "No valid PR curve",
            ha="center",
            va="center",
            fontsize=18,
        )

    ax.set_xlabel(
        "Recall",
        fontsize=CURVE_AXIS_LABEL_FONTSIZE,
    )
    ax.set_ylabel(
        "Precision",
        fontsize=CURVE_AXIS_LABEL_FONTSIZE,
    )
    ax.set_title(
        f"{title_prefix} PR ({positive_name})",
        fontsize=CURVE_TITLE_FONTSIZE,
        fontweight="bold",
        pad=12,
    )

    force_ticks(
        ax,
        xlim=(PR_XMIN, PR_XMAX),
        ylim=(PR_YMIN, PR_YMAX),
        step=CURVE_TICK_STEP,
    )

    try:
        ax.set_box_aspect(1)
    except Exception:
        ax.set_aspect("equal", adjustable="box")

    ax.tick_params(
        direction="out",
        length=6,
        width=3,
        labelsize=CURVE_TICK_FONTSIZE,
    )
    for spine in ax.spines.values():
        spine.set_linewidth(CURVE_SPINE_LINEWIDTH)

    fix_corner_ticklabel_overlap(ax)
    ax.legend(
        loc="lower left",
        prop={"size": CURVE_LEGEND_FONTSIZE},
        frameon=False,
    )

    plt.tight_layout()
    save_figure_all_formats(
        fig=fig,
        out_dir=out_dir,
        stem=f"{stem_prefix}_ALL_thresholds_PR",
        dpi=CURVE_DPI,
    )
    plt.close(fig)


def generate_per_threshold_roc_pr(
    global_rows: List[Dict[str, str]],
    group: str,
    y_true_all: np.ndarray,
    y_prob_all: np.ndarray,
    out_dir: Path,
    dataset_stem: str,
    positive_name: str,
):
    thresholds = collect_thresholds(
        global_rows,
        group=group,
    )

    count = 0
    for threshold in thresholds:
        y_true, y_prob = apply_confidence_threshold(
            y_true_all,
            y_prob_all,
            threshold,
        )

        if y_true.size == 0:
            continue

        condition = condition_name(threshold)
        condition_label = (
            "Argmax all"
            if threshold is None
            else f"max prob ≥ {threshold:.2f}"
        )

        plot_single_roc_pr(
            y_true=y_true,
            y_prob=y_prob,
            out_dir=out_dir,
            stem_prefix=f"{dataset_stem}_{condition}",
            positive_name=positive_name,
            title_prefix=f"{dataset_stem} | {condition_label}",
            line_color=(
                FIG2B_COLORS["POS_X"]
                if group == "POS_X"
                else FIG2B_COLORS["NEG_Y"]
            ),
        )
        count += 1

    print(
        f"  ✓ {group} 已重新绘制 "
        f"{count} 组单阈值 ROC/PR"
    )


def plot_combined_xy_roc_pr(
    y_true_x: np.ndarray,
    y_prob_x: np.ndarray,
    y_true_y: np.ndarray,
    y_prob_y: np.ndarray,
    out_dir: Path,
):
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    # Combined ROC
    fig, ax = plt.subplots(figsize=CURVE_FIGSIZE)

    for y_true, y_prob, name, color in [
        (
            y_true_x,
            y_prob_x,
            "X",
            FIG2B_COLORS["POS_X"],
        ),
        (
            y_true_y,
            y_prob_y,
            "Y",
            FIG2B_COLORS["NEG_Y"],
        ),
    ]:
        if len(np.unique(y_true)) < 2:
            continue

        fpr, tpr, _ = roc_curve(
            y_true,
            y_prob[:, 1],
        )
        auc_value = roc_auc_score(
            y_true,
            y_prob[:, 1],
        )

        ax.plot(
            fpr,
            tpr,
            linewidth=CURVE_LINEWIDTH,
            color=color,
            label=f"{name} AUROC={auc_value:.3f}",
        )

    ax.plot(
        [0, 1],
        [0, 1],
        linestyle="--",
        linewidth=1.5,
        color="gray",
        alpha=0.8,
    )
    ax.set_xlabel(
        "False Positive Rate",
        fontsize=CURVE_AXIS_LABEL_FONTSIZE,
    )
    ax.set_ylabel(
        "True Positive Rate",
        fontsize=CURVE_AXIS_LABEL_FONTSIZE,
    )
    ax.set_title(
        "CB (Canonical Base) vs X/Y ROC",
        fontsize=CURVE_TITLE_FONTSIZE,
        fontweight="bold",
        pad=12,
    )

    force_ticks(
        ax,
        xlim=(ROC_XMIN, ROC_XMAX),
        ylim=(ROC_YMIN, ROC_YMAX),
        step=CURVE_TICK_STEP,
    )

    try:
        ax.set_box_aspect(1)
    except Exception:
        ax.set_aspect("equal", adjustable="box")

    ax.tick_params(
        direction="out",
        length=6,
        width=3,
        labelsize=CURVE_TICK_FONTSIZE,
    )

    for spine in ax.spines.values():
        spine.set_linewidth(CURVE_SPINE_LINEWIDTH)

    fix_corner_ticklabel_overlap(ax)
    ax.legend(
        loc="lower right",
        prop={"size": CURVE_LEGEND_FONTSIZE},
        frameon=False,
    )

    plt.tight_layout()
    save_figure_all_formats(
        fig=fig,
        out_dir=out_dir,
        stem="Figure2B_litemamba_XY_combined_ROC",
        dpi=CURVE_DPI,
    )
    plt.close(fig)

    # Combined PR
    fig, ax = plt.subplots(figsize=CURVE_FIGSIZE)

    for y_true, y_prob, name, color in [
        (
            y_true_x,
            y_prob_x,
            "X",
            FIG2B_COLORS["POS_X"],
        ),
        (
            y_true_y,
            y_prob_y,
            "Y",
            FIG2B_COLORS["NEG_Y"],
        ),
    ]:
        if len(np.unique(y_true)) < 2:
            continue

        precision, recall, _ = precision_recall_curve(
            y_true,
            y_prob[:, 1],
        )
        ap_value = average_precision_score(
            y_true,
            y_prob[:, 1],
        )

        ax.plot(
            recall,
            precision,
            linewidth=CURVE_LINEWIDTH,
            color=color,
            label=f"{name} AUPRC={ap_value:.3f}",
        )

    ax.set_xlabel(
        "Recall",
        fontsize=CURVE_AXIS_LABEL_FONTSIZE,
    )
    ax.set_ylabel(
        "Precision",
        fontsize=CURVE_AXIS_LABEL_FONTSIZE,
    )
    ax.set_title(
        "CB (Canonical Base) vs X/Y PR",
        fontsize=CURVE_TITLE_FONTSIZE,
        fontweight="bold",
        pad=12,
    )

    force_ticks(
        ax,
        xlim=(PR_XMIN, PR_XMAX),
        ylim=(PR_YMIN, PR_YMAX),
        step=CURVE_TICK_STEP,
    )

    try:
        ax.set_box_aspect(1)
    except Exception:
        ax.set_aspect("equal", adjustable="box")

    ax.tick_params(
        direction="out",
        length=6,
        width=3,
        labelsize=CURVE_TICK_FONTSIZE,
    )

    for spine in ax.spines.values():
        spine.set_linewidth(CURVE_SPINE_LINEWIDTH)

    fix_corner_ticklabel_overlap(ax)
    ax.legend(
        loc="lower left",
        prop={"size": CURVE_LEGEND_FONTSIZE},
        frameon=False,
    )

    plt.tight_layout()
    save_figure_all_formats(
        fig=fig,
        out_dir=out_dir,
        stem="Figure2B_litemamba_XY_combined_PR",
        dpi=CURVE_DPI,
    )
    plt.close(fig)


# =============================================================================
# 8) Figure 2F
# =============================================================================
def collect_figure2f_rows(
    global_rows: List[Dict[str, str]],
) -> List[Dict[str, Any]]:
    group_to_series = {
        "POS_X": "X",
        "NEG_Y": "Y",
    }

    rows: List[Dict[str, Any]] = []

    for row in global_rows:
        group = str(row.get("group", ""))
        series = group_to_series.get(group)

        if series is None:
            continue
        if row.get("status") != "OK":
            continue

        threshold = numeric_threshold(
            row.get("threshold")
        )
        if threshold is None:
            continue

        accuracy = safe_float(
            row.get("accuracy")
        )
        coverage = safe_float(
            row.get("coverage")
        )

        if not np.isfinite(accuracy):
            continue
        if not np.isfinite(coverage):
            continue

        rows.append({
            "series": series,
            "group": group,
            "threshold": float(threshold),
            "accuracy": float(accuracy),
            "remaining_rate": float(coverage),
        })

    rows.sort(
        key=lambda item: (
            item["series"],
            item["threshold"],
        )
    )
    return rows


def auto_figure2f_ylim(
    values: np.ndarray,
    requested_min: Optional[float] = None,
    requested_max: Optional[float] = None,
    lower_bound: float = 0.0,
    upper_bound: float = 1.0,
) -> Tuple[float, float]:
    values = np.asarray(values, dtype=float)
    values = values[np.isfinite(values)]

    if values.size == 0:
        return lower_bound, upper_bound

    data_min = float(np.min(values))
    data_max = float(np.max(values))
    data_span = data_max - data_min
    padding = max(0.005, data_span * 0.20)

    if requested_min is None:
        ymin = max(
            lower_bound,
            data_min - padding,
        )
        ymin = np.floor(ymin / 0.005) * 0.005
    else:
        ymin = float(requested_min)

    if requested_max is None:
        ymax = min(
            upper_bound,
            data_max + padding,
        )
        ymax = np.ceil(ymax / 0.005) * 0.005
    else:
        ymax = float(requested_max)

    if ymax - ymin < 0.01:
        center = (ymax + ymin) / 2.0
        ymin = max(
            lower_bound,
            center - 0.005,
        )
        ymax = min(
            upper_bound,
            center + 0.005,
        )

    if ymax <= ymin:
        ymin = max(
            lower_bound,
            data_min - 0.01,
        )
        ymax = min(
            upper_bound,
            max(data_max + 0.01, ymin + 0.01),
        )

    return float(ymin), float(ymax)


def set_figure2f_y_ticks(
    ax,
    ymin: float,
    ymax: float,
):
    span = ymax - ymin

    if span <= 0.03:
        step = 0.005
        format_string = "%.3f"
    elif span <= 0.10:
        step = 0.01
        format_string = "%.2f"
    elif span <= 0.30:
        step = 0.05
        format_string = "%.2f"
    else:
        step = 0.10
        format_string = "%.1f"

    ax.yaxis.set_major_locator(
        MultipleLocator(step)
    )
    ax.yaxis.set_major_formatter(
        FormatStrFormatter(format_string)
    )


def plot_single_figure2f_series(
    rows: List[Dict[str, Any]],
    series: str,
    out_dir: Path,
):
    series_rows = sorted(
        [
            row
            for row in rows
            if row.get("series") == series
        ],
        key=lambda row: float(row["threshold"]),
    )

    if not series_rows:
        print(
            f"  ⚠ Figure 2F {series} 未绘制："
            "没有有效阈值结果"
        )
        return

    thresholds = np.asarray(
        [row["threshold"] for row in series_rows],
        dtype=float,
    )
    accuracy = np.asarray(
        [row["accuracy"] for row in series_rows],
        dtype=float,
    )
    remaining_rate = np.asarray(
        [row["remaining_rate"] for row in series_rows],
        dtype=float,
    )

    accuracy_ymin, accuracy_ymax = auto_figure2f_ylim(
        accuracy,
        requested_min=FIG2F_ACCURACY_YMIN,
        requested_max=FIG2F_ACCURACY_YMAX,
    )
    remaining_ymin, remaining_ymax = auto_figure2f_ylim(
        remaining_rate,
        requested_min=None,
        requested_max=1.0,
    )

    series_color = FIG2F_COLORS[series]
    remaining_color = "#555555"

    fig, ax_accuracy = plt.subplots(
        figsize=FIG2F_FIGSIZE
    )
    ax_remaining = ax_accuracy.twinx()

    accuracy_line, = ax_accuracy.plot(
        thresholds,
        accuracy,
        color=series_color,
        linewidth=FIG2F_LINEWIDTH,
        marker="o",
        markersize=FIG2F_MARKERSIZE,
        markeredgewidth=2.0,
        markeredgecolor="white",
        label="Accuracy",
        zorder=4,
    )

    remaining_line, = ax_remaining.plot(
        thresholds,
        remaining_rate,
        color=remaining_color,
        linewidth=FIG2F_LINEWIDTH,
        linestyle="--",
        marker="s",
        markersize=FIG2F_MARKERSIZE - 1,
        markeredgewidth=1.8,
        markeredgecolor="white",
        label="Remaining Rate",
        zorder=3,
    )

    if thresholds.size == 1:
        x_padding = 0.05
    else:
        unique_thresholds = np.unique(thresholds)
        min_step = float(
            np.min(np.diff(unique_thresholds))
        )
        x_padding = max(
            0.02,
            min_step * 0.25,
        )

    ax_accuracy.set_xlim(
        float(np.min(thresholds) - x_padding),
        float(np.max(thresholds) + x_padding),
    )
    ax_accuracy.set_xticks(thresholds)

    if np.allclose(
        thresholds * 10.0,
        np.round(thresholds * 10.0),
        atol=1e-8,
    ):
        ax_accuracy.xaxis.set_major_formatter(
            FormatStrFormatter("%.1f")
        )
    else:
        ax_accuracy.xaxis.set_major_formatter(
            FormatStrFormatter("%.2f")
        )

    ax_accuracy.set_ylim(
        accuracy_ymin,
        accuracy_ymax,
    )
    ax_remaining.set_ylim(
        remaining_ymin,
        remaining_ymax,
    )

    set_figure2f_y_ticks(
        ax_accuracy,
        accuracy_ymin,
        accuracy_ymax,
    )
    set_figure2f_y_ticks(
        ax_remaining,
        remaining_ymin,
        remaining_ymax,
    )

    ax_accuracy.set_xlabel(
        "Threshold",
        fontsize=FIG2F_AXIS_LABEL_FONTSIZE,
    )
    ax_accuracy.set_ylabel(
        "Accuracy",
        fontsize=FIG2F_AXIS_LABEL_FONTSIZE,
        color=series_color,
        labelpad=10,
    )
    ax_remaining.set_ylabel(
        "Remaining Rate",
        fontsize=FIG2F_AXIS_LABEL_FONTSIZE,
        color=remaining_color,
        labelpad=12,
    )

    ax_accuracy.tick_params(
        axis="x",
        direction="out",
        length=7,
        width=3,
        labelsize=FIG2F_TICK_FONTSIZE,
        pad=8,
        colors="black",
    )
    ax_accuracy.tick_params(
        axis="y",
        direction="out",
        length=7,
        width=3,
        labelsize=FIG2F_TICK_FONTSIZE,
        pad=8,
        colors=series_color,
    )
    ax_remaining.tick_params(
        axis="y",
        direction="out",
        length=7,
        width=3,
        labelsize=FIG2F_TICK_FONTSIZE,
        pad=8,
        colors=remaining_color,
    )

    for side in ["bottom", "top"]:
        ax_accuracy.spines[side].set_linewidth(3)
        ax_accuracy.spines[side].set_color("black")

    ax_accuracy.spines["left"].set_linewidth(3)
    ax_accuracy.spines["left"].set_color(series_color)
    ax_accuracy.spines["right"].set_visible(False)

    ax_remaining.spines["right"].set_linewidth(3)
    ax_remaining.spines["right"].set_color(remaining_color)
    ax_remaining.spines["left"].set_visible(False)
    ax_remaining.spines["top"].set_visible(False)
    ax_remaining.spines["bottom"].set_visible(False)

    ax_accuracy.text(
        0.02,
        0.97,
        series,
        transform=ax_accuracy.transAxes,
        ha="left",
        va="top",
        fontsize=FIG2F_AXIS_LABEL_FONTSIZE,
        color=series_color,
        fontweight="bold",
    )

    ax_accuracy.legend(
        handles=[
            accuracy_line,
            remaining_line,
        ],
        labels=[
            "Accuracy",
            "Remaining Rate",
        ],
        loc="lower right",
        frameon=False,
        prop={
            "size": FIG2F_LEGEND_FONTSIZE,
            "weight": "bold",
        },
        handlelength=2.6,
    )

    fig.subplots_adjust(
        left=0.10,
        right=0.985,
        bottom=0.23,
        top=0.97,
    )

    stem = (
        f"Figure2F_{series}_"
        "accuracy_remaining_rate"
    )
    save_figure_all_formats(
        fig=fig,
        out_dir=out_dir,
        stem=stem,
        dpi=FIG2F_DPI,
    )
    plt.close(fig)

    print(
        f"  ✓ Figure 2F {series} 已输出到: "
        f"{out_dir}"
    )


def generate_figure2f(
    global_rows: List[Dict[str, str]],
    out_dir: Path,
):
    rows = collect_figure2f_rows(global_rows)

    if not rows:
        print(
            "  ⚠ Figure 2F 未绘制："
            "没有找到有效阈值指标行"
        )
        return

    for series in ["X", "Y"]:
        plot_single_figure2f_series(
            rows=rows,
            series=series,
            out_dir=out_dir,
        )


# =============================================================================
# 9) Figure 2B 总体性能图
# =============================================================================
def figure2b_metric_items() -> List[Tuple[str, str]]:
    return [
        ("accuracy", "Accuracy"),
        ("precision_pos_label1", "Precision"),
        (
            "recall_sensitivity_TPR_pos_label1",
            "Recall",
        ),
        ("f1_pos_label1", "F1-score"),
        ("roc_auc_pos", "AUROC"),
        ("average_precision_pos", "AUPRC"),
        ("specificity_TNR", "Specificity"),
        ("mcc", "MCC"),
    ]


def collect_figure2b_rows(
    global_rows: List[Dict[str, str]],
    condition: str,
) -> List[Dict[str, str]]:
    rows: List[Dict[str, str]] = []

    for row in global_rows:
        if row.get("status") != "OK":
            continue
        if row.get("condition") != condition:
            continue
        if row.get("group") not in {
            "POS_X",
            "NEG_Y",
        }:
            continue

        rows.append(row)

    return rows


def auto_figure2b_ylim(
    values: List[float],
) -> Tuple[float, float]:
    finite_values = [
        value
        for value in values
        if np.isfinite(value)
    ]

    if not finite_values:
        return 0.0, 1.05

    value_min = min(finite_values)
    value_max = max(finite_values)

    if value_min >= 0.90:
        ymin = max(
            0.80,
            np.floor((value_min - 0.03) * 100.0)
            / 100.0,
        )
        tick_step = 0.02
    elif value_min >= 0.75:
        ymin = max(
            0.60,
            np.floor((value_min - 0.05) * 20.0)
            / 20.0,
        )
        tick_step = 0.05
    elif value_min >= 0.50:
        ymin = max(
            0.40,
            np.floor((value_min - 0.08) * 10.0)
            / 10.0,
        )
        tick_step = 0.10
    else:
        ymin = min(
            0.0,
            np.floor((value_min - 0.10) * 10.0)
            / 10.0,
        )
        tick_step = 0.10

    data_span = max(
        value_max - ymin,
        tick_step,
    )
    top_padding = max(
        0.025,
        data_span * 0.12,
    )
    ymax = value_max + top_padding
    ymax = (
        np.ceil(ymax / tick_step)
        * tick_step
    )
    ymax = max(
        ymax,
        1.02
        if value_max >= 0.98
        else value_max + top_padding,
    )

    return float(ymin), float(ymax)


def configure_figure2b_y_axis(
    ax,
    y_range: float,
):
    if y_range <= 0.18:
        step = 0.02
    elif y_range <= 0.35:
        step = 0.05
    else:
        step = 0.10

    ax.yaxis.set_major_locator(
        MultipleLocator(step)
    )
    ax.yaxis.set_major_formatter(
        FormatStrFormatter("%.2f")
    )


def plot_single_figure2b_task(
    row: Dict[str, str],
    group_key: str,
    task_label: str,
    out_dir: Path,
    metric_items: List[Tuple[str, str]],
):
    color = FIG2B_COLORS[group_key]
    x = np.arange(
        len(metric_items),
        dtype=float,
    )

    values = np.asarray(
        [
            safe_float(row.get(metric_key))
            for metric_key, _ in metric_items
        ],
        dtype=float,
    )

    if not np.any(np.isfinite(values)):
        print(
            f"  ⚠ Figure 2B {task_label} 未绘制："
            "指标值均无效"
        )
        return

    ymin, ymax = auto_figure2b_ylim(
        list(values)
    )
    y_range = ymax - ymin

    fig, ax = plt.subplots(
        figsize=FIG2B_SINGLE_FIGSIZE
    )

    ax.vlines(
        x,
        ymin,
        values,
        colors=color,
        linewidth=FIG2B_LINEWIDTH,
        zorder=2,
    )
    ax.scatter(
        x,
        values,
        s=FIG2B_POINTSIZE,
        color=color,
        edgecolor="white",
        linewidth=2.0,
        zorder=3,
    )

    for x_value, y_value in zip(x, values):
        if not np.isfinite(y_value):
            continue

        ax.annotate(
            f"{y_value:.3f}",
            xy=(x_value, y_value),
            xytext=(0, 3),
            textcoords="offset points",
            ha="center",
            va="bottom",
            fontsize=FIG2B_TEXT_FONTSIZE,
            color=color,
            fontweight="bold",
            clip_on=True,
            zorder=4,
        )

    ax.text(
        0.015,
        0.965,
        task_label,
        transform=ax.transAxes,
        ha="left",
        va="top",
        fontsize=FIG2B_SUBPANEL_FONTSIZE,
        color=color,
        fontweight="bold",
    )

    ax.set_xlim(
        -0.55,
        len(metric_items) - 0.45,
    )
    ax.set_ylim(
        ymin,
        ymax,
    )
    ax.set_ylabel(
        "Score",
        fontsize=FIG2B_AXIS_LABEL_FONTSIZE,
    )
    # ax.set_xlabel(
    #     "Metric",
    #     fontsize=FIG2B_AXIS_LABEL_FONTSIZE,
    # )
    ax.set_xticks(x)
    ax.set_xticklabels(
        [
            label
            for _, label in metric_items
        ],
        rotation=35,
        ha="right",
        fontsize=FIG2B_XTICK_FONTSIZE,
    )

    configure_figure2b_y_axis(
        ax,
        y_range,
    )
    style_axes_frame(
        ax,
        spine_lw=3,
        tick_lw=3,
        tick_len=7,
    )
    ax.tick_params(
        axis="y",
        labelsize=FIG2B_TICK_FONTSIZE,
        pad=8,
    )
    ax.tick_params(
        axis="x",
        pad=8,
    )

    fig.subplots_adjust(
        left=0.10,
        right=0.985,
        bottom=0.27,
        top=0.97,
    )

    suffix = (
        "X"
        if group_key == "POS_X"
        else "Y"
    )
    stem = (
        f"Figure2B_litemamba_{suffix}_"
        "overall_performance"
    )

    save_figure_all_formats(
        fig=fig,
        out_dir=out_dir,
        stem=stem,
        dpi=FIG2B_DPI,
    )
    plt.close(fig)

    print(
        f"  ✓ Figure 2B {task_label} 已输出到: "
        f"{out_dir}"
    )


def plot_combined_figure2b(
    rows: List[Dict[str, str]],
    out_dir: Path,
    metric_items: List[Tuple[str, str]],
):
    row_x = next(
        (
            row
            for row in rows
            if row.get("group") == "POS_X"
        ),
        None,
    )
    row_y = next(
        (
            row
            for row in rows
            if row.get("group") == "NEG_Y"
        ),
        None,
    )

    if row_x is None or row_y is None:
        print(
            "  ⚠ Figure 2B combined 未绘制："
            "缺少 X 或 Y 的结果"
        )
        return

    values_x = np.asarray(
        [
            safe_float(row_x.get(metric_key))
            for metric_key, _ in metric_items
        ],
        dtype=float,
    )
    values_y = np.asarray(
        [
            safe_float(row_y.get(metric_key))
            for metric_key, _ in metric_items
        ],
        dtype=float,
    )

    finite_values = [
        float(value)
        for value in np.concatenate(
            [values_x, values_y]
        )
        if np.isfinite(value)
    ]

    if not finite_values:
        print(
            "  ⚠ Figure 2B combined 未绘制："
            "X 和 Y 指标值均无效"
        )
        return

    ymin, ymax = auto_figure2b_ylim(
        finite_values
    )
    y_range = ymax - ymin

    x = np.arange(
        len(metric_items),
        dtype=float,
    )
    x_position_x = (
        x - FIG2B_COMBINED_X_OFFSET
    )
    x_position_y = (
        x + FIG2B_COMBINED_X_OFFSET
    )

    color_x = FIG2B_COLORS["POS_X"]
    color_y = FIG2B_COLORS["NEG_Y"]

    fig, ax = plt.subplots(
        figsize=FIG2B_COMBINED_FIGSIZE
    )

    ax.vlines(
        x_position_x,
        ymin,
        values_x,
        colors=color_x,
        linewidth=FIG2B_LINEWIDTH,
        zorder=2,
    )
    ax.scatter(
        x_position_x,
        values_x,
        s=FIG2B_POINTSIZE,
        color=color_x,
        edgecolor="white",
        linewidth=2.0,
        zorder=3,
    )

    ax.vlines(
        x_position_y,
        ymin,
        values_y,
        colors=color_y,
        linewidth=FIG2B_LINEWIDTH,
        zorder=2,
    )
    ax.scatter(
        x_position_y,
        values_y,
        s=FIG2B_POINTSIZE,
        color=color_y,
        edgecolor="white",
        linewidth=2.0,
        zorder=3,
    )

    for x_value, y_value in zip(
        x_position_x,
        values_x,
    ):
        if not np.isfinite(y_value):
            continue

        ax.annotate(
            f"{y_value:.3f}",
            xy=(x_value, y_value),
            xytext=FIG2B_X_LABEL_OFFSET,
            textcoords="offset points",
            ha="right",
            va="bottom",
            fontsize=FIG2B_TEXT_FONTSIZE,
            color=color_x,
            fontweight="bold",
            clip_on=False,
            zorder=4,
        )

    for x_value, y_value in zip(
        x_position_y,
        values_y,
    ):
        if not np.isfinite(y_value):
            continue

        ax.annotate(
            f"{y_value:.3f}",
            xy=(x_value, y_value),
            xytext=FIG2B_Y_LABEL_OFFSET,
            textcoords="offset points",
            ha="left",
            va="top",
            fontsize=FIG2B_TEXT_FONTSIZE,
            color=color_y,
            fontweight="bold",
            clip_on=False,
            zorder=4,
        )

    ax.set_xlim(
        -0.60,
        len(metric_items) - 0.40,
    )
    ax.set_ylim(
        ymin,
        ymax,
    )
    ax.set_ylabel(
        "Score",
        fontsize=FIG2B_AXIS_LABEL_FONTSIZE,
    )
    # ax.set_xlabel(
    #     "Metric",
    #     fontsize=FIG2B_AXIS_LABEL_FONTSIZE,
    # )
    ax.set_xticks(x)
    ax.set_xticklabels(
        [
            label
            for _, label in metric_items
        ],
        rotation=35,
        ha="right",
        fontsize=FIG2B_XTICK_FONTSIZE,
    )

    configure_figure2b_y_axis(
        ax,
        y_range,
    )
    style_axes_frame(
        ax,
        spine_lw=3,
        tick_lw=3,
        tick_len=7,
    )
    ax.tick_params(
        axis="y",
        labelsize=FIG2B_TICK_FONTSIZE,
        pad=8,
    )
    ax.tick_params(
        axis="x",
        pad=8,
    )

    legend_handles = [
        Line2D(
            [0],
            [0],
            color=color_x,
            linewidth=FIG2B_LINEWIDTH,
            marker="o",
            markersize=10,
            markerfacecolor=color_x,
            markeredgecolor="white",
            markeredgewidth=2.0,
            label="CB (Canonical Base) vs X",
        ),
        Line2D(
            [0],
            [0],
            color=color_y,
            linewidth=FIG2B_LINEWIDTH,
            marker="o",
            markersize=10,
            markerfacecolor=color_y,
            markeredgecolor="white",
            markeredgewidth=2.0,
            label="CB (Canonical Base) vs Y",
        ),
    ]

    ax.legend(
        handles=legend_handles,
        loc="best",
        frameon=False,
        prop={
            "size": FIG2B_SUBPANEL_FONTSIZE,
            "weight": "bold",
        },
        handlelength=2.2,
    )

    fig.subplots_adjust(
        left=0.10,
        right=0.985,
        bottom=0.27,
        top=0.97,
    )

    save_figure_all_formats(
        fig=fig,
        out_dir=out_dir,
        stem=(
            "Figure2B_litemamba_XY_"
            "overall_performance_combined"
        ),
        dpi=FIG2B_DPI,
    )
    plt.close(fig)

    print(
        "  ✓ Figure 2B combined 总体性能图已输出到: "
        f"{out_dir}"
    )


def generate_figure2b_overall(
    global_rows: List[Dict[str, str]],
    out_dir: Path,
):
    rows = collect_figure2b_rows(
        global_rows,
        condition=FIGURE2B_CONDITION,
    )

    if not rows:
        print(
            "  ⚠ Figure 2B 未绘制："
            "没有找到指定条件下的有效结果"
        )
        return

    metric_items = figure2b_metric_items()

    task_order = [
        (
            "POS_X",
            "CB (Canonical Base) vs X",
        ),
        (
            "NEG_Y",
            "CB (Canonical Base) vs Y",
        ),
    ]

    for group_key, task_label in task_order:
        row = next(
            (
                item
                for item in rows
                if item.get("group") == group_key
            ),
            None,
        )

        if row is None:
            print(
                f"  ⚠ Figure 2B {task_label} 未绘制："
                "未找到对应结果"
            )
            continue

        plot_single_figure2b_task(
            row=row,
            group_key=group_key,
            task_label=task_label,
            out_dir=out_dir,
            metric_items=metric_items,
        )

    plot_combined_figure2b(
        rows=rows,
        out_dir=out_dir,
        metric_items=metric_items,
    )


# =============================================================================
# 10) Figure 2B ROC/PR
# =============================================================================
def generate_figure2b_roc_pr(
    pos_predictions_csv: Path,
    neg_predictions_csv: Path,
    out_dir: Path,
):
    curve_dir = Path(out_dir) / "roc_pr_curves"
    curve_dir.mkdir(
        parents=True,
        exist_ok=True,
    )

    y_true_x, y_prob_x = load_predictions_csv(
        pos_predictions_csv
    )
    y_true_y, y_prob_y = load_predictions_csv(
        neg_predictions_csv
    )

    plot_single_roc_pr(
        y_true=y_true_x,
        y_prob=y_prob_x,
        out_dir=curve_dir,
        stem_prefix="Figure2B_litemamba_X",
        positive_name="X",
        title_prefix="CB (Canonical Base) vs X",
        line_color=FIG2B_COLORS["POS_X"],
    )

    plot_single_roc_pr(
        y_true=y_true_y,
        y_prob=y_prob_y,
        out_dir=curve_dir,
        stem_prefix="Figure2B_litemamba_Y",
        positive_name="Y",
        title_prefix="CB (Canonical Base) vs Y",
        line_color=FIG2B_COLORS["NEG_Y"],
    )

    plot_combined_xy_roc_pr(
        y_true_x=y_true_x,
        y_prob_x=y_prob_x,
        y_true_y=y_true_y,
        y_prob_y=y_prob_y,
        out_dir=curve_dir,
    )

    print(
        "  ✓ Figure 2B X/Y/combined ROC/PR 已输出到: "
        f"{curve_dir}"
    )

    return (
        y_true_x,
        y_prob_x,
        y_true_y,
        y_prob_y,
    )


# =============================================================================
# 11) 参数解析
# =============================================================================
def parse_args():
    parser = argparse.ArgumentParser(
        description=(
            "Read existing LiteMamba evaluation CSV files and redraw Figure 2 "
            "without model inference."
        )
    )
    parser.add_argument(
        "--global_summary_csv",
        required=True,
        help="Global summary CSV produced by evaluate_binary_litemamba.py.",
    )
    parser.add_argument(
        "--output_root",
        required=True,
        help="Root directory for all replotted Figure 2 outputs.",
    )
    parser.add_argument(
        "--figure2b_out_dir",
        default=None,
        help="Optional Figure 2B output directory. Default: <output_root>/figure2b",
    )
    parser.add_argument(
        "--figure2f_out_dir",
        default=None,
        help="Optional Figure 2F output directory. Default: <output_root>/figure2f",
    )
    parser.add_argument(
        "--confusion_out_dir",
        default=None,
        help="Optional confusion-matrix output directory. Default: <output_root>/confusion_matrices",
    )
    parser.add_argument(
        "--pos_predictions_csv",
        default=None,
        help="Optional X-task predictions CSV containing y_true, prob_0, and prob_1.",
    )
    parser.add_argument(
        "--neg_predictions_csv",
        default=None,
        help="Optional Y-task predictions CSV containing y_true, prob_0, and prob_1.",
    )
    parser.add_argument(
        "--pos_dataset_stem",
        default=POS_DATASET_STEM,
        help=f"Filename stem used for X threshold curves. Default: {POS_DATASET_STEM}",
    )
    parser.add_argument(
        "--neg_dataset_stem",
        default=NEG_DATASET_STEM,
        help=f"Filename stem used for Y threshold curves. Default: {NEG_DATASET_STEM}",
    )

    parser.add_argument("--plot_confusion_matrices", action="store_true")
    parser.add_argument("--skip_figure2b_overall", action="store_true")
    parser.add_argument("--skip_figure2f", action="store_true")
    parser.add_argument("--skip_figure2b_roc_pr", action="store_true")
    parser.add_argument("--plot_per_threshold_roc_pr", action="store_true")
    parser.add_argument("--plot_multi_threshold_roc_pr", action="store_true")

    parser.add_argument("--figure2b_condition", default="argmax_all")
    parser.add_argument("--min_kept_for_curve", type=int, default=20)
    parser.add_argument("--figure2f_accuracy_ymin", type=float, default=0.97)
    parser.add_argument("--figure2f_accuracy_ymax", type=float, default=0.99)

    args = parser.parse_args()
    if args.min_kept_for_curve <= 0:
        parser.error("--min_kept_for_curve must be > 0")
    if args.figure2f_accuracy_ymax <= args.figure2f_accuracy_ymin:
        parser.error("Figure 2F accuracy ymax must be greater than ymin.")
    if bool(args.pos_predictions_csv) != bool(args.neg_predictions_csv):
        parser.error(
            "--pos_predictions_csv and --neg_predictions_csv must be supplied together."
        )
    return args


def main():
    global PLOT_CONFUSION_MATRICES
    global PLOT_FIGURE2B_OVERALL
    global PLOT_FIGURE2F
    global PLOT_FIGURE2B_ROC_PR
    global PLOT_PER_THRESHOLD_ROC_PR
    global PLOT_MULTI_THRESHOLD_ROC_PR
    global FIGURE2B_CONDITION
    global MIN_KEPT_FOR_CURVE
    global FIG2F_ACCURACY_YMIN
    global FIG2F_ACCURACY_YMAX
    global POS_DATASET_STEM
    global NEG_DATASET_STEM

    args = parse_args()

    PLOT_CONFUSION_MATRICES = bool(args.plot_confusion_matrices)
    PLOT_FIGURE2B_OVERALL = not args.skip_figure2b_overall
    PLOT_FIGURE2F = not args.skip_figure2f
    PLOT_FIGURE2B_ROC_PR = not args.skip_figure2b_roc_pr
    PLOT_PER_THRESHOLD_ROC_PR = bool(args.plot_per_threshold_roc_pr)
    PLOT_MULTI_THRESHOLD_ROC_PR = bool(args.plot_multi_threshold_roc_pr)
    FIGURE2B_CONDITION = args.figure2b_condition
    MIN_KEPT_FOR_CURVE = int(args.min_kept_for_curve)
    FIG2F_ACCURACY_YMIN = float(args.figure2f_accuracy_ymin)
    FIG2F_ACCURACY_YMAX = float(args.figure2f_accuracy_ymax)
    POS_DATASET_STEM = args.pos_dataset_stem
    NEG_DATASET_STEM = args.neg_dataset_stem

    global_summary_csv = Path(args.global_summary_csv)
    output_root = Path(args.output_root)
    figure2b_out_dir = Path(args.figure2b_out_dir) if args.figure2b_out_dir else output_root / "figure2b"
    figure2f_out_dir = Path(args.figure2f_out_dir) if args.figure2f_out_dir else output_root / "figure2f"
    confusion_out_dir = Path(args.confusion_out_dir) if args.confusion_out_dir else output_root / "confusion_matrices"
    pos_predictions_csv = Path(args.pos_predictions_csv) if args.pos_predictions_csv else None
    neg_predictions_csv = Path(args.neg_predictions_csv) if args.neg_predictions_csv else None

    output_root.mkdir(parents=True, exist_ok=True)

    print("=" * 100)
    print("LiteMamba Figure 2 plotting-only workflow")
    print(f"global_summary_csv : {global_summary_csv}")
    print(f"output_root        : {output_root}")
    print(f"figure2b_out_dir   : {figure2b_out_dir}")
    print(f"figure2f_out_dir   : {figure2f_out_dir}")
    print(f"pos_predictions    : {pos_predictions_csv or 'not provided'}")
    print(f"neg_predictions    : {neg_predictions_csv or 'not provided'}")
    print("=" * 100)

    global_rows = load_csv_rows(global_summary_csv)
    print(f"Loaded summary rows: {len(global_rows)}")

    if PLOT_CONFUSION_MATRICES:
        print("\n[1] Redraw normalized confusion matrices")
        generate_confusion_matrices(global_rows=global_rows, output_root=confusion_out_dir)

    if PLOT_FIGURE2B_OVERALL:
        print("\n[2] Redraw Figure 2B overall-performance panels")
        generate_figure2b_overall(global_rows=global_rows, out_dir=figure2b_out_dir)

    if PLOT_FIGURE2F:
        print("\n[3] Redraw Figure 2F")
        generate_figure2f(global_rows=global_rows, out_dir=figure2f_out_dir)

    predictions_available = (
        pos_predictions_csv is not None
        and neg_predictions_csv is not None
        and pos_predictions_csv.exists()
        and neg_predictions_csv.exists()
    )

    needs_predictions = (
        PLOT_FIGURE2B_ROC_PR
        or PLOT_PER_THRESHOLD_ROC_PR
        or PLOT_MULTI_THRESHOLD_ROC_PR
    )

    if needs_predictions and not predictions_available:
        print("\n[4] ROC/PR plots were skipped")
        if pos_predictions_csv is None:
            print("  Reason: prediction CSV files were not provided.")
        else:
            print("  Reason: one or both prediction CSV files do not exist.")
            print(f"  X predictions: {pos_predictions_csv}")
            print(f"  Y predictions: {neg_predictions_csv}")
    elif needs_predictions:
        y_true_x = y_prob_x = None
        y_true_y = y_prob_y = None

        if PLOT_FIGURE2B_ROC_PR:
            print("\n[4] Redraw Figure 2B X/Y/combined ROC and PR curves")
            y_true_x, y_prob_x, y_true_y, y_prob_y = generate_figure2b_roc_pr(
                pos_predictions_csv=pos_predictions_csv,
                neg_predictions_csv=neg_predictions_csv,
                out_dir=figure2b_out_dir,
            )

        if y_true_x is None:
            y_true_x, y_prob_x = load_predictions_csv(pos_predictions_csv)
        if y_true_y is None:
            y_true_y, y_prob_y = load_predictions_csv(neg_predictions_csv)

        if PLOT_PER_THRESHOLD_ROC_PR:
            print("\n[5] Redraw per-threshold ROC/PR curves")
            generate_per_threshold_roc_pr(
                global_rows=global_rows,
                group="POS_X",
                y_true_all=y_true_x,
                y_prob_all=y_prob_x,
                out_dir=figure2b_out_dir / "per_threshold_roc_pr" / "POS_X",
                dataset_stem=POS_DATASET_STEM,
                positive_name="X",
            )
            generate_per_threshold_roc_pr(
                global_rows=global_rows,
                group="NEG_Y",
                y_true_all=y_true_y,
                y_prob_all=y_prob_y,
                out_dir=figure2b_out_dir / "per_threshold_roc_pr" / "NEG_Y",
                dataset_stem=NEG_DATASET_STEM,
                positive_name="Y",
            )

        if PLOT_MULTI_THRESHOLD_ROC_PR:
            print("\n[6] Redraw multi-threshold ROC/PR curves")
            plot_multi_threshold_roc_pr(
                y_true_all=y_true_x,
                y_prob_all=y_prob_x,
                thresholds=collect_thresholds(global_rows, group="POS_X"),
                out_dir=figure2b_out_dir / "multi_threshold_roc_pr" / "POS_X",
                stem_prefix=POS_DATASET_STEM,
                positive_name="X",
                title_prefix=POS_DATASET_STEM,
            )
            plot_multi_threshold_roc_pr(
                y_true_all=y_true_y,
                y_prob_all=y_prob_y,
                thresholds=collect_thresholds(global_rows, group="NEG_Y"),
                out_dir=figure2b_out_dir / "multi_threshold_roc_pr" / "NEG_Y",
                stem_prefix=NEG_DATASET_STEM,
                positive_name="Y",
                title_prefix=NEG_DATASET_STEM,
            )

    print("\n" + "=" * 100)
    print("Figure 2 plotting completed.")
    print("No model checkpoint was loaded and no inference was rerun.")
    print("=" * 100)


if __name__ == "__main__":
    main()
