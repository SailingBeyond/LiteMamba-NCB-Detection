#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""Generate the final Figure 4A-C panels from external-validation outputs.

This script does not run model inference. It reads the directory produced by
scripts/07_evaluation/evaluate_external_templates.py:

    <evaluation_root>/analysis_data/control_template_metrics_long.csv
    <evaluation_root>/analysis_data/target_template_metrics_long.csv
    <evaluation_root>/raw_evaluation/.../eval_litemamba/predictions_*.csv[.gz]

For each requested task (6-class, 5X and 5Y), it generates:
    Figure 4A: DNA01-16 false non-canonical call heatmap.
    Figure 4B: XNA01-16 target-recall lollipop plot.
    Figure 4C: XNA17-20 residual-error composition.

Figure 4D is intentionally excluded because it is not part of the final main
figure design.
"""

from __future__ import annotations

import argparse
import gzip
import re
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Tuple

import matplotlib
matplotlib.use("Agg")
import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

mpl.rcParams.update({
    "font.family": "sans-serif",
    "font.sans-serif": ["Arial", "Liberation Sans", "DejaVu Sans"],
    "pdf.fonttype": 42,
    "ps.fonttype": 42,
    "svg.fonttype": "none",
    "axes.linewidth": 1.0,
    "font.size": 11,
    "axes.labelsize": 12,
    "xtick.labelsize": 10,
    "ytick.labelsize": 10,
    "legend.fontsize": 10,
    "axes.unicode_minus": False,
})

MODEL_NAME = "litemamba"
SUPPORTED_TASKS = ["6-class", "5X", "5Y"]
DNA_TEMPLATES = [f"XNA{i}" for i in range(1, 17)]
XNA_1_16 = [f"XNA{i}" for i in range(1, 17)]
XNA_17_20 = [f"XNA{i}" for i in range(17, 21)]

TASK_CONFIG: Dict[str, Dict[str, object]] = {
    "6-class": {
        "slug": "6class",
        "prediction_keys": ["xna17_20_6class", "xna17_20_6_class"],
        "target_labels": [4, 5],
        "class_names": ["A", "T", "C", "G", "X", "Y"],
        "residual_order": ["A", "T", "C", "G", "opposite NCB"],
        "false_call_label": "False X/Y call rate",
        "target_recall_label": "Target X/Y recall",
    },
    "5X": {
        "slug": "5X",
        "prediction_keys": ["xna17_20_5X", "xna17_20_5x"],
        "target_labels": [4],
        "class_names": ["A", "T", "C", "G", "X"],
        "residual_order": ["A", "T", "C", "G"],
        "false_call_label": "False X call rate",
        "target_recall_label": "Target X recall",
    },
    "5Y": {
        "slug": "5Y",
        "prediction_keys": ["xna17_20_5Y", "xna17_20_5y"],
        "target_labels": [4],
        "class_names": ["A", "T", "C", "G", "Y"],
        "residual_order": ["A", "T", "C", "G"],
        "false_call_label": "False Y call rate",
        "target_recall_label": "Target Y recall",
    },
}

COLOR_DNA_HEAT = "YlOrRd"
COLOR_LITEMAMBA = "#1B9E77"
COLOR_GROUP_SHADE_1 = "#eef6ff"
COLOR_GROUP_SHADE_2 = "#f9f4ea"
RESIDUAL_ERROR_COLORS = {
    "A": "#9ecae1",
    "T": "#fdd0a2",
    "C": "#c7e9c0",
    "G": "#dadaeb",
    "opposite NCB": "#fb9a99",
}


def ensure_dir(path: Path) -> Path:
    path.mkdir(parents=True, exist_ok=True)
    return path


def template_number(template: object) -> int:
    match = re.search(r"(\d+)$", str(template))
    return int(match.group(1)) if match else 10**9


def normalize_task(value: object) -> str:
    text = str(value).strip()
    mapping = {
        "6class": "6-class",
        "6_class": "6-class",
        "6-class": "6-class",
        "5x": "5X",
        "5X": "5X",
        "5y": "5Y",
        "5Y": "5Y",
    }
    return mapping.get(text, mapping.get(text.lower(), text))


def parse_tasks(text: str) -> List[str]:
    tasks: List[str] = []
    for token in text.split(","):
        if not token.strip():
            continue
        task = normalize_task(token)
        if task not in SUPPORTED_TASKS:
            raise ValueError(f"Unsupported task {token!r}; choose from {SUPPORTED_TASKS}")
        if task not in tasks:
            tasks.append(task)
    if not tasks:
        raise ValueError("--tasks cannot be empty")
    return tasks


def parse_formats(text: str) -> List[str]:
    formats = [item.strip().lower() for item in text.split(",") if item.strip()]
    allowed = {"png", "pdf", "svg"}
    if not formats or any(fmt not in allowed for fmt in formats):
        raise ValueError("--formats must contain only png,pdf,svg")
    return formats


def validate_columns(df: pd.DataFrame, required: Sequence[str], name: str) -> None:
    missing = [column for column in required if column not in df.columns]
    if missing:
        raise ValueError(f"{name} is missing columns: {missing}")


def save_figure(fig: plt.Figure, stem: Path, formats: Sequence[str], dpi: int) -> None:
    ensure_dir(stem.parent)
    for fmt in formats:
        kwargs = {"bbox_inches": "tight"}
        if fmt == "png":
            kwargs["dpi"] = dpi
        fig.savefig(stem.with_suffix(f".{fmt}"), **kwargs)
    plt.close(fig)


def read_prediction_file(path: Path) -> pd.DataFrame:
    if path.suffix.lower() == ".gz":
        with gzip.open(path, "rt", encoding="utf-8") as handle:
            return pd.read_csv(handle)
    return pd.read_csv(path)


def find_prediction_files(directory: Path) -> List[Path]:
    return sorted(
        list(directory.glob("predictions_*.csv"))
        + list(directory.glob("predictions_*.csv.gz"))
    )


def load_analysis_tables(evaluation_root: Path) -> Tuple[pd.DataFrame, pd.DataFrame]:
    data_dir = evaluation_root / "analysis_data"
    control_path = data_dir / "control_template_metrics_long.csv"
    target_path = data_dir / "target_template_metrics_long.csv"
    if not control_path.exists():
        raise FileNotFoundError(f"Missing analysis table: {control_path}")
    if not target_path.exists():
        raise FileNotFoundError(f"Missing analysis table: {target_path}")

    control_df = pd.read_csv(control_path)
    target_df = pd.read_csv(target_path)
    validate_columns(
        control_df,
        ["task", "model", "template", "false_noncanonical_call_rate"],
        control_path.name,
    )
    validate_columns(
        target_df,
        ["task", "model", "template", "target_recall_micro"],
        target_path.name,
    )
    control_df = control_df.copy()
    target_df = target_df.copy()
    control_df["task"] = control_df["task"].map(normalize_task)
    target_df["task"] = target_df["task"].map(normalize_task)
    if "target_support" not in target_df.columns:
        target_df["target_support"] = np.nan
    return control_df, target_df


def resolve_prediction_dir(evaluation_root: Path, task: str) -> Path:
    raw_root = evaluation_root / "raw_evaluation"
    if not raw_root.exists():
        raise FileNotFoundError(f"Missing raw_evaluation directory: {raw_root}")

    for key in TASK_CONFIG[task]["prediction_keys"]:
        candidate = raw_root / str(key) / "eval_litemamba"
        if candidate.exists() and find_prediction_files(candidate):
            return candidate

    for candidate in sorted(raw_root.glob("**/eval_litemamba")):
        token = candidate.parent.name.lower().replace("-", "").replace("_", "")
        if "xna1720" not in token:
            continue
        if task == "6-class" and "6class" in token and find_prediction_files(candidate):
            return candidate
        if task == "5X" and "5x" in token and find_prediction_files(candidate):
            return candidate
        if task == "5Y" and "5y" in token and find_prediction_files(candidate):
            return candidate

    raise FileNotFoundError(
        f"Could not locate XNA17-20 LiteMamba prediction files for task {task} under {raw_root}"
    )


def build_figure4a_table(control_df: pd.DataFrame, task: str) -> pd.DataFrame:
    df = control_df[
        (control_df["task"] == task)
        & (control_df["model"].astype(str).str.lower() == MODEL_NAME)
        & (control_df["template"].isin(DNA_TEMPLATES))
    ].copy()
    if df.empty:
        raise ValueError(f"No LiteMamba DNA01-16 controls found for task {task}")
    df["template_num"] = df["template"].map(template_number)
    df = df.sort_values("template_num").drop_duplicates("template", keep="first")
    df["dna_label"] = df["template_num"].map(lambda number: f"DNA{number:02d}")
    return df[[
        "task", "template", "template_num", "dna_label",
        "false_noncanonical_call_rate",
    ]].reset_index(drop=True)


def build_figure4b_table(target_df: pd.DataFrame, task: str) -> pd.DataFrame:
    df = target_df[
        (target_df["task"] == task)
        & (target_df["model"].astype(str).str.lower() == MODEL_NAME)
        & (target_df["template"].isin(XNA_1_16))
    ].copy()
    if df.empty:
        raise ValueError(f"No LiteMamba XNA01-16 target metrics found for task {task}")
    df["template_num"] = df["template"].map(template_number)
    df = df.sort_values("template_num").drop_duplicates("template", keep="first")
    df["xna_label"] = df["template_num"].map(lambda number: f"XNA{number:02d}")
    return df[[
        "task", "template", "template_num", "xna_label",
        "target_recall_micro", "target_support",
    ]].reset_index(drop=True)


def classify_residual_error(true_label: int, pred_label: int, task: str) -> Optional[str]:
    if pred_label == true_label:
        return None
    class_names = list(TASK_CONFIG[task]["class_names"])
    if pred_label in [0, 1, 2, 3]:
        return str(class_names[pred_label])
    if task == "6-class" and (
        (true_label == 4 and pred_label == 5)
        or (true_label == 5 and pred_label == 4)
    ):
        return "opposite NCB"
    return None


def safe_fraction(numerator: float, denominator: float) -> float:
    return float(numerator / denominator) if denominator > 0 else np.nan


def build_figure4c_tables(
    evaluation_root: Path,
    target_df: pd.DataFrame,
    task: str,
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    prediction_dir = resolve_prediction_dir(evaluation_root, task)
    prediction_files = find_prediction_files(prediction_dir)
    required = [
        "cohort", "task", "model", "template", "reference_role",
        "true_label", "pred_label",
    ]
    parts: List[pd.DataFrame] = []
    for path in prediction_files:
        part = read_prediction_file(path)
        validate_columns(part, required, path.name)
        part = part[required].copy()
        part["task"] = part["task"].map(normalize_task)
        parts.append(part)
    if not parts:
        raise FileNotFoundError(f"No prediction files found in {prediction_dir}")

    pred_df = pd.concat(parts, ignore_index=True)
    pred_df = pred_df[
        (pred_df["cohort"] == "XNA17-20")
        & (pred_df["task"] == task)
        & (pred_df["model"].astype(str).str.lower() == MODEL_NAME)
        & (pred_df["reference_role"] == "target")
        & (pred_df["template"].isin(XNA_17_20))
    ].copy()
    if pred_df.empty:
        raise ValueError(f"No XNA17-20 target predictions found for task {task}")

    pred_df["true_label"] = pd.to_numeric(pred_df["true_label"], errors="raise").astype(int)
    pred_df["pred_label"] = pd.to_numeric(pred_df["pred_label"], errors="raise").astype(int)
    target_labels = list(TASK_CONFIG[task]["target_labels"])
    pred_df = pred_df[pred_df["true_label"].isin(target_labels)].copy()
    pred_df["error_type"] = pred_df.apply(
        lambda row: classify_residual_error(
            int(row["true_label"]), int(row["pred_label"]), task
        ),
        axis=1,
    )

    recall_df = target_df[
        (target_df["task"] == task)
        & (target_df["model"].astype(str).str.lower() == MODEL_NAME)
        & (target_df["template"].isin(XNA_17_20))
    ][["template", "target_recall_micro", "target_support"]].copy()
    recall_df = recall_df.drop_duplicates("template", keep="first")
    recall_map = {
        str(row["template"]): float(row["target_recall_micro"])
        for _, row in recall_df.iterrows()
    }

    detail_rows: List[Dict[str, object]] = []
    summary_rows: List[Dict[str, object]] = []
    residual_order = list(TASK_CONFIG[task]["residual_order"])
    for template in XNA_17_20:
        sub = pred_df[pred_df["template"] == template]
        if sub.empty:
            continue
        total_targets = int(len(sub))
        total_errors = int((sub["pred_label"] != sub["true_label"]).sum())
        recall = recall_map.get(template, 1.0 - safe_fraction(total_errors, total_targets))
        for error_type in residual_order:
            error_count = int((sub["error_type"] == error_type).sum())
            detail_rows.append({
                "task": task,
                "template": template,
                "template_num": template_number(template),
                "error_type": error_type,
                "error_count": error_count,
                "target_total": total_targets,
                "error_per_10000": safe_fraction(error_count * 10000, total_targets),
                "target_recall_micro": recall,
            })
        summary_rows.append({
            "task": task,
            "template": template,
            "template_num": template_number(template),
            "target_total": total_targets,
            "total_errors": total_errors,
            "total_error_per_10000": safe_fraction(total_errors * 10000, total_targets),
            "target_recall_micro": recall,
        })

    detail_df = pd.DataFrame(detail_rows)
    summary_df = pd.DataFrame(summary_rows)
    if detail_df.empty or summary_df.empty:
        raise ValueError(f"Could not derive Figure 4C data for task {task}")
    return (
        detail_df.sort_values(["template_num", "error_type"]).reset_index(drop=True),
        summary_df.sort_values("template_num").reset_index(drop=True),
    )


def percent_text(value: float) -> str:
    return "NA" if pd.isna(value) else f"{value * 100:.2f}%"


def plot_figure4a(
    heat_df: pd.DataFrame,
    task: str,
    stem: Path,
    formats: Sequence[str],
    dpi: int,
    vmax: float,
) -> None:
    matrix = np.full((4, 4), np.nan)
    labels = [[("", np.nan) for _ in range(4)] for _ in range(4)]
    for _, row in heat_df.iterrows():
        number = int(row["template_num"])
        r, c = (number - 1) // 4, (number - 1) % 4
        value = float(row["false_noncanonical_call_rate"])
        matrix[r, c] = value
        labels[r][c] = (str(row["dna_label"]), value)

    observed = float(np.nanmax(matrix)) if np.isfinite(matrix).any() else 0.0
    color_max = max(vmax, observed)
    fig, ax = plt.subplots(figsize=(5.0, 4.8))
    image = ax.imshow(matrix, cmap=COLOR_DNA_HEAT, vmin=0.0, vmax=color_max, aspect="equal")
    ax.set_xticks([])
    ax.set_yticks([])
    ax.set_xticks(np.arange(-0.5, 4, 1), minor=True)
    ax.set_yticks(np.arange(-0.5, 4, 1), minor=True)
    ax.grid(which="minor", color="white", linewidth=1.8)
    ax.tick_params(which="minor", bottom=False, left=False)
    ax.axhline(2.5, color="black", linewidth=2.3)

    threshold = color_max * 0.55
    for i in range(4):
        for j in range(4):
            label, value = labels[i][j]
            cell = matrix[i, j]
            color = "black" if np.isfinite(cell) and cell < threshold else "white"
            ax.text(j, i, f"{label}\n{percent_text(value)}", ha="center", va="center",
                    fontsize=10.2, fontweight="bold", color=color)

    cbar = fig.colorbar(image, ax=ax, fraction=0.047, pad=0.04)
    cbar.set_label(str(TASK_CONFIG[task]["false_call_label"]))
    cbar.set_ticks(np.linspace(0, color_max, 6))
    save_figure(fig, stem, formats, dpi)


def plot_figure4b(
    df: pd.DataFrame,
    task: str,
    stem: Path,
    formats: Sequence[str],
    dpi: int,
    ymin: float,
) -> None:
    df = df.sort_values("template_num").reset_index(drop=True)
    x = np.arange(len(df))
    y = df["target_recall_micro"].to_numpy(dtype=float)
    fig, ax = plt.subplots(figsize=(10.2, 3.0))
    ax.axvspan(-0.5, 11.5, color=COLOR_GROUP_SHADE_1, alpha=0.8, zorder=0)
    ax.axvspan(11.5, 15.5, color=COLOR_GROUP_SHADE_2, alpha=0.8, zorder=0)
    ax.axvline(11.5, color="#666666", linestyle="--", linewidth=1.3, zorder=1)
    ax.vlines(x, ymin, y, color=COLOR_LITEMAMBA, linewidth=1.8, zorder=2)
    ax.scatter(x, y, s=64, color=COLOR_LITEMAMBA, edgecolor="black", linewidth=0.7, zorder=3)
    ax.set_xlim(-0.7, len(df) - 0.3)
    ax.set_ylim(ymin, 1.0)
    ax.set_ylabel(str(TASK_CONFIG[task]["target_recall_label"]))
    ax.set_xticks(x)
    ax.set_xticklabels(df["xna_label"], rotation=45, ha="right")
    ax.grid(axis="y", linestyle="--", linewidth=0.8, alpha=0.45)
    save_figure(fig, stem, formats, dpi)


def plot_figure4c(
    detail_df: pd.DataFrame,
    summary_df: pd.DataFrame,
    task: str,
    stem: Path,
    formats: Sequence[str],
    dpi: int,
) -> None:
    detail_df = detail_df.sort_values(["template_num", "error_type"]).reset_index(drop=True)
    summary_df = summary_df.sort_values("template_num").reset_index(drop=True)
    templates = summary_df["template"].tolist()
    x = np.arange(len(templates))
    residual_order = [
        value for value in TASK_CONFIG[task]["residual_order"]
        if value in set(detail_df["error_type"])
    ]

    fig, ax = plt.subplots(figsize=(5.8, 3.6))
    bottom = np.zeros(len(templates), dtype=float)
    for error_type in residual_order:
        values = []
        for template in templates:
            sub = detail_df[
                (detail_df["template"] == template)
                & (detail_df["error_type"] == error_type)
            ]
            values.append(float(sub.iloc[0]["error_per_10000"]) if not sub.empty else 0.0)
        values_array = np.asarray(values, dtype=float)
        ax.bar(
            x, values_array, bottom=bottom, width=0.62,
            color=RESIDUAL_ERROR_COLORS.get(error_type, "#cccccc"),
            edgecolor="white", linewidth=0.8, label=error_type,
        )
        bottom += values_array

    ax.set_xticks(x)
    ax.set_xticklabels(templates)
    ax.set_ylabel("Misclassified calls per 10,000 target NCBs")
    ax.grid(axis="y", linestyle="--", linewidth=0.8, alpha=0.45)

    max_height = max(float(np.nanmax(bottom)), 1.0)
    label_offset = max(max_height * 0.04, 3.5)
    label_positions = []
    for index, row in summary_df.iterrows():
        height = float(row["total_error_per_10000"])
        recall = float(row["target_recall_micro"])
        y_text = height + label_offset
        label_positions.append(y_text)
        ax.text(index, y_text, f"Recall = {recall * 100:.2f}%", ha="center", va="bottom",
                fontsize=9.0, fontweight="bold")

    y_top = max(max(label_positions, default=0.0), max_height) * 1.10
    ax.set_ylim(0, max(y_top, 1.0))
    ax.legend(
        title="Residual error type",
        frameon=False,
        ncol=max(1, len(residual_order)),
        loc="upper center",
        bbox_to_anchor=(0.5, 1.18),
    )
    save_figure(fig, stem, formats, dpi)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Plot final Figure 4A-C from external-validation outputs.")
    parser.add_argument("--evaluation_root", required=True, help="Output root from evaluate_external_templates.py")
    parser.add_argument("--figure_dir", default=None, help="Default: <evaluation_root>/figures_final")
    parser.add_argument("--table_dir", default=None, help="Default: <evaluation_root>/analysis_data/figure4abc_plot_data")
    parser.add_argument("--tasks", default="6-class,5X,5Y")
    parser.add_argument("--formats", default="png,pdf,svg")
    parser.add_argument("--dpi", type=int, default=600)
    parser.add_argument("--false_call_vmax", type=float, default=0.20)
    parser.add_argument("--lollipop_ymin", type=float, default=0.0)
    args = parser.parse_args()
    if args.dpi < 72:
        parser.error("--dpi must be at least 72")
    if args.false_call_vmax <= 0:
        parser.error("--false_call_vmax must be positive")
    if not (0 <= args.lollipop_ymin < 1):
        parser.error("--lollipop_ymin must satisfy 0 <= value < 1")
    return args


def main() -> None:
    args = parse_args()
    evaluation_root = Path(args.evaluation_root)
    if not evaluation_root.exists():
        raise FileNotFoundError(f"Evaluation root not found: {evaluation_root}")
    figure_dir = ensure_dir(
        Path(args.figure_dir) if args.figure_dir else evaluation_root / "figures_final"
    )
    table_dir = ensure_dir(
        Path(args.table_dir)
        if args.table_dir
        else evaluation_root / "analysis_data" / "figure4abc_plot_data"
    )
    tasks = parse_tasks(args.tasks)
    formats = parse_formats(args.formats)
    control_df, target_df = load_analysis_tables(evaluation_root)

    all_a: List[pd.DataFrame] = []
    all_b: List[pd.DataFrame] = []
    all_c_detail: List[pd.DataFrame] = []
    all_c_summary: List[pd.DataFrame] = []

    for task in tasks:
        slug = str(TASK_CONFIG[task]["slug"])
        figure4a_df = build_figure4a_table(control_df, task)
        figure4b_df = build_figure4b_table(target_df, task)
        figure4c_detail, figure4c_summary = build_figure4c_tables(
            evaluation_root, target_df, task
        )

        figure4a_df.to_csv(table_dir / f"figure4A_dna01_16_false_call_{slug}.csv", index=False)
        figure4b_df.to_csv(table_dir / f"figure4B_xna01_16_target_recall_{slug}.csv", index=False)
        figure4c_detail.to_csv(table_dir / f"figure4C_xna17_20_residual_detail_{slug}.csv", index=False)
        figure4c_summary.to_csv(table_dir / f"figure4C_xna17_20_residual_summary_{slug}.csv", index=False)

        all_a.append(figure4a_df)
        all_b.append(figure4b_df)
        all_c_detail.append(figure4c_detail)
        all_c_summary.append(figure4c_summary)

        plot_figure4a(
            figure4a_df, task,
            figure_dir / f"Figure4A_DNA01_16_false_call_heatmap_{slug}",
            formats, args.dpi, args.false_call_vmax,
        )
        plot_figure4b(
            figure4b_df, task,
            figure_dir / f"Figure4B_XNA01_16_target_recall_lollipop_{slug}",
            formats, args.dpi, args.lollipop_ymin,
        )
        plot_figure4c(
            figure4c_detail, figure4c_summary, task,
            figure_dir / f"Figure4C_XNA17_20_residual_error_composition_{slug}",
            formats, args.dpi,
        )

    pd.concat(all_a, ignore_index=True).to_csv(table_dir / "figure4A_all_tasks.csv", index=False)
    pd.concat(all_b, ignore_index=True).to_csv(table_dir / "figure4B_all_tasks.csv", index=False)
    pd.concat(all_c_detail, ignore_index=True).to_csv(table_dir / "figure4C_residual_detail_all_tasks.csv", index=False)
    pd.concat(all_c_summary, ignore_index=True).to_csv(table_dir / "figure4C_residual_summary_all_tasks.csv", index=False)

    print("=" * 100)
    print("Final Figure 4A-C plotting completed")
    print(f"Evaluation root : {evaluation_root}")
    print(f"Figure directory: {figure_dir}")
    print(f"Plot-data tables: {table_dir}")
    print(f"Tasks           : {', '.join(tasks)}")
    print("Figure 4D was intentionally not generated.")
    print("=" * 100)


if __name__ == "__main__":
    main()
