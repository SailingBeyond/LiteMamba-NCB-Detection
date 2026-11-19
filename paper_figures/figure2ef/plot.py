#!/usr/bin/env python3
"""Replot Figure 2 E--K from frozen binary GatedConv outputs, without inference."""
import argparse
import csv
import hashlib
import gzip
import json
import math
import os
import sys
from io import BytesIO
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib import font_manager
from matplotlib.lines import Line2D
from matplotlib.ticker import MaxNLocator, FormatStrFormatter, MultipleLocator
from mpl_toolkits.axes_grid1 import make_axes_locatable

VERSION = "1.0.6"
PACKAGE_DIR = Path(__file__).resolve().parent
DEFAULT_RESULTS = PACKAGE_DIR / "data"
SEEDS = (42, 43, 44)
TASKS = {"pos": "X", "neg": "Y"}
COLORS = {"X": "#377DB8", "Y": "#D98484"}
METRICS = ("accuracy", "precision", "recall", "f1", "auroc", "auprc", "specificity", "mcc")
METRIC_LABELS = ("Accuracy", "Precision", "Recall", "F1score", "AUROC", "AUPRC", "Specificity", "MCC")
THRESHOLDS = (.5, .6, .7, .8, .9)
# G's requested 0.92--1.00 range is represented by five ticks spaced by 0.02.
G_YLIM = (.92, 1.00)
G_TICK_STEP = .02
# Panel-specific settings keep the standard E/F panels unchanged while enlarging
# their zoom variants and the requested text in G--K. All sizes are in points.
ZOOM_AXIS_FONTSIZE = 18
ZOOM_TICK_FONTSIZE = 16
ZOOM_LEGEND_FONTSIZE = 12
ZOOM_TITLE_FONTSIZE = 14
G_AXIS_FONTSIZE = 24
G_TICK_FONTSIZE = 20
G_XTICK_FONTSIZE = 22
G_LEGEND_FONTSIZE = 16
CM_AXIS_FONTSIZE = 24
CM_TICK_FONTSIZE = 24
CM_COLORBAR_FONTSIZE = 22
CM_NUMBER_FONTSIZE = 32
JK_AXIS_FONTSIZE = 20
JK_TICK_FONTSIZE = 18
JK_LEGEND_FONTSIZE = 16
JK_TASK_FONTSIZE = 26
JK_SPINE_LINEWIDTH = 3.0
# Match the supplied original Figure2B canvas AND plotting-frame proportions.
G_FIGSIZE = (13.5, 6.8)
G_AXES_RECT = (.10, .27, .885, .70)
G_BOX_ASPECT = (G_FIGSIZE[1] * G_AXES_RECT[3]) / (G_FIGSIZE[0] * G_AXES_RECT[2])
G_SPINE_LINEWIDTH = 3.0
G_TICK_LINEWIDTH = 3.0
G_TICK_LENGTH = 7.0
G_TICK_PAD = 8.0
JK_FIGSIZE = (6.2, 4.8)


def panel_figsize(panel):
    if panel == "G":
        return G_FIGSIZE
    if panel in ("J", "K"):
        return JK_FIGSIZE
    return (5.2, 4.8)


def create_panel_figure(panel):
    if panel == "G":
        # Fixed margins from the reference Figure2B; constrained_layout would
        # change the physical aspect ratio of the plotting frame.
        fig = plt.figure(figsize=G_FIGSIZE, constrained_layout=False)
        return fig, fig.add_axes(G_AXES_RECT)
    return plt.subplots(figsize=panel_figsize(panel), constrained_layout=True)


def require(condition, message):
    if not condition:
        raise ValueError(message)


def close(a, b, message, atol=1e-10):
    require(math.isfinite(float(a)) and math.isfinite(float(b))
            and math.isclose(float(a), float(b), rel_tol=1e-9, abs_tol=atol), message)


def sha256(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


class Inputs:
    def __init__(self, root):
        self.root = root.resolve()
        self.files = {}

    def record(self, path):
        path = Path(path).resolve()
        key = str(path.relative_to(self.root))
        if key not in self.files:
            self.files[key] = {"sha256": sha256(path), "size": path.stat().st_size}

    def json(self, path):
        self.record(path)
        return json.loads(Path(path).read_text(encoding="utf-8"))

    def csv(self, path):
        if not path.is_file():
            path = path.with_suffix(path.suffix + ".gz")
        self.record(path)
        opener = gzip.open if path.suffix == ".gz" else open
        with opener(path, "rt", encoding="utf-8-sig", newline="") as handle:
            return list(csv.DictReader(handle))


def validate_counts(raw, context):
    counts = [int(raw[key]) for key in ("tn", "fp", "fn", "tp")]
    require(all(v >= 0 for v in counts), context + ": negative counts")
    tn, fp, fn, tp = counts
    n = tn + fp + fn + tp
    require(n > 0 and n == int(raw["n"]), context + ": n/count mismatch")
    require(tn + fp == int(raw["negative_support"]) and fn + tp == int(raw["positive_support"]),
            context + ": support/count mismatch")
    require(tn + fp > 0 and fn + tp > 0, context + ": both classes required")
    checks = {"accuracy": (tn + tp) / n, "precision": tp / (tp + fp),
              "recall": tp / (tp + fn), "specificity": tn / (tn + fp),
              "f1": 2 * tp / (2 * tp + fp + fn),
              "mcc": (tp * tn - fp * fn) / math.sqrt((tp + fp) * (tp + fn) * (tn + fp) * (tn + fn))}
    for key, value in checks.items():
        close(raw[key], value, context + ": inconsistent " + key)
    for key in METRICS:
        require(math.isfinite(float(raw[key])), context + ": nonfinite " + key)
    return np.array([[tn, fp], [fn, tp]], dtype=np.int64)


def fingerprint(manifest):
    fields = manifest["input_fingerprints"]["test"]
    result = {kind: {key: fields[kind][key] for key in ("sha256", "size")}
              for kind in ("feature", "metadata")}
    for entry in result.values():
        require(len(entry["sha256"]) == 64 and int(entry["size"]) > 0, "Invalid test fingerprint")
    return result


def load_runs(inputs, panels):
    root = inputs.root
    summary = inputs.json(root / "summary/summary.json")
    require(summary.get("status") == "complete", "Result summary is not complete")
    queue = inputs.json(root / "queue_manifest.json")
    frozen = inputs.json(root / "evaluation_plan.json")
    frozen_runs = {x["run_id"]: x for x in frozen["experiments"]}
    experiments = queue["experiments"]
    expected = {(strand, seed) for strand in TASKS for seed in SEEDS}
    observed, runs, signatures = set(), [], {}
    require(len(experiments) == 6 and len(frozen_runs) == 6, "Exactly six frozen runs are required")
    for experiment in experiments:
        cfg = experiment["requested_config"]
        strand, seed = cfg["strand"], int(cfg["seed"])
        identity = (strand, seed)
        require(identity in expected and identity not in observed, "Unexpected or duplicate task/seed")
        observed.add(identity)
        run_id = experiment["run_id"]
        relative = Path(run_id)
        require(not relative.is_absolute() and ".." not in relative.parts, "Unsafe run_id")
        run_dir = (root / relative).resolve()
        try:
            run_dir.relative_to(root)
        except ValueError:
            raise ValueError("run_id escapes result directory")
        context = f"{TASKS[strand]}/seed_{seed}"
        for key, value in {"model_type": "litemamba", "litemamba_variant": "legacy_gatedconv",
                           "readout": "target_attention", "feature_scheme": "full"}.items():
            require(cfg.get(key) == value, context + ": wrong " + key)
        require(experiment.get("train_queue_status") == "complete" and
                experiment.get("evaluation_queue_status") == "complete", context + ": incomplete run")
        result = inputs.json(run_dir / "results.json")
        model = inputs.json(run_dir / "model_config.json")
        require(result.get("status") == "complete" and result.get("seed") == seed and
                result.get("strand") == strand, context + ": inconsistent results identity")
        for key, value in {"num_classes": 2, "target_index": 3, "seq_len": 6, "model_dim": 96,
                           "num_blocks": 4, "readout": "target_attention",
                           "litemamba_variant": "legacy_gatedconv", "feature_scheme": "full"}.items():
            require(model.get(key) == value, context + ": wrong model " + key)
        manifest = inputs.json(run_dir / "data_manifest.json")
        signature = fingerprint(manifest)
        pre = manifest["preprocessing"]
        require(pre.get("mask_index") == 3 and pre.get("mask_base") == "N", context + ": fourth base not masked")
        task_signature = (signature, pre)
        if strand in signatures:
            require(signatures[strand] == task_signature, context + ": test data/preprocessing differ across seeds")
        signatures[strand] = task_signature
        exported = inputs.json(run_dir / "test/export_complete.json")
        require(exported.get("split") == "test", context + ": wrong split")
        require(run_id in frozen_runs and exported["checkpoint_sha256"] == frozen_runs[run_id]["checkpoint_sha256"],
                context + ": checkpoint differs from frozen test plan")
        require(exported["dataset_identity"]["input_fingerprints"] == signature and
                exported["dataset_identity"]["preprocessing"] == pre, context + ": exported data identity differs")
        require(datetime.fromisoformat(exported["completed_at"]) >= datetime.fromisoformat(frozen["frozen_at"]),
                context + ": export predates freeze plan")
        metrics = inputs.json(run_dir / "test/analysis/metrics.json")
        raw = metrics["raw"]
        require(metrics["row_count"] == exported["n_samples"] == raw["n"], context + ": export row counts differ")
        require(metrics["metadata"]["seed"] == seed and metrics["metadata"]["strand"] == strand,
                context + ": analysis identity differs")
        cm = validate_counts(raw, context)
        require(np.array_equal(cm, raw["confusion_matrix"]), context + ": raw matrix differs")
        for other in (exported["metrics"]["raw"], result["metrics_by_split"]["test"]["raw"]):
            for key in (*METRICS, "n", "negative_support", "positive_support", "tn", "fp", "fn", "tp"):
                close(raw[key], other[key], context + ": metric binding differs: " + key)
        cm_rows = [x for x in inputs.csv(run_dir / "test/analysis/confusion_matrix.csv")
                   if x["probability_type"] == "raw"]
        require(len(cm_rows) == 4, context + ": expected four matrix cells")
        cells = {(int(x["true_label"]), int(x["predicted_label"])): int(x["count"]) for x in cm_rows}
        require(cells == {(i, j): int(cm[i, j]) for i in (0, 1) for j in (0, 1)}, context + ": CM CSV differs")
        run = {"task": TASKS[strand], "strand": strand, "seed": seed, "run_id": run_id,
               "raw": raw, "cm": cm, "cm_norm": cm / cm.sum(axis=1, keepdims=True),
               "test_fingerprints": signature, "checkpoint_sha256": exported["checkpoint_sha256"]}
        scan_rows = [x for x in inputs.csv(run_dir / "test/analysis/confidence_threshold_scan.csv")
                     if x["probability_type"] == "raw"]
        selected = []
        for threshold in THRESHOLDS:
            candidates = [x for x in scan_rows if math.isclose(float(x["confidence_threshold"]), threshold, abs_tol=1e-12)]
            require(len(candidates) == 1, context + f": missing/duplicate confidence threshold {threshold}")
            row = candidates[0]
            total, kept = int(row["total_count"]), int(row["retained_count"])
            counts = [int(row[key]) for key in ("tn", "fp", "fn", "tp")]
            require(total == raw["n"] and 0 < kept <= total and sum(counts) == kept,
                    context + ": confidence counts differ")
            close(row["decision_threshold"], .5, context + ": unexpected decision threshold")
            close(row["coverage"], kept / total, context + ": confidence coverage differs")
            close(row["accuracy_among_retained"], (counts[0] + counts[3]) / kept,
                  context + ": retained accuracy differs")
            selected.append({"threshold": threshold, "total_count": total, "retained_count": kept,
                             "accuracy": float(row["accuracy_among_retained"]), "retained_fraction": kept / total,
                             **dict(zip(("tn", "fp", "fn", "tp"), counts))})
        require(selected[0]["retained_count"] == raw["n"], context + ": threshold .5 must retain all samples")
        close(selected[0]["accuracy"], raw["accuracy"], context + ": threshold .5 accuracy differs")
        require(all(a["retained_count"] >= b["retained_count"] for a, b in zip(selected, selected[1:])),
                context + ": retained counts are not monotonic")
        run["confidence"] = selected
        for panel, filename, xkey, ykey, metric in (
            ("E", "roc_curve.csv", "fpr", "tpr", "auroc"),
            ("F", "pr_curve.csv", "recall", "precision", "auprc")):
            if panel not in panels:
                continue
            path = run_dir / "test/analysis" / filename
            require(path.is_file() or path.with_suffix(path.suffix + ".gz").is_file(),
                    f"Missing curve CSV for {context}: {filename}")
            rows = [x for x in inputs.csv(path) if x["probability_type"] == "raw"]
            x, y = (np.array([float(row[key]) for row in rows]) for key in (xkey, ykey))
            require(len(x) >= 2 and np.isfinite(x).all() and np.isfinite(y).all(), context + ": invalid curve")
            require(((x >= 0) & (x <= 1) & (y >= 0) & (y <= 1)).all(), context + ": curve outside [0,1]")
            if panel == "E":
                require((np.diff(x) >= -1e-12).all() and (np.diff(y) >= -1e-12).all(), context + ": nonmonotonic ROC")
                close(x[0], 0, context + ": ROC origin"); close(y[0], 0, context + ": ROC origin")
                close(x[-1], 1, context + ": ROC end"); close(y[-1], 1, context + ": ROC end")
                area = float(np.trapezoid(y, x) if hasattr(np, "trapezoid") else np.trapz(y, x))
            else:
                require((np.diff(x) <= 1e-12).all(), context + ": PR recall must be descending")
                close(x[0], 1, context + ": PR start"); close(x[-1], 0, context + ": PR end")
                area = float(-np.sum(np.diff(x) * y[:-1]))
            close(area, raw[metric], context + ": curve/statistic differs", atol=1e-8)
            run[panel] = (x, y)
        runs.append(run)
    require(observed == expected, "All six requested runs are required")
    runs.sort(key=lambda run: (0 if run["task"] == "X" else 1, run["seed"]))
    for task in ("X", "Y"):
        task_runs = [run for run in runs if run["task"] == task]
        require(len({(x["raw"]["n"], x["raw"]["positive_support"], x["raw"]["negative_support"])
                     for x in task_runs}) == 1, task + ": support differs across seeds")
    return runs


def moments(values):
    array = np.asarray(values, dtype=float)
    return np.mean(array, axis=0), np.std(array, axis=0, ddof=1)


def write_csv(path, rows):
    require(bool(rows), "Cannot write an empty table: " + str(path))
    with path.open("w", encoding="utf-8-sig", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader(); writer.writerows(rows)


def write_tables(runs, output):
    per_seed, summary, confusion, confidence, confidence_summary = [], [], [], [], []
    for run in runs:
        per_seed.append({"task": run["task"], "seed": run["seed"], "n": run["raw"]["n"],
                         "negative_support": run["raw"]["negative_support"], "positive_support": run["raw"]["positive_support"],
                         **{key: run["raw"][key] for key in METRICS}})
        for row in run["confidence"]:
            confidence.append({"task": run["task"], "seed": run["seed"], **row})
    for task in ("X", "Y"):
        selected = [run for run in runs if run["task"] == task]
        for key in METRICS:
            mean, sd = moments([run["raw"][key] for run in selected])
            summary.append({"task": task, "metric": key, "mean": float(mean), "sample_sd": float(sd), "seed_count": 3})
        mean, sd = moments([run["cm_norm"] for run in selected])
        for i in (0, 1):
            for j in (0, 1):
                confusion.append({"task": task, "true_label": i, "predicted_label": j,
                                  "mean_proportion": float(mean[i, j]), "sample_sd": float(sd[i, j]),
                                  "true_class_support_per_seed": int(selected[0]["cm"][i].sum()), "seed_count": 3})
        for index, threshold in enumerate(THRESHOLDS):
            row = {"task": task, "threshold": threshold, "seed_count": 3}
            for key in ("accuracy", "retained_fraction", "retained_count"):
                mean, sd = moments([run["confidence"][index][key] for run in selected])
                row[key + "_mean"], row[key + "_sample_sd"] = float(mean), float(sd)
            confidence_summary.append(row)
    for name, rows in (("per_seed_metrics", per_seed), ("metrics_mean_sd", summary),
                       ("confusion_mean_sd", confusion), ("confidence_per_seed", confidence),
                       ("confidence_mean_sd", confidence_summary)):
        write_csv(output / (name + ".csv"), rows)


def style(ax, panel, zoom=False):
    if panel in ("E", "F"):
        tick_size = ZOOM_TICK_FONTSIZE if zoom else 12
    elif panel == "G":
        tick_size = G_TICK_FONTSIZE
    elif panel in ("H", "I"):
        tick_size = CM_TICK_FONTSIZE
    else:
        tick_size = JK_TICK_FONTSIZE
    if panel == "G":
        ax.tick_params(direction="out", width=G_TICK_LINEWIDTH, length=G_TICK_LENGTH,
                       pad=G_TICK_PAD, labelsize=tick_size, colors="black")
        ax.tick_params(axis="x", labelsize=G_XTICK_FONTSIZE)
        ax.set_box_aspect(G_BOX_ASPECT)
    else:
        ax.tick_params(direction="out", width=1.6, length=5, labelsize=tick_size)
    ax.xaxis.get_offset_text().set_fontsize(tick_size)
    ax.yaxis.get_offset_text().set_fontsize(tick_size)
    for spine in ax.spines.values():
        spine.set_linewidth(G_SPINE_LINEWIDTH if panel == "G" else
                            JK_SPINE_LINEWIDTH if panel in ("J", "K") else 1.6)
        if panel == "G":
            spine.set_color("black")
    # Panel letters are added later in the user's figure layout software.


def bounds(values, lower=0., upper=1., min_span=.01):
    array = np.asarray(values, dtype=float)
    low, high = np.nanmin(array), np.nanmax(array)
    pad = max((high - low) * .15, min_span / 2)
    return max(lower, low - pad), min(upper, high + pad)


def draw_panel(panel, ax, runs, full_range=False, zoom=False):
    require(not zoom or panel in ("E", "F"), "zoom is only supported for E/F")
    style(ax, panel, zoom=zoom)
    if panel in ("E", "F"):
        for task in ("X", "Y"):
            selected = [run for run in runs if run["task"] == task]
            key = "auroc" if panel == "E" else "auprc"
            mean, sd = moments([run["raw"][key] for run in selected])
            for index, run in enumerate(selected):
                x, y = run[panel]
                label = f"{task}: {'AUROC' if panel == 'E' else 'AP'} {mean:.4f} ± {sd:.4f}" if index == 0 else None
                ax.plot(x, y, color=COLORS[task], lw=1.6, alpha=.8,
                        linestyle=("-", "--", ":")[index], label=label)
        if panel == "E":
            ax.plot([0, 1], [0, 1], color=".6", linestyle="--", lw=1)
            ax.set(xlabel="False positive rate", ylabel="True positive rate")
        else:
            # Both test sets are balanced; the no-skill PR reference equals prevalence.
            prevalence = {run["raw"]["positive_support"] / run["raw"]["n"] for run in runs}
            require(len(prevalence) == 1, "PR baseline requires common prevalence")
            ax.axhline(prevalence.pop(), color=".6", linestyle="--", lw=1)
            ax.set(xlabel="Recall", ylabel="Precision")
        ax.set(xlim=(0, 1), ylim=(0, 1))
        ax.legend(loc="lower right" if panel == "E" else "lower left", frameon=False,
                  prop={"size": ZOOM_LEGEND_FONTSIZE if zoom else 10, "weight": "bold"})
        ax.set_box_aspect(1)
        if zoom:
            ax.set(xlim=(0., .4) if panel == "E" else (.6, 1.), ylim=(.6, 1.))
            for axis in (ax.xaxis, ax.yaxis):
                axis.set_major_locator(MultipleLocator(.1))
                axis.set_major_formatter(FormatStrFormatter("%.1f"))
            ax.set_title("CB (Canonical Base) vs X/Y " + ("ROC" if panel == "E" else "PR"),
                         fontsize=ZOOM_TITLE_FONTSIZE, fontweight="bold", pad=12)
    elif panel == "G":
        all_values = []
        for offset, task in ((-.14, "X"), (.14, "Y")):
            selected = [run for run in runs if run["task"] == task]
            data = np.array([[run["raw"][key] for key in METRICS] for run in selected])
            mean, sd = moments(data)
            x = np.arange(len(METRICS)) + offset
            all_values.extend((data.ravel(), mean - sd, mean + sd))
            ax.errorbar(x, mean, yerr=sd, fmt="o", color=COLORS[task], ms=7, capsize=4,
                        lw=1.6, label=f"CB (Canonical Base) vs {task}", zorder=4)
            for index, values in enumerate(data):
                ax.scatter(x + (index - 1) * .022, values, s=13, color=COLORS[task], alpha=.55, zorder=5)
        low, high = (0., 1.) if full_range else G_YLIM
        for offset, task in ((-.14, "X"), (.14, "Y")):
            mean, _ = moments([[run["raw"][key] for key in METRICS] for run in runs if run["task"] == task])
            ax.vlines(np.arange(len(METRICS)) + offset, low, mean, color=COLORS[task], alpha=.4, lw=1.5)
        ax.set(ylim=(low, high), ylabel="Score", xlim=(-.6, 7.6))
        ax.set_xticks(range(len(METRICS)), METRIC_LABELS, rotation=35, ha="right")
        ax.yaxis.set_major_formatter(FormatStrFormatter("%.2f"))
        if not full_range:
            ax.set_yticks(np.arange(G_YLIM[0], G_YLIM[1] + G_TICK_STEP / 2, G_TICK_STEP))
            ax.set_ylim(*G_YLIM)
        # The reference Figure2B uses automatic placement. Its wider tick text
        # and the matched frame make a fixed upper-right legend obscure AP.
        # Keep automatic placement inside the main frame; the full-range view
        # retains its downward offset from the near-1.0 data cluster.
        ax.legend(frameon=False, loc="upper right" if full_range else "best",
                  bbox_to_anchor=(1., .90) if full_range else None,
                  borderaxespad=.4, handlelength=1., handletextpad=.4,
                  prop={"size": G_LEGEND_FONTSIZE, "weight": "bold"})
        ax.grid(False)
    elif panel in ("H", "I"):
        task = "X" if panel == "H" else "Y"
        selected = [run for run in runs if run["task"] == task]
        mean, _ = moments([run["cm_norm"] for run in selected])
        image = ax.imshow(mean, vmin=0, vmax=1, cmap="Blues")
        for i in (0, 1):
            for j in (0, 1):
                ax.text(j, i, f"{mean[i, j]:.3f}", ha="center", va="center", fontsize=CM_NUMBER_FONTSIZE,
                        fontweight="bold",
                        color="white" if mean[i, j] > .5 else "black")
        ax.set_xticks([0, 1], ["CB", task]); ax.set_yticks([0, 1], ["CB", task])
        ax.tick_params(axis="both", labelsize=CM_TICK_FONTSIZE)
        ax.set(xlabel="Predicted label", ylabel="True label")
        # AxesDivider ties both colorbar endpoints to the displayed matrix,
        # including when the panel is embedded in a SubFigure.
        divider = make_axes_locatable(ax)
        colorbar_ax = divider.append_axes("right", size="4.6%", pad=.16)
        bar = ax.figure.colorbar(image, cax=colorbar_ax)
        bar.set_label("Proportion", fontsize=CM_COLORBAR_FONTSIZE)
        bar.ax.tick_params(labelsize=CM_COLORBAR_FONTSIZE)
        bar.ax.yaxis.get_offset_text().set_fontsize(CM_COLORBAR_FONTSIZE)
        bar.ax.yaxis.set_major_formatter(FormatStrFormatter("%.1f"))
    else:
        task = "X" if panel == "J" else "Y"
        selected = [run for run in runs if run["task"] == task]
        right = ax.twinx()
        accuracy = np.array([[x["accuracy"] for x in run["confidence"]] for run in selected])
        coverage = np.array([[x["retained_fraction"] for x in run["confidence"]] for run in selected])
        for index in range(3):
            ax.plot(THRESHOLDS, accuracy[index], color=COLORS[task], lw=.8, alpha=.22)
            right.plot(THRESHOLDS, coverage[index], color=".3", lw=.8, alpha=.15)
        for target, data, color, marker, line in ((ax, accuracy, COLORS[task], "o", "-"),
                                                 (right, coverage, ".25", "s", "--")):
            mean, sd = moments(data)
            target.errorbar(THRESHOLDS, mean, yerr=sd, color=color, marker=marker, linestyle=line,
                            lw=2, ms=6, capsize=3)
            # Keep the same scale for X and Y so their confidence plots can be compared.
            key = "accuracy" if target is ax else "retained_fraction"
            candidates = []
            for other_task in ("X", "Y"):
                other_data = np.array([[row[key] for row in run["confidence"]]
                                       for run in runs if run["task"] == other_task])
                other_mean, other_sd = moments(other_data)
                candidates.extend((other_data.ravel(), other_mean - other_sd, other_mean + other_sd))
            low, high = bounds(np.concatenate(candidates), min_span=.02)
            target.set_ylim(max(0., math.floor(low * 200) / 200), min(1., math.ceil(high * 200) / 200))
            target.yaxis.set_major_locator(MaxNLocator(5))
            target.yaxis.set_major_formatter(FormatStrFormatter("%.3f"))
        ax.set(xlabel="Confidence threshold", ylabel="Accuracy", xlim=(.48, .92))
        ax.set_xticks(THRESHOLDS)
        right.set_ylabel("Retaining rate", fontsize=JK_AXIS_FONTSIZE, color=".25")
        right.tick_params(direction="out", width=1.6, length=5, labelsize=JK_TICK_FONTSIZE)
        right.yaxis.get_offset_text().set_fontsize(JK_TICK_FONTSIZE)
        right.tick_params(axis="y", colors=".25")
        ax.yaxis.label.set_color(COLORS[task])
        ax.spines["left"].set_color(COLORS[task])
        ax.tick_params(axis="y", colors=COLORS[task])
        # A twinx frame must not paint a black left spine over the colored one.
        right.spines["left"].set_visible(False)
        right.spines["right"].set_color(".25")
        right.spines["right"].set_linewidth(JK_SPINE_LINEWIDTH)
        # Draw each of the four visible borders exactly once: the primary
        # axes owns top/bottom/colored-left, the twin owns gray-right.
        ax.spines["right"].set_visible(False)
        right.spines["top"].set_visible(False)
        right.spines["bottom"].set_visible(False)
        legend = ax.legend(handles=[Line2D([0], [0], color=COLORS[task], marker="o", label="Accuracy"),
                                    Line2D([0], [0], color=".25", marker="s", linestyle="--", label="Retaining rate")],
                           frameon=False, prop={"size": JK_LEGEND_FONTSIZE, "weight": "bold"}, loc="lower center")
        legend.get_texts()[0].set_color(COLORS[task])
        legend.get_texts()[1].set_color(".25")
        ax.text(.97, .97, task, transform=ax.transAxes, ha="right", va="top",
                color=COLORS[task], fontsize=JK_TASK_FONTSIZE, fontweight="bold")
    if panel in ("H", "I"):
        label_size = CM_AXIS_FONTSIZE
    elif panel == "G":
        label_size = G_AXIS_FONTSIZE
    elif panel in ("J", "K"):
        label_size = JK_AXIS_FONTSIZE
    else:
        label_size = ZOOM_AXIS_FONTSIZE if zoom else 13
    ax.xaxis.label.set_size(label_size); ax.yaxis.label.set_size(label_size)
    if panel == "G":
        # Explicit local weights also cover tick labels created after style().
        # Do not alter global rcParams or the typography of other panels.
        for text in [ax.xaxis.label, ax.yaxis.label,
                     *ax.get_xticklabels(), *ax.get_yticklabels(),
                     ax.xaxis.get_offset_text(), ax.yaxis.get_offset_text()]:
            text.set_fontweight("bold")
        for text in ax.get_xticklabels():
            text.set_fontsize(G_XTICK_FONTSIZE)
        for text in ax.get_yticklabels():
            text.set_fontsize(G_TICK_FONTSIZE)


def create_montage(runs, panels):
    rows = []
    if "E" in panels or "F" in panels:
        rows.append(["E" if "E" in panels else ".", "F" if "F" in panels else "."])
    if "G" in panels:
        rows.append(["G", "G"])
    for left, right in (("H", "I"), ("J", "K")):
        if left in panels or right in panels:
            rows.append([left if left in panels else ".", right if right in panels else "."])
    # Match G's increased height in the montage; the confidence row keeps the
    # narrower standalone panel widths instead of stretching across G's width.
    has_jk = any(panel in panels for panel in ("J", "K"))
    width = max(G_FIGSIZE[0] if "G" in panels else 12., 2 * JK_FIGSIZE[0] if has_jk else 0.)
    heights = [G_FIGSIZE[1] if row == ["G", "G"] else 4.8 for row in rows]
    fig = plt.figure(figsize=(width, sum(heights)), constrained_layout=True)
    subrows = fig.subfigures(len(rows), 1, squeeze=False, hspace=.025,
                            height_ratios=heights)[:, 0]
    for row, subrow in zip(rows, subrows):
        if row == ["G", "G"]:
            draw_panel("G", subrow.add_axes(G_AXES_RECT), runs)
        else:
            if any(panel in ("J", "K") for panel in row) and width > 2 * JK_FIGSIZE[0]:
                margin = (width - 2 * JK_FIGSIZE[0]) / 2
                subrow = subrow.subfigures(1, 3, width_ratios=(margin, 2 * JK_FIGSIZE[0], margin),
                                          wspace=0.)[1]
            for panel, subpanel in zip(row, subrow.subfigures(1, 2, wspace=.025)):
                ax = subpanel.subplots()
                if panel == ".":
                    ax.set_axis_off()
                else:
                    draw_panel(panel, ax, runs)
    return fig


def save_figure(fig, output, stem, dpi):
    for extension in ("png", "pdf", "svg"):
        # Finish the backend export before replacing any existing output file.
        with BytesIO() as buffer:
            fig.savefig(buffer, format=extension, dpi=dpi, bbox_inches="tight", facecolor="white")
            payload = buffer.getvalue()
        if extension == "pdf":
            require(payload.rstrip().endswith(b"%%EOF"), f"Incomplete PDF export: {stem}")
        path = output / f"{stem}.{extension}"
        temporary = path.with_name(path.name + f".{os.getpid()}.tmp")
        try:
            temporary.write_bytes(payload)
            temporary.replace(path)
        finally:
            temporary.unlink(missing_ok=True)
    plt.close(fig)


def caption(runs):
    counts = {task: next(run["raw"]["n"] for run in runs if run["task"] == task) for task in ("X", "Y")}
    return (
        "Figure 2. Binary identification of non-canonical bases. "
        "(A, C) Representative median-MAD-normalized current traces for X versus A in ATAAGA (A) "
        "and Y versus C in ATACGA (C). Ten samples are shown per class. "
        "(B, D) Target-position signal summaries, base quality and mismatch, insertion and deletion responses. "
        "Boxes show medians, interquartile ranges and whiskers extending to 1.5 times the interquartile range. "
        "Bars show the proportion of samples with a nonzero response. "
        "(E, F) ROC and precision-recall curves for GatedConv across three training seeds (42, 43 and 44); "
        "legends report mean ± SD of AUROC and average precision (AP). "
        "(G) Classification metrics (mean ± SD; small points denote individual seeds). Precision, recall and F1 "
        "treat X or Y as positive; AUPRC is computed as AP. "
        "(H, I) Mean confusion matrices after true-label row normalization within each seed. "
        "(J, K) Accuracy among retained samples and retaining rate (retained fraction; mean ± SD) at confidence thresholds "
        "from 0.5 to 0.9; samples with max[p(CB), p(X or Y)] ≥ threshold are retained. "
        "E–K use uncalibrated probabilities and a positive-class decision threshold of 0.5. "
        f"The fixed read-disjoint test sets contain {counts['X']:,} X-task samples and {counts['Y']:,} Y-task samples, "
        "with equal positive and canonical-control counts within each task; the same test samples are evaluated "
        "for every seed. SD denotes sample standard deviation across training seeds. CB, canonical base.\n"
    )


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--results_dir", type=Path, default=Path(DEFAULT_RESULTS))
    parser.add_argument("--output-dir", "--output_dir", type=Path, default=PACKAGE_DIR / "outputs")
    parser.add_argument("--panels", nargs="+", choices=list("EFGHIJK"), default=["E", "F"])
    parser.add_argument("--dpi", type=int, default=600)
    args = parser.parse_args()
    require(72 <= args.dpi <= 1200, "dpi must be between 72 and 1200")
    require(len(args.panels) == len(set(args.panels)), "Duplicate panel requests")
    inputs = Inputs(args.results_dir)
    # Validate all requested inputs before creating any output; never silently skip seeds or curves.
    runs = load_runs(inputs, args.panels)
    output = args.output_dir.resolve()
    require(not any(output == (inputs.root / run["run_id"]).resolve() for run in runs), "Output cannot be a run directory")
    output.mkdir(parents=True, exist_ok=True)
    fonts = {font.name for font in font_manager.fontManager.ttflist}
    chosen_font = "Arial" if "Arial" in fonts else "DejaVu Sans"
    plt.rcParams.update({"font.family": chosen_font, "pdf.fonttype": 42, "ps.fonttype": 42,
                         "svg.fonttype": "none", "axes.unicode_minus": False})
    write_tables(runs, output)
    (output / "Figure2_caption.txt").write_text(caption(runs), encoding="utf-8")
    generated_stems = []
    for panel in args.panels:
        fig, ax = create_panel_figure(panel)
        draw_panel(panel, ax, runs)
        save_figure(fig, output, f"Figure2{panel}", args.dpi)
        generated_stems.append(f"Figure2{panel}")
        print(f"[PLOT] {output / ('Figure2' + panel)}.png/.pdf/.svg", flush=True)
        if panel in ("E", "F", "G"):
            fig, ax = create_panel_figure(panel)
            draw_panel(panel, ax, runs, full_range=(panel == "G"), zoom=panel in ("E", "F"))
            variant_stem = f"Figure2{panel}_{'full_range' if panel == 'G' else 'zoom'}"
            save_figure(fig, output, variant_stem, args.dpi)
            generated_stems.append(variant_stem)
            print(f"[PLOT] {output / variant_stem}.png/.pdf/.svg", flush=True)
    # A vector montage is convenient for review; individual panels retain the requested 600-dpi resolution.
    fig = create_montage(runs, args.panels)
    save_figure(fig, output, "Figure2_selected_panels", min(args.dpi, 300))
    generated_stems.append("Figure2_selected_panels")
    generated_names = [stem + "." + extension for stem in generated_stems for extension in ("png", "pdf", "svg")]
    generated_names += [name + ".csv" for name in ("per_seed_metrics", "metrics_mean_sd", "confusion_mean_sd",
                                                   "confidence_per_seed", "confidence_mean_sd")]
    generated_names.append("Figure2_caption.txt")
    files = {name: {"sha256": sha256(output / name), "size": (output / name).stat().st_size}
             for name in sorted(generated_names)}
    manifest = {
        "version": VERSION, "created_at": datetime.now(timezone.utc).isoformat(),
        "status": "complete_requested_panels", "requested_panels": args.panels,
        "dpi": args.dpi,
        "montage_dpi": min(args.dpi, 300), "font": chosen_font,
        "script_sha256": sha256(Path(__file__)), "inputs": inputs.files, "outputs": files,
        "seeds": list(SEEDS), "selection": "all six fixed completed runs; no test-based seed selection",
        "probability_type": "raw", "decision_threshold": .5,
        "confidence_rule": "max(p0,p1) >= threshold", "confidence_thresholds": list(THRESHOLDS),
        "metric_definitions": {"precision_recall_f1": "positive class X/Y", "auprc": "average precision (AP), not trapezoidal area",
                               "seed_error_bars": "sample standard deviation, ddof=1; not a test-sample confidence interval",
                               "confusion": "mean of each seed's true-row-normalized matrix",
                               "curves": "all individual seed curves, without averaging predictions or fabricating a mean curve"},
        "run_bindings": [{key: run[key] for key in ("task", "seed", "run_id", "checkpoint_sha256", "test_fingerprints")}
                         for run in runs],
        "verification_scope": "Saved metadata/checkpoint/data/metric bindings and internal count checks; no new feature or read-ID audit. "
                              "Input payload hashes are observed at plotting time, not claimed historical producer pins.",
        "caption_scope": "A–D text retained from supplied original caption; this script does not regenerate A–D.",
        "visual_settings": {"panel_letters": False, "bold_legends": True, "G_ylim": list(G_YLIM),
                            "G_tick_step": G_TICK_STEP, "G_tick_step_note": "0.2 requested; interpreted as 0.02 within 0.92–1.00",
                            "JK_accuracy_scope": "both binary classes among retained samples, (TN+TP)/retained_count",
                            "JK_right_axis_label": "Retaining rate", "zoom_titles_bold": True,
                            "G_legend_inside_frame": True, "CM_colorbar_matches_matrix_height": True,
                            "G_legend_placement": "best (original Figure2B rule); full-range upper right with downward offset",
                            "zoom_fontsizes": {"title": ZOOM_TITLE_FONTSIZE, "axis": ZOOM_AXIS_FONTSIZE,
                                               "tick": ZOOM_TICK_FONTSIZE, "legend": ZOOM_LEGEND_FONTSIZE},
                            "G_fontsizes": {"axis": G_AXIS_FONTSIZE, "tick": G_TICK_FONTSIZE,
                                            "x_tick": G_XTICK_FONTSIZE,
                                            "legend": G_LEGEND_FONTSIZE},
                            "G_axis_and_tick_fontweight": "bold",
                            "G_spine_linewidth": G_SPINE_LINEWIDTH,
                            "G_tick_linewidth": G_TICK_LINEWIDTH,
                            "G_tick_length": G_TICK_LENGTH, "G_tick_pad": G_TICK_PAD,
                            "G_xtick_rotation_degrees": 35,
                            "G_axes_rect": list(G_AXES_RECT), "G_box_aspect_height_over_width": G_BOX_ASPECT,
                            "G_style_reference": "supplied original Figure2B; explicit bold axis and tick text requested",
                            "CM_fontsizes": {"axis": CM_AXIS_FONTSIZE, "tick": CM_TICK_FONTSIZE,
                                             "colorbar": CM_COLORBAR_FONTSIZE, "number": CM_NUMBER_FONTSIZE},
                            "CM_number_fontweight": "bold",
                            "JK_fontsizes": {"axis": JK_AXIS_FONTSIZE, "tick": JK_TICK_FONTSIZE,
                                             "legend": JK_LEGEND_FONTSIZE, "task": JK_TASK_FONTSIZE},
                            "JK_all_four_border_linewidth": JK_SPINE_LINEWIDTH,
                            "G_figsize_inches": list(G_FIGSIZE), "JK_figsize_inches": list(JK_FIGSIZE),
                            "G_height_scale_vs_v1_0_4": G_FIGSIZE[1] / 4.5,
                            "JK_width_scale_vs_v1_0_4": JK_FIGSIZE[0] / 7.6,
                            "G_legend_handlelength": 1., "G_legend_handletextpad": .4,
                            "montage_confidence_row_centered": True,
                            "montage_figsize_inches": list(fig.get_size_inches())},
    }
    (output / "plot_manifest.json").write_text(json.dumps(manifest, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    print(f"[FULL PASS] Figure2 requested panels complete: {output}", flush=True)


if __name__ == "__main__":
    try:
        main()
    except (ValueError, KeyError, OSError) as error:
        print(f"[ERROR] {type(error).__name__}: {error}", file=sys.stderr)
        sys.exit(1)
