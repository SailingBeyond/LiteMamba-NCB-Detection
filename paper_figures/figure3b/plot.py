#!/usr/bin/env python3
"""Reproduce Figure 3B and Supplementary Table S1 from frozen test counts."""

import argparse
from csv import DictReader, writer
from math import sqrt
from pathlib import Path
from statistics import mean, stdev

import matplotlib

matplotlib.use("Agg")
from matplotlib import font_manager, pyplot as plt
from matplotlib.ticker import MaxNLocator


TASK_LABELS = {
    "ATCGXY": "ATCGXY",
    "ATCGX": "ATCGX",
    "ATCGY": "ATCGY",
}
MODEL_COLORS = {
    "BiGRU": "#4C72B0",
    "Transformer": "#DD8452",
    "ConvMixer": "#55A868",
    "GatedConv": "#C44E52",
}
SEEDS = (42, 2024, 2025)
TEST_SIZES = {"ATCGXY": 250188, "ATCGX": 1225160, "ATCGY": 1194720}
METRICS = ("accuracy", "macro_precision", "macro_recall", "macro_f1", "mcc")
COUNT_FIELDS = ("task", "model", "seed", "true_label", "predicted_label", "count")
DATA_FILE = Path(__file__).resolve().parent / "data" / "per_seed_confusion_counts.csv"


def read_matrices(path):
    counts = {}
    with path.open(encoding="utf-8", newline="") as stream:
        rows = DictReader(stream)
        if tuple(rows.fieldnames or ()) != COUNT_FIELDS:
            raise ValueError(f"Unexpected header in {path}")
        for row in rows:
            try:
                task, model, seed, actual, predicted, count = (row[field] for field in COUNT_FIELDS)
                seed, count = int(seed), int(count)
            except (TypeError, ValueError) as error:
                raise ValueError(f"Malformed count at line {rows.line_num}") from error
            key = (task, model, seed, actual, predicted)
            labels = TASK_LABELS.get(task, "")
            if (
                None in row or len(actual or "") != 1 or len(predicted or "") != 1
                or actual not in labels or predicted not in labels
                or model not in MODEL_COLORS or seed not in SEEDS or count < 0
                or key in counts
            ):
                raise ValueError(f"Invalid or duplicate count at line {rows.line_num}: {key}")
            counts[key] = count

    matrices = {}
    for task, labels in TASK_LABELS.items():
        for model in MODEL_COLORS:
            for seed in SEEDS:
                key = (task, model, seed)
                try:
                    matrix = [[counts[(*key, a, p)] for p in labels] for a in labels]
                except KeyError as error:
                    raise ValueError(f"Missing count for {error.args[0]}") from error
                if sum(map(sum, matrix)) != TEST_SIZES[task]:
                    raise ValueError(f"Unexpected test-set size for {key}")
                matrices[key] = matrix
    return matrices


def metrics_from_matrix(matrix):
    actual = [sum(row) for row in matrix]
    predicted = [sum(column) for column in zip(*matrix)]
    correct = [row[i] for i, row in enumerate(matrix)]
    total = sum(actual)
    trace = sum(correct)
    precision = mean(hit / pred if pred else 0.0 for hit, pred in zip(correct, predicted))
    recall = mean(hit / true if true else 0.0 for hit, true in zip(correct, actual))
    f1 = mean(
        2 * hit / (true + pred) if true + pred else 0.0
        for hit, true, pred in zip(correct, actual, predicted)
    )
    numerator = trace * total - sum(true * pred for true, pred in zip(actual, predicted))
    denominator = sqrt(
        (total**2 - sum(count**2 for count in actual))
        * (total**2 - sum(count**2 for count in predicted))
    )
    return trace / total, precision, recall, f1, numerator / denominator if denominator else 0.0


def summarize(per_seed):
    summary = {}
    for task in TASK_LABELS:
        for model in MODEL_COLORS:
            runs = (per_seed[task, model, seed] for seed in SEEDS)
            summary[task, model] = [(mean(values), stdev(values)) for values in zip(*runs)]
    return summary


def write_tables(output_dir, per_seed, summary):
    with (output_dir / "Figure3B_per_seed.csv").open("w", encoding="utf-8", newline="") as stream:
        table = writer(stream, lineterminator="\n")
        table.writerow(("task", "model", "seed", "n", *METRICS))
        for (task, model, seed), values in per_seed.items():
            table.writerow((task, model, seed, TEST_SIZES[task], *values))

    with (output_dir / "Supplementary_Table_S1.csv").open(
        "w", encoding="utf-8", newline=""
    ) as stream:
        table = writer(stream, lineterminator="\n")
        table.writerow(
            ("Task", "Model", "Accuracy (%)", "Macro precision (%)",
             "Macro recall (%)", "Macro F1 (%)", "MCC")
        )
        for (task, model), values in summary.items():
            percentages = [f"{100 * average:.2f} ± {100 * sd:.2f}" for average, sd in values[:4]]
            mcc, mcc_sd = values[4]
            table.writerow((task, model, *percentages, f"{mcc:.4f} ± {mcc_sd:.4f}"))


def draw_accuracy(output_dir, per_seed, summary, dpi):
    available = {font.name for font in font_manager.fontManager.ttflist}
    family = "Arial" if "Arial" in available else "DejaVu Sans"
    plt.rcParams.update(
        {
            "font.family": family, "font.size": 16, "axes.labelsize": 18,
            "xtick.labelsize": 15, "ytick.labelsize": 15, "axes.linewidth": 1.5,
            "xtick.major.width": 1.4, "ytick.major.width": 1.4,
            "xtick.major.size": 5.5, "ytick.major.size": 5.5, "pdf.fonttype": 42,
            "figure.facecolor": "white", "axes.facecolor": "white",
        }
    )

    fig, ax = plt.subplots(figsize=(5, 5.5))
    task_positions = range(len(TASK_LABELS))
    limits = []
    for (model, color), offset in zip(MODEL_COLORS.items(), (-0.24, -0.08, 0.08, 0.24)):
        positions = [x + offset for x in task_positions]
        averages, deviations = zip(*(summary[task, model][0] for task in TASK_LABELS))
        ax.errorbar(
            positions, averages, yerr=deviations, fmt="o", markersize=6.5, capsize=4,
            linewidth=1.7, color=color, label=model, zorder=4,
        )
        for average, sd in zip(averages, deviations):
            limits.extend((average - sd, average + sd))
        for index, seed in enumerate(SEEDS):
            offsets = [x + (index - 1) * 0.026 for x in positions]
            accuracy = [per_seed[task, model, seed][0] for task in TASK_LABELS]
            ax.scatter(offsets, accuracy, s=20, color=color, alpha=0.45, zorder=3)
            limits.extend(accuracy)

    low, high = min(limits), max(limits)
    pad = max(0.005, 0.18 * (high - low))
    ax.set_ylim(low - pad, high + pad)
    ax.set_xticks(
        task_positions, [f"{task}\n{len(labels)}-Class" for task, labels in TASK_LABELS.items()],
        fontsize=13,
    )
    ax.set_xlim(-0.55, 2.55)
    ax.set_xlabel("Classification task")
    ax.set_ylabel("Accuracy")
    ax.yaxis.set_major_locator(MaxNLocator(6))
    ax.grid(axis="y", alpha=0.22, linewidth=0.8)
    ax.legend(frameon=False, fontsize=11, loc="best", ncol=2)
    ax.tick_params(direction="out")
    fig.tight_layout()
    try:
        for extension in ("png", "pdf"):
            fig.savefig(
                output_dir / f"Figure3B_accuracy.{extension}",
                dpi=dpi, bbox_inches="tight", facecolor="white",
            )
    finally:
        plt.close(fig)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", type=Path, default=Path("reproduced/figure3b"))
    parser.add_argument("--dpi", type=int, default=600)
    args = parser.parse_args()
    if args.dpi < 1:
        parser.error("--dpi must be a positive integer.")
    per_seed = {key: metrics_from_matrix(matrix) for key, matrix in read_matrices(DATA_FILE).items()}
    summary = summarize(per_seed)
    args.output_dir.mkdir(parents=True, exist_ok=True)
    write_tables(args.output_dir, per_seed, summary)
    draw_accuracy(args.output_dir, per_seed, summary, args.dpi)
    print(f"Reproduced Figure 3B and Supplementary Table S1 in {args.output_dir.resolve()}")


if __name__ == "__main__":
    main()
