#!/usr/bin/env python3
"""Replot Supplementary Figure S7 from the archived UMAP coordinates."""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib import font_manager
from matplotlib.ticker import ScalarFormatter
import numpy as np
import pandas as pd
from PIL import Image, ImageDraw, ImageFont

PACKAGE_DIR = Path(__file__).resolve().parent

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


def compose_panels(figure_dir, dpi):
    panels = []
    for task in ("5X", "5Y"):
        path = figure_dir / f"Supplementary_UMAP_{task}_GatedConv_seed42.png"
        with Image.open(path) as source:
            width = 2280
            height = round(source.height * width / source.width)
            panels.append(source.convert("RGB").resize((width, height), Image.Resampling.LANCZOS))

    margin, gap, label_height = 28, 54, 112
    width = sum(panel.width for panel in panels) + 2 * margin + gap
    height = max(panel.height for panel in panels) + label_height + margin
    figure = Image.new("RGB", (width, height), "white")
    draw = ImageDraw.Draw(figure)
    font_path = font_manager.findfont(font_manager.FontProperties(family="DejaVu Sans", weight="bold"))
    font = ImageFont.truetype(font_path, 96)
    for index, (letter, panel) in enumerate(zip("AB", panels)):
        x = margin + index * (panel.width + gap)
        figure.paste(panel, (x, label_height))
        draw.text((x + 3, 8), letter, font=font, fill="black", anchor="lt")

    destination = figure_dir / "Supplementary_Figure_S7_GatedConv.png"
    figure.save(destination, dpi=(dpi, dpi), optimize=True)
    return destination


def main():
    args = parse_options()
    for task in ("5X", "5Y"):
        verify_embedding(f"Supplementary_UMAP_{task}")
    configure_style()
    report = replot_saved_umaps(PACKAGE_DIR, args.output_dir, tasks=("5X", "5Y"), dpi=args.dpi)
    if report["missing"]:
        raise MissingInput(report["missing"])
    destination = compose_panels(args.output_dir / "figures", args.dpi)
    print(f"Supplementary Figure S7 written to {destination.resolve()}")


if __name__ == "__main__":
    main()
