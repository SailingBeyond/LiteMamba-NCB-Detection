#!/usr/bin/env python3
"""Redraw the four external-evaluation panels from the packaged six-class counts."""

import argparse
import os
import tempfile
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

from figure4_data import COHORTS, load_data
from figure4_tables import build_tables
from panels import draw_panel


ROOT = Path(__file__).resolve().parent
DATA = ROOT / "data"
TABLE_NAMES = {
    "A": "identity_recall",
    "B": "templates",
    "C": "reference_strand_contexts",
    "D": "error_destinations",
}
FACETS = {"C": COHORTS, "D": ("X", "Y")}


def parse_arguments():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--panels", nargs="+", type=str.upper, choices=list(TABLE_NAMES),
                        default=list(TABLE_NAMES), help="Panels to write (default: A B C D)")
    parser.add_argument("--formats", nargs="+", type=str.lower, choices=["png", "pdf", "svg"],
                        default=["png", "pdf", "svg"])
    parser.add_argument("--dpi", type=int, default=300, help="PNG resolution (default: 300)")
    parser.add_argument("--output-dir", "--output_dir", type=Path,
                        default=ROOT / "outputs")
    parser.add_argument("--no-subpanels", action="store_true",
                        help="Write C and D only in their respective multi-axis layouts")
    args = parser.parse_args()
    if args.dpi <= 0:
        parser.error("--dpi must be positive")
    args.panels = list(dict.fromkeys(args.panels))
    args.formats = list(dict.fromkeys(args.formats))
    return args


def save_figure(fig, base, formats, dpi):
    base.parent.mkdir(parents=True, exist_ok=True)
    try:
        for fmt in formats:
            fig.savefig(base.with_suffix("." + fmt), format=fmt, dpi=dpi, facecolor="white")
    finally:
        plt.close(fig)


def save_overview(directory, dpi):
    """Arrange the four completed PNGs in the manuscript's three rows."""
    fig = plt.figure(figsize=(11, 10.9), facecolor="white")
    layout = {
        "A": (0.015, 0.671, 0.48, 0.30),
        "B": (0.508, 0.671, 0.48, 0.30),
        "C": (0.022, 0.337, 0.965, 0.329),
        "D": (0.022, 0.008, 0.965, 0.318),
    }
    try:
        for panel, rect in layout.items():
            axis = fig.add_axes(rect)
            axis.imshow(plt.imread(directory / f"Figure4{panel}.png"), aspect="equal")
            axis.set_axis_off()
            box = axis.get_position()
            fig.text(0.51 if panel == "B" else 0.018, box.y1 - 0.009 * box.height,
                     panel, va="top", fontsize=16, fontweight="bold", color="#1B2935")
        fig.savefig(directory / "Figure4_overview.png", dpi=dpi, facecolor="white")
    finally:
        plt.close(fig)


def main():
    args = parse_arguments()
    output = args.output_dir.expanduser().resolve()
    print("Reading the bundled external prediction counts ...", flush=True)
    data = load_data(DATA / "figure4_gatedconv_threshold_statistics_v1_0_0.tar.gz",
                     DATA / "figure4_gatedconv_per6mer_core_v1_0_0.tar.gz",
                     probability_type="raw", threshold=0.0)
    tables = build_tables(data)
    output.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory(dir=output, prefix=".render-") as tmp:
        stage = Path(tmp)
        (stage / "source_data").mkdir()
        settings = {"pdf.fonttype": 42, "ps.fonttype": 42, "svg.fonttype": "none"}
        with matplotlib.rc_context(settings):
            for panel in args.panels:
                prefix = "Figure4" + panel
                print("Drawing " + prefix, flush=True)
                for suffix in ("", "_by_seed"):
                    name = f"{prefix}_{TABLE_NAMES[panel]}{suffix}.csv"
                    tables[panel + suffix].to_csv(stage / "source_data" / name, index=False)
                save_figure(draw_panel(panel, tables), stage / "figures" / prefix,
                            args.formats, args.dpi)
                if not args.no_subpanels:
                    for facet in FACETS.get(panel, ()):
                        label = "true_" + facet if panel == "D" else facet
                        base = stage / "subpanels" / f"{prefix}_{label}"
                        save_figure(draw_panel(panel, tables, facet=facet), base,
                                    args.formats, args.dpi)
            if set(args.panels) == set(TABLE_NAMES) and "png" in args.formats:
                save_overview(stage / "figures", args.dpi)

        # Publish only complete files, leaving outputs for unrequested panels in place.
        for path in sorted(stage.rglob("*")):
            if path.is_file():
                dest = output / path.relative_to(stage)
                dest.parent.mkdir(parents=True, exist_ok=True)
                os.replace(path, dest)
    print(f"[DONE] {' '.join(args.panels)} | {output}", flush=True)


if __name__ == "__main__":
    main()
