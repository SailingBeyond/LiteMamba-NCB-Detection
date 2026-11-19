#!/usr/bin/env python3
"""Draw Supplementary Figure S6 from the frozen five-class test counts."""

import argparse
import hashlib
import json
from pathlib import Path

import matplotlib
from PIL import Image, ImageDraw, ImageFont

from collect_results import collect_results
from plot_common import configure_style, plot_heatmaps, plt


HERE = Path(__file__).resolve().parent
TASKS = ("5X", "5Y")
PANEL_DPI = 110
SYSTEM_LABEL_FONT = Path("/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf")
COMPOSITE = "Supplementary_Figure_S6_GatedConv.png"


def panel_name(task):
    return f"Supplementary_motif_heatmap_{task}_GatedConv.png"


def compose(panel_paths, destination, font_path):
    panels = []
    for panel_path in panel_paths:
        with Image.open(panel_path) as panel:
            panels.append(panel.convert("RGB"))

    width, height = panels[0].size
    if panels[1].size != (width, height):
        raise ValueError("The five-class heatmaps have different canvas sizes.")

    outer, gap = 18, 44
    image = Image.new("RGB", (width + 2 * outer, 2 * height + 2 * outer + gap), "white")
    draw = ImageDraw.Draw(image)
    font = ImageFont.truetype(str(font_path), 82)
    for index, (label, panel) in enumerate(zip("AB", panels)):
        top = outer + index * (height + gap)
        image.paste(panel, (outer, top))
        position = (outer + 8, top + 8)
        box = draw.textbbox(position, label, font=font, anchor="lt")
        if image.crop(box).getextrema() != ((255, 255),) * 3:
            raise ValueError(f"Panel letter {label} would cover the heatmap.")
        draw.text(position, label, fill="black", font=font, anchor="lt")

    image.save(destination, dpi=(600, 600), optimize=True)


def digest(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", type=Path, default=HERE / "results")
    parser.add_argument("--label-font", type=Path, help="DejaVu Sans Bold font for the A and B labels")
    parser.add_argument("--verify", action="store_true", help="Require byte-identical archived PNG files")
    args = parser.parse_args()

    output = args.output_dir.expanduser().resolve()
    default_font = SYSTEM_LABEL_FONT if SYSTEM_LABEL_FONT.is_file() else (
        Path(matplotlib.get_data_path()) / "fonts/ttf/DejaVuSans-Bold.ttf"
    )
    font = args.label_font or default_font

    collect_results(output)
    configure_style()
    plt.rcParams["font.family"] = "DejaVu Sans"
    plot_heatmaps(output, tasks=TASKS, dpi=PANEL_DPI)

    figures = output / "figures"
    names = [panel_name(task) for task in TASKS]
    compose([figures / name for name in names], figures / COMPOSITE, font)

    all_names = names + [COMPOSITE]
    outputs = {}
    for name in all_names:
        produced = digest(figures / name)
        archived = digest(HERE / "reference" / name)
        outputs[name] = {"sha256": produced, "reference_sha256": archived, "exact_match": produced == archived}

    report = output / "s6_replay.json"
    report.write_text(json.dumps(outputs, indent=2) + "\n", encoding="utf-8")
    print(f"Supplementary Figure S6: {figures / COMPOSITE}")
    print(f"PNG comparison report: {report}")
    if args.verify and not all(result["exact_match"] for result in outputs.values()):
        parser.exit(1, "One or more PNGs differ from the archived references. Check fonts and dependency versions.\n")


if __name__ == "__main__":
    main()
