# Supplementary Figure S6

Panel A shows the ATCGX classifier and panel B shows the ATCGY classifier. The heatmaps show the mean of seed-specific test accuracies for seeds 42, 2024 and 2025 across all 1,024 masked 6-mer contexts. The two panels share the 0.75–1.00 color scale. The test sets comprise 1,225,160 windows for ATCGX and 1,194,720 windows for ATCGY per seed.

## Reproduce the figure

Run from this directory with Python 3.12.

```bash
python -m pip install -r requirements.txt
python reproduce_s6.py --output-dir results
```

The script verifies the packaged count export, rebuilds both heatmaps and assembles `results/figures/Supplementary_Figure_S6_GatedConv.png`. It writes motif counts under `results/data/`, heatmap matrices and the split audit under `results/plot_data/`, and panel PDF and SVG files under `results/figures/`. The script reads frozen test counts and requires no model checkpoint or raw feature file.

The `reference/` directory contains the two archived full-resolution panel images and the exact composite embedded in the revised supplement. `results/s6_replay.json` records the SHA-256 comparison against those files. To require exact PNG matches, add `--verify`. Byte-identical reproduction was verified with the listed Python dependencies and the system DejaVu Sans Bold font at `/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf`. On another system, provide the same label font with `--label-font` if available. Fonts and the rendering environment may affect PNG hashes.

## Source

The plot uses the unchanged test counts from `figure3_updated_results_v1_0_4.zip`. The three-task frozen data determine the common lower color bound. The portable snapshot retains the count fields, task labels, seeds and GatedConv input hashes used for verification. The companion manifest records the original and portable snapshot hashes. Training configurations, lab paths and machine-specific timestamps were removed.

The two reference heatmaps are the archive previews that were combined to create the revised Supplementary Figure S6. The revised supplement embeds the composite with SHA-256 `4c75c424eda9ebb8117bdd520baac478dc59827fbeb2e61557495a869d8faa95`.
