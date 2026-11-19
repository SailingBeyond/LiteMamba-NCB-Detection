# Supplementary Figure S7

Replot the saved five-class GatedConv UMAP coordinates. Panel A shows ATCGX and panel B shows ATCGY. Each panel uses seed 42 and 25,000 held-out test embeddings, with 5,000 points per class.

From this directory, install the dependencies and run the plotter.

```bash
python3 -m pip install -r requirements.txt
python3 plot.py
```

The combined figure is written to `outputs/figures/Supplementary_Figure_S7_GatedConv.png`. The directory also contains both individual panels in PNG, PDF and SVG formats. Specify `--output-dir PATH` to change the destination.

The archived coordinate arrays and the accompanying UMAP parameters and class counts are included. The plotting script reads the saved coordinates directly.
