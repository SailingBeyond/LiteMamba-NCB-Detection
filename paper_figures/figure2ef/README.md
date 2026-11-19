# Figure 2E–F

Replot the binary GatedConv receiver operating characteristic (E) and precision–recall (F) curves for the fixed X and Y test sets across seeds 42, 43 and 44. The curves use the saved uncalibrated probability analysis. The legends report the mean and sample standard deviation of AUROC and average precision across seeds.

From this directory, install the dependencies and run the plotter.

```bash
python3 -m pip install -r requirements.txt
python3 plot.py
```

The figure panels and validation tables are written to `outputs/`. Use `Figure2E_zoom` and `Figure2F_zoom` for the manuscript panels. The corresponding files without the `_zoom` suffix show the full plotting ranges. Each panel is saved as PNG, PDF and SVG. Specify `--output-dir PATH` to change the destination.

The input tables come from the archived binary GatedConv results and the full curve export. The curve CSV files are gzip-compressed without changing their contents. The JSON files retain the fields used by the plotting and provenance checks. In those compatibility fields, `litemamba` and `legacy_gatedconv` identify the GatedConv runs. This entry point renders the archived results and verifies cross-file consistency before plotting.
