# Figure 3D–F

Panel D shows the saved UMAP coordinates for 30,000 six-class GatedConv test embeddings from seed 42, with 5,000 points per class. Panels E and F compare per-motif accuracy between the six-class task and the X and Y five-class tasks, respectively. The comparisons use the same audited test windows within each task pair and average seeds 42, 2024 and 2025 with equal weight.

From this directory, install the dependencies and run the plotter.

```bash
python3 -m pip install -r requirements.txt
python3 plot.py
```

The figures are written to `outputs/figures/` in PNG, PDF and SVG formats. The paired means, standard deviations and comparison summaries are written to `outputs/plot_data/`. Specify `--output-dir PATH` to change the destination.

The original coordinate archive and paired motif table are included. Their accompanying JSON records retain the parameters, sample counts and provenance used to verify these plotting inputs. The plotting script reads saved coordinates directly.
