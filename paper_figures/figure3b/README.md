# Figure 3B and Supplementary Table S1

This workflow regenerates the accuracy panel and recalculates the architecture benchmark in Supplementary Table S1 from frozen confusion counts. The input covers the held-out internal test sets of ATCGXY, ATCGX and ATCGY and the four architectures BiGRU, Transformer, ConvMixer and GatedConv. Each task–architecture combination has three runs (seeds 42, 2024 and 2025). Regeneration runs on a CPU and uses the bundled aggregate counts.

## Run

Use Python 3.12 and run the following from the repository root.

```bash
python3 -m pip install -r paper_figures/figure3b/requirements.txt
python3 paper_figures/figure3b/plot.py --output-dir reproduced/figure3b
```

| Output in `reproduced/figure3b/` | Contents |
| --- | --- |
| `Figure3B_accuracy.png`, `Figure3B_accuracy.pdf` | Fig. 3B. Small markers show individual runs. Larger markers and error bars show the mean and sample standard deviation across the three seeds. The vertical axis is accuracy on a 0–1 scale. |
| `Supplementary_Table_S1.csv` | Twelve task–architecture rows. Accuracy and macro precision, recall and F1 are percentages with two decimal places. Multiclass MCC is shown on its original scale with four decimal places. All values are reported as mean ± sample standard deviation. |
| `Figure3B_per_seed.csv` | Thirty-six run rows, with test-set size and unrounded metrics on their original scales. |

The script computes each run's metrics from its confusion counts. Macro metrics assign equal weight to the classes. A per-class precision, recall or F1 with a zero denominator contributes zero to its macro average. The workflow uses the multiclass Matthews correlation coefficient. Standard deviations use the sample denominator of two for the three seeds. The original plot style uses Arial when available and DejaVu Sans otherwise.

## Input

[`data/per_seed_confusion_counts.csv`](data/per_seed_confusion_counts.csv) stores one row per true-label and predicted-label pair for each run. The 1,032 rows include all cells of the 36 matrices, including zeros. Each matrix has 250,188, 1,225,160 or 1,194,720 test examples for ATCGXY, ATCGX or ATCGY, respectively. The published rows contain task, architecture, seed, class labels and integer counts.

Counts were extracted from the frozen confusion matrices used for the manuscript's Figure 3 benchmark (`figure3_updated_results_v1_0_4.zip`, SHA-256 `7d02cf919d6a4f5cfca78c45ba73efaf2439f24d42472360c059f5dc82660987`).

The bundled `per_seed_confusion_counts.csv` has SHA-256 `3904270b4e9de0aeca08a6cb5f39932583ddd56ffc95656cd63cf1c31364c47c`.
