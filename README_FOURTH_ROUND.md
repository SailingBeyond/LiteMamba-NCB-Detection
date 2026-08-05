# Fourth-round repository files: Figure 2 plotting

This package contains the cleaned Figure 2 plotting workflow for the
`LiteMamba-NCB-Detection` repository.

## Destination

Copy the two Python files into:

```text
LiteMamba-NCB-Detection/
└── scripts/
    └── 08_figures/
        ├── plot_figure2_feature_distributions.py
        └── plot_figure2_model_performance.py
```

The temporary documentation file may be placed in the repository root:

```text
LiteMamba-NCB-Detection/
└── README_FOURTH_ROUND.md
```

## 1. Feature-distribution panels

The original X- and Y-specific scripts were almost identical. They have been
merged into one parameterized script:

```text
scripts/08_figures/plot_figure2_feature_distributions.py
```

### X versus A

```bash
python scripts/08_figures/plot_figure2_feature_distributions.py \
  --input /path/to/xna_pc_6mer_features_pos.csv \
  --output_dir results/figure2/features_x \
  --target_motif ATAAGA \
  --natural_label 0 \
  --modified_label 1 \
  --natural_name A \
  --modified_name X
```

### Y versus C

```bash
python scripts/08_figures/plot_figure2_feature_distributions.py \
  --input /path/to/xna_pc_6mer_features_neg.csv \
  --output_dir results/figure2/features_y \
  --target_motif ATACGA \
  --natural_label 0 \
  --modified_label 1 \
  --natural_name C \
  --modified_name Y
```

By default, the script exports the compact target-site 1 x 8 panel and eight
six-position panels. Add `--compact_only` to generate only the compact panel.

## 2. Figure 2B, ROC/PR, and Figure 2F

Use:

```text
scripts/08_figures/plot_figure2_model_performance.py
```

The script is plotting-only: it does not load a model checkpoint or rerun
inference.

### Summary-only plotting

This generates Figure 2B overall-performance panels and Figure 2F from the
saved global summary table:

```bash
python scripts/08_figures/plot_figure2_model_performance.py \
  --global_summary_csv /path/to/global_summary.csv \
  --output_root results/figure2/model_performance
```

### Plotting with ROC/PR curves

The prediction CSV files must contain `y_true`, `prob_0`, and `prob_1`:

```bash
python scripts/08_figures/plot_figure2_model_performance.py \
  --global_summary_csv /path/to/global_summary.csv \
  --output_root results/figure2/model_performance \
  --pos_predictions_csv /path/to/x_predictions.csv \
  --neg_predictions_csv /path/to/y_predictions.csv
```

### Optional outputs

```bash
--plot_confusion_matrices
--plot_per_threshold_roc_pr
--plot_multi_threshold_roc_pr
```

Outputs can be disabled with:

```bash
--skip_figure2b_overall
--skip_figure2f
--skip_figure2b_roc_pr
```

## Scientific conventions retained

- Figure 2B contains Accuracy, Precision, Recall, F1-score, AUROC, AUPRC,
  Specificity, and MCC.
- Sensitivity is not plotted separately because it is numerically identical to
  recall for the positive class.
- ROC limits are x = 0.0-0.4 and y = 0.6-1.0.
- PR limits are x = 0.6-1.0 and y = 0.6-1.0.
- Figure 2F is split into separate X and Y panels.
- Figure 2F plots accuracy on the left axis and remaining rate on the right.
- PNG, PDF, and SVG outputs are retained for the model-performance script.
- Arial is requested, with DejaVu Sans as a fallback when Arial is unavailable.

## Dependencies

```text
matplotlib
numpy
pandas
scikit-learn
```

`plotnine` is no longer required by the cleaned Figure 2 feature script.
