# Fifth-round public plotting scripts

This package contains the final public plotting workflows for Figure 3 and the main-text Figure 4.

## Final panel mapping

- **Figure 3A**: 16 × 64 motif-accuracy heatmap.
- **Figure 3B**: overall performance across three classification tasks and four models.
- **Figure 3C**: empirical cumulative distribution of per-motif accuracy.
- **Figure 3D**: LiteMamba latent-feature UMAP.
- **Figure 3E**: paired motif accuracy, ATCGXY 6-Class versus ATCGX 5-Class.
- **Figure 3F**: paired motif accuracy, ATCGXY 6-Class versus ATCGY 5-Class.
- **Figure 4A**: DNA01–16 false non-canonical call heatmap.
- **Figure 4B**: XNA01–16 target-recall lollipop plot.
- **Figure 4C**: XNA17–20 residual-error composition.

Figure 4D is intentionally not generated because it is not included in the final main-text design.

## Repository placement

Copy the four Python files into:

```text
LiteMamba-NCB-Detection/
└── scripts/
    └── 08_figures/
        ├── plot_figure3a_motif_heatmap.py
        ├── plot_figure3bcef.py
        ├── plot_figure3d_umap.py
        └── plot_figure4abc.py
```

Place this README temporarily in the repository root:

```text
LiteMamba-NCB-Detection/README_FIFTH_ROUND.md
```

## 1. Figure 3A motif heatmap

Input requirements:

```text
kmer,accuracy
AAANAA,0.987
...
```

Only motifs matching `XXXNYY` are used.

PowerShell example:

```powershell
python scripts/08_figures/plot_figure3a_motif_heatmap.py `
  --input_csv "C:\path\to\per_kmer_metrics_litemamba.csv" `
  --output_dir "results\figure3\figure3a"
```

Main outputs:

```text
Figure3A_motif_accuracy_heatmap_16x64.png/pdf/svg
Figure3A_motif_accuracy_matrix_16x64.csv
```

## 2. Figure 3B, 3C, 3E and 3F

The result root should contain:

```text
<result_root>/
└── combined_data/
    ├── figure3_all_tasks_overall_metrics_long.csv
    └── figure3_all_tasks_per_motif_metrics_long.csv
```

PowerShell example:

```powershell
python scripts/08_figures/plot_figure3bcef.py `
  --result_root "C:\path\to\figure3_multitask_4models_eval"
```

Default output directories:

```text
<result_root>/figures/
<result_root>/combined_data/figure3_plot_data/
```

The script generates:

- Figure 3B metric panels and a four-metric summary.
- Figure 3C motif-accuracy ECDF.
- Figure 3E ATCGXY 6-Class versus ATCGX 5-Class.
- Figure 3F ATCGXY 6-Class versus ATCGY 5-Class.

The paired scatter plots do not annotate motif names. Their colorbar label is `Accuracy difference`.

An optional combined Figure 3E/F panel can be generated with:

```powershell
--plot_combined_ef
```

## 3. Figure 3D UMAP

The script reuses the public model definition in:

```text
scripts/06_training/model_benchmark_all.py
```

Therefore it should remain in `scripts/08_figures/` inside the complete repository.

Install the additional dependency:

```powershell
pip install umap-learn
```

PowerShell example:

```powershell
python scripts/08_figures/plot_figure3d_umap.py `
  --csv_path "C:\path\to\xna_pc_6mer_features_all_6_test.csv" `
  --model_path "C:\path\to\six_class_litemamba.pt" `
  --output_dir "results\figure3\figure3d" `
  --device cuda:0 `
  --raw_signal_len 30 `
  --dwell_divisor 60 `
  --quality_divisor 50
```

The preprocessing parameters must match those used for model training.

Main outputs:

```text
Figure3D_LiteMamba_UMAP.png/pdf
Figure3D_LiteMamba_UMAP_embedding.csv
Figure3D_LiteMamba_UMAP_summary.csv
```

By default, at most 5,000 observations per class are retained through reservoir sampling. Set `--max_points_per_label 0` to use all observations, although this may require substantial memory and time.

## 4. Final Figure 4A–C

Input is the output root from:

```text
scripts/07_evaluation/evaluate_external_templates.py
```

Required files include:

```text
<evaluation_root>/analysis_data/control_template_metrics_long.csv
<evaluation_root>/analysis_data/target_template_metrics_long.csv
<evaluation_root>/raw_evaluation/.../eval_litemamba/predictions_*.csv[.gz]
```

Figure 4C requires per-sample prediction files. Therefore external evaluation must not be run with `--no_predictions` when Figure 4C needs to be regenerated.

PowerShell example:

```powershell
python scripts/08_figures/plot_figure4abc.py `
  --evaluation_root "C:\path\to\figure4_external_validation"
```

Default outputs:

```text
<evaluation_root>/figures_final/
<evaluation_root>/analysis_data/figure4abc_plot_data/
```

The script generates 6-class, 5X and 5Y versions of Figure 4A, 4B and 4C. To generate only one task:

```powershell
--tasks 6-class
```

## Dependencies

The scripts use:

```text
numpy
pandas
matplotlib
scikit-learn
torch
umap-learn
```

`umap-learn` is required only for Figure 3D.

## Validation performed during cleanup

The public scripts were checked with synthetic data for:

- Python syntax and command-line parsing.
- Complete 1,024-cell Figure 3A matrix construction.
- Figure 3B/C/E/F input-table normalization and image generation.
- Loading a six-class LiteMamba checkpoint using the public model definition.
- Feature extraction, per-class sampling and UMAP output interfaces.
- Figure 4A/B table derivation from external-validation metrics.
- Figure 4C residual-error derivation from uncompressed prediction CSV files.
- Generation of task-specific and combined plotting-data tables.

These tests validate interfaces and workflow logic; they do not recalculate or independently verify the manuscript results from the full experimental datasets.
