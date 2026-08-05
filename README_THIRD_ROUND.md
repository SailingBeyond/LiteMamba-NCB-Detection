# Third-round repository files

This package adds template-level dataset splitting and model evaluation to the
repository. Copy the `scripts` and `configs` directories into the root of
`LiteMamba-NCB-Detection` and merge them with the existing directories.

## Target layout

```text
LiteMamba-NCB-Detection/
├── configs/
│   └── external_model_manifest.example.json
├── scripts/
│   ├── 05_dataset_preparation/
│   │   ├── balance_and_split_xna_pc_6mer_csv.py
│   │   └── split_features_by_reference_and_strand.py
│   ├── 06_training/
│   │   ├── model_benchmark_all.py
│   │   ├── train_all.py
│   │   └── run_all_4_models.py
│   └── 07_evaluation/
│       ├── evaluate_binary_litemamba.py
│       └── evaluate_external_templates.py
└── README_THIRD_ROUND.md
```

## 1. Split features by reference and strand

`scripts/05_dataset_preparation/split_features_by_reference_and_strand.py`
reads a feature CSV and its row-aligned metadata TSV, then creates test files
for three tasks:

- `ATCGXY_6class`
- `ATCGX_5class`
- `ATCGY_5class`

Each task is further split by `chrom` and `bam_strand`. The fourth base of the
output 6-mer is masked as `N`, and labels are remapped for the corresponding
multiclass model.

Example:

```bash
python scripts/05_dataset_preparation/split_features_by_reference_and_strand.py \
  --input /path/to/xna_pc_6mer_features_all.csv \
  --meta /path/to/xna_pc_6mer_features_meta_all.tsv \
  --outroot /path/to/external_test_splits \
  --header always \
  --label_check error \
  --xna_pos_is X
```

The feature CSV and metadata TSV must contain the same number of data rows in
the same order. The script aborts if one file has extra rows.

## 2. Evaluate the two binary LiteMamba models

`scripts/07_evaluation/evaluate_binary_litemamba.py` evaluates the CB-vs-X and
CB-vs-Y binary checkpoints, including confidence-threshold analysis.

Important architecture note: these binary checkpoints use A/C/T/G four-base
one-hot encoding plus eight scalar features (12 features per base). This is a
separate checkpoint architecture from the masked-kmer multiclass models in
`model_benchmark_all.py`, which use A/C/T/G/N one-hot encoding plus eight
scalar features (13 features per base).

Metrics-only example:

```bash
python scripts/07_evaluation/evaluate_binary_litemamba.py \
  --pos_csv /path/to/x_binary_test.csv \
  --neg_csv /path/to/y_binary_test.csv \
  --pos_model /path/to/x_litemamba.pt \
  --neg_model /path/to/y_litemamba.pt \
  --output_root /path/to/figure2_evaluation \
  --device cuda:0 \
  --batch_size 1024 \
  --workers 8 \
  --raw_signal_len 30 \
  --dwell_divisor 60 \
  --quality_divisor 50 \
  --thresholds 0.5 0.6 0.7 0.8 0.9 \
  --no_plots
```

Remove `--no_plots` to additionally generate confusion matrices, ROC/PR curves,
and the current Figure 2B/2F outputs. Add `--save_predictions` when row-level
probability output is needed.

## 3. Evaluate held-out external templates

`scripts/07_evaluation/evaluate_external_templates.py` evaluates XNA1-20 and
available DNA/PC controls for the 6-class, 5X, and 5Y tasks. It uses the four
model definitions in `scripts/06_training/model_benchmark_all.py` and exports:

```text
output_root/
├── raw_evaluation/
├── analysis_data/
│   ├── file_metrics_long.csv
│   ├── target_template_metrics_long.csv
│   ├── control_template_metrics_long.csv
│   ├── inference_efficiency_long.csv
│   └── figure4*.csv
└── run_manifest.json
```

The third-round public version intentionally does not contain the old Figure 4
radar/bubble plotting implementation. It only exports the validated source
tables. The final manuscript plotting script should be placed later under
`scripts/08_figures/`.

Copy the example manifest and replace each filename with the actual checkpoint
filename:

```bash
cp configs/external_model_manifest.example.json \
   configs/external_model_manifest.json
```

Full evaluation example:

```bash
python scripts/07_evaluation/evaluate_external_templates.py \
  --out_root /path/to/figure4_external_validation \
  --model_dir /path/to/checkpoints \
  --model_manifest configs/external_model_manifest.json \
  --dir_01_12_root /path/to/xna01_12_task_folders \
  --dir_13_16_6 /path/to/xna13_16_6class \
  --dir_13_16_5x /path/to/xna13_16_5X \
  --dir_13_16_5y /path/to/xna13_16_5Y \
  --dir_17_20_6 /path/to/xna17_20_6class \
  --dir_17_20_5x /path/to/xna17_20_5X \
  --dir_17_20_5y /path/to/xna17_20_5Y \
  --models litemamba transformer bigru convmixer \
  --device cuda:0 \
  --batch_size 256 \
  --workers 4 \
  --raw_signal_len 30 \
  --dwell_divisor 60 \
  --quality_divisor 50
```

Use `--no_predictions` to skip row-level prediction files. This substantially
reduces storage use when only aggregate metrics are needed.

## Validation performed

- All three Python files pass `py_compile`.
- The split script was compared against the uploaded original implementation;
  all generated task CSV files were byte-identical with the same parameters.
- The binary evaluator completed end-to-end inference and metric export on
  synthetic 11-column CSV files and compatible synthetic checkpoints.
- The external-template evaluator completed all nine cohort/task combinations
  on synthetic test directories and exported the complete analysis-data set.
- No server-specific `/harddisk...` or Windows absolute paths remain in the
  public files.
