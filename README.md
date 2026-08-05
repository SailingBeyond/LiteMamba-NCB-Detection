# LiteMamba-NCB-Detection

A reproducible analysis pipeline for detecting non-canonical bases in Oxford Nanopore sequencing data by combining resquiggled raw-current signals, basecalling/alignment-derived features, and neural-network classifiers.

The repository covers the workflow from FAST5 preprocessing to feature construction, dataset preparation, model training, held-out template evaluation, and reproduction of the main Figure 2–4 panels.

> **Terminology.** `X` and `Y` are internal symbols used in the scripts for the two non-canonical bases studied in the associated work. They are convenient computational labels rather than chemical names.

## Contents

- [Workflow](#workflow)
- [Repository layout](#repository-layout)
- [Requirements](#requirements)
- [Input and feature formats](#input-and-feature-formats)
- [1. Basecalling](#1-basecalling)
- [2. Tombo resquiggle](#2-tombo-resquiggle)
- [3. Alignment and BAM filtering](#3-alignment-and-bam-filtering)
- [4. Signal and 6-mer feature extraction](#4-signal-and-6-mer-feature-extraction)
- [5. Dataset preparation](#5-dataset-preparation)
- [6. Model training](#6-model-training)
- [7. Model evaluation](#7-model-evaluation)
- [8. Figure reproduction](#8-figure-reproduction)
- [Reproducibility notes](#reproducibility-notes)
- [Data and checkpoints](#data-and-checkpoints)

## Workflow

```text
multi-read FAST5
        │
        ▼
multi_to_single_fast5
        │
        ▼
single-read FAST5
        │
        ├──────────────► Guppy basecalling ──────────────► pass FASTQ
        │                                                     │
        │                                                     ├──► minimap2 + samtools
        │                                                     │         │
        │                                                     │         ▼
        └──► Tombo annotation and resquiggle ◄────────────────┘   filtered BAM
                              │                                      │
                              └──────────────────┬───────────────────┘
                                                 ▼
                                  per-reference-position signal TSV
                                                 │
                                                 ▼
                                      strand-aware 6-mer features
                                                 │
                                                 ▼
                                balanced train/test and held-out splits
                                                 │
                         ┌───────────────────────┴──────────────────────┐
                         ▼                                              ▼
              binary CB-vs-X / CB-vs-Y                    multiclass 5X / 5Y / 6-class
                         │                                              │
                         └───────────────────────┬──────────────────────┘
                                                 ▼
                                  evaluation and Figure 2–4 plotting
```

## Repository layout

```text
LiteMamba-NCB-Detection/
├── configs/
│   ├── preprocessing.example.env
│   └── external_model_manifest.example.json
├── scripts/
│   ├── run_preprocessing_pipeline.sh
│   ├── 01_basecalling/
│   │   └── run_guppy_basecalling.sh
│   ├── 02_resquiggle/
│   │   └── run_tombo_resquiggle.sh
│   ├── 03_alignment/
│   │   └── run_minimap2_samtools.sh
│   ├── 04_feature_extraction/
│   │   ├── extract_signal_from_fast5.py
│   │   ├── run_signal_feature_batches.sh
│   │   ├── merge_table_batches.py
│   │   ├── extract_xna_pc_6mer_features.py
│   │   └── extract_xna_6mer_features_all.py
│   ├── 05_dataset_preparation/
│   │   ├── balance_and_split_xna_pc_6mer_csv.py
│   │   └── split_features_by_reference_and_strand.py
│   ├── 06_training/
│   │   ├── model_benchmark_all.py
│   │   ├── train_all.py
│   │   └── run_all_4_models.py
│   ├── 07_evaluation/
│   │   ├── evaluate_binary_litemamba.py
│   │   └── evaluate_external_templates.py
│   └── 08_figures/
│       ├── plot_figure2_feature_distributions.py
│       ├── plot_figure2_model_performance.py
│       ├── plot_figure3a_motif_heatmap.py
│       ├── plot_figure3bcef.py
│       ├── plot_figure3d_umap.py
│       └── plot_figure4abc.py
├── .gitattributes
├── .gitignore
├── requirements.txt
└── README.md
```

## Requirements

Python 3.10 or newer is required because several scripts use modern built-in
generic and union type-hint syntax.

### External command-line software

The upstream preprocessing scripts record the tools used in the study:

- `multi_to_single_fast5` from ONT FAST5 tooling;
- Oxford Nanopore Guppy basecaller;
- Tombo;
- minimap2;
- samtools;
- Bash on Linux.

These tools are not installed through `requirements.txt`. Their executable paths and dataset-specific parameters are defined in `configs/preprocessing.env`.

### Python packages

Install a PyTorch build appropriate for the target CPU/GPU environment, then install the remaining dependencies:

```bash
pip install -r requirements.txt
```

PyTorch is intentionally not pinned in `requirements.txt`, because the correct
wheel depends on the local CPU/GPU and CUDA environment.

The core Python dependencies used by the repository are:

```text
numpy
pandas
matplotlib
scikit-learn
torch
h5py
tqdm
umap-learn
```

`umap-learn` is required only for Figure 3D.

## Input and feature formats

### Upstream signal TSV

`extract_signal_from_fast5.py` combines resquiggled FAST5 events with a filtered BAM and writes a 12-column tab-separated table:

```text
read_id
chrom
start
ref_seq
base_qualities
bam_strand
tombo_strand
strand_match
signal
mismatch
insertion
deletion
```

Per-base values are separated by `|`. Within each base, individual current measurements are separated by `*`.

### Model feature CSV

Feature extraction and all downstream model scripts use the following 11-column format:

```text
kmer,mean,std,median,dwell,quality,mismatch,insertion,deletion,signal,label
```

For a 6-mer:

- `mean`, `std`, `median`, `dwell`, `quality`, `mismatch`, `insertion`, and `deletion` each contain six `|`-separated values;
- `signal` contains six `|`-separated raw-signal segments;
- current measurements within one segment are separated by `*`;
- multiclass masked 6-mers use `N` at the target position.

## 1. Basecalling

Create a private, dataset-specific configuration from the public template:

```bash
cp configs/preprocessing.example.env configs/preprocessing.env
```

Edit only `configs/preprocessing.env`. This file is ignored by Git so that local data and software paths are not committed.

The template includes paths and parameters for:

- multi-read to single-read FAST5 conversion;
- Guppy model, device, caller count, and output directories;
- Tombo groups and process counts;
- minimap2 and samtools options;
- signal-extraction batches;
- target position and X/Y labels.

Check the expanded commands without running external software:

```bash
# Set DRY_RUN=1 in configs/preprocessing.env first.
bash scripts/run_preprocessing_pipeline.sh \
  configs/preprocessing.env all
```

Run the basecalling stage:

```bash
bash scripts/run_preprocessing_pipeline.sh \
  configs/preprocessing.env basecalling
```

This stage performs:

```text
multi-read FAST5
→ single-read FAST5
→ Guppy basecalling
→ merged pass FASTQ
```

The individual subcommands can also be run directly:

```bash
bash scripts/01_basecalling/run_guppy_basecalling.sh \
  configs/preprocessing.env convert

bash scripts/01_basecalling/run_guppy_basecalling.sh \
  configs/preprocessing.env basecall

bash scripts/01_basecalling/run_guppy_basecalling.sh \
  configs/preprocessing.env merge_fastq
```

## 2. Tombo resquiggle

Run FASTQ annotation and resquiggle:

```bash
bash scripts/run_preprocessing_pipeline.sh \
  configs/preprocessing.env resquiggle
```

Or invoke the stage script directly:

```bash
bash scripts/02_resquiggle/run_tombo_resquiggle.sh \
  configs/preprocessing.env all
```

The reference FASTA used for Tombo must be the same reference used for alignment and downstream signal extraction.

## 3. Alignment and BAM filtering

Run minimap2 alignment followed by samtools sorting, filtering, and indexing:

```bash
bash scripts/run_preprocessing_pipeline.sh \
  configs/preprocessing.env alignment
```

Or run the operations separately:

```bash
bash scripts/03_alignment/run_minimap2_samtools.sh \
  configs/preprocessing.env align

bash scripts/03_alignment/run_minimap2_samtools.sh \
  configs/preprocessing.env filter
```

The default BAM exclusion mask is `3844`, corresponding to unmapped, secondary, QC-fail, duplicate, and supplementary records.

The cleaned workflow pipes minimap2 output through samtools and does not retain unnecessary intermediate SAM and unsorted BAM files.

## 4. Signal and 6-mer feature extraction

### Complete batched workflow

Run signal-table generation and strand-aware 6-mer feature extraction:

```bash
bash scripts/run_preprocessing_pipeline.sh \
  configs/preprocessing.env features
```

`signal` remains an alias for `features` for compatibility:

```bash
bash scripts/run_preprocessing_pipeline.sh \
  configs/preprocessing.env signal
```

The batch runner:

1. discovers numeric FAST5 subdirectories or uses an explicit folder range;
2. processes the selected directories in configurable batches;
3. extracts a signal TSV for each batch;
4. creates the corresponding 6-mer feature CSV and metadata TSV;
5. resumes completed batches when `RESUME_BATCHES=1`;
6. merges feature and metadata batches with header validation.

Example batch configuration:

```bash
FOLDER_START="auto"
FOLDER_END="auto"
FOLDER_BATCH_SIZE=500
RESUME_BATCHES=1
MERGE_SIGNAL_TABLES=0
```

Signal TSV files can be very large. They are not merged by default; set `MERGE_SIGNAL_TABLES=1` only when a combined signal table is required and sufficient disk space is available.

### Signal extraction only

```bash
python scripts/04_feature_extraction/extract_signal_from_fast5.py \
  --fast5 /path/to/single_fast5 \
  --reference /path/to/reference.fasta \
  --bam /path/to/filtered.sorted.bam \
  --output /path/to/signal.tsv \
  --process 40 \
  --bam_threads 16 \
  --fast5_chunksize 64 \
  --with_header \
  --max_len_gap 10 \
  --strand_mismatch_output /path/to/strand_mismatch.tsv \
  --folder_start 0 \
  --folder_end 499
```

### PC/XNA feature extraction

`extract_xna_pc_6mer_features.py` filters target references and strand-consistent reads, extracts a strand-oriented target-centered 6-mer, and writes the common 11-column model format.

Default labels:

```text
PC                  0
XNA positive strand 1
XNA negative strand 2
```

Example:

```bash
python scripts/04_feature_extraction/extract_xna_pc_6mer_features.py \
  --input /path/to/upstream_signal.tsv \
  --output /path/to/xna_pc_6mer_features.csv \
  --meta_out /path/to/xna_pc_6mer_features_meta.tsv \
  --target_pos 1291 \
  --barcode_start 1240 \
  --barcode_end 1263 \
  --barcode_min_cover 15 \
  --max_ref_index 12 \
  --dwell_scale 1 \
  --quality_scale 1
```

The default `--max_ref_index 12` keeps templates 13–16 outside the default training extraction. Increase it only when that inclusion is intentional.

### X/Y six-class feature extraction

`extract_xna_6mer_features_all.py`:

- orients both strands to 5′→3′;
- places the target site at the fourth base;
- masks the target base with `N`;
- assigns X and Y labels;
- writes the common 11-column format and optional traceability metadata.

Default labels:

```text
X 4
Y 5
```

Example:

```bash
python scripts/04_feature_extraction/extract_xna_6mer_features_all.py \
  --input /path/to/upstream_signal.tsv \
  --output /path/to/xna_xy_features.csv \
  --meta_out /path/to/xna_xy_features_meta.tsv \
  --target_pos 1275 \
  --min_left_pos 1258 \
  --mask_base N \
  --label_x 4 \
  --label_y 5 \
  --dwell_scale 1 \
  --quality_scale 1
```

The extraction scripts write headers by default. Use `--no_header` only for a downstream program that explicitly requires headerless input.

## 5. Dataset preparation

### Balance and split model data

`balance_and_split_xna_pc_6mer_csv.py` balances each retained `kmer × label` group and creates deterministic train/test subsets.

Six-class example:

```bash
python scripts/05_dataset_preparation/balance_and_split_xna_pc_6mer_csv.py \
  --input /path/to/merged_6class_features.csv \
  --balanced_out /path/to/features_6class_balanced.csv \
  --train_out /path/to/features_6class_train.csv \
  --test_out /path/to/features_6class_test.csv \
  --summary_out /path/to/features_6class_split_summary.csv \
  --count_pivot_out /path/to/features_6class_count_pivot.csv \
  --num_labels 6 \
  --train_ratio 0.8 \
  --seed 12345
```

Five-class example excluding original label `4`:

```bash
python scripts/05_dataset_preparation/balance_and_split_xna_pc_6mer_csv.py \
  --input /path/to/merged_6class_features.csv \
  --balanced_out /path/to/features_5class_balanced.csv \
  --train_out /path/to/features_5class_train.csv \
  --test_out /path/to/features_5class_test.csv \
  --summary_out /path/to/features_5class_split_summary.csv \
  --labels 0,1,2,3,5 \
  --train_ratio 0.8 \
  --seed 12345
```

Selected original labels are remapped to contiguous model labels `0..N-1`.

### Split held-out templates by reference and strand

`split_features_by_reference_and_strand.py` reads a feature CSV and its row-aligned metadata TSV and generates external test files for:

```text
ATCGXY_6class
ATCGX_5class
ATCGY_5class
```

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

The feature CSV and metadata TSV must contain the same number of data rows in the same order. The script stops if either file contains unmatched rows.

## 6. Model training

The public benchmark contains four architectures:

```text
BiGRU
Transformer
ConvMixer
LiteMamba
```

### Train one model

Run the command from the repository root:

```bash
python scripts/06_training/train_all.py \
  --train_data /path/to/train.csv \
  --test_data /path/to/test.csv \
  --output_dir /path/to/results/litemamba \
  --model_type litemamba \
  --num_classes 6 \
  --seq_len 6 \
  --raw_signal_len 30 \
  --dwell_divisor 60 \
  --quality_divisor 50 \
  --batch_size 256 \
  --workers 8 \
  --epochs 10000 \
  --patience 50 \
  --seed 42
```

Use:

```text
--num_classes 2  for a binary task
--num_classes 5  for ATCGX or ATCGY
--num_classes 6  for ATCGXY
```

### Train all four models

```bash
python scripts/06_training/run_all_4_models.py \
  --train_data /path/to/train.csv \
  --test_data /path/to/test.csv \
  --output_dir /path/to/results/benchmark_6class \
  --num_classes 6 \
  --seq_len 6 \
  --raw_signal_len 30 \
  --dwell_divisor 60 \
  --quality_divisor 50 \
  --batch_size 256 \
  --workers 8 \
  --epochs 10000 \
  --patience 50 \
  --seed 42 \
  --stop_on_error
```

Each model receives its own output directory. The runner also creates `benchmark_summary.csv`.

### LiteMamba terminology

The implemented LiteMamba block is a dependency-free, Mamba-style gated mixing block using depthwise convolution. It is not the official `mamba_ssm` selective state-space implementation.

## 7. Model evaluation

### Binary CB-vs-X and CB-vs-Y evaluation

The binary checkpoints use a separate feature architecture from the masked-kmer multiclass models:

```text
binary models:    A/C/T/G one-hot + 8 scalar features = 12 features per base
multiclass models: A/C/T/G/N one-hot + 8 scalar features = 13 features per base
```

Binary checkpoints should therefore be evaluated with `evaluate_binary_litemamba.py`, not loaded into the multiclass model definition.

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
  --save_predictions \
  --no_plots
```

Remove `--no_plots` to let the evaluator generate its built-in plots. Keep `--save_predictions` when the plotting-only Figure 2 script needs row-level probabilities.

### Held-out external-template evaluation

Copy and edit the checkpoint manifest:

```bash
cp configs/external_model_manifest.example.json \
   configs/external_model_manifest.json
```

`configs/external_model_manifest.json` is ignored by Git so that local
checkpoint filenames and machine-specific model locations are not committed.

Example:

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

Main outputs:

```text
<out_root>/
├── raw_evaluation/
├── analysis_data/
│   ├── file_metrics_long.csv
│   ├── target_template_metrics_long.csv
│   ├── control_template_metrics_long.csv
│   ├── inference_efficiency_long.csv
│   └── additional source tables
└── run_manifest.json
```

Do not use `--no_predictions` when Figure 4C must be regenerated, because Figure 4C requires row-level prediction files.

## 8. Figure reproduction

### Figure 2 feature distributions

A single parameterized script replaces the original duplicated X- and Y-specific plotting scripts.

X versus A:

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

Y versus C:

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

Add `--compact_only` to export only the target-site 1 × 8 panel.

### Figure 2 model performance and confidence filtering

Summary-only plotting:

```bash
python scripts/08_figures/plot_figure2_model_performance.py \
  --global_summary_csv /path/to/global_summary.csv \
  --output_root results/figure2/model_performance
```

Add row-level probabilities to reproduce ROC/PR curves:

```bash
python scripts/08_figures/plot_figure2_model_performance.py \
  --global_summary_csv /path/to/global_summary.csv \
  --output_root results/figure2/model_performance \
  --pos_predictions_csv /path/to/x_predictions.csv \
  --neg_predictions_csv /path/to/y_predictions.csv
```

Optional outputs:

```text
--plot_confusion_matrices
--plot_per_threshold_roc_pr
--plot_multi_threshold_roc_pr
```

Figure 2 conventions retained by the public plotting script:

- Figure 2B reports Accuracy, Precision, Recall, F1-score, AUROC, AUPRC, Specificity, and MCC;
- Sensitivity is not plotted separately because it is identical to positive-class recall;
- ROC limits are x = 0.0–0.4 and y = 0.6–1.0;
- PR limits are x = 0.6–1.0 and y = 0.6–1.0;
- Figure 2F is split into separate X and Y panels;
- Figure 2F uses Accuracy on the left axis and Remaining Rate on the right.

### Final Figure 3 panel mapping

```text
Figure 3A  16 × 64 motif-accuracy heatmap
Figure 3B  overall performance across three tasks and four models
Figure 3C  empirical cumulative distribution of per-motif accuracy
Figure 3D  LiteMamba latent-feature UMAP
Figure 3E  paired motif accuracy: ATCGXY versus ATCGX
Figure 3F  paired motif accuracy: ATCGXY versus ATCGY
```

#### Figure 3A

```bash
python scripts/08_figures/plot_figure3a_motif_heatmap.py \
  --input_csv /path/to/per_kmer_metrics_litemamba.csv \
  --output_dir results/figure3/figure3a
```

The input must contain `kmer` and `accuracy`. Only motifs matching `XXXNYY` are used.

#### Figure 3B, 3C, 3E, and 3F

The result root must contain:

```text
<result_root>/combined_data/
├── figure3_all_tasks_overall_metrics_long.csv
└── figure3_all_tasks_per_motif_metrics_long.csv
```

Run:

```bash
python scripts/08_figures/plot_figure3bcef.py \
  --result_root /path/to/figure3_multitask_4models_eval
```

Add `--plot_combined_ef` to export an additional combined Figure 3E/F panel.

#### Figure 3D

```bash
python scripts/08_figures/plot_figure3d_umap.py \
  --csv_path /path/to/xna_pc_6mer_features_all_6_test.csv \
  --model_path /path/to/six_class_litemamba.pt \
  --output_dir results/figure3/figure3d \
  --device cuda:0 \
  --raw_signal_len 30 \
  --dwell_divisor 60 \
  --quality_divisor 50
```

The UMAP script reuses `scripts/06_training/model_benchmark_all.py`. Its preprocessing parameters must exactly match those used for training. By default, reservoir sampling retains at most 5,000 observations per class; use `--max_points_per_label 0` only when full-data UMAP is computationally feasible.

### Final Figure 4 panel mapping

```text
Figure 4A  DNA01–16 false non-canonical call heatmap
Figure 4B  XNA01–16 target-recall lollipop plot
Figure 4C  XNA17–20 residual-error composition
```

Figure 4D is intentionally not generated because it is not part of the final main-text design.

Run:

```bash
python scripts/08_figures/plot_figure4abc.py \
  --evaluation_root /path/to/figure4_external_validation
```

The script reads the output of `evaluate_external_templates.py` and writes:

```text
<evaluation_root>/figures_final/
<evaluation_root>/analysis_data/figure4abc_plot_data/
```

To plot one task only:

```bash
python scripts/08_figures/plot_figure4abc.py \
  --evaluation_root /path/to/figure4_external_validation \
  --tasks 6-class
```

## Reproducibility notes

- Use the same reference FASTA for Tombo resquiggle, minimap2 alignment, and signal extraction.
- Keep `raw_signal_len`, `dwell_divisor`, and `quality_divisor` identical between training, evaluation, and UMAP extraction.
- Keep the feature CSV and metadata TSV row-aligned when generating template-level splits.
- Binary and multiclass checkpoints use different per-base input dimensions and are not interchangeable.
- `configs/preprocessing.env` contains local paths and is intentionally ignored by Git.
- `configs/external_model_manifest.json` contains local checkpoint filenames and is intentionally ignored by Git.
- Raw FAST5/POD5, SAM/BAM, large feature tables, and model checkpoints are not intended to be committed to the source repository.
- Plotting scripts export vector formats where supported so that text and graphical elements remain editable.

## Data and checkpoints

This repository contains analysis code and configuration templates. Raw sequencing data, large intermediate files, and trained checkpoints are not bundled. Supply their locations through the command-line arguments and local configuration files described above.

## Citation

Citation information will be added after the associated manuscript is publicly available.
