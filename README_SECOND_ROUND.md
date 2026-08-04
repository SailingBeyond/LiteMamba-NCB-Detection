# Second-round repository files

This package contains the feature-extraction and dataset-preparation scripts
used before the model-training scripts in `scripts/06_training`.

## Destination layout

```text
LiteMamba-NCB-Detection/
└── scripts/
    ├── 04_feature_extraction/
    │   ├── extract_xna_pc_6mer_features.py
    │   └── extract_xna_6mer_features_all.py
    ├── 05_dataset_preparation/
    │   └── balance_and_split_xna_pc_6mer_csv.py
    └── 06_training/
        ├── model_benchmark_all.py
        ├── train_all.py
        └── run_all_4_models.py
```

## 1. `extract_xna_pc_6mer_features.py`

Purpose:

- reads the upstream 12-column signal TSV;
- filters target references and strand-consistent reads;
- extracts a strand-oriented 6-mer around the target site;
- calculates per-base signal statistics;
- writes the 11-column model feature CSV.

Default labels:

- PC: `0`;
- XNA positive strand: `1`;
- XNA negative strand: `2`.

The default `--max_ref_index 12` intentionally keeps templates 13–16 outside
the default training extraction. Use `--max_ref_index 16` only when that
inclusion is intentional.

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

## 2. `extract_xna_6mer_features_all.py`

Purpose:

- extracts the target-centered X/Y 6-mer;
- orients both strands to 5'->3';
- masks the fourth base with `N`;
- assigns X and Y labels;
- writes the same 11-column feature format.

Default labels:

- X: `4`;
- Y: `5`.

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

## 3. `balance_and_split_xna_pc_6mer_csv.py`

Purpose:

- balances every retained `kmer x label` group;
- creates exact training and testing counts within each group;
- remaps selected original labels to continuous labels `0..N-1`;
- supports a self-contained two-pass mode;
- optionally accepts a precomputed count pivot for compatibility.

Self-contained six-class example:

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

Five-class example excluding original label 4:

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

In the second example, original labels are remapped as follows:

```text
0 -> 0
1 -> 1
2 -> 2
3 -> 3
5 -> 4
```

## Output header behavior

The two extraction scripts now write headers by default. The balancing script
auto-detects the input header and writes headers by default. Use `--no_header`
only when a headerless downstream file is explicitly required.

## Validation performed

- all three scripts pass Python syntax compilation;
- the two extraction scripts reproduce the original CSV and metadata outputs
  exactly when run with the same parameters and a header;
- the balancing script was tested in both self-counting and precomputed-pivot
  modes;
- both balancing modes produced identical balanced, train, and test files;
- generated feature files were parsed successfully by the repository training
  pipeline.
