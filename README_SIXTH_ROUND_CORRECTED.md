# Sixth-round upstream preprocessing workflow — corrected directory layout

This corrected package preserves the repository's original eight-stage structure:

```text
scripts/
├── 01_basecalling/
├── 02_resquiggle/
├── 03_alignment/
├── 04_feature_extraction/
├── 05_dataset_preparation/
├── 06_training/
├── 07_evaluation/
└── 08_figures/
```

## File placement

```text
scripts/
├── run_preprocessing_pipeline.sh
├── 01_basecalling/
│   └── run_guppy_basecalling.sh
├── 02_resquiggle/
│   └── run_tombo_resquiggle.sh
├── 03_alignment/
│   └── run_minimap2_samtools.sh
└── 04_feature_extraction/
    ├── extract_signal_from_fast5.py
    ├── run_signal_feature_batches.sh
    ├── merge_table_batches.py
    ├── extract_xna_pc_6mer_features.py
    └── extract_xna_6mer_features_all.py
```

The final two feature extractors were added in the second round and are not duplicated in this package.

## Why signal-table generation belongs to 04_feature_extraction

`extract_signal_from_fast5.py` converts resquiggled FAST5 plus filtered BAM into the per-reference-position signal table consumed by the 6-mer feature extractor. It is therefore part of feature construction rather than a new top-level experimental stage.

## Main commands

```bash
cp configs/preprocessing.example.env configs/preprocessing.env
```

```bash
bash scripts/run_preprocessing_pipeline.sh configs/preprocessing.env basecalling
bash scripts/run_preprocessing_pipeline.sh configs/preprocessing.env resquiggle
bash scripts/run_preprocessing_pipeline.sh configs/preprocessing.env alignment
bash scripts/run_preprocessing_pipeline.sh configs/preprocessing.env features
```

Or run all stages:

```bash
bash scripts/run_preprocessing_pipeline.sh configs/preprocessing.env all
```

`signal` remains an alias of `features` for compatibility with the earlier draft package.
