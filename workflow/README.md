# Revision training and evaluation source

Run the commands from this directory. Every path beginning with `/path/to/` is a placeholder. Install the requirements file for the specific source package in an isolated environment. The FAST5 reader additionally uses h5py, pysam and samtools. The compute environment for model runs requires PyTorch and sufficient memory and disk. Files in `preprocessing/` and `source/` are source and configuration examples. The upstream package checksums and path substitutions are documented in `UPSTREAM_SHA256.txt` and `SITE_PATH_EDITS.txt`. Package manifests within `source/` document the original input archives and may list files or digests specific to those archives.

## Starting from feature tables

The six- and five-class preparation uses the same merged feature table and row-aligned metadata. In the commands below, set the inputs and output root to local paths before running them. The inputs comprise an aligned *E. coli* signal TSV, a motif-count CSV with counts for labels 4 and 5, and a positive feature CSV restricted to labels 4 and 5. The archived main multiclass splits use recorded sequencing read IDs for canonical windows and labelled source-row proxies for historical positives. The positive samples follow the design assumption of one source row per positive read. The audit checks partition disjointness for the recorded identities. Use `--positive_meta` when the input has corresponding read metadata and record that change as a new preparation. The canonical windows use minimum spacing 1. Split and sampling use seed 20260429 and the split ratio is 7:2:1.

```bash
set -eu
INPUT=/path/to/revision-inputs
OUT=/path/to/revision-results/preprocessing
mkdir -p "$OUT"

python preprocessing/extract_ecoli_6mer_features_spaced_by_read.py \
  --input "$INPUT/ecoli_aligned.tsv" \
  --positive_count_csv "$INPUT/positive_motif_counts.csv" \
  --output "$OUT/ecoli_6mer.csv" --meta_out "$OUT/ecoli_6mer_meta.tsv" \
  --min_window_spacing 1 --split_seed 20260429 --split_ratios 7,2,1

python preprocessing/combine_csv_header.py \
  --positive "$INPUT/xna_positive_6mer.csv" \
  --negative "$OUT/ecoli_6mer.csv" --negative_meta "$OUT/ecoli_6mer_meta.tsv" \
  --output "$OUT/merged_6class.csv" --meta_out "$OUT/merged_6class_meta.tsv" \
  --split_seed 20260429 --split_ratios 7,2,1

python preprocessing/balance_and_split_xna_pc_6class_by_read.py \
  --input "$OUT/merged_6class.csv" --meta "$OUT/merged_6class_meta.tsv" \
  --output_dir "$OUT/sixclass_overlap_read_split_721" \
  --split_seed 20260429 --sample_seed 20260429 --split_ratios 7,2,1

python preprocessing/balance_and_split_xna_pc_5class_by_read.py \
  --input "$OUT/merged_6class.csv" --meta "$OUT/merged_6class_meta.tsv" \
  --output_dir "$OUT/fiveclass_overlap_read_split_721" \
  --split_seed 20260429 --sample_seed 20260429 --split_ratios 7,2,1
```

Preprocessing also includes `extract_signal_from_fast5_XY.py` and the three XNA feature extractors. The upstream signal reader requires the corresponding annotated FAST5, reference FASTA and alignment BAM. The extractor `extract_xna_6mer_features_all.py` operates on the resulting TSV and accepts cohort-specific target and coverage arguments. The `extract_xna_pc_6mer_features_2_01_16.py` and `extract_xna_pc_6mer_features_2_13_16.py` scripts contain the two strand-specific extraction designs. Select extractor and arguments against the original cohort and alignment records. The feature-table commands above start from already staged inputs.

## Fresh fitting

The binary inputs for Fig. 2E,F are the two original strand-specific feature CSVs and their row-aligned metadata TSVs. The binary preparation splits by read. The GatedConv add-on fits three seeds for each strand. The base binary package also contains the comparator entry point `run_all_4_models.py` and the group-holdout preparation source.

```bash
set -eu
INPUT=/path/to/revision-inputs
OUT=/path/to/revision-results
mkdir -p "$OUT"

python preprocessing/balance_and_split_xna_pc_binary_by_read.py \
  --input_dir "$INPUT" --output_dir "$OUT/binary_read_split_721" \
  --features_pos "$INPUT/binary_pos.csv" --meta_pos "$INPUT/binary_pos_meta.tsv" \
  --features_neg "$INPUT/binary_neg.csv" --meta_neg "$INPUT/binary_neg_meta.tsv" \
  --seed 42

python source/figure2_binary_gatedconv_addon_v1_0_0/run_gatedconv_binary.py \
  --data_dir "$OUT/binary_read_split_721" --output_dir "$OUT/binary_gatedconv" \
  --seeds 42 43 44 --phase all
```

For the six-class task, the model identifier `litemamba` is retained from the training code. Select `legacy_gatedconv` and the `target_attention` readout for the manuscript GatedConv. The following command fits a fresh checkpoint per seed and defers test evaluation. The engine in `sixclass_training_auto_v1_0_4` contains the BiGRU, Transformer, ConvMixer and SSM comparator architecture. Its SSM is a different architecture from GatedConv and is used where specified in the comparison.

```bash
set -eu
OUT=/path/to/revision-results
for seed in 42 2024 2025; do
  python source/litemamba_structure_confirm3_v1_1_0/engine/train_all.py \
    --mode train --data_dir "$OUT/preprocessing/sixclass_overlap_read_split_721" \
    --output_dir "$OUT/sixclass_gatedconv_fresh/seed_$seed" \
    --model_type litemamba --litemamba_variant legacy_gatedconv \
    --readout target_attention --feature_scheme full --seed "$seed" \
    --monitor val_loss --batch_size 256 --epochs 200 --patience 30 \
    --lr 0.001 --defer_test
done
```

The five-class training launcher fits the four manuscript architectures on both the X and Y tasks at seeds 42, 2024 and 2025. Its source directory is the output of the five-class splitter containing `fiveclass_X/` and `fiveclass_Y/`.

```bash
set -eu
OUT=/path/to/revision-results
python source/figure3_fiveclass24_training_v1_0_4/run_fiveclass24.py \
  --source_dir "$OUT/preprocessing/fiveclass_overlap_read_split_721" \
  --output_dir "$OUT/fiveclass_training" --phase all
```

## Frozen evaluations and supplementary analyses

The normalized checkpoint attachment supplies model weights. Historical evaluation additionally requires the original frozen data, configuration and run-output records listed below. Use complete original run directories for those launchers. Fresh fitting with the commands above generates a new run.

| Analysis | Source entry point | Additional inputs |
| --- | --- | --- |
| Six-class structural diagnostic and confirmation | `source/litemamba_structure_diagnostic_v1_0_0/run_structure.py` and `source/litemamba_structure_confirm3_v1_1_0/run_structure.py` | The confirmation uses the original v1.0.4 base engine, completed seed-42 structure results and the prior `main_20/full/litemamba` SSM runs with validation outputs. |
| Final six-class internal test | `source/gatedconv_internal_test_v1_0_0/run_internal_test.py` | Staged original six-class split and row-aligned metadata, three original GatedConv run folders and the comparator results root. The run folders contain `best_checkpoint.pt`, `resolved_config.json`, `model_config.json`, `results.json`, `calibration.json` and accompanying original provenance and audit records. |
| External GatedConv versus SSM in Fig. 4 | `source/figure4_gated_comparison_v1_0_0/run_gated_comparison.py` | Complete corrected external run `figure4_external_xna13_16_corrected_v1_1_0`, its frozen plan, aligned feature/metadata inputs and existing SSM prediction exports. Also requires the original structural confirmation directory and linked baseline runs, including checkpoint, requested and resolved configurations, calibration, results and validation provenance for each relevant run. The example manifest documents the schema. |
| Internal ablation and single input | `source/gatedconv_ablation12_v1_0_0/run_ablation.py` and `source/gatedconv_singleinput9_v1_0_0/run_singleinput.py` | Six-class input split and the original completed full-input baseline run directories. |
| External single input in Table S4 | `source/figure4_singleinput9_v1_0_0/run_singleinput_external.py` | Corrected external run described above and completed internal single-input training directory. |
| External threshold counts | `source/figure4_gatedconv_threshold_collect_v1_0_0/collect_thresholds.py` | Existing external predictions and their corresponding validation outputs with original run provenance. |
| Same-source, direction-matched controls in Table S5 | See `same_source/`. | Source for both the direction-matched input preparation and six-class evaluation is provided separately. Follow that package's input contracts. |

Example invocation with the **original**, complete run directories and the corresponding data mounted. Substitute each path to match the original run layout. Some historical plans hold absolute paths or fingerprint source bytes. If those checks fail, restore the original mounts or obtain the designated upstream source and original runtime bundle before replay. Path anonymization in this distribution can change a source fingerprint. The launchers retain these checks. Archived plans, corresponding data inputs or completed run outputs absent from the distribution must be obtained separately.

```bash
set -eu
DATA=/path/to/revision-inputs
RUNS=/path/to/original-runs
NEW=/path/to/revision-results/replay
mkdir -p "$NEW"

python source/gatedconv_internal_test_v1_0_0/run_internal_test.py \
  --data_dir "$DATA/sixclass_overlap_read_split_721" \
  --results_dir "$RUNS/sixclass_results" \
  --seed42_dir "$RUNS/gatedconv_original/seed_42" \
  --seed2024_dir "$RUNS/gatedconv_original/seed_2024" \
  --seed2025_dir "$RUNS/gatedconv_original/seed_2025" \
  --output_dir "$NEW/internal_test"

python source/figure4_gated_comparison_v1_0_0/run_gated_comparison.py \
  --previous_output "$RUNS/figure4_external_xna13_16_corrected_v1_1_0" \
  --confirmation_dir "$RUNS/litemamba_structure_confirm3_v1_1_0" \
  --output_root "$NEW/figure4_gated_comparison"
```

The source for the corrected external comparison is present. Archived plans, read-level prediction exports, raw sequence inputs, row-aligned training manifests and private run logs are outside this source distribution. The publication figures and tables can be regenerated from the separate archived aggregate plotting inputs where supplied.

## Validation

Local source checks are listed in `SOURCE_CHECKS.txt`. Model training, frozen inference and full preprocessing runs require the staged datasets and runtime described above. Those end-to-end runs were not performed in the release environment.
