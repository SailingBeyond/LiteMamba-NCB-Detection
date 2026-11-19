# Supplementary Table S5 same-source controls

`six_class/` contains the six-class ATCGXY control. `direction_matched/` contains the strict `X_plus` and `Y_minus` five-class controls. Both use the CPLX-XNA-all sequencing batch and keep windows from a read in one split. The five-class workflows additionally match direction, BAM reference assignment and sequence context. The training coordinators first train the seed-42 pilot, then confirm with seeds 2024 and 2025, freeze checkpoints and evaluate the held-out test sets.

## Code checks

The following commands check the launchers without reading sequencing data or starting training. Run them from this directory.

```bash
python six_class/extract/extract_xna_6mer_features_all.py --help
python direction_matched/extract/extract_direction_5class.py --help
python six_class/train/run_same_source.py --dry_run
python direction_matched/train/run_direction_5class.py --dry_run
```

Install the dependencies in each subdirectory's `requirements.txt` before running that step. The direction-matched extractor checks for NumPy 1.23.5. The training scripts target the PyTorch 2.0.1 environment used for the archived runs.

## Run with the experiment inputs

This source bundle requires the original signal TSV and the fixed reference inputs. `SOURCE_MANIFEST.json` identifies the original archives and the reference inputs required for each workflow. Restore the references on the experiment server from the authorized original ZIP files with `python restore_reference_inputs.py "$CPLX_ARCHIVES"`. The installer checks the exact archive digests and preserves the portable engine identity record. Original signal files, feature tables, read IDs and reference inputs were omitted from this source bundle.

Set `CPLX_ARCHIVES` to the original archive directory, `CPLX_SIGNAL_TSV` to the original signal file and `CPLX_WORK_DIR` to a writable location outside this source directory. Set `CPLX_EXTRACT_PYTHON` to the Python executable with NumPy 1.23.5 and `CPLX_TRAIN_PYTHON` to the training environment's Python executable. Run from the same working directory throughout. The following commands preserve the distinct data and result directories for the two workflows.

```bash
set -eu
: "${CPLX_ARCHIVES:?Set the original archive directory}"
: "${CPLX_SIGNAL_TSV:?Set the original signal TSV path}"
: "${CPLX_WORK_DIR:?Set a work directory outside this source directory}"
: "${CPLX_EXTRACT_PYTHON:?Set the extraction Python executable}"
: "${CPLX_TRAIN_PYTHON:?Set the training Python executable}"

python restore_reference_inputs.py "$CPLX_ARCHIVES"

"$CPLX_EXTRACT_PYTHON" six_class/extract/extract_xna_6mer_features_all.py \
  --input "$CPLX_SIGNAL_TSV" --output_dir "$CPLX_WORK_DIR/six_class_input" \
  --phase all --with_header --resume
"$CPLX_EXTRACT_PYTHON" direction_matched/extract/extract_direction_5class.py \
  --input "$CPLX_SIGNAL_TSV" --output_dir "$CPLX_WORK_DIR/direction_matched_input" --resume

for phase in prepare pilot confirm freeze evaluate summary; do
  "$CPLX_TRAIN_PYTHON" six_class/train/run_same_source.py --phase "$phase" \
    --source_dir "$CPLX_WORK_DIR/six_class_input" --output_dir "$CPLX_WORK_DIR/six_class_results" --resume
  "$CPLX_TRAIN_PYTHON" direction_matched/train/run_direction_5class.py --phase "$phase" \
    --source_dir "$CPLX_WORK_DIR/direction_matched_input" --output_dir "$CPLX_WORK_DIR/direction_matched_results" --resume
done
```

Use a fresh result directory with the portable sources. Existing checkpoints and recovery records are tied to the original source and data fingerprints. Continue those runs from the corresponding original archives on the experiment server.

## Provenance and checks

`SOURCE_MANIFEST.json` records exact archive hashes and per-file original and portable SHA-256 values. The only source edits replace lab-machine default paths with relative paths. The portable engine records preserve the original configuration digests and pin the changed files. The model definitions, split rules, sampling logic, training schedule and evaluation code retain their archived behavior.

All four launchers passed help or dry-run checks. The reference installer and the pinned source checks passed against the original archives. The local NumPy 2.3.5 environment was correctly rejected by the strict extractor at its NumPy 1.23.5 version check. Full data extraction, training and checkpoint evaluation were not replayed while assembling this source bundle.
