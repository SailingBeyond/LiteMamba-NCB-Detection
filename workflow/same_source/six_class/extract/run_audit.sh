#!/usr/bin/env bash
set -euo pipefail
script_dir="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
cplx_input="${CPLX6_INPUT_TSV:-fast5_single_guppy_pass_signal_XY_all_notxt_fast.tsv}"
cplx_output="${CPLX6_OUTPUT_DIR:-cplx_all_same_source_6class_v1_0_0}"
exec "${CPLX6_PYTHON:-python3}" -u "$script_dir/extract_xna_6mer_features_all.py" \
  --input "$cplx_input" --output_dir "$cplx_output" \
  --batch_id CPLX-XNA-all --phase audit --with_header "$@"
