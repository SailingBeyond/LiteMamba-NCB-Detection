#!/usr/bin/env bash
set -euo pipefail
script_dir="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
exec "${CPLX_TRAIN_PYTHON:-python3}" -u "$script_dir/run_same_source.py" --phase prepare "$@"
