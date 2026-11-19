#!/usr/bin/env bash
set -euo pipefail
script_dir="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
exec "${FIVECLASS24_PYTHON:-python3}" -u "$script_dir/run_fiveclass24.py" --phase freeze "$@"
