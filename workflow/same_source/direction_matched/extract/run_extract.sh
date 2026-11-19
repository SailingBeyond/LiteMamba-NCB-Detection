#!/usr/bin/env bash
set -euo pipefail
script_dir="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
exec "${CPLX_EXTRACT_PYTHON:-python3}" -u "$script_dir/extract_direction_5class.py" --resume "$@"
