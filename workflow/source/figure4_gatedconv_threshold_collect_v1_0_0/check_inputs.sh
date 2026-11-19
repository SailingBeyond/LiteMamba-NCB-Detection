#!/usr/bin/env bash
set -euo pipefail
script_dir="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
exec "${FIGURE4_THRESHOLD_PYTHON:-python}" -u "$script_dir/collect_thresholds.py" --check_only "$@"
