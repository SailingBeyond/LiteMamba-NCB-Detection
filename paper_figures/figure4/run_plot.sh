#!/usr/bin/env bash
set -eu
script_dir="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
exec "${PYTHON:-python3}" "$script_dir/plot_figure4.py" "$@"
