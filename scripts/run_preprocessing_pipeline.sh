#!/usr/bin/env bash
set -euo pipefail

usage() {
    cat >&2 <<USAGE
Usage: bash $0 <preprocessing.env> [all|basecalling|alignment|signal]

  all          Run all three stages in order (default).
  basecalling  Convert FAST5, run Guppy, and merge pass FASTQ.
  alignment    Tombo annotation/resquiggle and minimap2/samtools processing.
  signal       Batched signal-table and 6-mer feature extraction.
USAGE
}

[[ $# -ge 1 && $# -le 2 ]] || { usage; exit 2; }
CONFIG_FILE=$1
STAGE=${2:-all}
case "$STAGE" in all|basecalling|alignment|signal) ;; *) usage; exit 2 ;; esac
[[ -f "$CONFIG_FILE" ]] || { echo "Config not found: $CONFIG_FILE" >&2; exit 1; }

SCRIPT_DIR=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)
BASECALL_SCRIPT="$SCRIPT_DIR/01_basecalling/run_guppy_basecalling.sh"
ALIGNMENT_SCRIPT="$SCRIPT_DIR/02_alignment_resquiggle/run_alignment_and_resquiggle.sh"
SIGNAL_SCRIPT="$SCRIPT_DIR/03_signal_table_generation/run_signal_feature_batches.sh"

run_basecalling() {
    bash "$BASECALL_SCRIPT" "$CONFIG_FILE" all
}

run_alignment() {
    bash "$ALIGNMENT_SCRIPT" "$CONFIG_FILE" all
}

run_signal() {
    bash "$SIGNAL_SCRIPT" "$CONFIG_FILE"
}

case "$STAGE" in
    all)
        run_basecalling
        run_alignment
        run_signal
        ;;
    basecalling) run_basecalling ;;
    alignment) run_alignment ;;
    signal) run_signal ;;
esac
