#!/usr/bin/env bash
set -euo pipefail

usage() {
    cat >&2 <<USAGE
Usage: bash $0 <preprocessing.env> [all|convert|basecall|merge_fastq]

Stages:
  all          Convert FAST5, run Guppy, and merge pass FASTQ files (default).
  convert      Run multi_to_single_fast5 only.
  basecall     Run Guppy only.
  merge_fastq  Merge Guppy pass/*.fastq only.
USAGE
}

[[ $# -ge 1 && $# -le 2 ]] || { usage; exit 2; }
CONFIG_FILE=$1
STAGE=${2:-all}
case "$STAGE" in all|convert|basecall|merge_fastq) ;; *) usage; exit 2 ;; esac
[[ -f "$CONFIG_FILE" ]] || { echo "Config not found: $CONFIG_FILE" >&2; exit 1; }
# shellcheck disable=SC1090
source "$CONFIG_FILE"

: "${MULTI_FAST5_DIR:?Missing MULTI_FAST5_DIR}"
: "${SINGLE_FAST5_DIR:?Missing SINGLE_FAST5_DIR}"
: "${BASECALL_DIR:?Missing BASECALL_DIR}"
: "${PASS_FASTQ:?Missing PASS_FASTQ}"
: "${MULTI_TO_SINGLE_BIN:?Missing MULTI_TO_SINGLE_BIN}"
: "${GUPPY_BIN:?Missing GUPPY_BIN}"

DRY_RUN=${DRY_RUN:-0}
ALLOW_EXISTING_OUTPUTS=${ALLOW_EXISTING_OUTPUTS:-0}

print_cmd() {
    printf '  '
    printf '%q ' "$@"
    printf '\n'
}

run_cmd() {
    print_cmd "$@"
    if [[ "$DRY_RUN" != "1" ]]; then
        "$@"
    fi
}

require_command() {
    local command_name=$1
    if [[ "$DRY_RUN" == "1" ]]; then
        return
    fi
    if [[ "$command_name" == */* ]]; then
        [[ -x "$command_name" ]] || { echo "Executable not found: $command_name" >&2; exit 1; }
    else
        command -v "$command_name" >/dev/null 2>&1 || { echo "Command not found: $command_name" >&2; exit 1; }
    fi
}

require_empty_or_allowed() {
    local path=$1
    if [[ -d "$path" ]] && find "$path" -mindepth 1 -print -quit | grep -q .; then
        if [[ "$ALLOW_EXISTING_OUTPUTS" != "1" ]]; then
            echo "Output directory is not empty: $path" >&2
            echo "Use a stage-specific rerun or set ALLOW_EXISTING_OUTPUTS=1 intentionally." >&2
            exit 1
        fi
    fi
}

run_convert() {
    require_command "$MULTI_TO_SINGLE_BIN"
    [[ -d "$MULTI_FAST5_DIR" ]] || { echo "Input FAST5 directory not found: $MULTI_FAST5_DIR" >&2; exit 1; }
    require_empty_or_allowed "$SINGLE_FAST5_DIR"
    mkdir -p "$SINGLE_FAST5_DIR"

    printf '\n[convert] Convert multi-read FAST5 to single-read FAST5\n'
    run_cmd "$MULTI_TO_SINGLE_BIN" \
        -i "$MULTI_FAST5_DIR" \
        -s "$SINGLE_FAST5_DIR" \
        -t "${MULTI_TO_SINGLE_THREADS:-40}" \
        --recursive
}

run_basecall() {
    require_command "$GUPPY_BIN"
    [[ -d "$SINGLE_FAST5_DIR" ]] || { echo "Single-read FAST5 directory not found: $SINGLE_FAST5_DIR" >&2; exit 1; }
    require_empty_or_allowed "$BASECALL_DIR"
    mkdir -p "$BASECALL_DIR"

    printf '\n[basecall] Guppy basecalling\n'
    read -r -a guppy_extra <<< "${GUPPY_EXTRA_ARGS:-}"
    run_cmd "$GUPPY_BIN" \
        -i "$SINGLE_FAST5_DIR" \
        -s "$BASECALL_DIR" \
        --config "${GUPPY_CONFIG:-dna_r9.4.1_450bps_hac.cfg}" \
        --device "${GUPPY_DEVICE:-cuda:all}" \
        --num_callers "${GUPPY_NUM_CALLERS:-40}" \
        --recursive \
        "${guppy_extra[@]}"
}

run_merge_fastq() {
    printf '\n[merge_fastq] Concatenate Guppy pass FASTQ files\n'
    if [[ "$DRY_RUN" == "1" ]]; then
        echo "  find '$BASECALL_DIR/pass' -type f -name '*.fastq' | natural-sort | concatenate > '$PASS_FASTQ'"
        return
    fi

    local pass_dir="$BASECALL_DIR/pass"
    [[ -d "$pass_dir" ]] || { echo "Guppy pass directory not found: $pass_dir" >&2; exit 1; }

    local -a fastq_files
    mapfile -d '' fastq_files < <(
        find "$pass_dir" -type f -name '*.fastq' -print0 | sort -z
    )
    (( ${#fastq_files[@]} > 0 )) || { echo "No .fastq files found under: $pass_dir" >&2; exit 1; }

    mkdir -p "$(dirname "$PASS_FASTQ")"
    : > "$PASS_FASTQ"
    for fastq_file in "${fastq_files[@]}"; do
        cat "$fastq_file" >> "$PASS_FASTQ"
    done
    echo "  Merged ${#fastq_files[@]} FASTQ files -> $PASS_FASTQ"
}

case "$STAGE" in
    all)
        run_convert
        run_basecall
        run_merge_fastq
        ;;
    convert) run_convert ;;
    basecall) run_basecall ;;
    merge_fastq) run_merge_fastq ;;
esac

printf '\nBasecalling stage completed: %s\n' "$STAGE"
