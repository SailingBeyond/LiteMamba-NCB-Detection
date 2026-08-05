#!/usr/bin/env bash
set -euo pipefail

usage() {
    cat >&2 <<USAGE
Usage: bash $0 <preprocessing.env> [all|annotate|resquiggle]

Stages:
  all         Run Tombo FASTQ annotation and resquiggle (default).
  annotate    Run annotate_raw_with_fastqs only.
  resquiggle  Run Tombo resquiggle only.
USAGE
}

[[ $# -ge 1 && $# -le 2 ]] || { usage; exit 2; }
CONFIG_FILE=$1
STAGE=${2:-all}
case "$STAGE" in all|annotate|resquiggle) ;; *) usage; exit 2 ;; esac
[[ -f "$CONFIG_FILE" ]] || { echo "Config not found: $CONFIG_FILE" >&2; exit 1; }
# shellcheck disable=SC1090
source "$CONFIG_FILE"

: "${SINGLE_FAST5_DIR:?Missing SINGLE_FAST5_DIR}"
: "${PASS_FASTQ:?Missing PASS_FASTQ}"
: "${SEQUENCING_SUMMARY:?Missing SEQUENCING_SUMMARY}"
: "${REFERENCE_FASTA:?Missing REFERENCE_FASTA}"

DRY_RUN=${DRY_RUN:-0}

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
    [[ "$DRY_RUN" == "1" ]] && return
    if [[ "$command_name" == */* ]]; then
        [[ -x "$command_name" ]] || { echo "Executable not found: $command_name" >&2; exit 1; }
    else
        command -v "$command_name" >/dev/null 2>&1 || { echo "Command not found: $command_name" >&2; exit 1; }
    fi
}

require_file() {
    local path=$1
    local label=$2
    if [[ "$DRY_RUN" != "1" && ! -f "$path" ]]; then
        echo "$label not found: $path" >&2
        exit 1
    fi
}

run_annotate() {
    require_command "${TOMBO_BIN:-tombo}"
    [[ "$DRY_RUN" == "1" || -d "$SINGLE_FAST5_DIR" ]] || {
        echo "Single-read FAST5 directory not found: $SINGLE_FAST5_DIR" >&2
        exit 1
    }
    require_file "$PASS_FASTQ" "Merged pass FASTQ"
    require_file "$SEQUENCING_SUMMARY" "Sequencing summary"

    printf '\n[annotate] Annotate FAST5 files with Guppy FASTQ calls\n'
    run_cmd "${TOMBO_BIN:-tombo}" preprocess annotate_raw_with_fastqs \
        --fast5-basedir "$SINGLE_FAST5_DIR" \
        --fastq-filenames "$PASS_FASTQ" \
        --processes "${TOMBO_ANNOTATE_PROCESSES:-16}" \
        --sequencing-summary-filenames "$SEQUENCING_SUMMARY" \
        --overwrite
}

run_resquiggle() {
    require_command "${TOMBO_BIN:-tombo}"
    [[ "$DRY_RUN" == "1" || -d "$SINGLE_FAST5_DIR" ]] || {
        echo "Single-read FAST5 directory not found: $SINGLE_FAST5_DIR" >&2
        exit 1
    }
    require_file "$REFERENCE_FASTA" "Reference FASTA"

    printf '\n[resquiggle] Tombo resquiggle\n'
    local -a tombo_args=(
        resquiggle
        "$SINGLE_FAST5_DIR"
        "$REFERENCE_FASTA"
        --processes "${TOMBO_RESQUIGGLE_PROCESSES:-10}"
        --corrected-group "${TOMBO_CORRECTED_GROUP:-RawGenomeCorrected_000}"
        --basecall-group "${TOMBO_BASECALL_GROUP:-Basecall_1D_000}"
        --overwrite
        --num-most-common-errors "${TOMBO_NUM_MOST_COMMON_ERRORS:-5}"
    )
    [[ "${TOMBO_FIT_GLOBAL_SCALE:-1}" == "1" ]] && tombo_args+=(--fit-global-scale)
    [[ "${TOMBO_INCLUDE_EVENT_STDEV:-1}" == "1" ]] && tombo_args+=(--include-event-stdev)
    run_cmd "${TOMBO_BIN:-tombo}" "${tombo_args[@]}"
}

case "$STAGE" in
    all)
        run_annotate
        run_resquiggle
        ;;
    annotate) run_annotate ;;
    resquiggle) run_resquiggle ;;
esac

printf '\nTombo stage completed: %s\n' "$STAGE"
