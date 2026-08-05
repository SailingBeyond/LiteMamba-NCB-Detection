#!/usr/bin/env bash
set -euo pipefail

usage() {
    cat >&2 <<USAGE
Usage: bash $0 <preprocessing.env> [all|align|filter]

Stages:
  all     Run minimap2 alignment, BAM sorting, filtering, and indexing (default).
  align   Generate coordinate-sorted BAM only.
  filter  Apply samtools -F filtering and index the filtered BAM only.
USAGE
}

[[ $# -ge 1 && $# -le 2 ]] || { usage; exit 2; }
CONFIG_FILE=$1
STAGE=${2:-all}
case "$STAGE" in all|align|filter) ;; *) usage; exit 2 ;; esac
[[ -f "$CONFIG_FILE" ]] || { echo "Config not found: $CONFIG_FILE" >&2; exit 1; }
# shellcheck disable=SC1090
source "$CONFIG_FILE"

: "${PASS_FASTQ:?Missing PASS_FASTQ}"
: "${REFERENCE_FASTA:?Missing REFERENCE_FASTA}"
: "${SORTED_BAM:?Missing SORTED_BAM}"
: "${FILTERED_BAM:?Missing FILTERED_BAM}"

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

run_align() {
    require_command "${MINIMAP2_BIN:-minimap2}"
    require_command "${SAMTOOLS_BIN:-samtools}"
    require_file "$PASS_FASTQ" "Merged pass FASTQ"
    require_file "$REFERENCE_FASTA" "Reference FASTA"
    mkdir -p "$(dirname "$SORTED_BAM")"

    printf '\n[align] Minimap2 alignment and coordinate-sorted BAM generation\n'
    if [[ "$DRY_RUN" == "1" ]]; then
        echo "  ${MINIMAP2_BIN:-minimap2} -t ${MINIMAP2_THREADS:-40} -ax ${MINIMAP2_PRESET:-map-ont} --secondary=${MINIMAP2_SECONDARY:-no} -w ${MINIMAP2_WINDOW:-5} '$REFERENCE_FASTA' '$PASS_FASTQ' | ${SAMTOOLS_BIN:-samtools} view -@ ${SAMTOOLS_THREADS:-56} -u - | ${SAMTOOLS_BIN:-samtools} sort -@ ${SAMTOOLS_THREADS:-56} -o '$SORTED_BAM' -"
    else
        "${MINIMAP2_BIN:-minimap2}" \
            -t "${MINIMAP2_THREADS:-40}" \
            -ax "${MINIMAP2_PRESET:-map-ont}" \
            --secondary="${MINIMAP2_SECONDARY:-no}" \
            -w "${MINIMAP2_WINDOW:-5}" \
            "$REFERENCE_FASTA" "$PASS_FASTQ" \
        | "${SAMTOOLS_BIN:-samtools}" view -@ "${SAMTOOLS_THREADS:-56}" -u - \
        | "${SAMTOOLS_BIN:-samtools}" sort -@ "${SAMTOOLS_THREADS:-56}" -o "$SORTED_BAM" -
    fi
    run_cmd "${SAMTOOLS_BIN:-samtools}" index "$SORTED_BAM"
}

run_filter() {
    require_command "${SAMTOOLS_BIN:-samtools}"
    require_file "$SORTED_BAM" "Sorted BAM"
    mkdir -p "$(dirname "$FILTERED_BAM")"

    printf '\n[filter] Filter BAM records and create index\n'
    run_cmd "${SAMTOOLS_BIN:-samtools}" view \
        -@ "${SAMTOOLS_THREADS:-56}" \
        -b \
        -F "${SAMTOOLS_FILTER_FLAGS:-3844}" \
        -o "$FILTERED_BAM" \
        "$SORTED_BAM"
    run_cmd "${SAMTOOLS_BIN:-samtools}" index "$FILTERED_BAM"
}

case "$STAGE" in
    all)
        run_align
        run_filter
        ;;
    align) run_align ;;
    filter) run_filter ;;
esac

printf '\nAlignment stage completed: %s\n' "$STAGE"
