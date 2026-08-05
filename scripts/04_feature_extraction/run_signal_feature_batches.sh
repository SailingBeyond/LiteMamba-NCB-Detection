#!/usr/bin/env bash
set -euo pipefail

usage() {
    echo "Usage: bash $0 <preprocessing.env>" >&2
}

[[ $# -eq 1 ]] || { usage; exit 2; }
CONFIG_FILE=$1
[[ -f "$CONFIG_FILE" ]] || { echo "Config not found: $CONFIG_FILE" >&2; exit 1; }
# shellcheck disable=SC1090
source "$CONFIG_FILE"

SCRIPT_DIR=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)
REPO_ROOT=$(cd "$SCRIPT_DIR/../.." && pwd)
SIGNAL_SCRIPT=${SIGNAL_SCRIPT:-"$SCRIPT_DIR/extract_signal_from_fast5.py"}
FEATURE_SCRIPT=${FEATURE_SCRIPT:-"$REPO_ROOT/scripts/04_feature_extraction/extract_xna_6mer_features_all.py"}
MERGE_SCRIPT=${MERGE_SCRIPT:-"$SCRIPT_DIR/merge_table_batches.py"}

: "${PYTHON_BIN:=python3}"
: "${SINGLE_FAST5_DIR:?Missing SINGLE_FAST5_DIR}"
: "${REFERENCE_FASTA:?Missing REFERENCE_FASTA}"
: "${FILTERED_BAM:?Missing FILTERED_BAM}"
: "${BATCH_OUTPUT_DIR:?Missing BATCH_OUTPUT_DIR}"
: "${FINAL_OUTPUT_DIR:?Missing FINAL_OUTPUT_DIR}"

DRY_RUN=${DRY_RUN:-0}
RESUME_BATCHES=${RESUME_BATCHES:-1}
mkdir -p "$BATCH_OUTPUT_DIR" "$FINAL_OUTPUT_DIR"

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

if [[ "$DRY_RUN" != "1" ]]; then
    command -v "$PYTHON_BIN" >/dev/null 2>&1 || { echo "Python not found: $PYTHON_BIN" >&2; exit 1; }
    [[ -f "$SIGNAL_SCRIPT" ]] || { echo "Signal extractor not found: $SIGNAL_SCRIPT" >&2; exit 1; }
    [[ -f "$FEATURE_SCRIPT" ]] || { echo "Feature extractor not found: $FEATURE_SCRIPT" >&2; exit 1; }
    [[ -f "$MERGE_SCRIPT" ]] || { echo "Merge script not found: $MERGE_SCRIPT" >&2; exit 1; }
    [[ -d "$SINGLE_FAST5_DIR" ]] || { echo "FAST5 directory not found: $SINGLE_FAST5_DIR" >&2; exit 1; }
    [[ -f "$REFERENCE_FASTA" ]] || { echo "Reference not found: $REFERENCE_FASTA" >&2; exit 1; }
    [[ -f "$FILTERED_BAM" ]] || { echo "Filtered BAM not found: $FILTERED_BAM" >&2; exit 1; }
fi

numeric_folder_bounds() {
    local ids
    mapfile -t ids < <(
        find "$SINGLE_FAST5_DIR" -mindepth 1 -maxdepth 1 -type d -printf '%f\n' 2>/dev/null \
        | grep -E '^[0-9]+$' \
        | sort -n
    )
    if (( ${#ids[@]} == 0 )); then
        return 1
    fi
    printf '%s %s\n' "${ids[0]}" "${ids[${#ids[@]}-1]}"
}

folder_start=${FOLDER_START:-auto}
folder_end=${FOLDER_END:-auto}
batch_size=${FOLDER_BATCH_SIZE:-500}
(( batch_size > 0 )) || { echo "FOLDER_BATCH_SIZE must be > 0" >&2; exit 1; }

range_mode=1
if [[ "$folder_start" == "auto" || "$folder_end" == "auto" ]]; then
    if bounds=$(numeric_folder_bounds); then
        read -r discovered_start discovered_end <<< "$bounds"
        [[ "$folder_start" == "auto" ]] && folder_start=$discovered_start
        [[ "$folder_end" == "auto" ]] && folder_end=$discovered_end
    else
        range_mode=0
    fi
fi

run_one_batch() {
    local start_label=$1
    local end_label=$2
    shift 2
    local range_args=("$@")
    local tag
    if [[ "$start_label" == "all" ]]; then
        tag="all"
    else
        printf -v tag '%06d_%06d' "$start_label" "$end_label"
    fi

    local signal_tsv="$BATCH_OUTPUT_DIR/signal_${tag}.tsv"
    local mismatch_tsv="$BATCH_OUTPUT_DIR/strand_mismatch_${tag}.tsv"
    local feature_csv="$BATCH_OUTPUT_DIR/features_${tag}.csv"
    local feature_meta="$BATCH_OUTPUT_DIR/features_meta_${tag}.tsv"

    if [[ "$RESUME_BATCHES" == "1" && -s "$feature_csv" && -s "$feature_meta" ]]; then
        echo "[skip] Completed batch: $tag"
        return
    fi

    echo
    echo "[batch $tag] Extract per-base signal table"
    signal_cmd=(
        "$PYTHON_BIN" "$SIGNAL_SCRIPT"
        -p "${SIGNAL_PROCESSES:-40}"
        --fast5 "$SINGLE_FAST5_DIR"
        --reference "$REFERENCE_FASTA"
        --bam "$FILTERED_BAM"
        --output "$signal_tsv"
        --samtools "${SAMTOOLS_BIN:-samtools}"
        --bam_threads "${SIGNAL_BAM_THREADS:-16}"
        --fast5_chunksize "${SIGNAL_FAST5_CHUNKSIZE:-64}"
        --with_header
        --max_len_gap "${SIGNAL_MAX_LEN_GAP:-10}"
        --strand_mismatch_output "$mismatch_tsv"
        --basecall_group "${TOMBO_CORRECTED_GROUP:-RawGenomeCorrected_000}"
        --basecall_subgroup "${TOMBO_EVENT_SUBGROUP:-BaseCalled_template}"
        "${range_args[@]}"
    )
    [[ "${SIGNAL_REQUIRE_STRAND_MATCH:-0}" == "1" ]] && signal_cmd+=(--require_strand_match)
    [[ "${SIGNAL_REQUIRE_SEQ_MATCH:-0}" == "1" ]] && signal_cmd+=(--require_seq_match)
    [[ "${SIGNAL_ALWAYS_TAKE_INTERSECTION:-0}" == "1" ]] && signal_cmd+=(--always_take_intersection)
    [[ "${SIGNAL_KEEP_CACHE:-0}" == "1" ]] && signal_cmd+=(--keep_cache)
    run_cmd "${signal_cmd[@]}"

    echo "[batch $tag] Extract strand-aware masked 6-mer features"
    feature_cmd=(
        "$PYTHON_BIN" "$FEATURE_SCRIPT"
        --input "$signal_tsv"
        --output "$feature_csv"
        --meta_out "$feature_meta"
        --target_pos "${TARGET_POS:-1275}"
        --min_left_pos "${MIN_LEFT_POS:-1258}"
        --mask_base "${MASK_BASE:-N}"
        --label_x "${LABEL_X:-4}"
        --label_y "${LABEL_Y:-5}"
        --dwell_scale "${DWELL_SCALE:-1.0}"
        --quality_scale "${QUALITY_SCALE:-1.0}"
        --mad_use_unique "${MAD_USE_UNIQUE:-1}"
        --require_strand_match "${FEATURE_REQUIRE_STRAND_MATCH:-1}"
        --signal_round "${SIGNAL_ROUND:-6}"
        --stat_round "${STAT_ROUND:-6}"
        --with_header
    )
    run_cmd "${feature_cmd[@]}"
}

if [[ "$range_mode" == "1" ]]; then
    [[ "$folder_start" =~ ^[0-9]+$ && "$folder_end" =~ ^[0-9]+$ ]] || {
        echo "FOLDER_START and FOLDER_END must be integers or auto." >&2
        exit 1
    }
    (( folder_start <= folder_end )) || { echo "FOLDER_START must be <= FOLDER_END" >&2; exit 1; }

    current=$folder_start
    while (( current <= folder_end )); do
        end=$(( current + batch_size - 1 ))
        (( end > folder_end )) && end=$folder_end
        run_one_batch "$current" "$end" --folder_start "$current" --folder_end "$end"
        current=$(( end + 1 ))
    done
else
    echo "No numeric immediate FAST5 subfolders detected; processing all FAST5 folders in one batch."
    run_one_batch all all
fi

if [[ "$DRY_RUN" == "1" ]]; then
    echo
    echo "Dry run completed; batch merging was not executed."
    exit 0
fi

echo
echo "[merge] Feature CSV batches"
run_cmd "$PYTHON_BIN" "$MERGE_SCRIPT" \
    --inputs "$BATCH_OUTPUT_DIR/features_*.csv" \
    --output "$FINAL_OUTPUT_DIR/features_all.csv" \
    --header_mode one

echo "[merge] Feature metadata TSV batches"
run_cmd "$PYTHON_BIN" "$MERGE_SCRIPT" \
    --inputs "$BATCH_OUTPUT_DIR/features_meta_*.tsv" \
    --output "$FINAL_OUTPUT_DIR/features_meta_all.tsv" \
    --header_mode one

if [[ "${MERGE_SIGNAL_TABLES:-0}" == "1" ]]; then
    echo "[merge] Intermediate signal TSV batches"
    run_cmd "$PYTHON_BIN" "$MERGE_SCRIPT" \
        --inputs "$BATCH_OUTPUT_DIR/signal_*.tsv" \
        --output "$FINAL_OUTPUT_DIR/signal_all.tsv" \
        --header_mode one
fi

if [[ "${MERGE_MISMATCH_TABLES:-1}" == "1" ]]; then
    echo "[merge] Diagnostic mismatch TSV batches"
    run_cmd "$PYTHON_BIN" "$MERGE_SCRIPT" \
        --inputs "$BATCH_OUTPUT_DIR/strand_mismatch_*.tsv" \
        --output "$FINAL_OUTPUT_DIR/strand_mismatch_all.tsv" \
        --header_mode one \
        --allow_empty_files
fi

echo
echo "Signal and feature batch stage completed."
echo "Merged feature CSV : $FINAL_OUTPUT_DIR/features_all.csv"
echo "Merged metadata TSV: $FINAL_OUTPUT_DIR/features_meta_all.tsv"
