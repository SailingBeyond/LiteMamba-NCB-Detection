#!/usr/bin/env bash
set -euo pipefail
SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
RESULTS_DEFAULT='/path/to/revision-inputs/CPLX-XNA-replenishment/CPLX-XNA-replenishment/fast5_single_guppy/sixclass_overlap_read_split_721/sixclass_auto_v1_0_4_results'
RESULTS_DIR="$RESULTS_DEFAULT"
OUTPUT_DIR=''
ARGS=("$@")
for ((i=0; i<${#ARGS[@]}; i++)); do
    case "${ARGS[$i]}" in
        --output_dir) OUTPUT_DIR="${ARGS[$((i+1))]:?--output_dir 需要路径}" ;;
        --output_dir=*) OUTPUT_DIR="${ARGS[$i]#*=}" ;;
        --results_dir) RESULTS_DIR="${ARGS[$((i+1))]:?--results_dir 需要路径}" ;;
        --results_dir=*) RESULTS_DIR="${ARGS[$i]#*=}" ;;
    esac
done
OUTPUT_DIR="${OUTPUT_DIR:-$RESULTS_DIR/gatedconv_internal_test_v1_0_0}"
mkdir -p -- "$OUTPUT_DIR/logs"
LAUNCH_LOG="$(mktemp "$OUTPUT_DIR/logs/launch_$(date +%Y%m%d_%H%M%S)_XXXXXX.log")"
nohup python3 -u "$SCRIPT_DIR/run_internal_test.py" "${ARGS[@]}" \
    --launcher_log "$LAUNCH_LOG" > "$LAUNCH_LOG" 2>&1 < /dev/null &
REQUEST_PID=$!
echo "已提交内部测试评估请求，PID=$REQUEST_PID"
echo "本次日志：$LAUNCH_LOG"
echo "查看日志：tail -F '$LAUNCH_LOG'"
echo '成功取得队列锁后，程序更新 logs/latest.log 与正式 PID。'
