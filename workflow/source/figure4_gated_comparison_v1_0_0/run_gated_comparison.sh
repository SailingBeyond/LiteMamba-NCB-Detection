#!/usr/bin/env bash
# 继承冻结外部数据，后台执行三种子门控卷积比较。
set -euo pipefail
HERE="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
PREVIOUS="/path/to/revision-inputs/CPLX-XNA-replenishment/CPLX-XNA-replenishment/fast5_single_guppy/sixclass_overlap_read_split_721/sixclass_auto_v1_0_4_results/figure4_external_xna13_16_corrected_v1_1_0"
OUTPUT=""
arguments=("$@")
for ((i=0; i<${#arguments[@]}; i++)); do
    case "${arguments[i]}" in
        --previous_output)
            if ((i+1>=${#arguments[@]})); then printf '%s\n' '--previous_output缺少路径' >&2; exit 2; fi
            PREVIOUS="${arguments[i+1]}"; i=$((i+1));;
        --previous_output=*) PREVIOUS="${arguments[i]#*=}";;
        --output_root|--output_dir)
            if ((i+1>=${#arguments[@]})); then printf '%s\n' '--output_root缺少路径' >&2; exit 2; fi
            OUTPUT="${arguments[i+1]}"; i=$((i+1));;
        --output_root=*|--output_dir=*) OUTPUT="${arguments[i]#*=}";;
    esac
done
OUTPUT="$(python3 -c 'from pathlib import Path; import sys; previous=Path(sys.argv[1]).expanduser().resolve(); print(Path(sys.argv[2]).expanduser().resolve() if sys.argv[2] else previous.parent/"figure4_gated_comparison_v1_0_0")' "$PREVIOUS" "$OUTPUT")"
mkdir -p "$OUTPUT/logs"
LOG="$(mktemp "$OUTPUT/logs/launch_$(date -u +%Y%m%d_%H%M%S)_XXXXXX.log")"
nohup python3 -u "$HERE/run_gated_comparison.py" "$@" \
    --launcher_log "$LOG" --pid_file "$OUTPUT/logs/gated_comparison.pid" \
    --latest_log "$OUTPUT/logs/gated_comparison_latest.log" > "$LOG" 2>&1 < /dev/null &
submitted_pid=$!
printf '已提交 Figure4 门控卷积比较，PID=%s\n' "$submitted_pid"
printf '本次独立日志：%s\n' "$LOG"
printf '查看本次日志：tail -F %q\n' "$LOG"
printf '输出目录：%s\n' "$OUTPUT"
printf '%s\n' '正式PID和latest日志在成功取得队列锁后更新；默认仅新增3组门控卷积预测，不训练。'
