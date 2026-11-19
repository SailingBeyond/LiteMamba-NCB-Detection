#!/usr/bin/env bash
# 保留当前 conda 环境；只读复用 full，后台运行九个单输入模型的外部评价。
set -euo pipefail
HERE="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
PYTHON_BIN="${PYTHON_BIN:-python3}"
PREVIOUS="/path/to/revision-inputs/CPLX-XNA-replenishment/CPLX-XNA-replenishment/fast5_single_guppy/sixclass_overlap_read_split_721/sixclass_auto_v1_0_4_results/figure4_gated_comparison_v1_0_0"
OUTPUT=""
foreground=0
arguments=()
while (($#)); do
    case "$1" in
        --foreground) foreground=1; shift;;
        --check_only|--help|-h|--version) foreground=1; arguments+=("$1"); shift;;
        --previous_output)
            if (($#<2)); then printf '%s\n' '--previous_output缺少路径' >&2; exit 2; fi
            PREVIOUS="$2"; arguments+=("$1" "$2"); shift 2;;
        --previous_output=*) PREVIOUS="${1#*=}"; arguments+=("$1"); shift;;
        --output_root|--output_dir)
            if (($#<2)); then printf '%s\n' '--output_root缺少路径' >&2; exit 2; fi
            OUTPUT="$2"; arguments+=("$1" "$2"); shift 2;;
        --output_root=*|--output_dir=*) OUTPUT="${1#*=}"; arguments+=("$1"); shift;;
        *) arguments+=("$1"); shift;;
    esac
done
if ((foreground)); then
    exec "$PYTHON_BIN" -u "$HERE/run_singleinput_external.py" "${arguments[@]}"
fi
OUTPUT="$("$PYTHON_BIN" -c 'from pathlib import Path; import sys; previous=Path(sys.argv[1]).expanduser().resolve(); print(Path(sys.argv[2]).expanduser().resolve() if sys.argv[2] else previous.parent/"figure4_singleinput9_v1_0_0")' "$PREVIOUS" "$OUTPUT")"
mkdir -p "$OUTPUT/logs"
LOG="$(mktemp "$OUTPUT/logs/launch_$(date -u +%Y%m%d_%H%M%S)_XXXXXX.log")"
nohup "$PYTHON_BIN" -u "$HERE/run_singleinput_external.py" "${arguments[@]}" \
    --launcher_log "$LOG" --pid_file "$OUTPUT/logs/singleinput_external.pid" \
    --latest_log "$OUTPUT/logs/singleinput_external_latest.log" > "$LOG" 2>&1 < /dev/null &
submitted_pid=$!
printf '已提交 Figure4 单输入外部评价，PID=%s\n' "$submitted_pid"
printf '本次独立日志：%s\n' "$LOG"
printf '查看本次日志：tail -F %q\n' "$LOG"
printf '输出目录：%s\n' "$OUTPUT"
printf '%s\n' '正式PID和latest日志在成功取得队列锁后更新。默认9项新预测+3项完整输入只读复用，无训练。'
