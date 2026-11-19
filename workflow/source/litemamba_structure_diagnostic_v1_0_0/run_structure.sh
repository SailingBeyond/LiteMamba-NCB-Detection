#!/usr/bin/env bash
# 使用当前conda环境后台提交两项结构诊断。
set -euo pipefail
HERE="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
DATA_DIR="/path/to/revision-inputs/CPLX-XNA-replenishment/CPLX-XNA-replenishment/fast5_single_guppy/sixclass_overlap_read_split_721"
OUTPUT_DIR=""
arguments=("$@")
for ((i=0; i<${#arguments[@]}; i++)); do
    case "${arguments[i]}" in
        --data_dir)
            if ((i+1>=${#arguments[@]})); then printf '%s\n' '--data_dir缺少路径' >&2; exit 2; fi
            DATA_DIR="${arguments[i+1]}"; i=$((i+1));;
        --data_dir=*) DATA_DIR="${arguments[i]#*=}";;
        --output_dir)
            if ((i+1>=${#arguments[@]})); then printf '%s\n' '--output_dir缺少路径' >&2; exit 2; fi
            OUTPUT_DIR="${arguments[i+1]}"; i=$((i+1));;
        --output_dir=*) OUTPUT_DIR="${arguments[i]#*=}";;
    esac
done
if [[ -z "$OUTPUT_DIR" ]]; then
    OUTPUT_DIR="$DATA_DIR/sixclass_auto_v1_0_4_results/litemamba_structure_seed42_v1_0_0"
fi
OUTPUT_DIR="$(python3 -c 'from pathlib import Path; import sys; print(Path(sys.argv[1]).expanduser().resolve())' "$OUTPUT_DIR")"
mkdir -p "$HERE/launcher_logs"
LOG="$(mktemp "$HERE/launcher_logs/launch_$(date -u +%Y%m%d_%H%M%S)_XXXXXX.log")"
nohup python3 -u "$HERE/run_structure.py" "$@" > "$LOG" 2>&1 < /dev/null &
submitted_pid=$!
printf '已提交结构诊断启动请求，PID=%s\n' "$submitted_pid"
printf '本次启动日志：%s\n' "$LOG"
printf '同时查看启动和训练：tail -F %q %q\n' "$LOG" "$OUTPUT_DIR/logs/structure_latest.log"
printf '输出目录：%s\n' "$OUTPUT_DIR"
printf '%s\n' '预检通过后只训练2项seed42结构对照；full使用原结果。'
