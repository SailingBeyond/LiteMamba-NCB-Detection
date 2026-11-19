#!/usr/bin/env bash
# 当前conda环境后台提交；预检与失败原因均保存在本次独立启动日志。
set -euo pipefail
HERE="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
for argument in "$@"; do
    case "$argument" in
        --dry_run|--check_only|--summary_only|--help|-h|--version)
            exec python3 -u "$HERE/run_ablation.py" "$@";;
    esac
done
mkdir -p "$HERE/launcher_logs"
LOG="$(mktemp "$HERE/launcher_logs/launch_$(date -u +%Y%m%d_%H%M%S)_XXXXXX.log")"
nohup python3 -u "$HERE/run_ablation.py" "$@" > "$LOG" 2>&1 < /dev/null &
submitted_pid=$!
printf '已提交12次去除组消融训练启动请求，PID=%s\n' "$submitted_pid"
printf '本次启动日志：%s\n' "$LOG"
printf '查看预检：tail -F %q\n' "$LOG"
printf '%s\n' '预检和磁盘预算通过后启动正式队列；各3次：no_raw/no_context/no_responses/no_statistics；优先no_raw。'
printf '%s\n' '默认输出：DATA_DIR/sixclass_auto_v1_0_4_results/gatedconv_ablation12_v1_0_0'
printf '%s\n' '查看队列：tail -F 输出目录/logs/queue_latest.log'
printf '%s\n' '中断或清理空间后：bash run_ablation12.sh --resume（保留首次使用的路径/资源参数）。'
