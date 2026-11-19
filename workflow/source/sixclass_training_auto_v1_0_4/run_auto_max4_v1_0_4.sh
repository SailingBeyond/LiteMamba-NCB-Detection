#!/usr/bin/env bash
# 直接后台执行：实测资源 -> 验证并发 -> 四模型 × 五个 seed。
# --probe_only 只测资源；--resume_main 恢复本版队列。
set -euo pipefail
HERE="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
cd "$HERE"

DATA_DIR="/path/to/revision-inputs/CPLX-XNA-replenishment/CPLX-XNA-replenishment/fast5_single_guppy/sixclass_overlap_read_split_721"
OUTPUT_ROOT=""
extra_args=("$@")
for ((i=0; i<${#extra_args[@]}; i++)); do
  case "${extra_args[i]}" in
    --data_dir)
      if ((i+1>=${#extra_args[@]})); then printf '%s\n' '--data_dir 缺少路径' >&2; exit 2; fi
      DATA_DIR="${extra_args[i+1]}"; i=$((i+1));;
    --data_dir=*) DATA_DIR="${extra_args[i]#*=}";;
    --output_root)
      if ((i+1>=${#extra_args[@]})); then printf '%s\n' '--output_root 缺少路径' >&2; exit 2; fi
      OUTPUT_ROOT="${extra_args[i+1]}"; i=$((i+1));;
    --output_root=*) OUTPUT_ROOT="${extra_args[i]#*=}";;
  esac
done
if [[ -z "$OUTPUT_ROOT" ]]; then OUTPUT_ROOT="$DATA_DIR/sixclass_auto_v1_0_4_results"; fi
OUTPUT_ROOT="$(python3 -c 'from pathlib import Path; import sys; print(Path(sys.argv[1]).expanduser().resolve())' "$OUTPUT_ROOT")"
mkdir -p "$OUTPUT_ROOT/logs"
# 每次启动使用独立日志。唯一正式 PID 和 latest 链接由 Python 获锁后写入。
LOG="$(mktemp "$OUTPUT_ROOT/logs/auto_max4_v1_0_4_$(date -u +%Y%m%d_%H%M%S)_XXXXXX.log")"
PID="$OUTPUT_ROOT/logs/auto_max4_v1_0_4.pid"
LATEST="$OUTPUT_ROOT/logs/auto_max4_v1_0_4.log"

nohup python3 -u "$HERE/auto_find_and_train.py" \
  --data_dir "$DATA_DIR" \
  --output_root "$OUTPUT_ROOT" \
  --max_parallel 4 --min_parallel 1 --probe_steps 8 \
  --batch_size 256 --workers 0 --cpu_threads 4 \
  --epochs 200 --patience 30 --lr 0.001 \
  --gpu_reserve_gib 2 --ram_reserve_gib 3 --reserve_fraction 0.10 \
  --memory_safety_factor 1.35 --train_ram_extra_gib 1.0 \
  "$@" \
  --launcher_log "$LOG" --pid_file "$PID" --latest_log "$LATEST" \
  > "$LOG" 2>&1 < /dev/null &
submitted_pid=$!
printf '已提交启动请求，PID=%s\n本次独立日志：%s\n' "$submitted_pid" "$LOG"
printf '查看本次日志：tail -f %q\n' "$LOG"
printf '正式 PID 与最新日志在程序成功取得队列锁后更新。\n'
