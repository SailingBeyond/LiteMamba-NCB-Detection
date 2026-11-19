#!/usr/bin/env python3
"""Submit the read-only threshold collection as one detached CPU process."""
from datetime import datetime
import os
from pathlib import Path
import subprocess
import sys

import collect_thresholds as collector


def main(argv=None):
    flags = list(sys.argv[1:] if argv is None else argv)
    args = collector.parser().parse_args(flags)
    if args.check_only:
        return collector.main(flags)
    source, output, thresholds, description, reference, reference_guard = collector.preflight(args)
    output.mkdir(parents=True, exist_ok=True)
    logs = collector.safe_directory(output, "logs")
    stamp = datetime.now().strftime("%Y%m%d_%H%M%S_%f") + "_" + str(os.getpid())
    log_path = logs / ("collect_thresholds_" + stamp + ".log")
    command = [sys.executable, "-u", str(Path(__file__).with_name("collect_thresholds.py")), *flags]
    env = dict(os.environ, OMP_NUM_THREADS="1", MKL_NUM_THREADS="1", OPENBLAS_NUM_THREADS="1", CUDA_VISIBLE_DEVICES="")
    with log_path.open("xb") as stream:
        child = subprocess.Popen(command, stdin=subprocess.DEVNULL, stdout=stream,
                                 stderr=subprocess.STDOUT, start_new_session=True,
                                 close_fds=True, env=env)
    try:
        log_path.with_suffix(".pid").write_text(str(child.pid) + "\n", encoding="utf-8")
    except OSError:
        print("[PID NOTE] 未保存PID小文件；收集任务仍按独立日志继续。", flush=True)
    print("[COLLECTION SUBMITTED] PID=" + str(child.pid))
    print("独立日志：" + str(log_path))
    print('查看日志：tail -F "' + str(log_path) + '"')
    print("完成后上传：" + str(output / collector.ARCHIVE))
    print("仅提交 CPU 统计请求；以日志 [FULL PASS] 和 [RESULT PACKAGE] 为完成标记。")
    print("无需暂停训练；本工具不加载模型、不调用GPU、不修改原预测和权重。")
    return child.pid


if __name__ == "__main__":
    try:
        main()
    except (OSError, ValueError, RuntimeError, KeyError) as error:
        print("[ERROR] " + type(error).__name__ + ": " + str(error), file=sys.stderr, flush=True)
        sys.exit(1)
