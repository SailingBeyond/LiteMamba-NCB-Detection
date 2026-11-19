#!/usr/bin/env python3
"""Display this collector's saved progress without touching model training."""
import argparse
import json
from pathlib import Path
import sys

from collect_thresholds import ARCHIVE, DEFAULT_RESULTS, PACKAGE


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output_dir", type=Path, default=DEFAULT_RESULTS.parent / PACKAGE)
    args = parser.parse_args(argv)
    output = args.output_dir.expanduser().resolve()
    path = output / "workflow_status.json"
    if not path.is_file():
        print("尚无收集进度记录，请先查看启动终端打印的独立日志。")
        return
    state = json.loads(path.read_text(encoding="utf-8"))
    print("统计输出目录：" + str(output))
    print("阶段：" + str(state.get("status")))
    print("已完成文件：%s / %s" % (state.get("completed_sources", 0), state.get("total_sources", "?")))
    if state.get("last_source"):
        print("最近完成：" + json.dumps(state["last_source"], ensure_ascii=False))
    if state.get("error"):
        print("错误：" + state["error"])
    if state.get("status") == "complete":
        print("上传文件：" + str(output / ARCHIVE))
    print("各文件窗口数不同，文件完成比例不等于时间进度；单文件进度见独立日志 [SCAN]。")


if __name__ == "__main__":
    try:
        main()
    except (OSError, ValueError) as error:
        print("[ERROR] " + str(error), file=sys.stderr)
        sys.exit(1)
