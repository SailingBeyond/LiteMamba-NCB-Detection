#!/usr/bin/env python3
"""后台提交固定阶段，给每次请求保留独立日志；队列锁由协调器取得。"""
from __future__ import annotations

from datetime import datetime
import os
from pathlib import Path
import subprocess
import sys


def main(argv=None):
    import run_direction_5class as runner

    flags = list(sys.argv[1:] if argv is None else argv)
    args = runner.validate_args(runner.parser().parse_args(flags))
    if args.dry_run:
        return runner.run(args)
    logs = args.output_dir / 'logs'
    logs.mkdir(parents=True, exist_ok=True)
    stamp = datetime.now().strftime('%Y%m%d_%H%M%S_%f')
    log_path = logs / ('launch_' + args.phase + '_' + stamp + '.log')
    pid_path = log_path.with_suffix('.pid')
    command = [sys.executable, '-u', str(Path(__file__).with_name('run_direction_5class.py')),
               '--resume', *flags]
    with log_path.open('xb') as log:
        child = subprocess.Popen(command, stdin=subprocess.DEVNULL, stdout=log,
                                 stderr=subprocess.STDOUT, start_new_session=True,
                                 close_fds=True)
    temporary = pid_path.with_suffix('.pid.writing')
    try:
        with temporary.open('w', encoding='utf-8') as stream:
            stream.write(str(child.pid) + '\n')
            stream.flush()
            os.fsync(stream.fileno())
        os.replace(temporary, pid_path)
    except BaseException:
        child.terminate()
        child.wait()
        raise
    finally:
        temporary.unlink(missing_ok=True)
    print('已提交 %s 启动请求，PID=%d' % (args.phase, child.pid))
    print('本次独立日志：' + str(log_path))
    print('查看日志：tail -F "' + str(log_path) + '"')
    print('启动请求不代表训练成功；以日志中的阶段完成标记为准。')
    return child.pid


if __name__ == '__main__':
    try:
        main()
    except (OSError, ValueError, RuntimeError) as error:
        print('[ERROR] ' + str(error), file=sys.stderr)
        raise SystemExit(1)
