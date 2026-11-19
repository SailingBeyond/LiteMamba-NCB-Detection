#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Add the fixed final GatedConv to the completed Figure 2 binary benchmark.

The uploaded trainer and four-model runner remain unchanged. This entry point
selects their existing legacy_gatedconv + target_attention branch, records the
new fixed queue and storage choices, and uses an independent results directory.
"""
from __future__ import annotations

import argparse
from datetime import datetime, timezone
import fcntl
import hashlib
import importlib.util
import json
import math
import os
from pathlib import Path
import shlex
import signal
import subprocess
import sys

VERSION = "1.0.0"
HERE = Path(__file__).resolve().parent
DEFAULT_DATA_DIR = (
    "/path/to/revision-inputs/POC-XNA_DNA-01_16/POC-XNA_DNA-01_16/"
    "fast5_single_guppy/binary_read_split_721"
)
DEFAULT_OUTPUT_DIR = "/path/to/revision-results/figure2_binary_gatedconv_results_v1_0_0"
PLAN_NAME = "gatedconv_addon_plan.json"
RETENTION = {
    "epoch_metric_logs": "all epochs retained",
    "checkpoints": ["best_checkpoint.pt", "last_checkpoint.pt", "best_state_dict.pt"],
    "resume_state": "optimizer, scheduler, scaler and RNG retained in checkpoints",
    "prediction_splits": ["train", "val", "test"],
    "retained_prediction_files": ["predictions.csv.gz", "logits.npy", "labels.npy"],
    "omitted": ["epoch_prediction_arrays", "embeddings", "attention_arrays", "periodic_checkpoint_copies"],
}


def now():
    return datetime.now(timezone.utc).isoformat()


def sha256(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for block in iter(lambda: handle.read(8 * 1024**2), b""):
            digest.update(block)
    return digest.hexdigest()


def read_json(path):
    return json.loads(Path(path).read_text(encoding="utf-8"))


def write_json(path, value):
    path = Path(path)
    temporary = path.with_name(path.name + ".writing." + str(os.getpid()))
    try:
        with temporary.open("w", encoding="utf-8") as stream:
            json.dump(value, stream, ensure_ascii=False, indent=2, allow_nan=False)
            stream.write("\n")
            stream.flush()
            os.fsync(stream.fileno())
        os.replace(temporary, path)
    finally:
        temporary.unlink(missing_ok=True)


def positive_int(text):
    value = int(text)
    if value < 1:
        raise argparse.ArgumentTypeError("必须为正整数")
    return value


def parser():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    p.add_argument("--data_dir", default=DEFAULT_DATA_DIR)
    p.add_argument("--output_dir", default=DEFAULT_OUTPUT_DIR)
    p.add_argument("--seeds", nargs="+", type=int, default=[42, 43, 44],
                   help="沿用原二分类默认种子；若原实验显式使用其他种子，应传入相同完整列表")
    p.add_argument("--strands", nargs="+", choices=("pos", "neg"), default=["pos", "neg"])
    p.add_argument("--phase", choices=("all", "train", "evaluate"), default="all")
    p.add_argument("--epochs", type=positive_int, default=200)
    p.add_argument("--patience", type=positive_int, default=30)
    p.add_argument("--lr", type=float, default=0.001)
    p.add_argument("--batch_size", type=positive_int, default=256)
    p.add_argument("--bootstrap", type=int, default=200)
    p.add_argument("--device", default="auto")
    p.add_argument("--workers", type=int, default=0)
    p.add_argument("--cpu_threads", type=positive_int, default=4)
    p.add_argument("--no_plots", action="store_true")
    p.add_argument("--baseline_dir", nargs="+", default=None,
                   help="可选：仅只读汇总已经完成的原四模型结果，不启动其训练或预测")
    p.add_argument("--resume_failed", action="store_true", default=True,
                   help="自动启用；仅配置相同且已有last检查点时恢复，缺少检查点不会静默重启")
    p.add_argument("--dry_run", action="store_true")
    p.add_argument("--background", action="store_true", help="后台提交；启动器自动设置")
    return p


def validate(args):
    args.data_dir = Path(args.data_dir).expanduser().resolve()
    args.output_dir = Path(args.output_dir).expanduser().resolve()
    if (args.output_dir == args.data_dir or args.output_dir in args.data_dir.parents
            or args.data_dir in args.output_dir.parents):
        raise ValueError("新增结果目录必须与输入数据目录独立，不能互相包含")
    if args.output_dir == HERE or args.output_dir in HERE.parents or HERE in args.output_dir.parents:
        raise ValueError("结果目录不能与代码目录重叠，请使用独立结果目录")
    if args.workers < 0 or args.bootstrap < 0 or not math.isfinite(args.lr) or args.lr <= 0:
        raise ValueError("workers/bootstrap须非负，学习率须为有限正数")
    if len(set(args.seeds)) != len(args.seeds) or any(seed < 0 or seed >= 2**32 for seed in args.seeds):
        raise ValueError("种子必须是不重复的0..2**32-1整数")
    if len(set(args.strands)) != len(args.strands):
        raise ValueError("链方向不能重复")
    if args.baseline_dir:
        args.baseline_dir = [str(Path(path).expanduser().resolve()) for path in args.baseline_dir]
        if len(set(args.baseline_dir)) != len(args.baseline_dir):
            raise ValueError("旧结果目录不能重复")
        for value in args.baseline_dir:
            baseline = Path(value)
            if baseline == args.output_dir or baseline in args.output_dir.parents or args.output_dir in baseline.parents:
                raise ValueError("新增结果目录必须与只读旧结果目录独立，不能互相包含")
    return args


def check_output_ownership(output):
    """Reject old result roots before adding any queue files or logs to them."""
    output = Path(output)
    if output.exists() and not output.is_dir():
        raise ValueError("结果路径不是目录：" + str(output))
    if not output.exists() or (output / PLAN_NAME).is_file():
        return
    allowed = {"logs", ".gatedconv_queue.lock", "dry_run_manifest.json"}
    found = {path.name for path in output.iterdir()}
    if not found.issubset(allowed):
        raise ValueError("该目录已有其他实验产物，不能作为新增GatedConv结果目录；请指定新的--output_dir：" + str(output))


def source_hashes():
    return {path.name: sha256(path) for path in sorted(HERE.glob("*.py"))}


def make_plan(args):
    return {
        "addon_version": VERSION,
        "display_name": "GatedConv",
        "model_type": "litemamba",
        "litemamba_variant": "legacy_gatedconv",
        "readout": "target_attention",
        "model_dim": 96,
        "num_blocks": 4,
        "target_index": 3,
        "feature_scheme": "full",
        "num_classes": 2,
        "class_names": {"0": "PC", "1": "XNA"},
        "seeds": list(args.seeds),
        "strands": list(args.strands),
        "data_dir": str(args.data_dir),
        "output_dir": str(args.output_dir),
        "training": {key: getattr(args, key) for key in (
            "epochs", "patience", "lr", "batch_size", "bootstrap", "device", "workers", "cpu_threads", "no_plots")},
        "fixed_preprocessing": {"seq_len": 6, "raw_signal_len": 30, "dwell_divisor": 60.0,
                                "quality_divisor": 50.0, "accum_steps": 1},
        "source_sha256": source_hashes(),
        "retention": RETENTION,
        "execution": "serial original binary queue; all new runs trained/validated before any new test evaluation",
        "initialization": "independent seeded random initialization; no old binary or six-class checkpoint reused",
    }


def commit_plan(args):
    path = args.output_dir / PLAN_NAME
    current = make_plan(args)
    if path.exists():
        if read_json(path) != current:
            raise ValueError("新增GatedConv方案、数据路径或源码与已有结果不一致；不会覆盖，请使用原配置或新输出目录")
    else:
        write_json(path, current)
    return current


def load_runner():
    path = HERE / "run_all_4_models.py"
    spec = importlib.util.spec_from_file_location("_unchanged_binary_queue_for_gatedconv", path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def upstream_arguments(args):
    values = ["--data_dir", str(args.data_dir), "--output_dir", str(args.output_dir),
              "--models", "litemamba", "--feature_schemes", "full",
              "--litemamba_variant", "legacy_gatedconv", "--readout", "target_attention",
              "--model_dim", "96", "--num_blocks", "4", "--state_dim", "16",
              "--raw_signal_len", "30", "--dwell_divisor", "60", "--quality_divisor", "50",
              "--accum_steps", "1", "--no_embeddings", "--embedding_dtype", "float32",
              "--resume_failed"]
    for key in ("phase", "epochs", "patience", "lr", "batch_size", "bootstrap", "device", "workers"):
        values.extend(["--" + key, str(getattr(args, key))])
    values += ["--seeds"] + [str(seed) for seed in args.seeds]
    values += ["--strands"] + list(args.strands)
    if args.no_plots:
        values.append("--no_plots")
    if args.dry_run:
        values.append("--dry_run")
    return values


def configure_runner(runner, args):
    """Only extend the new queue's recording and command construction."""
    original_config = runner.build_requested_config
    original_command = runner.build_command
    addon_identity = {"version": VERSION, "entry_sha256": sha256(Path(__file__)),
                      "display_name": "GatedConv", "retention": RETENTION}
    cache = str(args.output_dir / "shared_feature_cache")

    def requested(upstream_args, model, seed, strand, scheme, trainer):
        config = original_config(upstream_args, model, seed, strand, scheme, trainer)
        config.update(gatedconv_addon=addon_identity, no_epoch_logits=True, no_attention=True,
                      checkpoint_every=0, cpu_threads=args.cpu_threads, cache_dir=cache)
        return config

    def command(config, run_dir, mode, resume=False):
        value = original_command(config, run_dir, mode, resume=resume)
        value += ["--no_epoch_logits", "--no_attention", "--checkpoint_every", "0",
                  "--cpu_threads", str(config["cpu_threads"]), "--cache_dir", config["cache_dir"]]
        return value

    runner.build_requested_config = requested
    runner.build_command = command


def submit_background(args, raw_arguments):
    check_output_ownership(args.output_dir)
    logs = args.output_dir / "logs"
    logs.mkdir(parents=True, exist_ok=True)
    stamp = datetime.now().strftime("%Y%m%d_%H%M%S_%f") + "_" + str(os.getpid())
    log_path = logs / ("launch_gatedconv_" + stamp + ".log")
    command = [sys.executable, "-u", str(Path(__file__).resolve())]
    command += [value for value in raw_arguments if value != "--background"]
    with log_path.open("xb") as stream:
        process = subprocess.Popen(command, cwd=str(HERE), stdout=stream, stderr=subprocess.STDOUT,
                                   stdin=subprocess.DEVNULL, start_new_session=True)
    print("已提交新增GatedConv启动请求，PID=" + str(process.pid), flush=True)
    print("本次独立日志：" + str(log_path), flush=True)
    print("查看日志：tail -F " + shlex.quote(str(log_path)), flush=True)
    print("启动请求不代表训练成功；以日志中的 [FULL PASS] 标记为准。", flush=True)
    return 0


def run(args):
    check_output_ownership(args.output_dir)
    args.output_dir.mkdir(parents=True, exist_ok=True)
    with (args.output_dir / ".gatedconv_queue.lock").open("a") as lock:
        try:
            fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError as error:
            raise RuntimeError("该新增GatedConv结果目录已有队列运行，未重复启动") from error
        def interrupted(signum, frame):
            # Raising inside original execute() enters its BaseException cleanup:
            # its current trainer is terminated before our queue lock is released.
            raise InterruptedError("收到停止信号" + str(signum) + "；停止当前子任务并保留恢复状态")
        previous_handlers = {sig: signal.signal(sig, interrupted) for sig in (signal.SIGTERM, signal.SIGINT)}
        try:
            runner = load_runner()
            configure_runner(runner, args)
            if not args.dry_run:
                commit_plan(args)
                logs = args.output_dir / "logs"
                logs.mkdir(exist_ok=True)
                write_json(logs / "gatedconv_binary.pid.json", {"pid": os.getpid(), "accepted_at": now(),
                           "entry": str(Path(__file__).resolve()), "phase": args.phase})
                print("[ACCEPTED] GatedConv binary addon " + VERSION + "; independent runs="
                      + str(len(args.seeds) * len(args.strands)) + "; serial queue", flush=True)
            code = runner.main(upstream_arguments(args))
            if code:
                raise RuntimeError("新增GatedConv队列失败，保留检查点和日志；修复后以同一命令恢复，未运行原四模型")
            if args.dry_run:
                print("[DRY RUN] 仅打印新增GatedConv命令；未启动训练、预测或读取旧模型结果", flush=True)
                return 0
            from summarize_gatedconv_binary import summarize
            result = summarize(args.output_dir, baseline_dirs=args.baseline_dir, package_dir=HERE)
            print("[PACKAGED] " + str(result["core_path"]), flush=True)
            print("[FULL PASS] Figure2 GatedConv phase=" + args.phase + " output=" + str(args.output_dir), flush=True)
            return 0
        finally:
            for sig, old_handler in previous_handlers.items():
                signal.signal(sig, old_handler)


def main(argv=None):
    raw = list(sys.argv[1:] if argv is None else argv)
    args = validate(parser().parse_args(raw))
    if args.background and not args.dry_run:
        return submit_background(args, raw)
    return run(args)


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except Exception as error:
        print("[ERROR] " + type(error).__name__ + ": " + str(error), file=sys.stderr, flush=True)
        raise SystemExit(1)
