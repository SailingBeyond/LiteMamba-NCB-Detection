#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Predeclare, profile, concurrently train, freeze, then evaluate six-class runs.

No test prediction is requested until every planned training run succeeds and
all validation-selected checkpoint hashes are frozen. Resource observations,
reservations, commands, retries and process logs are retained for audit.
"""
from __future__ import annotations
import argparse
import fcntl
import hashlib
import importlib
import json
import os
from pathlib import Path
import shlex
import shutil
import signal
import subprocess
import sys
import time
import uuid as uuid_module
from datetime import datetime, timezone
from contextlib import contextmanager

from resource_scheduler import ResourceController, request_from_profile, training_request_from_profile, nvidia_snapshot, visible_gpu_mapping, classify_oom, available_cpu_slots, proc_directory, process_children

HERE = Path(__file__).resolve().parent
TRAINER = HERE / "train_all.py"
MODELS = ("bigru", "transformer", "convmixer", "litemamba")
STRUCTURE_MODELS = ("gatedconv_target", "ssm_global_only")
STRUCTURE_SETTINGS = {
    "gatedconv_target": ("legacy_gatedconv", "target_attention"),
    "ssm_global_only": ("selective_ssm", "global_attention_only"),
}
SCHEMES = ("full", "raw_only", "context_only", "responses_only", "statistics_only", "no_raw", "no_context", "no_responses", "no_statistics")
DEFAULT_DATA_DIR = "/path/to/revision-inputs/CPLX-XNA-replenishment/CPLX-XNA-replenishment/fast5_single_guppy/sixclass_overlap_read_split_721"
QUEUE_VERSION = "sixclass-profiled-staged-parallel-v1.0.4"


def utc_now():
    return datetime.now(timezone.utc).isoformat()


def sha256_file(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def load_json(path):
    return json.loads(Path(path).read_text(encoding="utf-8"))


def save_json(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + ".tmp")
    temporary.write_text(json.dumps(value, ensure_ascii=False, indent=2, allow_nan=False) + "\n", encoding="utf-8")
    temporary.replace(path)


def append_json(path, value):
    with Path(path).open("a", encoding="utf-8") as handle:
        handle.write(json.dumps(value, ensure_ascii=False, allow_nan=False) + "\n")


def parse_args(argv=None):
    parser = argparse.ArgumentParser(description="六分类四模型：资源实测、并发训练、全队列冻结后测试。", formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument("--data_dir", default=DEFAULT_DATA_DIR)
    parser.add_argument("--output_dir", default=None,
                        help="默认在 data_dir/sixclass_auto_v1_0_4_results/main_20 中保存全部队列产物")
    parser.add_argument("--cache_dir", default=None)
    parser.add_argument("--models", nargs="+", choices=MODELS + STRUCTURE_MODELS, default=list(MODELS))
    parser.add_argument("--structure_diagnostic", action="store_true", help="固定两个结构对照、seed42、full输入和仅train/val阶段，共用一个资源调度队列")
    parser.add_argument("--seeds", nargs="+", type=int, default=[42, 2024, 2025, 2026, 2027])
    parser.add_argument("--feature_schemes", nargs="+", choices=SCHEMES, default=["full"])
    parser.add_argument("--phase", choices=("all", "train", "evaluate"), default="all")
    parser.add_argument("--epochs", type=int, default=200)
    parser.add_argument("--patience", type=int, default=30)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--batch_size", type=int, default=256)
    parser.add_argument("--workers", type=int, default=0, help="每个训练进程的加载子进程数；memmap 数据默认 0，避免并发时重复预取")
    parser.add_argument("--cpu_threads", type=int, default=min(4, available_cpu_slots()))
    parser.add_argument("--accum_steps", type=int, default=1)
    parser.add_argument("--raw_signal_len", type=int, default=30)
    parser.add_argument("--dwell_divisor", type=float, default=60.0)
    parser.add_argument("--quality_divisor", type=float, default=50.0)
    parser.add_argument("--weight_decay", type=float, default=1e-4)
    parser.add_argument("--label_smoothing", type=float, default=0.0)
    parser.add_argument("--clip_grad", type=float, default=1.0)
    parser.add_argument("--monitor", choices=("val_loss", "val_macro_f1"), default="val_loss")
    parser.add_argument("--model_dim", type=int, default=96)
    parser.add_argument("--num_blocks", type=int, default=4)
    parser.add_argument("--state_dim", type=int, default=16)
    parser.add_argument("--litemamba_variant", choices=("selective_ssm", "legacy_gatedconv"), default="selective_ssm")
    parser.add_argument("--readout", choices=("target_attention", "legacy_last_attention", "global_attention_only"), default="target_attention")
    parser.add_argument("--bootstrap", type=int, default=200)
    parser.add_argument("--no_plots", action="store_true")
    parser.add_argument("--no_embeddings", action="store_true")
    parser.add_argument("--embedding_dtype", choices=("float16", "float32"), default="float32")
    parser.add_argument("--no_epoch_logits", action="store_true")
    parser.add_argument("--save_epoch_predictions", action="store_true")
    parser.add_argument("--external_manifest")
    parser.add_argument("--device", default="auto", help="auto/cpu/cuda/cuda:N；cuda:N 是 CUDA_VISIBLE_DEVICES 后的逻辑编号")
    parser.add_argument("--gpu_ids", nargs="+", type=int, help="可使用的 CUDA 可见逻辑 GPU 编号；默认全部可见 GPU")
    parser.add_argument("--max_parallel", type=int, default=4, help="并发硬上限；实测显存、RAM、CPU 预算仍可降低并发")
    parser.add_argument("--cpu_parallel_limit", type=int, default=2, help="CPU 模式额外并发上限")
    parser.add_argument("--cpu_budget", type=int, help="全队列 CPU 线程+worker 总预算；默认可用 CPU 数，至少 8 CPU 时保留 1 个")
    parser.add_argument("--gpu_reserve_gib", type=float, default=2.0)
    parser.add_argument("--ram_reserve_gib", type=float, default=2.0)
    parser.add_argument("--reserve_fraction", type=float, default=0.10, help="保留容量占比；与固定 GiB 保留值取较大者")
    parser.add_argument("--memory_safety_factor", type=float, default=1.35)
    parser.add_argument("--staged_exports", action="store_true", help="训练权重阶段并行，所有任务训练结束后再串行导出 train/val；新输出目录测试用")
    parser.add_argument("--train_ram_extra_gib", type=float, default=1.0, help="阶段化训练 RAM 预约的每进程额外 GiB 余量")
    parser.add_argument("--profile_steps", type=int, default=3, help="每个模型/特征方案实测完整训练更新步数")
    parser.add_argument("--poll_seconds", type=float, default=2.0)
    parser.add_argument("--resource_wait_seconds", type=float, default=1800.0, help="没有活动作业且资源不足时的最长等待；不改变 batch size")
    parser.add_argument("--oom_retries", type=int, choices=(0, 1, 2), default=1,
                        help="确认 OOM 或无 traceback 的 SIGKILL 疑似资源失败，降低并发后最多重试次数；SIGKILL 不记为已证实 OOM")
    parser.add_argument("--resume_queue", "--resume_failed", dest="resume_queue", action="store_true", help="配置/数据/源代码不变时恢复队列；有 last checkpoint 则续训")
    parser.add_argument("--continue_on_error", action="store_true", help="继续其余训练；任一训练失败仍禁止整队测试")
    parser.add_argument("--no_isolated_benchmark", action="store_true", help="跳过训练和评价进程全部退出后串行运行的独立推理基准")
    parser.add_argument("--dry_run", action="store_true")
    args = parser.parse_args(argv)
    if args.structure_diagnostic:
        supplied = list(sys.argv[1:] if argv is None else argv)
        fixed = {"models": list(STRUCTURE_MODELS), "feature_schemes": ["full"], "phase": "train"}
        for key, expected in fixed.items():
            explicitly_set = any(token == "--" + key or token.startswith("--" + key + "=") for token in supplied)
            if explicitly_set and getattr(args, key) != expected:
                parser.error("结构诊断固定--" + key + "=" + str(expected))
            setattr(args, key, expected)
        # seed42用于原诊断；确认阶段只新增2024/2025，旧seed42由外层只读复用。
        seeds_explicit = any(token == "--seeds" or token.startswith("--seeds=") for token in supplied)
        if not seeds_explicit:
            args.seeds = [42]
        if args.seeds not in ([42], [2024, 2025]):
            parser.error("结构任务的--seeds只能是42（原诊断）或2024 2025（三种子确认的新增任务）")
        if args.external_manifest:
            parser.error("结构诊断不接受external_manifest")
        if args.litemamba_variant != "selective_ssm" or args.readout != "target_attention":
            parser.error("结构诊断的基准参数固定selective_ssm和target_attention，各对照由独立身份映射")
        args.staged_exports = True
        args.no_isolated_benchmark = True
        args.max_parallel = min(args.max_parallel, 2)
        args.cpu_parallel_limit = min(args.cpu_parallel_limit, 2)
    elif any(model in STRUCTURE_MODELS for model in args.models):
        parser.error("结构任务别名只能与--structure_diagnostic一起使用")
    if args.output_dir is None:
        suffix = "litemamba_structure_diagnostic_v1_0_0" if args.structure_diagnostic else "main_20"
        args.output_dir = str(Path(args.data_dir).expanduser() / "sixclass_auto_v1_0_4_results" / suffix)
    for key in ("models", "seeds", "feature_schemes"):
        if len(set(getattr(args, key))) != len(getattr(args, key)):
            parser.error(f"--{key} 不能含重复项")
    positive = ("epochs", "patience", "batch_size", "cpu_threads", "accum_steps", "raw_signal_len", "model_dim", "num_blocks", "state_dim", "max_parallel", "cpu_parallel_limit", "profile_steps")
    if any(getattr(args, key) <= 0 for key in positive) or args.workers < 0 or args.bootstrap < 0:
        parser.error("计数参数必须为正，workers/bootstrap 必须 >= 0")
    if args.cpu_budget is not None and args.cpu_budget < args.cpu_threads + args.workers:
        parser.error("cpu_budget 至少应容纳一个作业的 cpu_threads + workers")
    if min(args.lr, args.dwell_divisor, args.quality_divisor, args.poll_seconds, args.resource_wait_seconds) <= 0:
        parser.error("学习率、缩放值、轮询和等待时间必须 > 0")
    if args.memory_safety_factor < 1 or not (0 <= args.reserve_fraction < 1) or min(args.gpu_reserve_gib, args.ram_reserve_gib, args.train_ram_extra_gib) < 0:
        parser.error("memory_safety_factor 必须 >= 1；reserve_fraction 应在 [0,1)；保留 GiB >= 0")
    if not 0 <= args.label_smoothing < 1 or args.weight_decay < 0 or args.clip_grad < 0:
        parser.error("非法优化器参数")
    return args


SEMANTIC_KEYS = ("epochs", "patience", "lr", "batch_size", "raw_signal_len", "dwell_divisor", "quality_divisor", "accum_steps", "weight_decay", "label_smoothing", "clip_grad", "monitor", "model_dim", "num_blocks", "state_dim", "litemamba_variant", "readout", "bootstrap", "no_plots", "no_embeddings", "embedding_dtype", "no_epoch_logits", "save_epoch_predictions", "staged_exports")
SOURCE_FILES = ("train_all.py", "model_benchmark_all.py", "config_6_all_6.py", "IterDataset.py", "analyze_results.py", "run_all_4_models.py", "resource_scheduler.py", "cuda_device_probe.py")


def build_requested_config(args, model, seed, scheme, input_fingerprints=None):
    config = {key: getattr(args, key) for key in SEMANTIC_KEYS}
    config.update(data_dir=str(Path(args.data_dir).expanduser().resolve()), model_type=model, seed=seed, feature_scheme=scheme,
                  source_sha256={name: sha256_file(HERE / name) for name in SOURCE_FILES if (HERE / name).is_file()},
                  input_fingerprints=input_fingerprints,
                  external_manifest=str(Path(args.external_manifest).expanduser().resolve()) if args.external_manifest else None)
    if model in STRUCTURE_SETTINGS:
        if not getattr(args, "structure_diagnostic", False):
            raise ValueError("结构别名缺少固定诊断开关")
        config.update(model_type="litemamba", structure_experiment=model)
        config["litemamba_variant"], config["readout"] = STRUCTURE_SETTINGS[model]
    if args.external_manifest:
        config["external_manifest_sha256"] = sha256_file(config["external_manifest"])
    return config


def build_command(config, run_dir, mode, args, device="cpu", resume=False, restart=False):
    command = [sys.executable, "-u", str(TRAINER), "--mode", mode]
    keys = ("data_dir", "model_type", "seed", "feature_scheme", "epochs", "patience", "lr", "batch_size", "raw_signal_len", "dwell_divisor", "quality_divisor", "accum_steps", "weight_decay", "label_smoothing", "clip_grad", "monitor", "model_dim", "num_blocks", "state_dim", "litemamba_variant", "readout", "bootstrap", "embedding_dtype")
    for key in keys:
        command += ["--" + key, str(config[key])]
    command += ["--output_dir", str(run_dir), "--cache_dir", str(args.cache_dir), "--device", device,
                "--cpu_threads", str(args.cpu_threads), "--workers", str(args.workers)]
    for key in ("no_plots", "no_embeddings", "no_epoch_logits", "save_epoch_predictions"):
        if config[key]:
            command.append("--" + key)
    if mode == "train":
        command += ["--defer_test", "--concurrent_training"]
        if args.staged_exports:
            command.append("--defer_train_export")
        if resume:
            command.append("--resume")
        elif restart:
            command.append("--restart_failed_before_checkpoint")
    if mode == "finalize":
        command.append("--defer_test")
    if mode == "profile":
        command += ["--profile_steps", str(args.profile_steps)]
    if mode == "evaluate" and config.get("external_manifest"):
        command += ["--external_manifest", config["external_manifest"]]
    return command


def results_status(run_dir):
    path = Path(run_dir) / "results.json"
    return load_json(path).get("status") if path.is_file() else None


def find_checkpoint(run_dir, kind):
    paths = [Path(run_dir) / f"{kind}_checkpoint.pt", Path(run_dir) / "checkpoints" / f"{kind}_checkpoint.pt", Path(run_dir) / "checkpoints" / f"{kind}.pt"]
    return next((path for path in paths if path.is_file()), None)


def check_existing(item):
    path = Path(item["run_dir"]) / "queue_requested_config.json"
    if path.is_file():
        if load_json(path) != item["requested_config"]:
            raise RuntimeError(f"已有运行的配置/数据/源代码不同：{item['run_dir']}。请使用新输出目录。")
        return True
    directory = Path(item["run_dir"])
    if directory.exists() and any(directory.iterdir()):
        raise RuntimeError(f"目录已有内容但缺少队列配置：{directory}。请使用新目录。")
    return False


def child_environment(args):
    # Per-process BLAS/OpenMP caps prevent concurrent processes from silently
    # each requesting all server CPU threads. DataLoader workers inherit caps.
    env = dict(os.environ, PYTHONUNBUFFERED="1", OMP_NUM_THREADS=str(args.cpu_threads),
               MKL_NUM_THREADS=str(args.cpu_threads), OPENBLAS_NUM_THREADS=str(args.cpu_threads),
               NUMEXPR_NUM_THREADS=str(args.cpu_threads))
    return env


@contextmanager
def queue_output_lock(output):
    """Lock before changing any queue log, plan or PID information."""
    path = Path(output) / ".queue.lock"
    handle = path.open("a+", encoding="utf-8")
    try:
        try:
            fcntl.flock(handle.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError as exc:
            raise RuntimeError(f"此输出目录已有运行中的队列；未覆盖日志或 PID：{output}") from exc
        handle.seek(0)
        handle.truncate()
        handle.write(json.dumps({"pid": os.getpid(), "started_at": utc_now()}))
        handle.flush()
        yield
    finally:
        handle.close()


def record_interruption(item, process, reason):
    """Prove an ended queue child, without forging a trainer failed result."""
    run_dir = Path(item["run_dir"])
    config = run_dir / "queue_requested_config.json"
    if not config.is_file() or process.poll() is None or process.returncode == 0:
        return
    save_json(run_dir / "queue_interruption.json", {
        "version": "queue-interruption-v1", "run_id": item["run_id"],
        "run_dir": str(run_dir.resolve()),
        "queue_requested_config_sha256": sha256_file(config),
        "ended_pid": process.pid, "returncode": process.returncode,
        "reason": reason, "finished_at": utc_now(),
        "no_test_started": not (run_dir / "test_started.json").exists(),
    })


def process_start_ticks(pid):
    try:
        directory = proc_directory(pid)
        if directory is None:
            return None
        fields = (directory / "stat").read_text().rsplit(") ", 1)[1].split()
        return None if fields[0] == "Z" else int(fields[19])
    except (OSError, ValueError, IndexError):
        return None


def capture_descendants(pid):
    owned, pending = {}, [int(pid)]
    while pending:
        current = pending.pop()
        for child in process_children(current):
            ticks = process_start_ticks(child)
            if ticks is not None and child not in owned:
                owned[child] = ticks
                pending.append(child)
    return owned


def signal_owned_process(pid, ticks, sig):
    """Use pidfd when available so PID reuse cannot target another job."""
    descriptor = None
    try:
        if hasattr(os, "pidfd_open") and hasattr(signal, "pidfd_send_signal"):
            descriptor = os.pidfd_open(pid)
        if process_start_ticks(pid) != ticks:
            return
        if descriptor is not None:
            signal.pidfd_send_signal(descriptor, sig)
        else:
            os.kill(pid, sig)
    except ProcessLookupError:
        pass
    finally:
        if descriptor is not None:
            os.close(descriptor)


def cleanup_descendants(owned):
    live = {pid: ticks for pid, ticks in owned.items() if process_start_ticks(pid) == ticks}
    for pid, ticks in live.items():
        signal_owned_process(pid, ticks, signal.SIGTERM)
    if live:
        time.sleep(0.05)
    for pid, ticks in live.items():
        signal_owned_process(pid, ticks, signal.SIGKILL)


def assert_no_live_queue_children(commands_path):
    """SIGKILL of a queue must not allow an orphan to be trained twice."""
    path = Path(commands_path)
    if not path.is_file():
        return
    for line in path.read_text(encoding="utf-8").splitlines():
        if not line.strip():
            continue
        entry = json.loads(line)
        if entry.get("event") != "queue_child_started":
            continue
        pid, ticks = entry.get("pid"), entry.get("process_start_ticks")
        if ticks is not None and process_start_ticks(pid) == ticks:
            raise RuntimeError(f"原队列仍有活动子进程 PID {pid}（{entry.get('run_id')}）；请先等待其退出或核实并停止，再恢复队列。")


def terminate_process(process):
    if process.poll() is not None:
        return
    try:
        os.killpg(process.pid, signal.SIGTERM)
    except ProcessLookupError:
        return
    try:
        process.wait(timeout=10)
    except subprocess.TimeoutExpired:
        try:
            os.killpg(process.pid, signal.SIGKILL)
        except ProcessLookupError:
            pass
        process.wait()


def launch(command, log_path, commands_path, item, stage, args, attempt=1):
    log_path = Path(log_path)
    invocation = getattr(args, "_invocation_id", None)
    if invocation:
        log_path = log_path.with_name(log_path.stem + "_" + invocation + log_path.suffix)
    record = {"started_at": utc_now(), "run_id": item["run_id"], "stage": stage, "attempt": attempt,
              "argv": command, "command": shlex.join(command), "log": str(log_path)}
    append_json(commands_path, record)
    Path(log_path).parent.mkdir(parents=True, exist_ok=True)
    log = Path(log_path).open("a", encoding="utf-8", buffering=1)
    log.write("\n" + json.dumps(record, ensure_ascii=False) + "\n")
    log.flush()
    try:
        process = subprocess.Popen(command, cwd=str(HERE), stdout=log, stderr=subprocess.STDOUT,
                                   env=child_environment(args), start_new_session=True)
    except BaseException:
        log.close()
        raise
    append_json(commands_path, {"event": "queue_child_started", "pid": process.pid,
                "process_start_ticks": process_start_ticks(process.pid), **record})
    print(f"[{stage.upper()}] {item['run_id']} pid={process.pid} log={log_path}", flush=True)
    return {"process": process, "log": log, "started": time.monotonic(), "record": record,
            "known_descendants": capture_descendants(process.pid)}


def execute_serial(command, log_path, commands_path, item, stage, args):
    active = launch(command, log_path, commands_path, item, stage, args)
    try:
        while active["process"].poll() is None:
            active["known_descendants"].update(capture_descendants(active["process"].pid))
            time.sleep(min(args.poll_seconds, 2.0))
        return {"returncode": active["process"].returncode, "elapsed_seconds": time.monotonic() - active["started"], "finished_at": utc_now(), "log": active["record"]["log"]}
    except BaseException:
        terminate_process(active["process"])
        raise
    finally:
        cleanup_descendants(active["known_descendants"])
        active["log"].close()


def canonical_gpu_uuid(value):
    """Normalize actual UUIDs, never physical GPU ordinals or device names."""
    value = str(value or "").strip()
    if value.lower().startswith("gpu-"):
        value = value[4:]
    try:
        return str(uuid_module.UUID(value))
    except (ValueError, AttributeError, TypeError):
        return None


def match_cuda_devices(observed, snapshot, requested, visible=None):
    """Match actual PyTorch-context UUIDs to the driver's memory telemetry.

    CUDA_VISIBLE_DEVICES can reorder logical devices. A numeric index, GPU
    name or amount of RAM cannot prove that mapping. Missing identity permits
    only an unambiguous, explicitly UUID-based visibility list as a fallback.
    """
    devices = observed["devices"]
    if len(set(requested)) != len(requested):
        raise ValueError("gpu_ids 不允许重复")
    if visible is None:
        visible = os.environ.get("CUDA_VISIBLE_DEVICES", "")
    tokens = [token.strip() for token in visible.split(",")]
    mapping, evidence = {}, []
    for logical in requested:
        if logical < 0 or logical >= len(devices) or devices[logical].get("logical") != logical:
            raise ValueError(f"无 CUDA 可见逻辑 GPU {logical}")
        item = devices[logical]
        if (item.get("unsupported_mig") or str(item.get("uuid", "")).upper().startswith("MIG-")
                or any(token.upper().startswith("MIG-") for token in tokens)):
            raise RuntimeError("自动共享 GPU 调度不支持 MIG 切片；不能使用父 GPU 的显存容量作为切片预算。")
        value = canonical_gpu_uuid(item.get("uuid"))
        if item.get("uuid") and value is None:
            raise RuntimeError(f"CUDA 设备 {logical} 返回了无效 UUID；拒绝猜测设备映射。")
        if value is not None:
            candidates = [key for key in snapshot["devices"] if canonical_gpu_uuid(key) == value]
            if len(candidates) != 1:
                raise RuntimeError(f"cuda:{logical} 的实际 UUID 不对应唯一完整 GPU；可能是 MIG/遥测不可见或设备环境不一致。")
            method = item.get("identity_method", "torch_device_properties_uuid")
        else:
            if len(tokens) != len(devices) or not all(token.upper().startswith("GPU-") for token in tokens):
                errors = item.get("identity_notes", item.get("identity_errors", []))
                raise RuntimeError("无法从 PyTorch/CUDA 驱动核实设备 UUID。请检查 libcuda.so.1，或设置明确的 CUDA_VISIBLE_DEVICES=GPU-实际UUID。探测信息：" + str(errors))
            token = tokens[logical].lower()
            candidates = [key for key in snapshot["devices"] if key.lower().startswith(token)]
            if len(candidates) != 1:
                raise RuntimeError("显式 CUDA_VISIBLE_DEVICES UUID 缺失或不唯一；拒绝猜测设备映射。")
            method = "explicit_uuid_visibility"
        mapping[logical] = candidates[0]
        evidence.append({"logical": logical, "uuid": candidates[0], "identity_method": method})
    if not mapping or len(set(mapping.values())) != len(mapping):
        raise RuntimeError("没有可用的、彼此独立的完整 GPU 身份；拒绝重复预算。")
    return mapping, evidence


def discover_devices(args):
    if args.device == "cpu":
        if args.gpu_ids:
            raise ValueError("--device cpu 不应同时指定 --gpu_ids")
        return {}, {"mode": "cpu", "reason": "explicit_cpu"}
    # A child discovers actual CUDA logical UUIDs. No parent CUDA context is
    # created, and CUDA_VISIBLE_DEVICES ordering is verified rather than guessed.
    probe_script = HERE / "cuda_device_probe.py"
    if not probe_script.is_file():
        raise FileNotFoundError("请完整替换补丁，缺少设备探测文件：" + str(probe_script))
    probe = subprocess.run([sys.executable, str(probe_script)], check=True, capture_output=True, text=True, timeout=60, env=child_environment(args))
    observed = json.loads(probe.stdout.strip().splitlines()[-1])
    if not observed["available"]:
        if args.device != "auto" or args.gpu_ids:
            raise RuntimeError("请求了 CUDA，但当前 Python/PyTorch 无可用 CUDA。")
        return {}, {"mode": "cpu", "reason": "torch_cuda_unavailable", "cuda_probe": observed}
    snapshot = nvidia_snapshot()  # Missing/unsupported telemetry fails closed.
    requested = args.gpu_ids
    if args.device.startswith("cuda:"):
        index = int(args.device.split(":", 1)[1])
        if requested and requested != [index]:
            raise ValueError("--device cuda:N 与 --gpu_ids 冲突")
        requested = [index]
    elif args.device not in ("auto", "cuda"):
        raise ValueError("device 必须为 auto/cpu/cuda/cuda:N")
    requested = list(range(len(observed["devices"]))) if requested is None else requested
    mapping, evidence = match_cuda_devices(observed, snapshot, requested)
    for item in evidence:
        print(f"[GPU] cuda:{item['logical']} -> {item['uuid']} ({item['identity_method']})", flush=True)
    return mapping, {"mode": "cuda", "logical_to_uuid": mapping, "mapping_evidence": evidence,
                     "cuda_probe": observed, "initial_snapshot": snapshot}


def prepare_queue_plan(plan_path, requested):
    """Archive a code-only hotfix plan only before any training has started."""
    plan_path = Path(plan_path)
    if not plan_path.exists() or load_json(plan_path) == requested:
        return
    previous, output = load_json(plan_path), plan_path.parent
    old_runs, new_runs = previous.get("experiments", []), requested.get("experiments", [])
    allowed_source_changes = {"run_all_4_models.py", "train_all.py", "cuda_device_probe.py"}
    if len(old_runs) != len(new_runs) or not old_runs:
        raise RuntimeError("原队列实验已不同；请使用新输出目录。")
    for old, new in zip(old_runs, new_runs):
        old_config, new_config = dict(old.get("requested_config", {})), dict(new.get("requested_config", {}))
        old_sources, new_sources = old_config.pop("source_sha256", {}), new_config.pop("source_sha256", {})
        if (old.get("run_id") != new.get("run_id") or old_config != new_config
                or any(old_sources.get(name) != new_sources.get(name)
                       for name in set(old_sources) | set(new_sources) if name not in allowed_source_changes)):
            raise RuntimeError("原队列的数据、训练参数、模型或分析源码已不同；请使用新输出目录。")
    allowed_entries = {"preflight", "feature_cache", "resource_profiles", "preflight_plan_history",
                       "queue_plan.json", "commands.jsonl", "queue_manifest.json", "resource_events.jsonl",
                       "queue_invocations.jsonl", "dry_run_manifest.json"}
    if any(path.name not in allowed_entries for path in output.iterdir()):
        raise RuntimeError("旧队列已有训练/测试目录或其他产物；补丁不会改写其冻结计划，请使用新输出目录。")
    commands = output / "commands.jsonl"
    if commands.exists():
        for line in commands.read_text(encoding="utf-8").splitlines():
            if line.strip() and json.loads(line).get("stage") not in ("preflight", "profile"):
                raise RuntimeError("旧队列已经启动过训练/评价；不能自动更新代码计划。")
    manifest = output / "queue_manifest.json"
    if manifest.exists():
        for item in load_json(manifest).get("experiments", []):
            if item.get("attempts") or item.get("train_queue_status") not in (None, "pending"):
                raise RuntimeError("旧队列有训练尝试记录；不能自动更新代码计划。")
    history = output / "preflight_plan_history"
    history.mkdir(exist_ok=True)
    stamp = datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S_%f")
    archived = history / ("queue_plan_" + stamp + ".json")
    shutil.copy2(plan_path, archived)
    save_json(history / ("refresh_" + stamp + ".json"), {
        "reason": "v1.0.1 device-identity hotfix; verified no training/evaluation started",
        "archived_plan_sha256": sha256_file(archived), "archived_plan": str(archived),
        "created_at": utc_now(), "queue_version": QUEUE_VERSION})
    print("[恢复] 仅预检的旧计划已归档；使用设备兼容补丁重新生成计划，复用特征缓存。", flush=True)


def freeze_evaluation_plan(items, output_dir):
    experiments = []
    for item in items:
        if results_status(item["run_dir"]) not in ("train_complete", "complete"):
            raise RuntimeError(f"禁止测试：训练未成功完成 {item['run_id']}")
        checkpoint = find_checkpoint(item["run_dir"], "best")
        if checkpoint is None:
            raise RuntimeError(f"缺少验证集选出的 checkpoint：{item['run_id']}")
        experiments.append({"run_id": item["run_id"], "run_dir": item["run_dir"], "requested_config": item["requested_config"],
                            "checkpoint": str(checkpoint), "checkpoint_sha256": sha256_file(checkpoint)})
    frozen = {"policy": "all_predeclared_training_succeeded_before_any_test;validation_only_selection", "experiments": experiments}
    path = Path(output_dir) / "evaluation_plan.json"
    if path.exists():
        previous = load_json(path)
        previous.pop("frozen_at", None)
        if previous != frozen:
            raise RuntimeError("已冻结测试计划不同；请用新输出目录，不能在看过测试后修改当前队列。")
    else:
        save_json(path, {"frozen_at": utc_now(), **frozen})
    return path


def verify_frozen_checkpoint(item, frozen):
    entry = next(value for value in frozen["experiments"] if value["run_id"] == item["run_id"])
    if sha256_file(entry["checkpoint"]) != entry["checkpoint_sha256"]:
        raise RuntimeError(f"冻结后 checkpoint 改变：{item['run_id']}")


def run_training_queue(items, profiles, controller, args, manifest, manifest_path, commands_path):
    pending, failures, active = [], [], {}
    complete_statuses = ("weights_complete", "train_complete", "complete") if args.staged_exports else ("train_complete", "complete")
    for item in items:
        if results_status(item["run_dir"]) in complete_statuses:
            if find_checkpoint(item["run_dir"], "best") is None:
                raise RuntimeError(f"完成状态缺少 best checkpoint：{item['run_dir']}")
            item["train_queue_status"] = "skipped_matching_completed"
        else:
            known = check_existing(item)
            if known and not args.resume_queue:
                raise RuntimeError(f"存在中断运行；请核查后用 --resume_queue：{item['run_dir']}")
            if known and (Path(item["run_dir"]) / "test_started.json").exists():
                raise RuntimeError("已启动测试的运行不能恢复训练：" + item["run_dir"])
            item["attempts"] = item.get("attempts", [])
            item["oom_retries_used"] = 0
            pending.append(item)
    idle_since, last_report, last_resource_log = None, 0.0, 0.0
    try:
        while pending or active:
            changed = False
            for run_id, process_info in list(active.items()):
                process = process_info["process"]
                if process.poll() is None:
                    process_info.setdefault("known_descendants", {}).update(capture_descendants(process.pid))
                    continue
                cleanup_descendants(process_info.get("known_descendants", {}))
                process_info["log"].close()
                item = process_info["item"]
                controller.release(run_id)
                del active[run_id]
                outcome = {"returncode": process.returncode, "elapsed_seconds": time.monotonic() - process_info["started"],
                           "finished_at": utc_now(), "device": process_info["device"], "log": process_info["record"]["log"]}
                item["attempts"].append(outcome)
                ok = process.returncode == 0 and results_status(item["run_dir"]) in complete_statuses
                if ok:
                    item["train_queue_status"] = "complete"
                    controller.record("job_complete", run_id=run_id, outcome=outcome)
                else:
                    oom = classify_oom(outcome["log"])
                    suspected_resource = process.returncode == -signal.SIGKILL and not oom
                    record_interruption(item, process, "oom" if oom else
                                        "sigkill_resource_suspected" if suspected_resource else "training_process_failed")
                    # Retry only before test, with unchanged scientific settings.
                    # Wait for other processes to drain before the retry starts.
                    if (oom or suspected_resource) and item["oom_retries_used"] < args.oom_retries:
                        item["oom_retries_used"] += 1
                        item["drain_before_retry"] = True
                        item["train_queue_status"] = "oom_waiting_retry" if oom else "suspected_resource_waiting_retry"
                        controller.reduce_after_oom("confirmed_oom" if oom else "sigkill_resource_suspected")
                        pending.insert(0, item)
                        controller.record("oom_retry_pending" if oom else "resource_suspected_retry_pending",
                                          run_id=run_id, outcome=outcome, retry=item["oom_retries_used"],
                                          confirmed_oom=oom, diagnosis="confirmed_oom" if oom else "SIGKILL_cause_unknown")
                    else:
                        item["train_queue_status"] = "failed"
                        failures.append(run_id)
                        controller.record("job_failed", run_id=run_id, oom=oom,
                                          suspected_resource=suspected_resource, outcome=outcome)
                        if not args.continue_on_error:
                            manifest.update(failed_runs=failures, evaluation_status="blocked_by_training_failures")
                            return failures
                changed = True
            snapshot = controller.snapshot()
            if time.monotonic() - last_resource_log >= max(5.0, args.poll_seconds):
                controller.record("resource_snapshot", snapshot=snapshot, active_jobs=list(active), pending_jobs=len(pending))
                last_resource_log = time.monotonic()
            # Admission has no blocking waits; current jobs continue to be polled.
            rejected = []
            for item in list(pending):
                if item.get("drain_before_retry") and active:
                    break
                profile = profiles[item["profile_key"]]
                request = (training_request_from_profile(profile, args.memory_safety_factor, args.cpu_threads,
                                args.workers, args.train_ram_extra_gib) if args.staged_exports
                           else request_from_profile(profile, args.memory_safety_factor, args.cpu_threads, args.workers))
                choice, reasons = controller.choose(request, snapshot)
                if choice is None:
                    rejected.append({"run_id": item["run_id"], "reasons": reasons})
                    continue
                run_dir = Path(item["run_dir"])
                run_dir.mkdir(parents=True, exist_ok=True)
                save_json(run_dir / "queue_requested_config.json", item["requested_config"])
                checkpoint = find_checkpoint(run_dir, "last")
                # Failed attempts before a first checkpoint may be restarted in
                # place only if no training checkpoint/results/config exists.
                resume = checkpoint is not None
                restart = not resume and (results_status(run_dir) == "failed" or
                                          (run_dir / "queue_interruption.json").is_file())
                attempt = len(item["attempts"]) + 1
                command = build_command(item["requested_config"], run_dir, "train", args, choice["device"], resume, restart)
                log_path = run_dir / f"stdout_train_attempt_{attempt:02d}.log"
                process_info = launch(command, log_path, commands_path, item, "train", args, attempt)
                process_info.update(item=item, device=choice["device"])
                active[item["run_id"]] = process_info
                controller.reserve(item["run_id"], process_info["process"].pid, request, choice["logical_gpu"])
                item["train_queue_status"] = "running"
                item["runtime_device"] = choice["device"]
                item["resource_request"] = {"host_bytes": request.host_bytes, "gpu_bytes": request.gpu_bytes, "cpu_slots": request.cpu_slots}
                controller.record("job_launched", run_id=item["run_id"], admission=choice, attempt=attempt, concurrent_jobs=len(active))
                pending.remove(item)
                changed = True
                # Recompute available telemetry; reservations cover allocation lag.
                snapshot = controller.snapshot()
                if item.pop("drain_before_retry", False):
                    break
            if changed:
                save_json(manifest_path, manifest)
            if pending and not active:
                if idle_since is None:
                    idle_since = time.monotonic()
                if time.monotonic() - last_report >= 30:
                    controller.record("waiting_for_resources", rejected=rejected)
                    print(f"资源不足，等待中；剩余 {len(pending)} 个作业。详情：{controller.event_path}", flush=True)
                    last_report = time.monotonic()
                if time.monotonic() - idle_since > args.resource_wait_seconds:
                    controller.record("queue_resource_wait_timeout", pending_jobs=[item["run_id"] for item in pending])
                    raise RuntimeError("资源等待超时；请核查 resource_events.jsonl，可在保持训练设置不变时调整资源预算后 --resume_queue。")
            else:
                idle_since = None
            if pending or active:
                time.sleep(min(args.poll_seconds, 5.0))
    finally:
        for run_id, process_info in list(active.items()):
            process_info.setdefault("known_descendants", {}).update(capture_descendants(process_info["process"].pid))
            terminate_process(process_info["process"])
            cleanup_descendants(process_info["known_descendants"])
            process_info["log"].close()
            item = process_info["item"]
            outcome = {"returncode": process_info["process"].returncode,
                       "elapsed_seconds": time.monotonic() - process_info["started"],
                       "finished_at": utc_now(), "device": process_info["device"],
                       "log": process_info["record"]["log"], "interrupted_by_queue": True}
            item["attempts"].append(outcome)
            item["train_queue_status"] = "interrupted"
            record_interruption(item, process_info["process"], "queue_interrupted_or_other_job_failed")
            controller.record("job_terminated", run_id=run_id, returncode=process_info["process"].returncode)
            controller.release(run_id)
        manifest["failed_runs"] = failures
        save_json(manifest_path, manifest)
    return failures


def finalize_training_exports(items, args, manifest, manifest_path, commands_path):
    """Serial train/val export; no test is accessed until every finalization succeeds."""
    failures = []
    for item in items:
        status = results_status(item["run_dir"])
        if status in ("train_complete", "complete"):
            item["finalize_status"] = "skipped_matching_completed"
            continue
        if status != "weights_complete":
            failures.append(item["run_id"] + ":not_weights_complete")
            if not args.continue_on_error:
                break
            continue
        run_dir = Path(item["run_dir"])
        if not find_checkpoint(run_dir, "best"):
            failures.append(item["run_id"] + ":best_checkpoint_missing")
            if not args.continue_on_error:
                break
            continue
        outcome = execute_serial(build_command(item["requested_config"], run_dir, "finalize", args,
                                 "cuda:" + str(next(iter(args._gpu_mapping))) if args._gpu_mapping else "cpu"),
                                 run_dir / "stdout_finalize.log", commands_path, item, "finalize", args)
        item["finalize_process"] = outcome
        ok = outcome["returncode"] == 0 and results_status(run_dir) == "train_complete"
        item["finalize_status"] = "complete" if ok else "failed"
        save_json(manifest_path, manifest)
        if not ok:
            failures.append(item["run_id"] + ":finalize")
            if not args.continue_on_error:
                break
    return failures


def _main_locked(args, output, data):
    args.cache_dir = str(Path(args.cache_dir).expanduser().resolve()) if args.cache_dir else str(output / "feature_cache")
    if output == data or output in data.parents:
        raise ValueError("output_dir 不能是数据目录或其父目录")
    commands = output / "commands.jsonl"
    assert_no_live_queue_children(commands)
    # No arbitrary external train_script/extra_args: only sibling versioned code.
    base = build_requested_config(args, args.models[0], args.seeds[0], args.feature_schemes[0])
    if args.dry_run:
        items = []
        combinations = ((scheme, model, seed)
                        for scheme in args.feature_schemes
                        for seed in args.seeds
                        for model in args.models) if args.staged_exports else (
                        (scheme, model, seed)
                        for scheme in args.feature_schemes
                        for model in args.models
                        for seed in args.seeds)
        for scheme, model, seed in combinations:
            run_id = f"{scheme}/{model}/seed_{seed}"
            config = build_requested_config(args, model, seed, scheme)
            item = {"run_id": run_id, "run_dir": str(output / run_id), "requested_config": config}
            items.append(item)
            print(shlex.join(build_command(config, item["run_dir"], "train", args, args.device)))
        save_json(output / "dry_run_manifest.json", {"queue_version": QUEUE_VERSION, "experiments": items, "warning": "No data read, hardware probe, training, evaluation or source publication performed."})
        return 0
    if not TRAINER.is_file() or not data.is_dir():
        raise FileNotFoundError(f"缺少完整代码或数据目录：{TRAINER}；{data}")
    for name in ("train.csv", "validation.csv", "test.csv", "train_meta.tsv", "validation_meta.tsv", "test_meta.tsv"):
        if not (data / name).is_file():
            raise FileNotFoundError(data / name)
    old_manifest_path = output / "queue_manifest.json"
    if old_manifest_path.exists() and not args.resume_queue and args.phase != "evaluate":
        raise RuntimeError("此输出目录已有队列，请核查后加 --resume_queue 或换新目录。")
    preflight_item = {"run_id": "preflight"}
    preflight_dir = output / "preflight"
    outcome = execute_serial(build_command(base, preflight_dir, "preflight", args, "cpu"), preflight_dir / "stdout.log", commands, preflight_item, "preflight", args)
    if outcome["returncode"]:
        raise RuntimeError(f"数据预检失败：{outcome['log']}")
    # Trainer provides content SHA fingerprints (not merely mtime/size).
    preflight_file = preflight_dir / "preflight.json"
    if not preflight_file.is_file():
        raise RuntimeError("预检没有生成 preflight.json；代码包版本可能不一致。")
    preflight = load_json(preflight_file)
    fingerprints = preflight["input_fingerprints"]
    # 阶段化训练按 seed -> model 交错排队，避免最开始同时启动多个 BiGRU，
    # 使已验证的混合架构并发模式与正式训练的首批任务一致。实验组合不变。
    items = []
    if args.staged_exports:
        combinations = ((scheme, model, seed)
                        for scheme in args.feature_schemes
                        for seed in args.seeds
                        for model in args.models)
    else:
        combinations = ((scheme, model, seed)
                        for scheme in args.feature_schemes
                        for model in args.models
                        for seed in args.seeds)
    for scheme, model, seed in combinations:
        run_id = f"{scheme}/{model}/seed_{seed}"
        items.append({"run_id": run_id, "run_dir": str(output / run_id), "profile_key": f"{scheme}/{model}",
                      "requested_config": build_requested_config(args, model, seed, scheme, fingerprints)})
    queue_plan = {"queue_version": QUEUE_VERSION, "experiments": [{"run_id": item["run_id"], "requested_config": item["requested_config"]} for item in items]}
    plan_path = output / "queue_plan.json"
    prepare_queue_plan(plan_path, queue_plan)
    if args.phase == "evaluate" and not plan_path.exists():
        raise RuntimeError("evaluate 必须使用本运行器已经完成训练的队列。")
    save_json(plan_path, queue_plan)
    for item in items:
        known = check_existing(item)
        if args.phase == "evaluate" and not known:
            raise RuntimeError("评价阶段缺少已有运行配置：" + item["run_dir"])
    mapping, hardware = discover_devices(args)
    args._gpu_mapping = mapping
    controller = ResourceController(args.max_parallel, args.cpu_budget, args.cpu_parallel_limit, args.gpu_reserve_gib,
                                    args.ram_reserve_gib, args.reserve_fraction, mapping, output / "resource_events.jsonl")
    resource_state = output / "resource_runtime_state.json"
    if args.resume_queue and resource_state.is_file():
        previous_cap = load_json(resource_state).get("max_parallel")
        if isinstance(previous_cap, int) and previous_cap > 0:
            controller.max_parallel = min(controller.max_parallel, previous_cap)
            controller.cpu_parallel_limit = min(controller.cpu_parallel_limit, controller.max_parallel)
            controller.record("resumed_resource_limit", requested_max_parallel=args.max_parallel,
                              effective_max_parallel=controller.max_parallel)
    if controller.cpu_budget < args.cpu_threads + args.workers:
        raise RuntimeError("可用 CPU 预算小于单作业 cpu_threads + workers；请降低这两个运行时参数。")
    runtime = {key: getattr(args, key) for key in ("workers", "cpu_threads", "device", "gpu_ids", "max_parallel", "cpu_parallel_limit", "cpu_budget", "gpu_reserve_gib", "ram_reserve_gib", "reserve_fraction", "memory_safety_factor", "profile_steps", "oom_retries", "staged_exports", "train_ram_extra_gib")}
    if old_manifest_path.exists():
        old_manifest = load_json(old_manifest_path)
        history = output / "queue_manifest_history"
        save_json(history / (args._invocation_id + ".json"), old_manifest)
        previous_items = {item["run_id"]: item for item in old_manifest.get("experiments", [])}
        for item in items:
            previous = previous_items.get(item["run_id"], {})
            if previous.get("requested_config") == item["requested_config"]:
                item["attempts"] = list(previous.get("attempts", []))
    manifest = {"queue_version": QUEUE_VERSION, "invocation_id": args._invocation_id,
                "started_at": utc_now(), "phase": args.phase, "hardware": hardware,
                "runtime": runtime, "experiments": items, "policy": "train_all_defer_test_then_freeze_all_then_evaluate", "status": "running"}
    save_json(old_manifest_path, manifest)
    append_json(output / "queue_invocations.jsonl", {"started_at": utc_now(), "argv": sys.argv, "runtime": runtime, "phase": args.phase})
    print(f"队列 {len(items)} 个实验；并发上限 {args.max_parallel}，按实测资源决定实际并发。所有训练完成后才测试。", flush=True)
    failures = []
    device = "cpu" if not mapping else "cuda:" + str(next(iter(mapping)))
    if args.phase in ("all", "train"):
        profiles = {}
        for item in items:
            key = item["profile_key"]
            if key in profiles or results_status(item["run_dir"]) in ("train_complete", "complete"):
                continue
            config = item["requested_config"]
            # Every invocation profiles in isolation, before launching jobs. It
            # measures the current server/PyTorch/batch/worker settings anew.
            device_profiles = []
            for profile_device in (["cuda:" + str(logical) for logical in mapping] if mapping else ["cpu"]):
                profile_dir = output / "resource_profiles" / key / profile_device.replace(":", "_")
                outcome = execute_serial(build_command(config, profile_dir, "profile", args, profile_device), profile_dir / "stdout.log", commands, {"run_id": "profile/" + key + "/" + profile_device}, "profile", args)
                if outcome["returncode"]:
                    raise RuntimeError(f"资源实测失败（未开始并发或测试）：{outcome['log']}")
                device_profiles.append(load_json(profile_dir / "resource_profile.json"))
            profile = dict(device_profiles[0])
            for metric in ("gpu_admission_estimate_bytes", "host_admission_estimate_bytes", "gpu_peak_reserved_bytes", "gpu_context_baseline_bytes", "gpu_driver_delta_bytes", "host_peak_rss_bytes",
                           "host_process_tree_peak_sampled_rss_bytes", "host_epoch_metric_buffer_budget_bytes",
                           "host_inference_buffer_budget_bytes", "host_worker_prefetch_budget_bytes",
                           "host_readonly_cache_budget_bytes"):
                profile[metric] = max(int(value.get(metric) or 0) for value in device_profiles)
            profile["profiled_devices"] = [value.get("device") for value in device_profiles]
            request = (training_request_from_profile(profile, args.memory_safety_factor, args.cpu_threads,
                           args.workers, args.train_ram_extra_gib) if args.staged_exports
                       else request_from_profile(profile, args.memory_safety_factor, args.cpu_threads, args.workers))
            # Ensure at least one job fits now; allow waiting if another user is
            # temporarily occupying resources, using the same admission loop.
            profiles[key] = profile
            controller.record("profile_measured", profile_key=key, profile=profile,
                              reservation={"host_bytes": request.host_bytes, "gpu_bytes": request.gpu_bytes, "cpu_slots": request.cpu_slots})
        if args.staged_exports:
            # One-time cache reserve (the distinct train/val/test memmaps are shared).
            controller.shared_cache_budget_bytes = max(int(p.get("host_readonly_cache_budget_bytes") or 0)
                                                       for p in profiles.values()) if profiles else 0
            controller.record("staged_budget_configured", shared_cache_budget_bytes=controller.shared_cache_budget_bytes,
                              extra_host_gib=args.train_ram_extra_gib, safety_factor=args.memory_safety_factor)
            print(f"[STAGED] 共享只读缓存仅预留一次: {controller.shared_cache_budget_bytes / 1024**3:.2f} GiB；并行训练不预留导出分析缓冲区", flush=True)
        try:
            failures = run_training_queue(items, profiles, controller, args, manifest, old_manifest_path, commands)
        finally:
            save_json(resource_state, {"max_parallel": controller.max_parallel,
                      "cpu_parallel_limit": controller.cpu_parallel_limit, "updated_at": utc_now(),
                      "reason": "measured_admission_and_bounded_oom_backoff"})
            manifest["effective_max_parallel"] = controller.max_parallel
            save_json(old_manifest_path, manifest)
        if not failures and args.staged_exports:
            print("[FINALIZE] 所有权重训练完成；开始逐个导出 train/validation，不进行 test 推理", flush=True)
            failures = finalize_training_exports(items, args, manifest, old_manifest_path, commands)
    if failures:
        manifest.update(status="failed", failed_runs=failures, evaluation_status="blocked_by_training_failures", finished_at=utc_now())
        save_json(old_manifest_path, manifest)
        print("至少一个训练失败，整队测试未启动。详见各运行日志和 resource_events.jsonl。", file=sys.stderr)
        return 1
    if args.phase in ("all", "evaluate"):
        plan = freeze_evaluation_plan(items, output)
        frozen = load_json(plan)
        manifest["evaluation_plan"] = str(plan)
        save_json(old_manifest_path, manifest)
        # Serial evaluation prevents bootstrap/plotting/embedding exports from
        # unexpectedly competing with a still-running trainer's GPU/RAM.
        for item in items:
            verify_frozen_checkpoint(item, frozen)
            if results_status(item["run_dir"]) == "complete":
                item["evaluation_queue_status"] = "skipped_matching_completed"
                continue
            run_dir = Path(item["run_dir"])
            outcome = execute_serial(build_command(item["requested_config"], run_dir, "evaluate", args, device), run_dir / "stdout_evaluate.log", commands, item, "evaluate", args)
            item["evaluation_process"] = outcome
            ok = outcome["returncode"] == 0 and results_status(run_dir) == "complete"
            item["evaluation_queue_status"] = "complete" if ok else "failed"
            save_json(old_manifest_path, manifest)
            if not ok:
                failures.append(item["run_id"])
                if not args.continue_on_error:
                    break
        if not failures and not args.no_isolated_benchmark:
            for item in items:
                verify_frozen_checkpoint(item, frozen)
                run_dir = Path(item["run_dir"])
                outcome = execute_serial(build_command(item["requested_config"], run_dir, "benchmark", args, device), run_dir / "stdout_benchmark.log", commands, item, "benchmark", args)
                item["isolated_benchmark_process"] = outcome
                if outcome["returncode"]:
                    failures.append(item["run_id"] + ":benchmark")
                    if not args.continue_on_error:
                        break
    usable = [item["run_dir"] for item in items if results_status(item["run_dir"]) in ("train_complete", "complete")]
    if usable and not args.structure_diagnostic:
        analyzer = importlib.import_module("analyze_results")
        analyzer.aggregate_runs(usable, output / "aggregate", split="val" if args.phase == "train" else "test")
    manifest.update(finished_at=utc_now(), failed_runs=failures, status="complete" if not failures else "failed")
    save_json(old_manifest_path, manifest)
    if args.structure_diagnostic:
        print(f"结构队列完成：{output}；各结构身份由独立汇总器比较，未进行通用跨结构聚合。", flush=True)
    else:
        print(f"队列完成：{output}；聚合结果：{output / 'aggregate'}", flush=True)
    return int(bool(failures))


def main(argv=None):
    args = parse_args(argv)
    output = Path(args.output_dir).expanduser().resolve()
    data = Path(args.data_dir).expanduser().resolve()
    if output == data or output in data.parents:
        raise ValueError("output_dir 不能是数据目录或其父目录")
    output.mkdir(parents=True, exist_ok=True)
    args._invocation_id = datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S_%f") + "_" + str(os.getpid())
    with queue_output_lock(output):
        try:
            return _main_locked(args, output, data)
        except BaseException as error:
            manifest_path = output / "queue_manifest.json"
            if manifest_path.is_file():
                manifest = load_json(manifest_path)
                if manifest.get("invocation_id") == args._invocation_id:
                    manifest.update(status="interrupted" if isinstance(error, KeyboardInterrupt) else "failed",
                                    queue_error=f"{type(error).__name__}: {error}", finished_at=utc_now())
                    save_json(manifest_path, manifest)
            raise


if __name__ == "__main__":
    def _stop_on_term(signum, frame):
        raise KeyboardInterrupt(f"收到信号 {signum}，安全终止子任务")
    signal.signal(signal.SIGTERM, _stop_on_term)
    try:
        raise SystemExit(main())
    except KeyboardInterrupt:
        print("已中断并终止本队列子进程；保持配置不变可用 --resume_queue 恢复。", file=sys.stderr)
        raise SystemExit(130)
    except Exception as exc:
        print(f"ERROR: {type(exc).__name__}: {exc}", file=sys.stderr)
        raise SystemExit(1)
