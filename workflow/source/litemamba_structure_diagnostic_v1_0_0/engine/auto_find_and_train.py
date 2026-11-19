#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""先串行实测四种模型，再按资源预算测试并发，最后启动 20 次正式实验。

只管理本程序创建的进程组。容量是当前资源下的短程上限，正式队列仍逐次准入。
不改变 batch size、seed 或科学设置，不把 SIGKILL 自动认定为 OOM。
"""
from __future__ import annotations

import argparse
from dataclasses import asdict
from datetime import datetime, timezone
import fcntl
import hashlib
import itertools
import json
import os
from pathlib import Path
import signal
import subprocess
import sys
import time
from types import SimpleNamespace

from resource_scheduler import (GIB, ResourceController, ResourceRequest, available_cpu_slots,
    classify_oom, memory_snapshot, nvidia_snapshot, process_rss_bytes, process_anonymous_rss_bytes, training_request_from_profile)
from run_all_4_models import discover_devices

HERE = Path(__file__).resolve().parent
DEFAULT_DATA = ("/path/to/revision-inputs/CPLX-XNA-replenishment/"
                "CPLX-XNA-replenishment/fast5_single_guppy/sixclass_overlap_read_split_721")
VERSION = "1.0.4"
CAPACITY_SCHEMA = "sixclass-capacity-v1.0.4"
MODEL_ORDER = ["transformer", "litemamba", "convmixer", "bigru"]
SEEDS = [42, 2024, 2025, 2026, 2027]
OWN_PROCESSES: dict[int, subprocess.Popen] = {}
LOCK_FILE = None


def now():
    return datetime.now(timezone.utc).isoformat()


def gib(value):
    return round(float(value or 0) / GIB, 3)


def save_json(path, obj):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temp = path.with_name(path.name + ".tmp")
    temp.write_text(json.dumps(obj, indent=2, ensure_ascii=False, allow_nan=False) + "\n", encoding="utf-8")
    temp.replace(path)


def tail_text(path, n=20):
    try:
        return "\n".join(Path(path).read_text(encoding="utf-8", errors="replace").splitlines()[-n:])
    except OSError:
        return ""


def child_env(threads):
    env = dict(os.environ, PYTHONUNBUFFERED="1")
    for key in ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS", "NUMEXPR_NUM_THREADS"):
        env[key] = str(threads)
    return env


def launch(command, log_path, threads):
    path = Path(log_path)
    path.parent.mkdir(parents=True, exist_ok=True)
    handle = path.open("a", encoding="utf-8", buffering=1)
    handle.write("\n>>> " + " ".join(command) + "\n")
    try:
        process = subprocess.Popen(command, cwd=HERE, env=child_env(threads), stdout=handle,
                                   stderr=subprocess.STDOUT, start_new_session=True)
        process._sixclass_leader_identity = process_identity(process.pid)
        OWN_PROCESSES[process.pid] = process
    except BaseException:
        handle.close()
        raise
    print(f"[LAUNCH] pid={process.pid} log={path}", flush=True)
    return process, handle


_PROC_PATH_CACHE = {}


def proc_directory(pid):
    """Resolve /proc correctly even when mounted from an ancestor PID namespace."""
    pid = int(pid)
    try:
        if int(Path("/proc/self").readlink().name) == os.getpid():
            path = Path(f"/proc/{pid}")
            return path if path.exists() else None
        namespace = Path("/proc/self/ns/pid").readlink()
        cached = _PROC_PATH_CACHE.get(pid)
        candidates = [cached] if cached else []
        candidates.extend(path for path in Path("/proc").iterdir() if path.name.isdigit() and path != cached)
        for path in candidates:
            try:
                if (path / "ns/pid").readlink() != namespace:
                    continue
                values = next(line.split()[1:] for line in (path / "status").read_text().splitlines()
                              if line.startswith("NSpid:"))
                if int(values[-1]) == pid:
                    _PROC_PATH_CACHE[pid] = path
                    return path
            except (OSError, StopIteration, ValueError):
                pass
    except OSError:
        pass
    return None


def process_identity(pid):
    try:
        directory = proc_directory(pid)
        if directory is None:
            return None
        # comm may contain spaces/parentheses; remaining fields start at state.
        fields = (directory / "stat").read_text().rsplit(")", 1)[1].split()
        return fields[19]
    except (OSError, IndexError):
        return None


def owned_tree(process):
    identities = dict(getattr(process, "_sixclass_descendants", {}))
    current_leader = process_identity(process.pid)
    expected = getattr(process, "_sixclass_leader_identity", None)
    if expected is None and process.poll() is None:
        expected = current_leader
        process._sixclass_leader_identity = expected
    pending = [process.pid] if expected is not None and current_leader == expected else []
    seen = set()
    while pending:
        pid = pending.pop()
        if pid in seen:
            continue
        seen.add(pid)
        identity = process_identity(pid)
        if identity is not None:
            identities[pid] = identity
        try:
            directory = proc_directory(pid)
            if directory is None:
                continue
            outer_pid = directory.name
            try:
                child_paths = [Path("/proc") / value for value in
                               (directory / f"task/{outer_pid}/children").read_text().split()]
            except OSError:
                # Some kernels omit task/*/children. Scan PPid only as fallback;
                # this still finds owned descendants that created new sessions.
                child_paths = []
                for candidate in Path("/proc").iterdir():
                    if not candidate.name.isdigit():
                        continue
                    try:
                        status_lines = (candidate / "status").read_text().splitlines()
                        parent = next(line.split()[1] for line in status_lines if line.startswith("PPid:"))
                        if parent == outer_pid:
                            child_paths.append(candidate)
                    except (OSError, StopIteration):
                        pass
            for child_path in child_paths:
                status = (child_path / "status").read_text().splitlines()
                ids = next(line.split()[1:] for line in status if line.startswith("NSpid:"))
                child_pid = int(ids[-1])
                _PROC_PATH_CACHE[child_pid] = child_path
                pending.append(child_pid)
        except (OSError, StopIteration, ValueError):
            pass
    process._sixclass_descendants = identities
    return identities


def stop_child(process, grace=15):
    # Queue workers can create their own sessions; killing only the queue's
    # process group would leave them alive. Capture only our process descendants
    # and verify /proc start ticks before sending each signal (PID-reuse guard).
    identities = owned_tree(process)
    # A live unreaped Popen PID cannot have been recycled, even if /proc is
    # unavailable. Signal that root directly as a safe fallback.
    if process.poll() is None:
        try:
            os.kill(process.pid, signal.SIGTERM)
        except ProcessLookupError:
            pass
    for pid, identity in identities.items():
        if process_identity(pid) == identity:
            try:
                os.kill(pid, signal.SIGTERM)
            except ProcessLookupError:
                pass
    if process.poll() is None:
        try:
            process.wait(timeout=grace)
        except subprocess.TimeoutExpired:
            pass
    if process.poll() is None:
        try:
            os.kill(process.pid, signal.SIGKILL)
        except ProcessLookupError:
            pass
    for pid, identity in identities.items():
        if process_identity(pid) == identity:
            try:
                os.kill(pid, signal.SIGKILL)
            except ProcessLookupError:
                pass
    process.wait()


def stop_all():
    for process in list(OWN_PROCESSES.values()):
        stop_child(process)


def on_signal(signum, _frame):
    raise KeyboardInterrupt(f"收到信号 {signum}，停止本脚本管理的进程组")


def get_args():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    p.add_argument("--data_dir", default=DEFAULT_DATA)
    p.add_argument("--output_root", default=None)
    p.add_argument("--cache_dir")
    p.add_argument("--device", default="cuda:0")
    p.add_argument("--max_parallel", type=int, default=4)
    p.add_argument("--min_parallel", type=int, default=1)
    p.add_argument("--probe_steps", type=int, default=8)
    p.add_argument("--probe_min_seconds", type=float, default=15, help="并发探针在同步起点后至少更新这么多秒")
    p.add_argument("--probe_timeout", type=float, default=1200)
    p.add_argument("--probe_only", action="store_true")
    p.add_argument("--resume_main", action="store_true")
    p.add_argument("--reprobe", action="store_true", help="保留旧容量报告并重新测量；已有正式队列仍须 --resume_main")
    p.add_argument("--batch_size", type=int, default=256)
    p.add_argument("--cpu_threads", type=int, default=4)
    p.add_argument("--workers", type=int, default=0)
    p.add_argument("--epochs", type=int, default=200)
    p.add_argument("--patience", type=int, default=30)
    p.add_argument("--lr", type=float, default=0.001)
    p.add_argument("--ram_reserve_gib", type=float, default=3.0)
    p.add_argument("--gpu_reserve_gib", type=float, default=2.0)
    p.add_argument("--reserve_fraction", type=float, default=0.10)
    p.add_argument("--memory_safety_factor", type=float, default=1.35)
    p.add_argument("--train_ram_extra_gib", type=float, default=1.0)
    p.add_argument("--interval", type=float, default=1.0)
    p.add_argument("--launcher_log", help=argparse.SUPPRESS)
    p.add_argument("--pid_file", help=argparse.SUPPRESS)
    p.add_argument("--latest_log", help=argparse.SUPPRESS)
    args = p.parse_args()
    args.data_dir = Path(args.data_dir).expanduser().resolve()
    args.output_root = Path(args.output_root).expanduser().resolve() if args.output_root else args.data_dir / "sixclass_auto_v1_0_4_results"
    args.cache_dir = Path(args.cache_dir).expanduser().resolve() if args.cache_dir else args.output_root / "feature_cache"
    try:
        args.gpu_index = int(args.device.split(":", 1)[1])
    except (ValueError, IndexError):
        p.error("--device 必须为 CUDA 逻辑设备 cuda:N")
    if not args.device.startswith("cuda:") or args.gpu_index < 0:
        p.error("--device 必须为 CUDA 逻辑设备 cuda:N")
    if not (1 <= args.min_parallel <= args.max_parallel <= 4):
        p.error("必须满足 1 <= min_parallel <= max_parallel <= 4")
    if min(args.probe_steps, args.cpu_threads, args.batch_size, args.epochs, args.patience) <= 0 or args.workers < 0:
        p.error("步数、线程数、batch size、epochs/patience 必须正数，workers>=0")
    if min(args.probe_timeout, args.interval, args.lr) <= 0 or args.probe_min_seconds < 0 or args.memory_safety_factor < 1:
        p.error("超时、间隔、学习率必须正数；min_seconds>=0，安全系数>=1")
    if args.probe_min_seconds >= args.probe_timeout:
        p.error("probe_min_seconds 必须小于 probe_timeout")
    if not 0 <= args.reserve_fraction < 1 or min(args.ram_reserve_gib, args.gpu_reserve_gib, args.train_ram_extra_gib) < 0:
        p.error("资源预留参数非法")
    return args


def command_for_profile(args, model, path):
    return [sys.executable, "-u", str(HERE / "train_all.py"), "--mode", "profile",
            "--data_dir", str(args.data_dir), "--cache_dir", str(args.cache_dir),
            "--output_dir", str(path), "--model_type", model, "--seed", "42",
            "--feature_scheme", "full", "--device", args.device,
            "--batch_size", str(args.batch_size), "--workers", str(args.workers),
            "--cpu_threads", str(args.cpu_threads), "--profile_steps", str(args.probe_steps),
            "--lr", str(args.lr)]


def wait_stage(command, folder, args, uuid=None):
    process, handle = launch(command, folder / "stdout.log", args.cpu_threads)
    started = time.monotonic()
    try:
        while process.poll() is None:
            owned_tree(process)
            if "preflight" not in command and time.monotonic() - started > args.probe_timeout:
                raise TimeoutError(f"串行阶段超时：{folder}")
            if uuid is not None:
                current = sampling(args, uuid, [], {})
                if below_floor(current):
                    raise RuntimeError(f"串行实测触及资源预留，已停止自有进程：{folder}")
            time.sleep(min(args.interval, 1.0))
        code = process.wait()
        if code:
            kind = classify_exit(code, folder / "stdout.log")
            raise RuntimeError(f"串行阶段失败({kind}) code={code}：{folder}\n" + tail_text(folder / "stdout.log"))
    finally:
        stop_child(process)
        OWN_PROCESSES.pop(process.pid, None)
        handle.close()


def preflight(args, folder):
    folder.mkdir(parents=True, exist_ok=True)
    command = command_for_profile(args, MODEL_ORDER[0], folder)
    command[command.index("--mode") + 1] = "preflight"
    command[command.index("--device") + 1] = "cpu"
    wait_stage(command, folder, args)
    return json.loads((folder / "preflight.json").read_text(encoding="utf-8"))


def fingerprint_content(value):
    if isinstance(value, dict):
        if "sha256" in value:
            return {"sha256": value["sha256"], "size": value.get("size")}
        return {k: fingerprint_content(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [fingerprint_content(v) for v in value]
    return value


def binding_for(args, uuid, checked):
    keys = ("batch_size", "workers", "cpu_threads", "epochs", "patience", "lr", "probe_steps",
            "memory_safety_factor", "train_ram_extra_gib", "ram_reserve_gib", "gpu_reserve_gib",
            "reserve_fraction", "probe_min_seconds")
    sources = {path.name: hashlib.sha256(path.read_bytes()).hexdigest() for path in sorted(HERE.glob("*.py"))}
    return {"schema": CAPACITY_SCHEMA, "gpu_uuid": uuid,
            "input_fingerprints": fingerprint_content(checked["input_fingerprints"]),
            "settings": {key: getattr(args, key) for key in keys}, "source_sha256": sources,
            "experiments": {"models": MODEL_ORDER, "seeds": SEEDS, "feature_schemes": ["full"]}}


def select_resume_capacity(report, binding, requested_max):
    if report.get("schema") != CAPACITY_SCHEMA or report.get("binding") != binding:
        raise ValueError("容量报告的代码、数据、设备或资源/训练设置已改变；请加 --reprobe 重新测量。旧版报告不能直接复用。")
    selected = report.get("selected_max_parallel")
    current = report.get("current_max_parallel", selected)
    if not isinstance(selected, int) or not isinstance(current, int) or min(selected, current) < 1:
        raise ValueError("上次容量未通过；请加 --reprobe。")
    return min(selected, current, int(requested_max))


def shared_cache_bytes(checked):
    paths = checked.get("data_manifest_paths", {})
    directories = {Path(path).parent for path in paths.values() if isinstance(path, str)}
    return sum(path.stat().st_size for directory in directories for path in directory.iterdir() if path.is_file())


def make_controller(args, uuid, n, cache_bytes, event_path=None):
    return ResourceController(max_parallel=n, cpu_budget=available_cpu_slots(), cpu_parallel_limit=n,
        gpu_reserve_gib=args.gpu_reserve_gib, ram_reserve_gib=args.ram_reserve_gib,
        reserve_fraction=args.reserve_fraction, gpu_mapping={args.gpu_index: uuid},
        shared_cache_budget_bytes=cache_bytes, event_path=event_path)


def snapshot_for(controller):
    # Explicit helpers make telemetry unit tests independent of physical GPUs.
    return {"host": memory_snapshot(), "gpu": nvidia_snapshot(),
            "process_rss": {r.pid: process_rss_bytes(r.pid) for r in controller.active.values()},
            "process_private_rss": {r.pid: process_anonymous_rss_bytes(r.pid) for r in controller.active.values()}}


def classify_exit(code, log_path):
    if code == 0:
        return "complete"
    if classify_oom(log_path):
        return "confirmed_memory_error"
    if code < 0:
        return "ambiguous_interrupt"  # e.g. -9: OOM, administrator kill, or another cause.
    return "fatal_model_or_data_error"


def serial_profiles(args, uuid, folder, checked):
    profiles = {}
    cache = shared_cache_bytes(checked)
    controller = make_controller(args, uuid, 1, cache, folder / "resource_events.jsonl")
    for model in MODEL_ORDER:
        # The first allocation has no measured model peak yet. Require conservative
        # startup space and monitor it; one real measurement cannot be guaranteed safe.
        previous = [training_request_from_profile(p, args.memory_safety_factor, args.cpu_threads,
                    args.workers, args.train_ram_extra_gib) for p in profiles.values()]
        request = ResourceRequest(max([int((1 + args.train_ram_extra_gib) * GIB * args.memory_safety_factor)] +
                                     [p.host_bytes for p in previous]),
                                  max([int(GIB * args.memory_safety_factor)] + [p.gpu_bytes for p in previous]),
                                  args.cpu_threads + args.workers)
        deadline = time.monotonic() + args.probe_timeout
        while True:
            allowed, reason, details = controller.admission(request, snapshot_for(controller), args.gpu_index)
            if allowed:
                break
            controller.record("serial_profile_resource_wait", model=model, reason=reason, details=details)
            if time.monotonic() >= deadline:
                raise RuntimeError(f"串行测量 {model} 等待资源超时({reason})；没有启动额外模型。")
            time.sleep(min(args.interval, 1.0))
        path = folder / model
        print(f"[SERIAL] {model} 实测 {args.probe_steps} 次 AdamW 更新及单批验证", flush=True)
        wait_stage(command_for_profile(args, model, path), path, args, uuid)
        profile = json.loads((path / "resource_profile.json").read_text(encoding="utf-8"))
        if fingerprint_content(profile["input_fingerprints"]) != fingerprint_content(checked["input_fingerprints"]):
            raise RuntimeError("串行探针与预检输入 SHA256 不一致，停止。")
        profiles[model] = profile
        cache = max(cache, int(profile.get("host_readonly_cache_budget_bytes") or 0))
        controller.shared_cache_budget_bytes = cache
        save_json(folder / "serial_profiles.json", profiles)
    return profiles


def budget_candidate(args, uuid, profiles, n, snapshot=None):
    """Choose a measured model mix that can all be reserved BEFORE any launch.

    Other mixes, including repeated seeds of a large model, remain subject to
    formal queue admission. Capacity is a process cap, not a promise all mixes fit.
    """
    cache = max(int(p.get("host_readonly_cache_budget_bytes") or 0) for p in profiles.values())
    requests = {m: training_request_from_profile(p, args.memory_safety_factor, args.cpu_threads,
                  args.workers, args.train_ram_extra_gib) for m, p in profiles.items()}
    base = snapshot or snapshot_for(make_controller(args, uuid, n, cache))
    admitted, rejected = [], []
    for combination in itertools.combinations(MODEL_ORDER, n):
        control = make_controller(args, uuid, n, cache)
        details_list = []
        for ordinal, model in enumerate(combination):
            allowed, reason, details = control.admission(requests[model], base, args.gpu_index)
            details_list.append({"model": model, "allowed": allowed, "reason": reason, "details": details})
            if not allowed:
                break
            control.reserve(model, -(ordinal + 1), requests[model], args.gpu_index)
        else:
            # Probe the most demanding admitted mixed-model combination.
            host = sum(requests[m].host_bytes for m in combination)
            gpu = sum(requests[m].gpu_bytes for m in combination)
            score = host / max(1, base["host"]["available_bytes"]) + gpu / max(1, base["gpu"]["devices"][uuid]["free_bytes"])
            admitted.append((score, list(combination), details_list))
            continue
        rejected.append({"models": list(combination), "checks": details_list})
    if not admitted:
        return None, requests, {"rejected": rejected, "shared_cache_budget_bytes": cache}
    _, models, details = max(admitted, key=lambda item: item[0])
    return models, requests, {"admitted_models": models, "checks": details, "rejected": rejected,
                             "shared_cache_budget_bytes": cache,
                             "requests": {m: asdict(r) for m, r in requests.items()}}


def sampling(args, uuid, models, procs):
    host = memory_snapshot()
    gpu = nvidia_snapshot()["devices"].get(uuid)
    if gpu is None:
        raise RuntimeError("GPU UUID 遥测不可用，停止本次自有任务")
    alive = [m for m in models if procs[m].poll() is None]
    pid_bytes = gpu.get("pid_bytes", {})
    present = [m for m in alive if procs[m].pid in pid_bytes or str(procs[m].pid) in pid_bytes]
    return {"at": now(), "host_available_bytes": host["available_bytes"], "gpu_free_bytes": gpu["free_bytes"],
            "host_floor_bytes": max(int(args.ram_reserve_gib * GIB), int(host["total_bytes"] * args.reserve_fraction)),
            "gpu_floor_bytes": max(int(args.gpu_reserve_gib * GIB), int(gpu["total_bytes"] * args.reserve_fraction)),
            "available_ram_gib": gib(host["available_bytes"]),
            "free_gpu_gib": gib(gpu["free_bytes"]),
            "rss_sum_gib": gib(sum(process_rss_bytes(procs[m].pid) for m in alive)),
            "alive": alive, "gpu_present": present,
            "ram_floor_gib": gib(max(int(args.ram_reserve_gib * GIB), int(host["total_bytes"] * args.reserve_fraction))),
            "gpu_floor_gib": gib(max(int(args.gpu_reserve_gib * GIB), int(gpu["total_bytes"] * args.reserve_fraction)))}


def below_floor(sample):
    return (sample["host_available_bytes"] < sample["host_floor_bytes"]
            or sample["gpu_free_bytes"] < sample["gpu_floor_bytes"])


def profile_overlap(profiles):
    """Intersect performed optimizer-update intervals, excluding barrier/polling.

    CUDA is synchronized inside every measured update. This proves overlapping
    training calls plus separate NVML PID evidence; it does not measure SM usage.
    """
    intersect = None
    for profile in profiles.values():
        intervals = []
        for item in profile.get("optimizer_update_intervals", []):
            if not item.get("optimizer_step_performed", True):
                continue
            try:
                begin, end = float(item["started_epoch_seconds"]), float(item["ended_epoch_seconds"])
            except (KeyError, ValueError, TypeError):
                continue
            if end > begin:
                intervals.append((begin, end))
        intervals.sort()
        if not intervals:
            return 0.0
        if intersect is None:
            intersect = intervals
            continue
        common = []
        left = right = 0
        while left < len(intersect) and right < len(intervals):
            begin = max(intersect[left][0], intervals[right][0])
            end = min(intersect[left][1], intervals[right][1])
            if end > begin:
                common.append((begin, end))
            if intersect[left][1] < intervals[right][1]:
                left += 1
            else:
                right += 1
        intersect = common
    return sum(end - begin for begin, end in (intersect or []))


def probe_candidate(args, uuid, root, n, profiles=None):
    if not profiles or set(profiles) != set(MODEL_ORDER):
        raise ValueError("并发前必须完成全部四模型的串行资源实测，不能使用固定每路估计。")
    models, requests, estimates = budget_candidate(args, uuid, profiles, n)
    folder = root / f"n{n}"
    folder.mkdir(parents=True, exist_ok=False)  # parent run attempt is always unique.
    result = {"n": n, "models": models, "started_at": now(), "samples": [], "status": "running", "budget": estimates}
    save_json(folder / "report.json", result)
    if models is None:
        result.update(status="rejected_resource_estimate", reason="没有按实测预算可同时准入的模型组合；未启动进程。", ended_at=now())
        save_json(folder / "report.json", result)
        return result
    cache = estimates["shared_cache_budget_bytes"]
    controller = make_controller(args, uuid, n, cache, folder / "resource_events.jsonl")
    processes, handles = {}, {}
    start_file = folder / "training_start.json"
    ready_files = {m: folder / m / "profile_ready.json" for m in models}
    started, overlap, stop_reason = time.monotonic(), 0, None
    released = False
    try:
        for model in models:
            # Recheck live resources for EACH launch; pending reservations protect
            # other admitted processes whose allocations have not appeared yet.
            allowed, reason, details = controller.admission(requests[model], snapshot_for(controller), args.gpu_index)
            if not allowed:
                stop_reason = f"launch_resource_changed:{reason}"
                break
            path = folder / model
            command = command_for_profile(args, model, path) + [
                "--profile_ready_file", str(ready_files[model]), "--profile_start_file", str(start_file),
                "--profile_barrier_timeout", str(args.probe_timeout),
                "--profile_min_seconds", str(args.probe_min_seconds)]
            process, handle = launch(command, path / "stdout.log", args.cpu_threads)
            processes[model], handles[model] = process, handle
            controller.reserve(model, process.pid, requests[model], args.gpu_index)
        while len(processes) == n and not stop_reason:
            sample = sampling(args, uuid, models, processes)
            result["samples"].append(sample)
            if released and len(sample["gpu_present"]) == n:
                overlap += 1
            if below_floor(sample):
                stop_reason = "resource_floor_reached"
                break
            if any(p.poll() is not None and p.returncode != 0 for p in processes.values()):
                stop_reason = "child_failure"
                break
            if not released and all(path.is_file() for path in ready_files.values()):
                save_json(start_file, {"released_at": now(), "models": models})
                result["barrier_released_at"] = now()
                released = True
                print(f"[PROBE {n}] 所有模型已就绪，同步开始实测训练", flush=True)
            if all(p.poll() is not None for p in processes.values()):
                break
            if time.monotonic() - started > args.probe_timeout:
                stop_reason = "probe_timeout"
                break
            if len(result["samples"]) % 5 == 1:
                save_json(folder / "report.json", result)
            time.sleep(args.interval)
        failure_codes = {m: p.returncode for m, p in processes.items() if p.poll() is not None and p.returncode != 0}
        if stop_reason:
            for p in processes.values():
                stop_child(p)
        codes = {m: p.wait() for m, p in processes.items()}
        classifications = {m: classify_exit(code, folder / m / "stdout.log") for m, code in failure_codes.items()}
        completed = {}
        for m in models:
            profile_path = folder / m / "resource_profile.json"
            if profile_path.is_file():
                completed[m] = json.loads(profile_path.read_text(encoding="utf-8"))
        seconds = profile_overlap(completed) if len(completed) == n else 0.0
        result.update(returncodes=codes, failure_classifications=classifications,
                      duration_seconds=round(time.monotonic() - started, 3),
                      gpu_overlap_samples=overlap, actual_training_overlap_seconds=seconds,
                      ended_at=now(), barrier_released=released,
                      observed_profiles=completed)
        if "fatal_model_or_data_error" in classifications.values():
            result.update(status="fatal_model_or_data_error", reason="非内存异常；降并发不会修复，停止自动流程。")
        elif stop_reason:
            status = "ambiguous_interrupt" if "ambiguous_interrupt" in classifications.values() else "resource_or_runtime_rejected"
            result.update(status=status, reason=stop_reason,
                          note="信号退出原因未知，不声称已证实 OOM；仅允许有限降低并发重试。")
        elif (all(code == 0 for code in codes.values()) and released and len(completed) == n
              and all(int(p.get("actual_optimizer_steps", 0)) >= args.probe_steps for p in completed.values())
              and (n == 1 or (overlap > 0 and seconds > 0))):
            result.update(status="passed_gpu_overlap_observed" if n > 1 else "passed_single_model",
                          reason="单模型实测完成" if n == 1 else "同步训练区间及 NVML 自有 PID 重叠均已核实")
        else:
            result.update(status="overlap_unverified", reason="未证明所有模型实际训练同时重叠，不据此提高并发。")
        save_json(folder / "report.json", result)
        return result
    except BaseException as err:
        result.update(status="interrupted" if isinstance(err, KeyboardInterrupt) else "fatal_probe_error",
                      reason=f"{type(err).__name__}: {err}", ended_at=now())
        save_json(folder / "report.json", result)
        raise
    finally:
        for m, p in processes.items():
            stop_child(p)
            OWN_PROCESSES.pop(p.pid, None)
            if m in controller.active:
                controller.release(m)
        for handle in handles.values():
            handle.close()


def command_for_full(args, selected, resume=False):
    # Enforce the current user hard cap even if a caller passes an older value.
    selected = min(int(selected), int(getattr(args, "max_parallel", selected)))
    cmd = [sys.executable, "-u", str(HERE / "run_all_4_models.py"),
           "--data_dir", str(args.data_dir), "--cache_dir", str(args.cache_dir),
           "--output_dir", str(args.output_root / "main_20"),
           "--models", *MODEL_ORDER, "--seeds", *[str(s) for s in SEEDS],
           "--feature_schemes", "full", "--phase", "all", "--device", args.device,
           "--staged_exports", "--max_parallel", str(selected), "--workers", str(args.workers),
           "--cpu_threads", str(args.cpu_threads), "--batch_size", str(args.batch_size),
           "--epochs", str(args.epochs), "--patience", str(args.patience), "--lr", str(args.lr),
           "--gpu_reserve_gib", str(args.gpu_reserve_gib), "--ram_reserve_gib", str(args.ram_reserve_gib),
           "--reserve_fraction", str(args.reserve_fraction), "--memory_safety_factor", str(args.memory_safety_factor),
           "--train_ram_extra_gib", str(args.train_ram_extra_gib), "--oom_retries", "1", "--profile_steps", str(getattr(args, "probe_steps", 8))]
    if resume:
        cmd.append("--resume_queue")
    return cmd


def evaluation_started(output):
    return (output / "evaluation_plan.json").exists() or any(output.glob("*/*/seed_*/test_started.json"))


def update_current_cap(root, cap, **details):
    path = root / "capacity_selection.json"
    report = json.loads(path.read_text())
    report["current_max_parallel"] = cap
    report.setdefault("runtime_history", []).append({"at": now(), "max_parallel": cap, **details})
    save_json(path, report)


def formal_run(args, n, uuid):
    root, output = args.output_root, args.output_root / "main_20"
    existing = (output / "queue_manifest.json").is_file()
    if existing and not args.resume_main:
        raise RuntimeError("已有正式队列；请检查并使用 --resume_main。")
    current = min(n, args.max_parallel)
    for attempt in range(1, current + 1):
        update_current_cap(root, current, event="formal_attempt", attempt=attempt)
        log = root / "logs" / f"main_20_{datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%S_%f')}_{os.getpid()}.log"
        event_path = output / "resource_events.jsonl"
        event_offset = event_path.stat().st_size if event_path.exists() else 0
        process, handle = launch(command_for_full(args, current, resume=existing or attempt > 1), log, args.cpu_threads)
        stopped_for_floor = False
        try:
            while process.poll() is None:
                owned_tree(process)
                snap = sampling(args, uuid, [], {})
                if below_floor(snap):
                    stopped_for_floor = True
                    print("[FULL SAFE_STOP] 资源低于预留，停止本次自有队列。", flush=True)
                    stop_child(process)
                    break
                time.sleep(max(1.0, args.interval))
            code = process.wait()
        finally:
            stop_child(process)
            OWN_PROCESSES.pop(process.pid, None)
            handle.close()
        runtime_state = output / "resource_runtime_state.json"
        if runtime_state.is_file():
            state = json.loads(runtime_state.read_text())
            current = min(current, int(state.get("max_parallel", current)))
            update_current_cap(root, current, event="queue_runtime_cap", returncode=code)
        if code == 0:
            update_current_cap(root, current, event="complete")
            print("[FULL PASS] 20次训练、串行导出、冻结测试及汇总全部结束", flush=True)
            return 0
        # Once frozen evaluation starts, changing the experiment set through
        # automatic training retry would be scientifically unsafe.
        if evaluation_started(output):
            raise RuntimeError(f"冻结评价已开始；不会修改并发并重跑训练。请检查后显式恢复：{log}")
        tail = tail_text(log, 100).lower()
        recent_events = []
        if event_path.exists():
            with event_path.open("rb") as stream:
                stream.seek(event_offset)
                for line in stream:
                    try:
                        recent_events.append(json.loads(line))
                    except (ValueError, UnicodeDecodeError):
                        pass
        confirmed = classify_oom(log) or any(e.get("event") == "oom_retry_pending" or
            (e.get("event") == "oom_concurrency_reduced" and e.get("reason") == "confirmed_oom") for e in recent_events)
        suspected = (code < 0 and not stopped_for_floor) or any(e.get("event") == "resource_suspected_retry_pending" for e in recent_events)
        wait_expired = any(e.get("event") == "queue_resource_wait_timeout" for e in recent_events)
        job_failures = [e for e in recent_events if e.get("event") == "job_failed"]
        if job_failures and not job_failures[-1].get("oom") and not job_failures[-1].get("suspected_resource"):
            raise RuntimeError(f"最终失败是非资源训练异常，不因更早的OOM事件重试：{log}")
        if not (stopped_for_floor or confirmed or suspected or wait_expired) or current <= 1:
            raise RuntimeError(f"正式队列 code={code}，未自动重跑；日志：{log}\n" + tail[-2000:])
        current -= 1
        update_current_cap(root, current, event="resource_fallback", returncode=code,
                           reason="floor" if stopped_for_floor else "confirmed_memory_error" if confirmed else "resource_wait_timeout" if wait_expired else "ambiguous_interrupt")
        existing = True
        print(f"[FULL RETRY] 降低到 {current} 路，同超参数、同输出目录恢复；信号退出不等同证实OOM。", flush=True)
    return 1


def acquire_run_lock(args):
    global LOCK_FILE
    args.output_root.mkdir(parents=True, exist_ok=True)
    LOCK_FILE = (args.output_root / "auto_runner.lock").open("a+")
    try:
        fcntl.flock(LOCK_FILE, fcntl.LOCK_EX | fcntl.LOCK_NB)
    except BlockingIOError:
        LOCK_FILE.close()
        LOCK_FILE = None
        raise RuntimeError("这个输出目录已有任务运行；拒绝重复启动，保留原 PID 和主日志。")
    pid_path = Path(args.pid_file) if args.pid_file else args.output_root / "auto_runner.pid"
    pid_path.parent.mkdir(parents=True, exist_ok=True)
    temporary = pid_path.with_name(pid_path.name + f".{os.getpid()}.tmp")
    temporary.write_text(str(os.getpid()) + "\n")
    temporary.replace(pid_path)
    save_json(args.output_root / "auto_runner_identity.json", {"pid": os.getpid(), "accepted_at": now(),
               "pid_file": str(pid_path), "invocation_log": args.launcher_log, "version": VERSION,
               "process_start_ticks": process_identity(os.getpid()), "process_group": os.getpgrp()})
    if args.launcher_log and args.latest_log:
        latest = Path(args.latest_log)
        latest.parent.mkdir(parents=True, exist_ok=True)
        link = latest.with_name(latest.name + f".{os.getpid()}.tmp")
        try:
            link.unlink(missing_ok=True)
            link.symlink_to(Path(args.launcher_log).resolve())
            link.replace(latest)
        finally:
            link.unlink(missing_ok=True)
    print(f"[ACCEPTED] pid={os.getpid()} pid_file={pid_path}", flush=True)
    return pid_path


def run():
    signal.signal(signal.SIGTERM, on_signal)
    signal.signal(signal.SIGINT, on_signal)
    args = get_args()
    for name in ("train.csv", "validation.csv", "test.csv", "train_meta.tsv", "validation_meta.tsv", "test_meta.tsv"):
        if not (args.data_dir / name).is_file():
            raise FileNotFoundError(args.data_dir / name)
    if args.output_root == HERE or args.output_root in HERE.parents or args.output_root == args.data_dir or args.output_root in args.data_dir.parents:
        raise ValueError("结果目录不能覆盖代码、原始数据或其父目录")
    pid_path = acquire_run_lock(args)
    try:
        args.cache_dir.mkdir(parents=True, exist_ok=True)
        mapping, _ = discover_devices(SimpleNamespace(device=args.device, gpu_ids=[args.gpu_index], cpu_threads=args.cpu_threads, workers=args.workers))
        uuid = mapping[args.gpu_index]
        print(f"[GPU] {args.device} -> {uuid}", flush=True)
        attempt = args.output_root / "probe" / (datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S_%f") + f"_{os.getpid()}")
        attempt.mkdir(parents=True)
        print("[PREFLIGHT] 串行预备共享只读缓存；不做 test 推理", flush=True)
        checked = preflight(args, attempt / "preflight")
        binding = binding_for(args, uuid, checked)
        report_path = args.output_root / "capacity_selection.json"
        if args.resume_main and not args.reprobe:
            if not report_path.exists():
                raise FileNotFoundError("--resume_main 需要已通过的容量报告；也可加 --reprobe 重新测量。")
            selection = json.loads(report_path.read_text())
            selected = select_resume_capacity(selection, binding, args.max_parallel)
            print(f"[RESUME] 当前硬上限={args.max_parallel}；采用 min(旧容量,已降档上限,当前上限)={selected}", flush=True)
        else:
            if report_path.exists() and not args.reprobe:
                raise RuntimeError("已有容量报告；恢复请用 --resume_main，重新测量请用 --reprobe。")
            if (args.output_root / "main_20" / "queue_manifest.json").exists() and not args.resume_main:
                raise RuntimeError("已有正式队列，重新测量后恢复还需 --resume_main。")
            if report_path.exists():
                save_json(attempt / "previous_capacity_selection.json", json.loads(report_path.read_text()))
            report = {"schema": CAPACITY_SCHEMA, "status": "profiling", "created_at": now(), "binding": binding,
                      "attempt_directory": str(attempt), "selected_max_parallel": None,
                      "max_requested": args.max_parallel, "trials": [],
                      "note": "短程资源上限不是无OOM保证；正式队列仍动态准入。同名旧尝试完整保留。"}
            save_json(report_path, report)
            try:
                profiles = serial_profiles(args, uuid, attempt / "serial", checked)
                report["serial_profiles"] = {m: str(attempt / "serial" / m / "resource_profile.json") for m in MODEL_ORDER}
                selected = None
                for n in range(args.max_parallel, args.min_parallel - 1, -1):
                    trial = probe_candidate(args, uuid, attempt, n, profiles)
                    report["trials"].append({k: v for k, v in trial.items() if k not in ("samples", "observed_profiles")})
                    save_json(report_path, report)
                    if trial["status"] in ("passed_gpu_overlap_observed", "passed_single_model"):
                        selected = n
                        break
                    if trial["status"] == "fatal_model_or_data_error":
                        raise RuntimeError(f"探针存在非内存错误；请检查 {attempt / f'n{n}' / 'report.json'}")
                report.update(status="passed" if selected else "no_capacity", selected_max_parallel=selected,
                              current_max_parallel=selected, ended_at=now())
                save_json(report_path, report)
                if selected is None:
                    raise RuntimeError("连最低档位也未通过；未启动正式训练。资源恢复后加 --reprobe 再测。")
            except BaseException as err:
                if report.get("status") == "profiling":
                    report.update(status="interrupted" if isinstance(err, KeyboardInterrupt) else "failed",
                                  reason=f"{type(err).__name__}: {err}", ended_at=now())
                    save_json(report_path, report)
                raise
        if args.probe_only:
            print(f"[PROBE ONLY] 已测得当前上限 {selected}，未启动正式20次训练。", flush=True)
            return 0
        return formal_run(args, selected, uuid)
    finally:
        # Stop all owned children BEFORE releasing the output lock/PID so a new
        # invocation cannot overlap the previous cleanup.
        stop_all()
        # Keep a later accepted launch's PID untouched if anything changed it.
        try:
            if pid_path.read_text().strip() == str(os.getpid()):
                pid_path.unlink()
        except OSError:
            pass
        if LOCK_FILE is not None:
            LOCK_FILE.close()


if __name__ == "__main__":
    try:
        raise SystemExit(run())
    except KeyboardInterrupt as err:
        print("[INTERRUPTED]", err, file=sys.stderr, flush=True)
        stop_all()
        raise SystemExit(130)
    except Exception as err:
        print(f"[ERROR] {type(err).__name__}: {err}", file=sys.stderr, flush=True)
        stop_all()
        raise SystemExit(1)
