#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""准备外部数据、实测资源并并行预测，随后串行统计和重绘 Figure 4。"""

from __future__ import annotations

import argparse
from contextlib import contextmanager
from datetime import datetime, timezone
import fcntl
import importlib
import json
import math
import os
from pathlib import Path
import shutil
import signal
import subprocess
import sys
import time
import uuid


HERE = Path(__file__).resolve().parent
VENDOR = HERE / "vendor"
GIB = 1024 ** 3
DEFAULT_RESULTS = "/path/to/revision-inputs/CPLX-XNA-replenishment/CPLX-XNA-replenishment/fast5_single_guppy/sixclass_overlap_read_split_721/sixclass_auto_v1_0_4_results/main_20"
DEFAULT_01_12_ROOT = "/path/to/staged-storage2/analysis/paper/xna_pc_6mer_features_for_model_test_X_to_T"
DEFAULT_13_16_6 = "/path/to/revision-inputs/POC-XNA_DNA-01_16/POC-XNA_DNA-01_16/fast5_single_guppy/xna_pc_13_16_6"
DEFAULT_17_20_6 = "/path/to/staged-storage2/analysis/paper/POC-XNA-17_20/fast5_single_guppy/xna_6mer_features_17_20_split_by_ref_strand"
DEFAULT_CORRECTED_13_16 = "/path/to/staged-storage2/analysis/paper/xna_pc_13_16_6_X_to_T"
MODELS = ("litemamba", "transformer", "convmixer", "bigru")
SEEDS = (42, 2024, 2025, 2026, 2027)


def utc_now():
    return datetime.now(timezone.utc).isoformat()


def read_json(path):
    return json.loads(Path(path).read_text(encoding="utf-8"))


def write_json(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + ".tmp-" + str(os.getpid()))
    with temporary.open("w", encoding="utf-8") as stream:
        json.dump(value, stream, ensure_ascii=False, indent=2, allow_nan=False)
        stream.write("\n")
        stream.flush()
        os.fsync(stream.fileno())
    os.replace(temporary, path)


def append_event(path, record):
    """每次事件一次追加写入；恢复依据原子队列状态，不依赖 JSONL 尾行。"""
    data = (json.dumps(record, ensure_ascii=False, allow_nan=False) + "\n").encode()
    descriptor = os.open(str(path), os.O_WRONLY | os.O_CREAT | os.O_APPEND, 0o600)
    try:
        offset = 0
        while offset < len(data):
            offset += os.write(descriptor, data[offset:])
        os.fsync(descriptor)
    finally:
        os.close(descriptor)


def validate_jsonl(path):
    """只修复断电/ENOSPC 留下的末尾半行；中间损坏必须报告。"""
    path = Path(path)
    if not path.exists():
        return
    offset, tail, line_number = 0, None, 0
    with path.open("rb") as stream:
        for line_number, line in enumerate(stream, 1):
            try:
                value = json.loads(line)
                if not isinstance(value, dict):
                    raise ValueError("事件不是 JSON object")
            except (ValueError, UnicodeDecodeError) as error:
                if line.endswith(b"\n") or stream.read(1):
                    raise RuntimeError("事件日志中间损坏：{} 第 {} 行".format(path, line_number)) from error
                tail = line
                break
            offset += len(line)
    if tail is not None:
        backup = path.with_name(path.name + ".truncated-" + str(os.getpid()) + ".bin")
        backup.write_bytes(tail)
        with path.open("r+b") as stream:
            stream.truncate(offset)
            stream.flush()
            os.fsync(stream.fileno())
        append_event(path, {"event": "truncated_tail_recovered", "line": line_number,
                            "saved_tail": str(backup), "recovered_at": utc_now()})
        print("[RECOVER] 已保存并移除事件日志末尾半行：" + str(backup), flush=True)
    elif offset:
        with path.open("r+b") as stream:
            stream.seek(-1, os.SEEK_END)
            if stream.read(1) != b"\n":
                stream.write(b"\n")
                stream.flush()
                os.fsync(stream.fileno())


def resource_module():
    if str(VENDOR) not in sys.path:
        sys.path.insert(0, str(VENDOR))
    return importlib.import_module("resource_scheduler")


def process_start_ticks(pid):
    directory = resource_module().proc_directory(int(pid))
    if directory is None:
        return None
    try:
        fields = (directory / "stat").read_text().rsplit(") ", 1)[1].split()
        return None if fields[0] == "Z" else int(fields[19])
    except (OSError, ValueError, IndexError):
        return None


def assert_no_live_children(state):
    for entry in state.get("active_children", []):
        ticks = entry.get("start_ticks")
        if ticks is not None and (process_start_ticks(entry["pid"]) == ticks or session_has_members(entry["pid"], ticks)):
            raise RuntimeError("上次队列仍有预测子进程，暂不重复启动：PID {}，任务 {}".format(entry["pid"], entry.get("run_id")))


@contextmanager
def output_lock(output):
    path = output / ".figure4_queue.lock"
    with path.open("a+", encoding="utf-8") as handle:
        try:
            fcntl.flock(handle.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError as error:
            raise RuntimeError("此输出目录已有 Figure 4 队列运行：" + str(output)) from error
        handle.seek(0)
        handle.truncate()
        handle.write(json.dumps({"pid": os.getpid(), "started_at": utc_now()}))
        handle.flush()
        yield


def environment(args):
    return dict(os.environ, PYTHONUNBUFFERED="1", OMP_NUM_THREADS=str(args.cpu_threads),
                MKL_NUM_THREADS=str(args.cpu_threads), OPENBLAS_NUM_THREADS=str(args.cpu_threads),
                NUMEXPR_NUM_THREADS=str(args.cpu_threads))


def terminate_owned(process):
    # start_new_session 使 PGID 等于此 Popen 的 PID；只在活着的 leader
    # 或仍可核实属于该 session 的后代存在时发送，避免误伤 PID 重用。
    leader_running = process.poll() is None
    if not leader_running and not session_has_members(process.pid, getattr(process, "_figure4_start_ticks", None)):
        return
    try:
        os.killpg(process.pid, signal.SIGTERM)
    except ProcessLookupError:
        return
    deadline = time.monotonic() + 10
    while time.monotonic() < deadline:
        process.poll()
        if process.returncode is not None and not session_has_members(process.pid, getattr(process, "_figure4_start_ticks", None)):
            return
        time.sleep(0.1)
    if process.poll() is None or session_has_members(process.pid, getattr(process, "_figure4_start_ticks", None)):
        try:
            os.killpg(process.pid, signal.SIGKILL)
        except ProcessLookupError:
            pass
        process.wait()


def session_has_members(session_id, original_ticks=None):
    """不依赖父子关系：DataLoader 子进程可能在 leader 退出后被重新收养。"""
    root = Path("/proc")
    current_ticks = process_start_ticks(session_id)
    if original_ticks is not None and current_ticks is not None and current_ticks != original_ticks:
        return False
    try:
        namespace = (root / "self/ns/pid").readlink()
        for directory in root.iterdir():
            if not directory.name.isdigit():
                continue
            try:
                if (directory / "ns/pid").readlink() != namespace:
                    continue
                fields = (directory / "stat").read_text().rsplit(") ", 1)[1].split()
                status = (directory / "status").read_text().splitlines()
                local_session = next(line.split()[-1] for line in status if line.startswith("NSsid:"))
                local_group = next(line.split()[-1] for line in status if line.startswith("NSpgid:"))
                if fields[0] != "Z" and int(local_session) == session_id and int(local_group) == session_id:
                    return True
            except (OSError, ValueError, IndexError, StopIteration):
                continue
    except OSError:
        return False
    return False


def start_process(command, log_path, args, stage, state, state_path, commands_path, run_id):
    log_path.parent.mkdir(parents=True, exist_ok=True)
    handle = log_path.open("w", encoding="utf-8")
    try:
        process = subprocess.Popen(command, cwd=str(HERE), env=environment(args), stdout=handle,
                                   stderr=subprocess.STDOUT, start_new_session=True)
    except BaseException:
        handle.close()
        raise
    process._figure4_start_ticks = process_start_ticks(process.pid)
    record = {"stage": stage, "run_id": run_id, "pid": process.pid,
              "start_ticks": process._figure4_start_ticks, "command": command,
              "log": str(log_path), "started_at": utc_now()}
    try:
        state["active_children"].append(record)
        write_json(state_path, state)
        append_event(commands_path, {"event": "child_started", **record})
    except BaseException:
        terminate_owned(process)
        handle.close()
        raise
    print("[{}] {} pid={} log={}".format(stage.upper(), run_id, process.pid, log_path), flush=True)
    return {"process": process, "handle": handle, "record": record, "log": log_path}


def finish_process(child, state, state_path, commands_path):
    process = child["process"]
    code = process.wait()
    terminate_owned(process)
    child["handle"].close()
    state["active_children"] = [item for item in state["active_children"] if item["pid"] != process.pid]
    write_json(state_path, state)
    append_event(commands_path, {"event": "child_finished", "pid": process.pid,
                                "run_id": child["record"]["run_id"], "returncode": code,
                                "finished_at": utc_now()})
    return code


def run_serial(command, log_path, args, stage, state, state_path, commands_path, run_id):
    child = start_process(command, log_path, args, stage, state, state_path, commands_path, run_id)
    try:
        while child["process"].poll() is None:
            time.sleep(min(args.poll_seconds, 2.0))
        code = finish_process(child, state, state_path, commands_path)
    except BaseException:
        terminate_owned(child["process"])
        child["handle"].close()
        raise
    if code:
        raise RuntimeError("{} 失败，退出码 {}；日志：{}".format(stage, code, log_path))


def canonical_uuid(value):
    value = str(value or "").strip()
    if value.lower().startswith("gpu-"):
        value = value[4:]
    try:
        return str(uuid.UUID(value))
    except (ValueError, AttributeError, TypeError):
        return None


def match_cuda_devices(observed, snapshot, requested):
    """使用实际 CUDA context UUID；绝不把逻辑编号当物理 GPU 编号。"""
    devices = observed["devices"]
    tokens = [value.strip() for value in os.environ.get("CUDA_VISIBLE_DEVICES", "").split(",") if value.strip()]
    mapping, evidence = {}, []
    if len(set(requested)) != len(requested):
        raise ValueError("GPU 逻辑编号重复")
    for index in requested:
        if index < 0 or index >= len(devices) or devices[index].get("logical") != index:
            raise ValueError("无 CUDA 可见逻辑 GPU：" + str(index))
        item = devices[index]
        if item.get("unsupported_mig") or any(value.upper().startswith("MIG-") for value in tokens):
            raise RuntimeError("此调度器不支持使用完整 GPU 遥测来预约 MIG 切片")
        actual = canonical_uuid(item.get("uuid"))
        if actual:
            matches = [key for key in snapshot["devices"] if canonical_uuid(key) == actual]
            method = item.get("identity_method", "cuda_context_uuid")
        else:
            if item.get("uuid"):
                raise RuntimeError("CUDA 返回的 UUID 无效，拒绝猜测映射")
            if len(tokens) != len(devices) or not all(value.upper().startswith("GPU-") for value in tokens):
                raise RuntimeError("无法核实 GPU UUID；请检查 CUDA 驱动，或显式设置 CUDA_VISIBLE_DEVICES=GPU-实际UUID。")
            matches = [key for key in snapshot["devices"] if key.lower().startswith(tokens[index].lower())]
            method = "explicit_uuid_visibility"
        if len(matches) != 1:
            raise RuntimeError("逻辑 GPU 的 UUID 未对应唯一完整 GPU，拒绝重复预算")
        mapping[index] = matches[0]
        evidence.append({"logical": index, "uuid": matches[0], "identity_method": method})
    if not mapping or len(set(mapping.values())) != len(mapping):
        raise RuntimeError("GPU 映射为空或重复")
    return mapping, evidence


def discover_devices(args):
    if args.device == "cpu":
        return {}, {"mode": "cpu", "reason": "explicit_cpu"}
    command = [sys.executable, str(VENDOR / "cuda_device_probe.py")]
    result = subprocess.run(command, check=True, capture_output=True, text=True, timeout=60, env=environment(args))
    observed = json.loads(result.stdout.strip().splitlines()[-1])
    if not observed.get("available"):
        if args.device != "auto":
            raise RuntimeError("当前 Python/PyTorch 无可用 CUDA")
        return {}, {"mode": "cpu", "reason": "torch_cuda_unavailable", "probe": observed}
    snapshot = resource_module().nvidia_snapshot()
    requested = list(range(len(observed["devices"])))
    if args.device.startswith("cuda:"):
        requested = [int(args.device.split(":", 1)[1])]
    mapping, evidence = match_cuda_devices(observed, snapshot, requested)
    for item in evidence:
        print("[GPU] cuda:{} -> {} ({})".format(item["logical"], item["uuid"], item["identity_method"]), flush=True)
    return mapping, {"mode": "cuda", "mapping": mapping, "evidence": evidence, "probe": observed}


def worker_command(args, plan_path, run, device, mode="predict", profile_out=None):
    command = [sys.executable, "-u", str(HERE / "figure4_worker.py"), "--plan", str(plan_path),
               "--run_id", run["run_id"], "--mode", mode, "--output_dir", str(args.output_root),
               "--device", device, "--batch_size", str(args.batch_size), "--workers", str(args.workers),
               "--cpu_threads", str(args.cpu_threads), "--embedding_dtype", args.embedding_dtype]
    if args.no_embeddings:
        command.append("--no_embeddings")
    if args.no_attention:
        command.append("--no_attention")
    if profile_out is not None:
        command += ["--profile_out", str(profile_out), "--profile_batches", str(args.profile_batches),
                    "--profile_min_seconds", str(args.profile_min_seconds)]
    return command


def prepare_command(args):
    if args.previous_output:
        command = [sys.executable, "-u", str(HERE / "prepare_figure4_correction.py"),
                   "--previous_output", str(args.previous_output),
                   "--corrected_13_16", str(args.corrected_13_16),
                   "--output_dir", str(args.output_root), "--reuse_mode", args.reuse_mode,
                   "--disk_reserve_gib", str(args.disk_reserve_gib),
                   "--parent_queue_pid", str(os.getpid())]
        if args.replace_controls:
            command.append("--replace_controls")
        for name in ("original_pos_csv", "original_neg_csv", "original_pos_meta", "original_neg_meta"):
            if getattr(args, name):
                command += ["--" + name, str(getattr(args, name))]
        if args.resume:
            command.append("--resume")
        return command
    command = [sys.executable, "-u", str(HERE / "prepare_figure4.py"),
               "--training_results", str(args.training_results), "--output_root", str(args.output_root),
               "--models", *args.models, "--seeds", *map(str, args.seeds),
               "--feature_schemes", *args.feature_schemes,
               "--disk_reserve_gib", str(args.disk_reserve_gib)]
    if args.data_manifest:
        command += ["--data_manifest", str(args.data_manifest)]
    else:
        command += ["--dir_01_12_root", args.dir_01_12_root,
                    "--dir_13_16_6", args.dir_13_16_6, "--dir_17_20_6", args.dir_17_20_6]
    if args.resume:
        command.append("--resume")
    if args.allow_partial_templates:
        command.append("--allow_partial_templates")
    return command


def prediction_settings(args):
    return {"batch_size": int(args.batch_size), "workers": int(args.workers),
            "cpu_threads": int(args.cpu_threads), "embedding_dtype": args.embedding_dtype,
            "no_embeddings": bool(args.no_embeddings), "no_attention": bool(args.no_attention),
            "amp_policy": "training_checkpoint_config"}


def dataset_is_complete(args, run, plan, dataset):
    # 此 helper 只使用标准库，流式核验所有实际输出的大小、SHA 和冻结身份。
    # 与 worker 共用，避免父队列将损坏或缺少 embedding 的结果直接跳过。
    from figure4_common import validate_completion
    folder = args.output_root / "predictions" / run["run_id"] / dataset["dataset_id"]
    return validate_completion(folder, plan["plan_id"], run["run_id"], dataset["dataset_id"],
                               prediction_settings(args), int(dataset["n_samples"]))


def run_is_complete(args, run, plan):
    marker = args.output_root / "predictions" / run["run_id"] / "run_complete.json"
    if not marker.is_file():
        return False
    data = read_json(marker)
    if data.get("plan_id") != plan.get("plan_id") or data.get("run_id") != run["run_id"]:
        raise RuntimeError("完成记录与当前冻结计划不一致：" + str(marker))
    if data.get("prediction_settings") != prediction_settings(args):
        raise RuntimeError("预测设置与已有结果不一致，请使用原 batch/CPU/输出设置 --resume，或指定新输出目录：" + str(marker))
    if data.get("status") != "complete":
        return False
    # 每文件 marker 和预测文件必须都在，避免把只剩总标记的目录跳过。
    for dataset in plan["datasets"]:
        if not dataset_is_complete(args, run, plan, dataset):
            return False
    return True


def check_disk(args, required=0):
    free = shutil.disk_usage(args.output_root).free
    reserve = math.ceil(args.disk_reserve_gib * GIB)
    if free < reserve + required:
        raise OSError(28, "输出盘空间不足：剩余 {:.2f} GiB，保留 {:.2f} GiB，估计当前新增需求 {:.2f} GiB".format(free / GIB, reserve / GIB, required / GIB))
    return free


def estimate_remaining_bytes(args, plan, pending):
    total = 0
    for run in pending:
        bytes_per_sample = args.disk_bytes_per_sample
        if not args.no_embeddings:
            dimension = 2 * int(run.get("model_kwargs", {}).get("model_dim", 96))
            bytes_per_sample += dimension * (4 if args.embedding_dtype == "float32" else 2)
        if not args.no_attention:
            bytes_per_sample += 6 * 4
        # 更正模式只会重新预测替换输入。未变文件若损坏，worker 会明确
        # 停止而不重建；不把旧64文件再次计入写盘预算或额外读一遍SHA。
        for dataset in prediction_datasets(plan):
            if dataset_is_complete(args, run, plan, dataset):
                continue
            total += int(dataset["n_samples"]) * bytes_per_sample
    return total


def prediction_datasets(plan):
    datasets = plan["datasets"]
    if "replacement_dataset_ids" not in plan:
        return datasets
    selected = plan["replacement_dataset_ids"]
    if not isinstance(selected, list) or not selected or len(set(selected)) != len(selected):
        raise ValueError("更正计划 replacement_dataset_ids 必须为非空且不重复的列表")
    by_id = {dataset["dataset_id"]: dataset for dataset in datasets}
    if len(by_id) != len(datasets) or any(value not in by_id for value in selected):
        raise ValueError("更正计划存在重复或未知 dataset_id")
    return [by_id[value] for value in selected]


def shared_cache_budget(plan, parallel):
    if "replacement_dataset_ids" in plan:
        sizes = []
        for dataset in prediction_datasets(plan):
            outputs = dataset.get("outputs", {})
            names = ("merged.npy", "raw.npy", "labels.npy")
            if all(isinstance(outputs.get(name), dict) and int(outputs[name].get("size_bytes", 0)) > 0 for name in names):
                sizes.append(sum(int(outputs[name]["size_bytes"]) for name in names))
            else:
                root = Path(dataset["cache_dir"])
                sizes.append(sum((root / name).stat().st_size for name in names))
        return min(sum(sizes), max(sizes) * max(1, int(parallel)))
    total = max(0, int(plan.get("total_cache_bytes", 0)))
    maximum = max(0, int(plan.get("max_file_cache_bytes", 0)))
    if total and maximum:
        return min(total, maximum * max(1, int(parallel)))
    return max(0, int(plan.get("shared_cache_budget_bytes", 0)))


def profile_runs(args, plan, pending, mapping, invocation, state, state_path, commands_path):
    representatives = {}
    for run in pending:
        key = str(run.get("profile_key") or run["model"] + "/" + run.get("feature_scheme", "full"))
        representatives.setdefault(key, run)
    profiles = {}
    resource = resource_module()
    for key, run in representatives.items():
        measured = []
        for logical in (list(mapping) if mapping else [None]):
            device = "cpu" if logical is None else "cuda:" + str(logical)
            folder = args.output_root / "resource_profiles" / invocation / key / device.replace(":", "_")
            target = folder / "resource_profile.json"
            command = worker_command(args, args.output_root / "evaluation_plan.json", run, device, "profile", target)
            run_serial(command, folder / "stdout.log", args, "profile", state, state_path, commands_path, run["run_id"])
            profile = read_json(target)
            if int(profile.get("measured_batches", 0)) < 1:
                raise RuntimeError("资源探测没有实际处理 batch：" + str(target))
            measured.append(profile)
        profile = dict(measured[0])
        for name in ("host_peak_rss_bytes", "host_admission_estimate_bytes", "gpu_peak_reserved_bytes",
                     "gpu_context_baseline_bytes", "gpu_driver_delta_bytes", "gpu_admission_estimate_bytes"):
            profile[name] = max(int(value.get(name) or 0) for value in measured)
        profiles[key] = resource.request_from_profile(profile, args.memory_safety_factor, args.cpu_threads, args.workers)
        write_json(args.output_root / "resource_profiles" / invocation / key / "aggregate.json",
                   {"profile_key": key, "representative_run_id": run["run_id"],
                    "prediction_settings": prediction_settings(args), "measured_devices": measured,
                    "memory_safety_factor": args.memory_safety_factor,
                    "resource_request": vars(profiles[key]), "created_at": utc_now()})
        print("[PROFILE] {} RAM={:.2f} GiB GPU={:.2f} GiB（含安全系数）".format(key, profiles[key].host_bytes / GIB, profiles[key].gpu_bytes / GIB), flush=True)
    return profiles


def predict_queue(args, plan, pending, profiles, mapping, invocation, state, state_path, commands_path):
    resource = resource_module()
    controller = resource.ResourceController(max_parallel=state["effective_max_parallel"],
        cpu_parallel_limit=args.cpu_parallel_limit, cpu_budget=args.cpu_budget,
        gpu_reserve_gib=args.gpu_reserve_gib, ram_reserve_gib=args.ram_reserve_gib,
        reserve_fraction=args.reserve_fraction, gpu_mapping=mapping,
        shared_cache_budget_bytes=shared_cache_budget(plan, state["effective_max_parallel"]),
        event_path=args.output_root / "resource_events.jsonl")
    active, retries, idle_since, last_wait = {}, {}, None, 0.0
    try:
        while pending or active:
            check_disk(args)
            for run_id, child in list(active.items()):
                if child["process"].poll() is None:
                    continue
                code = finish_process(child, state, state_path, commands_path)
                controller.release(run_id)
                del active[run_id]
                run = child["run"]
                if code == 0 and run_is_complete(args, run, plan):
                    state["runs"][run_id] = {"status": "complete", "finished_at": utc_now()}
                    write_json(state_path, state)
                    print("[COMPLETE] " + run_id, flush=True)
                    continue
                used = retries.get(run_id, 0)
                if resource.classify_oom(child["log"]) and used < args.oom_retries:
                    retries[run_id] = used + 1
                    controller.reduce_after_oom("confirmed_prediction_oom")
                    state["effective_max_parallel"] = controller.max_parallel
                    state["runs"][run_id] = {"status": "retry_pending", "reason": "confirmed_oom", "retry": used + 1}
                    pending.insert(0, run)
                    write_json(state_path, state)
                    print("[BACKOFF] {}：并发上限降至 {}，保留已完成文件后重试".format(run_id, controller.max_parallel), flush=True)
                    continue
                state["runs"][run_id] = {"status": "failed", "returncode": code, "log": str(child["log"])}
                write_json(state_path, state)
                raise RuntimeError("预测未完成：{} code={}；日志：{}。保留输出，清理问题后使用 --resume。".format(run_id, code, child["log"]))

            if not pending:
                if active:
                    time.sleep(args.poll_seconds)
                continue
            snapshot = controller.snapshot()
            admitted = False
            for run in list(pending):
                key = str(run.get("profile_key") or run["model"] + "/" + run.get("feature_scheme", "full"))
                request = profiles[key]
                choice, rejected = controller.choose(request, snapshot)
                if choice is None:
                    continue
                run_id = run["run_id"]
                attempt = retries.get(run_id, 0) + 1
                log = args.output_root / "logs" / invocation / run_id / ("predict_attempt_" + str(attempt) + ".log")
                child = start_process(worker_command(args, args.output_root / "evaluation_plan.json", run, choice["device"]),
                                      log, args, "predict", state, state_path, commands_path, run_id)
                child["run"] = run
                active[run_id] = child
                controller.reserve(run_id, child["process"].pid, request, choice["logical_gpu"])
                pending.remove(run)
                state["runs"][run_id] = {"status": "running", "device": choice["device"], "log": str(log)}
                write_json(state_path, state)
                admitted = True
                # 重新取样包含刚启动进程，并保留尚未出现的预约余额。
                snapshot = controller.snapshot()
            if admitted or active:
                idle_since = None
            else:
                idle_since = idle_since or time.monotonic()
                if time.monotonic() - last_wait > 30:
                    print("[WAIT] 正在等待可用 RAM/GPU/CPU；活动 {}，待处理 {}".format(len(active), len(pending)), flush=True)
                    controller.record("admission_wait", pending=len(pending), snapshot=snapshot)
                    last_wait = time.monotonic()
                if time.monotonic() - idle_since > args.resource_wait_seconds:
                    raise RuntimeError("资源等待超时；可在保持输入、权重和科学设置不变时调整资源上限后 --resume。")
            time.sleep(args.poll_seconds)
    finally:
        for child in active.values():
            terminate_owned(child["process"])
            child["handle"].close()
        state["active_children"] = []
        state["effective_max_parallel"] = controller.max_parallel
        try:
            write_json(state_path, state)
        except OSError:
            # 清理本次子进程优先；磁盘满不能阻止 finally 返回原始错误。
            pass


def inherit_correction_configuration(args, explicit_options):
    if not args.previous_output:
        return
    existing = args.output_root / "evaluation_plan.json"
    if args.phase in ("analyze", "plot") and existing.is_file():
        source_plan = read_json(existing)
        if "replacement_dataset_ids" not in source_plan:
            raise ValueError("指定输出不含更正计划；请确认 output_root")
        settings = source_plan.get("prediction_settings")
        replaced = set(source_plan["replacement_dataset_ids"])
        controls_replaced = any(dataset["dataset_id"] in replaced and dataset["role"] == "control"
                                for dataset in source_plan["datasets"])
        if "replace_controls" in explicit_options and args.replace_controls != controls_replaced:
            raise ValueError("replace_controls 与已有更正计划不一致")
        args.replace_controls = controls_replaced
    else:
        source_plan = read_json(args.previous_output / "evaluation_plan.json")
        # 此准备模块仅在实际准备缓存时才初始化运行时，不在父调度器加载 CUDA。
        from prepare_figure4_correction import inherit_prediction_settings
        settings = inherit_prediction_settings(args.previous_output, source_plan)
    fields = ("batch_size", "workers", "cpu_threads", "embedding_dtype", "no_embeddings", "no_attention")
    required = set(fields) | {"amp_policy"}
    if not isinstance(settings, dict) or set(settings) != required or settings.get("amp_policy") != "training_checkpoint_config":
        raise ValueError("旧结果 prediction_settings 字段不完整或AMP策略不兼容")
    for name in ("batch_size", "workers", "cpu_threads"):
        if type(settings[name]) is not int:
            raise ValueError("旧结果预测设置必须为整数：" + name)
    if settings["batch_size"] < 1 or settings["cpu_threads"] < 1 or settings["workers"] != 0:
        raise ValueError("旧结果 batch/threads/workers 不支持当前流式更正预测")
    if settings["embedding_dtype"] not in ("float16", "float32") or any(type(settings[name]) is not bool for name in ("no_embeddings", "no_attention")):
        raise ValueError("旧结果 embedding/attention 设置无效")
    for name in fields:
        if name in explicit_options and getattr(args, name) != settings[name]:
            raise ValueError("更正必须继承旧预测设置；显式 --{}={} 与旧值 {} 不一致".format(name, getattr(args, name), settings[name]))
        setattr(args, name, settings[name])
    for name in ("models", "seeds", "feature_schemes"):
        values = source_plan.get(name)
        if not isinstance(values, list) or not values:
            raise ValueError("旧计划缺少冻结任务列表：" + name)
        if name in explicit_options and getattr(args, name) != values:
            raise ValueError("更正复用完整旧任务列表，--{} 必须与旧计划一致".format(name))
        setattr(args, name, list(values))
    training = Path(source_plan["training_results"]).expanduser().resolve()
    if "training_results" in explicit_options and args.training_results != training:
        raise ValueError("更正必须复用旧计划的 training_results")
    args.training_results = training


def parse_args(argv=None):
    raw_arguments = list(sys.argv[1:] if argv is None else argv)
    explicit_options = {token.split("=", 1)[0][2:] for token in raw_arguments if token.startswith("--")}
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--version", action="version", version="Figure4 external revision 1.1.0")
    parser.add_argument("--training_results", type=Path, default=Path(DEFAULT_RESULTS))
    parser.add_argument("--output_root", "--output_dir", dest="output_root", type=Path)
    parser.add_argument("--previous_output", type=Path,
                        help="已完成的旧Figure4输出；启用只更正XNA13-16并复用未变预测")
    parser.add_argument("--corrected_13_16", type=Path, default=Path(DEFAULT_CORRECTED_13_16))
    parser.add_argument("--replace_controls", action="store_true",
                        help="同时替换XNA13-16 control；默认只替换target")
    parser.add_argument("--reuse_mode", choices=("hardlink", "symlink", "copy"), default="hardlink")
    for name in ("original_pos_csv", "original_neg_csv", "original_pos_meta", "original_neg_meta"):
        parser.add_argument("--" + name, type=Path,
                            help="更正split严格审计的未拆分原文件；默认由准备脚本按corrected目录查找")
    parser.add_argument("--data_manifest", type=Path)
    parser.add_argument("--dir_01_12_root", default=DEFAULT_01_12_ROOT)
    parser.add_argument("--dir_13_16_6", default=DEFAULT_13_16_6)
    parser.add_argument("--dir_17_20_6", default=DEFAULT_17_20_6)
    parser.add_argument("--models", nargs="+", choices=MODELS, default=list(MODELS))
    parser.add_argument("--seeds", type=int, nargs="+", default=list(SEEDS))
    parser.add_argument("--feature_schemes", nargs="+", default=["full"])
    parser.add_argument("--max_parallel", type=int, default=4)
    parser.add_argument("--device", default="auto")
    parser.add_argument("--batch_size", type=int, default=256)
    parser.add_argument("--workers", type=int, default=0,
                        help="直接流式读取 mmap，保持0；模型进程并发由 max_parallel 控制")
    parser.add_argument("--cpu_threads", type=int, default=4)
    parser.add_argument("--cpu_parallel_limit", type=int, default=2)
    parser.add_argument("--cpu_budget", type=int)
    parser.add_argument("--gpu_reserve_gib", type=float, default=2)
    parser.add_argument("--ram_reserve_gib", type=float, default=3)
    parser.add_argument("--reserve_fraction", type=float, default=0.10)
    parser.add_argument("--memory_safety_factor", type=float, default=1.35)
    parser.add_argument("--disk_reserve_gib", type=float, default=5)
    parser.add_argument("--disk_bytes_per_sample", type=int, default=1536,
                        help="剩余结果空间估算中的每窗口 CSV/logits 预算；另加 embedding/attention，非实际压缩率承诺")
    parser.add_argument("--profile_batches", type=int, default=8)
    parser.add_argument("--profile_min_seconds", type=float, default=5)
    parser.add_argument("--poll_seconds", type=float, default=2)
    parser.add_argument("--resource_wait_seconds", type=float, default=3600)
    parser.add_argument("--oom_retries", type=int, default=1)
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--phase", choices=("all", "predict", "analyze", "plot"), default="all")
    parser.add_argument("--bootstrap", type=int, default=200)
    parser.add_argument("--analysis_seed", type=int, default=42)
    parser.add_argument("--no_embeddings", action="store_true")
    parser.add_argument("--no_attention", action="store_true")
    parser.add_argument("--embedding_dtype", choices=("float16", "float32"), default="float16")
    parser.add_argument("--allow_partial_templates", action="store_true")
    parser.add_argument("--launcher_log", type=Path, help=argparse.SUPPRESS)
    parser.add_argument("--pid_file", type=Path, help=argparse.SUPPRESS)
    parser.add_argument("--latest_log", type=Path, help=argparse.SUPPRESS)
    args = parser.parse_args(raw_arguments)
    for name in ("max_parallel", "batch_size", "cpu_threads", "cpu_parallel_limit", "profile_batches", "disk_bytes_per_sample"):
        if getattr(args, name) < 1:
            parser.error(name + " 必须为正整数")
    for name in ("workers", "bootstrap", "oom_retries"):
        if getattr(args, name) < 0:
            parser.error(name + " 必须非负")
    if args.workers != 0:
        parser.error("流式 mmap 预测无需 DataLoader；workers 必须为0，并发请设置 max_parallel")
    for name in ("gpu_reserve_gib", "ram_reserve_gib", "disk_reserve_gib", "profile_min_seconds"):
        if not math.isfinite(getattr(args, name)) or getattr(args, name) < 0:
            parser.error(name + " 必须为非负有限数")
    if not 0 <= args.reserve_fraction < 1 or not math.isfinite(args.memory_safety_factor) or args.memory_safety_factor < 1:
        parser.error("reserve_fraction 必须在 [0,1)，memory_safety_factor 必须 >=1")
    if any(not math.isfinite(value) or value <= 0 for value in (args.poll_seconds, args.resource_wait_seconds)):
        parser.error("poll_seconds 和 resource_wait_seconds 必须为正数")
    if args.device not in ("auto", "cpu", "cuda") and not (args.device.startswith("cuda:") and args.device[5:].isdigit()):
        parser.error("device 必须为 auto/cpu/cuda/cuda:N")
    for name in ("models", "seeds", "feature_schemes"):
        if len(set(getattr(args, name))) != len(getattr(args, name)):
            parser.error(name + " 不允许重复")
    if args.cpu_budget is not None and args.cpu_budget < 1:
        parser.error("cpu_budget 必须为正整数")
    args.training_results = args.training_results.expanduser().resolve()
    if args.data_manifest:
        args.data_manifest = args.data_manifest.expanduser().resolve()
    for name in ("dir_01_12_root", "dir_13_16_6", "dir_17_20_6"):
        setattr(args, name, str(Path(getattr(args, name)).expanduser().resolve()))
    args.corrected_13_16 = args.corrected_13_16.expanduser().resolve()
    for name in ("original_pos_csv", "original_neg_csv", "original_pos_meta", "original_neg_meta"):
        if getattr(args, name):
            setattr(args, name, getattr(args, name).expanduser().resolve())
    if args.previous_output:
        args.previous_output = args.previous_output.expanduser().resolve()
        incompatible = explicit_options & {"data_manifest", "dir_01_12_root", "dir_13_16_6", "dir_17_20_6", "allow_partial_templates"}
        if incompatible:
            parser.error("更正模式使用旧冻结计划，不接受原始全量发现参数：" + ", ".join(sorted(incompatible)))
    elif explicit_options & {"corrected_13_16", "replace_controls", "reuse_mode", "original_pos_csv", "original_neg_csv", "original_pos_meta", "original_neg_meta"}:
        parser.error("更正参数需要 --previous_output；也可直接使用 run_figure4_xna13_16_corrected.sh")
    if args.output_root is None:
        if args.previous_output:
            args.output_root = args.previous_output.parent / "figure4_external_xna13_16_corrected_v1_1_0"
        else:
            parent = args.training_results.parent if args.training_results.name == "main_20" else args.training_results
            args.output_root = parent / "figure4_external_v1_1_0"
    args.output_root = args.output_root.expanduser().resolve()
    if args.previous_output and (args.output_root == args.previous_output or args.output_root in args.previous_output.parents or args.previous_output in args.output_root.parents):
        parser.error("更正输出必须是独立新目录，不能覆盖、嵌入旧输出或用其父目录")
    inherit_correction_configuration(args, explicit_options)
    if args.output_root == args.training_results or args.output_root in args.training_results.parents:
        parser.error("output_root 不能覆盖训练目录或其父目录")
    return args


def publish_launcher(args):
    if args.pid_file:
        args.pid_file.parent.mkdir(parents=True, exist_ok=True)
        temporary = args.pid_file.with_name(args.pid_file.name + ".tmp-" + str(os.getpid()))
        temporary.write_text(str(os.getpid()) + "\n", encoding="utf-8")
        os.replace(temporary, args.pid_file)
    if args.latest_log and args.launcher_log:
        args.latest_log.parent.mkdir(parents=True, exist_ok=True)
        temporary = args.latest_log.with_name(args.latest_log.name + ".tmp-" + str(os.getpid()))
        if temporary.is_symlink():
            temporary.unlink()
        temporary.symlink_to(args.launcher_log.resolve())
        os.replace(temporary, args.latest_log)
    print("[ACCEPTED] pid={} output={}".format(os.getpid(), args.output_root), flush=True)


def main(argv=None):
    args = parse_args(argv)
    print("[VERSION] Figure4 external revision 1.1.0; code=" + str(HERE), flush=True)
    if args.previous_output:
        print("[CORRECTION] 复用旧冻结权重与预测设置；{}；旧输出={}".format(
              "替换XNA13-16 target和control" if args.replace_controls else "仅替换XNA13-16 target", args.previous_output), flush=True)
    args.output_root.mkdir(parents=True, exist_ok=True)
    (args.output_root / "logs").mkdir(exist_ok=True)
    state_path = args.output_root / "queue_state.json"
    commands_path = args.output_root / "commands.jsonl"
    invocation = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S_%f_") + str(os.getpid())
    with output_lock(args.output_root):
        validate_jsonl(commands_path)
        validate_jsonl(args.output_root / "resource_events.jsonl")
        previous = read_json(state_path) if state_path.exists() else {}
        assert_no_live_children(previous)
        if previous and args.phase in ("all", "predict") and not args.resume:
            raise RuntimeError("此输出目录已有队列记录；保持设置不变后加 --resume 恢复，或使用新输出目录。")
        publish_launcher(args)
        old_cap = int(previous.get("effective_max_parallel", args.max_parallel))
        effective = min(args.max_parallel, old_cap) if args.resume else args.max_parallel
        if previous and args.phase in ("analyze", "plot"):
            effective = old_cap
        state = {"version": "figure4-resource-profiled-v1.1.0", "invocation_id": invocation,
                 "started_at": utc_now(), "status": "running", "phase": args.phase,
                 "effective_max_parallel": effective, "requested_max_parallel": args.max_parallel,
                 "active_children": [], "runs": previous.get("runs", {}), "history": previous.get("history", [])}
        if previous:
            state["history"].append({key: previous.get(key) for key in ("invocation_id", "started_at", "finished_at", "status", "error")})
        write_json(state_path, state)
        write_json(args.output_root / "logs" / invocation / "invocation.json",
                   {"invocation_id": invocation, "started_at": state["started_at"],
                    "python_executable": sys.executable, "python_version": sys.version,
                    "cuda_visible_devices": os.environ.get("CUDA_VISIBLE_DEVICES"),
                    "arguments": {key: str(value) if isinstance(value, Path) else value
                                  for key, value in vars(args).items()},
                    "prediction_settings": prediction_settings(args)})
        try:
            if args.phase in ("all", "predict"):
                check_disk(args)
                run_serial(prepare_command(args), args.output_root / "logs" / invocation / "prepare.log",
                           args, "prepare", state, state_path, commands_path, "prepare")
            plan_path = args.output_root / "evaluation_plan.json"
            if not plan_path.is_file():
                raise FileNotFoundError("没有冻结评估计划：" + str(plan_path))
            plan = read_json(plan_path)
            if args.previous_output:
                prediction_datasets(plan)
                if plan.get("prediction_settings") != prediction_settings(args):
                    raise RuntimeError("更正计划与继承的旧预测设置不一致，停止而不混合结果")
            state["plan_id"] = plan["plan_id"]
            write_json(state_path, state)
            if args.phase in ("all", "predict"):
                pending = []
                for run in plan["runs"]:
                    if run_is_complete(args, run, plan):
                        state["runs"][run["run_id"]] = {"status": "skipped_matching_completed"}
                    else:
                        pending.append(run)
                required = estimate_remaining_bytes(args, plan, pending)
                check_disk(args, required)
                state["remaining_output_estimate_bytes"] = required
                print("[DISK] 剩余预测输出保守估计 {:.2f} GiB；预留 {:.2f} GiB".format(required / GIB, args.disk_reserve_gib), flush=True)
                if pending:
                    mapping, hardware = discover_devices(args)
                    state["hardware"] = hardware
                    write_json(state_path, state)
                    profiles = profile_runs(args, plan, pending, mapping, invocation, state, state_path, commands_path)
                    predict_queue(args, plan, pending, profiles, mapping, invocation, state, state_path, commands_path)
                state["prediction_status"] = "complete"
                write_json(state_path, state)
            if args.phase in ("all", "analyze"):
                command = [sys.executable, "-u", str(HERE / "figure4_analyze.py"), "--output_root", str(args.output_root),
                           "--bootstrap", str(args.bootstrap), "--analysis_seed", str(args.analysis_seed)]
                run_serial(command, args.output_root / "logs" / invocation / "analysis.log", args, "analyze", state, state_path, commands_path, "analysis")
            if args.phase in ("all", "analyze", "plot"):
                command = [sys.executable, str(HERE / "plot_figure4_final_finetune.py"),
                           "--data_dir", str(args.output_root / "analysis_data"),
                           "--out_dir", str(args.output_root / "figures"), "--formats", "png", "pdf", "--dpi", "300"]
                run_serial(command, args.output_root / "logs" / invocation / "plot.log", args, "plot", state, state_path, commands_path, "plot")
            state.update(status="complete", finished_at=utc_now(), active_children=[])
            write_json(state_path, state)
            if args.phase in ("all", "analyze"):
                from package_figure4_results import package_results
                state["evaluation_stages_finished_at"] = state["finished_at"]
                state["packaging_started_at"] = utc_now()
                write_json(state_path, state)
                archive = package_results(args.output_root)
                state["result_archive"] = str(archive)
                state["packaging_finished_at"] = utc_now()
                state["finished_at"] = state["packaging_finished_at"]
                write_json(state_path, state)
                print("[RESULT PACKAGE] " + str(archive), flush=True)
            print("[FULL PASS] Figure 4 {} 阶段全部完成：{}".format(args.phase, args.output_root), flush=True)
            return 0
        except BaseException as error:
            state.update(status="interrupted" if isinstance(error, KeyboardInterrupt) else "failed",
                         error=type(error).__name__ + ": " + str(error), finished_at=utc_now())
            # ENOSPC 时保留最后一次已提交状态；异常本身继续写到独立 stdout 日志。
            try:
                write_json(state_path, state)
            except OSError:
                pass
            raise


if __name__ == "__main__":
    def stop_on_signal(signum, frame):
        raise KeyboardInterrupt("收到信号 " + str(signum))
    signal.signal(signal.SIGTERM, stop_on_signal)
    try:
        raise SystemExit(main())
    except KeyboardInterrupt as error:
        print("[INTERRUPTED] {}；保留输出，可用 --resume 继续。".format(error), file=sys.stderr)
        raise SystemExit(130)
    except Exception as error:
        print("[ERROR] {}: {}".format(type(error).__name__, error), file=sys.stderr)
        raise SystemExit(1)
