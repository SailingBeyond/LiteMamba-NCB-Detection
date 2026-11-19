#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""真实双模型并行压力测试（不运行完整训练、不会访问 test 标签/指标）。

流程：单独预检并准备共享只读缓存 -> 两次串行短程资源探测 ->
      改进的训练阶段预算/共享缓存一次性预算准入 -> 同时启动两个真实
      AdamW 前向/反向/更新 + 单批验证探针 -> 监控 RAM/VRAM/PID 重叠。

整个程序不修改 run_all_4_models.py 的正式运行记录，不导出 train/val/test。
如果真实资源不足，拒绝启动双模型；不会关闭服务器的其他任务。
"""
from __future__ import annotations

import argparse
from datetime import datetime, timezone
import json
import os
from pathlib import Path
import signal
import subprocess
import sys
import time
from types import SimpleNamespace

from resource_scheduler import (GIB, ResourceController, training_request_from_profile,
                                memory_snapshot, nvidia_snapshot, process_rss_bytes)
from run_all_4_models import discover_devices

HERE = Path(__file__).resolve().parent
DEFAULT_DATA = ("/path/to/revision-inputs/CPLX-XNA-replenishment/"
                "CPLX-XNA-replenishment/fast5_single_guppy/sixclass_overlap_read_split_721")
MODELS = ("bigru", "transformer", "convmixer", "litemamba")
LIVE_PROCESSES = {}


def save_json(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(path.name + ".tmp")
    tmp.write_text(json.dumps(value, ensure_ascii=False, indent=2, allow_nan=False) + "\n", encoding="utf-8")
    tmp.replace(path)


def gib(value):
    return round(float(value or 0) / GIB, 3)


def stop_one(process):
    if process.poll() is not None:
        return
    try:
        os.killpg(process.pid, signal.SIGTERM)
    except ProcessLookupError:
        return
    try:
        process.wait(timeout=12)
    except subprocess.TimeoutExpired:
        try:
            os.killpg(process.pid, signal.SIGKILL)
        except ProcessLookupError:
            pass
        process.wait()


def stop_all():
    for process in tuple(LIVE_PROCESSES.values()):
        stop_one(process)


def termination_signal(signum, frame):
    raise KeyboardInterrupt(f"收到信号 {signum}，停止本测试创建的子进程")


def env_with_threads(threads):
    env = dict(os.environ, PYTHONUNBUFFERED="1")
    for key in ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS", "NUMEXPR_NUM_THREADS"):
        env[key] = str(threads)
    return env


def launch(command, log_path, threads):
    log_path.parent.mkdir(parents=True, exist_ok=True)
    log = log_path.open("w", encoding="utf-8", buffering=1)
    log.write("命令：" + " ".join(command) + "\n")
    log.flush()
    try:
        process = subprocess.Popen(command, stdout=log, stderr=subprocess.STDOUT,
                                   cwd=str(HERE), env=env_with_threads(threads), start_new_session=True)
        LIVE_PROCESSES[process.pid] = process
    except BaseException:
        log.close()
        raise
    print(f"[LAUNCH] pid={process.pid} log={log_path}", flush=True)
    return process, log


def finish_serial(command, log_path, threads):
    process, log = launch(command, log_path, threads)
    try:
        code = process.wait()
    finally:
        if process.poll() is None:
            stop_one(process)
        LIVE_PROCESSES.pop(process.pid, None)
        log.close()
    if code:
        tail = "\n".join(log_path.read_text(encoding="utf-8", errors="replace").splitlines()[-30:])
        raise RuntimeError(f"阶段失败，退出码={code}，日志={log_path}\n{tail}")


def snapshot_live(gpu_uuid, pids, controller):
    host = memory_snapshot()
    gpu = nvidia_snapshot()
    target = gpu["devices"].get(gpu_uuid)
    if target is None:
        raise RuntimeError("原来识别的 GPU UUID 在监控中消失；终止测试")
    present = sorted(pid for pid in pids if pid in target.get("pid_bytes", {}))
    alive = sorted(pid for pid in pids if LIVE_PROCESSES[pid].poll() is None)
    return {"at": datetime.now(timezone.utc).isoformat(),
            "host_available_bytes": host["available_bytes"],
            "gpu_free_bytes": target["free_bytes"],
            "process_rss_bytes": {str(pid): process_rss_bytes(pid) for pid in alive},
            "alive_pids": alive, "gpu_present_pids": present,
            "gpu_telemetry_error": gpu.get("process_telemetry_error")}


def command_for(mode, args, output, name, steps=3, profile_out=None):
    command = [sys.executable, "-u", str(HERE / "train_all.py"),
               "--mode", mode, "--data_dir", str(args.data_dir),
               "--cache_dir", str(args.cache_dir), "--output_dir", str(output),
               "--model_type", name, "--seed", str(args.seed),
               "--feature_scheme", "full", "--device", args.device,
               "--batch_size", str(args.batch_size), "--workers", str(args.workers),
               "--cpu_threads", str(args.cpu_threads)]
    if mode == "profile":
        command += ["--profile_steps", str(steps)]
        if profile_out is not None:
            command += ["--profile_out", str(profile_out)]
    return command


def parser_args():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument("--data_dir", default=DEFAULT_DATA)
    parser.add_argument("--output_dir", default=str(HERE.parent / "sixclass_dual_stress_v1_0_2"))
    parser.add_argument("--cache_dir", default=None, help="默认复用 v1.0.1 检查脚本的已验证特征缓存")
    parser.add_argument("--models", nargs=2, choices=MODELS, default=["transformer", "litemamba"])
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--batch_size", type=int, default=256)
    parser.add_argument("--workers", type=int, default=0)
    parser.add_argument("--cpu_threads", type=int, default=4)
    parser.add_argument("--serial_steps", type=int, default=3)
    parser.add_argument("--parallel_steps", type=int, default=20)
    parser.add_argument("--memory_safety_factor", type=float, default=1.35)
    parser.add_argument("--train_ram_extra_gib", type=float, default=1.0)
    parser.add_argument("--ram_reserve_gib", type=float, default=3.0)
    parser.add_argument("--gpu_reserve_gib", type=float, default=2.0)
    parser.add_argument("--reserve_fraction", type=float, default=0.10)
    parser.add_argument("--interval_seconds", type=float, default=2.0)
    parser.add_argument("--timeout_seconds", type=float, default=3600)
    args = parser.parse_args()
    if len(set(args.models)) != 2:
        parser.error("--models 必须指定两个不同的模型。")
    if args.device not in ("cuda", "cuda:0") and not args.device.startswith("cuda:"):
        parser.error("压力测试必须使用可监控的 CUDA 逻辑设备。")
    if (min(args.batch_size, args.cpu_threads, args.serial_steps, args.parallel_steps) <= 0
            or args.workers < 0 or args.interval_seconds <= 0 or args.timeout_seconds <= 0
            or args.memory_safety_factor < 1 or args.train_ram_extra_gib < 0
            or min(args.ram_reserve_gib, args.gpu_reserve_gib) < 0 or not 0 <= args.reserve_fraction < 1):
        parser.error("非法的资源、安全系数或步数。")
    args.data_dir = Path(args.data_dir).expanduser().resolve()
    args.output_dir = Path(args.output_dir).expanduser().resolve()
    if args.cache_dir:
        args.cache_dir = Path(args.cache_dir).expanduser().resolve()
    else:
        old_cache = HERE.parent / "sixclass_training_revision_v1_0_1" / "sixclass_parallel_check" / "feature_cache"
        args.cache_dir = old_cache if old_cache.is_dir() else HERE.parent / "sixclass_parallel_shared_cache"
    return args


def main():
    signal.signal(signal.SIGTERM, termination_signal)
    signal.signal(signal.SIGINT, termination_signal)
    args = parser_args()
    if args.output_dir == args.data_dir or args.output_dir in args.data_dir.parents:
        raise ValueError("输出目录不能包含原始数据")
    if args.output_dir == HERE or args.output_dir in HERE.parents:
        raise ValueError("输出目录不能是训练代码目录或其父目录")
    for name in ("train.csv", "validation.csv", "test.csv", "train_meta.tsv", "validation_meta.tsv", "test_meta.tsv"):
        if not (args.data_dir / name).is_file():
            raise FileNotFoundError(args.data_dir / name)
    args.output_dir.mkdir(parents=True, exist_ok=True)
    args.cache_dir.mkdir(parents=True, exist_ok=True)
    stamp = datetime.now().strftime("%Y%m%d_%H%M%S_%f")
    run_root = args.output_dir / stamp
    run_root.mkdir()
    report_path = run_root / "dual_stress_report.json"
    report = {"status": "running", "arguments": {k: str(v) if isinstance(v, Path) else v for k, v in vars(args).items()},
              "created_at": datetime.now(timezone.utc).isoformat(), "profiles": {}, "samples": []}
    save_json(report_path, report)
    print("测试目录：", run_root, flush=True)
    print("使用共享只读特征缓存：", args.cache_dir, flush=True)
    print("只运行真实短程训练更新和单批验证推理；不会选最优模型、不会运行 test 预测。", flush=True)
    gpu_index = 0 if args.device == "cuda" else int(args.device.split(":", 1)[1])
    mapping, hardware = discover_devices(SimpleNamespace(device=args.device, gpu_ids=[gpu_index], cpu_threads=args.cpu_threads, workers=args.workers))
    if len(mapping) != 1:
        raise RuntimeError("无法唯一识别 CUDA GPU")
    gpu_uuid = mapping[gpu_index]
    print("[GPU]", f"cuda:{gpu_index}", "->", gpu_uuid, flush=True)
    try:
        # Preflight loads source fingerprints and creates/reuses all 3 memmaps.
        # It does not run model inference or inspect test accuracy.
        preflight = command_for("preflight", args, run_root / "preflight", args.models[0])
        preflight[preflight.index("--device") + 1] = "cpu"
        print("[PREFLIGHT] 共享缓存预备（CPU，串行）", flush=True)
        finish_serial(preflight, run_root / "preflight.log", args.cpu_threads)
        controller = ResourceController(max_parallel=2, cpu_budget=None,
                 gpu_reserve_gib=args.gpu_reserve_gib, ram_reserve_gib=args.ram_reserve_gib,
                 reserve_fraction=args.reserve_fraction, gpu_mapping=mapping)
        controller.cpu_budget = max(controller.cpu_budget, 1)
        profiles = {}
        requests = {}
        for name in args.models:
            folder = run_root / "serial" / name
            print(f"[SERIAL] {name}: {args.serial_steps} 个更新步 + 验证探针", flush=True)
            finish_serial(command_for("profile", args, folder, name, args.serial_steps), folder / "stdout.log", args.cpu_threads)
            profile = json.loads((folder / "resource_profile.json").read_text(encoding="utf-8"))
            profiles[name] = profile
            req = training_request_from_profile(profile, args.memory_safety_factor,
                       args.cpu_threads, args.workers, args.train_ram_extra_gib)
            requests[name] = req
            report["profiles"][name] = {
                "rss_gib": gib(profile.get("host_peak_rss_bytes")),
                "gpu_reserved_gib": gib(req.gpu_bytes),
                "training_ram_reserved_gib": gib(req.host_bytes),
                "export_budget_gib": gib(profile.get("host_export_buffer_budget_bytes")),
                "cache_budget_gib": gib(profile.get("host_readonly_cache_budget_bytes"))}
            save_json(report_path, report)
        controller.shared_cache_budget_bytes = max(int(p.get("host_readonly_cache_budget_bytes") or 0) for p in profiles.values())
        print(f"[BUDGET] 只读缓存统一预约一次: {gib(controller.shared_cache_budget_bytes)} GiB", flush=True)
        request_snapshot = controller.snapshot()
        report["before_pair"] = {"ram_available_gib": gib(request_snapshot["host"]["available_bytes"]),
                                 "gpu_free_gib": gib(request_snapshot["gpu"]["devices"][gpu_uuid]["free_bytes"]),
                                 "host_cache_once_gib": gib(controller.shared_cache_budget_bytes),
                                 "ram_reserve_gib": gib(max(controller.ram_reserve_bytes, int(request_snapshot["host"]["total_bytes"] * controller.reserve_fraction)))}
        planned = []
        for i, name in enumerate(args.models):
            choice, reject = controller.choose(requests[name], request_snapshot)
            if choice is None:
                report.update(status="admission_rejected", rejection={"model": name, "details": reject})
                save_json(report_path, report)
                print("[拒绝启动双模型]", name, reject, flush=True)
                return 2
            controller.reserve("simulate:" + name, -(i + 1), requests[name], gpu_index)
            planned.append(name)
        controller.active.clear()
        print("[ADMISSION] 两个模型在当前 RAM/显存/CPU 预约条件下均通过。", flush=True)
        print("[PARALLEL] 两个真实独立进程同时进行优化器更新与验证探针。", flush=True)
        processes = {}
        opened_logs = {}
        for name in args.models:
            folder = run_root / "parallel" / name
            process, handle = launch(command_for("profile", args, folder, name, args.parallel_steps),
                                      folder / "stdout.log", args.cpu_threads)
            processes[name] = process
            opened_logs[name] = handle
        pids = [processes[name].pid for name in args.models]
        started = time.monotonic()
        host_floor = max(controller.ram_reserve_bytes, int(memory_snapshot()["total_bytes"] * controller.reserve_fraction))
        total_vram = request_snapshot["gpu"]["devices"][gpu_uuid]["total_bytes"]
        gpu_floor = max(controller.gpu_reserve_bytes, int(total_vram * controller.reserve_fraction))
        peak_combined_rss = 0
        observed_gpu_overlap = 0
        observed_pid_overlap = 0
        stopped_reason = None
        while True:
            sample = snapshot_live(gpu_uuid, pids, controller)
            rss = sum(sample["process_rss_bytes"].values())
            peak_combined_rss = max(peak_combined_rss, rss)
            if len(sample["alive_pids"]) == 2:
                observed_pid_overlap += 1
            if len(sample["gpu_present_pids"]) == 2:
                observed_gpu_overlap += 1
            report["samples"].append(sample)
            if len(report["samples"]) % 5 == 1:
                print(f"[MONITOR] RAM可用 {gib(sample['host_available_bytes']):.2f} GiB；"
                      f"GPU空闲 {gib(sample['gpu_free_bytes']):.2f} GiB；"
                      f"两进程RSS合计 {gib(rss):.2f} GiB；"
                      f"存活 {len(sample['alive_pids'])}；GPU上同时出现 {len(sample['gpu_present_pids'])}", flush=True)
                save_json(report_path, report)
            if sample["host_available_bytes"] < host_floor:
                stopped_reason = f"RAM可用低于安全底线 {gib(host_floor)} GiB"
            elif sample["gpu_free_bytes"] < gpu_floor:
                stopped_reason = f"GPU空闲低于安全底线 {gib(gpu_floor)} GiB"
            elif time.monotonic() - started > args.timeout_seconds:
                stopped_reason = "超过压力测试超时时限"
            if stopped_reason:
                print("[SAFE_STOP]", stopped_reason, flush=True)
                break
            finished = [p for p in processes.values() if p.poll() is not None]
            if any(p.returncode != 0 for p in finished):
                stopped_reason = "至少一个子进程失败，停止另一个子进程"
                break
            if len(finished) == 2:
                break
            time.sleep(args.interval_seconds)
        if stopped_reason:
            for proc in processes.values():
                stop_one(proc)
        retcodes = {name: process.wait() for name, process in processes.items()}
        report["returncodes"] = retcodes
        report["pid_overlap_samples"] = observed_pid_overlap
        report["gpu_overlap_samples"] = observed_gpu_overlap
        report["combined_rss_peak_gib"] = gib(peak_combined_rss)
        report["minimum_ram_available_gib"] = gib(min(s["host_available_bytes"] for s in report["samples"]))
        report["minimum_gpu_free_gib"] = gib(min(s["gpu_free_bytes"] for s in report["samples"]))
        report["duration_seconds"] = round(time.monotonic() - started, 2)
        if stopped_reason:
            report["status"] = "stopped_for_safety_or_error"
            report["stop_reason"] = stopped_reason
            print("[FAIL] 并行压力测试安全停止：", stopped_reason, flush=True)
        elif all(code == 0 for code in retcodes.values()) and observed_gpu_overlap >= 1:
            report["status"] = "passed_gpu_overlap_observed"
            print("[PASS] 两个模型成功完成短程训练；曾在同一GPU上同时观察到两个训练进程。", flush=True)
        elif all(code == 0 for code in retcodes.values()):
            report["status"] = "completed_gpu_overlap_unverified"
            print("[WARN] 两进程均成功完成，但GPU上同时运行的证据未取得，不能据此声称真实GPU并行。", flush=True)
        else:
            report["status"] = "failed"
            print("[FAIL] 双模型试验出现错误，请查看 parallel/*/stdout.log", flush=True)
        save_json(report_path, report)
        print("最终报告：", report_path, flush=True)
        print("说明：本测试尚未覆盖全量200 epochs及全量train/val分析导出的峰值；进入正式运行需使用全新输出目录。", flush=True)
        return 0 if report["status"] == "passed_gpu_overlap_observed" else 1
    except BaseException as exc:
        report["status"] = "interrupted_or_exception"
        report["error"] = f"{type(exc).__name__}: {exc}"
        save_json(report_path, report)
        raise
    finally:
        stop_all()
        for pid in list(LIVE_PROCESSES):
            LIVE_PROCESSES.pop(pid, None)
        # Close logs of parallel subprocesses even if interrupted.
        for handle in locals().get("opened_logs", {}).values():
            handle.close()


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except KeyboardInterrupt:
        print("用户中断；仅终止本次测试启动的子进程。", file=sys.stderr)
        raise SystemExit(130)
    except Exception as exc:
        print("测试失败：", exc, file=sys.stderr)
        raise SystemExit(1)
