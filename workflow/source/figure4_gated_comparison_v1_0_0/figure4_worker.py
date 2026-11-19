#!/usr/bin/env python3
"""冻结 checkpoint 的流式外部预测；一个进程处理一组模型/seed。"""
from __future__ import annotations

import argparse
import contextlib
import csv
from datetime import datetime, timezone
import gzip
import json
import math
import os
from pathlib import Path
import platform
import resource
import shutil
import sys
import time

import numpy as np

from figure4_common import (atomic_json, read_json, runtime_modules, sha256,
                            torch_load, validate_completion, verify_run, VENDOR)

GIB = 1024 ** 3
META_FIELDS = ["sample_index", "sample_id", "source_file", "source_row", "source_sha256",
               "read_id", "read_id_kind", "original_kmer", "motif", "label", "true_class",
               "reference", "template", "cohort", "role", "file_strand", "bam_strand",
               "template_group", "pair_id", "run_id", "source_dataset", "position",
               "reference_identity_basis"]
CSV_FIELDS = (META_FIELDS + ["model", "seed", "feature_scheme"] +
              [f"{prefix}_{i}" for prefix in ("logit", "prob", "calibrated_prob") for i in range(6)] +
              ["prediction", "confidence", "calibrated_confidence"])


def prediction_settings(args):
    return {"batch_size": int(args.batch_size), "workers": int(args.workers),
            "cpu_threads": int(args.cpu_threads), "embedding_dtype": args.embedding_dtype,
            "no_embeddings": bool(args.no_embeddings), "no_attention": bool(args.no_attention),
            "amp_policy": "training_checkpoint_config"}


def replacement_dataset_ids(plan):
    """A correction may only open caches for the explicitly replaced files."""
    if "replacement_dataset_ids" not in plan:
        return None
    values = plan["replacement_dataset_ids"]
    if not isinstance(values, list) or not values or not all(isinstance(value, str) and value for value in values):
        raise ValueError("更正计划 replacement_dataset_ids 必须为非空 dataset_id 列表")
    if len(set(values)) != len(values):
        raise ValueError("更正计划 replacement_dataset_ids 不能重复")
    datasets = [dataset["dataset_id"] for dataset in plan["datasets"]]
    if len(set(datasets)) != len(datasets):
        raise ValueError("冻结计划的 dataset_id 重复")
    missing = set(values) - set(datasets)
    if missing:
        raise ValueError("更正计划中的替换 dataset_id 不在输入清单：" + ", ".join(sorted(missing)))
    return frozenset(values)


def validate_plan_settings(plan, args):
    replaced = replacement_dataset_ids(plan)
    frozen = plan.get("prediction_settings")
    if replaced is not None and not isinstance(frozen, dict):
        raise ValueError("更正计划缺少从旧结果继承的 prediction_settings；请重新准备更正计划")
    if frozen is not None and frozen != prediction_settings(args):
        raise ValueError("预测参数与更正计划继承的旧结果设置不一致；请保持旧 batch/workers/CPU/embedding/attention 设置")
    return replaced


def softmax(logits, temperature=1.0):
    values = np.asarray(logits, dtype=np.float64) / float(temperature)
    values -= values.max(axis=1, keepdims=True)
    exp = np.exp(values)
    return exp / exp.sum(axis=1, keepdims=True)


def check_inputs(dataset):
    for key in ("feature", "metadata"):
        item = dataset.get("identity", {}).get(key)
        if item:
            stat = Path(item["path"]).stat()
            if stat.st_size != item["size_bytes"] or stat.st_mtime_ns != item["mtime_ns"]:
                raise RuntimeError("外部输入在缓存冻结后改变：" + item["path"])
    root = Path(dataset["cache_dir"])
    saved = read_json(root / "cache_manifest.json")
    if saved.get("status") != "complete" or saved.get("identity") != dataset["identity"]:
        raise RuntimeError("缓存身份或状态不匹配：" + str(root))
    for name, info in saved.get("outputs", {}).items():
        stat = (root / name).stat()
        if stat.st_size != info["size_bytes"] or stat.st_mtime_ns != info["mtime_ns"]:
            raise RuntimeError("缓存文件在准备后改变：" + str(root / name))
    return root


def open_cache(dataset, raw_len):
    root = check_inputs(dataset)
    arrays = [np.load(root / name, mmap_mode="r", allow_pickle=False)
              for name in ("merged.npy", "raw.npy", "labels.npy")]
    n = int(dataset["n_samples"])
    if [a.shape for a in arrays] != [(n, 6, 13), (n, 6, raw_len), (n,)]:
        raise ValueError("缓存形状不匹配：" + str(root))
    if not all(a.dtype == np.float32 for a in arrays[:2]) or arrays[2].dtype != np.int64:
        raise ValueError("缓存dtype不匹配：" + str(root))
    return root, arrays


def load_model(plan, run, args):
    verify_run(run)
    import torch
    torch.set_num_threads(args.cpu_threads)
    torch.manual_seed(int(run["seed"]))
    np.random.seed(int(run["seed"]))
    if args.device.startswith("cuda") and not torch.cuda.is_available():
        raise RuntimeError("请求CUDA但当前PyTorch/CUDA不可用；不静默切换设备")
    device = torch.device(args.device)
    if device.type == "cuda":
        torch.cuda.set_device(device)
        torch.cuda.manual_seed_all(int(run["seed"]))
        torch.backends.cudnn.benchmark = False
        torch.backends.cudnn.deterministic = True
        torch.backends.cuda.matmul.allow_tf32 = False
        torch.backends.cudnn.allow_tf32 = False
    _, runtime = runtime_modules(run)
    config = read_json(run["resolved_config"])
    checkpoint = torch_load(run["checkpoint"])
    recorded = checkpoint.get("resolved_config", {})
    for key in ("model_type", "seed", "feature_scheme", "model_kwargs", "target_index", "class_names"):
        if key in recorded and recorded[key] != config.get(key):
            raise RuntimeError("checkpoint与冻结配置不一致：" + key)
    model = runtime.build_model(run.get("actual_model_type", run["model"]), num_classes=6, seq_len=6,
                                raw_signal_len=plan["preprocessing"]["raw_signal_len"], **run["model_kwargs"])
    model.load_state_dict(checkpoint["model_state_dict"], strict=True)
    del checkpoint
    model.to(device).eval()
    amp = bool(config.get("amp", False)) and device.type == "cuda"
    return torch, model, device, amp


def forward(torch, model, device, amp, merged, raw):
    # Copy only this batch. Shared mmap arrays remain read-only and on disk.
    merged_tensor = torch.from_numpy(np.array(merged, dtype=np.float32, copy=True)).to(device)
    raw_tensor = torch.from_numpy(np.array(raw, dtype=np.float32, copy=True)).to(device)
    context = torch.cuda.amp.autocast(enabled=True) if amp else contextlib.nullcontext()
    with torch.inference_mode(), context:
        logits, embedding, auxiliary = model((merged_tensor[:, :, :5], merged_tensor, raw_tensor), return_aux=True)
    result = (logits.float().cpu().numpy(), embedding.float().cpu().numpy(),
              auxiliary["attention"].float().cpu().numpy())
    if not all(np.isfinite(a).all() for a in result):
        raise ValueError("预测输出含非有限值，停止而不丢弃窗口")
    if result[0].ndim != 2 or result[0].shape[1] != 6 or result[2].shape != (len(result[0]), 6):
        raise ValueError("模型输出形状不符合冻结六分类接口")
    return result


def environment_info(torch, device, amp):
    code_files = ["figure4_worker.py", "figure4_common.py", "prepare_figure4.py"]
    correction_prepare = Path(__file__).parent / "prepare_figure4_correction.py"
    if correction_prepare.is_file():
        code_files.append(correction_prepare.name)
    return {"python": sys.version, "platform": platform.platform(), "numpy": np.__version__,
            "torch": str(torch.__version__), "cuda_build": torch.version.cuda,
            "device": str(device), "amp_enabled": amp,
            "device_name": torch.cuda.get_device_name(device) if device.type == "cuda" else platform.processor(),
            "cuda_visible_devices": os.environ.get("CUDA_VISIBLE_DEVICES"),
            "worker_version": "1.1.0",
            "code_sha256": {name: sha256(Path(__file__).parent / name) for name in code_files}}


def predict_dataset(plan, run, dataset, args, torch, model, device, amp):
    folder = Path(args.output_dir) / "predictions" / run["run_id"] / dataset["dataset_id"]
    settings = prediction_settings(args)
    if validate_completion(folder, plan["plan_id"], run["run_id"], dataset["dataset_id"], settings, dataset["n_samples"]):
        print("[SKIP] " + run["run_id"] + " " + dataset["dataset_id"], flush=True)
        return
    replaced = replacement_dataset_ids(plan)
    if replaced is not None and dataset["dataset_id"] not in replaced:
        raise RuntimeError("应复用的未更正预测缺失或校验失败：" + str(folder) +
                           "；请恢复旧结果并重新准备更正输出，不读取旧缓存、不自动重跑未更正数据。")
    root, (merged, raw, labels) = open_cache(dataset, plan["preprocessing"]["raw_signal_len"])
    temporary = folder.with_name(folder.name + ".partial")
    temporary.parent.mkdir(parents=True, exist_ok=True)
    if temporary.exists():
        shutil.rmtree(temporary)
    temporary.mkdir()
    n, started, arrays = len(labels), time.monotonic(), {}
    forward_seconds = 0.0
    if device.type == "cuda":
        torch.cuda.reset_peak_memory_stats(device)
    for name, shape, dtype in [("logits", (n, 6), "float32"), ("labels", (n,), "int64")]:
        arrays[name] = np.lib.format.open_memmap(temporary / (name + ".npy"), mode="w+", dtype=dtype, shape=shape)
    done = 0
    with (root / "metadata.jsonl").open(encoding="utf-8") as meta, gzip.open(
            temporary / "predictions.csv.gz", "wt", encoding="utf-8", newline="", compresslevel=4) as output:
        writer = csv.DictWriter(output, fieldnames=CSV_FIELDS)
        writer.writeheader()
        for start in range(0, n, args.batch_size):
            end = min(n, start + args.batch_size)
            forward_start = time.monotonic()
            logits, embeddings, attention = forward(torch, model, device, amp, merged[start:end], raw[start:end])
            forward_seconds += time.monotonic() - forward_start
            probs, calibrated = softmax(logits), softmax(logits, run["temperature"])
            arrays["logits"][start:end], arrays["labels"][start:end] = logits, labels[start:end]
            for name, enabled, values, dtype in [("embeddings", not args.no_embeddings, embeddings, args.embedding_dtype),
                                                  ("attention", not args.no_attention, attention, "float32")]:
                if enabled:
                    if name not in arrays:
                        arrays[name] = np.lib.format.open_memmap(temporary / (name + ".npy"), mode="w+",
                                                               dtype=dtype, shape=(n,) + values.shape[1:])
                    if dtype == "float16" and np.max(np.abs(values)) > np.finfo(np.float16).max:
                        raise ValueError("embedding超过float16范围，请用新输出目录并设置--embedding_dtype float32")
                    arrays[name][start:end] = values
            for index in range(end - start):
                line = meta.readline()
                if not line:
                    raise ValueError("缓存metadata缺行")
                record = json.loads(line)
                if record["sample_index"] != start + index or int(record["label"]) != int(labels[start + index]):
                    raise ValueError("缓存metadata/特征/标签逐行不匹配")
                row = {key: record[key] for key in META_FIELDS}
                row.update(model=run["model"], seed=run["seed"], feature_scheme=run["feature_scheme"])
                for prefix, values in [("logit", logits), ("prob", probs), ("calibrated_prob", calibrated)]:
                    row.update({f"{prefix}_{j}": float(values[index, j]) for j in range(6)})
                row.update(prediction=int(probs[index].argmax()), confidence=float(probs[index].max()),
                           calibrated_confidence=float(calibrated[index].max()))
                writer.writerow(row)
            done = end
            if start == 0 or end == n or end % (args.batch_size * 100) == 0:
                print(f"[PREDICT] {run['run_id']} {dataset['dataset_id']} {end:,}/{n:,}", flush=True)
        if meta.readline():
            raise ValueError("缓存metadata多行")
    for array in arrays.values():
        array.flush()
    arrays.clear()
    # A marker is written only after gzip trailers and all arrays are complete.
    outputs = {path.name: {"size_bytes": path.stat().st_size, "sha256": sha256(path)}
               for path in temporary.iterdir() if path.is_file()}
    check_inputs(dataset)
    elapsed = time.monotonic() - started
    atomic_json(temporary / "completion.json", {"status": "complete", "plan_id": plan["plan_id"],
        "run_id": run["run_id"], "dataset_id": dataset["dataset_id"], "n_samples": done,
        "prediction_settings": settings, "checkpoint_sha256": run["checkpoint_sha256"],
        "temperature": run["temperature"], "outputs": outputs, "seconds": elapsed,
        "forward_seconds_including_batch_transfer": forward_seconds,
        "end_to_end_samples_per_second": done / elapsed,
        "forward_samples_per_second_including_batch_transfer": done / forward_seconds,
        "gpu_peak_reserved_bytes": int(torch.cuda.max_memory_reserved(device)) if device.type == "cuda" else 0,
        "gpu_peak_allocated_bytes": int(torch.cuda.max_memory_allocated(device)) if device.type == "cuda" else 0,
        "timing_scope": "end-to-end includes streaming CSV/array export and output hashing; forward includes host/device transfer and auxiliaries; concurrent jobs may contend",
        "completed_at": datetime.now(timezone.utc).isoformat(), "environment": environment_info(torch, device, amp)})
    if folder.exists():
        shutil.rmtree(folder)
    temporary.replace(folder)


def profile(plan, run, args, torch, model, device, amp):
    import resource_scheduler as telemetry
    replaced = replacement_dataset_ids(plan)
    candidates = [dataset for dataset in plan["datasets"]
                  if replaced is None or dataset["dataset_id"] in replaced]
    if not candidates:
        raise ValueError("没有可实测的更正输入；不能使用未更正数据的旧缓存进行资源探测")
    dataset = max(candidates, key=lambda item: item["n_samples"])
    _, (merged, raw, labels) = open_cache(dataset, plan["preprocessing"]["raw_signal_len"])
    indices = np.arange(args.batch_size) % len(labels)
    # Always exercise a full requested batch, including on small test fixtures.
    batch_merged, batch_raw = np.asarray(merged[indices]), np.asarray(raw[indices])
    if device.type == "cuda":
        torch.cuda.synchronize(device)
        baseline = torch.cuda.memory_reserved(device)
        torch.cuda.reset_peak_memory_stats(device)
    else:
        baseline = 0
    start, batches = time.monotonic(), 0
    while batches < args.profile_batches or time.monotonic() - start < args.profile_min_seconds:
        logits, embedding, attention = forward(torch, model, device, amp, batch_merged, batch_raw)
        softmax(logits), softmax(logits, run["temperature"])
        if not args.no_embeddings:
            embedding.astype(args.embedding_dtype)
        batches += 1
    if device.type == "cuda":
        torch.cuda.synchronize(device)
    peak_host = int(resource.getrusage(resource.RUSAGE_SELF).ru_maxrss * (1 if sys.platform == "darwin" else 1024))
    peak_host = max(peak_host, telemetry.process_rss_bytes(os.getpid()))
    driver, driver_note = 0, "cpu"
    if device.type == "cuda":
        try:
            directory = telemetry.proc_directory(os.getpid())
            pids = {os.getpid()}
            if directory is not None:
                pids.add(int(directory.name))
            snapshot = telemetry.nvidia_snapshot()
            driver = max((gpu["pid_bytes"].get(pid, 0) for gpu in snapshot["devices"].values() for pid in pids), default=0)
            driver_note = "nvidia_smi_process_bytes" if driver else "process_bytes_unavailable_conservative_context"
        except Exception as error:
            driver_note = "unavailable: " + str(error)
    reserved = int(torch.cuda.max_memory_reserved(device)) if device.type == "cuda" else 0
    allocated = int(torch.cuda.max_memory_allocated(device)) if device.type == "cuda" else 0
    context = max(0, driver - int(torch.cuda.memory_reserved(device))) if driver else (GIB if device.type == "cuda" else 0)
    elapsed = time.monotonic() - start
    value = {"status": "complete", "mode": "resource_profile_only_no_metric_selection", "plan_id": plan["plan_id"],
             "run_id": run["run_id"], "device": str(device), "measured_batches": batches,
             "profile_dataset_id": dataset["dataset_id"],
             "profile_dataset_scope": "replacement_datasets_only" if replaced is not None else "all_planned_datasets",
             "measured_seconds": elapsed, "batch_size": args.batch_size,
             "parameter_count": sum(p.numel() for p in model.parameters()),
             "state_tensor_bytes": sum(p.numel() * p.element_size() for p in model.state_dict().values()),
             "samples_per_second_including_warmup_and_transfers": batches * args.batch_size / elapsed,
             "timing_scope": "serial resource probe, includes warmup and auxiliaries; not a standardized deployment latency benchmark",
             "host_peak_rss_bytes": peak_host, "host_admission_estimate_bytes": peak_host + 256 * 1024 ** 2,
             "gpu_peak_reserved_bytes": reserved, "gpu_peak_allocated_bytes": allocated,
             "gpu_context_baseline_bytes": context, "gpu_driver_delta_bytes": driver,
             "gpu_admission_estimate_bytes": max(driver, reserved + context), "driver_telemetry": driver_note,
             "prediction_settings": prediction_settings(args), "environment": environment_info(torch, device, amp),
             "shared_cache_accounting": "parent reserves min(total_cache_bytes,max_file_cache_bytes*parallel) once"}
    atomic_json(args.profile_out, value)
    print("[PROFILE] " + json.dumps(value, ensure_ascii=False), flush=True)


def parser():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--plan", required=True)
    p.add_argument("--run_id", required=True)
    p.add_argument("--mode", choices=("profile", "predict"), required=True)
    p.add_argument("--output_dir", required=True)
    p.add_argument("--device", default="cpu")
    p.add_argument("--batch_size", type=int, default=256)
    p.add_argument("--workers", type=int, default=0)
    p.add_argument("--cpu_threads", type=int, default=4)
    p.add_argument("--profile_batches", type=int, default=8)
    p.add_argument("--profile_min_seconds", type=float, default=5)
    p.add_argument("--profile_out")
    p.add_argument("--embedding_dtype", choices=("float16", "float32"), default="float16")
    p.add_argument("--no_embeddings", action="store_true")
    p.add_argument("--no_attention", action="store_true")
    return p


def main():
    args = parser().parse_args()
    if min(args.batch_size, args.cpu_threads, args.profile_batches) < 1 or args.workers != 0 or not math.isfinite(args.profile_min_seconds) or args.profile_min_seconds < 0:
        raise ValueError("batch/threads/profile_batches须正数；流式mmap预测请保持--workers 0")
    if args.mode == "profile" and not args.profile_out:
        raise ValueError("profile模式需要--profile_out")
    plan = read_json(args.plan)
    replaced = validate_plan_settings(plan, args)
    run = next((r for r in plan["runs"] if r["run_id"] == args.run_id), None)
    if run is None:
        raise ValueError("冻结计划中没有此run_id")
    torch, model, device, amp = load_model(plan, run, args)
    if args.mode == "profile":
        profile(plan, run, args, torch, model, device, amp)
    else:
        for dataset in plan["datasets"]:
            predict_dataset(plan, run, dataset, args, torch, model, device, amp)
        verify_run(run)
        atomic_json(Path(args.output_dir) / "predictions" / run["run_id"] / "run_complete.json",
                    {"status": "complete", "plan_id": plan["plan_id"], "run_id": run["run_id"],
                     "prediction_settings": prediction_settings(args), "datasets": [d["dataset_id"] for d in plan["datasets"]],
                     "replacement_dataset_ids": sorted(replaced) if replaced is not None else None,
                     "completed_at": datetime.now(timezone.utc).isoformat()})
        print("[RUN COMPLETE] " + run["run_id"], flush=True)


if __name__ == "__main__":
    try:
        main()
    except Exception as error:
        print(f"[ERROR] {type(error).__name__}: {error}", file=sys.stderr, flush=True)
        raise
