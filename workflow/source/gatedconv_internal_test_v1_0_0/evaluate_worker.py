#!/usr/bin/env python3
"""Frozen GatedConv inference and CPU-only export; never trains or fits a temperature."""
from __future__ import annotations

import argparse
import contextlib
import importlib
import json
import math
import os
from pathlib import Path
import platform
import resource
import shutil
import sys
import time

from common import (CLASSES, HERE, SEEDS, VENDOR, atomic_json, read_json,
                    sha256, timestamp, verify_plan)

GIB = 1024 ** 3
ARRAY_NAMES = ("logits.npy", "labels.npy", "embeddings.npy", "attention.npy")


def run_directory(output_dir, seed):
    return Path(output_dir) / "runs" / "gatedconv_target" / ("seed_" + str(seed))


def _run(plan, seed):
    matches = [run for run in plan["runs"] if int(run["seed"]) == int(seed)]
    if len(matches) != 1:
        raise ValueError("Frozen plan must contain exactly one requested seed")
    return matches[0]


def file_manifest(root, names=None):
    root = Path(root)
    paths = ([root / name for name in names] if names is not None else
             [p for p in root.rglob("*") if p.is_file()])
    result = {}
    for path in sorted(paths):
        if not path.is_file() or path.is_symlink():
            raise ValueError("Missing or linked evaluation artifact: " + str(path))
        result[str(path.relative_to(root))] = {
            "size": path.stat().st_size, "sha256": sha256(path)}
    return result


def _verify_files(root, files):
    root = Path(root)
    if not isinstance(files, dict) or not files:
        raise ValueError("Completion marker has no artifact manifest")
    for relative, info in files.items():
        name = Path(relative)
        if name.is_absolute() or ".." in name.parts:
            raise ValueError("Invalid artifact path in completion marker")
        path = root / name
        if (not path.is_file() or path.is_symlink() or
                path.stat().st_size != int(info["size"]) or
                sha256(path) != info["sha256"]):
            raise ValueError("Evaluation artifact changed or is incomplete: " + str(path))


def _identity_matches(marker, run, plan):
    if (marker.get("plan_id") != plan["plan_id"] or
            int(marker.get("seed", -1)) != int(run["seed"]) or
            marker.get("checkpoint_sha256") != run["checkpoint_sha256"] or
            float(marker.get("temperature", -1)) != float(run["temperature"]) or
            marker.get("n_samples") != int(plan["dataset"]["n_samples"])):
        raise ValueError("Completion marker does not match the frozen evaluation")


def validate_prediction_complete(output_dir, seed, plan, *, strict=False):
    """Validate reusable inference arrays without importing torch."""
    directory = run_directory(output_dir, seed) / "test"
    path = directory / "inference_complete.json"
    try:
        marker = read_json(path)
        _identity_matches(marker, _run(plan, seed), plan)
        if marker.get("status") != "inference_complete" or set(marker.get("files", {})) != set(ARRAY_NAMES):
            raise ValueError("Incomplete prediction array manifest")
        _verify_files(directory, marker["files"])
        return True
    except (OSError, ValueError, KeyError, TypeError) as exc:
        if strict:
            raise RuntimeError("Cannot reuse prediction arrays for seed " + str(seed)) from exc
        return False


def validate_run_complete(output_dir, seed, plan, *, strict=False):
    """Validate complete metrics/plots/probabilities and arrays without torch."""
    directory = run_directory(output_dir, seed)
    try:
        marker = read_json(directory / "run_complete.json")
        _identity_matches(marker, _run(plan, seed), plan)
        if marker.get("status") != "complete" or marker.get("test_evaluated") is not True:
            raise ValueError("Run has not completed its frozen internal test")
        if set(marker.get("metrics_by_split", {})) != {"test"}:
            raise ValueError("Only the frozen internal test may be exported here")
        _verify_files(directory / "test", marker["files"])
        return True
    except (OSError, ValueError, KeyError, TypeError) as exc:
        if strict:
            raise RuntimeError("Cannot reuse completed test export for seed " + str(seed)) from exc
        return False


def _runtime():
    # Fixed local vendor modules only. No implicit import from a user's old code folder.
    sys.path.insert(0, str(VENDOR))
    return (importlib.import_module("train_all"),
            importlib.import_module("IterDataset"),
            importlib.import_module("model_benchmark_all"))


def _choose_device(torch, requested):
    if requested == "auto":
        requested = "cuda:0" if torch.cuda.is_available() else "cpu"
    device = torch.device(requested)
    if device.type not in ("cpu", "cuda"):
        raise ValueError("Only CPU and CUDA evaluation are supported")
    if device.type == "cuda":
        if not torch.cuda.is_available():
            raise RuntimeError("CUDA was explicitly requested but is unavailable")
        torch.cuda.set_device(device)
        torch.cuda.init()
        torch.empty(1, device=device)
        torch.cuda.synchronize(device)
    return device


def snapshot(args):
    """Read free memory from the requested logical CUDA device directly."""
    import torch
    sys.path.insert(0, str(VENDOR))
    telemetry = importlib.import_module("resource_scheduler")
    device = _choose_device(torch, args.device)
    value = {"status": "complete", "mode": "snapshot", "created_at": timestamp(),
             "chosen_device": str(device), "host": telemetry.memory_snapshot(),
             "available_cpu_slots": telemetry.available_cpu_slots(),
             "cuda_visible_devices": os.environ.get("CUDA_VISIBLE_DEVICES")}
    if device.type == "cuda":
        free, total = torch.cuda.mem_get_info(device)
        value["gpu"] = {"free_bytes": int(free), "total_bytes": int(total),
                        "name": torch.cuda.get_device_name(device),
                        "mapping_policy": "torch logical device queried in its own CUDA context; no physical-index guessing"}
    else:
        value["gpu"] = None
    if args.profile_out:
        atomic_json(args.profile_out, value)
    print(json.dumps(value, ensure_ascii=False), flush=True)
    return value


def _verify_cache(plan):
    root = Path(plan["dataset"]["cache_path"])
    hashes = plan["dataset"].get("cache_files_sha256", {})
    required = {"audit.json", "merged.npy", "raw.npy", "labels.npy",
                "metadata_offsets.npy", "manifest.jsonl"}
    if not required.issubset(hashes):
        raise ValueError("Frozen test cache has an incomplete content manifest")
    for name, item in hashes.items():
        relative = Path(name)
        if relative.is_absolute() or ".." in relative.parts:
            raise ValueError("Invalid frozen cache path")
        digest = item.get("sha256") if isinstance(item, dict) else item
        if sha256(root / name) != digest:
            raise ValueError("Frozen test cache changed: " + name)


def _args_for_run(train, run, plan, args, device):
    values = vars(train.parser().parse_args([]))
    values.update(run["config"])
    settings = plan["prediction_settings"]
    values.update(mode="evaluate", seed=int(run["seed"]), model_type="litemamba",
                  output_dir=str(run_directory(args.output_dir, run["seed"])),
                  batch_size=int(settings["batch_size"]), workers=int(settings["workers"]),
                  cpu_threads=int(settings["cpu_threads"]), device=str(device),
                  embedding_dtype=settings["embedding_dtype"], no_embeddings=False,
                  no_attention=False, force_evaluate=False, bootstrap=int(args.bootstrap),
                  no_plots=False, amp=bool(run["config"].get("amp")) and device.type == "cuda")
    return argparse.Namespace(**values)


def _load(plan, run, args):
    import numpy as np
    import torch
    train, dataset_runtime, model_runtime = _runtime()
    torch.set_num_threads(int(plan["prediction_settings"]["cpu_threads"]))
    device = _choose_device(torch, args.device)
    train.set_seed(int(run["seed"]))
    if device.type == "cuda":
        torch.backends.cudnn.benchmark = False
        torch.backends.cudnn.deterministic = True
        torch.backends.cuda.matmul.allow_tf32 = False
        torch.backends.cudnn.allow_tf32 = False
    resolved = _args_for_run(train, run, plan, args, device)
    if sha256(run["checkpoint"]) != run["checkpoint_sha256"]:
        raise ValueError("Frozen checkpoint changed")
    checkpoint = train.torch_load(run["checkpoint"], "cpu")
    recorded = checkpoint.get("resolved_config")
    if not isinstance(recorded, dict):
        raise ValueError("Checkpoint lacks its frozen training configuration")
    for key in ("model_type", "seed", "feature_scheme", "model_kwargs", "target_index",
                "class_names", "raw_signal_len", "dwell_divisor", "quality_divisor"):
        if recorded.get(key) != run["config"].get(key):
            raise ValueError("Checkpoint and original run configuration differ: " + key)
    model = model_runtime.build_model("litemamba", num_classes=6, seq_len=6,
                raw_signal_len=int(run["config"]["raw_signal_len"]), **run["config"]["model_kwargs"])
    model.load_state_dict(checkpoint["model_state_dict"], strict=True)
    del checkpoint
    model.to(device).eval()
    dataset = dataset_runtime.CachedFeatureDataset(plan["dataset"]["cache_path"], 0)
    if len(dataset) != int(plan["dataset"]["n_samples"]):
        raise ValueError("Frozen test cache sample count changed")
    return torch, train, dataset, model, device, resolved


def _environment(torch, device, amp):
    return {"python": sys.version, "platform": platform.platform(),
            "torch": str(torch.__version__), "cuda_build": torch.version.cuda,
            "device": str(device), "amp_enabled": bool(amp),
            "device_name": torch.cuda.get_device_name(device) if device.type == "cuda" else platform.processor(),
            "cuda_visible_devices": os.environ.get("CUDA_VISIBLE_DEVICES"),
            "scope": "frozen internal-test evaluation; no optimizer or temperature fitting"}


def profile(plan, run, args):
    import numpy as np
    torch, train, dataset, model, device, resolved = _load(plan, run, args)
    indices = np.arange(resolved.batch_size) % len(dataset)
    batch = next(iter(train.loader(dataset, resolved)))
    # For real fixed test N > batch size; repeat only for small integration fixtures.
    if len(batch["label"]) < resolved.batch_size:
        from torch.utils.data._utils.collate import default_collate
        batch = default_collate([dataset[int(i)] for i in indices])
    inputs, _ = train.batch_to_device(batch, device)
    if device.type == "cuda":
        torch.cuda.synchronize(device)
        torch.cuda.reset_peak_memory_stats(device)
    start = time.monotonic()
    with torch.no_grad():
        for _ in range(8):
            with train.autocast_context(resolved, device):
                output, embedding, auxiliary = model(inputs, return_aux=True)
            if not all(torch.isfinite(x).all().item() for x in (output, embedding, auxiliary["attention"])):
                raise ValueError("Non-finite output during inference-only resource probe")
    if device.type == "cuda":
        torch.cuda.synchronize(device)
    peak = int(resource.getrusage(resource.RUSAGE_SELF).ru_maxrss *
               (1 if sys.platform == "darwin" else 1024))
    reserved = int(torch.cuda.max_memory_reserved(device)) if device.type == "cuda" else 0
    allocated = int(torch.cuda.max_memory_allocated(device)) if device.type == "cuda" else 0
    value = {"status": "complete", "mode": "inference_resource_profile_only",
             "plan_id": plan["plan_id"], "seed": run["seed"], "chosen_device": str(device),
             "checkpoint_sha256": run["checkpoint_sha256"], "batch_size": resolved.batch_size,
             "measured_batches": 8, "measured_seconds": time.monotonic() - start,
             "host_peak_rss_bytes": peak, "host_admission_estimate_bytes": peak + 256 * 1024 ** 2,
             "gpu_peak_reserved_bytes": reserved, "gpu_peak_allocated_bytes": allocated,
             "gpu_admission_estimate_bytes": reserved + GIB if device.type == "cuda" else 0,
             "gpu_context_allowance_bytes": GIB if device.type == "cuda" else 0,
             "parameter_count": sum(p.numel() for p in model.parameters()),
             "environment": _environment(torch, device, resolved.amp),
             "timing_scope": "resource admission only; includes warmup and does not constitute an isolated efficiency benchmark",
             "created_at": timestamp()}
    atomic_json(args.profile_out, value)
    dataset.close()
    print("[PROFILE] " + json.dumps(value, ensure_ascii=False), flush=True)
    return value


def _check_arrays(path, plan):
    import numpy as np
    n = int(plan["dataset"]["n_samples"])
    shapes = {"logits.npy": (n, 6), "labels.npy": (n,),
              "embeddings.npy": (n, 192), "attention.npy": (n, 6)}
    for name, shape in shapes.items():
        value = np.load(Path(path) / name, mmap_mode="r", allow_pickle=False)
        if value.shape != shape or value.dtype != (np.int64 if name == "labels.npy" else np.float32):
            raise ValueError("Frozen output shape/dtype mismatch: " + name)
        if name == "labels.npy":
            counts = np.bincount(value, minlength=6)
            expected = plan["dataset"].get("class_counts")
            if isinstance(expected, dict):
                expected = [expected.get(str(i), expected.get(CLASSES[i], 0)) for i in range(6)]
            if expected is not None and counts.tolist() != list(expected):
                raise ValueError("Test class counts differ from the frozen split")
        else:
            for start in range(0, n, 65536):
                if not np.isfinite(value[start:start + 65536]).all():
                    raise ValueError("Non-finite saved output: " + name)


def predict(plan, run, args):
    directory = run_directory(args.output_dir, run["seed"])
    final = directory / "test"
    if (final / "inference_complete.json").exists():
        validate_prediction_complete(args.output_dir, run["seed"], plan, strict=True)
        print("[SKIP PREDICT] seed=" + str(run["seed"]), flush=True)
        return
    if final.exists() and any(final.iterdir()):
        raise RuntimeError("Unmarked final test directory exists; preserve it for inspection: " + str(final))
    partial = directory / "test.partial"
    directory.mkdir(parents=True, exist_ok=True)
    if partial.exists():
        shutil.rmtree(partial)
    partial.mkdir()
    atomic_json(directory / "test_started.json", {"plan_id": plan["plan_id"],
        "seed": run["seed"], "checkpoint_sha256": run["checkpoint_sha256"],
        "started_at": timestamp(), "scope": "this frozen output only; original training run remains read-only"})
    torch, train, dataset, model, device, resolved = _load(plan, run, args)
    labels, logits, seconds = train.infer_arrays(model, dataset, resolved, device, partial)
    del labels, logits
    _check_arrays(partial, plan)
    if sha256(run["checkpoint"]) != run["checkpoint_sha256"]:
        raise ValueError("Checkpoint changed during inference")
    _verify_cache(plan)
    marker = {"status": "inference_complete", "plan_id": plan["plan_id"],
        "seed": run["seed"], "model": "gatedconv_target", "n_samples": len(dataset),
        "checkpoint_sha256": run["checkpoint_sha256"], "temperature": run["temperature"],
        "inference_seconds": seconds, "environment": _environment(torch, device, resolved.amp),
        "files": file_manifest(partial, ARRAY_NAMES), "completed_at": timestamp()}
    atomic_json(partial / "inference_complete.json", marker)
    if final.exists():
        final.rmdir()
    partial.rename(final)
    dataset.close()
    print("[PREDICT COMPLETE] seed=" + str(run["seed"]) + " n=" + str(marker["n_samples"]), flush=True)


def analyze(plan, run, args):
    import numpy as np
    if (run_directory(args.output_dir, run["seed"]) / "run_complete.json").exists():
        validate_run_complete(args.output_dir, run["seed"], plan, strict=True)
        print("[SKIP ANALYZE] seed=" + str(run["seed"]), flush=True)
        return
    validate_prediction_complete(args.output_dir, run["seed"], plan, strict=True)
    train, data_runtime, _ = _runtime()
    import torch
    # This stage does not instantiate a model and never opens a CUDA context.
    torch.set_num_threads(int(plan["prediction_settings"]["cpu_threads"]))
    directory = run_directory(args.output_dir, run["seed"])
    out = directory / "test"
    dataset = data_runtime.CachedFeatureDataset(plan["dataset"]["cache_path"], 0)
    resolved = _args_for_run(train, run, plan, args, torch.device("cpu"))
    marker = read_json(out / "inference_complete.json")
    labels = np.load(out / "labels.npy", mmap_mode="r", allow_pickle=False)
    logits = np.load(out / "logits.npy", mmap_mode="r", allow_pickle=False)
    info = train.finish_export(dataset, resolved, "test", out, labels, logits,
            run["temperature"], run["checkpoint_sha256"], marker["inference_seconds"])
    # Ensure the analysis consumed precisely the preserved inference, not changed arrays.
    validate_prediction_complete(args.output_dir, run["seed"], plan, strict=True)
    if info["metrics"]["raw"]["n"] != int(plan["dataset"]["n_samples"]):
        raise ValueError("Exported metrics have an unexpected test sample count")
    result = {"status": "complete", "test_evaluated": True, "plan_id": plan["plan_id"],
        "model": "gatedconv_target", "seed": run["seed"], "feature_scheme": "full",
        "n_samples": int(plan["dataset"]["n_samples"]),
        "checkpoint_sha256": run["checkpoint_sha256"], "temperature": run["temperature"],
        "source_run_dir": run["run_dir"], "dataset_identity": info["dataset_identity"],
        "metrics_by_split": {"test": info["metrics"]}, "exports_by_split": {"test": info},
        "inference_environment": marker["environment"],
        "bootstrap_replicates": int(args.bootstrap), "files": file_manifest(out),
        "completed_at": timestamp()}
    atomic_json(directory / "run_complete.json", result)
    dataset.close()
    print("[ANALYZE COMPLETE] seed=" + str(run["seed"]), flush=True)


def parser():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--mode", choices=("snapshot", "profile", "predict", "analyze"), required=True)
    p.add_argument("--output_dir")
    p.add_argument("--seed", type=int, choices=SEEDS)
    p.add_argument("--device", default="auto")
    p.add_argument("--cpu_threads", type=int, default=4)
    p.add_argument("--profile_out")
    p.add_argument("--bootstrap", type=int, default=200)
    return p


def main(argv=None):
    args = parser().parse_args(argv)
    if args.cpu_threads < 1 or args.bootstrap < 0:
        raise ValueError("CPU threads must be positive; bootstrap must be nonnegative")
    if args.mode == "snapshot":
        return snapshot(args)
    if args.output_dir is None or args.seed is None:
        raise ValueError("Evaluation modes require output_dir and seed")
    if args.mode == "profile" and not args.profile_out:
        raise ValueError("profile requires profile_out")
    plan = read_json(Path(args.output_dir) / "evaluation_plan.json")
    verify_plan(plan)
    if int(plan["prediction_settings"]["cpu_threads"]) != args.cpu_threads:
        raise ValueError("CPU thread count differs from the frozen prediction plan")
    frozen_bootstrap = plan.get("analysis_settings", {}).get("bootstrap", 200)
    if args.bootstrap != frozen_bootstrap:
        raise ValueError("Bootstrap count differs from the frozen analysis settings")
    _verify_cache(plan)
    run = _run(plan, args.seed)
    if args.mode == "profile":
        return profile(plan, run, args)
    if args.mode == "predict":
        return predict(plan, run, args)
    return analyze(plan, run, args)


if __name__ == "__main__":
    try:
        main()
    except Exception as error:
        import traceback
        traceback.print_exc()
        print("[ERROR] " + type(error).__name__ + ": " + str(error), file=sys.stderr, flush=True)
        raise SystemExit(1)
