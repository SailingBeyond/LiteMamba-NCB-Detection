#!/usr/bin/env python3
"""共享文件、身份和冻结配置处理；不在父进程初始化 CUDA。"""

from __future__ import annotations

import csv
import hashlib
import importlib.util
import json
import math
import os
from pathlib import Path
import re
import shutil
import sys
from decimal import Decimal, InvalidOperation

VERSION = "1.1.0"
HERE = Path(__file__).resolve().parent
VENDOR = HERE / "vendor"
CLASSES = ["A", "T", "C", "G", "X", "Y"]
MODELS = ["litemamba", "transformer", "bigru", "convmixer"]
SEEDS = [42, 2024, 2025, 2026, 2027]
UNKNOWN = {"", "unknown", "na", "n/a", "none", "null", "nan", "."}
DEFAULT_RESULTS = "/path/to/revision-inputs/CPLX-XNA-replenishment/CPLX-XNA-replenishment/fast5_single_guppy/sixclass_overlap_read_split_721/sixclass_auto_v1_0_4_results/main_20"
DEFAULT_01_12 = "/path/to/staged-storage2/analysis/paper/xna_pc_6mer_features_for_model_test_X_to_T"
DEFAULT_13_16 = "/path/to/revision-inputs/POC-XNA_DNA-01_16/POC-XNA_DNA-01_16/fast5_single_guppy/xna_pc_13_16_6"
DEFAULT_17_20 = "/path/to/staged-storage2/analysis/paper/POC-XNA-17_20/fast5_single_guppy/xna_6mer_features_17_20_split_by_ref_strand"


def read_json(path):
    return json.loads(Path(path).read_text(encoding="utf-8"))


def atomic_json(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + ".tmp." + str(os.getpid()))
    with temporary.open("w", encoding="utf-8") as stream:
        json.dump(value, stream, ensure_ascii=False, indent=2, allow_nan=False)
        stream.write("\n")
        stream.flush()
        os.fsync(stream.fileno())
    temporary.replace(path)


def sha256(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(8 * 1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def fingerprint(path):
    path = Path(path).resolve()
    before = path.stat()
    digest, lines, last = hashlib.sha256(), 0, b""
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(8 * 1024 * 1024), b""):
            digest.update(block)
            lines += block.count(b"\n")
            last = block[-1:]
    after = path.stat()
    if (before.st_size, before.st_mtime_ns) != (after.st_size, after.st_mtime_ns):
        raise RuntimeError("文件在读取时发生变化：" + str(path))
    return {"path": str(path), "sha256": digest.hexdigest(), "size_bytes": before.st_size,
            "mtime_ns": before.st_mtime_ns, "physical_lines": lines + int(bool(last) and last != b"\n")}


def identity_hash(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True, ensure_ascii=False, separators=(",", ":")).encode()).hexdigest()


def known(value):
    return str(value if value is not None else "").strip().lower() not in UNKNOWN


def strict_label(value, where):
    try:
        number = Decimal(str(value).strip())
    except InvalidOperation as error:
        raise ValueError(f"{where}：label 不是整数") from error
    if not number.is_finite() or number != number.to_integral_value() or not 0 <= number <= 5:
        raise ValueError(f"{where}：六分类 label 必须为0..5整数，可接受1.0但不接受1.2")
    return int(number)


def ensure_disk(path, required_bytes=0, reserve_gib=5.0):
    path = Path(path)
    path.mkdir(parents=True, exist_ok=True)
    free = shutil.disk_usage(path).free
    reserve = int(float(reserve_gib) * 1024 ** 3)
    if free < required_bytes + reserve:
        raise OSError(28, f"空间不足：{path} 可用{free / 1024**3:.2f}GiB，预计写入{required_bytes / 1024**3:.2f}GiB，另保留{reserve_gib}GiB")


def runtime_modules(run=None):
    """每套模型源码使用独立模块名，避免full与门控运行时互相串用。"""
    directory = Path(run.get("runtime_dir", VENDOR) if run else VENDOR).resolve()
    prefix = "_figure4_runtime_" + identity_hash(str(directory))[:16] + "_"
    modules = []
    for filename in ("IterDataset.py", "model_benchmark_all.py"):
        module_name = prefix + Path(filename).stem
        if module_name not in sys.modules:
            spec = importlib.util.spec_from_file_location(module_name, directory / filename)
            if spec is None or spec.loader is None:
                raise ImportError("不能加载冻结运行时：" + str(directory / filename))
            module = importlib.util.module_from_spec(spec)
            sys.modules[module_name] = module
            try:
                spec.loader.exec_module(module)
            except BaseException:
                sys.modules.pop(module_name, None)
                raise
        modules.append(sys.modules[module_name])
    return tuple(modules)


def torch_load(path):
    import torch
    try:
        return torch.load(path, map_location="cpu", weights_only=False)
    except TypeError:
        return torch.load(path, map_location="cpu")


def verify_run(run):
    """检查原始 checkpoint、val 温度和配置是否仍与计划一致。"""
    for field, hash_field in [("checkpoint", "checkpoint_sha256"),
                              ("resolved_config", "resolved_config_sha256"),
                              ("calibration", "calibration_sha256")]:
        if sha256(run[field]) != run[hash_field]:
            raise RuntimeError(f"冻结后文件改变：{run['run_id']} {field}")
    for name, expected in run["runtime_sha256"].items():
        if sha256(Path(run.get("runtime_dir", VENDOR)) / name) != expected:
            raise RuntimeError("冻结模型运行时发生变化：" + name)
    for name, expected in run.get("frozen_metadata_sha256", {}).items():
        if Path(name).is_absolute() or ".." in Path(name).parts:
            raise ValueError("非法冻结元数据路径：" + name)
        if sha256(Path(run["run_dir"]) / name) != expected:
            raise RuntimeError("冻结训练元数据发生变化：" + str(Path(run["run_dir"]) / name))
    for name, expected in run.get("selection_sources", {}).items():
        if sha256(name) != expected:
            raise RuntimeError("冻结结构确认来源发生变化：" + name)


def validate_completion(folder, plan_id, run_id, dataset_id, settings, n_samples):
    """Only an intact, matching export may be reused after interruption."""
    folder = Path(folder)
    marker = folder / "completion.json"
    if not marker.exists():
        return False
    try:
        value = read_json(marker)
    except (ValueError, OSError):
        return False
    if value.get("status") != "complete":
        return False
    for key, expected in {"plan_id": plan_id, "run_id": run_id, "dataset_id": dataset_id,
                          "prediction_settings": settings, "n_samples": int(n_samples)}.items():
        if value.get(key) != expected:
            raise RuntimeError(f"预测完成记录与本次设置不一致：{folder} {key}；请使用新的输出目录")
    required = {"predictions.csv.gz", "logits.npy", "labels.npy"}
    if not settings["no_embeddings"]:
        required.add("embeddings.npy")
    if not settings["no_attention"]:
        required.add("attention.npy")
    outputs = value.get("outputs", {})
    if not required.issubset(outputs):
        return False
    for name, info in outputs.items():
        if Path(name).name != name:
            raise ValueError("非法输出清单路径：" + str(name))
        path = folder / name
        if not path.is_file() or path.stat().st_size != info.get("size_bytes") or sha256(path) != info.get("sha256"):
            return False
    return True


def discover_runs(training_results, models, seeds, feature_schemes):
    root = Path(training_results).expanduser().resolve()
    if (root / "main_20/queue_manifest.json").is_file():
        root = root / "main_20"
    manifest = read_json(root / "queue_manifest.json")
    if manifest.get("status") != "complete" or manifest.get("failed_runs"):
        raise ValueError("训练队列尚未全部完成；请先完成并冻结队列")
    frozen = read_json(root / "evaluation_plan.json")
    entries = {row["run_id"]: row for row in frozen["experiments"]}
    runs, preprocessing = [], None
    for scheme in feature_schemes:
        for model in models:
            for seed in seeds:
                run_id = f"{scheme}/{model}/seed_{seed}"
                directory = root / run_id
                config_path = directory / "resolved_config.json"
                config = read_json(config_path)
                result = read_json(directory / "results.json")
                calibration_path = directory / "calibration.json"
                calibration = read_json(calibration_path)
                checkpoint = directory / "best_checkpoint.pt"
                if result.get("status") != "complete" or not result.get("test_evaluated"):
                    raise ValueError("运行未完成：" + run_id)
                if config.get("model_type") != model or int(config.get("seed", -1)) != seed or config.get("feature_scheme") != scheme:
                    raise ValueError("训练目录和记录身份不一致：" + run_id)
                class_names = config.get("class_names")
                if isinstance(class_names, dict):
                    class_names = [class_names.get(str(i), class_names.get(i)) for i in range(6)]
                if class_names != CLASSES or int(config.get("target_index", -1)) != 3:
                    raise ValueError("当前入口只接受第四位N的新版六分类权重，不混入旧5X/5Y权重")
                current = {"seq_len": 6, "raw_signal_len": int(config["raw_signal_len"]),
                           "dwell_divisor": float(config["dwell_divisor"]),
                           "quality_divisor": float(config["quality_divisor"]), "mask_index": 3,
                           "mask_base": "N", "one_hot_base_order": ["A", "C", "T", "G", "N"],
                           "class_names": CLASSES, "signal_rectification": "symmetric_zero_pad_or_floor_linspace_indices",
                           "signal_round_decimals": 4}
                if preprocessing is not None and current != preprocessing:
                    raise ValueError("运行之间的输入预处理不同；请分开评估")
                preprocessing = current
                checkpoint_hash = sha256(checkpoint)
                expected = entries.get(run_id, {}).get("checkpoint_sha256")
                hashes = [checkpoint_hash, expected, calibration.get("checkpoint_sha256"), result["provenance"].get("checkpoint_sha256")]
                if len(set(hashes)) != 1:
                    raise ValueError("checkpoint 与测试冻结/结果/温度不一致：" + run_id)
                if calibration.get("fitted_on") != "val" or not math.isfinite(float(calibration["temperature"])) or float(calibration["temperature"]) <= 0:
                    raise ValueError("温度必须为仅用validation拟合的正数：" + run_id)
                source = config["source_snapshot"]["files_sha256"]
                runtime_hashes = {}
                for name in ("model_benchmark_all.py", "IterDataset.py"):
                    runtime_hashes[name] = sha256(VENDOR / name)
                    if source.get(name) != runtime_hashes[name]:
                        raise ValueError("提供的模型运行时与训练快照不匹配：" + name)
                kwargs = dict(config["model_kwargs"])
                profile_key = identity_hash({"model": model, "kwargs": kwargs, "preprocessing": preprocessing})[:20]
                runs.append({"run_id": run_id, "model": model, "seed": seed, "feature_scheme": scheme,
                             "profile_key": profile_key, "run_dir": str(directory), "checkpoint": str(checkpoint),
                             "checkpoint_sha256": checkpoint_hash, "resolved_config": str(config_path),
                             "resolved_config_sha256": sha256(config_path), "calibration": str(calibration_path),
                             "calibration_sha256": sha256(calibration_path), "temperature": float(calibration["temperature"]),
                             "model_kwargs": kwargs, "runtime_sha256": runtime_hashes,
                             "training_input_fingerprints": config["input_fingerprints"]})
    return root, runs, preprocessing


def parse_reference(name):
    upper = str(name).strip().upper()
    aliases = {"84DS4-AA": 17, "84DS4-AB": 18, "84DS4-AC": 19, "84DS4-AD": 20}
    if upper in aliases:
        return f"XNA{aliases[upper]}", "target"
    match = re.fullmatch(r"(XNA|DNA|PC)0*([1-9][0-9]?)", upper)
    if not match or not 1 <= int(match[2]) <= 20:
        raise ValueError("无法可靠识别参考名称：" + str(name))
    return "XNA" + str(int(match[2])), "target" if match[1] == "XNA" else "control"


def parse_legacy_filename(path):
    stem = Path(path).stem
    patterns = [r"(?:^|_)(?P<ref>XNA0*\d+|DNA0*\d+|PC0*\d+|84Ds4-A[A-D])_(?P<strand>pos|neg)$",
                r"_(?P<strand>pos|neg)_(?P<ref>XNA0*\d+|DNA0*\d+|PC0*\d+|84Ds4-A[A-D])$"]
    for pattern in patterns:
        match = re.search(pattern, stem, flags=re.IGNORECASE)
        if match:
            reference, strand = match.group("ref"), match.group("strand").lower()
            template, role = parse_reference(reference)
            return {"reference": reference, "file_strand": strand, "template": template, "role": role}
    raise ValueError("文件名无法可靠解析：" + str(path))


def metadata_candidate(feature):
    candidates = [feature.with_name(feature.stem + "_meta.tsv"),
                  feature.with_name(feature.name + ".meta.tsv"), feature.with_suffix(".meta.tsv")]
    found = list(dict.fromkeys(str(path.resolve()) for path in candidates if path.is_file()))
    if len(found) > 1:
        raise ValueError("metadata 匹配到多个文件，请用 --data_manifest 明确指定：" + str(feature))
    return found[0] if found else ""


def discover_datasets(args):
    rows = []
    if args.data_manifest:
        manifest = Path(args.data_manifest).expanduser().resolve()
        with manifest.open(encoding="utf-8-sig", newline="") as stream:
            reader = csv.DictReader(stream, delimiter="\t" if manifest.suffix.lower() == ".tsv" else ",")
            for row in reader:
                feature = Path(row["feature_csv"]).expanduser()
                row["feature_csv"] = str((feature if feature.is_absolute() else manifest.parent / feature).resolve())
                meta = row.get("metadata_tsv", "").strip()
                if meta:
                    path = Path(meta).expanduser()
                    row["metadata_tsv"] = str((path if path.is_absolute() else manifest.parent / path).resolve())
                rows.append(row)
    else:
        roots = [("XNA1-12", Path(args.dir_01_12_root) / "ATCGXY_6class"),
                 ("XNA13-16", Path(args.dir_13_16_6)), ("XNA17-20", Path(args.dir_17_20_6))]
        for cohort, root in roots:
            if not root.is_dir():
                raise FileNotFoundError("外部特征目录不存在：" + str(root))
            for feature in sorted(root.glob("*.csv")):
                if any(word in feature.name.lower() for word in ("summary", "prediction", "metrics", "label_count", "confusion")):
                    continue
                parsed = parse_legacy_filename(feature)
                rows.append({**parsed, "feature_csv": str(feature.resolve()), "cohort": cohort,
                             "metadata_tsv": metadata_candidate(feature), "source_dataset": "unknown"})
    if not rows:
        raise ValueError("没有外部测试文件")
    result, paths = [], set()
    for row in rows:
        feature = Path(row["feature_csv"]).resolve()
        if not feature.is_file() or str(feature) in paths:
            raise ValueError("文件缺失或重复列入：" + str(feature))
        paths.add(str(feature))
        number = int(re.search(r"\d+$", str(row["template"]))[0])
        template = "XNA" + str(number)
        if not 1 <= number <= 20 or row["role"] not in ("control", "target"):
            raise ValueError("非法template/role：" + str(row))
        cohort = "XNA1-12" if number <= 12 else "XNA13-16" if number <= 16 else "XNA17-20"
        if row.get("cohort") and row["cohort"] != cohort:
            raise ValueError("cohort与模板编号冲突：" + str(row))
        item = {key: str(row.get(key, "unknown") or "unknown") for key in
                ("reference", "file_strand", "source_dataset", "run_id", "template_group", "pair_id")}
        item.update(feature_csv=str(feature), metadata_tsv=row.get("metadata_tsv", ""),
                    template=template, cohort=cohort, role=row["role"])
        result.append(item)
    expected = {(f"XNA{i}", role) for i in range(1, 21) for role in (["control", "target"] if i <= 16 else ["target"])}
    actual = {(r["template"], r["role"]) for r in result}
    missing = sorted(expected - actual)
    if missing and not args.allow_partial_templates:
        raise ValueError("缺少预定模板/对照：" + str(missing) + "；仅做局部评估请显式 --allow_partial_templates")
    return result, missing
