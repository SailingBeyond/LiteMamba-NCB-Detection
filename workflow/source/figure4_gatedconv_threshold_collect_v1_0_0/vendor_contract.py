#!/usr/bin/env python3
"""Frozen Figure4 source contracts copied from the audited per-6mer collector.

No model is loaded, no inference is performed, and the source result tree is
never written. Three fixed model seeds are kept separate. All rates can later
be derived from integer counts with explicit cohort/role/truth denominators.
"""
from __future__ import annotations

import argparse
from collections import Counter
import csv
from datetime import datetime, timezone
import fcntl
import gzip
import hashlib
import json
import math
import os
from pathlib import Path
import re
import shutil
import sys
import time

VERSION = "1.0.0"
SEEDS = (42, 2024, 2025)
CLASSES = ("A", "T", "C", "G", "X", "Y")
DEFAULT_RESULTS = Path("/path/to/revision-inputs/CPLX-XNA-replenishment/CPLX-XNA-replenishment/fast5_single_guppy/sixclass_overlap_read_split_721/sixclass_auto_v1_0_4_results/figure4_gated_comparison_v1_0_0")
MODEL = "gatedconv_target"
DISK_RESERVE_BYTES = 8 * 1024 ** 3
MAX_JOINT_CELLS_PER_FILE = 250000
UNKNOWN = {"", "unknown", "na", "n/a", "none", "null", "nan", "."}
META_FIELDS = ("sample_index", "sample_id", "source_file", "source_row", "source_sha256",
               "read_id", "read_id_kind", "original_kmer", "motif", "label", "true_class",
               "reference", "template", "cohort", "role", "file_strand", "bam_strand",
               "template_group", "pair_id", "run_id", "source_dataset", "position",
               "reference_identity_basis")
GROUP_SOURCE_FIELDS = ("cohort", "template", "role", "file_strand", "bam_strand", "reference",
                       "source_dataset", "run_id", "template_group", "pair_id", "position",
                       "original_kmer", "motif")
GROUP_OUTPUT_FIELDS = tuple("sequencing_run_id" if name == "run_id" else name for name in GROUP_SOURCE_FIELDS)
COUNT_FIELDS = ("training_run_id", "model", "seed", "dataset_id", "probability_type") + GROUP_OUTPUT_FIELDS + (
    "true_label", "true_class", "predicted_label", "predicted_class", "count")
AVAIL_FIELDS = ("training_run_id", "model", "seed", "dataset_id", "cohort", "template", "role", "file_strand",
                "sample_n", "real_read_rows", "unknown_or_proxy_rows", "original_center_unmasked_rows",
                "original_center_masked_rows", "motif_unique", "original_kmer_unique", "known_position_rows",
                "known_bam_strand_rows", "label_counts_json", "ordered_identity_sha256")
SAMPLE_FIELDS = ("training_run_id", "model", "seed", "dataset_id") + META_FIELDS + ("prediction",)


def require(condition, message):
    if not condition:
        raise ValueError(message)


def now():
    return datetime.now(timezone.utc).isoformat()


def sha256(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(8 * 1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def file_identity(path):
    value = Path(path).stat()
    return {"size_bytes": value.st_size, "mtime_ns": value.st_mtime_ns,
            "device": value.st_dev, "inode": value.st_ino}


def read_json(path):
    return json.loads(Path(path).read_text(encoding="utf-8"))


def write_json(path, value):
    path = Path(path)
    check_disk(path.parent)
    temporary = path.with_name(path.name + ".tmp-" + str(os.getpid()))
    try:
        with temporary.open("w", encoding="utf-8") as stream:
            json.dump(value, stream, ensure_ascii=False, indent=2, allow_nan=False)
            stream.write("\n")
        temporary.replace(path)
    finally:
        temporary.unlink(missing_ok=True)


def write_csv(path, fields, rows):
    path = Path(path)
    check_disk(path.parent)
    temporary = path.with_name(path.name + ".tmp-" + str(os.getpid()))
    try:
        opener = gzip.open if path.name.endswith(".gz") else open
        with opener(temporary, "wt", encoding="utf-8", newline="") as stream:
            writer = csv.DictWriter(stream, fieldnames=fields)
            writer.writeheader()
            for number, row in enumerate(rows):
                if number % 100000 == 0:
                    check_disk(path.parent)
                writer.writerow(row)
        temporary.replace(path)
    finally:
        temporary.unlink(missing_ok=True)


def safe_relative(value, label):
    path = Path(str(value))
    require(str(value) and not path.is_absolute() and ".." not in path.parts, "不安全的" + label)
    return path


def digest_json(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True, ensure_ascii=False,
                                    separators=(",", ":"), allow_nan=False).encode()).hexdigest()


def check_disk(directory):
    available = shutil.disk_usage(directory).free
    require(available >= DISK_RESERVE_BYTES, "输出盘空间低于预留值：可用%.3f GiB，预留%.3f GiB" %
            (available / 2**30, DISK_RESERVE_BYTES / 2**30))


def validate_plan(plan):
    require(plan.get("version") == "gated-comparison-1.0.0" and plan.get("task") == "6-class",
            "请选择当前六分类GatedConv结构比较的evaluation_plan.json")
    require(isinstance(plan.get("plan_id"), str) and re.fullmatch(r"[0-9a-f]{64}", plan["plan_id"]),
            "冻结plan_id无效")
    require(plan.get("seeds") == list(SEEDS) and plan.get("feature_schemes") == ["full"],
            "冻结种子或特征方案与当前Figure4流程不同")
    require(plan.get("inherited_reference_correction", {}).get("kind") == "reference_sequence_correction",
            "缺少已修正XNA13–16参考的继承记录")
    pre = plan.get("preprocessing", {})
    require(pre.get("seq_len") == 6 and pre.get("mask_index") == 3 and pre.get("mask_base") == "N"
            and pre.get("class_names") == list(CLASSES), "六分类/第四位N预处理定义不一致")
    runs, datasets = plan.get("runs", []), plan.get("datasets", [])
    expected_runs = {(model, seed) for model in ("litemamba", MODEL) for seed in SEEDS}
    require(len(runs) == 6 and {(r.get("model"), r.get("seed")) for r in runs} == expected_runs,
            "计划必须包含两个结构、各三个固定种子")
    require(len({r.get("run_id") for r in runs}) == 6, "计划含重复运行ID")
    selected = []
    for run in runs:
        safe_relative(run.get("run_id", ""), "run_id")
        require(run.get("feature_scheme") == "full", "运行不是full特征")
        if run["model"] != MODEL:
            continue
        require(run["run_id"] == f"full/{MODEL}/seed_{run['seed']}", "GatedConv运行ID不一致")
        require(run.get("model_kwargs", {}).get("litemamba_variant") == "legacy_gatedconv"
                and run.get("model_kwargs", {}).get("readout") == "target_attention",
                "运行不是最终GatedConv＋第4位目标读出")
        require(re.fullmatch(r"[0-9a-f]{64}", str(run.get("checkpoint_sha256", ""))) is not None,
                "冻结检查点SHA256无效")
        require(math.isfinite(float(run.get("temperature", 0))) and float(run["temperature"]) > 0,
                "冻结validation温度无效")
        selected.append(run)
    expected_design = {(f"XNA{i}", role, strand) for i in range(1, 21)
                       for role in (("target", "control") if i <= 16 else ("target",))
                       for strand in ("pos", "neg")}
    observed_design = [(d.get("template"), d.get("role"), d.get("file_strand")) for d in datasets]
    require(len(datasets) == 72 and len(set(observed_design)) == 72 and set(observed_design) == expected_design,
            "必须保留完整72个实验文件，不能筛选模板/角色/文件方向子集")
    require(len({d.get("dataset_id") for d in datasets}) == 72, "计划含重复dataset_id")
    for dataset in datasets:
        path = safe_relative(dataset.get("dataset_id", ""), "dataset_id")
        require(len(path.parts) == 1, "dataset_id必须为单个目录名称")
        require(dataset.get("status") == "complete" and type(dataset.get("n_samples")) is int
                and dataset["n_samples"] > 0, "数据文件准备未完成：" + str(path))
        number = int(dataset["template"][3:])
        cohort = "XNA1-12" if number <= 12 else "XNA13-16" if number <= 16 else "XNA17-20"
        require(dataset.get("cohort") == cohort, "cohort与模板分组不一致：" + str(path))
    selected.sort(key=lambda run: SEEDS.index(run["seed"]))
    return selected, sorted(datasets, key=lambda d: d["dataset_id"])


def read_marker(path, plan, run, dataset):
    value = read_json(path)
    expected = {"status": "complete", "plan_id": plan["plan_id"], "run_id": run["run_id"],
                "dataset_id": dataset["dataset_id"], "n_samples": dataset["n_samples"],
                "checkpoint_sha256": run["checkpoint_sha256"], "temperature": run["temperature"]}
    for field, target in expected.items():
        require(value.get(field) == target, f"完成标记不匹配：{path}/{field}")
    require(value.get("prediction_settings") == plan.get("prediction_settings"), "预测设置完成记录不同")
    declared = value.get("outputs", {}).get("predictions.csv.gz", {})
    require(type(declared.get("size_bytes")) is int and declared["size_bytes"] > 0
            and re.fullmatch(r"[0-9a-f]{64}", str(declared.get("sha256", ""))),
            "完成标记缺少预测CSV完整SHA256/长度：" + str(path))
    return value


def validate_available_files(root, plan, runs, datasets):
    missing = []
    expected_ids = {dataset["dataset_id"] for dataset in datasets}
    for run in runs:
        directory = root / "predictions" / run["run_id"]
        marker = directory / "run_complete.json"
        if not marker.is_file():
            missing.append(str(marker))
        else:
            value = read_json(marker)
            require(value.get("status") == "complete" and value.get("plan_id") == plan["plan_id"]
                    and value.get("run_id") == run["run_id"]
                    and set(value.get("datasets", [])) == expected_ids
                    and len(value.get("datasets", [])) == len(expected_ids)
                    and value.get("prediction_settings") == plan.get("prediction_settings"),
                    "运行完成记录不匹配：" + str(marker))
        for dataset in datasets:
            for name in ("predictions.csv.gz", "completion.json"):
                path = directory / dataset["dataset_id"] / name
                if not path.is_file():
                    missing.append(str(path))
    require(not missing, "已有预测尚未齐备；不会启动推理。缺失文件：\n" + "\n".join(missing))


def parse_integer(value, where):
    text = str(value)
    require(re.fullmatch(r"[0-9]+", text) is not None, "非整数标签/索引：" + where)
    return int(text)


def count_rows(path):
    with gzip.open(path, "rt", encoding="utf-8", newline="") as stream:
        reader = csv.DictReader(stream)
        require(tuple(reader.fieldnames or ()) == COUNT_FIELDS, "计数缓存字段与本版不同")
        total = 0
        for row in reader:
            value = parse_integer(row["count"], str(path))
            require(value > 0, "稀疏计数含非正值")
            total += value
        return total


