#!/usr/bin/env python3
"""Read frozen six-class predictions into compact confidence-bin counts.

Standard-library only. No checkpoint, feature array, training or GPU code is
loaded. Source result files are never written. All 216 external files remain
represented; validation is deliberately aggregated across motifs to bound RAM.
"""
from __future__ import annotations

from bisect import bisect_right
from collections import Counter
import csv
import gzip
import hashlib
import json
import math
import os
from pathlib import Path
import re
import time

import vendor_contract as contract

VERSION = "1.0.0"
CLASSES = contract.CLASSES
SEEDS = contract.SEEDS
DEFAULT_RESULTS = contract.DEFAULT_RESULTS
BIN_FIELDS = ("scope", "seed", "dataset_id", "cohort", "template", "role", "motif",
              "true_label", "predicted_label", "probability_type", "max_passed_index", "count")
MAX_CELLS_PER_SOURCE = 200000
VAL_IDENTITY_FIELDS = ("sample_index", "sample_id", "read_id", "chrom", "pair_id", "template_group",
                       "bam_strand", "motif", "original_kmer", "label", "class_name", "run_id",
                       "source_dataset", "source_file", "source_row", "read_id_kind", "read_identity_basis",
                       "original_read_id", "reference_identity_known", "positive_source_sha256",
                       "positive_data_record", "merge_input_role")


require = contract.require
sha256 = contract.sha256
read_json = contract.read_json
file_identity = contract.file_identity
now = contract.now


def _checked_json(path, expected=None):
    path = Path(path)
    require(path.is_file(), "缺少已有结果文件；不会启动推理：" + str(path))
    before = file_identity(path)
    digest = sha256(path)
    if expected is not None:
        require(digest == expected, "冻结元数据 SHA256 不匹配：" + str(path))
    value = read_json(path)
    require(file_identity(path) == before, "元数据读取期间变化：" + str(path))
    return value, {"path": str(path), "sha256": digest, "stat": before}


def _mapped_path(value, original_root, training_root):
    path = Path(value).expanduser()
    require(path.is_absolute(), "冻结训练路径不是绝对路径：" + str(path))
    if training_root is not None:
        try:
            relative = path.relative_to(original_root)
        except ValueError as error:
            raise ValueError("训练路径不在明确声明的原训练结果根下，不能自动猜测迁移：" + str(path)) from error
        path = Path(training_root) / relative
    return path.resolve()


def _class_names(value):
    if isinstance(value, dict):
        return [value.get(str(i), value.get(i)) for i in range(6)]
    return value


def _run_metadata(run, original_root, training_root):
    run_dir = _mapped_path(run["run_dir"], original_root, training_root)
    cfg_path = _mapped_path(run["resolved_config"], original_root, training_root)
    calibration_path = _mapped_path(run["calibration"], original_root, training_root)
    require(cfg_path == run_dir / "resolved_config.json" and calibration_path == run_dir / "calibration.json",
            "冻结配置/校准文件不属于指定训练运行")
    cfg, cfg_guard = _checked_json(cfg_path, run["resolved_config_sha256"])
    cal, cal_guard = _checked_json(calibration_path, run["calibration_sha256"])
    require(cfg.get("seed") == run["seed"] and cfg.get("feature_scheme") == "full"
            and cfg.get("model_type") == "litemamba"
            and cfg.get("litemamba_variant") == "legacy_gatedconv"
            and cfg.get("readout") == "target_attention"
            and _class_names(cfg.get("class_names")) == list(CLASSES),
            "训练配置不是最终的六分类 GatedConv/target_attention")
    require(cal.get("fitted_on") == "val" and cal.get("temperature") == run["temperature"]
            and cal.get("checkpoint_sha256") == run["checkpoint_sha256"],
            "温度不是对应检查点的已有 validation 校准")
    expected_result_hash = run.get("frozen_metadata_sha256", {}).get("results.json")
    result, result_guard = _checked_json(run_dir / "results.json", expected_result_hash)
    require(result.get("seed") == run["seed"] and result.get("feature_scheme") == "full"
            and result.get("litemamba_variant") == "legacy_gatedconv"
            and result.get("readout") == "target_attention"
            and _class_names(result.get("class_names")) == list(CLASSES), "训练结果身份不同")
    return run_dir, cfg, cal, result, [cfg_guard, cal_guard, result_guard]


def _prediction_header(path, scope):
    require(Path(path).is_file(), "缺少已有预测；不会重训或推理：" + str(path))
    with gzip.open(path, "rt", encoding="utf-8-sig", newline="") as stream:
        reader = csv.reader(stream)
        fields = next(reader, [])
    identity = contract.META_FIELDS if scope == "external" else VAL_IDENTITY_FIELDS
    needed = set(identity) | {"prediction", "confidence", "calibrated_confidence", "seed", "feature_scheme"}
    needed |= {"prob_" + str(i) for i in range(6)} | {"calibrated_prob_" + str(i) for i in range(6)}
    needed |= {"model"} if scope == "external" else {"model_type", "split", "predicted_class_name"}
    require(len(fields) == len(set(fields)), "预测CSV列名重复：" + str(path))
    require(needed.issubset(fields), "预测CSV缺少已有概率或身份字段：%s / %s" % (path, ",".join(sorted(needed - set(fields)))))
    return fields


def load_sources(results_dir, external_only=False, training_root=None):
    """Precheck the whole frozen input, returning only small source descriptors.

    training_root, when explicit, remaps the common result root immediately
    above evaluation_plan['training_results']; no filesystem model discovery is
    performed. Probability/metadata files remain the sole runtime input.
    """
    root = Path(results_dir).expanduser().resolve()
    require(root.is_dir(), "外部评估结果目录不存在：" + str(root))
    plan, plan_guard = _checked_json(root / "evaluation_plan.json")
    runs, datasets = contract.validate_plan(plan)
    contract.validate_available_files(root, plan, runs, datasets)
    original_root = Path(plan["training_results"]).parent
    require(original_root.is_absolute(), "训练结果根必须为冻结绝对路径")
    if training_root is not None:
        training_root = Path(training_root).expanduser().resolve()
        require(training_root.is_dir(), "显式训练结果根不存在：" + str(training_root))
    sources, provenance = [], [plan_guard]
    for run in runs:
        run_dir, cfg, cal, result, run_guards = _run_metadata(run, original_root, training_root)
        provenance.extend(run_guards)
        parent = root / "predictions" / run["run_id"]
        run_marker, run_marker_guard = _checked_json(parent / "run_complete.json")
        provenance.append(run_marker_guard)
        for dataset in datasets:
            folder = parent / dataset["dataset_id"]
            marker_path = folder / "completion.json"
            _marker_value, marker_guard = _checked_json(marker_path)
            marker = contract.read_marker(marker_path, plan, run, dataset)
            require(file_identity(marker_path) == marker_guard["stat"] and sha256(marker_path) == marker_guard["sha256"],
                    "完成标记校验期间变化：" + str(marker_path))
            prediction = folder / "predictions.csv.gz"
            _prediction_header(prediction, "external")
            declared = marker["outputs"]["predictions.csv.gz"]
            require(prediction.stat().st_size == declared["size_bytes"], "预测CSV长度与完成记录不同：" + str(prediction))
            sources.append({"scope": "external", "seed": run["seed"], "run_id": run["run_id"],
                "dataset_id": dataset["dataset_id"], "cohort": dataset["cohort"], "template": dataset["template"],
                "role": dataset["role"], "file_strand": dataset["file_strand"], "path": str(prediction),
                "n_samples": dataset["n_samples"], "declared_sha256": declared["sha256"],
                "temperature": run["temperature"], "checkpoint_sha256": run["checkpoint_sha256"],
                "plan_id": plan["plan_id"], "plan_sha256": plan_guard["sha256"], "results_root": str(root),
                "protected_roots": [str(root), str(run_dir)],
                "label_counts": dataset.get("label_counts", {}),
                "guard_records": [plan_guard, run_marker_guard, marker_guard, *run_guards]})
        if external_only:
            continue
        exported = result.get("exports_by_split", {}).get("val")
        require(isinstance(exported, dict), "缺少已有验证集预测导出记录：" + str(run_dir / "results.json"))
        require(exported.get("split") == "val" and type(exported.get("n_samples")) is int
                and exported["n_samples"] > 0
                and exported.get("checkpoint_sha256") == run["checkpoint_sha256"]
                and exported.get("temperature") == run["temperature"], "验证集导出与冻结检查点/温度不一致")
        prediction = _mapped_path(exported["predictions"], original_root, training_root)
        require(prediction == run_dir / "val" / "predictions.csv.gz", "验证预测不在该运行的 val 目录")
        complete, complete_guard = _checked_json(run_dir / "val" / "export_complete.json")
        for field in ("split", "n_samples", "checkpoint_sha256", "temperature", "dataset_identity", "predictions"):
            require(complete.get(field) == exported.get(field), "验证集完成记录与结果快照不同：" + field)
        expected_identity = exported.get("dataset_identity", {})
        pre = expected_identity.get("preprocessing", {})
        require(pre.get("seq_len") == 6 and pre.get("mask_index") == 3 and pre.get("mask_base") == "N"
                and pre.get("class_names") == list(CLASSES), "验证集预处理不是第四位N六分类")
        require(expected_identity.get("input_fingerprints", {}) == {
            key: {k: cfg["input_fingerprints"]["val"][key][k] for k in ("sha256", "size")}
            for key in ("feature", "metadata")}, "验证集导出与原输入内容指纹不一致")
        require(cal.get("n_samples") == exported["n_samples"], "校准与验证样本量不同")
        _prediction_header(prediction, "validation")
        provenance.append(complete_guard)
        raw_metrics = exported.get("metrics", {}).get("raw", {})
        sources.append({"scope": "validation", "seed": run["seed"], "run_id": run["run_id"],
            "dataset_id": "validation", "cohort": "VAL", "template": "validation", "role": "validation",
            "path": str(prediction), "n_samples": exported["n_samples"], "declared_sha256": None,
            "temperature": run["temperature"], "checkpoint_sha256": run["checkpoint_sha256"],
            "plan_id": plan["plan_id"], "plan_sha256": plan_guard["sha256"], "results_root": str(root),
            "protected_roots": [str(root), str(run_dir)],
            "expected_confusion_matrix": raw_metrics.get("confusion_matrix"),
            "guard_records": [plan_guard, *run_guards, complete_guard],
            "validation_aggregation": "All real validation rows verified; motif/role collapsed only in bin counts for bounded memory."})
    return {"plan": plan, "runs": runs, "datasets": datasets, "sources": sources,
            "provenance": list({record["path"]: record for record in provenance}.values()),
            "external_only": bool(external_only), "training_root": str(training_root) if training_root else None}


def check_guards(source):
    for guard in source.get("guard_records", []):
        path = Path(guard["path"])
        require(file_identity(path) == guard["stat"] and sha256(path) == guard["sha256"],
                "冻结来源元数据在预检后发生变化：" + str(path))


def _atomic_json(path, value, compressed=False):
    path = Path(path)
    temporary = path.with_name(path.name + ".tmp-" + str(os.getpid()))
    try:
        opener = gzip.open if compressed else open
        with opener(temporary, "wt", encoding="utf-8") as stream:
            json.dump(value, stream, ensure_ascii=False, separators=(",", ":"), allow_nan=False)
            stream.write("\n")
        os.replace(temporary, path)
    finally:
        temporary.unlink(missing_ok=True)


def validate_thresholds(thresholds):
    require(isinstance(thresholds, (list, tuple)) and 1 <= len(thresholds) <= 101,
            "阈值网格必须包含1到101个数值")
    require(all(isinstance(x, (int, float)) and not isinstance(x, bool) and math.isfinite(x) and 0 <= x <= 1
                for x in thresholds), "阈值必须为0到1之间有限数")
    require(thresholds[0] == 0 and list(thresholds) == sorted(set(thresholds)),
            "阈值必须严格递增并从0开始，以保留原argmax基线")


def _probabilities(row, prefix, pred, confidence_field):
    values = [float(row[prefix + str(i)]) for i in range(6)]
    require(all(math.isfinite(x) and 0 <= x <= 1 for x in values)
            and math.isclose(sum(values), 1., abs_tol=2e-5), "六类概率不是有限归一化概率：" + prefix)
    require(max(range(6), key=values.__getitem__) == pred, "六类概率argmax与prediction不同：" + prefix)
    confidence = float(row[confidence_field])
    require(math.isfinite(confidence) and math.isclose(confidence, max(values), abs_tol=2e-6),
            "confidence与六类最大概率不一致：" + confidence_field)
    # The source confidence column is checked for consistency, but the gate is
    # defined on the actual six-class maximum, including exact >= boundaries.
    return max(values)


def _motif(row):
    original, motif = row["original_kmer"], row["motif"]
    require(len(original) == 6 and all(base in "ACGTN" or (i == 3 and base in "XY")
                for i, base in enumerate(original)), "original_kmer不是合法6mer")
    require(motif == original[:3] + "N" + original[4:] and len(motif) == 6,
            "motif不是第4位N掩码6mer")
    return motif


def collect_source(source, thresholds, cache_dir, force=False):
    """Stream one saved CSV, returning compact records and a source audit.

    One confidence bin per probability family is recorded for each window;
    there is no per-window threshold loop and no removal of rejected windows.
    """
    validate_thresholds(thresholds)
    path = Path(source["path"])
    cache = Path(cache_dir).expanduser()
    require(not cache.is_symlink() and cache.resolve() == cache.absolute(), "缓存输出不能经过符号链接")
    root = Path(source["results_root"]).resolve()
    protected_roots = source.get("protected_roots", [str(root), str(path.parent)])
    for protected_value in protected_roots:
        protected = Path(protected_value).resolve()
        require(cache.resolve() != protected and protected not in cache.resolve().parents,
                "阈值缓存必须在原结果/训练运行目录以外")
    require(path.resolve() != cache.resolve() and cache.resolve() not in path.resolve().parents,
            "预测来源不能属于阈值输出缓存")
    cache.mkdir(parents=True, exist_ok=True)
    check_guards(source)
    before = file_identity(path)
    print("[VERIFY CSV] %s seed=%s %s" % (source["scope"], source["seed"], source["dataset_id"]), flush=True)
    digest = sha256(path)
    require(file_identity(path) == before, "预测文件校验期间变化：" + str(path))
    if source.get("declared_sha256") is not None:
        require(digest == source["declared_sha256"], "预测CSV SHA256与原完成记录不同：" + str(path))
    binding = {"collector_version": VERSION, "source": source, "source_sha256": digest,
               "source_stat": before, "thresholds": list(thresholds),
               "collector_sha256": sha256(Path(__file__)),
               "contract_sha256": sha256(Path(contract.__file__))}
    stem = "%s_seed%s_%s" % (source["scope"], source["seed"], source["dataset_id"])
    require(re.fullmatch(r"[A-Za-z0-9_.-]+", stem), "缓存身份不是安全文件名")
    cache_path = cache / (stem + ".json.gz")
    if not force and cache_path.is_file():
        with gzip.open(cache_path, "rt", encoding="utf-8") as stream:
            previous = json.load(stream)
        if previous.get("binding") == binding:
            records, audit = previous["records"], previous["audit"]
            require(contract.digest_json(records) == previous.get("records_sha256"), "阈值计数缓存内容被修改")
            require(contract.digest_json(audit) == previous.get("audit_sha256"), "阈值来源审计缓存内容被修改")
            _validate_records(records, thresholds, source["n_samples"])
            print("[REUSE COUNTS] " + stem, flush=True)
            return records, dict(audit, reused=True)
    _prediction_header(path, source["scope"])
    identity_fields = contract.META_FIELDS if source["scope"] == "external" else VAL_IDENTITY_FIELDS
    counts, label_counts = Counter(), Counter()
    ordered = hashlib.sha256()
    confusion = [[0] * 6 for _ in range(6)]
    n, real_reads, known_positions = 0, 0, 0
    started = last_progress = time.monotonic()
    with gzip.open(path, "rt", encoding="utf-8-sig", newline="") as stream:
        reader = csv.DictReader(stream)
        for row in reader:
            require(None not in row and all(value is not None for value in row.values()), "预测CSV行列数量不一致")
            require(contract.parse_integer(row["sample_index"], "sample_index") == n, "sample_index不连续")
            truth, pred = contract.parse_integer(row["label"], "label"), contract.parse_integer(row["prediction"], "prediction")
            require(0 <= truth < 6 and 0 <= pred < 6, "标签不在六分类范围内")
            require(contract.parse_integer(row["seed"], "seed") == source["seed"] and row["feature_scheme"] == "full",
                    "预测seed或特征身份不同")
            motif = _motif(row)
            original = row["original_kmer"]
            require(truth >= 4 or original[3] not in "ATCG" or original[3] == CLASSES[truth],
                    "original_kmer第4位天然碱基与真值冲突")
            if source["scope"] == "external":
                require(row["model"] == contract.MODEL and row["true_class"] == CLASSES[truth], "外部预测模型或类别名不同")
                for field in ("cohort", "template", "role", "file_strand"):
                    require(row[field] == source[field], "外部逐窗声明与计划不同：" + field)
                require(source["role"] != "control" or truth < 4, "天然对照中出现X/Y真值")
                role = source["role"]
            else:
                require(row["split"] == "val" and row["model_type"] == "litemamba"
                        and row["class_name"] == CLASSES[truth]
                        and row["predicted_class_name"] == CLASSES[pred], "验证CSV身份/类别字段不同")
                role, motif = "validation", "ALL"
            for probability_type, prefix, confidence_field in (("raw", "prob_", "confidence"),
                                                               ("calibrated", "calibrated_prob_", "calibrated_confidence")):
                confidence = _probabilities(row, prefix, pred, confidence_field)
                passed = bisect_right(thresholds, confidence) - 1
                require(passed >= 0, "概率没有通过argmax基线阈值")
                key = (source["scope"], source["seed"], source["dataset_id"], source["cohort"], source["template"],
                       role, motif, truth, pred, probability_type, passed)
                require(key in counts or len(counts) < MAX_CELLS_PER_SOURCE,
                        "单文件阈值计数超过内存保护上限，未静默删除或合并外部背景")
                counts[key] += 1
            label_counts[truth] += 1
            confusion[truth][pred] += 1
            real_reads += row.get("read_id_kind") == "sequencing_read_id" and row["read_id"].strip().lower() not in contract.UNKNOWN
            known_positions += row.get("position", "unknown").strip().lower() not in contract.UNKNOWN
            ordered.update(json.dumps([row[field] for field in identity_fields], ensure_ascii=False, separators=(",", ":")).encode())
            ordered.update(b"\n")
            n += 1
            if n % 50000 == 0 and time.monotonic() - last_progress >= 20:
                print("[SCAN] %s seed=%s %s %s/%s rows" % (source["scope"], source["seed"], source["dataset_id"], n, source["n_samples"]), flush=True)
                last_progress = time.monotonic()
    require(n == source["n_samples"], "预测行数与冻结分母不同：" + str(path))
    if source.get("label_counts"):
        require({str(key): value for key, value in label_counts.items()} == source["label_counts"], "外部标签数量与冻结计划不同")
    if source.get("expected_confusion_matrix") is not None:
        require(confusion == source["expected_confusion_matrix"], "验证argmax混淆矩阵与原导出指标不同")
    require(file_identity(path) == before, "预测CSV在统计期间发生变化：" + str(path))
    check_guards(source)
    records = [dict(zip(BIN_FIELDS, (*key, count))) for key, count in sorted(counts.items())]
    _validate_records(records, thresholds, n)
    audit = {"scope": source["scope"], "seed": source["seed"], "run_id": source["run_id"],
             "dataset_id": source["dataset_id"], "path": str(path), "source_sha256": digest,
             "source_stat": before, "n_samples": n, "real_read_rows": int(real_reads),
             "unknown_or_proxy_rows": n - int(real_reads), "known_position_rows": int(known_positions),
             "ordered_identity_sha256": ordered.hexdigest(), "label_counts": dict(label_counts),
             "argmax_confusion_matrix": confusion, "bin_cells": len(records),
             "elapsed_seconds": round(time.monotonic() - started, 3), "completed_at": now(),
             "prior_prediction_hash_available": source.get("declared_sha256") is not None,
             "validation_aggregation": source.get("validation_aggregation"), "reused": False}
    _atomic_json(cache_path, {"binding": binding, "records": records, "audit": audit,
                              "records_sha256": contract.digest_json(records),
                              "audit_sha256": contract.digest_json(audit)}, compressed=True)
    print("[SOURCE COMPLETE] " + stem + " rows=" + str(n), flush=True)
    return records, audit


def _validate_records(records, thresholds, n_samples):
    totals = Counter()
    for row in records:
        require(tuple(row) == BIN_FIELDS and type(row["count"]) is int and row["count"] > 0,
                "计数缓存字段/数量异常")
        require(row["probability_type"] in ("raw", "calibrated")
                and type(row["max_passed_index"]) is int
                and 0 <= row["max_passed_index"] < len(thresholds), "计数缓存阈值索引异常")
        totals[row["probability_type"]] += row["count"]
    require(totals == {"raw": n_samples, "calibrated": n_samples}, "两种概率的阈值计数分母不守恒")
