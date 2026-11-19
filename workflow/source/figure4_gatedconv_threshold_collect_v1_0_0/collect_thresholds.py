#!/usr/bin/env python3
"""Collect saved GatedConv confidence thresholds without training or inference."""
from __future__ import annotations

import argparse
from collections import Counter, defaultdict
from contextlib import contextmanager
from datetime import datetime, timezone
import fcntl
import hashlib
import json
import math
import os
from pathlib import Path
import shutil
import subprocess
import sys
import tarfile
import time

PACKAGE = "figure4_gatedconv_threshold_collect_v1_0_0"
ARCHIVE = "figure4_gatedconv_threshold_statistics_v1_0_0.tar.gz"
DEFAULT_RESULTS = Path("/path/to/revision-inputs/CPLX-XNA-replenishment/CPLX-XNA-replenishment/fast5_single_guppy/sixclass_overlap_read_split_721/sixclass_auto_v1_0_4_results/figure4_gated_comparison_v1_0_0")
DEFAULT_THRESHOLDS = (0., .5, .6, .7, .8, .85, .9, .92, .95, .97, .98, .99, .995, .999, 1.)
BASELINE_KEYS = ("seed", "dataset_id", "cohort", "template", "role", "motif", "true_label", "predicted_label")


def require(condition, message):
    if not condition:
        raise ValueError(message)


def now():
    return datetime.now(timezone.utc).isoformat()


def jsonable(value):
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, dict):
        return {str(key): jsonable(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [jsonable(item) for item in value]
    return value


def digest_file(path):
    value = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(4 * 1024 ** 2), b""):
            value.update(block)
    return value.hexdigest()


def atomic_json(path, value):
    path = Path(path)
    require(not path.is_symlink(), "输出文件不能是符号链接：" + str(path))
    temporary = path.with_name(path.name + ".tmp-" + str(os.getpid()))
    try:
        with temporary.open("w", encoding="utf-8") as stream:
            json.dump(jsonable(value), stream, ensure_ascii=False, indent=2, allow_nan=False)
            stream.write("\n")
        os.replace(temporary, path)
    finally:
        temporary.unlink(missing_ok=True)


def parser():
    result = argparse.ArgumentParser(description=__doc__)
    result.add_argument("--results_dir", type=Path, default=DEFAULT_RESULTS)
    result.add_argument("--output_dir", type=Path)
    result.add_argument("--training_root", type=Path, help="Explicit relocated common training result root, only if paths moved")
    result.add_argument("--thresholds", default=",".join(str(value) for value in DEFAULT_THRESHOLDS))
    result.add_argument("--reference_baseline", type=Path, default=Path(__file__).with_name("reference_baseline.json"))
    result.add_argument("--external_only", action="store_true", help="Explicit exploratory external-only collection; cannot choose a validation threshold")
    result.add_argument("--force_rebuild", action="store_true", help="Rebuild this tool's count caches, leaving source predictions unchanged")
    result.add_argument("--check_only", action="store_true", help="Read-only preflight without creating an output directory")
    result.add_argument("--disk_reserve_gib", type=float, default=2.)
    result.add_argument("--max_records", type=int, default=250000)
    return result


def settings(args):
    import threshold_sources as sources
    source = args.results_dir.expanduser().resolve()
    output_arg = args.output_dir.expanduser() if args.output_dir else source.parent / PACKAGE
    require(not output_arg.is_symlink(), "统计输出根不能是符号链接")
    output = output_arg.resolve()
    require(source.is_dir(), "Figure4 原结果目录不存在：" + str(source))
    require(source != output and source not in output.parents and output not in source.parents,
            "新统计目录必须独立于原 Figure4 结果目录")
    require(output != Path(__file__).resolve().parent, "请勿把统计结果写进收集工具代码目录")
    try:
        thresholds = [float(value.strip()) for value in args.thresholds.split(",")]
    except ValueError as error:
        raise ValueError("thresholds 必须是逗号分隔的数值") from error
    sources.validate_thresholds(thresholds)
    require(math.isfinite(args.disk_reserve_gib) and args.disk_reserve_gib > 0,
            "磁盘预留必须为正有限数")
    require(args.max_records >= 1000, "max_records 必须至少为1000")
    return source, output, thresholds


def baseline_reference(args, description):
    path = args.reference_baseline.expanduser().resolve()
    reference = json.loads(path.read_text(encoding="utf-8"))
    require(reference.get("plan_id") == description["plan"]["plan_id"],
            "当前评估计划与已上传的 Figure4 计数基线不同，未自动替换参考数据")
    plan_path = Path(description["sources"][0]["results_root"]) / "evaluation_plan.json"
    require(reference.get("plan_sha256") == digest_file(plan_path),
            "评估计划内容与已有 Figure4 基线不同")
    require(isinstance(reference.get("rows"), list) and reference["rows"], "参考基线缺少整数计数")
    return reference, {"path": str(path), "sha256": digest_file(path)}


def preflight(args):
    import threshold_sources as sources
    source, output, thresholds = settings(args)
    description = sources.load_sources(source, external_only=args.external_only, training_root=args.training_root)
    reference, reference_guard = baseline_reference(args, description)
    for item in description["sources"]:
        path = Path(item["path"]).resolve()
        require(output != path and output not in path.parents, "原预测文件不能位于统计输出内")
    for run in description["runs"]:
        source_file = next(item for item in description["sources"] if item["seed"] == run["seed"])
        run_directory = Path(next(record["path"] for record in source_file["guard_records"]
                                  if Path(record["path"]).name == "resolved_config.json")).parent.resolve()
        require(output != run_directory and run_directory not in output.parents and output not in run_directory.parents,
                "统计目录必须与原模型运行目录独立")
    return source, output, thresholds, description, reference, reference_guard


def lower_priority():
    status = {}
    try:
        current = os.getpriority(os.PRIO_PROCESS, 0)
        if current < 10:
            os.nice(10 - current)
        status["nice"] = os.getpriority(os.PRIO_PROCESS, 0)
    except (OSError, AttributeError) as error:
        status["nice_note"] = str(error)
    ionice = shutil.which("ionice")
    if ionice:
        try:
            value = subprocess.run([ionice, "-c", "3", "-p", str(os.getpid())],
                                   stdin=subprocess.DEVNULL, stdout=subprocess.PIPE,
                                   stderr=subprocess.PIPE, text=True, timeout=10)
            status["ionice_idle_requested"] = value.returncode == 0
            if value.returncode:
                status["ionice_note"] = value.stderr.strip()
        except (OSError, subprocess.SubprocessError) as error:
            status["ionice_idle_requested"] = False
            status["ionice_note"] = type(error).__name__ + ": " + str(error)
    else:
        status["ionice_note"] = "ionice unavailable; CPU niceness still attempted"
    print("[PRIORITY] " + json.dumps(status, ensure_ascii=False), flush=True)
    return status


@contextmanager
def workflow_lock(output):
    lock_path = output / ".workflow.lock"
    require(not lock_path.is_symlink(), "输出锁不能是符号链接")
    with lock_path.open("a") as handle:
        try:
            fcntl.flock(handle, fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError as error:
            raise RuntimeError("同一阈值收集目录已有任务运行，请查看原日志") from error
        try:
            yield
        finally:
            fcntl.flock(handle, fcntl.LOCK_UN)


def check_disk(output, reserve_gib):
    require(shutil.disk_usage(output).free >= reserve_gib * 1024 ** 3,
            "统计输出盘可用空间低于预留 %.2f GiB" % reserve_gib)


def verify_identity_and_baseline(records, audits, reference, external_only):
    identity_groups = defaultdict(list)
    for item in audits:
        identity_groups[(item["scope"], item["dataset_id"])].append(item)
    for key, values in identity_groups.items():
        require({item["seed"] for item in values} == {42, 2024, 2025} and len(values) == 3,
                "同一数据文件缺少三个固定 seed：" + str(key))
        require(len({(item["n_samples"], item["ordered_identity_sha256"]) for item in values}) == 1,
                "不同 seed 使用的数据身份或窗口顺序不一致：" + str(key))
    expected_groups = 72 if external_only else 73
    require(len(identity_groups) == expected_groups, "未保留完整外部数据及所需验证集")
    expected_hashes = {(item["seed"], item["dataset_id"]): item["prediction_sha256"]
                       for item in reference.get("source_predictions", [])}
    actual_hashes = {(item["seed"], item["dataset_id"]): item["source_sha256"]
                     for item in audits if item["scope"] == "external"}
    require(len(expected_hashes) == 216 and actual_hashes == expected_hashes,
            "全量外部预测 SHA256 与原上传结果不一致；未用新的概率悄悄替换旧结果")
    actual, expected = Counter(), Counter()
    for row in records:
        if row["scope"] == "external" and row["probability_type"] == "raw":
            actual[tuple(row[field] for field in BASELINE_KEYS)] += row["count"]
    for row in reference["rows"]:
        expected[tuple(row[field] for field in BASELINE_KEYS)] += row["count"]
    require(actual == expected, "阈值0的全量 argmax 计数与原上传 Figure4 基线不一致；未继续生成结论")
    return {"status": "passed", "cross_seed_identity_groups": len(identity_groups),
            "original_prediction_hashes_verified": len(expected_hashes),
            "baseline_cells": len(expected), "baseline_prediction_rows": sum(expected.values())}


def safe_directory(output, relative):
    current = output
    for part in Path(relative).parts:
        current = current / part
        require(not current.is_symlink(), "统计目录含符号链接：" + str(current))
        current.mkdir(exist_ok=True)
    return current


def make_archive(output, stats, payload):
    archive = output / ARCHIVE
    require(not archive.is_symlink(), "结果包目标不能是符号链接")
    temporary = archive.with_name(archive.name + ".tmp-" + str(os.getpid()))
    total = 0
    try:
        with tarfile.open(temporary, "w:gz", compresslevel=6) as packed:
            for name in sorted(payload):
                relative = Path(name)
                require(not relative.is_absolute() and ".." not in relative.parts, "非法统计产物路径")
                path = stats / relative
                require(path.is_file() and not path.is_symlink() and stats.resolve() in path.resolve().parents,
                        "统计产物不是独立目录中的普通文件：" + str(path))
                require(path.stat().st_size <= 32 * 1024 ** 2, "单个统计文件超过轻量包大小保护")
                total += path.stat().st_size
                require(total <= 128 * 1024 ** 2, "统计文件合计超过轻量包保护上限")
                packed.add(path, arcname=str(relative), recursive=False)
        with tarfile.open(temporary, "r:gz") as packed:
            require(len(packed.getmembers()) == len(payload), "统计包文件清单异常")
            for member in packed.getmembers():
                expected = digest_file(stats / member.name)
                stream = packed.extractfile(member)
                value = hashlib.sha256()
                for block in iter(lambda: stream.read(1024 ** 2), b""):
                    value.update(block)
                require(value.hexdigest() == expected, "统计包内容 SHA256 核对失败")
        os.replace(temporary, archive)
    finally:
        temporary.unlink(missing_ok=True)
    return {"path": str(archive), "size_bytes": archive.stat().st_size, "sha256": digest_file(archive)}


def main(argv=None):
    import threshold_sources as sources
    import threshold_metrics as metrics
    args = parser().parse_args(argv)
    priority = None if args.check_only else lower_priority()
    source, output, thresholds, description, reference, reference_guard = preflight(args)
    scope_counts = Counter(item["scope"] for item in description["sources"])
    print("[PREFLIGHT PASS] external=%d validation=%d; thresholds=%d" %
          (scope_counts["external"], scope_counts["validation"], len(thresholds)), flush=True)
    print("[OUTPUT DIRECTORY] " + str(output), flush=True)
    if args.check_only:
        print("[CHECK ONLY] 仅核对已有输入；未创建统计输出，也未启动训练或推理", flush=True)
        return {"status": "preflight_passed", "source_counts": dict(scope_counts)}
    output.mkdir(parents=True, exist_ok=True)
    with workflow_lock(output):
        cache = safe_directory(output, "count_cache")
        stats = safe_directory(output, "statistics")
        check_disk(output, args.disk_reserve_gib)
        code_files = ("collect_thresholds.py", "threshold_sources.py", "vendor_contract.py", "threshold_metrics.py")
        binding = {"version": "1.0.0", "results_dir": str(source), "plan_id": description["plan"]["plan_id"],
                   "plan_sha256": reference["plan_sha256"], "thresholds": thresholds,
                   "external_only": args.external_only, "training_root": description.get("training_root"),
                   "reference_sha256": reference_guard["sha256"],
                   "code_sha256": {name: digest_file(Path(__file__).with_name(name)) for name in code_files}}
        binding_path = output / "collection_binding.json"
        if binding_path.exists():
            require(json.loads(binding_path.read_text(encoding="utf-8")) == binding,
                    "该统计目录已绑定其他输入/阈值/代码，请指定新的 output_dir")
        else:
            atomic_json(binding_path, binding)
        state_path = output / "workflow_status.json"
        state = {"status": "running", "started_at": now(), "completed_sources": 0,
                 "total_sources": len(description["sources"]), "output_dir": str(output)}
        atomic_json(state_path, state)
        started = time.monotonic()
        try:
            records, audits = [], []
            for index, item in enumerate(description["sources"], 1):
                check_disk(output, args.disk_reserve_gib)
                cells, audit = sources.collect_source(item, thresholds, cache, force=args.force_rebuild)
                records.extend(cells)
                require(len(records) <= args.max_records, "聚合计数超过内存保护上限；已完成文件缓存保留，未截断数据")
                audits.append(audit)
                state.update(completed_sources=index, last_source={"scope": item["scope"], "seed": item["seed"], "dataset_id": item["dataset_id"]})
                atomic_json(state_path, state)
                print("[PROGRESS] %d/%d sources; compact_cells=%d" % (index, len(description["sources"]), len(records)), flush=True)
            source_qa = verify_identity_and_baseline(records, audits, reference, args.external_only)
            state.update(status="summarizing")
            atomic_json(state_path, state)
            products = metrics.summarize(records, thresholds, stats)
            require(products["qa"].get("status") == "passed", "阈值计数/指标核查未通过")
            for item in description["sources"]:
                sources.check_guards(item)
            for audit in audits:
                require(sources.file_identity(Path(audit["path"])) == audit["source_stat"],
                        "此前已统计的预测文件在打包前发生变化：" + audit["path"])
            require(all(digest_file(Path(__file__).with_name(name)) == expected
                        for name, expected in binding["code_sha256"].items()),
                    "统计代码在运行期间发生变化，未生成完成结果包")
            require(digest_file(reference_guard["path"]) == reference_guard["sha256"],
                    "参考基线在运行期间发生变化")
            definitions = {"model": "gatedconv_target", "seeds": [42, 2024, 2025],
                "class_names": list(sources.CLASSES), "primary_probability_type": "calibrated",
                "raw_probability_purpose": "Predeclared sensitivity analysis; not selected by external performance.",
                "decision_rule": "Keep original six-class argmax only when its maximum probability >= threshold; otherwise no_call=6.",
                "denominator_rule": "Keep every original window in FPR, identity-recall and coverage denominators. Rejection is not a correct canonical call.",
                "analysis_status": "Post-hoc threshold sensitivity analysis of previously examined external predictions. No automatic final threshold and no external calibration/model fitting.",
                "thresholds": thresholds, "source_qa": source_qa, "metric_qa": products["qa"],
                "validation_aggregation": "All validation rows included; only pooled truth/prediction confidence bins retained. External data retain each masked 6mer.",
                "uncertainty_scope": "Training-seed variation only; no read-independent or biological replicate confidence intervals inferred.",
                "reference_baseline": reference_guard}
            atomic_json(stats / "collection_manifest.json", dict(binding, **definitions, completed_at=now(),
                        source_file_count=len(audits), source_prediction_rows=sum(x["n_samples"] for x in audits),
                        source_files=audits, scheduling_priority=priority,
                        elapsed_seconds=round(time.monotonic() - started, 3)))
            atomic_json(stats / "source_provenance.json", description["provenance"])
            atomic_json(stats / "evaluation_plan.json", description["plan"])
            payload = list(products["outputs"]) + ["collection_manifest.json", "source_provenance.json", "evaluation_plan.json"]
            file_manifest = {name: {"size_bytes": (stats / name).stat().st_size,
                                   "sha256": digest_file(stats / name)} for name in payload}
            atomic_json(stats / "RESULT_PACKAGE_MANIFEST.json", {"files": file_manifest,
                         "scope": "Small count tables and JSON only. No per-window predictions, feature arrays, checkpoints or read identifiers are packed."})
            payload.append("RESULT_PACKAGE_MANIFEST.json")
            package = make_archive(output, stats, payload)
            state.update(status="complete", completed_at=now(), package=package,
                         elapsed_seconds=round(time.monotonic() - started, 3))
            atomic_json(state_path, state)
            print("[RESULT PACKAGE] " + package["path"], flush=True)
            print("[FULL PASS] 全部输入、阈值计数、原始分母、跨seed身份及轻量包核查完成", flush=True)
            return state
        except BaseException as error:
            state.update(status="failed", failed_at=now(), error=type(error).__name__ + ": " + str(error))
            atomic_json(state_path, state)
            raise


if __name__ == "__main__":
    try:
        main()
    except (OSError, ValueError, RuntimeError, KeyError, TimeoutError, subprocess.SubprocessError) as error:
        print("[ERROR] " + type(error).__name__ + ": " + str(error), file=sys.stderr, flush=True)
        sys.exit(1)
