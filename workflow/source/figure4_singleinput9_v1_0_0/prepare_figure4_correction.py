#!/usr/bin/env python3
"""以新参考的 XNA13–16 替换原外部评估，复用通过完整性校验的其余预测。"""
from __future__ import annotations

import argparse
from contextlib import contextmanager
import copy
import csv
import fcntl
import json
import os
from pathlib import Path
import shutil
import sys
import time

from figure4_common import (VERSION, atomic_json, ensure_disk, fingerprint,
    identity_hash, metadata_candidate, parse_legacy_filename, read_json, sha256,
    validate_completion, verify_run)
from prepare_figure4 import independence_audit, prepare_cache

DEFAULT_CORRECTED = "/path/to/staged-storage2/analysis/paper/xna_pc_13_16_6_X_to_T"
SETTING_KEYS = {"batch_size", "workers", "cpu_threads", "embedding_dtype",
                "no_embeddings", "no_attention", "amp_policy"}
DECLARATION_KEYS = ("feature_csv", "metadata_tsv", "reference", "file_strand",
    "source_dataset", "run_id", "template_group", "pair_id", "template", "cohort", "role")


def _validate_old_plan(plan):
    if plan.get("task") != "6-class" or not plan.get("plan_id"):
        raise ValueError("旧目录不是有效的六分类 Figure 4 冻结评估")
    if not plan.get("runs") or not plan.get("datasets"):
        raise ValueError("旧评估计划没有模型或外部文件")
    for field, rows, key in [("runs", plan["runs"], "run_id"),
                             ("datasets", plan["datasets"], "dataset_id")]:
        values = [row[key] for row in rows]
        if len(set(values)) != len(values):
            raise ValueError("旧计划身份重复：" + field)
    for run in plan["runs"]:
        parts = Path(run["run_id"]).parts
        if len(parts) != 3 or any(p in (".", "..") for p in parts) or Path(run["run_id"]).is_absolute():
            raise ValueError("旧计划 run_id 路径非法")
    for dataset in plan["datasets"]:
        name = dataset["dataset_id"]
        if Path(name).name != name or name in (".", ".."):
            raise ValueError("旧计划 dataset_id 路径非法")
    # 不能只相信字符串plan_id：kwargs/掩码位置即使改动后权重形状仍可能兼容。
    bookkeeping = {"plan_id", "total_cache_bytes", "max_file_cache_bytes",
                   "shared_cache_budget_bytes", "cache_budget_scope", "created_at"}
    scientific = {key: value for key, value in plan.items() if key not in bookkeeping}
    scientific["datasets"] = [_scientific_dataset(d) for d in plan["datasets"]]
    if identity_hash(scientific) != plan["plan_id"]:
        raise ValueError("旧冻结计划内容与plan_id的科学配置SHA不一致，不能复用或猜测修复")


def inherit_prediction_settings(previous_output, plan=None):
    """不导入/初始化 CUDA；继承所有旧模型/种子一致的实际导出设置。"""
    root = Path(previous_output).expanduser().resolve()
    plan = read_json(root / "evaluation_plan.json") if plan is None else plan
    _validate_old_plan(plan)
    expected_datasets = {d["dataset_id"] for d in plan["datasets"]}
    inherited = plan.get("prediction_settings")
    for run in plan["runs"]:
        path = root / "predictions" / run["run_id"] / "run_complete.json"
        if not path.is_file():
            raise ValueError("旧评估缺少完整运行记录：" + str(path))
        value = read_json(path)
        if (value.get("status") != "complete" or value.get("plan_id") != plan["plan_id"]
                or value.get("run_id") != run["run_id"]
                or set(value.get("datasets", [])) != expected_datasets):
            raise ValueError("旧评估运行记录与冻结计划不一致：" + str(path))
        settings = value.get("prediction_settings", {})
        if set(settings) != SETTING_KEYS:
            raise ValueError("旧预测设置字段不完整/不支持：" + str(path))
        if inherited is None:
            inherited = settings
        elif inherited != settings:
            raise ValueError("旧模型/种子使用了不同导出设置，不能自动混合复用")
    if (inherited["workers"] != 0 or inherited["amp_policy"] != "training_checkpoint_config"
            or inherited["embedding_dtype"] not in ("float16", "float32")
            or int(inherited["batch_size"]) < 1 or int(inherited["cpu_threads"]) < 1):
        raise ValueError("旧预测设置不受本修订入口支持")
    return copy.deepcopy(inherited)


@contextmanager
def previous_read_lock(root):
    # 与旧队列的排他锁协作，避免边预测边迁移。只读打开，不改旧文件内容。
    lock = root / ".figure4_queue.lock"
    if not lock.is_file():
        yield
        return
    with lock.open("r", encoding="utf-8") as stream:
        try:
            fcntl.flock(stream.fileno(), fcntl.LOCK_SH | fcntl.LOCK_NB)
        except BlockingIOError as error:
            raise RuntimeError("旧 Figure 4 队列仍在运行，请等它退出后再复用") from error
        yield


@contextmanager
def correction_write_lock(output, parent_queue_pid=None):
    """独立CLI也遵守队列锁；父driver持锁时只允许它的直接准备子进程。"""
    output.mkdir(parents=True, exist_ok=True)
    queue_path = output / ".figure4_queue.lock"
    with queue_path.open("a+", encoding="utf-8") as queue, \
         (output / ".figure4_correction_prepare.lock").open("a+", encoding="utf-8") as prepare:
        if parent_queue_pid is not None:
            if int(parent_queue_pid) != os.getppid():
                raise RuntimeError("准备进程的父队列PID不匹配")
            queue.seek(0)
            try:
                owner = json.load(queue)
            except ValueError as error:
                raise RuntimeError("找不到可验证的父队列锁记录") from error
            if owner.get("pid") != int(parent_queue_pid):
                raise RuntimeError("父队列PID与输出锁记录不一致")
            try:
                fcntl.flock(queue.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
            except BlockingIOError:
                pass  # 父队列确实仍持有它的排他锁。
            else:
                fcntl.flock(queue.fileno(), fcntl.LOCK_UN)
                raise RuntimeError("父队列没有持有输出排他锁")
        else:
            try:
                fcntl.flock(queue.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
            except BlockingIOError as error:
                raise RuntimeError("此新输出目录已有Figure 4队列；不能同时单独准备") from error
        try:
            fcntl.flock(prepare.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError as error:
            raise RuntimeError("此新输出目录已有参考修正准备进程") from error
        yield


def corrected_items(root, old_datasets, replace_controls=False):
    """匹配设计身份，不依赖旧新文件 basename 完全相同。"""
    roles = {"target", "control"} if replace_controls else {"target"}
    expected = {(f"XNA{i}", role, strand) for i in range(13, 17)
                for role in roles for strand in ("pos", "neg")}
    old = {}
    for dataset in old_datasets:
        key = (dataset["template"], dataset["role"], dataset["file_strand"])
        if key in expected:
            if key in old:
                raise ValueError("旧评估中待替换设计身份重复：" + str(key))
            old[key] = dataset
    if set(old) != expected:
        raise ValueError("旧评估缺少待替换的模板/角色/方向：" + str(sorted(expected - set(old))))
    found = {}
    if not root.is_dir():
        raise FileNotFoundError("新参考拆分目录不存在：" + str(root))
    for feature in sorted(root.glob("*.csv")):
        if any(word in feature.name.lower() for word in ("summary", "prediction", "metrics", "label_count", "confusion")):
            continue
        parsed = parse_legacy_filename(feature)
        key = (parsed["template"], parsed["role"], parsed["file_strand"])
        if key not in expected:
            continue
        if key in found:
            raise ValueError("新目录中同一模板/角色/方向匹配多个文件：" + str(key))
        metadata = metadata_candidate(feature)
        if not metadata:
            raise ValueError("新拆分文件缺少逐行 metadata；请使用 --write_split_meta：" + str(feature))
        previous = old[key]
        item = {name: previous["identity"]["declarations"].get(name, previous.get(name, "unknown"))
                for name in DECLARATION_KEYS}
        item.update(parsed, feature_csv=str(feature.resolve()), metadata_tsv=metadata,
                    cohort="XNA13-16")
        found[key] = (previous, item)
    if set(found) != expected:
        raise ValueError("新目录缺少待替换模板/角色/方向：" + str(sorted(expected - set(found))))
    return [found[key] for key in sorted(found)]


def _training_hashes(runs):
    values = set()
    def collect(value):
        if isinstance(value, dict):
            if "sha256" in value:
                values.add(value["sha256"])
            for child in value.values():
                collect(child)
        elif isinstance(value, list):
            for child in value:
                collect(child)
    for run in runs:
        collect(run.get("training_input_fingerprints", {}))
    return values


def _merged_identity_audit(old_plan, replacements, all_datasets, output):
    fresh = independence_audit(replacements, old_plan["runs"], output)
    inherited = old_plan.get("external_identity_audit", {})
    unchanged = [d for d in all_datasets if d["dataset_id"] not in {x["dataset_id"] for x in replacements}]
    old_has_real = any(d.get("real_read_id_rows", 0) for d in unchanged)
    old_checked = inherited.get("read_overlap_status") == "verified_on_available_real_ids"
    fresh_has_real = bool(fresh.get("real_read_id_rows"))
    fresh_checked = fresh.get("read_overlap_status") == "verified_on_available_real_ids"
    checked = (not old_has_real or old_checked) and (not fresh_has_real or fresh_checked)
    result = {"read_overlap_status": "verified_on_available_real_ids" if checked and (old_has_real or fresh_has_real) else "not_fully_established",
        "real_read_id_rows": sum(int(d.get("real_read_id_rows", 0)) for d in all_datasets),
        "total_rows": sum(int(d["n_samples"]) for d in all_datasets),
        "unique_available_real_reads": None,
        "unique_read_count_scope": "Not recomputed across reused and corrected cohorts; no sum of potentially overlapping unique counts.",
        "overlap_counts": {}, "training_membership_database_available": fresh["training_membership_database_available"],
        "template_run_independence_status": "not_fully_established",
        "training_evaluation_scope": fresh.get("training_evaluation_scope", "unknown"),
        "inherited_audit_for_unchanged_subset": inherited,
        "corrected_subset_audit": fresh,
        "audit_database_scope": "external_read_audit.sqlite contains corrected subset only; reused subset retains prior audit provenance.",
        "experimenter_read_independence_statement": "The experimenter states that each external read contributes one 6mer window; this statement is retained separately from automated ID availability checks.",
        "notes": ["保留实验设计提供的read级独立性说明；缺少可追溯ID时不合成ID或伪造read聚类区间。",
                  "新参考的正确性仍取决于实际提取使用的参考序列；这里验证文件对齐、转换和冻结预测。",
                  "旧XNA13–16 target及其旧汇总已被本次参考修正替代，不能用旧错误参考的结果解释生物学泛化。"]}
    atomic_json(output / "external_identity_audit.json", result)
    return result


def _reuse_one(previous, output, old_plan, plan, run, dataset, mode, reserve):
    old_folder = previous / "predictions" / run["run_id"] / dataset["dataset_id"]
    folder = output / "predictions" / run["run_id"] / dataset["dataset_id"]
    settings = plan["prediction_settings"]
    if folder.exists() and validate_completion(folder, plan["plan_id"], run["run_id"], dataset["dataset_id"], settings, dataset["n_samples"]):
        completed = read_json(folder / "completion.json")
        if completed.get("checkpoint_sha256") != run["checkpoint_sha256"] or completed.get("temperature") != run["temperature"]:
            raise RuntimeError("已复用结果checkpoint/温度与冻结权重不一致：" + str(folder))
        reused = completed.get("reuse_provenance", {})
        if reused.get("reuse_mode") != mode:
            raise ValueError("恢复不能改变reuse_mode；请沿用原方式或使用新的输出目录")
        if reused.get("previous_plan_id") != old_plan["plan_id"] or reused.get("source_completion_sha256") != sha256(old_folder / "completion.json"):
            raise RuntimeError("已复用结果的来源完成记录发生变化：" + str(folder))
        return False
    if not validate_completion(old_folder, old_plan["plan_id"], run["run_id"], dataset["dataset_id"], settings, dataset["n_samples"]):
        raise ValueError("无法复用旧结果：文件缺失、未完成或SHA不匹配：" + str(old_folder)
                         + "；请先恢复该旧文件，或在单独目录完整运行全部外部预测。")
    marker_path = old_folder / "completion.json"
    marker_hash = sha256(marker_path)
    marker = read_json(marker_path)
    if marker.get("checkpoint_sha256") != run["checkpoint_sha256"] or marker.get("temperature") != run["temperature"]:
        raise ValueError("旧预测 checkpoint/校准温度与冻结权重不一致：" + str(old_folder))
    required = sum(v["size_bytes"] for v in marker["outputs"].values()) if mode == "copy" else 65536
    ensure_disk(output, required, reserve)
    temporary = folder.with_name(folder.name + ".reuse-partial")
    temporary.parent.mkdir(parents=True, exist_ok=True)
    if temporary.exists():
        shutil.rmtree(temporary)
    temporary.mkdir()
    for name, info in marker["outputs"].items():
        source, destination = old_folder / name, temporary / name
        if mode == "hardlink":
            try:
                os.link(source, destination)
            except OSError as error:
                raise OSError(error.errno, "无法建立硬链接；新旧目录应在同一文件系统。跨盘可显式使用 --reuse_mode symlink（须保留旧目录），或有足够容量时 --reuse_mode copy。" ) from error
        elif mode == "symlink":
            destination.symlink_to(source.resolve())
        elif mode == "copy":
            shutil.copyfile(source, destination)
            if sha256(destination) != info["sha256"]:
                raise RuntimeError("复用副本SHA不匹配：" + str(destination))
        else:
            raise ValueError("不支持的reuse_mode：" + str(mode))
        if destination.stat().st_size != info["size_bytes"]:
            raise RuntimeError("复用期间源文件发生变化：" + str(source))
    if sha256(marker_path) != marker_hash:
        raise RuntimeError("复用期间源完成记录发生变化：" + str(marker_path))
    marker["plan_id"] = plan["plan_id"]
    marker["reuse_provenance"] = {"previous_output": str(previous), "previous_plan_id": old_plan["plan_id"],
        "source_completion": str(marker_path), "source_completion_sha256": marker_hash,
        "reuse_mode": mode, "reason": "Unchanged external dataset, checkpoint, validation temperature and prediction settings; no new inference.",
        "inference_timing_scope": "Inherited original inference timing; not newly measured inference performance."}
    atomic_json(temporary / "completion.json", marker)
    if folder.exists():
        shutil.rmtree(folder)
    temporary.replace(folder)
    return True


def _scientific_dataset(dataset):
    return {**dataset, "outputs": {name: {key: value for key, value in info.items() if key != "mtime_ns"}
                                    for name, info in dataset["outputs"].items()}}


def _prepare_correction_locked(args):
    previous = Path(args.previous_output).expanduser().resolve()
    output_value = getattr(args, "output_dir", None) or getattr(args, "output_root", None)
    if output_value is None:
        raise ValueError("必须指定新的output_dir/output_root")
    output = Path(output_value).expanduser().resolve()
    corrected = Path(args.corrected_13_16).expanduser().resolve()
    if previous == output or previous in output.parents or output in previous.parents:
        raise ValueError("修订结果必须使用与旧结果互不包含的新目录，不能覆盖旧结果")
    if corrected == output or corrected in output.parents or output in corrected.parents:
        raise ValueError("新输出目录不能覆盖或包含新拆分输入目录，也不能放入其中")
    output.mkdir(parents=True, exist_ok=True)
    plan_path = output / "evaluation_plan.json"
    existing = read_json(plan_path) if plan_path.is_file() else None
    if existing is not None and not getattr(args, "resume", False):
        raise ValueError("新目录已有冻结计划；继续相同修正请加 --resume")
    with previous_read_lock(previous):
        old_path = previous / "evaluation_plan.json"
        old_fp = fingerprint(old_path)
        old_plan = read_json(old_path)
        _validate_old_plan(old_plan)
        settings = inherit_prediction_settings(previous, old_plan)
        for run in old_plan["runs"]:
            verify_run(run)
        training = Path(old_plan["training_results"]).resolve()
        if output == training or output in training.parents or training in output.parents:
            raise ValueError("新输出不能覆盖或放入原训练目录")
        replace_controls = bool(getattr(args, "replace_controls", False))
        mode = getattr(args, "reuse_mode", "hardlink")
        pairs = corrected_items(corrected, old_plan["datasets"], replace_controls)
        from audit_corrected_split import audit_corrected_split
        split_audit = audit_corrected_split(corrected, [item for _, item in pairs],
            original_pos_csv=getattr(args, "original_pos_csv", None),
            original_neg_csv=getattr(args, "original_neg_csv", None),
            original_pos_meta=getattr(args, "original_pos_meta", None),
            original_neg_meta=getattr(args, "original_neg_meta", None))
        split_audit_hash = identity_hash(split_audit)
        if existing is not None:
            provenance = existing.get("correction_provenance", {})
            if (provenance.get("previous_plan_file") != old_fp
                    or provenance.get("corrected_split_audit_sha256") != split_audit_hash
                    or provenance.get("replace_controls") != replace_controls
                    or provenance.get("reuse_mode") != mode
                    or existing.get("prediction_settings") != settings):
                raise ValueError("原结果/新数据/替换范围在冻结后改变；请用新的输出目录")
        reserve = float(getattr(args, "disk_reserve_gib", 5))
        replacements = []
        replaced = {}
        for previous_dataset, item in pairs:
            dataset = prepare_cache(item, old_plan["preprocessing"], output, reserve)
            replacements.append(dataset)
            replaced[previous_dataset["dataset_id"]] = dataset
        for dataset in replacements:
            if dataset["identity"]["feature"]["sha256"] in _training_hashes(old_plan["runs"]):
                raise ValueError("新外部文件与内部训练/验证/测试文件相同：" + dataset["feature_csv"])
        datasets = [replaced.get(d["dataset_id"], copy.deepcopy(d)) for d in old_plan["datasets"]]
        audit = _merged_identity_audit(old_plan, replacements, datasets, output)
        provenance = {"kind": "reference_sequence_correction", "previous_output": str(previous),
            "previous_plan_id": old_plan["plan_id"], "previous_plan_file": old_fp,
            "corrected_root": str(corrected), "corrected_split_audit_sha256": split_audit_hash,
            "replace_controls": replace_controls, "reuse_mode": mode, "superseded_dataset_ids": sorted(replaced),
            "replacement_map": {key: value["dataset_id"] for key, value in replaced.items()},
            "invalid_previous_scope": "XNA13–16 target predictions and all summaries containing those old-reference samples.",
            "interpretation": "Reference/data correction, not retraining, architecture improvement, seed selection or external calibration.",
            "controls_policy": "Replace corrected controls too" if replace_controls else "Retain original DNA/PC13–16 controls; no assumption that newly extracted controls are byte-identical.",
            "reuse_policy": "Reuse only unchanged files after complete output SHA/size/plan/settings/checkpoint/temperature verification."}
        core = {key: copy.deepcopy(old_plan[key]) for key in ("task", "training_results", "runs", "preprocessing",
                "missing_template_roles", "models", "seeds", "feature_schemes", "selection_policy") if key in old_plan}
        core.update(version=VERSION, datasets=datasets, prediction_settings=settings,
                    external_identity_audit=audit, correction_provenance=provenance,
                    replacement_dataset_ids=[d["dataset_id"] for d in replacements])
        scientific = {**core, "datasets": [_scientific_dataset(d) for d in datasets]}
        core["plan_id"] = identity_hash(scientific)
        sizes = [sum(d["outputs"][name]["size_bytes"] for name in ("merged.npy", "raw.npy", "labels.npy")) for d in replacements]
        core.update(total_cache_bytes=sum(sizes), max_file_cache_bytes=max(sizes),
                    shared_cache_budget_bytes=max(sizes), cache_budget_scope="corrected subset only",
                    created_at=time.time())
        if existing is not None and existing["plan_id"] != core["plan_id"]:
            raise ValueError("冻结计划与本次修正不一致；请用新的输出目录")
        if fingerprint(old_path) != old_fp:
            raise RuntimeError("准备期间旧评估计划发生变化")
        atomic_json(output / "corrected_split_audit.json", split_audit)
        atomic_json(output / "reference_correction_provenance.json", provenance)
        # 先提交科学计划；ENOSPC/链接中断后按同计划 --resume 可继续复用剩余文件。
        atomic_json(plan_path, core)
        fields = ["dataset_id", "cohort", "template", "role", "reference", "file_strand", "feature_csv", "metadata_tsv", "n_samples", "real_read_id_rows", "unavailable_read_id_rows"]
        with (output / "discovered_inputs.csv").open("w", encoding="utf-8", newline="") as stream:
            writer = csv.DictWriter(stream, fieldnames=fields, extrasaction="ignore")
            writer.writeheader()
            writer.writerows(datasets)
        unchanged = [d for d in datasets if d["dataset_id"] not in set(core["replacement_dataset_ids"])]
        reused_count = 0
        for run in core["runs"]:
            for dataset in unchanged:
                reused_count += _reuse_one(previous, output, old_plan, core, run, dataset, mode, reserve)
            print(f"[REUSE] {run['run_id']}：已核验 {len(unchanged)} 个不变文件；待重算 {len(replacements)} 个文件", flush=True)
        atomic_json(output / "reuse_manifest.json", {"status": "complete", "plan_id": core["plan_id"],
            "previous_plan_id": old_plan["plan_id"], "reuse_mode": mode,
            "reused_dataset_ids": [d["dataset_id"] for d in unchanged],
            "replacement_dataset_ids": core["replacement_dataset_ids"], "runs": [r["run_id"] for r in core["runs"]],
            "newly_linked_or_copied_exports": reused_count,
            "source_directory_required": "Symlinks require the old directory; hardlinks keep file contents without data duplication. Preserve original records for provenance."})
        print(f"[CORRECTION PLAN] {len(core['runs'])}组冻结权重 × {len(replacements)}个新文件；复用其余{len(unchanged)}个文件，不重新训练。", flush=True)
        return core


def prepare_correction(args):
    output_value = getattr(args, "output_dir", None) or getattr(args, "output_root", None)
    if output_value is None:
        raise ValueError("必须指定新的output_dir/output_root")
    output = Path(output_value).expanduser().resolve()
    previous = Path(args.previous_output).expanduser().resolve()
    corrected = Path(args.corrected_13_16).expanduser().resolve()
    # 锁文件本身也不能写入任何源目录。
    if (previous == output or previous in output.parents or output in previous.parents
            or corrected == output or corrected in output.parents or output in corrected.parents):
        raise ValueError("新输出必须与旧结果及新拆分输入目录互不包含")
    with correction_write_lock(output, getattr(args, "parent_queue_pid", None)):
        return _prepare_correction_locked(args)


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--previous_output", required=True)
    parser.add_argument("--corrected_13_16", default=DEFAULT_CORRECTED)
    parser.add_argument("--output_dir", "--output_root", dest="output_dir", required=True)
    parser.add_argument("--replace_controls", action="store_true")
    parser.add_argument("--reuse_mode", choices=("hardlink", "symlink", "copy"), default="hardlink")
    for name in ("original_pos_csv", "original_neg_csv", "original_pos_meta", "original_neg_meta"):
        parser.add_argument("--" + name)
    parser.add_argument("--disk_reserve_gib", type=float, default=5)
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--parent_queue_pid", type=int, help=argparse.SUPPRESS)
    args = parser.parse_args()
    if args.disk_reserve_gib < 0:
        parser.error("disk_reserve_gib不能小于0")
    return args


if __name__ == "__main__":
    try:
        prepare_correction(parse_args())
    except Exception as error:
        print(f"[ERROR] {type(error).__name__}: {error}", file=sys.stderr)
        raise SystemExit(1)
