#!/usr/bin/env python3
"""审计外部CSV，建立所有模型共享的只读缓存和冻结预测计划。"""

from __future__ import annotations

import argparse
import csv
import json
from collections import Counter
from itertools import zip_longest
from pathlib import Path
import re
import shutil
import sqlite3
import sys
import time
from urllib.parse import quote

import numpy as np

from figure4_common import (VERSION, CLASSES, MODELS, SEEDS, DEFAULT_RESULTS, DEFAULT_01_12,
    DEFAULT_13_16, DEFAULT_17_20, atomic_json, discover_datasets, discover_runs,
    ensure_disk, fingerprint, identity_hash, known, parse_reference, read_json,
    runtime_modules, sha256, strict_label)


def check_metadata_kmer(sequence, other, where):
    """核对侧翼；双方保留天然目标碱基时，同时核对完整原始6mer。"""
    motif = sequence[:3] + "N" + sequence[4:]
    if other[:3] + "N" + other[4:] != motif:
        raise ValueError(where + "：metadata motif不对应")
    if sequence[3] in "ATCG" and other[3] in "ATCG" and sequence != other:
        raise ValueError(where + "：metadata原始kmer不对应")


def metadata_read_identity(metadata):
    """只使用声明为真实身份的read ID，未知或代理身份保持未知。"""
    if not metadata:
        return "unknown", "unavailable"
    read_id = next((str(metadata.get(key, "")).strip() for key in
                    ("original_read_id", "read_id") if known(metadata.get(key))), "unknown")
    kind = str(metadata.get("read_id_kind", "sequencing_read_id")).strip().lower()
    invalid_kind = not known(kind) or any(word in kind for word in
                                        ("proxy", "unavailable", "unknown", "synthetic", "row"))
    if not known(read_id) or invalid_kind or read_id.lower().startswith("positive-row:"):
        return "unknown", "unavailable"
    return read_id, "sequencing_read_id"


def metadata_reference(metadata, item, where):
    """保留metadata参考名称，并拒绝能识别但与清单冲突的模板或角色。"""
    reference, basis = item["reference"], "declared_file_alias"
    for field in ("chrom", "reference"):
        value = str(metadata.get(field, "")).strip()
        if not known(value):
            continue
        try:
            template, role = parse_reference(value)
        except ValueError:
            # 参考可以是无法映射的真实名称，照实保存，不臆测模板身份。
            pass
        else:
            if template != item["template"] or role != item["role"]:
                raise ValueError(where + "：metadata." + field + "与清单template/role冲突")
        if basis != "metadata":
            reference, basis = value, "metadata"
    return reference, basis


def metadata_rows(path):
    path = Path(path)
    with path.open(encoding="utf-8-sig", newline="") as stream:
        reader = csv.DictReader(stream, delimiter="\t" if path.suffix.lower() == ".tsv" else ",")
        if not reader.fieldnames or len(set(reader.fieldnames)) != len(reader.fieldnames):
            raise ValueError("metadata 缺表头或表头重复：" + str(path))
        if "read_id" not in reader.fieldnames:
            raise ValueError("metadata 必须包含真实read_id；没有时请不提供该文件")
        for row in reader:
            if None in row or any(value is None for value in row.values()):
                raise ValueError(f"metadata格式错误：{path}:{reader.line_num}")
            yield row


def cache_valid(directory, identity):
    try:
        saved = read_json(directory / "cache_manifest.json")
        if saved.get("identity") != identity or saved.get("status") != "complete":
            return False
        outputs = saved.get("outputs", {})
        if set(outputs) != {"merged.npy", "raw.npy", "labels.npy", "metadata.jsonl"}:
            return False
        for name, info in outputs.items():
            path = directory / name
            stat = path.stat()
            if (stat.st_size != info["size_bytes"] or stat.st_mtime_ns != info["mtime_ns"]
                    or sha256(path) != info["sha256"]):
                return False
        for name, shape in [("merged.npy", [saved["n_samples"], 6, 13]),
                            ("raw.npy", [saved["n_samples"], 6, identity["preprocessing"]["raw_signal_len"]]),
                            ("labels.npy", [saved["n_samples"]])]:
            array = np.load(directory / name, mmap_mode="r", allow_pickle=False)
            expected_dtype = np.int64 if name == "labels.npy" else np.float32
            if list(array.shape) != shape or array.dtype != expected_dtype:
                return False
        return True
    except (OSError, ValueError, KeyError, TypeError):
        return False


def prepare_cache(item, preprocessing, output_root, disk_reserve_gib):
    feature = Path(item["feature_csv"])
    feature_fp = fingerprint(feature)
    metadata_fp = fingerprint(item["metadata_tsv"]) if item.get("metadata_tsv") else None
    identity = {"feature": feature_fp, "metadata": metadata_fp,
                "declarations": item, "preprocessing": preprocessing, "cache_version": VERSION,
                "metadata_identity_policy": "strict-kmer-reference-read-kind-v1"}
    dataset_id = item["template"] + "_" + item["role"] + "_" + item["file_strand"] + "_" + identity_hash(identity)[:12]
    directory = output_root / "cache" / dataset_id
    if cache_valid(directory, identity):
        saved = read_json(directory / "cache_manifest.json")
        print(f"[CACHE] 复用 {dataset_id}：{saved['n_samples']:,}行", flush=True)
        return {**item, **saved, "dataset_id": dataset_id, "cache_dir": str(directory)}
    runtime, _ = runtime_modules()
    with feature.open(encoding="utf-8-sig", newline="") as stream:
        first = next(csv.reader(stream), [])
    n = feature_fp["physical_lines"] - int(bool(first) and first[0].strip().lower() == "kmer")
    if n < 1:
        raise ValueError("外部文件没有样本：" + str(feature))
    bytes_needed = n * (6 * 13 * 4 + 6 * preprocessing["raw_signal_len"] * 4 + 8 + 1800)
    ensure_disk(output_root, bytes_needed, disk_reserve_gib)
    directory.parent.mkdir(parents=True, exist_ok=True)
    temporary = directory.with_name(directory.name + ".building")
    if temporary.exists():
        shutil.rmtree(temporary)
    temporary.mkdir()
    merged = np.lib.format.open_memmap(temporary / "merged.npy", mode="w+", dtype="float32", shape=(n, 6, 13))
    raw = np.lib.format.open_memmap(temporary / "raw.npy", mode="w+", dtype="float32", shape=(n, 6, preprocessing["raw_signal_len"]))
    labels = np.lib.format.open_memmap(temporary / "labels.npy", mode="w+", dtype="int64", shape=(n,))
    count, real_reads, mask_replacements = 0, 0, 0
    label_counts, strands = Counter(), Counter()
    metadata_iter = metadata_rows(item["metadata_tsv"]) if item.get("metadata_tsv") else None
    sentinel = object()
    features = runtime._feature_rows(feature)
    iterator = zip_longest(features, metadata_iter, fillvalue=sentinel) if metadata_iter is not None else ((row, {}) for row in features)
    try:
        with (temporary / "metadata.jsonl").open("w", encoding="utf-8") as manifest:
            for feature_item, metadata in iterator:
                if feature_item is sentinel or metadata is sentinel:
                    raise ValueError("外部CSV/metadata行数不一致：" + str(feature))
                line, row = feature_item
                where = f"{feature}:{line}"
                if count >= n:
                    raise ValueError(where + "：记录数超过物理行数")
                label = strict_label(row[-1], where)
                seq = runtime._sequence(row[0], 6, where)
                motif = seq[:3] + "N" + seq[4:]
                if item["role"] == "control" and label >= 4:
                    raise ValueError(where + "：control含非天然真值，请核对文件角色和标签")
                if label < 4 and seq[3] in "ATCG" and seq[3] != CLASSES[label]:
                    raise ValueError(where + "：第四位天然碱基与CSV真值冲突")
                if metadata:
                    if metadata.get("label", "").strip() and strict_label(metadata["label"], where + " metadata") != label:
                        raise ValueError(where + "：metadata标签不对应")
                    if metadata.get("kmer", "").strip():
                        other = runtime._sequence(metadata["kmer"], 6, where + " metadata.kmer")
                        check_metadata_kmer(seq, other, where)
                read_id, read_kind = metadata_read_identity(metadata)
                if read_kind == "sequencing_read_id":
                    real_reads += 1
                reference, reference_basis = metadata_reference(metadata, item, where)
                strand = metadata.get("bam_strand") or "unknown"
                if strand not in ("+", "-", "unknown"):
                    raise ValueError(where + "：bam_strand必须为+/-/unknown，不从pos/neg猜测")
                mask_replacements += seq[3] != "N"
                onehot = np.zeros((6, 5), dtype=np.float32)
                onehot[np.arange(6), [runtime.BASES[base] for base in motif]] = 1
                scalars = np.stack([runtime._scalar_values(row[i], 6, where) for i in range(1, 9)], axis=1)
                scalars[:, 3] /= preprocessing["dwell_divisor"]
                scalars[:, 4] /= preprocessing["quality_divisor"]
                if not np.isfinite(scalars).all():
                    raise ValueError(where + "：归一化出现非有限值")
                merged[count, :, :5], merged[count, :, 5:] = onehot, scalars
                raw[count] = runtime.rectify_signal(row[9], 6, preprocessing["raw_signal_len"], where)
                labels[count] = label
                record = {"sample_index": count, "sample_id": feature_fp["sha256"] + ":" + str(line),
                          "source_file": str(feature), "source_row": line, "source_sha256": feature_fp["sha256"],
                          "read_id": read_id, "read_id_kind": read_kind, "original_kmer": seq,
                          "motif": motif, "label": label, "true_class": CLASSES[label],
                          "reference": reference,
                          "template": item["template"], "cohort": item["cohort"], "role": item["role"],
                          "file_strand": item["file_strand"], "bam_strand": strand,
                          "template_group": metadata.get("template_group") or item["template_group"],
                          "pair_id": metadata.get("pair_id") or item["pair_id"],
                          "run_id": metadata.get("run_id") or item["run_id"],
                          "source_dataset": metadata.get("source_dataset") or item["source_dataset"],
                          "position": metadata.get("position") or metadata.get("pos") or "unknown",
                          "reference_identity_basis": reference_basis,
                          "original_metadata": metadata}
                manifest.write(json.dumps(record, ensure_ascii=False, allow_nan=False) + "\n")
                label_counts[str(label)] += 1
                strands[strand] += 1
                count += 1
                if count % 100000 == 0:
                    print(f"[CACHE] {dataset_id} {count:,}/{n:,}", flush=True)
        if count != n:
            raise ValueError("外部文件记录数不一致：" + str(feature))
        for array in (merged, raw, labels):
            array.flush()
        del merged, raw, labels
        if fingerprint(feature) != feature_fp or (metadata_fp is not None and fingerprint(item["metadata_tsv"]) != metadata_fp):
            raise ValueError("准备过程中输入发生变化：" + str(feature))
        saved = {"status": "complete", "identity": identity, "n_samples": count,
                 "label_counts": dict(label_counts), "real_read_id_rows": real_reads,
                 "unavailable_read_id_rows": count - real_reads, "bam_strand_counts": dict(strands),
                 "fourth_base_replacements": int(mask_replacements),
                 "file_identity_note": "pos/neg是原始文件分组，不等同于已核实的BAM方向；真值只来自CSV。"}
        saved["outputs"] = {}
        for name in ("merged.npy", "raw.npy", "labels.npy", "metadata.jsonl"):
            path = temporary / name
            before = path.stat()
            digest = sha256(path)
            after = path.stat()
            if (before.st_size, before.st_mtime_ns) != (after.st_size, after.st_mtime_ns):
                raise ValueError("缓存写入后仍在变化：" + str(path))
            saved["outputs"][name] = {"size_bytes": before.st_size,
                                      "mtime_ns": before.st_mtime_ns, "sha256": digest}
        atomic_json(temporary / "cache_manifest.json", saved)
        if not cache_valid(temporary, identity):
            raise ValueError("新缓存完整性核验失败：" + str(temporary))
        invalid_backup = None
        if directory.exists():
            invalid_backup = directory.with_name(directory.name + ".invalid." + str(time.time_ns()))
            directory.rename(invalid_backup)
        try:
            temporary.rename(directory)
        except BaseException:
            if invalid_backup is not None and not directory.exists():
                invalid_backup.rename(directory)
            raise
        if invalid_backup is not None:
            # 仅清理本次替换成功后的损坏派生缓存，绝不触碰用户源文件。
            shutil.rmtree(invalid_backup)
            print(f"[CACHE] 重建成功，已清理本次损坏派生缓存：{invalid_backup}", flush=True)
        print(f"[CACHE] 完成 {dataset_id}：{count:,}行", flush=True)
        return {**item, **saved, "dataset_id": dataset_id, "cache_dir": str(directory)}
    except BaseException:
        # 保留本次building供故障定位；重试时只重建未完成缓存。
        raise


def independence_audit(datasets, runs, output_root):
    first = Path(runs[0]["run_dir"])
    training_audit = read_json(first / "data_audit.json") if (first / "data_audit.json").is_file() else {}
    database = first / "read_membership.sqlite"
    connection = None
    if database.is_file():
        connection = sqlite3.connect("file:" + quote(str(database)) + "?mode=ro", uri=True)
        connection.execute("PRAGMA query_only=ON")
    overlap, real_rows, examples = Counter(), 0, []
    membership = output_root / "external_read_audit.sqlite"
    audit_db = sqlite3.connect(membership)
    audit_db.execute("CREATE TABLE IF NOT EXISTS external_reads(read_id TEXT PRIMARY KEY)")
    audit_db.execute("DELETE FROM external_reads")
    try:
        for dataset in datasets:
            with (Path(dataset["cache_dir"]) / "metadata.jsonl").open(encoding="utf-8") as stream:
                for line in stream:
                    row = json.loads(line)
                    if row["read_id_kind"] != "sequencing_read_id":
                        continue
                    real_rows += 1
                    inserted = audit_db.execute("INSERT OR IGNORE INTO external_reads VALUES(?)", (row["read_id"],)).rowcount
                    if inserted and connection is not None:
                        matches = connection.execute("SELECT DISTINCT split FROM read_membership WHERE read_id=?", (row["read_id"],)).fetchall()
                        for (split,) in matches:
                            overlap[split] += 1
                            if len(examples) < 20:
                                examples.append({"read_id": row["read_id"], "split": split})
        audit_db.commit()
        unique = audit_db.execute("SELECT COUNT(*) FROM external_reads").fetchone()[0]
    finally:
        audit_db.close()
        if connection is not None:
            connection.close()
    audit = {"read_overlap_status": "verified_on_available_real_ids" if connection is not None and real_rows else "not_established",
             "real_read_id_rows": real_rows, "unique_available_real_reads": unique,
             "total_rows": sum(d["n_samples"] for d in datasets), "overlap_counts": dict(overlap),
             "overlap_examples": examples, "training_membership_database_available": database.is_file(),
             "template_run_independence_status": "not_fully_established",
             "training_evaluation_scope": training_audit.get("evaluation_scope", "unknown"),
             "notes": ["缺少原始read UUID时不合成read身份、不提供假read-cluster区间。",
                       "文件名声明的模板与pos/neg分组不等于已验证测序run、来源或方向独立。",
                       "训练阳性参考/run未知时，不能仅凭不同文件名证明模板/run完全留出。"]}
    atomic_json(output_root / "external_identity_audit.json", audit)
    if overlap:
        raise ValueError("外部真实read与内部划分重叠，不能作为留出测试：" + str(dict(overlap)))
    return audit


def prepare(args):
    output = Path(args.output_root).expanduser().resolve()
    output.mkdir(parents=True, exist_ok=True)
    root, runs, preprocessing = discover_runs(args.training_results, args.models, args.seeds, args.feature_schemes)
    if output == root or output in root.parents:
        raise ValueError("外部输出不能覆盖训练目录或它的父目录")
    files, missing = discover_datasets(args)
    datasets = [prepare_cache(item, preprocessing, output, args.disk_reserve_gib) for item in files]
    training_hashes = set()
    def collect(value):
        if isinstance(value, dict):
            if "sha256" in value:
                training_hashes.add(value["sha256"])
            for child in value.values():
                collect(child)
        elif isinstance(value, list):
            for child in value:
                collect(child)
    collect(runs[0]["training_input_fingerprints"])
    for dataset in datasets:
        if dataset["identity"]["feature"]["sha256"] in training_hashes:
            raise ValueError("外部文件与训练/验证/内部测试输入完全相同：" + dataset["feature_csv"])
    audit = independence_audit(datasets, runs, output)
    core = {"version": VERSION, "task": "6-class", "training_results": str(root),
            "runs": runs, "datasets": datasets, "preprocessing": preprocessing,
            "missing_template_roles": missing, "external_identity_audit": audit,
            "models": args.models, "seeds": args.seeds, "feature_schemes": args.feature_schemes,
            "selection_policy": "全部预先指定模型/种子，使用冻结最佳checkpoint和validation温度；外部数据不拟合参数。"}
    # 重建损坏缓存可改变mtime，但同一内容不应变成不同科学评估计划。
    # mtime保留供worker轻量检查；计划身份冻结其内容hash和大小。
    scientific_core = {**core, "datasets": [
        {**dataset, "outputs": {name: {key: value for key, value in info.items()
                                       if key != "mtime_ns"}
                                 for name, info in dataset["outputs"].items()}}
        for dataset in datasets]}
    core["plan_id"] = identity_hash(scientific_core)
    cache_sizes = [sum((Path(d["cache_dir"]) / name).stat().st_size for name in ("merged.npy", "raw.npy", "labels.npy")) for d in datasets]
    core.update(total_cache_bytes=sum(cache_sizes), max_file_cache_bytes=max(cache_sizes),
                shared_cache_budget_bytes=max(cache_sizes), created_at=time.time())
    previous = output / "evaluation_plan.json"
    if previous.exists():
        if read_json(previous).get("plan_id") != core["plan_id"]:
            raise ValueError("输出目录已有不同冻结计划，请使用新输出目录；不混合已有预测")
        if not args.resume:
            raise ValueError("目录已有冻结计划；保持输入不变时加 --resume")
        print("[PLAN] 复用同一冻结计划", flush=True)
        return read_json(previous)
    atomic_json(previous, core)
    with (output / "discovered_inputs.csv").open("w", encoding="utf-8", newline="") as stream:
        fields = ["dataset_id", "cohort", "template", "role", "reference", "file_strand", "feature_csv", "metadata_tsv", "n_samples", "real_read_id_rows", "unavailable_read_id_rows"]
        writer = csv.DictWriter(stream, fieldnames=fields, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(datasets)
    print(f"[PLAN] {len(runs)}组冻结权重，{len(datasets)}个文件，{sum(d['n_samples'] for d in datasets):,}窗口；不重新训练", flush=True)
    return core


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--training_results", default=DEFAULT_RESULTS)
    parser.add_argument("--output_root", required=True)
    parser.add_argument("--data_manifest")
    parser.add_argument("--dir_01_12_root", default=DEFAULT_01_12)
    parser.add_argument("--dir_13_16_6", default=DEFAULT_13_16)
    parser.add_argument("--dir_17_20_6", default=DEFAULT_17_20)
    parser.add_argument("--models", nargs="+", choices=MODELS, default=MODELS)
    parser.add_argument("--seeds", nargs="+", type=int, default=SEEDS)
    parser.add_argument("--feature_schemes", nargs="+", default=["full"])
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--allow_partial_templates", action="store_true")
    parser.add_argument("--disk_reserve_gib", type=float, default=5)
    args = parser.parse_args()
    for name in ("models", "seeds", "feature_schemes"):
        values = getattr(args, name)
        if len(set(values)) != len(values):
            parser.error(name + "不允许重复")
    if args.disk_reserve_gib < 0:
        parser.error("disk_reserve_gib不能小于0")
    return args


if __name__ == "__main__":
    try:
        prepare(parse_args())
    except Exception as error:
        print(f"[ERROR] {type(error).__name__}: {error}", file=sys.stderr)
        raise SystemExit(1)
