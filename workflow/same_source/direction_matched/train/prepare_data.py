#!/usr/bin/env python3
"""只读核验严格匹配五分类导出，并保留冻结 read split 发布训练输入。

生产 CLI 不提供 reference_dir；显式 Python API 的 reference_dir 仅供小型测试。
原始 CSV、metadata、报告均不修改，不重复下采样或重新划分。
"""
from __future__ import annotations

import argparse
from collections import Counter
from contextlib import ExitStack
import csv
import fcntl
import hashlib
from itertools import zip_longest
import json
import math
import os
from pathlib import Path
import re
import shutil
import sqlite3
import time

from reference6 import ReferenceLibrary, interval_for_target

HERE = Path(__file__).resolve().parent
VERSION = "1.0.0"
OWNER = "cplx-direction-fiveclass-preparer-v1"
SPLITS = ("train", "validation", "test")
SCENARIOS = {"X_plus": {"strand": "+", "classes": ("A", "T", "C", "G", "X"), "original_ncb_label": 4},
             "Y_minus": {"strand": "-", "classes": ("A", "T", "C", "G", "Y"), "original_ncb_label": 5}}
FEATURE_COLUMNS = ("kmer", "mean", "std", "median", "dwell", "quality",
                   "mismatch", "insertion", "deletion", "signal", "label")
REPORTS = ("EXTRACT_PACKAGE_MANIFEST.json", "extraction_summary.json", "selected_counts.csv",
           "read_audit.json", "source_identity.json", "sampling_plan.json", "README_中文.txt")
REQUIRED_META = {"feature_row", "batch_id", "read_id", "chrom", "bam_strand", "tombo_strand",
                 "strand_match", "split", "source_line", "row_start", "row_end", "target_pos",
                 "kmer_ref_start", "kmer_ref_end", "raw_oriented_6mer", "masked_6mer", "true_base",
                 "label", "strict_match", "original_label", "scenario", "label_mapping"}
EXTRA_META = ("kmer", "original_kmer", "source_dataset", "original_read_id", "read_id_kind",
              "read_identity_basis", "reference_identity_known", "source_row", "source_feature_row",
              "source_file", "run_id", "template_group", "template_identity_basis")
GIB = 1024 ** 3


def sha256(path):
    h = hashlib.sha256()
    with Path(path).open("rb") as f:
        for chunk in iter(lambda: f.read(4 * 1024 ** 2), b""):
            h.update(chunk)
    return h.hexdigest()


def _json(path):
    return json.loads(Path(path).read_text(encoding="utf-8-sig"))


def _write_json(path, value):
    path = Path(path)
    temporary = path.with_name(path.name + ".writing")
    try:
        with temporary.open("w", encoding="utf-8") as f:
            json.dump(value, f, ensure_ascii=False, indent=2, allow_nan=False)
            f.write("\n"); f.flush(); os.fsync(f.fileno())
        os.replace(temporary, path)
    finally:
        temporary.unlink(missing_ok=True)


def _integer(text, name, minimum=0):
    if not re.fullmatch(r"[0-9]+", str(text)):
        raise ValueError("%s必须为非负十进制整数，得到%r" % (name, text))
    value = int(text)
    if value < minimum:
        raise ValueError("%s小于%d" % (name, minimum))
    return value


def assigned_split(read_id, batch_id, seed, ratios):
    digest = hashlib.sha256("\0".join(str(x) for x in (seed, batch_id, read_id)).encode()).digest()
    u = int.from_bytes(digest[:8], "big") / 2 ** 64
    total = sum(ratios)
    return "train" if u < ratios[0] / total else "validation" if u < sum(ratios[:2]) / total else "test"


def _disk(path, reserve):
    if shutil.disk_usage(path).free < reserve * GIB:
        raise OSError(28, "准备目录剩余空间低于 %.2f GiB预留；清理后使用--resume" % reserve)


def _frozen(source_dir, reference_dir, scenario):
    identity = {"version": VERSION, "owner": OWNER, "scenario": scenario,
                "adapter_sha256": sha256(__file__), "reference_helper_sha256": sha256(HERE / "reference6.py"),
                "source_dir": str(source_dir), "reports": {}, "inputs": {}}
    frozen_core = reference_dir / "extract_core"
    for name in REPORTS:
        expected, actual = frozen_core / name, source_dir / name
        if not expected.is_file() or not actual.is_file():
            raise FileNotFoundError("缺少冻结导出报告：" + str(actual))
        if sha256(actual) != sha256(expected):
            raise ValueError("导出报告与随包冻结副本不同：" + name)
        identity["reports"][name] = sha256(actual)
    manifest = _json(source_dir / REPORTS[0])
    if manifest.get("status") != "complete" or manifest.get("file_count") != len(REPORTS) - 1:
        raise ValueError("导出精简包清单未完成或文件数量不正确")
    if set(manifest.get("files", {})) != set(REPORTS) - {REPORTS[0]}:
        raise ValueError("导出清单缺少预期报告")
    for name, info in manifest["files"].items():
        path = source_dir / name
        if path.stat().st_size != info["size_bytes"] or sha256(path) != info["sha256"]:
            raise ValueError("导出清单中报告SHA/大小不一致：" + name)
    summary = _json(source_dir / "extraction_summary.json")
    plan = _json(source_dir / "sampling_plan.json")
    audit = _json(source_dir / "read_audit.json")
    source_identity = _json(source_dir / "source_identity.json")
    contract = _json(reference_dir / "input_contract.json")
    strict = _json(reference_dir / "strict_plan.json")
    if summary.get("status") != "complete" or audit.get("status") != "PASS" or plan.get("status") != "frozen":
        raise ValueError("导出/真实read审计/采样计划尚未完整完成")
    if summary["batch_id"] != contract["batch_id"] or source_identity["batch_id"] != contract["batch_id"]:
        raise ValueError("数据批次与冻结来源不一致")
    if summary["runtime"] != contract["runtime"] or source_identity["runtime"] != contract["runtime"]:
        raise ValueError("原特征提取数值环境记录与冻结凭据不一致")
    if summary["source_chain_sha256"] != audit["full_source_chain_sha256"] or summary["source_chain_sha256"] != source_identity["source_chain_sha256"]:
        raise ValueError("原始信号完整哈希链记录不一致")
    if tuple(contract["split_ratios"]) != (7, 2, 1) or contract["split_seed"] != source_identity["split_seed"]:
        raise ValueError("固定read划分规则改变")
    if audit["cross_split_read_id_count"] != 0 or audit["selected_window_count"] != summary["rows"]:
        raise ValueError("read跨集合或导出总数不一致")
    strict_sha = sha256(reference_dir / "strict_plan.json")
    if strict_sha != plan["plan_sha256"] or strict_sha != summary["frozen_plan_sha256"]:
        raise ValueError("严格匹配方案的SHA与导出不一致")
    if plan["groups"] != strict["groups"] or plan["label_mapping"] != summary["label_mapping"]:
        raise ValueError("严格匹配组合或标签映射不同于冻结导出")
    expected_mapping = {"X_plus": {str(i): i for i in range(5)},
                        "Y_minus": {"0": 0, "1": 1, "2": 2, "3": 3, "5": 4}}
    if summary["label_mapping"] != expected_mapping:
        raise ValueError("不是预期的X/Y五分类标签映射")
    identity["reference_files"] = {}
    for name in ("input_contract.json", "strict_plan.json", "strict_plan_cells.csv", "CPLX_XNA_all_original.fasta"):
        identity["reference_files"][name] = sha256(reference_dir / name)
        if source_identity.get("reference_files", {}).get(name) != identity["reference_files"][name]:
            raise ValueError("原特征参考/数值凭据与冻结导出不一致：" + name)
    fasta = reference_dir / "CPLX_XNA_all_original.fasta"
    if sha256(fasta) != contract["reference_sha256"]:
        raise ValueError("原始X/Y参考FASTA的SHA不一致")
    # Both tasks are bound to the same four committed files. A later source
    # mutation must not silently leave one task using an earlier data version.
    for task in SCENARIOS:
        spec = SCENARIOS[task]
        record = summary["scenarios"][task]
        if record["strand"] != spec["strand"] or record["class_names"] != list(spec["classes"]):
            raise ValueError("导出类别名称/方向不符合五分类任务：" + task)
        for kind, filename in (("features", "features.csv"), ("metadata", "features_meta.tsv")):
            info = summary["scenarios"][task][kind]
            path = source_dir / task / filename
            before = path.stat()
            if before.st_size != info["size_bytes"] or sha256(path) != info["sha256"]:
                raise ValueError("原始%s/%s大小或SHA与完成提取不同" % (task, filename))
            after = path.stat()
            if (before.st_size, before.st_mtime_ns) != (after.st_size, after.st_mtime_ns):
                raise ValueError("哈希核验期间源文件发生变化：" + str(path))
            identity["inputs"][task + "/" + kind] = {"path": str(path), "size_bytes": before.st_size,
                                                       "mtime_ns": before.st_mtime_ns, "sha256": info["sha256"]}
    return identity, summary, contract, plan, strict, fasta


def _quotas(source_dir, reference_dir, scenario, summary, plan):
    with (source_dir / "selected_counts.csv").open(encoding="utf-8-sig", newline="") as f:
        actual = list(csv.DictReader(f))
    with (reference_dir / "strict_plan_cells.csv").open(encoding="utf-8-sig", newline="") as f:
        frozen = list(csv.DictReader(f))
    if len(actual) != len(frozen):
        raise ValueError("严格配额单元数量不一致")
    quotas = {}
    for a, b in zip(actual, frozen):
        if {k: v for k, v in a.items() if k != "selected_windows"} != b or a["selected_windows"] != a["planned_windows"]:
            raise ValueError("实际提取单元/配额与冻结计划不同")
        if a["scenario"] == scenario:
            key = (a["split"], a["chrom"], a["strand"], a["context"], int(a["future_label"]))
            if key in quotas or int(a["selected_windows"]) < 1:
                raise ValueError("严格五分类配额有重复或空单元")
            quotas[key] = int(a["selected_windows"])
    groups = [g for g in plan["groups"] if g["scenario"] == scenario]
    if len(quotas) != len(groups) * 3 * 5:
        raise ValueError("严格五分类缺少组合/split/类别单元")
    for group in groups:
        for split, multiplier in (("train", 7), ("validation", 2), ("test", 1)):
            for label in range(5):
                key = (split, group["chrom"], group["strand"], group["context"], label)
                if quotas.get(key) != multiplier * group["k_exact_721"]:
                    raise ValueError("严格五分类未满足逐组合7k/2k/k配额")
    for split in SPLITS:
        if sum(v for key, v in quotas.items() if key[0] == split) != summary["scenarios"][scenario]["rows_by_split"][split]:
            raise ValueError("冻结配额与导出各split行数不一致")
    return quotas, groups


def _validate_row(row, meta, number, scenario, contract, lib, quotas):
    where = "%s数据行%d" % (scenario, number)
    spec = SCENARIOS[scenario]
    if len(row) != len(FEATURE_COLUMNS) or _integer(meta["feature_row"], where + " feature_row", 1) != number:
        raise ValueError(where + "：原11列格式或连续feature_row不正确")
    label = _integer(row[-1], where + " label")
    if label not in range(5) or _integer(meta["label"], where + " metadata.label") != label:
        raise ValueError(where + "：特征与metadata标签不一致或不是0–4五分类")
    original_label = spec["original_ncb_label"] if label == 4 else label
    if (_integer(meta["original_label"], where + " original_label") != original_label
            or meta["label_mapping"] != ("5->4" if original_label == 5 else "identity")
            or meta["scenario"] != scenario):
        raise ValueError(where + "：原始标签、Y5→4映射或scenario不一致")
    motif, original = row[0], meta["raw_oriented_6mer"]
    if not re.fullmatch(r"[ACGT]{3}N[ACGT]{2}", motif):
        raise ValueError(where + "：kmer必须为第4位N的大写6mer")
    if (len(original) != 6 or original[:3] + "N" + original[4:] != motif
            or meta["masked_6mer"] != motif or original[3] != spec["classes"][label]
            or meta["true_base"] != spec["classes"][label]):
        raise ValueError(where + "：原始6mer、mask、真值和标签冲突")
    batch, rid, split = meta["batch_id"], meta["read_id"], meta["split"]
    _integer(meta["source_line"], where + " source_line", 1)
    if batch != contract["batch_id"] or not rid or rid != rid.strip() or rid.lower() in {"unknown", "na", "none", "null", "nan"}:
        raise ValueError(where + "：batch或真实read_id无效")
    if split not in SPLITS or assigned_split(rid, batch, contract["split_seed"], contract["split_ratios"]) != split:
        raise ValueError(where + "：split与冻结真实read哈希划分不同")
    chrom, strand = meta["chrom"], meta["bam_strand"]
    key = (split, chrom, strand, motif, label)
    if key not in quotas:
        raise ValueError(where + "：参考/方向/context/类别不在冻结配额内")
    if chrom not in lib.sequences or strand != spec["strand"] or meta["tombo_strand"] != strand or meta["strand_match"] not in ("1", "true", "True"):
        raise ValueError(where + "：方向匹配或参考记录不一致")
    if meta["strict_match"] not in ("1", "true", "True"):
        raise ValueError(where + "：不属于同参考/方向/context的严格匹配窗口")
    pos = _integer(meta["target_pos"], where + " target_pos", 1)
    lo, hi = interval_for_target(pos, strand)
    if (_integer(meta["kmer_ref_start"], where + " kmer_ref_start", 1), _integer(meta["kmer_ref_end"], where + " kmer_ref_end", 1)) != (lo, hi):
        raise ValueError(where + "：窗口坐标与目标第4位不一致")
    row_start, row_end = _integer(meta["row_start"], where + " row_start", 1), _integer(meta["row_end"], where + " row_end", 1)
    if row_start > lo or row_end < hi or lib.get_window(chrom, pos, strand) != original:
        raise ValueError(where + "：窗口未被read覆盖或与原始X/Y参考不同")
    if label < 4:
        if any(max(lo, a) <= min(hi, b) for a, b in lib.exclude_intervals):
            raise ValueError(where + "：天然窗口与排除区间相交")
        if any(min(abs(p - j) for j in range(lo, hi + 1)) <= lib.ncb_guard for p in lib.nc_positions(chrom)):
            raise ValueError(where + "：天然窗口离NCB过近")
    elif pos not in lib.nc_positions(chrom) or row_start > contract["min_left_pos_ncb_only"]:
        raise ValueError(where + "：NCB窗口不符合冻结位点或覆盖规则")
    return key, rid


def _reuse(destination, identity):
    if destination.is_symlink():
        raise ValueError("准备目录不能是符号链接")
    marker = destination / "prepare_manifest.json"
    if not marker.is_file():
        raise ValueError("准备目录已有未知或未完成内容，拒绝覆盖：" + str(destination))
    previous = _json(marker)
    if previous.get("status") != "complete" or previous.get("identity") != identity:
        raise ValueError("已有准备结果的来源/源码身份不同，拒绝复用")
    for name, expected in previous["files"].items():
        if Path(name).name != name:
            raise ValueError("准备清单文件名不合法")
        path = destination / name
        if path.is_symlink() or not path.is_file() or path.stat().st_size != expected["size_bytes"] or sha256(path) != expected["sha256"]:
            raise ValueError("已准备文件被改动或缺失：" + name)
    return {"data_dir": str(destination), "audit_path": str(destination / "prepare_audit.json"),
            "manifest_path": str(marker), "reused": True, "scenario": identity["scenario"]}


def prepare_inputs(source_dir, work_dir, scenario, resume=False, disk_reserve_gib=4.0, reference_dir=None):
    if scenario not in SCENARIOS:
        raise ValueError("scenario必须为X_plus或Y_minus")
    source_dir, work_dir = Path(source_dir).expanduser().resolve(), Path(work_dir).expanduser().resolve()
    reference_dir = Path(reference_dir or HERE / "reference").resolve()
    if not math.isfinite(disk_reserve_gib) or disk_reserve_gib < 0:
        raise ValueError("磁盘预留量必须为有限非负数")
    destination, incomplete = work_dir / "prepared_data", work_dir / ".prepared_data.incomplete"
    if source_dir in (destination, incomplete) or source_dir.is_relative_to(destination) or source_dir.is_relative_to(incomplete):
        raise ValueError("源目录不能位于准备输出内部")
    work_dir.mkdir(parents=True, exist_ok=True)
    with (work_dir / ".prepare.lock").open("a+") as lock:
        try:
            fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError as exc:
            raise RuntimeError("此scenario已有准备进程运行") from exc
        _disk(work_dir, disk_reserve_gib)
        identity, summary, contract, plan, strict, fasta = _frozen(source_dir, reference_dir, scenario)
        quotas, groups = _quotas(source_dir, reference_dir, scenario, summary, plan)
        if destination.exists() or destination.is_symlink():
            return _reuse(destination, identity)
        if incomplete.exists() or incomplete.is_symlink():
            marker = incomplete / "prepare_identity.json"
            if incomplete.is_symlink() or not resume or not marker.is_file() or _json(marker) != identity:
                raise ValueError("已有未知/不同身份的准备staging；仅同一配置加--resume可恢复")
            shutil.rmtree(incomplete)
        incomplete.mkdir()
        _write_json(incomplete / "prepare_identity.json", identity)
        started = time.time()
        lib = ReferenceLibrary(fasta, ncb_guard=contract["ncb_guard"],
                               exclude_intervals=tuple(map(tuple, contract["exclude_intervals"])),
                               context_scope=contract["context_scope"])
        counts, strand_counts = Counter(), Counter()
        database = sqlite3.connect(str(incomplete / "read_membership.sqlite"))
        database.execute("PRAGMA journal_mode=DELETE"); database.execute("PRAGMA synchronous=FULL")
        database.execute("CREATE TABLE reads (read_id TEXT PRIMARY KEY, split TEXT NOT NULL, windows INTEGER NOT NULL, read_signature TEXT NOT NULL)")
        database.execute("CREATE TABLE cells (read_id TEXT NOT NULL, context TEXT NOT NULL, original_label INTEGER NOT NULL, PRIMARY KEY(read_id,context,original_label)) WITHOUT ROWID")
        n = 0
        feature_path, metadata_path = source_dir / scenario / "features.csv", source_dir / scenario / "features_meta.tsv"
        try:
            csv.field_size_limit(16 * 1024 ** 2)
            with ExitStack() as stack:
                source = stack.enter_context(feature_path.open(encoding="utf-8-sig", newline=""))
                metadata = stack.enter_context(metadata_path.open(encoding="utf-8-sig", newline=""))
                feature_reader = csv.reader(source)
                if tuple(next(feature_reader, ())) != FEATURE_COLUMNS:
                    raise ValueError("原特征CSV表头不是冻结11列格式")
                meta_reader = csv.DictReader(metadata, delimiter="\t")
                columns = list(meta_reader.fieldnames or [])
                if len(columns) != len(set(columns)) or not REQUIRED_META.issubset(columns):
                    raise ValueError("metadata表头缺必要字段或存在重复列")
                fields = columns + [name for name in EXTRA_META if name not in columns]
                writers, streams = {}, []
                for split in SPLITS:
                    fs = stack.enter_context((incomplete / (split + ".csv")).open("w", encoding="utf-8", newline=""))
                    ms = stack.enter_context((incomplete / (split + "_meta.tsv")).open("w", encoding="utf-8", newline=""))
                    fw, mw = csv.writer(fs, lineterminator="\n"), csv.DictWriter(ms, fields, delimiter="\t", lineterminator="\n")
                    fw.writerow(FEATURE_COLUMNS); mw.writeheader()
                    writers[split] = fw, mw; streams.extend((fs, ms))
                sentinel = object()
                for number, (row, meta) in enumerate(zip_longest(feature_reader, meta_reader, fillvalue=sentinel), 1):
                    if row is sentinel or meta is sentinel or None in meta or any(v is None for v in meta.values()):
                        raise ValueError("特征/metadata行数或TSV列数不对应，数据行%d" % number)
                    key, rid = _validate_row(row, meta, number, scenario, contract, lib, quotas)
                    split, chrom, strand, motif, label = key
                    signature = hashlib.sha256(json.dumps({name: meta[name] for name in
                        ("batch_id", "chrom", "bam_strand", "tombo_strand", "strand_match", "source_line", "row_start", "row_end")},
                        sort_keys=True, separators=(",", ":")).encode()).hexdigest()
                    previous = database.execute("SELECT split,read_signature FROM reads WHERE read_id=?", (rid,)).fetchone()
                    if previous and (previous[0] != split or previous[1] != signature):
                        raise ValueError("同一read的split、参考、方向或源记录冲突：" + rid)
                    database.execute("INSERT INTO reads VALUES (?,?,1,?) ON CONFLICT(read_id) DO UPDATE SET windows=windows+1", (rid, split, signature))
                    try:
                        database.execute("INSERT INTO cells VALUES (?,?,?)", (rid, motif, int(meta["original_label"])))
                    except sqlite3.IntegrityError as exc:
                        raise ValueError("同一read/context/original_class重复：" + rid) from exc
                    counts[key] += 1
                    if counts[key] > quotas[key]:
                        raise ValueError("实际窗口超过冻结单元配额")
                    strand_counts[(split, label, strand)] += 1
                    adapted = dict(meta)
                    adapted.update(kmer=motif, original_kmer=meta["raw_oriented_6mer"], source_dataset=contract["batch_id"],
                        original_read_id=rid, read_id_kind="sequencing_read_id", read_identity_basis="provided_row_aligned_extraction_metadata",
                        reference_identity_known="true", source_row=str(number), source_feature_row=str(number),
                        source_file=str(feature_path), run_id="unknown", template_group=chrom,
                        template_identity_basis="BAM reference assignment; shared natural backbone, not independent physical-template verification")
                    writers[split][0].writerow(row); writers[split][1].writerow(adapted); n = number
                    if number % 10000 == 0:
                        database.commit(); _disk(work_dir, disk_reserve_gib)
                        print("[PREPARE] %s 已核验并对齐%s/%s行" % (scenario, f"{number:,}", f"{summary['scenarios'][scenario]['rows']:,}"), flush=True)
                if n != summary["scenarios"][scenario]["rows"] or dict(counts) != quotas:
                    raise ValueError("实际总行数或逐参考/方向/context/类/split数量与冻结配额不同")
                for stream in streams:
                    stream.flush(); os.fsync(stream.fileno())
            database.commit()
            read_stats = {}
            for split in SPLITS:
                values = database.execute("SELECT COUNT(*),SUM(windows),SUM(windows>1),MAX(windows) FROM reads WHERE split=?", (split,)).fetchone()
                read_stats[split] = dict(zip(("unique_real_reads", "windows", "reads_with_multiple_windows", "max_windows_per_read"), values))
                reported = _json(source_dir / "read_audit.json")["scenarios"][scenario][split]
                if read_stats[split]["unique_real_reads"] != reported["unique_real_reads"] or read_stats[split]["windows"] != reported["windows"]:
                    raise ValueError("逐read重算统计与服务器提取审计不同")
            with (incomplete / "read_membership.tsv").open("w", newline="", encoding="utf-8") as f:
                writer = csv.writer(f, delimiter="\t", lineterminator="\n")
                writer.writerow(("read_id", "split", "windows")); writer.writerows(database.execute("SELECT read_id,split,windows FROM reads ORDER BY split,read_id"))
                f.flush(); os.fsync(f.fileno())
        finally:
            database.close()
        for info in identity["inputs"].values():
            stat = Path(info["path"]).stat()
            if (stat.st_size, stat.st_mtime_ns) != (info["size_bytes"], info["mtime_ns"]):
                raise ValueError("准备期间源文件发生变化，拒绝发布")
        with (incomplete / "selected_counts.csv").open("w", newline="", encoding="utf-8") as f:
            writer = csv.writer(f, lineterminator="\n")
            writer.writerow(("split", "chrom", "strand", "context", "label", "count"))
            writer.writerows((*key, value) for key, value in sorted(counts.items()))
            f.flush(); os.fsync(f.fileno())
        audit = {"version": VERSION, "status": "PASS", "scenario": scenario,
            "class_names": list(SCENARIOS[scenario]["classes"]), "rows": n, "contexts": len({g["context"] for g in groups}),
            "matched_groups": len(groups), "rows_by_split": {s: sum(v for k, v in counts.items() if k[0] == s) for s in SPLITS},
            "class_counts_by_split": {s: {str(c): sum(v for k, v in counts.items() if k[0] == s and k[-1] == c) for c in range(5)} for s in SPLITS},
            "read_statistics": read_stats, "read_overlap_counts": {"train__validation": 0, "train__test": 0, "validation__test": 0},
            "read_cell_duplicates": 0, "all_original_read_hash_assignments_verified": True,
            "sampling_policy": "Frozen extraction split and strict quotas retained; no balancing or new random splitting",
            "all_class_sources": contract["batch_id"], "original_real_read_ids_present": True,
            "physical_run_id": "unknown; batch_id is not substituted for a physical sequencing-run ID",
            "strand_counts": [{"split": s, "label": c, "strand": st, "windows": value} for (s, c, st), value in sorted(strand_counts.items())],
            "reference_sha256": contract["reference_sha256"], "started_unix": started, "completed_unix": time.time(),
            "limits": ["Same-source, same BAM-assigned reference/context/direction; natural and NCB positions differ.",
                       "Read holdout, not template/batch holdout; no within-direction X-versus-Y comparison.",
                       "Multiple windows of each real read remain together; read-cluster uncertainty is required.",
                       "Frozen engine independently validates all finite scalar and raw-signal values."]}
        _write_json(incomplete / "prepare_audit.json", audit)
        files = {p.name: {"sha256": sha256(p), "size_bytes": p.stat().st_size} for p in incomplete.iterdir()
                 if p.is_file() and p.name != "prepare_manifest.json"}
        _write_json(incomplete / "prepare_manifest.json", {"status": "complete", "identity": identity, "files": files})
        _disk(work_dir, disk_reserve_gib)
        os.replace(incomplete, destination)
        print("[PREPARE PASS] %s 原split/%s行已发布：%s" % (scenario, f"{n:,}", destination), flush=True)
        return {"data_dir": str(destination), "audit_path": str(destination / "prepare_audit.json"),
                "manifest_path": str(destination / "prepare_manifest.json"), "reused": False, "scenario": scenario}


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--source_dir", default="cplx_all_direction_matched_5class_v1_0_0")
    p.add_argument("--work_dir", required=True)
    p.add_argument("--scenario", required=True, choices=tuple(SCENARIOS))
    p.add_argument("--resume", action="store_true")
    p.add_argument("--disk_reserve_gib", type=float, default=4.0)
    print(json.dumps(prepare_inputs(**vars(p.parse_args())), ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
