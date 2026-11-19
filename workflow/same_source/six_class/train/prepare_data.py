#!/usr/bin/env python3
"""把已经完成的 CPLX 同来源导出按冻结 split 写为训练引擎输入。

不重新平衡、不重新划分、不修改源文件。生产入口固定核验随包 reference
中的导出凭据；reference_dir 仅供自动化测试构造独立的小型凭据。
"""
from __future__ import annotations

import argparse
from collections import Counter, defaultdict
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

from reference6 import ReferenceLibrary, LABELS, interval_for_target

HERE = Path(__file__).resolve().parent
VERSION = "1.0.0"
SPLITS = ("train", "validation", "test")
CLASSES = ("A", "T", "C", "G", "X", "Y")
FEATURE_COLUMNS = ("kmer", "mean", "std", "median", "dwell", "quality",
                   "mismatch", "insertion", "deletion", "signal", "label")
REPORTS = ("extraction_summary.json", "input_contract.json", "sampling_plan.json",
           "selected_counts.csv", "reference_context_scope.json", "audit_summary.json")
REQUIRED_META = {"feature_row", "batch_id", "read_id", "chrom", "bam_strand",
                 "tombo_strand", "strand_match", "split", "source_line", "row_start",
                 "row_end", "target_pos", "kmer_ref_start", "kmer_ref_end",
                 "raw_oriented_6mer", "masked_6mer", "true_base", "label", "strict_match"}
EXTRA_META = ("kmer", "original_kmer", "source_dataset", "original_read_id",
              "read_id_kind", "read_identity_basis", "reference_identity_known",
              "source_row", "source_feature_row", "source_file", "run_id",
              "template_group", "template_identity_basis")
GIB = 1024 ** 3


def sha256(path):
    value = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for chunk in iter(lambda: stream.read(4 * 1024 ** 2), b""):
            value.update(chunk)
    return value.hexdigest()


def _json(path):
    return json.loads(Path(path).read_text(encoding="utf-8"))


def _write_json(path, value):
    path = Path(path)
    temporary = path.with_name(path.name + ".writing")
    with temporary.open("w", encoding="utf-8") as stream:
        json.dump(value, stream, ensure_ascii=False, indent=2, allow_nan=False)
        stream.write("\n")
        stream.flush()
        os.fsync(stream.fileno())
    os.replace(temporary, path)


def _integer(text, name, minimum=0):
    if not re.fullmatch(r"[0-9]+", str(text)):
        raise ValueError("%s必须为非负十进制整数，得到%r" % (name, text))
    value = int(text)
    if value < minimum:
        raise ValueError("%s小于%d" % (name, minimum))
    return value


def assigned_split(read_id, batch_id, seed, ratios):
    """与冻结提取器完全相同的 SHA256 read 划分，不能改用窗口随机划分。"""
    digest = hashlib.sha256("\0".join(str(x) for x in (seed, batch_id, read_id)).encode()).digest()
    u = int.from_bytes(digest[:8], "big") / 2 ** 64
    total = sum(ratios)
    return "train" if u < ratios[0] / total else "validation" if u < sum(ratios[:2]) / total else "test"


def _disk(path, reserve):
    free = shutil.disk_usage(path).free
    if free < reserve * GIB:
        raise OSError(28, "准备目录剩余空间不足预留 %.2f GiB；清理空间后使用 --resume" % reserve)


def _frozen(source_dir, reference_dir):
    identity = {"version": VERSION, "adapter_sha256": sha256(__file__),
                "reference_helper_sha256": sha256(HERE / "reference6.py"),
                "source_dir": str(source_dir), "reports": {}}
    for name in REPORTS:
        expected = reference_dir / name
        actual = source_dir / name
        if not expected.is_file() or not actual.is_file():
            raise FileNotFoundError("缺少冻结导出报告：" + str(actual))
        a, b = sha256(actual), sha256(expected)
        if a != b:
            raise ValueError("导出报告与本包冻结参考不一致：" + name)
        identity["reports"][name] = a
    summary, contract, plan = (_json(source_dir / n) for n in REPORTS[:3])
    if summary.get("status") != "complete" or plan.get("status") != "ready":
        raise ValueError("提取任务尚未完成或采样计划未ready")
    if summary["batch_id"] != contract["batch_id"]:
        raise ValueError("导出与输入凭据的batch_id不一致")
    if summary["rows"] != plan["total_rows"] or summary["rows_by_split"] != plan["rows_by_split"]:
        raise ValueError("导出行数与采样计划不一致")
    if summary["contexts_kept"] != plan["contexts_kept"]:
        raise ValueError("导出context数量与计划不一致")
    identity["inputs"] = {}
    for name, filename, offset_key in (("features", "features.csv", "feature_offset"),
                                        ("metadata", "features_meta.tsv", "meta_offset")):
        path = source_dir / filename
        before = path.stat()
        if before.st_size != summary[offset_key] or sha256(path) != summary[name]["sha256"]:
            raise ValueError("原始%s的大小或SHA256与已完成提取不一致" % filename)
        after = path.stat()
        if (before.st_size, before.st_mtime_ns) != (after.st_size, after.st_mtime_ns):
            raise ValueError("哈希核验期间源文件变化：" + filename)
        identity["inputs"][name] = {"path": str(path), "sha256": summary[name]["sha256"],
                                     "size_bytes": before.st_size, "mtime_ns": before.st_mtime_ns}
    fasta = reference_dir / "CPLX_XNA_all_original.fasta"
    if sha256(fasta) != contract["reference_sha256"]:
        raise ValueError("原始X/Y参考FASTA与冻结导出凭据不一致")
    identity["reference_sha256"] = contract["reference_sha256"]
    return identity, summary, contract, plan, fasta


def _validate_row(row, meta, number, contract, lib, quotas):
    where = "数据行 %d" % number
    if len(row) != len(FEATURE_COLUMNS):
        raise ValueError(where + "：特征CSV必须是原始11列")
    if _integer(meta["feature_row"], where + " feature_row", 1) != number:
        raise ValueError(where + "：feature_row不是连续的原始行序号，可能错行")
    label = _integer(row[-1], where + " label")
    if label not in range(6) or _integer(meta["label"], where + " metadata.label") != label:
        raise ValueError(where + "：特征与metadata label不一致或不是六分类")
    motif, original = row[0], meta["raw_oriented_6mer"]
    if not re.fullmatch(r"[ACGT]{3}N[ACGT]{2}", motif):
        raise ValueError(where + "：kmer必须为第4位N的大写6mer")
    if (meta["masked_6mer"] != motif or len(original) != 6
            or original[:3] + "N" + original[4:] != motif
            or original[3] != CLASSES[label] or meta["true_base"] != CLASSES[label]):
        raise ValueError(where + "：原始6mer、mask、真值或标签发生冲突")
    batch, rid, split = meta["batch_id"], meta["read_id"], meta["split"]
    if batch != contract["batch_id"] or not rid or rid != rid.strip() or rid.lower() in {"unknown", "na", "none", "null", "nan"}:
        raise ValueError(where + "：batch或真实read_id无效")
    if split not in SPLITS or assigned_split(rid, batch, contract["split_seed"], contract["split_ratios"]) != split:
        raise ValueError(where + "：split与冻结read哈希划分不一致")
    key = (split, motif, label)
    if key not in quotas:
        raise ValueError(where + "：此split/context/label不在冻结采样计划内")
    chrom, strand = meta["chrom"], meta["bam_strand"]
    if chrom not in lib.sequences or strand not in ("+", "-"):
        raise ValueError(where + "：未知参考或方向")
    if meta["tombo_strand"] != strand or meta["strand_match"] not in ("1", "True", "true"):
        raise ValueError(where + "：方向匹配信息不一致")
    pos = _integer(meta["target_pos"], where + " target_pos", 1)
    lo, hi = interval_for_target(pos, strand)
    if (_integer(meta["kmer_ref_start"], where + " kmer_ref_start", 1),
            _integer(meta["kmer_ref_end"], where + " kmer_ref_end", 1)) != (lo, hi):
        raise ValueError(where + "：窗口坐标与目标第4位不一致")
    row_start = _integer(meta["row_start"], where + " row_start", 1)
    row_end = _integer(meta["row_end"], where + " row_end", 1)
    if row_start > lo or row_end < hi or lib.get_window(chrom, pos, strand) != original:
        raise ValueError(where + "：窗口未被read覆盖或与原始X/Y参考不一致")
    if label < 4:
        if any(max(lo, a) <= min(hi, b) for a, b in lib.exclude_intervals):
            raise ValueError(where + "：天然窗口与排除区间相交")
        if any(min(abs(p - j) for j in range(lo, hi + 1)) <= lib.ncb_guard for p in lib.nc_positions(chrom)):
            raise ValueError(where + "：天然窗口离NCB过近")
    else:
        if pos not in lib.nc_positions(chrom) or row_start > contract["min_left_pos_ncb_only"]:
            raise ValueError(where + "：NCB窗口不符合已冻结位点/覆盖规则")
    return key, rid


def _reuse(destination, identity):
    marker = destination / "prepare_manifest.json"
    if not marker.is_file():
        raise ValueError("准备目录已有不完整/未知内容，拒绝覆盖：" + str(destination))
    previous = _json(marker)
    if previous.get("status") != "complete" or previous.get("identity") != identity:
        raise ValueError("已有准备结果的来源/源码身份不同，不能静默复用或覆盖")
    for name, expected in previous["files"].items():
        path = destination / name
        if not path.is_file() or path.stat().st_size != expected["size_bytes"] or sha256(path) != expected["sha256"]:
            raise ValueError("已准备文件被改动或缺失：" + name)
    return {"data_dir": str(destination), "audit_path": str(destination / "prepare_audit.json"),
            "manifest_path": str(marker), "reused": True}


def prepare_inputs(source_dir, work_dir, resume=False, disk_reserve_gib=2.0, reference_dir=None):
    """核验源数据并按原metadata.split发布三份引擎输入；不使用模型分数。"""
    source_dir = Path(source_dir).expanduser().resolve()
    work_dir = Path(work_dir).expanduser().resolve()
    reference_dir = Path(reference_dir or HERE / "reference").resolve()
    if not math.isfinite(disk_reserve_gib) or disk_reserve_gib < 0:
        raise ValueError("磁盘预留量必须为有限非负数")
    destination = work_dir / "prepared_data"
    incomplete = work_dir / ".prepared_data.incomplete"
    if source_dir in (destination, incomplete) or source_dir.is_relative_to(destination) or source_dir.is_relative_to(incomplete):
        raise ValueError("源目录不能位于准备输出内部")
    work_dir.mkdir(parents=True, exist_ok=True)
    with (work_dir / ".prepare.lock").open("a+") as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        _disk(work_dir, disk_reserve_gib)
        identity, summary, contract, plan, fasta = _frozen(source_dir, reference_dir)
        if destination.exists():
            return _reuse(destination, identity)
        if incomplete.exists():
            old = incomplete / "prepare_identity.json"
            if not resume or not old.is_file() or _json(old) != identity:
                raise ValueError("发现未完成准备任务；同一配置加 --resume 继续，不同配置用新目录")
            shutil.rmtree(incomplete)
        incomplete.mkdir()
        _write_json(incomplete / "prepare_identity.json", identity)
        started = time.time()
        quotas = {}
        for raw_key, value in plan["quota"].items():
            split, motif, label = raw_key.split("|")
            key = (split, motif, int(label))
            if split not in SPLITS or key in quotas or not isinstance(value, int) or value < 1:
                raise ValueError("冻结配额格式不合法")
            quotas[key] = value
        with (source_dir / "selected_counts.csv").open(encoding="utf-8-sig", newline="") as stream:
            selected = {}
            for row in csv.DictReader(stream):
                key = (row["split"], row["context"], int(row["label"]))
                if key in selected:
                    raise ValueError("selected_counts包含重复单元")
                selected[key] = int(row["count"])
        if selected != quotas:
            raise ValueError("已提取计数与冻结配额不一致")
        contexts = {motif for _, motif, _ in quotas}
        if len(contexts) != plan["contexts_kept"] or len(quotas) != len(contexts) * len(SPLITS) * 6:
            raise ValueError("冻结计划缺少context/split/类别单元")
        for split in SPLITS:
            for motif in contexts:
                if len({quotas.get((split, motif, c)) for c in range(6)}) != 1:
                    raise ValueError("冻结计划的六类配额不均衡")
        lib = ReferenceLibrary(fasta, ncb_guard=contract["ncb_guard"],
                               exclude_intervals=tuple(map(tuple, contract["exclude_intervals"])),
                               context_scope=contract["context_scope"])
        counts, strand_counts = Counter(), Counter()
        database = sqlite3.connect(str(incomplete / "read_membership.sqlite"))
        database.execute("PRAGMA journal_mode=DELETE")
        database.execute("PRAGMA synchronous=FULL")
        database.execute("CREATE TABLE reads (read_id TEXT PRIMARY KEY, split TEXT NOT NULL, windows INTEGER NOT NULL, read_signature TEXT NOT NULL)")
        database.execute("CREATE TABLE cells (read_id TEXT NOT NULL, context TEXT NOT NULL, label INTEGER NOT NULL, PRIMARY KEY(read_id,context,label)) WITHOUT ROWID")
        n = 0
        try:
            csv.field_size_limit(16 * 1024 ** 2)
            with ExitStack() as stack:
                source = stack.enter_context((source_dir / "features.csv").open(encoding="utf-8-sig", newline=""))
                metadata = stack.enter_context((source_dir / "features_meta.tsv").open(encoding="utf-8-sig", newline=""))
                feature_reader = csv.reader(source)
                if tuple(next(feature_reader, ())) != FEATURE_COLUMNS:
                    raise ValueError("原始特征CSV表头不是冻结的11列格式")
                meta_reader = csv.DictReader(metadata, delimiter="\t")
                columns = list(meta_reader.fieldnames or [])
                if len(columns) != len(set(columns)) or not REQUIRED_META.issubset(columns):
                    raise ValueError("metadata表头缺少必要字段或存在重复字段")
                fields = columns + [name for name in EXTRA_META if name not in columns]
                writers, streams = {}, []
                for split in SPLITS:
                    fs = stack.enter_context((incomplete / (split + ".csv")).open("w", encoding="utf-8", newline=""))
                    ms = stack.enter_context((incomplete / (split + "_meta.tsv")).open("w", encoding="utf-8", newline=""))
                    fw, mw = csv.writer(fs, lineterminator="\n"), csv.DictWriter(ms, fields, delimiter="\t", lineterminator="\n")
                    fw.writerow(FEATURE_COLUMNS); mw.writeheader()
                    writers[split] = fw, mw
                    streams.extend((fs, ms))
                sentinel = object()
                for number, pair in enumerate(zip_longest(feature_reader, meta_reader, fillvalue=sentinel), 1):
                    row, meta = pair
                    if row is sentinel or meta is sentinel or None in meta or any(v is None for v in meta.values()):
                        raise ValueError("特征/metadata行数不同或TSV列数错误，数据行%d" % number)
                    key, rid = _validate_row(row, meta, number, contract, lib, quotas)
                    split, motif, label = key
                    read_signature = hashlib.sha256(json.dumps(
                        {name: meta[name] for name in ("batch_id", "chrom", "bam_strand", "tombo_strand", "strand_match", "source_line", "row_start", "row_end")},
                        sort_keys=True, separators=(",", ":")).encode()).hexdigest()
                    previous = database.execute("SELECT split,read_signature FROM reads WHERE read_id=?", (rid,)).fetchone()
                    if previous and previous[0] != split:
                        raise ValueError("同一真实read跨越两个split：" + rid)
                    if previous and previous[1] != read_signature:
                        raise ValueError("同一真实read对应了冲突的参考、方向或原始记录：" + rid)
                    database.execute("INSERT INTO reads VALUES (?,?,1,?) ON CONFLICT(read_id) DO UPDATE SET windows=windows+1", (rid, split, read_signature))
                    try:
                        database.execute("INSERT INTO cells VALUES (?,?,?)", (rid, motif, label))
                    except sqlite3.IntegrityError as exc:
                        raise ValueError("同一read/context/class重复出现：" + rid) from exc
                    counts[key] += 1
                    if counts[key] > quotas[key]:
                        raise ValueError("实际提取单元数量超过冻结配额")
                    strand_counts[(split, label, meta["bam_strand"])] += 1
                    adapted = dict(meta)
                    adapted.update(kmer=motif, original_kmer=meta["raw_oriented_6mer"],
                                   source_dataset=contract["batch_id"], original_read_id=rid,
                                   read_id_kind="sequencing_read_id", read_identity_basis="provided_row_aligned_extraction_metadata",
                                   reference_identity_known="true", source_row=str(number), source_feature_row=str(number),
                                   source_file=str(source_dir / "features.csv"), run_id="unknown", template_group=meta["chrom"],
                                   template_identity_basis="BAM reference assignment; shared natural backbone, not an independent experimental replicate")
                    writers[split][0].writerow(row)
                    writers[split][1].writerow(adapted)
                    n = number
                    if number % 10000 == 0:
                        database.commit(); _disk(work_dir, disk_reserve_gib)
                        print("[PREPARE] 已对齐并核验 %s/%s 行" % (f"{number:,}", f"{summary['rows']:,}"), flush=True)
                if n != summary["rows"] or dict(counts) != quotas:
                    raise ValueError("总行数或逐context/class/split实际配额与冻结计划不一致")
                for stream in streams:
                    stream.flush(); os.fsync(stream.fileno())
            database.commit()
            read_stats = {}
            for split in SPLITS:
                a = database.execute("SELECT COUNT(*),SUM(windows),SUM(windows>1),MAX(windows) FROM reads WHERE split=?", (split,)).fetchone()
                read_stats[split] = dict(zip(("unique_real_reads", "windows", "reads_with_multiple_windows", "max_windows_per_read"), a))
            with (incomplete / "read_membership.tsv").open("w", newline="", encoding="utf-8") as stream:
                writer = csv.writer(stream, delimiter="\t", lineterminator="\n")
                writer.writerow(("read_id", "split", "windows"))
                writer.writerows(database.execute("SELECT read_id,split,windows FROM reads ORDER BY split,read_id"))
                stream.flush(); os.fsync(stream.fileno())
        finally:
            database.close()
        for value in identity["inputs"].values():
            stat = Path(value["path"]).stat()
            if (stat.st_size, stat.st_mtime_ns) != (value["size_bytes"], value["mtime_ns"]):
                raise ValueError("准备期间源文件变化，拒绝发布")
        with (incomplete / "selected_counts.csv").open("w", newline="", encoding="utf-8") as stream:
            writer = csv.writer(stream, lineterminator="\n")
            writer.writerow(("split", "context", "label", "count"))
            writer.writerows((*key, value) for key, value in sorted(counts.items()))
        audit = {"version": VERSION, "status": "PASS", "rows": n, "contexts": plan["contexts_kept"],
                 "rows_by_split": {s: sum(v for (a, _, _), v in counts.items() if a == s) for s in SPLITS},
                 "class_counts_by_split": {s: {str(c): sum(v for (a, _, b), v in counts.items() if a == s and b == c) for c in range(6)} for s in SPLITS},
                 "read_statistics": read_stats, "read_overlap_counts": {"train__validation": 0, "train__test": 0, "validation__test": 0},
                 "read_cell_duplicates": 0, "all_original_read_hash_assignments_verified": True,
                 "sampling_policy": "Original extraction split and quotas retained; no new balancing or random splitting",
                 "all_six_class_sources": contract["batch_id"], "original_real_read_ids_present": True,
                 "physical_run_id": "unknown; experimental batch_id is not substituted for a physical sequencing-run ID",
                 "strand_counts": [{"split": s, "label": c, "strand": st, "windows": value} for (s, c, st), value in sorted(strand_counts.items())],
                 "reference_sha256": identity["reference_sha256"], "started_unix": started, "completed_unix": time.time(),
                 "limits": ["Same-source read-held-out evaluation, not batch/template-held-out validation.",
                            "X/Y remain associated with opposite directions; natural and NCB sites differ in position.",
                            "Natural windows share one common backbone; reference assignment is inherited from BAM.",
                            "Scalar and raw-signal finite float32 validation is repeated by the frozen training engine."]}
        _write_json(incomplete / "prepare_audit.json", audit)
        files = {p.name: {"sha256": sha256(p), "size_bytes": p.stat().st_size}
                 for p in incomplete.iterdir() if p.is_file() and p.name != "prepare_manifest.json"}
        _write_json(incomplete / "prepare_manifest.json", {"status": "complete", "identity": identity, "files": files})
        os.replace(incomplete, destination)
        print("[PREPARE PASS] 保留原split，已准备 %s 行：%s" % (f"{n:,}", destination), flush=True)
        return {"data_dir": str(destination), "audit_path": str(destination / "prepare_audit.json"),
                "manifest_path": str(destination / "prepare_manifest.json"), "reused": False}


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--source_dir", default="cplx_all_same_source_6class_v1_0_0")
    p.add_argument("--work_dir", required=True)
    p.add_argument("--resume", action="store_true")
    p.add_argument("--disk_reserve_gib", type=float, default=2.0)
    args = p.parse_args()
    print(json.dumps(prepare_inputs(**vars(args)), ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
