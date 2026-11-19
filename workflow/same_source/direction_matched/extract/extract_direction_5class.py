#!/usr/bin/env python3
"""一次扫描既有信号TSV，按冻结严格配额导出两个同方向五分类数据集。"""
from __future__ import annotations

import argparse
from collections import Counter
from contextlib import ExitStack
import csv
from datetime import datetime, timezone
import fcntl
import hashlib
import json
import math
import os
from pathlib import Path
import platform
import shutil
import signal
import sqlite3
import sys
import tarfile
import time
from types import SimpleNamespace

import numpy as np

import legacy_extract as legacy
from feature6 import FEATURE_HEADER
from reference6 import ReferenceLibrary

VERSION = "1.0.0"
HERE = Path(__file__).resolve().parent
DEFAULT_INPUT = "fast5_single_guppy_pass_signal_XY_all_notxt_fast.tsv"
DEFAULT_OUTPUT = "cplx_all_direction_matched_5class_v1_0_0"
CORE_NAME = "cplx_direction_5class_extract_core_v1_0_0.tar.gz"
SPLITS = ("train", "validation", "test")
SCENARIOS = {"X_plus": {"strand": "+", "labels": (0, 1, 2, 3, 4)},
             "Y_minus": {"strand": "-", "labels": (0, 1, 2, 3, 5)}}
META_HEADER = list(legacy.META_HEADER) + ["original_label", "scenario", "label_mapping"]
CELL_FIELDS = ["scenario", "split", "chrom", "strand", "context", "original_label",
               "future_label", "true_base", "available_raw_candidates", "planned_windows"]
PINS = {
    "input_contract.json": "83e13ea633569a861ec5bb13ad1fcca4b1edffebb35e0cee4b6ecf8b5370b42a",
    "audit_summary.json": "c25b52d5bfa6d0f4c90f986094682fcac419f81ba1f2cdef1c5982a840274cbd",
    "strict_candidate_counts.csv": "25e6b092f230f69635b1236811eeec215a8eb1dd759a10b78e1b58442fd4b84b",
    "strict_plan.json": "4f2df41f81444567e3acfa76f630746e5752fc19dc2255a582dac80072cd66b8",
    "strict_plan_cells.csv": "449de3afb80df79e942eeb11964392373d18652fd5d80d95eb6f425a82e590bc",
    "CPLX_XNA_all_original.fasta": "da8157b20963794e9a7986bad9d92d2a3654bfd4eb912e730e9e8c3b832e7e7b",
}
HELPER_PINS = {
    "legacy_extract.py": "2a09c903993e067e48a4d1660aecafa1d7b30ca1243ede5ea9134b246af72a8c",
    "reference6.py": "32b762ebdd567137750c31621d5af1d68005134a55a01e5748c5e4f253470b73",
    "feature6.py": "3246aec7cd891f9ff77dec22e063713a29ae0f38049cba6384004528350d4890",
}
STOP = False


def file_sha(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for chunk in iter(lambda: stream.read(4 * 1024 ** 2), b""):
            digest.update(chunk)
    return digest.hexdigest()


def read_json(path):
    return json.loads(Path(path).read_text(encoding="utf-8-sig"))


def atomic_json(path, data):
    legacy.atomic_json(Path(path), data)


def now():
    return datetime.now(timezone.utc).isoformat()


def handle_stop(signum, frame):
    global STOP
    STOP = True
    print("[STOP] 当前read结束后保存检查点；随后用同一命令--resume继续。", flush=True)


def cell_key(scenario, split, chrom, strand, context, label):
    return "|".join(map(str, (scenario, split, chrom, strand, context, label)))


def file_fingerprint(path):
    return legacy.input_fingerprint(path)


def load_reference(reference_dir=None):
    """Explicit reference_dir is an API-only fixture injection; CLI always pins production inputs."""
    folder = Path(reference_dir).resolve() if reference_dir is not None else HERE / "reference"
    for name in PINS:
        if not (folder / name).is_file():
            raise FileNotFoundError("缺少冻结参考文件：" + str(folder / name))
        if reference_dir is None and file_sha(folder / name) != PINS[name]:
            raise ValueError("冻结参考文件的SHA不符：" + name)
    for name, expected in HELPER_PINS.items():
        if file_sha(HERE / name) != expected:
            raise ValueError("特征数值助手代码改变，拒绝混用：" + name)
    contract = read_json(folder / "input_contract.json")
    audit = read_json(folder / "audit_summary.json")
    plan = read_json(folder / "strict_plan.json")
    expected_numpy = contract.get("runtime", {}).get("numpy")
    if not expected_numpy or np.__version__ != expected_numpy:
        raise RuntimeError("本任务必须保持原特征提取的NumPy版本 %s，当前为 %s。请使用原特征提取环境（通常conda activate base），不要改归一化定义或自动安装包。" % (expected_numpy, np.__version__))
    if audit.get("status") != "complete" or len(audit.get("chain_sha256", "")) != 64:
        raise ValueError("原始信号审计不完整。")
    if tuple(contract.get("split_ratios", ())) != (7, 2, 1):
        raise ValueError("read划分比例不是已固定的7:2:1。")
    fasta = folder / "CPLX_XNA_all_original.fasta"
    if file_sha(fasta) != contract["reference_sha256"]:
        raise ValueError("原始X/Y参考与原信号审计不一致。")
    lib = ReferenceLibrary(fasta, ncb_guard=contract["ncb_guard"],
                           exclude_intervals=tuple(tuple(x) for x in contract["exclude_intervals"]),
                           context_scope=contract["context_scope"])
    args = SimpleNamespace(require_strand_match=contract["require_strand_match"],
        reference_placeholder=contract["reference_placeholder"], min_left_pos=contract["min_left_pos_ncb_only"],
        selection_seed=contract["selection_seed"], batch_id=contract["batch_id"])
    raw_counts = {}
    with (folder / "strict_candidate_counts.csv").open(encoding="utf-8-sig", newline="") as stream:
        for row in csv.DictReader(stream):
            key = (row["split"], row["chrom"], row["strand"], row["context"], int(row["label"]))
            if key in raw_counts:
                raise ValueError("原候选统计包含重复单元。")
            raw_counts[key] = int(row["valid_read_cell_windows"])
    with (folder / "strict_plan_cells.csv").open(encoding="utf-8-sig", newline="") as stream:
        reader = csv.DictReader(stream)
        if reader.fieldnames != CELL_FIELDS:
            raise ValueError("冻结单元配额表表头不正确。")
        cells = list(reader)
    group_map = {}
    for group in plan["groups"]:
        key = (group["scenario"], group["chrom"], group["strand"], group["context"])
        if key in group_map or group["scenario"] not in SCENARIOS or group["k_exact_721"] < 5:
            raise ValueError("冻结严格分组无效。")
        group_map[key] = group
    expected_keys, seen = set(), set()
    for group_key, group in group_map.items():
        scenario, chrom, strand, context = group_key
        if strand != SCENARIOS[scenario]["strand"] or chrom not in lib.sequences:
            raise ValueError("冻结参考或方向无效。")
        available = [min(raw_counts.get((split, chrom, strand, context, label), 0)
                         for label in SCENARIOS[scenario]["labels"]) for split in SPLITS]
        k = min(available[0] // 7, available[1] // 2, available[2])
        if k != group["k_exact_721"]:
            raise ValueError("冻结配额不符合原候选计数。")
        for split in SPLITS:
            for label in SCENARIOS[scenario]["labels"]:
                expected_keys.add(cell_key(scenario, split, chrom, strand, context, label))
    for row in cells:
        for name in ("original_label", "future_label", "available_raw_candidates", "planned_windows"):
            row[name] = int(row[name])
        key = cell_key(row["scenario"], row["split"], row["chrom"], row["strand"], row["context"], row["original_label"])
        if key not in expected_keys or key in seen:
            raise ValueError("冻结配额有重复或未知单元。")
        seen.add(key)
        group = group_map[(row["scenario"], row["chrom"], row["strand"], row["context"])]
        ratio = (7, 2, 1)[SPLITS.index(row["split"])]
        available = raw_counts[(row["split"], row["chrom"], row["strand"], row["context"], row["original_label"])]
        if (row["planned_windows"] != ratio * group["k_exact_721"]
                or row["available_raw_candidates"] != available or available < row["planned_windows"]
                or row["future_label"] != (4 if row["original_label"] == 5 else row["original_label"])
                or row["true_base"] != "ATCGXY"[row["original_label"]]):
            raise ValueError("冻结单元的数量或标签映射错误。")
    if seen != expected_keys or not all(any(r["scenario"] == s for r in cells) for s in SCENARIOS):
        raise ValueError("两个五分类任务的冻结配额不完整。")
    return contract, audit, plan, cells, lib, args


class RestrictedLibrary:
    """Only whole reference/strand/context cells are filtered; position ranking stays frozen."""
    def __init__(self, original, plan):
        self.original = original
        self.allowed = {(g["chrom"], g["strand"], g["context"]) for g in plan["groups"]}
        self.by_pair = {(chrom, strand) for chrom, strand, _ in self.allowed}
        self.cache = {}

    def __getattr__(self, name):
        return getattr(self.original, name)

    def candidate_sites(self, chrom, strand):
        pair = (chrom, strand)
        if pair not in self.cache:
            self.cache[pair] = tuple(site for site in self.original.candidate_sites(chrom, strand)
                if site["strict_match"] and (chrom, strand, site["masked_6mer"]) in self.allowed)
        return self.cache[pair]


def connect(path):
    db = sqlite3.connect(str(path))
    db.execute("PRAGMA journal_mode=WAL")
    db.execute("PRAGMA synchronous=FULL")
    db.execute("CREATE TABLE IF NOT EXISTS state (name TEXT PRIMARY KEY, payload TEXT NOT NULL)")
    db.execute("CREATE TABLE IF NOT EXISTS reads (read_id TEXT PRIMARY KEY, line INTEGER NOT NULL, digest TEXT NOT NULL, split TEXT NOT NULL, chrom TEXT, strand TEXT, accepted INTEGER NOT NULL)")
    db.execute("CREATE TABLE IF NOT EXISTS selected (scenario TEXT, read_id TEXT, context TEXT, original_label INTEGER, final_label INTEGER, split TEXT, chrom TEXT, strand TEXT, source_line INTEGER, feature_row INTEGER, target_pos INTEGER, PRIMARY KEY(scenario,read_id,context,original_label))")
    db.commit()
    return db


def db_get(db, name):
    row = db.execute("SELECT payload FROM state WHERE name=?", (name,)).fetchone()
    return json.loads(row[0]) if row else None


def db_set(db, name, value):
    db.execute("INSERT OR REPLACE INTO state VALUES (?,?)", (name, json.dumps(value, ensure_ascii=False, allow_nan=False)))


def disk_check(output, reserve_gib):
    free = shutil.disk_usage(output).free
    if free < reserve_gib * 2 ** 30:
        raise OSError(28, "剩余空间低于预留 %.2f GiB；清理后用同一命令--resume继续" % reserve_gib)


def validate_input(input_path, contract):
    current = file_fingerprint(input_path)
    recorded = contract["input"]
    # A byte-identical copied file may have a different mtime.  Its actual
    # path/mtime are bound in this task's identity; the old full hash-chain is
    # still mandatory at EOF, and start/end/resume identities must be equal.
    for key in ("size", "head_tail_sha256"):
        if current[key] != recorded[key]:
            raise ValueError("输入信号TSV与已完成审计不一致（%s）；请保留原文件内容与时间身份。" % key)
    return current


def make_identity(input_path, reference_dir, contract, audit, plan):
    folder = Path(reference_dir).resolve() if reference_dir is not None else HERE / "reference"
    return {"version": VERSION, "input": validate_input(input_path, contract),
        "reference_files": {name: file_sha(folder / name) for name in PINS},
        "code": {name: file_sha(HERE / name) for name in (*HELPER_PINS, Path(__file__).name)},
        "runtime": {"python": platform.python_version(), "numpy": np.__version__, "platform": platform.system()},
        "batch_id": contract["batch_id"], "split_seed": contract["split_seed"], "selection_seed": contract["selection_seed"],
        "split_ratios": contract["split_ratios"], "sampling_namespace": "strict5_v1", "source_chain_sha256": audit["chain_sha256"],
        "source_expected_lines": audit["line"], "feature_definition": "Frozen float32 whole-TSV-overlap unique median/MAD, scales and rounding from original input_contract."}


def _outputs(output):
    return [output / scenario / name for scenario in SCENARIOS for name in ("features.csv", "features_meta.tsv")]


def committed_output_hashes(output, state):
    """Read-only validate every committed prefix before changing any output."""
    digests = {}
    for path in _outputs(output):
        relative = str(path.relative_to(output))
        committed = state["output_offsets"][relative]
        expected = state.get("output_prefix_sha256", {}).get(relative)
        if expected is None:
            raise ValueError("检查点缺少输出前缀哈希，不能猜测恢复：" + relative)
        digest = hashlib.sha256()
        if path.exists():
            before = path.stat()
            if before.st_size < committed:
                raise ValueError("产物短于已提交检查点，不能恢复：" + relative)
            with path.open("rb") as stream:
                left = committed
                while left:
                    data = stream.read(min(left, 4 * 1024 * 1024))
                    if not data:
                        raise ValueError("读取已提交产物时长度不足：" + relative)
                    digest.update(data);left -= len(data)
            after = path.stat()
            if (before.st_size, before.st_mtime_ns) != (after.st_size, after.st_mtime_ns):
                raise ValueError("核验已提交产物期间文件改变：" + relative)
        elif committed:
            raise FileNotFoundError("已提交产物缺失：" + relative)
        if digest.hexdigest() != expected:
            raise ValueError("已提交产物前缀SHA改变，拒绝截断或恢复：" + relative)
        digests[relative] = digest
    return digests


def _fast_header(raw):
    fields, left = [], 0
    for _ in range(8):
        right = raw.find(b"\t", left)
        if right < 0:
            break
        fields.append(raw[left:right])
        left = right + 1
    if len(fields) < 8:
        raise ValueError("信号TSV头部字段不足；无法核验真实read、参考和方向。")
    return fields[0].decode("utf-8"), fields[1].decode("utf-8"), fields[5].decode("utf-8")


def build_reports(output, db, state, identity, contract, plan, cells, hook=None):
    files = {}
    for relative, info in state["output_fingerprints"].items():
        path = output / relative
        if path.stat().st_size != info["size_bytes"] or file_sha(path) != info["sha256"]:
            raise ValueError("已完成特征产物改变，拒绝伪装完成：" + relative)
        files[relative] = dict(info, path=str(path))
    selected_rows = []
    for row in cells:
        key = cell_key(row["scenario"], row["split"], row["chrom"], row["strand"], row["context"], row["original_label"])
        actual = state["selected_counts"].get(key, 0)
        if actual != row["planned_windows"]:
            raise ValueError("实际选择数量与冻结配额不符：" + key)
        selected_rows.append(dict(row, selected_windows=actual))
    legacy.write_csv(output / "selected_counts.csv", [*CELL_FIELDS, "selected_windows"],
                     ([row[name] for name in [*CELL_FIELDS, "selected_windows"]] for row in selected_rows))
    cross = db.execute("SELECT COUNT(*) FROM (SELECT read_id FROM selected GROUP BY read_id HAVING COUNT(DISTINCT split)>1)").fetchone()[0]
    if cross:
        raise ValueError("同一真实read跨集合，不能完成导出。")
    scenarios = {}
    read_stats = {}
    for scenario in SCENARIOS:
        grouped = [r for r in plan["groups"] if r["scenario"] == scenario]
        per_split = {}
        for split in SPLITS:
            n, reads = db.execute("SELECT COUNT(*), COUNT(DISTINCT read_id) FROM selected WHERE scenario=? AND split=?", (scenario, split)).fetchone()
            classes = dict(db.execute("SELECT final_label,COUNT(*) FROM selected WHERE scenario=? AND split=? GROUP BY final_label", (scenario, split)))
            required = sum(r["planned_windows"] for r in cells if r["scenario"] == scenario and r["split"] == split)
            if n != required or set(classes) != set(range(5)) or len(set(classes.values())) != 1:
                raise ValueError("五类均衡或划分数量不正确：%s/%s" % (scenario, split))
            per_split[split] = {"windows": n, "unique_real_reads": reads, "class_counts": classes}
        scenarios[scenario] = {"strand": SCENARIOS[scenario]["strand"], "class_names": ["A", "T", "C", "G", "X" if scenario == "X_plus" else "Y"],
            "rows": state["rows"][scenario], "rows_by_split": {s: per_split[s]["windows"] for s in SPLITS},
            "matched_groups": len(grouped), "contexts": len({r["context"] for r in grouped}), "references": len({r["chrom"] for r in grouped}),
            "features": files[scenario + "/features.csv"], "metadata": files[scenario + "/features_meta.tsv"]}
        read_stats[scenario] = per_split
    audit = {"status": "PASS", "cross_split_read_id_count": cross, "sampling_unit": "At most one valid window per real read/context/original class; multiple classes/windows per read are retained together.",
        "scenarios": read_stats, "selected_window_count": sum(state["rows"].values()),
        "selected_unique_real_reads": db.execute("SELECT COUNT(DISTINCT read_id) FROM selected").fetchone()[0],
        "source_unique_read_ids_indexed": db.execute("SELECT COUNT(*) FROM reads").fetchone()[0],
        "source_identity_method": "Full byte hash-chain including header and duplicates, plus source size/mtime/head-tail identity.",
        "full_source_chain_sha256": state["chain_sha256"], "rows_are_not_independent_reads": True}
    summary = {"version": VERSION, "status": "complete", "completed_at": state["completed_at"], "batch_id": contract["batch_id"],
        "rows": sum(state["rows"].values()), "source_lines": state["line"], "source_bytes_scanned": state["offset"],
        "single_scan": "Both scenarios processed in the same sequential source traversal; resumed runs continue from committed byte offset.",
        "scenarios": scenarios, "input": identity["input"], "frozen_plan_sha256": identity["reference_files"]["strict_plan.json"],
        "source_chain_sha256": state["chain_sha256"], "started_at": state["started_at"], "stats": state["stats"],
        "label_mapping": {"X_plus": {str(i): i for i in range(5)}, "Y_minus": {"0": 0, "1": 1, "2": 2, "3": 3, "5": 4}},
        "feature_normalization": contract["feature_normalization"], "runtime": identity["runtime"],
        "limitations": ["BAM reference assignment is a matching proxy; natural and NCB windows occupy different positions.",
            "This is same-source read-held-out control data, not a new template/run/batch-held-out experiment.",
            "The original six-class test has already been inspected; these controls are supplementary sensitivity analyses.",
            "TSV-relative signal coordinates are retained; absolute FAST5 raw offsets are not invented."]}
    atomic_json(output / "source_identity.json", identity)
    atomic_json(output / "sampling_plan.json", {"status": "frozen", "plan_sha256": identity["reference_files"]["strict_plan.json"], "groups": plan["groups"], "minimum_k": 5,
        "rule": "per-stratum per-class exact 7k/2k/k; fixed true-read split, no resplitting", "label_mapping": summary["label_mapping"]})
    atomic_json(output / "read_audit.json", audit)
    atomic_json(output / "extraction_summary.json", summary)
    (output / "README_中文.txt").write_text("X_plus和Y_minus各自包含features.csv/features_meta.tsv。CSV为原11列，metadata保留原26字段及original_label/scenario/label_mapping。\n"
        "特征第4位为N，五分类label均为0至4；Y原label5映射到4，true_base仍为Y。\n"
        "按冻结的真实read划分保留7:2:1配额，同read窗口不跨集合。没有训练或测试模型。\n"
        "两套数据是同来源/参考分配/context/方向匹配的补充控制，天然/NCB位置仍不同；不是同位点替换或模板/批次留出。\n"
        "只需上传cplx_direction_5class_extract_core_v1_0_0.tar.gz；大特征、逐行metadata和SQLite留在服务器。\n", encoding="utf-8")
    names = ["extraction_summary.json", "selected_counts.csv", "read_audit.json", "source_identity.json", "sampling_plan.json", "README_中文.txt"]
    manifest = {"version": VERSION, "status": "complete", "file_count": len(names),
        "files": {name: {"size_bytes": (output / name).stat().st_size, "sha256": file_sha(output / name)} for name in names}}
    atomic_json(output / "EXTRACT_PACKAGE_MANIFEST.json", manifest)
    temporary = output / (CORE_NAME + ".writing")
    try:
        with tarfile.open(temporary, "w:gz", compresslevel=6) as archive:
            for name in ["EXTRACT_PACKAGE_MANIFEST.json", *names]:
                archive.add(output / name, arcname=name, recursive=False)
        os.replace(temporary, output / CORE_NAME)
    finally:
        temporary.unlink(missing_ok=True)
    return summary


def run_export(input_path, output_dir, resume=False, checkpoint_every=1000, disk_reserve_gib=2.0,
               reference_dir=None, test_hooks=None, log_every=10000, log_seconds=30):
    global STOP
    STOP = False
    if (checkpoint_every < 1 or not math.isfinite(disk_reserve_gib) or disk_reserve_gib < 0
            or log_every < 0 or not math.isfinite(log_seconds) or log_seconds <= 0):
        raise ValueError("检查点/磁盘预留/日志参数无效。")
    input_path, output = Path(input_path).expanduser().resolve(), Path(output_dir).expanduser().resolve()
    if output == input_path or input_path.is_relative_to(output):
        raise ValueError("输出目录不能覆盖原信号TSV或其父目录。")
    contract, source_audit, plan, cells, original_lib, legacy_args = load_reference(reference_dir)
    lib = RestrictedLibrary(original_lib, plan)
    identity = make_identity(input_path, reference_dir, contract, source_audit, plan)
    output.mkdir(parents=True, exist_ok=True)
    with (output / ".extract.lock").open("a") as lock:
        try:
            fcntl.flock(lock.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError as exc:
            raise RuntimeError("此输出目录已有导出任务；不允许重复启动。") from exc
        disk_check(output, disk_reserve_gib)
        db_path = output / "extraction_state.sqlite"
        if not db_path.exists():
            allowed = {".extract.lock", "logs"}
            if any(p.name not in allowed for p in output.iterdir()):
                raise ValueError("新输出目录含未知已有文件，拒绝覆盖；请使用独立新目录。")
        with ExitStack() as stack:
            db = connect(db_path)
            stack.callback(db.close)
            saved_identity = db_get(db, "identity")
            state = db_get(db, "extract")
            if saved_identity is not None and saved_identity != identity:
                raise ValueError("源文件、冻结计划、源码或数值环境改变，拒绝恢复并截断已有输出。")
            if state is not None and not resume:
                raise ValueError("任务已存在；继续请使用同一命令--resume。")
            if state is not None and state["status"] == "complete":
                if test_hooks:
                    test_hooks("before_reports", state)
                summary = build_reports(output, db, state, identity, contract, plan, cells)
                print("[RESUME] 已完成产物及哈希通过核验；未重新扫描信号TSV。", flush=True)
                print("[CORE] " + str(output / CORE_NAME), flush=True)
                print("[FULL PASS] 两套严格五分类特征导出已完成。", flush=True)
                return summary
            if state is None:
                if saved_identity is not None:
                    raise ValueError("SQLite存在身份但缺少已提交状态，拒绝猜测恢复。")
                allowed = {".extract.lock", "logs", "extraction_state.sqlite", "extraction_state.sqlite-wal", "extraction_state.sqlite-shm"}
                if any(p.name not in allowed for p in output.iterdir()):
                    raise ValueError("空任务状态旁存在未知文件，拒绝覆盖已有输出。")
                paths = _outputs(output)
                if any(p.exists() for p in paths):
                    raise ValueError("特征输出已经存在，拒绝覆盖。")
                state = {"status": "running", "line": 0, "offset": 0, "chain_sha256": "00" * 32,
                    "output_offsets": {str(p.relative_to(output)): 0 for p in paths}, "rows": {s: 0 for s in SCENARIOS},
                    "output_prefix_sha256": {str(p.relative_to(output)): hashlib.sha256(b"").hexdigest() for p in paths},
                    "remaining": {}, "needed": {}, "selected_counts": {}, "stats": {}, "started_at": now()}
                for row in cells:
                    key = cell_key(row["scenario"], row["split"], row["chrom"], row["strand"], row["context"], row["original_label"])
                    state["remaining"][key] = row["available_raw_candidates"]
                    state["needed"][key] = row["planned_windows"]
                db_set(db, "identity", identity)
                db_set(db, "extract", state)
                db.commit()
            output_hashes = committed_output_hashes(output, state)
            streams = {}
            for path in _outputs(output):
                relative = str(path.relative_to(output));committed = state["output_offsets"][relative]
                path.parent.mkdir(parents=True, exist_ok=True)
                if path.exists():
                    if path.stat().st_size < committed:
                        raise ValueError("产物短于已提交检查点，不能恢复：" + relative)
                    stream = stack.enter_context(path.open("r+b", buffering=1024 * 1024))
                    stream.seek(committed);stream.truncate()
                elif committed:
                    raise FileNotFoundError("已提交产物缺失：" + relative)
                else:
                    stream = stack.enter_context(path.open("w+b", buffering=1024 * 1024))
                if committed == 0:
                    data = legacy.row_bytes(FEATURE_HEADER if path.name == "features.csv" else META_HEADER,
                                             "," if path.name == "features.csv" else "\t")
                    if stream.write(data) != len(data):
                        raise OSError(28, "输出表头未完整写入：" + relative)
                    output_hashes[relative].update(data)
                streams[relative] = stream
            stats = Counter(state["stats"])

            def write_output(relative, data):
                if streams[relative].write(data) != len(data):
                    raise OSError(28, "输出未完整写入：" + relative)
                output_hashes[relative].update(data)

            def checkpoint():
                if test_hooks:
                    test_hooks("before_checkpoint", state)
                disk_check(output, disk_reserve_gib)
                for relative, stream in streams.items():
                    stream.flush();os.fsync(stream.fileno())
                    state["output_offsets"][relative] = stream.tell()
                    state["output_prefix_sha256"][relative] = output_hashes[relative].hexdigest()
                state["stats"] = dict(stats)
                db_set(db, "extract", state)
                db.commit()
                if test_hooks:
                    test_hooks("after_checkpoint", state)

            checkpoint()
            input_stream = stack.enter_context(input_path.open("rb", buffering=4 * 1024 * 1024))
            input_stream.seek(state["offset"])
            dirty, last_log = 0, time.monotonic()
            try:
                while True:
                    raw = input_stream.readline()
                    if not raw:
                        break
                    state["line"] += 1;state["offset"] = input_stream.tell();dirty += 1
                    digest = hashlib.sha256(raw).digest()
                    state["chain_sha256"] = hashlib.sha256(bytes.fromhex(state["chain_sha256"]) + digest).hexdigest()
                    if state["line"] == 1 and raw.lstrip(b"\xef\xbb\xbf").startswith(b"read_id\t"):
                        stats["headers"] += 1
                    else:
                        stats["source_rows"] += 1
                        rid, chrom, strand = _fast_header(raw)
                        if not rid or rid.strip() != rid or any(c in rid for c in "\r\n"):
                            raise ValueError("信号TSV缺少可核验的真实read ID。")
                        previous = db.execute("SELECT digest FROM reads WHERE read_id=?", (rid,)).fetchone()
                        if previous is not None:
                            if previous[0] != digest.hex():
                                raise ValueError("同一真实read ID出现不同源记录：" + rid)
                            stats["identical_duplicate_source_rows"] += 1
                        else:
                            split = legacy.assign_split(rid, contract["batch_id"], contract["split_seed"], contract["split_ratios"])
                            accepted = -1
                            if (chrom, strand) in lib.by_pair:
                                try:
                                    read = legacy.parse_read(raw, legacy_args, lib)
                                    candidates = legacy.read_candidates(read, lib, legacy_args)
                                except (ValueError, UnicodeError) as exc:
                                    accepted = 0;stats["planned_pair_rejected:" + str(exc).split("\n")[0][:100]] += 1
                                else:
                                    accepted = 1;stats["parsed_valid_planned_reads"] += 1
                                    scenario = "X_plus" if strand == "+" else "Y_minus"
                                    for (context, label), site in candidates.items():
                                        key = cell_key(scenario, split, chrom, strand, context, label)
                                        if key not in state["remaining"]:
                                            raise ValueError("出现冻结计划以外的候选单元：" + key)
                                        remaining, needed = state["remaining"][key], state["needed"][key]
                                        if remaining <= 0:
                                            raise ValueError("候选数量超过原审计：" + key)
                                        u = int.from_bytes(legacy.digest_text("strict5_v1", contract["selection_seed"], contract["batch_id"], rid, key)[:8], "big") / 2 ** 64
                                        chosen = needed > 0 and (needed == remaining or u < needed / remaining)
                                        state["remaining"][key] -= 1
                                        stats["planned_candidate_windows_seen"] += 1
                                        if chosen:
                                            values = read.features(site, dwell_scale=contract["dwell_scale"], quality_scale=contract["quality_scale"],
                                                mad_use_unique=bool(contract["mad_use_unique"]), signal_round=contract["signal_round"], stat_round=contract["stat_round"])
                                            final_label = 4 if label == 5 else label
                                            values[-1] = final_label
                                            number = state["rows"][scenario] + 1
                                            metadata = legacy.metadata(read, site, legacy_args, split, state["line"], number)
                                            metadata[legacy.META_HEADER.index("label")] = final_label
                                            metadata += [label, scenario, "5->4" if label == 5 else "identity"]
                                            write_output(scenario + "/features.csv", legacy.row_bytes(values))
                                            write_output(scenario + "/features_meta.tsv", legacy.row_bytes(metadata, "\t"))
                                            db.execute("INSERT INTO selected VALUES (?,?,?,?,?,?,?,?,?,?,?)", (scenario, rid, context, label, final_label, split, chrom, strand, state["line"], number, site["target_pos"]))
                                            state["rows"][scenario] = number
                                            state["needed"][key] -= 1
                                            state["selected_counts"][key] = state["selected_counts"].get(key, 0) + 1
                            else:
                                stats["source_reads_skipped_outside_planned_reference_direction"] += 1
                            db.execute("INSERT INTO reads VALUES (?,?,?,?,?,?,?)", (rid, state["line"], digest.hex(), split, chrom, strand, accepted))
                    requested_stop = test_hooks("after_row", state) == "stop" if test_hooks else False
                    if dirty >= checkpoint_every or STOP or requested_stop:
                        checkpoint();dirty = 0
                        if STOP or requested_stop:
                            raise InterruptedError("检查点已保存；用相同命令--resume继续。")
                    if (log_every and state["line"] % log_every == 0) or time.monotonic() - last_log >= log_seconds:
                        print("[SCAN] lines=%s scanned=%.2fGiB X=%s Y=%s remaining_to_select=%s" %
                            (f"{state['line']:,}", state["offset"] / 2 ** 30, f"{state['rows']['X_plus']:,}", f"{state['rows']['Y_minus']:,}", f"{sum(state['needed'].values()):,}"), flush=True)
                        last_log = time.monotonic()
                if (state["line"] != source_audit["line"] or state["offset"] != source_audit["offset"]
                        or state["chain_sha256"] != source_audit["chain_sha256"]):
                    raise ValueError("完整信号TSV行数/字节数/哈希链与原审计不一致，不能标记完成。")
                if any(state["remaining"].values()) or any(state["needed"].values()):
                    raise ValueError("候选计数或冻结配额未完全满足，拒绝将导出标记为完成。")
                if validate_input(input_path, contract) != identity["input"]:
                    raise ValueError("导出期间原信号TSV身份改变。")
                checkpoint()
                state["output_fingerprints"] = {relative: {"sha256": file_sha(output / relative), "size_bytes": (output / relative).stat().st_size}
                                                for relative in streams}
                state["status"] = "complete";state["completed_at"] = now()
                checkpoint()
                if test_hooks:
                    test_hooks("before_reports", state)
                summary = build_reports(output, db, state, identity, contract, plan, cells)
            except BaseException:
                db.rollback()
                raise
    print("[CORE] " + str(output / CORE_NAME), flush=True)
    print("[FULL PASS] 两套严格五分类特征导出已完成，总窗口=%s。" % f"{summary['rows']:,}", flush=True)
    return summary


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", default=DEFAULT_INPUT)
    parser.add_argument("--output_dir", default=DEFAULT_OUTPUT)
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--checkpoint_every", type=int, default=1000)
    parser.add_argument("--disk_reserve_gib", type=float, default=2.0)
    parser.add_argument("--log_every", type=int, default=10000)
    parser.add_argument("--log_seconds", type=float, default=30.0)
    parser.add_argument("--version", action="version", version=VERSION)
    args = parser.parse_args(argv)
    for signum in (signal.SIGINT, signal.SIGTERM):
        signal.signal(signum, handle_stop)
    try:
        run_export(args.input, args.output_dir, args.resume, args.checkpoint_every,
                   args.disk_reserve_gib, log_every=args.log_every, log_seconds=args.log_seconds)
    except InterruptedError as exc:
        print("[INTERRUPTED] " + str(exc), file=sys.stderr)
        return 130
    except (OSError, ValueError, RuntimeError, KeyError, TypeError) as exc:
        print("[ERROR] %s: %s" % (type(exc).__name__, exc), file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    sys.exit(main())
