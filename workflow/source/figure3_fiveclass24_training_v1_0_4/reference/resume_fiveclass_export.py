#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Recover v1.0.0 five-class exports from its saved SQLite selections.

Never resample or append partial feature files. --inspect performs no writes.
The original script remains unchanged and is loaded only after a SHA256 check.
"""
from __future__ import annotations

import argparse
from collections import Counter
from contextlib import contextmanager
import csv
from datetime import datetime, timezone
from decimal import Decimal, InvalidOperation
import fcntl
import hashlib
import importlib.util
import itertools
import json
import os
from pathlib import Path
import shutil
import sqlite3
import sys
import tempfile
import time
import traceback

VERSION = "1.0.0"
SOURCE_SHA256 = "f9c1551c37d4657ddb8fa470e2a13e75573d905c94cc73fbf6a5767b21bea8de"
BASE = Path("/path/to/revision-inputs/CPLX-XNA-replenishment/CPLX-XNA-replenishment/fast5_single_guppy")
DEFAULT_OUTPUT = BASE / "fiveclass_overlap_read_split_721"
WINDOW_COLUMNS = ["ordinal", "read_id", "motif", "label", "chrom", "split", "source", "strand", "run_id", "read_id_kind", "reference_identity_known", "priority", "physical_key"]


def log(message):
    print(message, file=sys.stderr, flush=True)


def digest(path):
    result = hashlib.sha256()
    with open(path, "rb") as handle:
        for block in iter(lambda: handle.read(8 * 1024 * 1024), b""):
            result.update(block)
    return result.hexdigest()


def utc():
    return datetime.now(timezone.utc).isoformat()


def json_atomic(path, value):
    path = Path(path)
    temporary = path.with_name(path.name + ".recovery-tmp-" + str(os.getpid()))
    try:
        with open(temporary, "w", encoding="utf-8") as handle:
            json.dump(value, handle, ensure_ascii=False, indent=2, allow_nan=False)
            handle.write("\n")
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary, path)
    finally:
        if temporary.exists():
            temporary.unlink()


def load_source(path):
    path = Path(path).expanduser().resolve()
    if not path.is_file() or digest(path) != SOURCE_SHA256:
        raise ValueError("需要原版 v1.0.0 五分类脚本（SHA256 必须匹配），不会使用修改过的逻辑：" + str(path))
    spec = importlib.util.spec_from_file_location("original_fiveclass_recovery_source", path)
    module = importlib.util.module_from_spec(spec)
    # Suppress __pycache__ creation, including during read-only --inspect.
    previous = sys.dont_write_bytecode
    sys.dont_write_bytecode = True
    try:
        spec.loader.exec_module(module)
    finally:
        sys.dont_write_bytecode = previous
    return module, path


def find_stage(options):
    if options.resume_from:
        stage = Path(options.resume_from).expanduser().resolve()
        if not stage.is_dir():
            raise ValueError("恢复目录不存在：" + str(stage))
        return stage
    target = Path(options.output_dir).expanduser().resolve() if options.output_dir else DEFAULT_OUTPUT
    parent = target.parent
    candidates = sorted(set(list(parent.glob("." + target.name + ".partial-*")) +
                            list(parent.glob(target.name + ".failed-*")) +
                            list(parent.glob("." + target.name + ".recovery-*"))))
    candidates = [path for path in candidates if path.is_dir() and (path / "selection.sqlite").is_file() and (path / "requested_config.json").is_file()]
    if not candidates:
        raise ValueError("没有找到带 selection.sqlite 和 requested_config.json 的恢复目录。请用 --resume_from 指定原 .partial-* 或 .failed-* 目录。查找位置：" + str(parent))
    if len(candidates) != 1:
        raise ValueError("发现多个恢复目录，请用 --resume_from 明确选择，不能猜测：\n" + "\n".join(str(path) for path in candidates))
    return candidates[0].resolve()


def load_arguments(stage, options, module):
    with open(stage / "requested_config.json", encoding="utf-8") as handle:
        config = json.load(handle)
    required = ("input", "meta", "output_dir", "cases", "split_seed", "sample_seed", "split_ratios", "reference_balance")
    missing = [key for key in required if key not in config]
    if missing:
        raise ValueError("保存的配置不完整：" + ", ".join(missing))
    args = argparse.Namespace(**config)
    args.input = Path(args.input).expanduser().resolve()
    args.meta = Path(args.meta).expanduser().resolve()
    args.output_dir = Path(options.output_dir or args.output_dir).expanduser().resolve()
    args.normalized_ratios = module.parse_split_ratios(args.split_ratios)
    args.integer_weights = module.integer_weights(args.split_ratios)
    args.log_every = int(getattr(args, "log_every", 100000))
    args.log_seconds = float(getattr(args, "log_seconds", 30.0))
    if not args.cases or len(set(args.cases)) != len(args.cases) or any(case not in module.TASK_SOURCE_LABELS for case in args.cases):
        raise ValueError("保存的 cases 无效")
    if args.reference_balance not in ("equal", "none"):
        raise ValueError("保存的 reference_balance 无效")
    if args.output_dir == stage:
        raise ValueError("最终输出目录不能与恢复状态目录相同")
    if stage in args.output_dir.parents or args.output_dir in stage.parents:
        raise ValueError("最终输出目录与恢复状态目录不能互为父子，避免递归复制或覆盖状态")
    if args.output_dir.exists():
        raise FileExistsError("最终输出目录已存在，不会覆盖：" + str(args.output_dir))
    if args.output_dir in args.input.parents or args.output_dir in args.meta.parents or args.output_dir in (args.input, args.meta):
        raise ValueError("最终输出目录不能覆盖输入文件或其父目录")
    if stage in args.input.parents or stage in args.meta.parents or args.input == stage or args.meta == stage:
        raise ValueError("原始输入不能位于恢复状态目录内，避免将输入当作派生文件改写")
    for case in args.cases:
        folder = stage / ("fiveclass_" + case)
        if folder.is_symlink():
            raise ValueError("任务子目录不能是符号链接：" + str(folder))
    summary_path = stage / "summary.json"
    if summary_path.is_file():
        try:
            with open(summary_path, encoding="utf-8") as handle:
                old_summary = json.load(handle)
        except (OSError, ValueError):
            old_summary = {}
        if old_summary.get("status") == "complete":
            try:
                with open(stage / "recovery_state.json", encoding="utf-8") as handle:
                    prior_recovery = json.load(handle)
            except (OSError, ValueError):
                prior_recovery = {}
            if prior_recovery.get("original_script_sha256") != SOURCE_SHA256 or prior_recovery.get("frozen_selection_reused") is not True:
                raise ValueError("该目录已有完整原始 summary，不是中断导出状态，拒绝重建已完成数据集")
    return args, config


def open_readonly(path):
    # mode=ro respects rollback/WAL journals; immutable=1 is intentionally absent.
    con = sqlite3.connect(Path(path).resolve().as_uri() + "?mode=ro", uri=True)
    con.execute("PRAGMA temp_store=FILE")
    con.execute("BEGIN")
    return con


def validate_database(module, con, args):
    log("[恢复审计] SQLite quick_check；只读，不修改冻结选择")
    checks = list(con.execute("PRAGMA quick_check"))
    if checks != [("ok",)]:
        raise ValueError("selection.sqlite 完整性检查未通过：" + repr(checks[:10]))
    log("[恢复审计] quick_check 通过；核验表结构、输入 ordinal 与 read 分区")
    for table, columns in (("windows", WINDOW_COLUMNS + ["selected"]), ("task_selections", ["case_name", "ordinal", "split"])):
        actual = {row[1] for row in con.execute("PRAGMA table_info(" + table + ")")}
        if not set(columns).issubset(actual):
            raise ValueError("selection.sqlite 缺少完整表结构：" + table)
    total, minimum, maximum = con.execute("SELECT count(*),min(ordinal),max(ordinal) FROM main.windows").fetchone()
    if not total or minimum != 1 or maximum != total:
        raise ValueError("审计输入的 ordinal 不完整，不能恢复导出")
    invalid = con.execute("SELECT ordinal FROM main.windows WHERE label NOT IN (0,1,2,3,4,5) OR split NOT IN ('train','validation','test') LIMIT 1").fetchone()
    if invalid:
        raise ValueError("数据库中的类别/分区无效：" + repr(invalid))
    cases = {row[0] for row in con.execute("SELECT DISTINCT case_name FROM main.task_selections")}
    if cases != set(args.cases):
        raise ValueError("冻结选择的任务不完整或不匹配配置：" + repr(cases))
    bad = con.execute("SELECT s.ordinal FROM main.task_selections s LEFT JOIN main.windows w ON w.ordinal=s.ordinal WHERE w.ordinal IS NULL OR s.split!=w.split LIMIT 1").fetchone()
    if bad:
        raise ValueError("冻结选择存在丢失输入或分区冲突：" + repr(bad))
    duplicate = con.execute("SELECT case_name,ordinal FROM main.task_selections GROUP BY case_name,ordinal HAVING count(*)!=1 LIMIT 1").fetchone()
    if duplicate:
        raise ValueError("冻结选择存在重复记录：" + repr(duplicate))
    conflict = con.execute("SELECT read_id FROM main.windows GROUP BY read_id HAVING count(DISTINCT split)>1 OR count(DISTINCT read_id_kind)>1 LIMIT 1").fetchone()
    if conflict:
        raise ValueError("原输入 read 分区或身份类型冲突：" + str(conflict[0]))
    cross = con.execute("SELECT w.read_id FROM main.task_selections s JOIN main.windows w ON s.ordinal=w.ordinal GROUP BY w.read_id HAVING count(DISTINCT s.split)>1 LIMIT 1").fetchone()
    if cross:
        raise ValueError("两任务间同一 read 跨分区：" + str(cross[0]))
    log("[恢复审计] read 分区通过；汇总完整输入 motif / 标签 / 参考 / 分区容量")
    counts = Counter({(motif, label, chrom, split): n for motif, label, chrom, split, n in con.execute("SELECT motif,label,chrom,split,count(*) FROM main.windows GROUP BY motif,label,chrom,split")})
    source_counts = Counter({("input", source, strand, run_id, chrom, label, split): n for source, strand, run_id, chrom, label, split, n in con.execute("SELECT source,strand,run_id,chrom,label,split,count(*) FROM main.windows GROUP BY source,strand,run_id,chrom,label,split")})
    log("[恢复审计] 输入容量汇总完成；核验每个任务的冻结整数配额")
    tasks = {}
    for case in args.cases:
        args.case, args.source_labels = case, module.TASK_SOURCE_LABELS[case]
        filtered = Counter({key: n for key, n in counts.items() if key[1] in args.source_labels})
        quotas, motifs, references, retained = module.make_quotas(args, filtered)
        if not retained:
            raise ValueError("任务 " + case + " 没有可恢复的完整配额")
        selected = Counter({(motif, label, chrom, split): n for motif, label, chrom, split, n in con.execute("SELECT w.motif,w.label,w.chrom,w.split,count(*) FROM main.task_selections s JOIN main.windows w ON w.ordinal=s.ordinal WHERE s.case_name=? GROUP BY w.motif,w.label,w.chrom,w.split", (case,))})
        aggregate = Counter()
        for (motif, label, chrom, split), n in selected.items():
            if label not in args.source_labels:
                raise ValueError("冻结选择含任务外类别：" + case)
            aggregate[(motif, label, split)] += n
        expected = Counter({(motif, label, chrom, split): n for motif, label, chrom, split, n in quotas})
        actual = selected if args.reference_balance == "equal" else Counter({(motif, label, None, split): n for (motif, label, split), n in aggregate.items()})
        if actual != expected:
            raise ValueError("任务 " + case + " 冻结选择与原始精确配额不一致，不会重新抽样")
        per_class = Counter()
        for (motif, label, chrom, split), n in selected.items():
            per_class[(split, label)] += n
        tasks[case] = {"counts": filtered, "quotas": quotas, "motif_report": motifs, "reference_report": references, "retained": retained, "selected": selected, "per_class": per_class}
        log("[冻结选择 {}] {:,} 窗口；{} motif；配额一致".format(case, sum(selected.values()), len(retained)))
    return total, counts, source_counts, tasks


def select_task_view(con, case):
    con.execute("DROP VIEW IF EXISTS temp.windows")
    columns = ",".join("w." + name for name in WINDOW_COLUMNS)
    # case values were checked against fixed internal constants.
    con.execute("CREATE TEMP VIEW windows AS SELECT " + columns + ",EXISTS(SELECT 1 FROM main.task_selections s WHERE s.ordinal=w.ordinal AND s.case_name='" + case + "') AS selected FROM main.windows w")


def storage_report(folder):
    usage = shutil.disk_usage(folder)
    stats = os.statvfs(folder)
    return {"path": str(folder), "free_bytes": usage.free, "total_bytes": usage.total, "available_inodes": stats.f_favail}


def derived_files(module, args, stage):
    return [stage / ("fiveclass_" + case) / (split + suffix) for case in args.cases for split in module.SPLIT_NAMES for suffix in (".csv", "_meta.tsv")]


def verify_persisted_fingerprints(stage, cases, current):
    """Reuse any hashes actually saved before this recovery attempt.

    A previous recovery-start hash establishes continuity since that recovery,
    not the missing hash of the first original audit.
    """
    evidence = []
    original_initial_available = False
    candidates = [(stage / "recovery_state.json", "input_fingerprints_at_recovery_start", False)]
    candidates += [(stage / "summary.json", "input_fingerprints", True)]
    candidates += [(stage / ("fiveclass_" + case) / "summary.json", "input_fingerprints", True) for case in cases]
    for path, field, original_summary in candidates:
        if not path.is_file():
            continue
        try:
            with open(path, encoding="utf-8") as handle:
                saved = json.load(handle)
        except (OSError, ValueError):
            continue
        if field == "input_fingerprints_at_recovery_start" and (saved.get("original_script_sha256") != SOURCE_SHA256 or saved.get("frozen_selection_reused") is not True):
            continue
        fingerprints = saved.get(field)
        if not isinstance(fingerprints, dict) or not all(isinstance(fingerprints.get(name), dict) and isinstance(fingerprints[name].get("sha256"), str) and len(fingerprints[name]["sha256"]) == 64 for name in ("features", "metadata")):
            continue
        for name in ("features", "metadata"):
            if fingerprints[name] != current[name]:
                raise ValueError("当前输入指纹与已保存的恢复/原始记录不一致，拒绝改写半成品：{}；记录 {}".format(name, path))
        # Summaries generated by this helper hold current recovery fingerprints,
        # so they cannot turn a previously unknown original hash into proof.
        is_legacy = original_summary and "recovery" not in saved and saved.get("script_version") == "1.0.0"
        previously_verified = (not original_summary and saved.get("persisted_fingerprint_verification", {}).get("original_initial_fingerprints_available") is True)
        original_initial_available |= is_legacy or previously_verified
        evidence.append({"file": str(path), "field": field, "scope": "original_completed_audit" if is_legacy else "previous_recovery_start", "matched": True})
    return {"matched_persisted_records": evidence, "original_initial_fingerprints_available": original_initial_available}


@contextmanager
def state_lock(stage):
    # The directory itself is flock'ed, avoiding a new lock file for --inspect.
    descriptor = os.open(stage, os.O_RDONLY | getattr(os, "O_DIRECTORY", 0))
    try:
        try:
            fcntl.flock(descriptor, fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError as error:
            raise RuntimeError("同一恢复目录已有恢复进程运行") from error
        yield
    finally:
        os.close(descriptor)


def row_identity(module, args, ordinal, feature, meta, motif):
    split = module.validate_assignment(meta, args, ordinal)
    label = int(feature[-1])
    source = module.normalize_known(meta.get("source_dataset") or meta.get("source_type"))
    strand = module.normalize_known(meta.get("bam_strand"))
    run_id = module.normalize_known(meta.get("run_id"))
    kind = "positive_row_proxy" if module.is_positive_row_proxy(meta, label) else "provided_real_read_id"
    known = module.normalize_known(meta.get("chrom")) != "unknown"
    chrom = module.normalize_known(meta.get("chrom"))
    priority = hashlib.blake2b((str(args.sample_seed) + "\x00" + str(ordinal)).encode("utf-8"), digest_size=16, person=b"sixclass-sample").digest()
    physical = None
    if meta.get("chrom") != meta.get("chrom", "").strip():
        raise ValueError("chrom 首尾含空白，数据行 " + str(ordinal))
    values = [meta.get(name, "").strip() for name in ("kmer_ref_start", "kmer_ref_end")]
    if all(values) and strand in ("+", "-") and known and kind != "positive_row_proxy":
        try:
            coordinates = [Decimal(value) for value in values]
            if any(not value.is_finite() or value != value.to_integral_value() for value in coordinates):
                raise ValueError("非有限整数坐标")
            start, end = map(int, coordinates)
            if start > end:
                raise ValueError("坐标反向")
        except (ValueError, InvalidOperation) as error:
            raise ValueError("窗口坐标无效，数据行 " + str(ordinal)) from error
        physical = json.dumps([meta["read_id"], meta["chrom"], strand, start, end], ensure_ascii=False, separators=(",", ":"))
    return (ordinal, meta["read_id"], motif, label, chrom, split, source, strand, run_id, kind, int(known), priority, physical)


def export_verified(module, args, stage, con, header, total):
    original = module.iter_input
    seen = [0]
    def verified(features, metadata):
        saved = iter(con.execute("SELECT " + ",".join(WINDOW_COLUMNS) + " FROM main.windows ORDER BY ordinal"))
        last_log = time.monotonic()
        for ordinal, feature, meta, motif in original(features, metadata):
            expected = next(saved, None)
            actual = row_identity(module, args, ordinal, feature, meta, motif)
            if expected != actual:
                differing = [name for name, old, new in zip(WINDOW_COLUMNS, expected or (), actual) if old != new]
                raise ValueError("当前输入与冻结审计不一致，数据行 {}，字段 {}；不会继续复用选择".format(ordinal, ",".join(differing) or "ordinal/row_count"))
            seen[0] = ordinal
            now = time.monotonic()
            if (args.log_every and ordinal % args.log_every == 0) or now - last_log >= args.log_seconds:
                log("[恢复导出] 已复核并扫描 {:,} / {:,} 行".format(ordinal, total))
                last_log = now
            yield ordinal, feature, meta, motif
        if next(saved, None) is not None or seen[0] != total:
            raise ValueError("当前输入未覆盖完整冻结审计")
    module.iter_input = verified
    try:
        return module.export_all_tasks(args, stage, con, header)
    finally:
        module.iter_input = original


def restore_audits(module, args, stage, con, source_counts):
    """Compare existing reports with frozen state; rewrite only incomplete ones."""
    original = module.csv_write
    invalid = set()
    reused = []
    def checking(path, header, rows, delimiter=","):
        path = Path(path)
        if not path.is_file():
            invalid.add(path.name)
            return
        try:
            with open(path, encoding="utf-8", newline="") as handle:
                reader = csv.reader(handle, delimiter=delimiter, strict=True)
                if next(reader, None) != list(header):
                    invalid.add(path.name)
                    return
                sentinel = object()
                for actual, expected in itertools.zip_longest(reader, rows, fillvalue=sentinel):
                    if actual is sentinel or expected is sentinel or actual != ["" if value is None else str(value) for value in expected]:
                        invalid.add(path.name)
                        return
            reused.append(path.name)
        except (OSError, csv.Error, UnicodeError):
            invalid.add(path.name)
    module.csv_write = checking
    try:
        module.export_task_audits(args, stage, con, source_counts)
    finally:
        module.csv_write = original
    if invalid:
        log("[审计报告] 重建缺失/不完整文件：" + ", ".join(sorted(invalid)))
        def rebuilding(path, header, rows, delimiter=","):
            if Path(path).name in invalid:
                # Unlink first: a copied recovery stage can hardlink old reports.
                Path(path).unlink(missing_ok=True)
                original(path, header, rows, delimiter)
        module.csv_write = rebuilding
        try:
            module.export_task_audits(args, stage, con, source_counts)
        finally:
            module.csv_write = original
    return {"existing_reports_revalidated_and_reused": reused, "rebuilt_reports": sorted(invalid)}


def copy_state(source, target, module, args):
    omitted = {path.relative_to(source) for path in derived_files(module, args, source)}
    omitted.update((Path("summary.json"), Path("error.json"), Path("recovery_error.json"), Path("recovery_state.json")))
    database_storage = None
    for path in source.rglob("*"):
        relative = path.relative_to(source)
        if relative in omitted or (path.is_symlink() and relative != Path("selection.sqlite")):
            continue
        destination = target / relative
        if path.is_dir():
            destination.mkdir(parents=True, exist_ok=True)
        elif path.is_file():
            destination.parent.mkdir(parents=True, exist_ok=True)
            if relative == Path("selection.sqlite") and path.is_symlink():
                destination.symlink_to(path.resolve())
                database_storage = {"mode": "external_symlink", "self_contained": False, "file": "selection.sqlite", "external_database": str(path.resolve()), "source_stage_must_be_preserved": str(path.resolve().parent)}
                continue
            # State is read-only; hardlinks preserve space within this filesystem.
            try:
                os.link(path, destination)
                if relative == Path("selection.sqlite"):
                    database_storage = {"mode": "hardlink", "self_contained": True, "file": "selection.sqlite"}
            except OSError:
                if relative == Path("selection.sqlite"):
                    # Across filesystems, preserve the single frozen database.
                    # The old state must remain present while the link is used.
                    destination.symlink_to(path.resolve())
                    database_storage = {"mode": "external_symlink", "self_contained": False, "file": "selection.sqlite", "external_database": str(path.resolve()), "source_stage_must_be_preserved": str(source)}
                else:
                    shutil.copy2(path, destination)
    if database_storage is None:
        raise ValueError("恢复副本没有找到冻结 selection.sqlite")
    return database_storage


def execute(options):
    module, source_path = load_source(options.source_script)
    source_stage = find_stage(options)
    args, requested = load_arguments(source_stage, options, module)
    log("[恢复目录] " + str(source_stage))
    log("[最终目录] " + str(args.output_dir))
    con = open_readonly(source_stage / "selection.sqlite")
    stage = None
    try:
        total, counts, source_counts, tasks = validate_database(module, con, args)
        sqlite_tmpdir = Path(os.environ.get("SQLITE_TMPDIR") or tempfile.gettempdir()).expanduser().resolve()
        spaces = [storage_report(source_stage), storage_report(sqlite_tmpdir)]
        for space in spaces:
            log("[磁盘] {}：可用 {:.2f} GiB；可用 inode {}".format(space["path"], space["free_bytes"] / 2**30, space["available_inodes"]))
        report = {"status": "inspect_ok", "helper_version": VERSION, "state_directory": str(source_stage), "output_directory": str(args.output_dir), "input_windows": total, "cases": {case: {"selected_windows": sum(task["selected"].values()), "retained_motifs": len(task["retained"])} for case, task in tasks.items()}, "storage": spaces, "sqlite_tmpdir_environment": os.environ.get("SQLITE_TMPDIR"), "read_only_inspection": True, "frozen_selection_reused": True, "input_numeric_payload_initial_sha_available": False}
        if not options.resume:
            print(json.dumps(report, ensure_ascii=False, indent=2))
            log("[检查完成] 冻结选择可恢复；未写入、删除或重新抽样。继续导出请加 --resume --in_place。")
            return report
        if not args.input.is_file() or not args.meta.is_file():
            raise ValueError("原始合并 CSV 或配对 metadata 不存在，无法恢复数值特征")
        log("[输入指纹] 读取原始 CSV / metadata 并比对已有记录；若原失败任务未保存初始指纹，无法独立核验旧数值 payload")
        fingerprints = {"features": module.fingerprint(args.input), "metadata": module.fingerprint(args.meta)}
        persisted = verify_persisted_fingerprints(source_stage, args.cases, fingerprints)
        if persisted["matched_persisted_records"]:
            log("[输入指纹] 与 {} 份已保存指纹一致".format(len(persisted["matched_persisted_records"])))
        db_before = module.fingerprint(source_stage / "selection.sqlite")
        with state_lock(source_stage):
            if args.output_dir.exists():
                raise FileExistsError("最终输出目录已存在，不会覆盖：" + str(args.output_dir))
            args.output_dir.parent.mkdir(parents=True, exist_ok=True)
            if options.in_place:
                if source_stage.stat().st_dev != args.output_dir.parent.stat().st_dev:
                    raise ValueError("--in_place 要求最终目录与恢复状态位于同一文件系统，以便原子重命名；跨盘请移除 --in_place，数据库会引用原冻结文件")
                stage = source_stage
                database_storage = {"mode": "in_place", "self_contained": not (source_stage / "selection.sqlite").is_symlink(), "file": "selection.sqlite"}
                if not database_storage["self_contained"]:
                    database_storage.update(external_database=str((source_stage / "selection.sqlite").resolve()), source_stage_must_be_preserved=str((source_stage / "selection.sqlite").resolve().parent))
            else:
                args.output_dir.parent.mkdir(parents=True, exist_ok=True)
                stage = Path(tempfile.mkdtemp(prefix="." + args.output_dir.name + ".recovery-", dir=args.output_dir.parent))
                database_storage = copy_state(source_stage, stage, module, args)
            marker = {"status": "exporting", "created_utc": utc(), "source_stage": str(source_stage), "current_stage": str(stage), "final_output": str(args.output_dir), "original_script_sha256": SOURCE_SHA256, "helper_sha256": digest(Path(__file__)), "frozen_selection_reused": True, "no_resampling": True, "input_fingerprints_at_recovery_start": fingerprints, "initial_numeric_payload_hash_not_saved": not persisted["original_initial_fingerprints_available"], "persisted_fingerprint_verification": persisted, "partial_feature_exports_rebuilt_from_start": True, "in_place": options.in_place, "database_storage": database_storage}
            json_atomic(stage / "recovery_state.json", marker)
            # Only the recognized derived split files may be removed in place.
            # All database/config/input/audit files remain available.
            for path in derived_files(module, args, stage):
                path.unlink(missing_ok=True)
            summaries = {}
            for case, task in tasks.items():
                args.case, args.source_labels = case, module.TASK_SOURCE_LABELS[case]
                folder = stage / ("fiveclass_" + case)
                folder.mkdir(exist_ok=True)
                select_task_view(con, case)
                log("[恢复 {}] 比对和复用已有审计报告；不复制完整 provenance 文件".format(case))
                # Rebuild small deterministic reports atomically where practical.
                for name in ("label_mapping.json", "motif_balance_report.csv", "reference_balance_report.csv", "summary.json"):
                    (folder / name).unlink(missing_ok=True)
                json_atomic(folder / "label_mapping.json", module.label_mapping(case))
                module.write_task_reports(folder, task["motif_report"], task["reference_report"], task["selected"])
                audit = restore_audits(module, args, folder, con, source_counts)
                log("[恢复 {}] 审计报告完成；重建 read / reference 统计 summary".format(case))
                summary = module.task_summary(args, con, fingerprints, total, task["counts"], task["retained"], task["per_class"])
                summary["recovery"] = {**marker, "audit_reports": audit, "input_identity_rows_to_be_revalidated": total, "numeric_payload_limit": None if persisted["original_initial_fingerprints_available"] else "The failed original run did not persist its input hashes; recovery verifies every stored row identity and fingerprints current inputs before/after export, but cannot independently prove original numeric payload equality. Prior recovery-start hashes, when present, are checked and establish continuity only since that recovery."}
                summaries[case] = summary
            con.execute("DROP VIEW IF EXISTS temp.windows")
            observed = export_verified(module, args, stage, con, module.read_metadata_header(args.meta), total)
            for case, task in tasks.items():
                actual = Counter({(split, label): n for (task_name, split, label), n in observed.items() if task_name == case})
                if actual != task["per_class"]:
                    raise RuntimeError("恢复输出数量与冻结选择不一致：" + case)
            for name, path in (("features", args.input), ("metadata", args.meta)):
                if module.fingerprint(path) != fingerprints[name]:
                    raise RuntimeError("恢复导出期间输入发生变化：" + str(path))
            if module.fingerprint(source_stage / "selection.sqlite") != db_before:
                raise RuntimeError("冻结数据库在恢复期间发生变化")
            for case, summary in summaries.items():
                folder = stage / ("fiveclass_" + case)
                summary["selection_database"] = {"file": "../selection.sqlite", "task_selections_case_name": case, "authoritative": "task_selections; windows.selected remains its original interrupted scratch state and must not be interpreted as a union"}
                summary["recovery"]["input_identity_rows_revalidated"] = total
                summary["recovery"]["status"] = "complete"
                summary["code_sha256"] = {"original_script": SOURCE_SHA256, "recovery_helper": digest(Path(__file__))}
                summary["output_fingerprints"] = {path.name: {"size_bytes": path.stat().st_size, "sha256": digest(path)} for path in sorted(folder.iterdir()) if path.is_file() and path.name != "summary.json"}
                json_atomic(folder / "summary.json", summary)
            marker.update(status="complete", completed_utc=utc(), input_identity_rows_revalidated=total)
            json_atomic(stage / "recovery_state.json", marker)
            root_summary = {"status": "complete", "script_version": module.VERSION, "recovery_helper_version": VERSION, "created_utc": utc(), "input_fingerprints": fingerprints, "input_windows": total, "input_class_counts": {str(label): sum(n for key, n in counts.items() if key[1] == label) for label in range(6)}, "num_classes_per_task": 5, "code_sha256": {"original_script": SOURCE_SHA256, "recovery_helper": digest(Path(__file__))}, "tasks": {case: {"directory": "fiveclass_" + case, "label_mapping": module.label_mapping(case), "retained_motif_count": summary["retained_motif_count"], "split_counts": summary["split_counts"]} for case, summary in summaries.items()}, "same_read_in_different_splits_across_tasks": 0, "all_sixclass_input_rows_validated_before_filtering": True, "split_assignments_reused_without_task_specific_hashing": True, "frozen_selection_reused": True, "selection_database": "selection.sqlite; task_selections(case_name,ordinal,split) is authoritative. windows.selected is unchanged interrupted scratch state, not a union.", "recovery": marker, "initial_numeric_payload_identity_independently_verified": persisted["original_initial_fingerprints_available"]}
            root_summary["output_fingerprints"] = {path.relative_to(stage).as_posix(): {"size_bytes": path.stat().st_size, "sha256": digest(path)} for path in sorted(stage.rglob("*")) if path.is_file() and path.name not in ("summary.json", "error.json", "recovery_error.json")}
            json_atomic(stage / "summary.json", root_summary)
            if args.output_dir.exists():
                raise FileExistsError("最终输出目录在恢复中被创建，不会覆盖")
            # Read-only SQLite state has never been mutated.
            con.close()
            con = None
            os.rename(stage, args.output_dir)
            log("[恢复完成] " + str(args.output_dir))
            return root_summary
    except BaseException as error:
        if stage is not None and stage.exists():
            log("[恢复状态已保留] " + str(stage))
            try:
                json_atomic(stage / "recovery_error.json", {"status": "failed", "created_utc": utc(), "type": type(error).__name__, "error": str(error), "traceback": traceback.format_exc()})
            except OSError as write_error:
                log("[错误记录未写入] " + str(write_error))
        raise
    finally:
        if con is not None:
            con.close()


def parser():
    result = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    mode = result.add_mutually_exclusive_group()
    mode.add_argument("--inspect", action="store_true", help="只读检查保存状态（默认），不重写导出")
    mode.add_argument("--resume", action="store_true", help="复用冻结选择，重新导出未完整完成的 split 文件")
    result.add_argument("--resume_from", help="原 .partial-* 或 .failed-* 状态目录；多个候选时必填")
    result.add_argument("--source_script", default=str(Path(__file__).with_name("balance_and_split_xna_pc_5class_by_read.py")), help="原版 v1.0.0 五分类脚本，SHA256 必须匹配")
    result.add_argument("--output_dir", help="最终输出路径，默认沿用保存的 requested_config；必须尚不存在")
    result.add_argument("--in_place", action="store_true", help="在原状态目录重建 12 个派生 split 文件，避免旧半成品和新导出同时占空间；保留 SQLite/config/完整审计报告")
    result.add_argument("--version", action="version", version="%(prog)s " + VERSION)
    return result


def main(argv=None):
    options = parser().parse_args(argv)
    if options.in_place and not options.resume:
        log("[ERROR] --in_place 需要同时使用 --resume")
        return 2
    try:
        execute(options)
        return 0
    except (Exception, KeyboardInterrupt) as error:
        log("[ERROR] " + str(error))
        return 130 if isinstance(error, KeyboardInterrupt) else 1


if __name__ == "__main__":
    raise SystemExit(main())
