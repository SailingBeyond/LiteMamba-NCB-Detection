#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Run the binary benchmark queue, then freeze and evaluate the whole queue.

Only the sibling train.py is executed. Successful validation-selected runs are
reused only when their complete requested configuration is unchanged.
"""

import argparse
from collections import Counter
import csv
from decimal import Decimal, InvalidOperation
import hashlib
import importlib
import json
import os
from pathlib import Path
import shlex
import sqlite3
import subprocess
import sys
import time
from datetime import datetime, timezone


HERE = Path(__file__).resolve().parent
MODELS = ("bigru", "transformer", "convmixer", "litemamba")
SCHEMES = (
    "full", "raw_only", "context_only", "responses_only", "statistics_only",
    "no_raw", "no_context", "no_responses", "no_statistics",
)
DEFAULT_DATA_DIR = (
    "/path/to/revision-inputs/POC-XNA_DNA-01_16/POC-XNA_DNA-01_16/"
    "fast5_single_guppy/binary_read_split_721"
)


def utc_now():
    return datetime.now(timezone.utc).isoformat()


def sha256_file(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def load_json(path):
    with Path(path).open("r", encoding="utf-8") as handle:
        return json.load(handle)


def save_json(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + ".tmp")
    with temporary.open("w", encoding="utf-8") as handle:
        json.dump(value, handle, ensure_ascii=False, indent=2, allow_nan=False)
        handle.write("\n")
    temporary.replace(path)


def joint_read_metadata_preflight(data_dir, output_dir, strands=("pos", "neg")):
    """Audit one shared read assignment across the requested strand metadata.

    This deliberately reads no raw-signal feature CSV. Each trainer subsequently
    performs its stricter CSV/metadata alignment and SHA256 preprocessing audit.
    Multiple windows of a read may remain in one split; a read cannot change
    split or PC/XNA reference identity between the pos and neg collections.
    """
    import re

    data_dir, output_dir = Path(data_dir).resolve(), Path(output_dir).resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    database = output_dir / "read_membership.db"
    connection = sqlite3.connect(str(database))
    connection.execute("DROP TABLE IF EXISTS reads")
    connection.execute("DROP TABLE IF EXISTS membership")
    connection.execute("CREATE TABLE reads(read_id TEXT PRIMARY KEY, label INTEGER, chrom TEXT, label_conflict INTEGER, chrom_conflict INTEGER)")
    connection.execute("CREATE TABLE membership(read_id TEXT, strand TEXT, split TEXT, windows INTEGER, PRIMARY KEY(read_id,strand,split))")
    unknown = {"", "unknown", "na", "n/a", "none", "null", "nan"}
    required = {"read_id", "kmer", "label", "chrom"}
    audit = {
        "status": "RUNNING", "checked_strands": list(strands),
        "scope": "joint read identity across strand collections; not a template/source/run holdout claim",
        "raw_feature_csv_read": False, "metadata_sources": {},
        "read_membership_database": str(database),
    }
    counts, row_count = {}, 0
    try:
        for strand in strands:
            if strand not in ("pos", "neg"):
                raise ValueError(f"Unknown strand collection: {strand}")
            for split in ("train", "val", "test"):
                feature = data_dir / f"xna_pc_6mer_features_{strand}_balanced_{split}.csv"
                if not feature.is_file():
                    raise FileNotFoundError(feature)
                candidates = (
                    feature.with_name(feature.stem + "_meta.tsv"),
                    Path(str(feature) + ".meta.tsv"), feature.with_suffix(".meta.tsv"),
                )
                metadata_files = list(dict.fromkeys(path for path in candidates if path.is_file()))
                if len(metadata_files) != 1:
                    raise ValueError(f"Exactly one metadata file is required for {feature}; found {metadata_files}")
                metadata = metadata_files[0]
                before, digest = metadata.stat(), hashlib.sha256()
                class_counts, unknown_fields = Counter(), Counter()
                read_batch, membership_batch = [], []
                with metadata.open("rb") as binary_stream:
                    def tracked_lines():
                        for line_number, raw_line in enumerate(binary_stream, 1):
                            digest.update(raw_line)
                            if not raw_line.strip():
                                raise ValueError(f"{metadata}:{line_number}: blank metadata row")
                            yield raw_line.decode("utf-8-sig" if line_number == 1 else "utf-8")

                    reader = csv.DictReader(tracked_lines(), delimiter="\t")
                    header = [name.strip() for name in (reader.fieldnames or [])]
                    if len(set(header)) != len(header) or not required.issubset(header):
                        raise ValueError(f"{metadata}: unique metadata columns {sorted(required)} are required")
                    reader.fieldnames = header
                    for row in reader:
                        where = f"{metadata}:{reader.line_num}"
                        if None in row or any(value is None for value in row.values()):
                            raise ValueError(f"{where}: malformed metadata row")
                        if any("\n" in value or "\r" in value for value in row.values()):
                            raise ValueError(f"{where}: multiline metadata fields are unsupported")
                        read_id, chrom = row["read_id"].strip(), row["chrom"].strip()
                        if read_id.lower() in unknown or chrom.lower() in unknown:
                            raise ValueError(f"{where}: known read_id and chrom are required")
                        try:
                            parsed_label = Decimal(row["label"].strip())
                        except (InvalidOperation, ValueError) as exc:
                            raise ValueError(f"{where}: invalid binary label") from exc
                        if not parsed_label.is_finite() or parsed_label not in (0, 1):
                            raise ValueError(f"{where}: labels must be exactly binary 0/1")
                        label = int(parsed_label)
                        sequence = row["kmer"].strip().upper()
                        if len(sequence) != 6 or any(
                            base not in "ATCGN" and not (position == 3 and base in "XY")
                            for position, base in enumerate(sequence)
                        ):
                            raise ValueError(f"{where}: invalid 6mer; X/Y may occur only at the fourth target base")
                        if "bam_strand" in row and row["bam_strand"].strip() != ("+" if strand == "pos" else "-"):
                            raise ValueError(f"{where}: bam_strand disagrees with its pos/neg collection")
                        family = re.match(r"^(PC|DNA|XNA)(?:[_-]?\d|$)", chrom, re.I)
                        if family and label != int(family.group(1).upper() == "XNA"):
                            raise ValueError(f"{where}: PC/DNA/XNA reference identity disagrees with its binary label")
                        read_batch.append((read_id, label, chrom, 0, 0))
                        membership_batch.append((read_id, strand, split, 1))
                        class_counts[str(label)] += 1
                        row_count += 1
                        for field in ("run_id", "source_dataset"):
                            unknown_fields[field] += int(row.get(field, "").strip().lower() in unknown)
                        if len(read_batch) >= 5000:
                            _store_joint_read_batch(connection, read_batch, membership_batch)
                            read_batch.clear()
                            membership_batch.clear()
                _store_joint_read_batch(connection, read_batch, membership_batch)
                connection.commit()
                after = metadata.stat()
                if (before.st_size, before.st_mtime_ns) != (after.st_size, after.st_mtime_ns):
                    raise ValueError(f"Metadata changed during the joint read audit: {metadata}")
                if not class_counts:
                    raise ValueError(f"Empty metadata split: {metadata}")
                key = f"{strand}_{split}"
                counts[key] = dict(class_counts)
                audit["metadata_sources"][key] = {
                    "path": str(metadata), "sha256": digest.hexdigest(),
                    "size": after.st_size, "mtime_ns": after.st_mtime_ns,
                    "class_counts": dict(class_counts), "unknown_identity_rows": dict(unknown_fields),
                }
        overlaps = connection.execute(
            "SELECT read_id,GROUP_CONCAT(DISTINCT split) FROM membership GROUP BY read_id HAVING COUNT(DISTINCT split)>1"
        ).fetchall()
        conflicts = connection.execute(
            "SELECT read_id,label,chrom,label_conflict,chrom_conflict FROM reads WHERE label_conflict=1 OR chrom_conflict=1"
        ).fetchall()
        repeated = connection.execute(
            "SELECT COUNT(*) FROM (SELECT read_id FROM membership GROUP BY read_id HAVING SUM(windows)>1)"
        ).fetchone()[0]
        audit.update(
            sample_count=row_count, unique_read_count=connection.execute("SELECT COUNT(*) FROM reads").fetchone()[0],
            class_counts=counts, reads_with_multiple_windows=repeated,
            read_cross_split_overlap_count=len(overlaps),
            read_cross_split_overlap_examples=[{"read_id": value[0], "splits": value[1]} for value in overlaps[:20]],
            read_identity_conflict_count=len(conflicts),
            read_identity_conflict_examples=[dict(zip(("read_id", "first_label", "first_chrom", "label_conflict", "chrom_conflict"), value)) for value in conflicts[:20]],
        )
        if overlaps or conflicts:
            raise ValueError(
                f"Joint pos/neg read audit rejected: {len(overlaps)} reads cross splits; "
                f"{len(conflicts)} reads have conflicting binary/reference identity. See {output_dir / 'joint_read_audit.json'}"
            )
        audit["status"] = "PASS"
        save_json(output_dir / "joint_read_audit.json", audit)
        print(f"联合 read 审计通过：{row_count:,} 行，{audit['unique_read_count']:,} 条 read，跨集合交集为 0。", flush=True)
        return audit
    except BaseException as exc:
        connection.commit()
        audit.update(status="REJECTED", error=f"{type(exc).__name__}: {exc}")
        save_json(output_dir / "joint_read_audit.json", audit)
        raise
    finally:
        connection.close()


def _store_joint_read_batch(connection, reads, memberships):
    connection.executemany(
        "INSERT INTO reads VALUES (?,?,?,?,?) ON CONFLICT(read_id) DO UPDATE SET "
        "label_conflict=label_conflict OR label!=excluded.label, "
        "chrom_conflict=chrom_conflict OR chrom!=excluded.chrom", reads,
    )
    connection.executemany(
        "INSERT INTO membership VALUES (?,?,?,?) ON CONFLICT(read_id,strand,split) "
        "DO UPDATE SET windows=windows+1", memberships,
    )


def positive_int(text):
    value = int(text)
    if value <= 0:
        raise argparse.ArgumentTypeError("必须为正整数")
    return value


def parse_args(argv=None):
    parser = argparse.ArgumentParser(
        description="二分类四模型批量训练；全队列完成后固定计划并统一测试。",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("--data_dir", default=DEFAULT_DATA_DIR)
    parser.add_argument("--output_dir", default="./binary_training_runs")
    parser.add_argument("--models", nargs="+", choices=MODELS, default=list(MODELS))
    parser.add_argument("--seeds", nargs="+", type=int, default=[42, 43, 44])
    parser.add_argument("--strands", nargs="+", choices=["pos", "neg"], default=["pos", "neg"])
    parser.add_argument("--feature_schemes", nargs="+", choices=SCHEMES, default=["full"])
    parser.add_argument("--phase", choices=["all", "train", "evaluate"], default="all")
    parser.add_argument("--epochs", type=positive_int, default=200)
    parser.add_argument("--patience", type=positive_int, default=30)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--batch_size", type=positive_int, default=256)
    parser.add_argument("--workers", type=int, default=4)
    parser.add_argument("--raw_signal_len", type=positive_int, default=30)
    parser.add_argument("--dwell_divisor", type=float, default=60.0)
    parser.add_argument("--quality_divisor", type=float, default=50.0)
    parser.add_argument("--accum_steps", type=positive_int, default=1)
    parser.add_argument("--device", default="auto")
    parser.add_argument("--model_dim", type=positive_int, default=96)
    parser.add_argument("--num_blocks", type=positive_int, default=4)
    parser.add_argument("--state_dim", type=positive_int, default=16)
    parser.add_argument("--litemamba_variant", choices=["selective_ssm", "legacy_gatedconv"], default="selective_ssm")
    parser.add_argument("--readout", choices=["target_attention", "legacy_last_attention"], default="target_attention")
    parser.add_argument("--bootstrap", type=int, default=200)
    parser.add_argument("--no_plots", action="store_true")
    parser.add_argument("--no_embeddings", action="store_true")
    parser.add_argument("--embedding_dtype", choices=["float16", "float32"], default="float32")
    parser.add_argument("--external_manifest", default=None)
    parser.add_argument("--resume_failed", action="store_true", help="配置不变且存在 last checkpoint 时恢复中断训练")
    parser.add_argument("--continue_on_error", action="store_true", help="继续训练剩余实验；只要有失败，就不进入整队测试")
    parser.add_argument("--dry_run", action="store_true", help="只打印并保存计划命令，不启动训练或读取测试结果")
    args = parser.parse_args(argv)
    if args.workers < 0 or args.bootstrap < 0:
        parser.error("workers 和 bootstrap 必须 >= 0")
    if min(args.lr, args.dwell_divisor, args.quality_divisor) <= 0:
        parser.error("lr、dwell_divisor、quality_divisor 必须 > 0")
    for key in ("models", "seeds", "strands", "feature_schemes"):
        values = getattr(args, key)
        if len(set(values)) != len(values):
            parser.error("--" + key + " 不允许重复项目")
    return args


def build_requested_config(args, model, seed, strand, scheme, trainer):
    # Runtime flags that affect results are recorded as well as architecture and
    # optimization choices. Changing them cannot silently reuse an old run.
    names = (
        "epochs", "patience", "lr", "batch_size", "workers", "raw_signal_len",
        "dwell_divisor", "quality_divisor", "accum_steps", "device", "model_dim",
        "num_blocks", "state_dim", "litemamba_variant", "readout", "bootstrap",
        "no_plots", "no_embeddings", "embedding_dtype",
    )
    config = {key: getattr(args, key) for key in names}
    config.update({
        "data_dir": str(Path(args.data_dir).expanduser().resolve()),
        "model_type": model, "seed": seed, "strand": strand,
        "feature_scheme": scheme, "trainer": str(trainer),
        "trainer_sha256": sha256_file(trainer),
        "external_manifest": (
            str(Path(args.external_manifest).expanduser().resolve())
            if args.external_manifest else None
        ),
    })
    data_dir = Path(config["data_dir"])
    config["data_file_stats"] = {
        str(path.name): {"size": path.stat().st_size, "mtime_ns": path.stat().st_mtime_ns}
        for path in sorted(data_dir.iterdir())
        if path.is_file() and path.suffix.lower() in (".csv", ".tsv")
    } if data_dir.is_dir() else {}
    if args.external_manifest:
        config["external_manifest_sha256"] = sha256_file(config["external_manifest"])
    # The full training suite records full data manifests. Source hashes here
    # ensure an edited model/data loader cannot be mistaken for the same queue.
    config["source_sha256"] = {
        name: sha256_file(HERE / name)
        for name in ("model_benchmark.py", "config_19.py", "IterDataset.py")
        if (HERE / name).is_file()
    }
    return config


def build_command(config, run_dir, mode, resume=False):
    command = [sys.executable, "-u", str(HERE / "train.py"), "--mode", mode]
    keys = (
        "data_dir", "model_type", "seed", "strand", "feature_scheme", "epochs",
        "patience", "lr", "batch_size", "workers", "raw_signal_len",
        "dwell_divisor", "quality_divisor", "accum_steps", "device", "model_dim",
        "num_blocks", "state_dim", "litemamba_variant", "readout", "bootstrap", "embedding_dtype",
    )
    for key in keys:
        command.extend(["--" + key, str(config[key])])
    command.extend(["--output_dir", str(run_dir)])
    for key in ("no_plots", "no_embeddings"):
        if config[key]:
            command.append("--" + key)
    if mode == "evaluate" and config.get("external_manifest"):
        command.extend(["--external_manifest", config["external_manifest"]])
    if mode == "train":
        command.append("--defer_test")
        if resume:
            command.append("--resume")
    return command


def results_status(run_dir):
    path = Path(run_dir) / "results.json"
    if not path.is_file():
        return None
    return load_json(path).get("status")


def find_checkpoint(run_dir, kind):
    # Accommodate the suite's checkpoints subdirectory and flat layouts without
    # guessing success from a checkpoint's mere existence.
    names = [
        Path(run_dir) / "checkpoints" / f"{kind}_checkpoint.pt",
        Path(run_dir) / f"{kind}_checkpoint.pt",
        Path(run_dir) / "checkpoints" / f"{kind}.pt",
    ]
    return next((path for path in names if path.is_file()), None)


def verify_existing_config(item):
    path = Path(item["run_dir"]) / "queue_requested_config.json"
    if path.is_file():
        if load_json(path) != item["requested_config"]:
            raise RuntimeError(
                f"配置与已有实验不一致：{item['run_dir']}。请换 output_dir；不会复用或覆盖旧实验。"
            )
        return True
    run_dir = Path(item["run_dir"])
    if run_dir.exists() and any(run_dir.iterdir()):
        raise RuntimeError(
            f"目录已有内容但缺少 queue_requested_config.json：{run_dir}。请使用新目录。"
        )
    return False


def execute(command, log_path, commands_path, item, stage):
    command_record = {
        "started_at": utc_now(), "run_id": item["run_id"], "stage": stage,
        "argv": command, "command": shlex.join(command), "log": str(log_path),
    }
    with commands_path.open("a", encoding="utf-8") as handle:
        handle.write(json.dumps(command_record, ensure_ascii=False) + "\n")
    print("\n" + command_record["command"], flush=True)
    started = time.monotonic()
    log_path.parent.mkdir(parents=True, exist_ok=True)
    with log_path.open("a", encoding="utf-8", buffering=1) as log:
        log.write("\n" + json.dumps(command_record, ensure_ascii=False) + "\n")
        process = subprocess.Popen(
            command, cwd=str(HERE), stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT, text=True, encoding="utf-8", errors="replace",
            bufsize=1, env={**os.environ, "PYTHONUNBUFFERED": "1"},
        )
        try:
            for line in process.stdout:
                print(line, end="", flush=True)
                log.write(line)
            code = process.wait()
        except BaseException:
            process.terminate()
            try:
                process.wait(timeout=10)
            except subprocess.TimeoutExpired:
                process.kill()
                process.wait()
            raise
    return {"returncode": code, "elapsed_seconds": time.monotonic() - started, "finished_at": utc_now()}


def freeze_evaluation_plan(items, output_dir):
    experiments = []
    for item in items:
        status = results_status(item["run_dir"])
        if status not in ("train_complete", "complete"):
            raise RuntimeError(f"实验尚未完成训练：{item['run_dir']} (status={status!r})")
        checkpoint = find_checkpoint(item["run_dir"], "best")
        if checkpoint is None:
            raise RuntimeError(f"缺少验证集选出的 best checkpoint：{item['run_dir']}")
        experiments.append({
            "run_id": item["run_id"], "run_dir": item["run_dir"],
            "requested_config": item["requested_config"],
            "checkpoint": str(checkpoint), "checkpoint_sha256": sha256_file(checkpoint),
        })
    frozen = {"policy": "validation_only_selection_then_fixed_queue_evaluation", "experiments": experiments}
    path = output_dir / "evaluation_plan.json"
    if path.exists():
        previous = load_json(path)
        previous.pop("frozen_at", None)
        if previous != frozen:
            raise RuntimeError("evaluation_plan.json 已固定且与当前队列不一致；请换 output_dir，勿在看过测试后静默改队列。")
    else:
        save_json(path, {"frozen_at": utc_now(), **frozen})
    return path


def main(argv=None):
    args = parse_args(argv)
    trainer = HERE / "train.py"
    if not trainer.is_file():
        raise FileNotFoundError(f"请将完整代码包放在同一目录，缺少：{trainer}")
    output_dir = Path(args.output_dir).expanduser().resolve()
    data_dir = Path(args.data_dir).expanduser().resolve()
    if not args.dry_run and not data_dir.is_dir():
        raise FileNotFoundError(f"数据目录不存在：{data_dir}；可用 --data_dir 指定已通过检查的 7:2:1 输出目录。")
    if output_dir == data_dir or output_dir in data_dir.parents:
        raise ValueError("output_dir 不得等于 data_dir 或是其父目录")
    output_dir.mkdir(parents=True, exist_ok=True)
    items = []
    for strand in args.strands:
        for scheme in args.feature_schemes:
            for model in args.models:
                variant = args.litemamba_variant if model == "litemamba" else "backbone"
                for seed in args.seeds:
                    run_id = f"{strand}/{scheme}/{model}/{variant}_{args.readout}/seed_{seed}"
                    run_dir = output_dir / run_id
                    config = build_requested_config(args, model, seed, strand, scheme, trainer)
                    items.append({"run_id": run_id, "run_dir": str(run_dir), "requested_config": config})
    manifest = {"created_at": utc_now(), "phase": args.phase, "dry_run": args.dry_run, "experiments": items}
    manifest_path = output_dir / "queue_manifest.json"
    commands_path = output_dir / "commands.jsonl"
    if args.dry_run:
        save_json(output_dir / "dry_run_manifest.json", manifest)
        phases = ["train", "evaluate"] if args.phase == "all" else [args.phase]
        for phase in phases:
            for item in items:
                print(shlex.join(build_command(item["requested_config"], item["run_dir"], phase)))
        print(f"\n计划：{len(items)} 个实验；未启动训练和测试。")
        return 0
    joint_read_metadata_preflight(data_dir, output_dir, strands=tuple(args.strands))
    for item in items:
        existed = verify_existing_config(item)
        if args.phase == "evaluate" and not existed:
            raise RuntimeError(f"evaluate 阶段要求已有本运行器的训练配置：{item['run_dir']}")
    save_json(manifest_path, manifest)
    print(f"队列包含 {len(items)} 个实验。训练只监控 validation；所有实验训练成功后才进入测试。", flush=True)
    failures = []
    if args.phase in ("all", "train"):
        for index, item in enumerate(items, 1):
            status = results_status(item["run_dir"])
            if status in ("train_complete", "complete"):
                if find_checkpoint(item["run_dir"], "best") is None:
                    raise RuntimeError(f"完成状态缺少 best checkpoint：{item['run_dir']}")
                item["train_queue_status"] = "skipped_matching_completed_run"
                save_json(manifest_path, manifest)
                continue
            run_dir = Path(item["run_dir"])
            known = (run_dir / "queue_requested_config.json").is_file()
            last_checkpoint = find_checkpoint(run_dir, "last")
            resume = bool(args.resume_failed and last_checkpoint)
            if known and not args.resume_failed:
                raise RuntimeError(f"发现未完成实验：{run_dir}；核查后用 --resume_failed 恢复，或改用新目录。")
            if known and args.resume_failed and last_checkpoint is None:
                raise RuntimeError(f"无法恢复，缺少 last checkpoint：{run_dir}。请核查故障并使用新输出目录。")
            run_dir.mkdir(parents=True, exist_ok=True)
            save_json(run_dir / "queue_requested_config.json", item["requested_config"])
            item["train_queue_status"] = "running"
            save_json(manifest_path, manifest)
            print(f"\n[{index}/{len(items)}] TRAIN {item['run_id']}", flush=True)
            outcome = execute(
                build_command(item["requested_config"], run_dir, "train", resume=resume),
                run_dir / "stdout_train.log", commands_path, item, "train",
            )
            item["train_process"] = outcome
            ok = outcome["returncode"] == 0 and results_status(run_dir) in ("train_complete", "complete")
            item["train_queue_status"] = "complete" if ok else "failed"
            save_json(manifest_path, manifest)
            if not ok:
                failures.append(item["run_id"])
                if not args.continue_on_error:
                    print(f"训练失败，已停止；日志：{run_dir / 'stdout_train.log'}", file=sys.stderr)
                    return 1
    if failures:
        manifest["failed_runs"] = failures
        manifest["evaluation_status"] = "blocked_by_training_failures"
        save_json(manifest_path, manifest)
        print("有训练失败，整队测试未启动。修复并恢复训练后再评估。", file=sys.stderr)
        return 1
    if args.phase in ("all", "evaluate"):
        plan = freeze_evaluation_plan(items, output_dir)
        print(f"固定测试计划：{plan}", flush=True)
        for index, item in enumerate(items, 1):
            if results_status(item["run_dir"]) == "complete":
                item["evaluation_queue_status"] = "skipped_matching_completed_run"
                save_json(manifest_path, manifest)
                continue
            print(f"\n[{index}/{len(items)}] EVALUATE {item['run_id']}", flush=True)
            run_dir = Path(item["run_dir"])
            item["evaluation_queue_status"] = "running"
            save_json(manifest_path, manifest)
            outcome = execute(
                build_command(item["requested_config"], run_dir, "evaluate"),
                run_dir / "stdout_evaluate.log", commands_path, item, "evaluate",
            )
            item["evaluation_process"] = outcome
            ok = outcome["returncode"] == 0 and results_status(run_dir) == "complete"
            item["evaluation_queue_status"] = "complete" if ok else "failed"
            save_json(manifest_path, manifest)
            if not ok:
                failures.append(item["run_id"])
                if not args.continue_on_error:
                    print(f"测试失败，已停止；日志：{run_dir / 'stdout_evaluate.log'}", file=sys.stderr)
                    return 1
    usable = [
        item["run_dir"] for item in items
        if results_status(item["run_dir"]) in ("train_complete", "complete")
    ]
    if usable:
        analyzer = importlib.import_module("analyze_results")
        analyzer.aggregate_runs(usable, output_dir / "aggregate", split="val" if args.phase == "train" else "test")
    manifest["finished_at"] = utc_now()
    manifest["failed_runs"] = failures
    manifest["status"] = "complete" if not failures else "failed"
    save_json(manifest_path, manifest)
    print(f"\n队列完成；聚合结果：{output_dir / 'aggregate'}", flush=True)
    return 1 if failures else 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except KeyboardInterrupt:
        print("\n用户中断；使用相同配置和 --resume_failed 可恢复已有 last checkpoint。", file=sys.stderr)
        raise SystemExit(130)
    except Exception as exc:
        print(f"ERROR: {type(exc).__name__}: {exc}", file=sys.stderr)
        raise SystemExit(1)
