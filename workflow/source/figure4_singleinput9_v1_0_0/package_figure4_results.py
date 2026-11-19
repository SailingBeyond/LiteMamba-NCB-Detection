#!/usr/bin/env python3
"""打包已完成的Figure 4统计、图和溯源，排除大型逐窗预测/缓存/权重。"""
from __future__ import annotations

import argparse
import hashlib
import io
import json
import os
from pathlib import Path
import sys
import tarfile
import time


def _json(path):
    return json.loads(path.read_text(encoding="utf-8"))


def _sha(path):
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(8 * 1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def package_results(output_root, destination=None):
    root = Path(output_root).expanduser().resolve()
    plan = _json(root / "evaluation_plan.json")
    analysis = _json(root / "analysis_data/analysis_manifest.json")
    if analysis.get("status") != "complete" or analysis.get("plan_id") != plan.get("plan_id"):
        raise ValueError("分析未完整完成或不属于当前冻结计划；先运行 --phase analyze")
    target = (Path(destination).expanduser().resolve() if destination else
              root / "figure4_results_for_analysis_v1_1_0.tar.gz")
    if not target.name.endswith(".tar.gz"):
        raise ValueError("结果压缩包文件名必须以 .tar.gz 结尾")
    target.parent.mkdir(parents=True, exist_ok=True)
    files = {}
    def add(path, archive_name=None):
        path = Path(path)
        if path.is_file():
            name = archive_name or path.relative_to(root).as_posix()
            files[name] = path
    for name in ("evaluation_plan.json", "discovered_inputs.csv", "external_identity_audit.json",
                 "corrected_split_audit.json", "reference_correction_provenance.json", "reuse_manifest.json",
                 "queue_state.json", "commands.jsonl", "resource_events.jsonl"):
        add(root / name)
    for directory in ("analysis_data", "figures"):
        for path in sorted((root / directory).rglob("*")):
            if path.is_file() and path.suffix.lower() != ".sqlite":
                add(path)
    run_ids = []
    for run in plan["runs"]:
        run_id = run["run_id"]
        run_ids.append(run_id)
        base = root / "predictions" / run_id
        complete = base / "run_complete.json"
        value = _json(complete)
        if value.get("status") != "complete" or value.get("plan_id") != plan["plan_id"]:
            raise ValueError("预测运行尚未完成：" + run_id)
        if set(value.get("datasets", [])) != {d["dataset_id"] for d in plan["datasets"]}:
            raise ValueError("预测运行的文件集合与计划不一致：" + run_id)
        add(complete)
        for dataset in plan["datasets"]:
            marker = base / dataset["dataset_id"] / "completion.json"
            info = _json(marker)
            if (info.get("status") != "complete" or info.get("plan_id") != plan["plan_id"]
                    or info.get("run_id") != run_id or info.get("dataset_id") != dataset["dataset_id"]
                    or info.get("n_samples") != dataset["n_samples"]):
                raise ValueError("逐文件完成记录不一致：" + str(marker))
            add(marker)
        training = Path(run["run_dir"])
        for name in ("resolved_config.json", "calibration.json", "model_config.json", "results.json", "data_audit.json"):
            add(training / name, "training_run_metadata/" + run_id + "/" + name)
    for path in sorted((root / "resource_profiles").rglob("*.json")):
        add(path)
    for path in sorted((root / "logs").rglob("invocation.json")):
        add(path)
    # 不重复读取巨大的逐窗文件。完整推理/分析阶段已校验其SHA，包中保留完成清单。
    records = {}
    for name, path in sorted(files.items()):
        before = path.stat()
        digest = _sha(path)
        after = path.stat()
        if (before.st_size, before.st_mtime_ns) != (after.st_size, after.st_mtime_ns):
            raise ValueError("打包时文件仍在变化：" + str(path))
        records[name] = {"size_bytes": after.st_size, "sha256": digest,
                         "source_mtime_ns": after.st_mtime_ns}
    manifest = {"status": "complete", "package_kind": "figure4_analysis_and_provenance",
        "package_version": "1.1.0", "plan_id": plan["plan_id"], "run_ids": run_ids,
        "created_at_unix": time.time(), "files": records,
        "included": "All standard analysis tables, figures, frozen input declarations, correction audits, export completion records and available training JSON metadata.",
        "excluded": "Per-window prediction CSV, logits, labels, embeddings, attention, feature caches, read-audit databases, checkpoints and training datasets remain on the server.",
        "verification_scope": "This package hashes each included file; it does not repeat full per-window prediction integrity checks or rerun inference."}
    temporary = target.with_name(target.name + ".partial-" + str(os.getpid()))
    try:
        with tarfile.open(temporary, "w:gz", compresslevel=5) as archive:
            for name, path in sorted(files.items()):
                stat = path.stat()
                expected = records[name]
                if (stat.st_size, stat.st_mtime_ns) != (expected["size_bytes"], expected["source_mtime_ns"]):
                    raise ValueError("打包前文件发生变化：" + str(path))
                # dereference=True 兼容手动使用的分析结果符号链接；归档不含外部链接。
                archive.dereference = True
                archive.add(path, arcname=name, recursive=False)
                after = path.stat()
                if (after.st_size, after.st_mtime_ns) != (stat.st_size, stat.st_mtime_ns):
                    raise ValueError("打包期间文件发生变化：" + str(path))
            data = (json.dumps(manifest, ensure_ascii=False, indent=2, allow_nan=False) + "\n").encode()
            info = tarfile.TarInfo("RESULT_PACKAGE_MANIFEST.json")
            info.size, info.mtime = len(data), int(time.time())
            archive.addfile(info, io.BytesIO(data))
        temporary.replace(target)
    except BaseException:
        temporary.unlink(missing_ok=True)
        raise
    print("[PACKAGED] " + str(target), flush=True)
    return target


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output_root", required=True)
    parser.add_argument("--destination")
    args = parser.parse_args()
    package_results(args.output_root, args.destination)


if __name__ == "__main__":
    try:
        main()
    except Exception as error:
        print(f"[ERROR] {type(error).__name__}: {error}", file=sys.stderr)
        raise SystemExit(1)
