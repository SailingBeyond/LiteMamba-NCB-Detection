#!/usr/bin/env python3
"""Summarize the two fixed five-class tasks without loading PyTorch or weights.

X_plus and Y_minus are never pooled as independent repeats of the same task.
All rates are fractions. Large prediction arrays and curve tables stay on server;
the result core contains complete compact evidence for every committed run.
"""
from __future__ import annotations

import argparse
import csv
from datetime import datetime, timezone
import hashlib
import json
import math
import os
from pathlib import Path
import statistics
import tarfile

VERSION = "1.0.0"
SEEDS = (42, 2024, 2025)
SCENARIOS = {"X_plus": ("A", "T", "C", "G", "X"),
             "Y_minus": ("A", "T", "C", "G", "Y")}
SUMMARY_DIR = "summary"
PROBABILITY_METRICS = ("nll", "brier", "ece_15bins", "macro_roc_auc",
                       "macro_average_precision", "micro_roc_auc", "micro_average_precision")
MAX_FILE_BYTES = 16 * 2 ** 20
MAX_PAYLOAD_BYTES = 128 * 2 ** 20
MAX_CORE_BYTES = 20 * 2 ** 20


def _clean(value):
    if isinstance(value, float) and not math.isfinite(value):
        return None
    if isinstance(value, dict):
        return {str(key): _clean(item) for key, item in value.items()}
    if isinstance(value, (tuple, list)):
        return [_clean(item) for item in value]
    return value


def _read(path, default=None):
    path = Path(path)
    return json.loads(path.read_text(encoding="utf-8")) if path.is_file() else default


def _json(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + ".writing." + str(os.getpid()))
    try:
        with temporary.open("w", encoding="utf-8") as stream:
            json.dump(_clean(value), stream, ensure_ascii=False, indent=2, allow_nan=False)
            stream.write("\n"); stream.flush(); os.fsync(stream.fileno())
        os.replace(temporary, path)
    finally:
        temporary.unlink(missing_ok=True)


def _csv(path, rows, fields):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + ".writing." + str(os.getpid()))
    try:
        with temporary.open("w", encoding="utf-8-sig", newline="") as stream:
            writer = csv.DictWriter(stream, fieldnames=fields, extrasaction="ignore")
            writer.writeheader(); writer.writerows(_clean(rows))
            stream.flush(); os.fsync(stream.fileno())
        os.replace(temporary, path)
    finally:
        temporary.unlink(missing_ok=True)


def _fingerprint(path):
    path = Path(path); before = path.stat(); digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(2 ** 20), b""):
            digest.update(block)
    after = path.stat()
    if (before.st_size, before.st_mtime_ns) != (after.st_size, after.st_mtime_ns):
        raise ValueError("File changed while reading evidence: " + str(path))
    return {"sha256": digest.hexdigest(), "size_bytes": after.st_size}


def _ratio(a, b):
    return a / b if b else None


def _names(value):
    if isinstance(value, dict):
        try:
            return tuple(value[str(k)] if str(k) in value else value[k] for k in range(5)) if len(value) == 5 else ()
        except KeyError:
            return ()
    return tuple(value) if isinstance(value, (list, tuple)) else ()


def _count_metrics(saved, scenario):
    """Recompute classification and binary-NCB rates from saved integer counts."""
    names = SCENARIOS[scenario]; matrix = saved.get("confusion_matrix")
    if (not isinstance(matrix, list) or len(matrix) != 5
            or any(not isinstance(row, list) or len(row) != 5 for row in matrix)
            or any(type(x) is not int or x < 0 for row in matrix for x in row)):
        raise ValueError("Committed metrics require a nonnegative integer 5x5 confusion matrix")
    support = [sum(row) for row in matrix]
    predicted = [sum(row[c] for row in matrix) for c in range(5)]
    total = sum(support)
    if not total or saved.get("n", total) != total:
        raise ValueError("Confusion matrix and sample count disagree")
    classes = []
    for index, name in enumerate(names):
        tp = matrix[index][index]; fp = predicted[index] - tp; fn = support[index] - tp
        tn = total - tp - fp - fn
        classes.append({"class_id": index, "class_name": name, "support": support[index],
                        "predicted_support": predicted[index], "tp": tp, "fp": fp, "fn": fn, "tn": tn,
                        "precision": _ratio(tp, tp + fp), "recall": _ratio(tp, tp + fn),
                        "specificity": _ratio(tn, tn + fp), "fpr": _ratio(fp, tn + fp),
                        "fdr": _ratio(fp, tp + fp), "f1": _ratio(2 * tp, 2 * tp + fp + fn)})
    defined_recall = [row["recall"] for row in classes if row["recall"] is not None]
    canonical = sum(support[:4]); positive = classes[4]
    metrics = {
        "accuracy": sum(matrix[k][k] for k in range(5)) / total,
        "macro_f1": sum(row["f1"] or 0 for row in classes) / 5,
        "balanced_accuracy": statistics.mean(defined_recall) if defined_recall else None,
        "canonical_accuracy": _ratio(sum(matrix[k][k] for k in range(4)), canonical),
        "canonical_false_positive_rate": positive["fpr"],
        "ncb_recall": positive["recall"], "ncb_precision": positive["precision"],
        "ncb_specificity": positive["specificity"], "ncb_fdr": positive["fdr"], "ncb_f1": positive["f1"],
        "ncb_to_canonical_rate": _ratio(sum(matrix[4][:4]), support[4]),
        "target_recall_micro": positive["recall"],
        names[4] + "_recall": positive["recall"], names[4] + "_f1": positive["f1"],
    }
    for key in ("accuracy", "macro_f1", "balanced_accuracy"):
        if saved.get(key) is not None and not math.isclose(float(saved[key]), metrics[key], rel_tol=1e-9, abs_tol=1e-10):
            raise ValueError("Saved metric differs from integer confusion counts: " + key)
    for key in PROBABILITY_METRICS:
        value = saved.get(key)
        metrics[key] = float(value) if isinstance(value, (int, float)) and math.isfinite(value) else None
    return metrics, classes, matrix, total


def _aggregate(rows):
    groups = {}
    for row in rows:
        key = (row["scenario"], row["split"], row["probability_type"], row["metric"])
        groups.setdefault(key, []).append(row)
    output = []
    for (scenario, split, probability_type, metric), items in sorted(groups.items()):
        if len({r["seed"] for r in items}) != len(items):
            raise ValueError("Duplicate seed in scenario-specific aggregate")
        defined = [item for item in items if item["value"] is not None]
        values = [item["value"] for item in defined]
        output.append({"scenario": scenario, "split": split, "probability_type": probability_type, "metric": metric,
                       "expected_seeds": len(SEEDS), "available_seeds": ",".join(str(item["seed"]) for item in items),
                       "defined_seeds": ",".join(str(item["seed"]) for item in defined), "n_available": len(items),
                       "n_defined": len(values), "mean": statistics.mean(values) if values else None,
                       "sample_sd": statistics.stdev(values) if len(values) > 1 else None,
                       "status": "complete_n3" if len(items) == len(SEEDS) else "partial_n" + str(len(items))})
    return output


def _numerical(scenario, seed, history, result):
    keys = ("optimizer_step_attempts", "optimizer_steps_successful", "optimizer_steps_skipped_amp", "grad_norm_nonfinite_count")
    counts = dict.fromkeys(keys, 0); available = bool(history); finite = True
    for item in history:
        for key in keys:
            value = item.get("train", {}).get(key)
            if type(value) is not int:
                available = False
            else:
                counts[key] += value
        for split in ("train", "val"):
            loss = item.get(split, {}).get("loss")
            finite = finite and isinstance(loss, (int, float)) and math.isfinite(loss)
    selected = next((row for row in history if row.get("epoch") == result.get("best_epoch")), None)
    return {"scenario": scenario, "seed": seed, "epochs_recorded": len(history), "selected_epoch": result.get("best_epoch"),
            "diagnostics_available": available, **{key: value if available else None for key, value in counts.items()},
            "skipped_attempt_fraction": _ratio(counts["optimizer_steps_skipped_amp"], counts["optimizer_step_attempts"]) if available else None,
            "all_train_val_losses_finite": finite if history else None,
            "selected_val_loss": selected.get("val", {}).get("loss") if selected else None,
            "optimizer_count_scope": "GradScaler protective skipped updates remain recorded; no omission of failed update attempts"}


def _frozen_records(output):
    """Require the global six-checkpoint plan before accepting any test metrics."""
    plan_path = output / "evaluation_plan.json"
    plan = _read(plan_path)
    if plan is None:
        return None, {}
    expected = {(scenario, seed) for scenario in SCENARIOS for seed in SEEDS}
    records = {}
    for record in plan.get("models", []):
        key = (record.get("scenario"), record.get("seed"))
        if key in records:
            raise ValueError("Duplicate scenario/seed in global frozen evaluation plan")
        records[key] = record
    if set(records) != expected or plan.get("fixed_seeds") != list(SEEDS) or plan.get("fixed_scenarios") != list(SCENARIOS):
        raise ValueError("Global evaluation plan must freeze all six fixed scenario/seed checkpoints")
    if plan.get("experiment_plan_sha256") != _fingerprint(output / "experiment_plan.json")["sha256"]:
        raise ValueError("Global frozen experiment plan identity changed")
    for (scenario, seed), record in records.items():
        run = output / "scenarios" / scenario / "runs" / ("seed_" + str(seed))
        for field, filename in (("checkpoint_sha256", "best_checkpoint.pt"), ("state_dict_sha256", "best_state_dict.pt"),
                                ("calibration_sha256", "calibration.json"), ("resolved_config_sha256", "resolved_config.json")):
            if record.get(field) != _fingerprint(run / filename)["sha256"]:
                raise ValueError("Frozen model or calibration identity changed: " + scenario + "/seed_" + str(seed) + "/" + filename)
        if record.get("temperature") != _read(run / "calibration.json", {}).get("temperature"):
            raise ValueError("Frozen validation temperature changed")
    return plan, records


def _verify_run(run, result, scenario, seed, test_done, frozen, output):
    if result.get("seed") != seed or result.get("feature_scheme") != "full":
        raise ValueError("Unexpected seed or feature scheme: " + str(run))
    if result.get("litemamba_variant") != "legacy_gatedconv" or result.get("readout") != "target_attention":
        raise ValueError("Committed result is not the fixed final GatedConv")
    if _names(result.get("class_names")) != SCENARIOS[scenario]:
        raise ValueError("Result five-class definition does not match scenario: " + scenario)
    config = _read(run / "resolved_config.json", {})
    if config.get("seed") != seed or _names(config.get("class_names")) != SCENARIOS[scenario]:
        raise ValueError("Resolved configuration class or seed identity conflicts with scenario")
    checkpoint = _fingerprint(run / "best_checkpoint.pt")["sha256"]
    calibration = _read(run / "calibration.json", {})
    t = calibration.get("temperature")
    if (result.get("provenance", {}).get("checkpoint_sha256") != checkpoint
            or calibration.get("fitted_on") != "val" or calibration.get("checkpoint_sha256") != checkpoint
            or not isinstance(t, (float, int)) or not math.isfinite(t) or t <= 0):
        raise ValueError("Calibration is not tied to this checkpoint and validation data")
    if test_done:
        record = frozen.get((scenario, seed))
        started = _read(output / "test_started.json", {})
        if (not record or record.get("checkpoint_sha256") != checkpoint
                or started.get("evaluation_plan_sha256") != _fingerprint(output / "evaluation_plan.json")["sha256"]):
            raise ValueError("Test metrics are not bound to the globally frozen evaluation plan")
    versions = result.get("metrics_by_split", {})
    if not test_done and "test" in versions:
        raise ValueError("Uncommitted test metrics appear in a training-only result")
    for split in ("train", "val") + (("test",) if test_done else ()):
        marker = _read(run / split / "export_complete.json", {})
        if marker.get("checkpoint_sha256") != checkpoint or marker.get("temperature") != t or marker.get("split") != split:
            raise ValueError("Missing or mismatched committed export marker: " + str(run / split))
        if marker.get("metrics") != versions.get(split) or _read(run / split / "prediction_metrics.json") != versions.get(split):
            raise ValueError("Prediction metric identity conflicts with committed results: " + str(run / split))
    return checkpoint, calibration


def _plots(summary_dir, scenario, learning, confusion, complete_seeds):
    stems = (scenario + "_learning_curves", scenario + "_confusion_val", scenario + "_confusion_test")
    for stem in stems:
        for extension in ("png", "svg"):
            (summary_dir / (stem + "." + extension)).unlink(missing_ok=True)
    if not complete_seeds:
        return []
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    plt.rcParams.update({"font.family": "DejaVu Sans", "svg.fonttype": "none", "font.size": 10,
                         "axes.spines.top": False, "axes.spines.right": False})
    files = []
    rows = [row for row in learning if row["scenario"] == scenario]
    if rows:
        fig, axes = plt.subplots(1, 2, figsize=(10.2, 3.8))
        for seed in complete_seeds:
            for ax, key, label in ((axes[0], "val_loss", "Validation loss"), (axes[1], "val_macro_f1", "Validation Macro-F1 (%)")):
                usable = [r for r in rows if r["seed"] == seed and r.get(key) is not None]
                if usable:
                    ax.plot([r["epoch"] for r in usable], [r[key] * (100 if key.endswith("f1") else 1) for r in usable], label="Seed " + str(seed))
                ax.set(xlabel="Epoch", ylabel=label); ax.grid(alpha=.15)
                if usable:
                    ax.legend(fontsize=8)
        fig.suptitle(scenario + " — fixed GatedConv training/validation")
        fig.tight_layout()
        for ext in ("png", "svg"):
            name = stems[0] + "." + ext; fig.savefig(summary_dir / name, bbox_inches="tight", dpi=180); files.append(name)
        plt.close(fig)
    for split in ("val", "test"):
        data = [r for r in confusion if r["scenario"] == scenario and r["split"] == split and r["probability_type"] == "raw"]
        seeds = sorted({r["seed"] for r in data})
        if not seeds:
            continue
        matrix = [[0.0] * 5 for _ in range(5)]
        for row in data:
            matrix[row["true_label"]][row["predicted_label"]] += (row["row_fraction"] or 0) * 100 / len(seeds)
        fig, ax = plt.subplots(figsize=(5.7, 4.8)); image = ax.imshow(matrix, cmap="Blues", vmin=0, vmax=100)
        for i in range(5):
            for j in range(5):
                ax.text(j, i, format(matrix[i][j], ".1f"), ha="center", va="center", color="white" if matrix[i][j] > 60 else "#183543", fontsize=9)
        ax.set_xticks(range(5), SCENARIOS[scenario]); ax.set_yticks(range(5), SCENARIOS[scenario])
        ax.set(xlabel="Predicted class", ylabel="True class", title=scenario + ": " + split + "; n=" + str(len(seeds)) + " seed(s)")
        fig.colorbar(image, ax=ax, label="Within-class windows (%)"); fig.tight_layout()
        for ext in ("png", "svg"):
            name = scenario + "_confusion_" + split + "." + ext
            fig.savefig(summary_dir / name, bbox_inches="tight", dpi=180); files.append(name)
        plt.close(fig)
    return files


def _include_output(path, root, committed_splits):
    relative = path.relative_to(root); parts = relative.parts; name = path.name.lower()
    if path.is_symlink() or any(p.startswith(".") or p == "__pycache__" for p in parts):
        return False
    if any(p in ("feature_cache", "cache", "failed_before_checkpoint", "epoch_predictions") for p in parts):
        return False
    if any(t in name for t in (".writing", ".tmp")) or name in ("manifest.jsonl", "metadata.jsonl", "read_splits.tsv", "result_package_manifest.json"):
        return False
    if path.suffix.lower() in (".pt", ".pth", ".npy", ".npz", ".sqlite", ".db", ".gz", ".zip", ".tar", ".tsv"):
        return False
    if "prepared_data" in parts:
        return path.suffix.lower() == ".json" or (path.suffix.lower() == ".csv" and any(token in name for token in ("count", "audit")))
    if len(parts) >= 6 and parts[0] == "scenarios" and parts[2] == "runs" and parts[4] in ("train", "val", "test"):
        try:
            key = (parts[1], int(parts[3].removeprefix("seed_")), parts[4])
        except ValueError:
            return False
        if key not in committed_splits:
            return False
        if path.suffix.lower() in (".png", ".svg", ".pdf"):
            return False
        if name in ("roc_curve.csv", "pr_curve.csv"):
            return False
    if name.startswith("predictions."):
        return False
    return path.suffix.lower() in (".json", ".csv", ".jsonl", ".md", ".txt", ".py", ".log", ".png", ".svg", ".sh", ".yaml", ".yml")


def _include_code(path, root):
    parts = path.relative_to(root).parts
    if path.is_symlink() or any(p.startswith(".") or p in ("__pycache__", "node_modules", "tests", "validation") for p in parts):
        return False
    return path.suffix.lower() in (".py", ".sh", ".json", ".csv", ".md", ".txt", ".fasta", ".yml", ".yaml")


def _package(output, package_dir, name, summary, committed_splits):
    selected = {}
    for path in sorted(output.rglob("*")):
        if path.is_file() and _include_output(path, output, committed_splits):
            selected[str(path.relative_to(output))] = path
    if package_dir is not None:
        package_dir = Path(package_dir).resolve()
        for path in sorted(package_dir.rglob("*")):
            if path.is_file() and not path.is_relative_to(output) and _include_code(path, package_dir):
                selected["code/" + str(path.relative_to(package_dir))] = path
    fingerprints = {name: _fingerprint(path) for name, path in selected.items()}
    too_large = [name for name, r in fingerprints.items() if r["size_bytes"] > MAX_FILE_BYTES]
    if too_large or sum(r["size_bytes"] for r in fingerprints.values()) > MAX_PAYLOAD_BYTES:
        raise ValueError("Compact package size budget exceeded; committed training remains intact. Re-run summary after inspecting these evidence files: " + ", ".join(too_large[:8]))
    manifest = {"status": summary["status"], "package_version": VERSION, "stage": summary["stage"],
                "fixed_scenarios": list(SCENARIOS), "fixed_seeds": list(SEEDS),
                "completed_training_jobs": summary["completed_training_jobs"], "test_evaluated_jobs": summary["test_evaluated_jobs"],
                "file_count": len(selected), "files": fingerprints,
                "excluded_heavy_files": ["weights", "tensor arrays", "epoch logits", "per-window predictions", "cache", "prepared rows", "ROC/PR threshold-point tables", "per-run redundant plots"],
                "server_retention": "All omitted predictions, raw/calibrated probability arrays, logits, embeddings, attention, ROC/PR tables and checkpoints remain on the server for reanalysis without inference.",
                "size_limits_bytes": {"per_file": MAX_FILE_BYTES, "uncompressed_payload": MAX_PAYLOAD_BYTES, "compressed_core": MAX_CORE_BYTES},
                "policy": "No seed/performance selection and no evidence truncation; only committed split exports included. Two tasks are summarized separately."}
    marker = output / "RESULT_PACKAGE_MANIFEST.json"; _json(marker, manifest)
    destination = output / name; temporary = destination.with_name(destination.name + ".writing." + str(os.getpid()))
    try:
        with tarfile.open(temporary, "w:gz", compresslevel=6) as archive:
            archive.add(marker, arcname=marker.name, recursive=False)
            for relative, path in selected.items():
                archive.add(path, arcname=relative, recursive=False)
                if _fingerprint(path) != fingerprints[relative]:
                    raise ValueError("File changed during packaging: " + str(path))
        if temporary.stat().st_size > MAX_CORE_BYTES:
            raise ValueError("Compressed core exceeds 20MiB; committed results are retained. Inspect evidence size and re-run --phase summary; no inference is needed.")
        os.replace(temporary, destination)
    finally:
        temporary.unlink(missing_ok=True)
    return destination


def summarize_and_package(output_dir, package_dir=None):
    output = Path(output_dir).resolve(); output.mkdir(parents=True, exist_ok=True)
    summary_dir = output / SUMMARY_DIR; summary_dir.mkdir(exist_ok=True)
    frozen_plan, frozen = _frozen_records(output)
    status_rows, metric_rows, class_rows, confusion_rows, learning_rows, diagnostics, runtime_rows, uncertainty_rows = ([] for _ in range(8))
    completed = {scenario: [] for scenario in SCENARIOS}; tested = {scenario: [] for scenario in SCENARIOS}; committed_splits = set()
    for scenario in SCENARIOS:
        for seed in SEEDS:
            run = output / "scenarios" / scenario / "runs" / ("seed_" + str(seed))
            result = _read(run / "results.json", {}); status = result.get("status", "not_started")
            finished = status in ("train_complete", "complete"); test_done = finished and result.get("test_evaluated") is True
            if status == "complete" and not test_done:
                raise ValueError("A complete evaluation result must explicitly record test_evaluated=true")
            if not finished and result.get("test_evaluated"):
                raise ValueError("An incomplete training run cannot have committed test evaluation")
            checkpoint, calibration = (None, {})
            if finished:
                checkpoint, calibration = _verify_run(run, result, scenario, seed, test_done, frozen, output)
                completed[scenario].append(seed)
                if test_done:
                    tested[scenario].append(seed)
            status_rows.append({"scenario": scenario, "seed": seed, "run_dir": str(run), "status": status,
                                "training_complete": finished, "test_evaluated": test_done,
                                "selected_epoch": result.get("best_epoch"), "temperature": calibration.get("temperature"),
                                "checkpoint_sha256": checkpoint,
                                "uncommitted_test_artifacts_present": not test_done and (run / "test").is_dir() and any((run / "test").iterdir())})
            if not finished:
                continue
            for split in ("train", "val") + (("test",) if test_done else ()):
                versions = result.get("metrics_by_split", {}).get(split)
                if not isinstance(versions, dict) or not all(v in versions for v in ("raw", "calibrated")):
                    raise ValueError("Committed result missing raw/calibrated metrics: " + str(run / split))
                committed_splits.add((scenario, seed, split)); matrices = []
                for probability_type in ("raw", "calibrated"):
                    values, per_class, matrix, count = _count_metrics(versions[probability_type], scenario); matrices.append(matrix)
                    for metric, value in values.items():
                        metric_rows.append({"scenario": scenario, "seed": seed, "split": split, "probability_type": probability_type,
                                            "n": count, "metric": metric, "value": value,
                                            "unit": "native loss" if metric in ("nll", "brier") else "fraction", "checkpoint_sha256": checkpoint})
                    for row in per_class:
                        class_rows.append({"scenario": scenario, "seed": seed, "split": split, "probability_type": probability_type, **row})
                    for i, row in enumerate(matrix):
                        for j, value in enumerate(row):
                            confusion_rows.append({"scenario": scenario, "seed": seed, "split": split, "probability_type": probability_type,
                                                   "true_label": i, "true_name": SCENARIOS[scenario][i], "predicted_label": j,
                                                   "predicted_name": SCENARIOS[scenario][j], "count": value, "row_fraction": _ratio(value, sum(row))})
                if matrices[0] != matrices[1]:
                    raise ValueError("A positive validation temperature unexpectedly changed argmax")
                analysis = _read(run / split / "analysis/metrics.json", {})
                if analysis and _names(analysis.get("class_definition")) != SCENARIOS[scenario]:
                    raise ValueError("Analysis class definition conflicts with scenario")
                for probability_type, uncertainty in analysis.get("uncertainty", {}).items():
                    if not isinstance(uncertainty, dict):
                        continue
                    intervals = dict(uncertainty.get("intervals", {}))
                    for alias, key in (("canonical_false_positive_rate", "class_4_fpr"), ("ncb_recall", "class_4_recall"),
                                       ("ncb_precision", "class_4_precision"), ("ncb_f1", "class_4_f1")):
                        if key in intervals:
                            intervals[alias] = intervals[key]
                    for metric, bounds in intervals.items():
                        uncertainty_rows.append({"scenario": scenario, "seed": seed, "split": split, "probability_type": probability_type,
                                                 "metric": metric, "ci_low": bounds.get("ci_low"), "ci_high": bounds.get("ci_high"),
                                                 "valid_replicates": bounds.get("valid_replicates"), "cluster_count": uncertainty.get("cluster_count"),
                                                 "replicates": uncertainty.get("replicates"), "method": uncertainty.get("method")})
            history_path = run / "epoch_metrics.jsonl"
            history = [json.loads(line) for line in history_path.read_text(encoding="utf-8").splitlines() if line.strip()] if history_path.is_file() else []
            for item in history:
                learning_rows.append({"scenario": scenario, "seed": seed, "epoch": item["epoch"], "lr": item.get("lr"),
                                      "train_loss": item.get("train", {}).get("loss"), "val_loss": item.get("val", {}).get("loss"),
                                      "train_macro_f1": item.get("train", {}).get("macro_f1"), "val_macro_f1": item.get("val", {}).get("macro_f1")})
            diagnostics.append(_numerical(scenario, seed, history, result)); runtime_rows.append({"scenario": scenario, "seed": seed, **result.get("runtime", {})})
    all_training = all(completed[s] == list(SEEDS) for s in SCENARIOS)
    all_test = all(tested[s] == list(SEEDS) for s in SCENARIOS)
    if all_test:
        stage, status, stem = "evaluation", "evaluation_complete", "results"
    elif all_training and not any(tested.values()):
        stage, status, stem = "training", "training_complete", "training"
    elif all(completed[s] == [42] for s in SCENARIOS) and not any(tested.values()):
        stage, status, stem = "pilot", "pilot_complete", "pilot"
    else:
        stage, status, stem = "partial", "partial", "partial"
    aggregates = _aggregate(metric_rows)
    _csv(summary_dir / "run_status.csv", status_rows, ["scenario", "seed", "run_dir", "status", "training_complete", "test_evaluated", "selected_epoch", "temperature", "checkpoint_sha256", "uncommitted_test_artifacts_present"])
    _csv(summary_dir / "metrics_per_seed.csv", metric_rows, ["scenario", "seed", "split", "probability_type", "n", "metric", "value", "unit", "checkpoint_sha256"])
    _csv(summary_dir / "mean_sd.csv", aggregates, ["scenario", "split", "probability_type", "metric", "expected_seeds", "available_seeds", "defined_seeds", "n_available", "n_defined", "mean", "sample_sd", "status"])
    _csv(summary_dir / "class_metrics.csv", class_rows, ["scenario", "seed", "split", "probability_type", "class_id", "class_name", "support", "predicted_support", "tp", "fp", "fn", "tn", "precision", "recall", "specificity", "fpr", "fdr", "f1"])
    _csv(summary_dir / "confusion_counts.csv", confusion_rows, ["scenario", "seed", "split", "probability_type", "true_label", "true_name", "predicted_label", "predicted_name", "count", "row_fraction"])
    _csv(summary_dir / "learning_curves.csv", learning_rows, ["scenario", "seed", "epoch", "lr", "train_loss", "val_loss", "train_macro_f1", "val_macro_f1"])
    _csv(summary_dir / "read_cluster_intervals.csv", uncertainty_rows, ["scenario", "seed", "split", "probability_type", "metric", "ci_low", "ci_high", "valid_replicates", "cluster_count", "replicates", "method"])
    _csv(summary_dir / "numerical_diagnostics.csv", diagnostics, list(dict.fromkeys(k for row in diagnostics for k in row)) or ["scenario", "seed"])
    _csv(summary_dir / "runtime.csv", runtime_rows, list(dict.fromkeys(k for row in runtime_rows for k in row)) or ["scenario", "seed"])
    figures = []
    for scenario in SCENARIOS:
        figures.extend(_plots(summary_dir, scenario, learning_rows, confusion_rows, completed[scenario]))
    complete_jobs = [{"scenario": scenario, "seed": seed} for scenario in SCENARIOS for seed in completed[scenario]]
    test_jobs = [{"scenario": scenario, "seed": seed} for scenario in SCENARIOS for seed in tested[scenario]]
    summary = {"version": VERSION, "status": status, "stage": stage, "created_at": datetime.now(timezone.utc).isoformat(),
               "fixed_scenarios": list(SCENARIOS), "fixed_seeds": list(SEEDS), "class_names_by_scenario": SCENARIOS,
               "completed_training_jobs": complete_jobs, "test_evaluated_jobs": test_jobs,
               "completed_training_seeds": completed, "test_evaluated_seeds": tested,
               "test_evaluation_status": "all_six_complete" if all_test else "partial" if test_jobs else "not_run",
               "all_training_complete": all_training, "all_test_complete": all_test,
               "architecture": "Final GatedConv, legacy_gatedconv + target_attention; classifier output changed to five; no SSM claim",
               "scope": "Supplementary read-held-out same-source/same-direction/context/reference-assignment matched controls; two separately trained tasks",
               "probability_metrics": "Saved probability analysis; classification and binary-NCB rates independently recomputed from saved confusion counts",
               "missing_values": "Undefined denominators remain missing; sample SD is null with one seed; two scenarios never pooled",
               "runs": status_rows, "mean_sd": aggregates, "figures": figures,
               "limitations": ["Only fixed X/+43 and Y/−34 matched groups; not all 1,024 contexts and not 77 independent physical templates.",
                               "Natural and NCB sites differ; BAM reference assignment is a proxy, not same-site replacement or batch/template holdout.",
                               "No within-direction X-versus-Y discrimination is tested by the two five-class tasks.",
                               "Read-cluster intervals retain all windows of each sampled read; three seeds share a fixed test set.",
                               "New controlled retraining does not show that the earlier six-class model used no technical confounders.",
                               "Training-only cores contain no committed test performance; raw/calibrated are not independent repeated trainings."]}
    summary_path = summary_dir / "summary.json"; _json(summary_path, summary)
    core = _package(output, package_dir, "cplx_direction_5class_" + stem + "_core_v1_0_0.tar.gz", summary, committed_splits)
    return {"summary_path": str(summary_path), "core_path": str(core), "status": status, "stage": stage,
            "completed_training_jobs": complete_jobs, "test_evaluated_jobs": test_jobs,
            "completed_training_seeds": completed, "test_evaluated_seeds": tested}


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output_dir", required=True)
    parser.add_argument("--package_dir", default=str(Path(__file__).resolve().parent))
    args = parser.parse_args(argv)
    print(json.dumps(summarize_and_package(args.output_dir, args.package_dir), ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
