#!/usr/bin/env python3
"""Summarize the two fixed five-class tasks without loading PyTorch or weights.

X and Y are never pooled as independent repeats of the same task.
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

VERSION = "1.0.2"
SEEDS = (42, 2024, 2025)
SCENARIOS = {"X": ("A", "T", "C", "G", "X"), "Y": ("A", "T", "C", "G", "Y")}
MODELS = ("gatedconv", "transformer", "convmixer", "bigru")
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


def _output_policy(result, config):
    """Legacy exports are full; compact records must agree on their omissions."""
    declared = [value["output_policy"] for value in (result, config) if "output_policy" in value]
    if any(policy not in ("full", "compact") for policy in declared) or len(set(declared)) > 1:
        raise ValueError("Output retention policy conflicts with committed configuration")
    return declared[0] if declared else "full"


def _export_splits(policy, test_done):
    return (("val",) if policy == "compact" else ("train", "val")) + (("test",) if test_done else ())


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
        key = (row["scenario"], row["model"], row["split"], row["probability_type"], row["metric"])
        groups.setdefault(key, []).append(row)
    output = []
    for (scenario, model, split, probability_type, metric), items in sorted(groups.items()):
        if len({r["seed"] for r in items}) != len(items):
            raise ValueError("Duplicate seed in scenario-specific aggregate")
        defined = [item for item in items if item["value"] is not None]
        values = [item["value"] for item in defined]
        output.append({"scenario": scenario, "model": model, "split": split, "probability_type": probability_type, "metric": metric,
                       "expected_seeds": len(SEEDS), "available_seeds": ",".join(str(item["seed"]) for item in items),
                       "defined_seeds": ",".join(str(item["seed"]) for item in defined), "n_available": len(items),
                       "n_defined": len(values), "mean": statistics.mean(values) if values else None,
                       "sample_sd": statistics.stdev(values) if len(values) > 1 else None,
                       "status": "complete_n3" if len(items) == len(SEEDS) else "partial_n" + str(len(items))})
    return output


def _numerical(scenario, model, seed, history, result):
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
    return {"scenario": scenario, "model": model, "seed": seed, "epochs_recorded": len(history), "selected_epoch": result.get("best_epoch"),
            "diagnostics_available": available, **{key: value if available else None for key, value in counts.items()},
            "skipped_attempt_fraction": _ratio(counts["optimizer_steps_skipped_amp"], counts["optimizer_step_attempts"]) if available else None,
            "all_train_val_losses_finite": finite if history else None,
            "selected_val_loss": selected.get("val", {}).get("loss") if selected else None,
            "optimizer_count_scope": "GradScaler protective skipped updates remain recorded; no omission of failed update attempts"}


def _frozen_records(output):
    """Require the global 24-checkpoint plan before accepting any test metrics."""
    plan_path = output / "evaluation_plan.json"
    plan = _read(plan_path)
    if plan is None:
        return None, {}
    expected = {(scenario, model, seed) for scenario in SCENARIOS for model in MODELS for seed in SEEDS}
    records = {}
    for record in plan.get("models", []):
        key = (record.get("scenario"), record.get("model"), record.get("seed"))
        if key in records:
            raise ValueError("Duplicate scenario/seed in global frozen evaluation plan")
        records[key] = record
    if set(records) != expected or plan.get("fixed_seeds") != list(SEEDS) or plan.get("fixed_scenarios") != list(SCENARIOS) or plan.get("fixed_models") != list(MODELS):
        raise ValueError("Global evaluation plan must freeze all 24 fixed scenario/model/seed checkpoints")
    if plan.get("experiment_plan_sha256") != _fingerprint(output / "experiment_plan.json")["sha256"]:
        raise ValueError("Global frozen experiment plan identity changed")
    for (scenario, model, seed), record in records.items():
        run = output / "scenarios" / scenario / "runs" / model / ("seed_" + str(seed))
        for field, filename in (("checkpoint_sha256", "best_checkpoint.pt"), ("state_dict_sha256", "best_state_dict.pt"),
                                ("calibration_sha256", "calibration.json"), ("resolved_config_sha256", "resolved_config.json")):
            if record.get(field) != _fingerprint(run / filename)["sha256"]:
                raise ValueError("Frozen model or calibration identity changed: " + scenario + "/seed_" + str(seed) + "/" + filename)
        if record.get("temperature") != _read(run / "calibration.json", {}).get("temperature"):
            raise ValueError("Frozen validation temperature changed")
    return plan, records


def _verify_run(run, result, scenario, model, seed, test_done, frozen, output):
    if result.get("seed") != seed or result.get("feature_scheme") != "full":
        raise ValueError("Unexpected seed or feature scheme: " + str(run))
    expected_type = "litemamba" if model == "gatedconv" else model
    if result.get("model_type") != expected_type:
        raise ValueError("Committed result model does not match its fixed model directory")
    if model == "gatedconv" and (result.get("litemamba_variant") != "legacy_gatedconv" or result.get("readout") != "target_attention"):
        raise ValueError("Committed GatedConv is not the final fixed architecture")
    if _names(result.get("class_names")) != SCENARIOS[scenario]:
        raise ValueError("Result five-class definition does not match scenario: " + scenario)
    config = _read(run / "resolved_config.json", {})
    policy = _output_policy(result, config)
    if config.get("seed") != seed or _names(config.get("class_names")) != SCENARIOS[scenario] or config.get("model_type") != expected_type:
        raise ValueError("Resolved configuration class or seed identity conflicts with scenario")
    checkpoint = _fingerprint(run / "best_checkpoint.pt")["sha256"]
    calibration = _read(run / "calibration.json", {})
    t = calibration.get("temperature")
    if (result.get("provenance", {}).get("checkpoint_sha256") != checkpoint
            or calibration.get("fitted_on") != "val" or calibration.get("checkpoint_sha256") != checkpoint
            or not isinstance(t, (float, int)) or not math.isfinite(t) or t <= 0):
        raise ValueError("Calibration is not tied to this checkpoint and validation data")
    if test_done:
        record = frozen.get((scenario, model, seed))
        started = _read(output / "test_started.json", {})
        if (not record or record.get("checkpoint_sha256") != checkpoint
                or started.get("evaluation_plan_sha256") != _fingerprint(output / "evaluation_plan.json")["sha256"]):
            raise ValueError("Test metrics are not bound to the globally frozen evaluation plan")
    versions = result.get("metrics_by_split", {})
    if policy == "compact" and "train" in versions:
        raise ValueError("Compact runs omit full training-set inference; epoch training metrics cannot substitute for a checkpoint export")
    if not test_done and "test" in versions:
        raise ValueError("Uncommitted test metrics appear in a training-only result")
    for split in _export_splits(policy, test_done):
        marker = _read(run / split / "export_complete.json", {})
        if marker.get("output_policy", policy) != policy:
            raise ValueError("Export retention policy conflicts with committed results")
        if marker.get("checkpoint_sha256") != checkpoint or marker.get("temperature") != t or marker.get("split") != split:
            raise ValueError("Missing or mismatched committed export marker: " + str(run / split))
        if marker.get("metrics") != versions.get(split) or _read(run / split / "prediction_metrics.json") != versions.get(split):
            raise ValueError("Prediction metric identity conflicts with committed results: " + str(run / split))
    return checkpoint, calibration


def _plots(directory, aggregates):
    """One readable comparison per split; means/SD are within each task/model."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    plt.rcParams.update({"font.family": "DejaVu Sans", "svg.fonttype": "none", "font.size": 10})
    files = []
    for split in ("val", "test"):
        for ext in ("png", "svg"):
            (directory / ("model_comparison_" + split + "." + ext)).unlink(missing_ok=True)
        subset = [r for r in aggregates if r["split"] == split and r["probability_type"] == "raw"]
        if not subset:
            continue
        fig, axes = plt.subplots(2, 2, figsize=(10.8, 6.4))
        for row, scenario in enumerate(SCENARIOS):
            for col, (metric, label) in enumerate((("macro_f1", "Macro-F1 (%)"), ("canonical_false_positive_rate", "Canonical → NCB FPR (%)"))):
                ax = axes[row, col]
                usable = {r["model"]: r for r in subset if r["scenario"] == scenario and r["metric"] == metric and r["mean"] is not None}
                for index, model in enumerate(MODELS):
                    if model in usable:
                        r = usable[model]
                        ax.errorbar(index, r["mean"] * 100, yerr=None if r["sample_sd"] is None else r["sample_sd"] * 100,
                                    fmt="o", capsize=4, color=("#31678B" if scenario == "X" else "#93648E"))
                ax.set(xticks=range(4), xticklabels=MODELS, ylabel=label, title=scenario + " five-class — " + split)
                ax.tick_params(axis="x", labelsize=8); ax.grid(axis="y", alpha=.18)
                if metric.endswith("rate"):
                    ax.set_ylim(bottom=0)
        fig.suptitle("Fixed models; mean ± sample SD across available fixed seeds")
        fig.tight_layout()
        for extension in ("png", "svg"):
            name = "model_comparison_" + split + "." + extension
            fig.savefig(directory / name, dpi=180, bbox_inches="tight"); files.append(name)
        plt.close(fig)
    return files


def _paired(rows):
    """Paired-seed differences are descriptive; three seeds are not test replicates."""
    indexes = {}
    for row in rows:
        if row["probability_type"] == "raw":
            indexes[(row["scenario"], row["model"], row["seed"], row["split"], row["metric"])] = row["value"]
    output = []
    for scenario in SCENARIOS:
        for split in ("train", "val", "test"):
            for comparator in MODELS[1:]:
                for metric in ("macro_f1", "accuracy", "ncb_recall", "ncb_precision", "canonical_false_positive_rate"):
                    values = []
                    for seed in SEEDS:
                        left = indexes.get((scenario, "gatedconv", seed, split, metric))
                        right = indexes.get((scenario, comparator, seed, split, metric))
                        if left is not None and right is not None:
                            values.append((seed, left - right))
                    if values:
                        deltas = [v for _, v in values]
                        output.append({"scenario": scenario, "split": split, "model": "gatedconv", "comparator": comparator,
                                       "metric": metric, "paired_seeds": ",".join(str(s) for s, _ in values), "n_pairs": len(values),
                                       "mean_difference": statistics.mean(deltas),
                                       "sample_sd_difference": statistics.stdev(deltas) if len(deltas) > 1 else None,
                                       "positive_difference_seeds": sum(v > 0 for v in deltas),
                                       "negative_difference_seeds": sum(v < 0 for v in deltas),
                                       "direction": "lower is better" if metric.endswith("rate") else "higher is better",
                                       "scope": "Descriptive paired initialization differences on the same observations; no seed selection or significance claim."})
    return output


def _class_aggregate(rows):
    tidy = []
    for row in rows:
        for metric in ("precision", "recall", "specificity", "fpr", "fdr", "f1"):
            tidy.append({**row, "metric": str(row["class_id"]) + ":" + metric, "value": row[metric]})
    output = _aggregate(tidy)
    for row in output:
        label, metric = row["metric"].split(":")
        row.update(class_id=int(label), class_name=SCENARIOS[row["scenario"]][int(label)], metric=metric)
    return output


def _compact_subgroups(run, summary_dir, scenario, model, seed, split):
    """Keep all nonzero raw confusion counts with short columns, not redundant CSVs."""
    analysis = run / split / "analysis"
    destination = summary_dir / "compact_subgroups" / scenario / model / ("seed_" + str(seed))
    for kind in ("motif", "reference", "source", "strand", "run", "template"):
        source = analysis / ("per_" + kind + "_confusion.csv")
        if not source.is_file():
            continue
        with source.open(encoding="utf-8-sig", newline="") as stream:
            reader = csv.DictReader(stream)
            fields = reader.fieldnames or []
            # The engine writes group, probability_type, true/predicted labels.
            group_key = next((x for x in (kind, "group", "group_value", "value", "masked_motif", "reference_name", "chrom", "source_dataset", "bam_strand", "run_id", "template_group") if x in fields), None)
            truth = next((x for x in ("true_label", "true_class", "true_class_id") if x in fields), None)
            predicted = next((x for x in ("predicted_label", "pred_label", "predicted_class", "predicted_class_id") if x in fields), None)
            count_key = next((x for x in ("count", "windows", "n") if x in fields), None)
            probability = next((x for x in ("probability_type", "probability_version") if x in fields), None)
            if not all((group_key, truth, predicted, count_key)):
                raise ValueError("Unsupported subgroup confusion schema: " + str(source) + " " + repr(fields))
            destination.mkdir(parents=True, exist_ok=True)
            temporary = destination / (split + "_" + kind + "_confusion_sparse.csv.writing")
            target = destination / (split + "_" + kind + "_confusion_sparse.csv")
            try:
                with temporary.open("w", encoding="utf-8", newline="") as out:
                    writer = csv.writer(out); writer.writerow(["group", "true_label", "predicted_label", "count"])
                    for row in reader:
                        if probability and row[probability] != "raw":
                            continue
                        count = int(row[count_key])
                        if count < 0:
                            raise ValueError("Negative subgroup confusion count")
                        if count:
                            writer.writerow([row[group_key], row[truth], row[predicted], count])
                os.replace(temporary, target)
            finally:
                temporary.unlink(missing_ok=True)


def _include_output(path, root, committed_splits):
    parts = path.relative_to(root).parts; name = path.name.lower()
    if path.is_symlink() or any(p.startswith(".") or p == "__pycache__" for p in parts):
        return False
    if any(p in ("feature_cache", "cache", "failed_before_checkpoint", "epoch_predictions") for p in parts):
        return False
    if any(t in name for t in (".writing", ".tmp")) or name in ("manifest.jsonl", "metadata.jsonl", "read_splits.tsv", "result_package_manifest.json"):
        return False
    if path.suffix.lower() in (".pt", ".pth", ".npy", ".npz", ".sqlite", ".db", ".gz", ".zip", ".tar", ".tsv"):
        return False
    if "prepared_data" in parts:
        return path.suffix.lower() == ".json" or (path.suffix.lower() == ".csv" and any(t in name for t in ("count", "audit")))
    if len(parts) >= 7 and parts[0] == "scenarios" and parts[2] == "runs" and parts[5] in ("train", "val", "test"):
        try:
            key = (parts[1], parts[3], int(parts[4].removeprefix("seed_")), parts[5])
        except ValueError:
            return False
        if key not in committed_splits:
            return False
        if len(parts) == 7:
            return name in ("export_complete.json", "prediction_metrics.json")
        if parts[6] != "analysis":
            return False
        # Complete subgroup counts are losslessly compacted separately for val/test.
        # Per-window long curve tables and repeated plotting reports remain on server.
        if name == "bootstrap_replicates.csv":
            return parts[5] == "test"
        return name in ("metrics.json", "confusion_matrix.csv", "per_class_metrics.csv", "ncb_calls.csv",
                        "metadata_coverage.csv", "calibration_bins.csv", "confidence_threshold_scan.csv",
                        "confidence_per_class.csv", "class_threshold_scan.csv", "prevalence_scenarios.csv",
                        "probability_histogram.csv")
    if "resource_profiles" in parts or "preflight" in parts:
        return path.suffix.lower() in (".json", ".log", ".txt", ".csv") and "source_snapshot" not in parts
    if "source_snapshot" in parts:
        return name == "manifest.json"
    if name.startswith("predictions."):
        return False
    return path.suffix.lower() in (".json", ".csv", ".jsonl", ".md", ".txt", ".py", ".log", ".png", ".svg", ".sh", ".yaml", ".yml")


def _include_code(path, root):
    parts = path.relative_to(root).parts
    return (not path.is_symlink() and not any(p.startswith(".") or p in ("__pycache__", "node_modules", "tests", "validation") for p in parts)
            and path.suffix.lower() in (".py", ".sh", ".json", ".csv", ".md", ".txt", ".yml", ".yaml"))


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
    too_large = [name for name, item in fingerprints.items() if item["size_bytes"] > MAX_FILE_BYTES]
    if too_large or sum(item["size_bytes"] for item in fingerprints.values()) > MAX_PAYLOAD_BYTES:
        raise ValueError("Compact evidence exceeds size budget. Full training and predictions remain intact. Inspect: " + repr(too_large[:8]))
    manifest = {"status": summary["status"], "package_version": VERSION, "stage": summary["stage"],
                "fixed_scenarios": list(SCENARIOS), "fixed_models": list(MODELS), "fixed_seeds": list(SEEDS),
                "completed_training_jobs": summary["completed_training_jobs"], "test_evaluated_jobs": summary["test_evaluated_jobs"],
                "file_count": len(selected), "files": fingerprints,
                "excluded_heavy_files": ["weights", "tensor arrays", "epoch logits", "per-window predictions", "cache", "prepared rows", "ROC/PR threshold-point tables", "redundant plots and subgroup CSVs"],
                "subgroup_policy": "All nonzero raw subgroup confusion counts for val/test retained as sparse tables, omitted cells mean zero; calibrated argmax is identical. Full original subgroup reports stay on server.",
                "server_retention": summary["retention_policy"],
                "size_limits_bytes": {"per_file": MAX_FILE_BYTES, "uncompressed_payload": MAX_PAYLOAD_BYTES, "compressed_core": MAX_CORE_BYTES},
                "policy": "All fixed seeds/models, separately per task; no performance selection or truncation."}
    marker = output / "RESULT_PACKAGE_MANIFEST.json"; _json(marker, manifest)
    destination = output / name; temporary = destination.with_name(destination.name + ".writing." + str(os.getpid()))
    try:
        with tarfile.open(temporary, "w:gz", compresslevel=6) as archive:
            archive.add(marker, arcname=marker.name, recursive=False)
            for relative, path in selected.items():
                archive.add(path, arcname=relative, recursive=False)
                if _fingerprint(path) != fingerprints[relative]:
                    raise ValueError("Evidence changed during packaging: " + str(path))
        if temporary.stat().st_size > MAX_CORE_BYTES:
            raise ValueError("Compressed core exceeds 20MiB. All results retained; inspect compact evidence and rerun summary only.")
        os.replace(temporary, destination)
    finally:
        temporary.unlink(missing_ok=True)
    return destination


def summarize_and_package(output_dir, package_dir=None):
    output = Path(output_dir).resolve(); output.mkdir(parents=True, exist_ok=True)
    directory = output / SUMMARY_DIR; directory.mkdir(exist_ok=True)
    frozen_plan, frozen = _frozen_records(output)
    statuses, metrics, classes, confusions, learning, numerical, runtime, intervals = ([] for _ in range(8))
    complete, tested, committed = [], [], set()
    policies = set()
    for scenario in SCENARIOS:
        for model in MODELS:
            for seed in SEEDS:
                identity = {"scenario": scenario, "model": model, "seed": seed}
                run = output / "scenarios" / scenario / "runs" / model / ("seed_" + str(seed))
                result = _read(run / "results.json", {}); status = result.get("status", "not_started")
                policy = _output_policy(result, _read(run / "resolved_config.json", {}))
                finished = status in ("train_complete", "complete"); test_done = finished and result.get("test_evaluated") is True
                if status == "complete" and not test_done:
                    raise ValueError("Complete result must explicitly record test_evaluated=true")
                if not finished and result.get("test_evaluated"):
                    raise ValueError("Incomplete training cannot contain committed test results")
                checkpoint, calibration = (None, {})
                if finished:
                    checkpoint, calibration = _verify_run(run, result, scenario, model, seed, test_done, frozen, output)
                    complete.append(identity)
                    if test_done:
                        tested.append(identity)
                    policies.add(policy)
                statuses.append({**identity, "run_dir": str(run), "status": status, "training_complete": finished,
                                 "test_evaluated": test_done, "selected_epoch": result.get("best_epoch"),
                                 "temperature": calibration.get("temperature"), "checkpoint_sha256": checkpoint,
                                 "output_policy": policy if result else None,
                                 "uncommitted_test_artifacts_present": not test_done and (run / "test").is_dir() and any((run / "test").iterdir())})
                if not finished:
                    continue
                for split in _export_splits(policy, test_done):
                    versions = result.get("metrics_by_split", {}).get(split)
                    if not isinstance(versions, dict) or not all(v in versions for v in ("raw", "calibrated")):
                        raise ValueError("Committed result lacks raw/calibrated metrics: " + str(run / split))
                    committed.add((scenario, model, seed, split)); matrices = []
                    for probability in ("raw", "calibrated"):
                        values, per_class, matrix, count = _count_metrics(versions[probability], scenario); matrices.append(matrix)
                        for metric, value in values.items():
                            metrics.append({**identity, "split": split, "probability_type": probability, "n": count,
                                            "metric": metric, "value": value, "unit": "native loss" if metric in ("nll", "brier") else "fraction",
                                            "checkpoint_sha256": checkpoint})
                        classes.extend({**identity, "split": split, "probability_type": probability, **row} for row in per_class)
                        for t, row in enumerate(matrix):
                            for p, count in enumerate(row):
                                confusions.append({**identity, "split": split, "probability_type": probability,
                                                   "true_label": t, "true_name": SCENARIOS[scenario][t], "predicted_label": p,
                                                   "predicted_name": SCENARIOS[scenario][p], "count": count, "row_fraction": _ratio(count, sum(row))})
                    if matrices[0] != matrices[1]:
                        raise ValueError("Positive validation temperature changed argmax")
                    analysis = _read(run / split / "analysis/metrics.json", {})
                    if analysis and _names(analysis.get("class_definition")) != SCENARIOS[scenario]:
                        raise ValueError("Analysis class definition differs from fixed scenario")
                    for probability, uncertainty in analysis.get("uncertainty", {}).items():
                        if not isinstance(uncertainty, dict):
                            continue
                        bounds_map = dict(uncertainty.get("intervals", {}))
                        for alias, key in (("canonical_false_positive_rate", "class_4_fpr"), ("ncb_recall", "class_4_recall"), ("ncb_precision", "class_4_precision"), ("ncb_f1", "class_4_f1")):
                            if key in bounds_map:
                                bounds_map[alias] = bounds_map[key]
                        for metric, bounds in bounds_map.items():
                            intervals.append({**identity, "split": split, "probability_type": probability, "metric": metric,
                                              "ci_low": bounds.get("ci_low"), "ci_high": bounds.get("ci_high"),
                                              "valid_replicates": bounds.get("valid_replicates"), "cluster_count": uncertainty.get("cluster_count"),
                                              "replicates": uncertainty.get("replicates"), "method": uncertainty.get("method")})
                    if split in ("val", "test"):
                        _compact_subgroups(run, directory, scenario, model, seed, split)
                history_path = run / "epoch_metrics.jsonl"
                history = [json.loads(line) for line in history_path.read_text(encoding="utf-8").splitlines() if line.strip()] if history_path.is_file() else []
                for row in history:
                    learning.append({**identity, "epoch": row["epoch"], "lr": row.get("lr"), "train_loss": row.get("train", {}).get("loss"),
                                     "val_loss": row.get("val", {}).get("loss"), "train_macro_f1": row.get("train", {}).get("macro_f1"),
                                     "val_macro_f1": row.get("val", {}).get("macro_f1"),
                                     "training_metric_scope": "In-epoch training pass while weights update; not a frozen-checkpoint training-set evaluation.",
                                     "validation_metric_scope": "End-of-epoch validation inference used for checkpoint selection."})
                numerical.append(_numerical(scenario, model, seed, history, result)); runtime.append({**identity, **result.get("runtime", {})})
    all_training, all_test = len(complete) == 24, len(tested) == 24
    stage, status, stem = ("evaluation", "evaluation_complete", "results") if all_test else (("training", "training_complete", "training") if all_training and not tested else ("partial", "partial", "partial"))
    aggregates = _aggregate(metrics); class_aggregates = _class_aggregate(classes); paired = _paired(metrics)
    tables = [("run_status", statuses), ("metrics_per_seed", metrics), ("mean_sd", aggregates), ("class_metrics", classes),
              ("class_mean_sd", class_aggregates), ("confusion_counts", confusions), ("learning_curves", learning),
              ("read_cluster_intervals", intervals), ("numerical_diagnostics", numerical), ("runtime", runtime), ("paired_model_comparisons", paired)]
    for name, values in tables:
        fields = list(dict.fromkeys(k for row in values for k in row)) or ["scenario", "model", "seed"]
        _csv(directory / (name + ".csv"), values, fields)
    figures = _plots(directory, aggregates)
    # Data audits, rather than class labels, determine how positive identity is described.
    audits = {scenario: _read(output / "scenarios" / scenario / "preflight/data_audit.json", {}) for scenario in SCENARIOS}
    retention = {"policies_in_completed_runs": sorted(policies),
                 "compact": {"saved": ["all scalar epoch training/validation metrics", "best and last resumable checkpoints plus best state_dict", "validation and test float32 logits, labels, raw/calibrated probabilities, per-window prediction CSV and analyses", "validation-fitted temperature and provenance"],
                             "omitted": ["full training-set checkpoint inference and per-window exports", "per-epoch validation logits and sample arrays", "embeddings", "attention arrays", "periodic checkpoint copies"],
                             "reanalysis_scope": "Saved validation/test scores and metadata support offline performance analysis and plotting without neural inference. Embedding or attention analyses and full frozen-checkpoint training-set predictions would require later inference from saved checkpoints."},
                 "full": "Legacy full-policy runs retain train/validation/test predictions, embeddings and attention according to their resolved configuration."}
    summary = {"version": VERSION, "status": status, "stage": stage, "created_at": datetime.now(timezone.utc).isoformat(),
               "fixed_scenarios": list(SCENARIOS), "fixed_models": list(MODELS), "fixed_seeds": list(SEEDS), "class_names_by_scenario": SCENARIOS,
               "completed_training_jobs": complete, "test_evaluated_jobs": tested, "expected_training_jobs": 24,
               "test_evaluation_status": "all_24_complete" if all_test else "partial" if tested else "not_run",
               "all_training_complete": all_training, "all_test_complete": all_test, "runs": statuses, "mean_sd": aggregates,
               "class_mean_sd": class_aggregates, "paired_model_comparisons": paired, "figures": figures,
               "architecture": "Final GatedConv (legacy_gatedconv + target_attention), Transformer, ConvMixer, BiGRU; five-class output heads; no architecture retuning.",
               "retention_policy": retention,
               "training_metric_scope": "learning_curves.csv contains in-epoch training metrics; compact runs have no frozen-checkpoint training-set evaluation in metrics_per_seed.csv.",
               "scope": "Two original broad-context five-class tasks, separately balanced and trained; source and positive-identity limitations retained from data audits.",
               "data_audit_scope": {s: {k: audits[s].get(k) for k in ("evaluation_scope", "read_id_overlap_check_scope", "original_positive_read_uuid_isolation_independently_verified", "positive_one_row_per_read_assumption")} for s in SCENARIOS},
               "probability_metrics": "Saved probability analysis; classification and binary-NCB metrics independently reconstructed from integer confusion counts.",
               "missing_values": "Undefined values remain null. Sample SD is null for a single seed. Tasks are never pooled; same windows across seeds are not independent experimental repeats.",
               "limitations": ["Positive CSV-row proxies, if present, preserve the user's one-row-per-positive-read assumption, not verified original read UUIDs.",
                               "Read/record grouping does not establish template or sequencing-run independence; unknown provenance remains unknown.",
                               "Source-confounded original main tasks differ from the completed same-source/direction-matched supplementary controls.",
                               "Training-only cores contain no committed test performance; raw/calibrated outputs are not separate trainings.",
                               "Timing in runtime.csv reflects concurrent operational training, not fair isolated speed comparisons.",
                               "Compact output retains validation/test predictions and scores, not embeddings, attention or full training-set checkpoint inference; see retention_policy."]}
    summary_path = directory / "summary.json"; _json(summary_path, summary)
    core = _package(output, package_dir, "fiveclass24_" + stem + "_core_v1_0_0.tar.gz", summary, committed)
    return {"summary_path": str(summary_path), "core_path": str(core), "status": status, "stage": stage,
            "completed_training_jobs": complete, "test_evaluated_jobs": tested}


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output_dir", required=True)
    parser.add_argument("--package_dir", default=str(Path(__file__).resolve().parent))
    args = parser.parse_args(argv)
    print(json.dumps(summarize_and_package(args.output_dir, args.package_dir), ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()

