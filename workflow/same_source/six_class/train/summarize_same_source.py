#!/usr/bin/env python3
"""Summarize committed same-source GatedConv results without loading a model.

All rates remain fractions in machine-readable tables.  The core archive retains
complete small evidence and excludes only explicitly named heavy data products.
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
CLASS_NAMES = ("A", "T", "C", "G", "X", "Y")
SUMMARY_DIR = "summary"
PROBABILITY_METRICS = ("nll", "brier", "ece_15bins", "macro_roc_auc",
                       "macro_average_precision", "micro_roc_auc", "micro_average_precision")


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
            stream.write("\n")
            stream.flush()
            os.fsync(stream.fileno())
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
            writer.writeheader()
            writer.writerows(_clean(rows))
            stream.flush()
            os.fsync(stream.fileno())
        os.replace(temporary, path)
    finally:
        temporary.unlink(missing_ok=True)


def _fingerprint(path):
    path = Path(path)
    before = path.stat()
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(2 ** 20), b""):
            digest.update(block)
    after = path.stat()
    if (before.st_size, before.st_mtime_ns) != (after.st_size, after.st_mtime_ns):
        raise ValueError("File changed during packaging: " + str(path))
    return {"sha256": digest.hexdigest(), "size_bytes": after.st_size}


def _ratio(numerator, denominator):
    return numerator / denominator if denominator else None


def _count_metrics(saved):
    matrix = saved.get("confusion_matrix")
    if (not isinstance(matrix, list) or len(matrix) != 6
            or any(not isinstance(row, list) or len(row) != 6 for row in matrix)
            or any(type(x) is not int or x < 0 for row in matrix for x in row)):
        raise ValueError("Committed metrics require a nonnegative integer 6x6 confusion matrix")
    support = [sum(row) for row in matrix]
    predicted = [sum(row[column] for row in matrix) for column in range(6)]
    total = sum(support)
    if not total or saved.get("n", total) != total:
        raise ValueError("Confusion matrix and sample count disagree")
    classes = []
    for index, name in enumerate(CLASS_NAMES):
        tp = matrix[index][index]
        fp, fn = predicted[index] - tp, support[index] - tp
        tn = total - tp - fp - fn
        classes.append({"class_id": index, "class_name": name, "support": support[index],
                        "predicted_support": predicted[index], "tp": tp, "fp": fp, "fn": fn,
                        "tn": tn, "precision": _ratio(tp, tp + fp), "recall": _ratio(tp, tp + fn),
                        "specificity": _ratio(tn, tn + fp), "fpr": _ratio(fp, tn + fp),
                        "fdr": _ratio(fp, tp + fp), "f1": _ratio(2 * tp, 2 * tp + fp + fn)})
    canonical = sum(support[:4])
    ncb = sum(support[4:])
    detected = sum(sum(row[4:]) for row in matrix[4:])
    false_ncb = sum(sum(row[4:]) for row in matrix[:4])
    swap = matrix[4][5] + matrix[5][4]
    defined_recall = [row["recall"] for row in classes if row["recall"] is not None]
    metrics = {
        "accuracy": sum(matrix[k][k] for k in range(6)) / total,
        "macro_f1": sum(row["f1"] or 0 for row in classes) / 6,
        "balanced_accuracy": statistics.mean(defined_recall) if defined_recall else None,
        "canonical_accuracy": _ratio(sum(matrix[k][k] for k in range(4)), canonical),
        "canonical_false_positive_rate": _ratio(false_ncb, canonical),
        "target_recall_micro": _ratio(matrix[4][4] + matrix[5][5], ncb),
        "ncb_recall": _ratio(detected, ncb), "ncb_precision": _ratio(detected, detected + false_ncb),
        "xy_swap_rate": _ratio(swap, ncb),
        "xy_to_canonical_rate": _ratio(sum(sum(row[:4]) for row in matrix[4:]), ncb),
        "X_recall": classes[4]["recall"], "X_f1": classes[4]["f1"],
        "Y_recall": classes[5]["recall"], "Y_f1": classes[5]["f1"],
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
        key = (row["split"], row["probability_type"], row["metric"])
        groups.setdefault(key, []).append(row)
    output = []
    for (split, probability_type, metric), items in sorted(groups.items()):
        defined = [item for item in items if item["value"] is not None]
        values = [item["value"] for item in defined]
        output.append({"split": split, "probability_type": probability_type, "metric": metric,
                       "expected_seeds": len(SEEDS), "available_seeds": ",".join(str(item["seed"]) for item in items),
                       "defined_seeds": ",".join(str(item["seed"]) for item in defined), "n_available": len(items),
                       "n_defined": len(values), "mean": statistics.mean(values) if values else None,
                       "sample_sd": statistics.stdev(values) if len(values) > 1 else None,
                       "status": "complete_n3" if len(items) == len(SEEDS) else "partial_n" + str(len(items))})
    return output


def _numerical(seed, run, result):
    path = run / "epoch_metrics.jsonl"
    history = [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()] if path.is_file() else []
    counts = {key: 0 for key in ("optimizer_step_attempts", "optimizer_steps_successful",
                                 "optimizer_steps_skipped_amp", "grad_norm_nonfinite_count")}
    available = bool(history)
    finite_losses = True
    for item in history:
        train = item.get("train", {})
        for key in counts:
            value = train.get(key)
            if value is None:
                available = False
            elif type(value) is int:
                counts[key] += value
        for split in ("train", "val"):
            loss = item.get(split, {}).get("loss")
            finite_losses = finite_losses and isinstance(loss, (int, float)) and math.isfinite(loss)
    attempts = counts["optimizer_step_attempts"]
    best_epoch = result.get("best_epoch")
    selected = next((item for item in history if item.get("epoch") == best_epoch), None)
    return {"seed": seed, "epochs_recorded": len(history), "selected_epoch": best_epoch,
            "diagnostics_available": available, **{key: value if available else None for key, value in counts.items()},
            "skipped_attempt_fraction": _ratio(counts["optimizer_steps_skipped_amp"], attempts) if available else None,
            "all_train_val_losses_finite": finite_losses if history else None,
            "selected_val_loss": selected.get("val", {}).get("loss") if selected else None,
            "optimizer_count_scope": "Recorded GradScaler scale-backoff accounting; protective skipped updates are not failed runs"}


def _plots(summary_dir, learning, confusion, complete_seeds):
    """Plots are descriptive and show only actual available runs."""
    for stem in ("learning_curves", "confusion_val", "confusion_test"):
        for extension in ("png", "svg"):
            (summary_dir / (stem + "." + extension)).unlink(missing_ok=True)
    if not complete_seeds:
        return []
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    files = []
    plt.rcParams.update({"font.family": "DejaVu Sans", "svg.fonttype": "none", "font.size": 10,
                         "axes.spines.top": False, "axes.spines.right": False})
    if learning:
        figure, axes = plt.subplots(1, 2, figsize=(11.2, 4.0))
        for seed in complete_seeds:
            rows = [row for row in learning if row["seed"] == seed]
            for axis, key, label in ((axes[0], "val_loss", "Validation loss"),
                                     (axes[1], "val_macro_f1", "Validation Macro-F1 (%)")):
                usable = [row for row in rows if row.get(key) is not None]
                scale = 100 if key.endswith("f1") else 1
                axis.plot([row["epoch"] for row in usable], [row[key] * scale for row in usable], label="Seed " + str(seed))
                axis.set_xlabel("Epoch")
                axis.set_ylabel(label)
                axis.grid(alpha=.15)
                axis.legend(fontsize=8)
        figure.suptitle("Same-source GatedConv — training/validation only")
        figure.tight_layout()
        for extension in ("png", "svg"):
            path = summary_dir / ("learning_curves." + extension)
            figure.savefig(path, bbox_inches="tight", dpi=180)
            files.append(path.name)
        plt.close(figure)
    for split in ("val", "test"):
        available = sorted({row["seed"] for row in confusion if row["split"] == split and row["probability_type"] == "raw"})
        if not available:
            continue
        matrix = [[0.0] * 6 for _ in range(6)]
        for seed in available:
            for row in confusion:
                if row["split"] == split and row["probability_type"] == "raw" and row["seed"] == seed:
                    matrix[row["true_label"]][row["predicted_label"]] += row["row_fraction"] / len(available) * 100
        figure, axis = plt.subplots(figsize=(6.1, 5.1))
        image = axis.imshow(matrix, cmap="Blues", vmin=0, vmax=100)
        for i in range(6):
            for j in range(6):
                axis.text(j, i, format(matrix[i][j], ".1f"), ha="center", va="center",
                          color="white" if matrix[i][j] > 60 else "#183543", fontsize=9)
        axis.set_xticks(range(6), CLASS_NAMES)
        axis.set_yticks(range(6), CLASS_NAMES)
        axis.set_xlabel("Predicted class")
        axis.set_ylabel("True class")
        axis.set_title(("Validation" if split == "val" else "Frozen internal test") + " confusion; n=" + str(len(available)) + " seed(s)")
        figure.colorbar(image, ax=axis, label="Within-class windows (%)")
        figure.tight_layout()
        for extension in ("png", "svg"):
            path = summary_dir / ("confusion_" + split + "." + extension)
            figure.savefig(path, bbox_inches="tight", dpi=180)
            files.append(path.name)
        plt.close(figure)
    return files


def _include_output(path, root):
    relative = path.relative_to(root)
    parts = relative.parts
    if path.is_symlink() or any(part.startswith(".") or part == "__pycache__" for part in parts):
        return False
    if any(part in ("feature_cache", "cache") for part in parts):
        return False
    name = path.name.lower()
    if (".writing" in name or ".tmp" in name or name.startswith("predictions.")
            or name in ("manifest.jsonl", "metadata.jsonl", "read_splits.tsv")):
        return False
    if path.suffix.lower() in (".pt", ".pth", ".npy", ".npz", ".sqlite", ".db", ".gz", ".zip", ".tar"):
        return False
    if parts[0] == "prepared_data":
        return path.suffix.lower() == ".json" or (path.suffix.lower() == ".csv" and any(token in name for token in ("count", "audit")))
    return path.suffix.lower() in (".json", ".csv", ".jsonl", ".md", ".txt", ".py", ".log", ".png", ".svg", ".sh", ".yaml", ".yml", ".fasta")


def _include_code(path, root):
    relative = path.relative_to(root)
    if path.is_symlink() or any(part.startswith(".") or part in ("__pycache__", "node_modules") for part in relative.parts):
        return False
    return path.suffix.lower() in (".py", ".sh", ".json", ".csv", ".md", ".txt", ".fasta", ".log", ".yml", ".yaml")


def _package(output, package_dir, name, summary):
    destination = output / name
    selected = {}
    for path in sorted(output.rglob("*")):
        if path.is_file() and _include_output(path, output):
            selected[str(path.relative_to(output))] = path
    if package_dir is not None:
        package_dir = Path(package_dir).resolve()
        for path in sorted(package_dir.rglob("*")):
            if path.is_file() and not path.is_relative_to(output) and _include_code(path, package_dir):
                selected["code/" + str(path.relative_to(package_dir))] = path
    manifest = {"status": summary["status"], "package_version": VERSION, "stage": summary["stage"],
                "fixed_seeds": list(SEEDS), "completed_training_seeds": summary["completed_training_seeds"],
                "test_evaluated_seeds": summary["test_evaluated_seeds"], "file_count": len(selected),
                "files": {name: _fingerprint(path) for name, path in selected.items()},
                "excluded_heavy_files": "Weights, tensor arrays, epoch logits, per-window prediction CSV, cache and prepared data rows stay on the server.",
                "policy": "No size-budget truncation of included evidence; no test metrics are fabricated; raw/calibrated are the same run."}
    manifest_path = output / "RESULT_PACKAGE_MANIFEST.json"
    # The previous manifest must never include itself or its stale hash.
    selected.pop("RESULT_PACKAGE_MANIFEST.json", None)
    manifest["files"].pop("RESULT_PACKAGE_MANIFEST.json", None)
    manifest["file_count"] = len(selected)
    _json(manifest_path, manifest)
    temporary = destination.with_name(destination.name + ".writing." + str(os.getpid()))
    try:
        with tarfile.open(temporary, "w:gz") as archive:
            archive.add(manifest_path, arcname=manifest_path.name, recursive=False)
            for relative, path in selected.items():
                archive.add(path, arcname=relative, recursive=False)
                if _fingerprint(path) != manifest["files"][relative]:
                    raise ValueError("File changed while creating core: " + str(path))
        os.replace(temporary, destination)
    finally:
        temporary.unlink(missing_ok=True)
    return destination


def summarize_and_package(output_dir, package_dir=None):
    """Summarize available committed runs and return the compact evidence path."""
    output = Path(output_dir).resolve()
    output.mkdir(parents=True, exist_ok=True)
    summary_dir = output / SUMMARY_DIR
    summary_dir.mkdir(exist_ok=True)
    status_rows, metric_rows, class_rows, confusion_rows, learning_rows, diagnostics, runtime_rows = [], [], [], [], [], [], []
    completed, tested = [], []
    for seed in SEEDS:
        run = output / "runs" / ("seed_" + str(seed))
        result = _read(run / "results.json", {})
        status = result.get("status", "not_started")
        finished = status in ("train_complete", "complete")
        test_done = finished and result.get("test_evaluated") is True
        if finished:
            if status == "complete" and not test_done:
                raise ValueError("A complete evaluation result must explicitly record test_evaluated=true")
            if result.get("seed", seed) != seed or result.get("feature_scheme", "full") != "full":
                raise ValueError("Unexpected seed or feature scheme in committed result: " + str(run))
            if result.get("litemamba_variant", "legacy_gatedconv") != "legacy_gatedconv" or result.get("readout", "target_attention") != "target_attention":
                raise ValueError("Committed result is not the fixed final GatedConv")
            if not test_done and "test" in result.get("metrics_by_split", {}):
                raise ValueError("Uncommitted test metrics appear in a training-only result")
            completed.append(seed)
            if test_done:
                tested.append(seed)
        calibration = _read(run / "calibration.json", {})
        checkpoint_sha = result.get("provenance", {}).get("checkpoint_sha256")
        if finished:
            temperature = calibration.get("temperature")
            if (not isinstance(checkpoint_sha, str) or len(checkpoint_sha) != 64
                    or calibration.get("fitted_on") != "val" or calibration.get("checkpoint_sha256") != checkpoint_sha
                    or not isinstance(temperature, (int, float)) or not math.isfinite(temperature) or temperature <= 0):
                raise ValueError("Calibration is not tied to this checkpoint and validation data")
        status_rows.append({"seed": seed, "run_dir": str(run), "status": status,
                            "training_complete": finished, "test_evaluated": test_done,
                            "selected_epoch": result.get("best_epoch"), "temperature": calibration.get("temperature"),
                            "checkpoint_sha256": checkpoint_sha})
        if not finished:
            continue
        versions = result.get("metrics_by_split", {})
        for split in ("train", "val") + (("test",) if test_done else ()):
            probability_versions = versions.get(split)
            if not isinstance(probability_versions, dict) or not all(key in probability_versions for key in ("raw", "calibrated")):
                raise ValueError("Committed run is missing exported metrics: " + str(run) + "/" + split)
            matrices = []
            for probability_type in ("raw", "calibrated"):
                values, per_class, matrix, count = _count_metrics(probability_versions[probability_type])
                matrices.append(matrix)
                for metric, value in values.items():
                    metric_rows.append({"seed": seed, "split": split, "probability_type": probability_type,
                                        "n": count, "metric": metric, "value": value,
                                        "unit": "native loss" if metric in ("nll", "brier") else "fraction",
                                        "checkpoint_sha256": checkpoint_sha})
                for row in per_class:
                    class_rows.append({"seed": seed, "split": split, "probability_type": probability_type, **row})
                for true_index, row in enumerate(matrix):
                    for predicted_index, count_at_cell in enumerate(row):
                        confusion_rows.append({"seed": seed, "split": split, "probability_type": probability_type,
                                               "true_label": true_index, "true_name": CLASS_NAMES[true_index],
                                               "predicted_label": predicted_index, "predicted_name": CLASS_NAMES[predicted_index],
                                               "count": count_at_cell, "row_fraction": _ratio(count_at_cell, sum(row))})
            if matrices[0] != matrices[1]:
                raise ValueError("A positive temperature unexpectedly changed classification argmax")
        history_path = run / "epoch_metrics.jsonl"
        if history_path.is_file():
            for line in history_path.read_text(encoding="utf-8").splitlines():
                if not line.strip():
                    continue
                item = json.loads(line)
                learning_rows.append({"seed": seed, "epoch": item["epoch"], "lr": item.get("lr"),
                                      "train_loss": item.get("train", {}).get("loss"), "val_loss": item.get("val", {}).get("loss"),
                                      "train_macro_f1": item.get("train", {}).get("macro_f1"), "val_macro_f1": item.get("val", {}).get("macro_f1")})
        diagnostics.append(_numerical(seed, run, result))
        runtime_rows.append({"seed": seed, **result.get("runtime", {})})
    if completed == list(SEEDS) and tested == list(SEEDS):
        stage, status, core_name = "evaluation", "evaluation_complete", "cplx_same_source_results_core_v1_0_0.tar.gz"
    elif completed == list(SEEDS) and not tested:
        stage, status, core_name = "training", "training_complete", "cplx_same_source_training_core_v1_0_0.tar.gz"
    elif completed == [42] and not tested:
        stage, status, core_name = "pilot", "pilot_complete", "cplx_same_source_pilot_core_v1_0_0.tar.gz"
    else:
        stage, status, core_name = "partial", "partial", "cplx_same_source_partial_core_v1_0_0.tar.gz"
    aggregates = _aggregate(metric_rows)
    _csv(summary_dir / "run_status.csv", status_rows, ["seed", "run_dir", "status", "training_complete", "test_evaluated", "selected_epoch", "temperature", "checkpoint_sha256"])
    _csv(summary_dir / "metrics_per_seed.csv", metric_rows, ["seed", "split", "probability_type", "n", "metric", "value", "unit", "checkpoint_sha256"])
    _csv(summary_dir / "mean_sd.csv", aggregates, ["split", "probability_type", "metric", "expected_seeds", "available_seeds", "defined_seeds", "n_available", "n_defined", "mean", "sample_sd", "status"])
    _csv(summary_dir / "class_metrics.csv", class_rows, ["seed", "split", "probability_type", "class_id", "class_name", "support", "predicted_support", "tp", "fp", "fn", "tn", "precision", "recall", "specificity", "fpr", "fdr", "f1"])
    _csv(summary_dir / "confusion_counts.csv", confusion_rows, ["seed", "split", "probability_type", "true_label", "true_name", "predicted_label", "predicted_name", "count", "row_fraction"])
    _csv(summary_dir / "learning_curves.csv", learning_rows, ["seed", "epoch", "lr", "train_loss", "val_loss", "train_macro_f1", "val_macro_f1"])
    numeric_fields = list(dict.fromkeys(key for row in diagnostics for key in row)) or ["seed", "diagnostics_available"]
    _csv(summary_dir / "numerical_diagnostics.csv", diagnostics, numeric_fields)
    runtime_fields = list(dict.fromkeys(key for row in runtime_rows for key in row)) or ["seed"]
    _csv(summary_dir / "runtime.csv", runtime_rows, runtime_fields)
    figures = _plots(summary_dir, learning_rows, confusion_rows, completed)
    summary = {"version": VERSION, "status": status, "stage": stage,
               "created_at": datetime.now(timezone.utc).isoformat(), "fixed_seeds": list(SEEDS),
               "completed_training_seeds": completed, "test_evaluated_seeds": tested,
               "test_evaluation_status": "all_three_complete" if tested == list(SEEDS) else "not_run" if not tested else "partial",
               "all_training_complete": completed == list(SEEDS), "all_test_complete": tested == list(SEEDS),
               "architecture": "Final GatedConv, legacy_gatedconv + target_attention; no SSM claim",
               "scope": "Same CPLX batch, real-read-grouped internal six-class evaluation; no cross-batch/template independence claim",
               "probability_metrics": "Read from saved probability analyses; classification rates independently recomputed from saved confusion counts",
               "missing_values": "Unavailable seeds and undefined denominators remain missing; sample SD is null with one seed",
               "runs": status_rows, "mean_sd": aggregates, "figures": figures,
               "limitations": ["Context support is the frozen common subset, not all 1,024 contexts.",
                               "Same-source sampling does not remove all NCB-position or X/Y-direction associations.",
                               "Windows from one real read remain in one split; uncertainty should retain all windows of that read.",
                               "Raw and calibrated are two probability versions of the same trained model, not independent repetitions.",
                               "Training-only packages contain no test performance; pilot results must not select or tune on the frozen test set."]}
    summary_path = summary_dir / "summary.json"
    _json(summary_path, summary)
    core_path = _package(output, package_dir, core_name, summary)
    return {"summary_path": str(summary_path), "core_path": str(core_path), "status": status,
            "stage": stage, "completed_training_seeds": completed, "test_evaluated_seeds": tested}


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output_dir", required=True)
    parser.add_argument("--package_dir", default=str(Path(__file__).resolve().parent))
    args = parser.parse_args(argv)
    print(json.dumps(summarize_and_package(args.output_dir, args.package_dir), ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
