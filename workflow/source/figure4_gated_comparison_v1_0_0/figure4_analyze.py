#!/usr/bin/env python3
"""离线分析 Figure 4 保存的逐窗口预测，不加载模型或重新推理。"""
from __future__ import annotations

import argparse
from collections import Counter, defaultdict
from contextlib import ExitStack
import csv
import gzip
import hashlib
import json
import math
from pathlib import Path
import sqlite3
import tempfile
import time

import numpy as np
import pandas as pd


CLASS_NAMES = ["A", "T", "C", "G", "X", "Y"]
THRESHOLDS = [0.0, 0.5, 0.7, 0.8, 0.9, 0.95, 0.99, 0.995, 0.999]
PREVALENCES = [0.0001, 0.001, 0.01, 0.05, 0.1, 1 / 3]
UNKNOWN = {"", "unknown", "na", "n/a", "none", "null", "nan", "."}
SEED_METRICS = ["accuracy", "macro_f1", "balanced_accuracy", "ncb_precision", "ncb_recall", "ncb_f1",
                "canonical_false_positive_rate", "target_recall_micro", "nll", "brier", "ece_15bins"]
GROUP_COLUMNS = ["training_run_id", "model", "seed", "feature_scheme", "probability_type", "group_level",
                 "group_key", "cohort", "template", "role", "file_strand", "bam_strand",
                 "source_dataset", "sequencing_run_id", "reference", "motif", "pair_id", "template_group"]


def known(value):
    return str(value or "").strip().lower() not in UNKNOWN


def ratio(a, b):
    return float(a / b) if b else None


def wilson(success, n):
    if not n:
        return None, None
    z = 1.959963984540054
    p = success / n
    denominator = 1 + z * z / n
    center = (p + z * z / (2 * n)) / denominator
    half = z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / denominator
    return max(0.0, center - half), min(1.0, center + half)


def json_dump(path, value):
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2, allow_nan=False) + "\n", encoding="utf-8")


def file_sha256(path):
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(8 * 1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def verify_prediction_marker(directory, plan, run, dataset):
    marker, prediction = directory / "completion.json", directory / "predictions.csv.gz"
    if not marker.is_file() or not prediction.is_file():
        raise ValueError("prediction or completion marker missing")
    value = json.loads(marker.read_text(encoding="utf-8"))
    if value.get("status") != "complete" or (plan.get("plan_id") and value.get("plan_id") != plan["plan_id"]):
        raise ValueError("completion status/plan identity mismatch")
    for field, expected in [("run_id", run["run_id"]), ("dataset_id", dataset["dataset_id"]),
                            ("n_samples", dataset.get("n_samples")), ("checkpoint_sha256", run.get("checkpoint_sha256")),
                            ("temperature", run.get("temperature"))]:
        if field in value and expected is not None and value[field] != expected:
            raise ValueError(f"completion {field} differs from evaluation plan")
    declared = value.get("outputs", {}).get("predictions.csv.gz")
    before = prediction.stat()
    if declared is not None:
        if before.st_size != int(declared["size_bytes"]):
            raise ValueError("prediction size differs from completion manifest")
        if file_sha256(prediction) != declared["sha256"]:
            raise ValueError("prediction SHA256 differs from completion manifest")
    after = prediction.stat()
    if (before.st_size, before.st_mtime_ns) != (after.st_size, after.st_mtime_ns):
        raise ValueError("prediction changed during integrity verification")
    value["_verified_prediction_stat"] = (after.st_size, after.st_mtime_ns)
    return value, {"run_id": run["run_id"], "dataset_id": dataset["dataset_id"],
                   "prediction_sha256_verified": declared is not None,
                   "prediction_sha256": declared.get("sha256") if declared is not None else None,
                   "prediction_size_bytes": after.st_size,
                   "integrity_note": "size and SHA256 verified" if declared is not None else "legacy marker has no output hash declaration"}


def counts_metrics(cm):
    cm = np.asarray(cm, dtype=np.int64)
    n = int(cm.sum())
    supports, predicted = cm.sum(axis=1), cm.sum(axis=0)
    classes = []
    for k in range(6):
        tp = int(cm[k, k])
        fn = int(supports[k] - tp)
        fp = int(predicted[k] - tp)
        tn = n - tp - fn - fp
        row = {"class_id": k, "class_name": CLASS_NAMES[k], "support": int(supports[k]),
               "predicted_support": int(predicted[k]), "tp": tp, "fp": fp, "fn": fn, "tn": tn,
               "precision": ratio(tp, tp + fp), "recall": ratio(tp, tp + fn),
               "specificity": ratio(tn, tn + fp), "fpr": ratio(fp, tn + fp),
               "fdr": ratio(fp, tp + fp), "f1": ratio(2 * tp, 2 * tp + fp + fn)}
        for metric, numerator, denominator in [("precision", tp, tp + fp), ("recall", tp, tp + fn),
                                               ("specificity", tn, tn + fp), ("fpr", fp, tn + fp),
                                               ("fdr", fp, tp + fp)]:
            row[metric + "_wilson_low"], row[metric + "_wilson_high"] = wilson(numerator, denominator)
        classes.append(row)
    positive_support = int(supports[4:].sum())
    canonical_support = int(supports[:4].sum())
    tp = int(cm[4:, 4:].sum())
    fp = int(cm[:4, 4:].sum())
    fn = positive_support - tp
    recalls = [r["recall"] for r in classes if r["recall"] is not None]
    result = {"n": n, "correct": int(np.trace(cm)), "accuracy": ratio(np.trace(cm), n),
              "macro_f1": float(np.mean([r["f1"] or 0.0 for r in classes])),
              "macro_f1_definition": "fixed six labels; absent-class F1 set to zero",
              "balanced_accuracy": float(np.mean(recalls)) if recalls else None,
              "present_class_count": int((supports > 0).sum()),
              "canonical_support": canonical_support, "ncb_support": positive_support,
              "canonical_false_positive_calls": fp, "canonical_false_positive_rate": ratio(fp, canonical_support),
              "false_noncanonical_call_rate": ratio(fp, canonical_support),
              "ncb_precision": ratio(tp, tp + fp), "ncb_recall": ratio(tp, positive_support),
              "ncb_f1": ratio(2 * tp, 2 * tp + fp + fn),
              "target_support": positive_support, "target_correct": int(cm[4, 4] + cm[5, 5]),
              "target_recall_micro": ratio(cm[4, 4] + cm[5, 5], positive_support)}
    result["accuracy_wilson_low"], result["accuracy_wilson_high"] = wilson(result["correct"], n)
    result["canonical_fpr_wilson_low"], result["canonical_fpr_wilson_high"] = wilson(fp, canonical_support)
    return result, classes


class Accumulator:
    def __init__(self):
        self.cm = np.zeros((6, 6), dtype=np.int64)
        self.nll = self.brier = 0.0
        self.bins = np.zeros((15, 3), dtype=np.float64)
        self.real_read_rows = 0

    def add(self, truth, pred, probability, real_read):
        self.cm[truth, pred] += 1
        self.nll -= math.log(max(float(probability[truth]), np.finfo(float).tiny))
        self.brier += float(np.dot(probability, probability) - 2 * probability[truth] + 1)
        confidence = float(probability[pred])
        index = min(14, int(confidence * 15))
        self.bins[index] += [1, confidence, int(pred == truth)]
        self.real_read_rows += int(real_read)

    def metrics(self):
        values, classes = counts_metrics(self.cm)
        n = values["n"]
        gap = np.abs(self.bins[:, 1] - self.bins[:, 2]).sum()
        values.update(nll=ratio(self.nll, n), brier=ratio(self.brier, n), ece_15bins=ratio(gap, n),
                      real_read_rows=self.real_read_rows, unknown_read_rows=n - self.real_read_rows,
                      interval_scope="Wilson intervals describe windows; within-read dependence is not removed.")
        return values, classes


class ThresholdAccumulator:
    def __init__(self):
        self.total_support = np.zeros(6, dtype=np.int64)
        self.cm = np.zeros((len(THRESHOLDS), 6, 6), dtype=np.int64)

    def add(self, truth, pred, confidence):
        self.total_support[truth] += 1
        for index, threshold in enumerate(THRESHOLDS):
            if confidence >= threshold:
                self.cm[index, truth, pred] += 1

    def rows(self):
        total = int(self.total_support.sum())
        canonical = int(self.total_support[:4].sum())
        ncb = int(self.total_support[4:].sum())
        for threshold, cm in zip(THRESHOLDS, self.cm):
            retained = int(cm.sum())
            ncb_tp = int(cm[4:, 4:].sum())
            ncb_fp = int(cm[:4, 4:].sum())
            row = {"confidence_threshold": threshold, "total_count": total, "retained_count": retained,
                   "abstained_count": total - retained, "coverage": ratio(retained, total),
                   "accuracy_among_retained": ratio(np.trace(cm), retained),
                   "ncb_precision": ratio(ncb_tp, ncb_tp + ncb_fp), "ncb_recall": ratio(ncb_tp, ncb),
                   "canonical_false_positive_rate": ratio(ncb_fp, canonical),
                   "decision": "retain if max probability >= fixed threshold; abstention counts as missed recall"}
            for k in [4, 5]:
                name = CLASS_NAMES[k]
                row[name + "_precision"] = ratio(cm[k, k], cm[:, k].sum())
                row[name + "_recall"] = ratio(cm[k, k], self.total_support[k])
                row[name + "_retained_true_fraction"] = ratio(cm[k].sum(), self.total_support[k])
            yield row


def identity_groups(row, dataset):
    cohort = str(row.get("cohort") or dataset.get("cohort", "unknown"))
    template = str(row.get("template") or dataset.get("template", "unknown"))
    role = str(row.get("role") or dataset.get("role", "unknown"))
    file_strand = str(row.get("file_strand") or dataset.get("file_strand", "unknown"))
    bam_strand = str(row.get("bam_strand") or "unknown")
    source = str(row.get("source_dataset") or "unknown")
    run = str(row.get("run_id") or "unknown")
    reference = str(row.get("reference") or dataset.get("reference", "unknown"))
    motif = str(row.get("motif") or "unknown")
    pair_id = str(row.get("pair_id") or "unknown")
    template_group = str(row.get("template_group") or "unknown")
    fields = [
        ("overall", {}), ("cohort", {"cohort": cohort}),
        ("template", {"cohort": cohort, "template": template}),
        ("template_role", {"cohort": cohort, "template": template, "role": role}),
        ("template_strand", {"cohort": cohort, "template": template, "role": role, "file_strand": file_strand}),
        ("template_source_run_strand", {"cohort": cohort, "template": template, "source_dataset": source,
                                         "sequencing_run_id": run, "bam_strand": bam_strand}),
        ("reference", {"reference": reference, "role": role}),
        ("source", {"source_dataset": source}),
        ("run", {"sequencing_run_id": run}),
        ("strand", {"bam_strand": bam_strand}),
        ("motif", {"motif": motif}),
    ]
    if known(pair_id):
        fields.append(("template_pair", {"cohort": cohort, "template": template,
                                         "pair_id": pair_id, "template_group": template_group}))
    for level, values in fields:
        key = json.dumps(values, sort_keys=True, ensure_ascii=False, separators=(",", ":"))
        yield level, key, values


def real_read_identity(row):
    read_id = str(row.get("read_id") or "unknown").strip()
    kind = str(row.get("read_id_kind") or "unknown").lower()
    real = known(read_id) and known(kind) and not any(word in kind for word in ["proxy", "unavailable", "synthetic", "row"])
    return read_id, real


def cluster_intervals(connection, scope, expected_n, real_n, repetitions, rng, replicate_writer, base):
    if repetitions == 0 or real_n != expected_n or expected_n == 0:
        return {}, {"read_cluster_ci_available": False,
                    "uncertainty_method": "not computed" if repetitions == 0 else "unavailable: real read IDs missing for some/all windows",
                    "bootstrap_replicates": repetitions, "read_cluster_count": None}
    profiles = Counter()
    current_id, matrix = None, np.zeros(36, dtype=np.int64)
    cursor = connection.execute("SELECT read_id, cell, n FROM clusters WHERE scope=? ORDER BY read_id, cell", (scope,))
    for read_id, cell, count in cursor:
        if current_id is not None and read_id != current_id:
            profiles[tuple(matrix)] += 1
            matrix.fill(0)
        current_id = read_id
        matrix[cell] = count
    if current_id is not None:
        profiles[tuple(matrix)] += 1
    if not profiles:
        raise ValueError("Real read coverage was complete but no read clusters were saved")
    vectors = np.asarray(list(profiles), dtype=np.int64)
    frequencies = np.asarray(list(profiles.values()), dtype=np.int64)
    total_reads = int(frequencies.sum())
    assert int((vectors * frequencies[:, None]).sum()) == expected_n
    distributions = defaultdict(list)
    for iteration in range(repetitions):
        sampled = rng.multinomial(total_reads, frequencies / total_reads)
        cm = (sampled @ vectors).reshape(6, 6)
        overall, classes = counts_metrics(cm)
        values = {key: overall[key] for key in ["accuracy", "ncb_precision", "ncb_recall", "ncb_f1",
                                               "canonical_false_positive_rate", "target_recall_micro"]}
        for item in classes:
            if item["class_id"] in [4, 5]:
                for metric in ["precision", "recall", "specificity", "fpr", "fdr", "f1"]:
                    values[f'class_{item["class_id"]}_{metric}'] = item[metric]
        replicate_writer.writerow({**base, "replicate": iteration, **values})
        for key, value in values.items():
            if value is not None and math.isfinite(value):
                distributions[key].append(value)
    intervals = {}
    for key, values in distributions.items():
        lo, hi = np.quantile(values, [0.025, 0.975])
        intervals[key + "_read_cluster_ci_low"] = float(lo)
        intervals[key + "_read_cluster_ci_high"] = float(hi)
        intervals[key + "_read_cluster_valid_replicates"] = len(values)
    return intervals, {"read_cluster_ci_available": True,
                       "uncertainty_method": "read-cluster percentile bootstrap; retains every window and its own truth",
                       "bootstrap_replicates": repetitions, "read_cluster_count": total_reads}


def write_table(folder, name, rows, columns=None):
    frame = pd.DataFrame(rows)
    if columns:
        frame = frame.reindex(columns=list(dict.fromkeys([*columns, *frame.columns])))
    frame.to_csv(folder / (name + ".csv"), index=False)
    return frame


def gzip_writer(stack, path, fields):
    stream = stack.enter_context(gzip.open(path, "wt", encoding="utf-8", newline=""))
    writer = csv.DictWriter(stream, fieldnames=fields)
    writer.writeheader()
    return writer


def seed_summaries(overall_rows, planned_runs):
    frame = pd.DataFrame(overall_rows)
    keys = ["model", "feature_scheme", "probability_type", "group_level", "group_key"]
    rows = []
    planned = defaultdict(set)
    for run in planned_runs:
        planned[(run["model"], run.get("feature_scheme", "full"))].add(int(run["seed"]))
    for _, part in frame.groupby(keys, dropna=False, sort=True):
        first = part.iloc[0]
        expected = planned[(first.model, first.feature_scheme)]
        observed = set(part.seed.astype(int))
        if observed != expected or len(part) != len(expected):
            raise ValueError(f"Seed coverage differs across groups: {first.model} {first.group_level} {first.group_key}: {observed} != {expected}")
        for metric in SEED_METRICS:
            values = pd.to_numeric(part[metric], errors="coerce").dropna()
            rows.append({**{key: first.get(key) for key in GROUP_COLUMNS if key not in ["seed", "training_run_id"]},
                         "metric": metric, "mean": float(values.mean()) if len(values) else None,
                         "std": float(values.std(ddof=1)) if len(values) > 1 else None,
                         "seed_sd": float(values.std(ddof=1)) if len(values) > 1 else None,
                         "valid_seeds": len(values), "n_seeds": len(part), "planned_seeds": len(expected),
                         "seed_ids": ",".join(map(str, sorted(expected))),
                         "aggregation": "per-seed metric mean and sample SD; repeated predictions are not independent samples"})
    return pd.DataFrame(rows)


def figure_tables(folder, summaries, confusion_rows):
    data = summaries[summaries.probability_type.eq("raw") & summaries.feature_scheme.eq("full")]
    lite = data[data.model.eq("litemamba") & data.group_level.eq("template_role")]
    a, b, detail, c = [], [], [], []
    for template, part in lite.groupby("template", sort=False):
        try:
            number = int(str(template).replace("XNA", ""))
        except ValueError:
            continue
        control = part[part.role.eq("control") & part.metric.eq("canonical_false_positive_rate")]
        target = part[part.role.eq("target") & part.metric.eq("target_recall_micro")]
        if len(control) == 1 and number <= 16:
            row = control.iloc[0]
            a.append({"template_num": number, "dna_label": f"DNA{number}",
                      "false_noncanonical_call_rate": row["mean"], "seed_sd": row["std"], "n_seeds": row.n_seeds})
        if len(target) == 1 and number <= 16:
            row = target.iloc[0]
            b.append({"template_num": number, "xna_label": f"XNA{number}",
                      "target_recall_micro": row["mean"], "seed_sd": row["std"], "n_seeds": row.n_seeds})
        if len(target) == 1 and 17 <= number <= 20:
            row = target.iloc[0]
            c.append({"template_num": number, "template": template,
                      "total_error_per_10000": (1 - row["mean"]) * 10000,
                      "target_recall_micro": row["mean"], "seed_sd": row["std"] * 10000 if pd.notna(row["std"]) else None,
                      "target_recall_seed_sd": row["std"], "n_seeds": row.n_seeds})
    counts = pd.DataFrame(confusion_rows)
    if not counts.empty:
        counts = counts[counts.model.eq("litemamba") & counts.feature_scheme.eq("full") &
                        counts.probability_type.eq("raw") & counts.group_level.eq("template_role") & counts.role.eq("target")]
        for template, part in counts.groupby("template", sort=False):
            try:
                number = int(str(template).replace("XNA", ""))
            except ValueError:
                continue
            if not 17 <= number <= 20:
                continue
            per_seed = []
            for seed, run in part.groupby("seed"):
                cm = run.pivot(index="true_label", columns="predicted_label", values="count").reindex(index=range(6), columns=range(6)).to_numpy()
                support = float(cm[4:].sum())
                if support == 0:
                    continue
                errors = {CLASS_NAMES[k]: float(cm[4:, k].sum()) / support * 10000 for k in range(4)}
                errors["opposite NCB"] = float(cm[4, 5] + cm[5, 4]) / support * 10000
                per_seed.append({"seed": seed, **errors})
            for error in ["A", "T", "C", "G", "opposite NCB"]:
                values = [row[error] for row in per_seed]
                if values:
                    detail.append({"template_num": number, "template": template, "error_type": error,
                                   "error_per_10000": float(np.mean(values)),
                                   "seed_sd": float(np.std(values, ddof=1)) if len(values) > 1 else None,
                                   "n_seeds": len(values)})
    profile = []
    definitions = [("dna_false_call", "DNA01–16 false X/Y calls ↓", "control", "canonical_false_positive_rate", 1, 16),
                   ("single_ncb_recall", "XNA01–12 target recall ↑", "target", "target_recall_micro", 1, 12),
                   ("multiple_ncb_recall", "XNA13–16 target recall ↑", "target", "target_recall_micro", 13, 16),
                   ("four_ncb_recall", "XNA17–20 target recall ↑", "target", "target_recall_micro", 17, 20)]
    for model in sorted(data.model.unique()):
        for order, (key, label, role, metric, low, high) in enumerate(definitions):
            part = data[data.model.eq(model) & data.group_level.eq("template_role") & data.role.eq(role) & data.metric.eq(metric)].copy()
            numbers = part.template.astype(str).str.extract(r"(\d+)$")[0].astype(float)
            part = part[numbers.between(low, high)]
            if not len(part):
                continue
            # Reconstruct each seed's template-macro profile separately, not SD of template means.
            original = pd.read_csv(folder / "per_template_role_metrics.csv")
            original = original[original.model.eq(model) & original.feature_scheme.eq("full") & original.probability_type.eq("raw") & original.role.eq(role)]
            nums = original.template.astype(str).str.extract(r"(\d+)$")[0].astype(float)
            original = original[nums.between(low, high)]
            values = original.groupby("seed")[metric].mean().dropna()
            profile.append({"metric_key": key, "metric_label": label, "order": order, "model": model,
                            "value": float(values.mean()), "seed_sd": float(values.std(ddof=1)) if len(values) > 1 else None,
                            "n_seeds": len(values), "n_templates": original.template.nunique(),
                            "aggregation": "equal template means within each seed, then equal seed mean"})
    write_table(folder, "figure4A_dna01_16_false_call_heatmap_data_6class", a,
                ["template_num", "dna_label", "false_noncanonical_call_rate", "seed_sd", "n_seeds"])
    write_table(folder, "figure4B_xna01_16_target_recall_lollipop_data_6class", b,
                ["template_num", "xna_label", "target_recall_micro", "seed_sd", "n_seeds"])
    write_table(folder, "figure4C_xna17_20_residual_error_detail_6class", detail,
                ["template_num", "template", "error_type", "error_per_10000", "seed_sd", "n_seeds"])
    write_table(folder, "figure4C_xna17_20_residual_error_summary_6class", c,
                ["template_num", "template", "total_error_per_10000", "target_recall_micro", "seed_sd", "target_recall_seed_sd", "n_seeds"])
    write_table(folder, "figure4D_external_validation_profile_data_6class", profile,
                ["metric_key", "metric_label", "order", "model", "value", "seed_sd", "n_seeds", "n_templates", "aggregation"])


def chunk_metadata(frame, dataset):
    """Keep the legacy fallback rules and exact grouping keys, without per-window JSON."""
    def column(name, default="unknown"):
        if name not in frame:
            return pd.Series(str(default), index=frame.index, dtype=object)
        return frame[name].where(frame[name].ne(""), str(default))
    values = {name: column(name, dataset.get(name, "unknown"))
              for name in ["cohort", "template", "role", "file_strand", "reference"]}
    values.update({name: column(name) for name in ["bam_strand", "source_dataset", "motif", "pair_id", "template_group"]})
    values["sequencing_run_id"] = column("run_id")
    return pd.DataFrame(values, index=frame.index)


CHUNK_GROUPS = [
    ("overall", []), ("cohort", ["cohort"]), ("template", ["cohort", "template"]),
    ("template_role", ["cohort", "template", "role"]),
    ("template_strand", ["cohort", "template", "role", "file_strand"]),
    ("template_source_run_strand", ["cohort", "template", "source_dataset", "sequencing_run_id", "bam_strand"]),
    ("reference", ["reference", "role"]), ("source", ["source_dataset"]),
    ("run", ["sequencing_run_id"]), ("strand", ["bam_strand"]), ("motif", ["motif"]),
    ("template_pair", ["cohort", "template", "pair_id", "template_group"]),
]


def consume_chunked_dataset(path, dataset, run, temperature, accumulators, group_info, thresholds,
                            metadata, cluster_batch, flush_clusters, bootstrap, chunk_size=50000):
    """Vectorize window validation and additive statistics; retain read-cluster semantics."""
    with gzip.open(path, "rt", encoding="utf-8-sig", newline="") as stream:
        fields = next(csv.reader(stream), [])
    required = {"sample_index", "sample_id", "label", *[f"prob_{k}" for k in range(6)],
                *[f"calibrated_prob_{k}" for k in range(6)]}
    if not required.issubset(fields):
        raise ValueError(f"Prediction CSV fields missing: {path}")
    has_logits = [f"logit_{k}" in fields for k in range(6)]
    if any(has_logits) and not all(has_logits):
        raise ValueError(f"Incomplete saved logits: {path}")
    if temperature is not None and (not math.isfinite(float(temperature)) or float(temperature) <= 0):
        raise ValueError(f"Frozen calibration temperature is invalid: {run['run_id']}")
    text_fields = ["sample_id", "model", "seed", "feature_scheme", "read_id", "read_id_kind", "run_id",
                   "cohort", "template", "role", "file_strand", "reference", "bam_strand", "source_dataset",
                   "motif", "pair_id", "template_group"]
    numeric_fields = [f"{prefix}{k}" for prefix in ["prob_", "calibrated_prob_", "logit_"] for k in range(6)]
    numeric_fields += ["confidence", "calibrated_confidence"]
    int_fields = ["sample_index", "label", "prediction"]
    used = [name for name in fields if name in text_fields + numeric_fields + int_fields]
    dtype = {name: str for name in text_fields if name in used}
    dtype.update({name: np.float64 for name in numeric_fields if name in used})
    dtype.update({name: np.int64 for name in int_fields if name in used})
    count = 0
    for frame in pd.read_csv(path, encoding="utf-8-sig", compression="gzip", usecols=used,
                             dtype=dtype, keep_default_na=False, chunksize=chunk_size, float_precision="round_trip"):
        n = len(frame)
        truth = frame.label.to_numpy(dtype=np.int64)
        if not np.array_equal(frame.sample_index.to_numpy(), np.arange(count, count + n)) or frame.sample_id.eq("").any():
            raise ValueError(f"Prediction sample order/identity invalid at {path}:{count}")
        if np.any((truth < 0) | (truth >= 6)):
            raise ValueError(f"Prediction truth outside six labels: {path}:{count}")
        for field, expected in [("model", run["model"]), ("seed", str(int(run["seed"]))),
                                ("feature_scheme", run.get("feature_scheme", "full"))]:
            if field in frame and not frame[field].eq(expected).all():
                raise ValueError(f"Saved row {field} differs from evaluation plan: {path}:{count}")
        read_ids = frame.get("read_id", pd.Series("unknown", index=frame.index)).str.strip()
        read_kinds = frame.get("read_id_kind", pd.Series("unknown", index=frame.index)).str.lower()
        real = (~read_ids.str.lower().isin(UNKNOWN) & ~read_kinds.str.strip().isin(UNKNOWN) &
                ~read_kinds.str.contains("proxy|unavailable|synthetic|row", regex=True)).to_numpy()
        for field, counts in metadata.items():
            if field == "read_id":
                counts["known"] += int(real.sum())
                counts["unknown_or_proxy"] += int(n - real.sum())
            else:
                series = frame.get(field, pd.Series("unknown", index=frame.index))
                good = ~series.str.strip().str.lower().isin(UNKNOWN)
                counts["known"] += int(good.sum())
                counts["unknown"] += int(n - good.sum())
        probabilities = {}
        for variant, prefix in [("raw", "prob_"), ("calibrated", "calibrated_prob_")]:
            p = frame[[prefix + str(k) for k in range(6)]].to_numpy(dtype=np.float64)
            if not np.isfinite(p).all() or np.any(p < 0) or np.any(p > 1) or np.any(np.abs(p.sum(axis=1) - 1) > 2e-4):
                raise ValueError(f"Invalid saved probabilities at {path}:{count}")
            probabilities[variant] = p
        predictions = {variant: p.argmax(axis=1) for variant, p in probabilities.items()}
        if not np.array_equal(predictions["raw"], predictions["calibrated"]):
            raise ValueError("Scalar-temperature calibration changed argmax; inspect saved predictions")
        if all(has_logits):
            logits = frame[[f"logit_{k}" for k in range(6)]].to_numpy(dtype=np.float64)
            if not np.isfinite(logits).all():
                raise ValueError(f"Nonfinite saved logits: {path}:{count}")
            for variant, temp in [("raw", 1.0), ("calibrated", temperature)]:
                if temp is None:
                    continue
                transformed = logits / float(temp)
                expected = np.exp(transformed - transformed.max(axis=1, keepdims=True))
                expected /= expected.sum(axis=1, keepdims=True)
                if not np.allclose(probabilities[variant], expected, rtol=2e-7, atol=1e-10):
                    raise ValueError(f"Saved {variant} probabilities disagree with logits/frozen temperature: {path}:{count}")
        if "prediction" in frame and not np.array_equal(frame.prediction.to_numpy(), predictions["raw"]):
            raise ValueError(f"Saved prediction disagrees with probabilities: {path}:{count}")
        for field, variant in [("confidence", "raw"), ("calibrated_confidence", "calibrated")]:
            if field in frame and not np.allclose(frame[field].to_numpy(), probabilities[variant].max(axis=1), rtol=2e-7, atol=1e-10):
                raise ValueError(f"Saved {field} disagrees with probabilities: {path}:{count}")
        values = chunk_metadata(frame, dataset)
        prepared = {}
        for variant, p in probabilities.items():
            pred = predictions[variant]
            confidence = p[np.arange(n), pred]
            prepared[variant] = {"cell": truth * 6 + pred, "confidence": confidence,
                "bin": np.minimum(14, (confidence * 15).astype(np.int64)), "correct": (pred == truth).astype(np.int64),
                "nll": -np.log(np.maximum(p[np.arange(n), truth], np.finfo(float).tiny)),
                "brier": (p * p).sum(axis=1) - 2 * p[np.arange(n), truth] + 1}
        for level, names in CHUNK_GROUPS:
            mask = ~values.pair_id.str.strip().str.lower().isin(UNKNOWN).to_numpy() if level == "template_pair" else np.ones(n, dtype=bool)
            if not mask.any():
                continue
            # One scalar key per distinct group, rather than one JSON key per window.
            if names:
                code, unique = pd.factorize(pd.MultiIndex.from_frame(values.loc[mask, names]), sort=False)
                descriptions = [dict(zip(names, item)) for item in unique]
            else:
                code, descriptions = np.zeros(int(mask.sum()), dtype=np.int64), [{}]
            group_keys = [json.dumps(item, sort_keys=True, ensure_ascii=False, separators=(",", ":")) for item in descriptions]
            ngroups = len(descriptions)
            local_real = real[mask]
            real_counts = np.bincount(code, weights=local_real.astype(np.int64), minlength=ngroups).astype(np.int64)
            for variant, data in prepared.items():
                cell, confidence, bins = data["cell"][mask], data["confidence"][mask], data["bin"][mask]
                cms = np.bincount(code * 36 + cell, minlength=ngroups * 36).reshape(ngroups, 6, 6)
                nll = np.bincount(code, weights=data["nll"][mask], minlength=ngroups)
                brier = np.bincount(code, weights=data["brier"][mask], minlength=ngroups)
                index = code * 15 + bins
                bin_count = np.bincount(index, minlength=ngroups * 15).reshape(ngroups, 15)
                bin_confidence = np.bincount(index, weights=confidence, minlength=ngroups * 15).reshape(ngroups, 15)
                bin_correct = np.bincount(index, weights=data["correct"][mask], minlength=ngroups * 15).reshape(ngroups, 15)
                retained = []
                if level in ["overall", "template", "template_role"]:
                    for threshold in THRESHOLDS:
                        keep = confidence >= threshold
                        retained.append(np.bincount(code[keep] * 36 + cell[keep], minlength=ngroups * 36).reshape(ngroups, 6, 6))
                for k, key in enumerate(group_keys):
                    identity = (variant, level, key)
                    accumulator = accumulators.setdefault(identity, Accumulator())
                    group_info[identity] = descriptions[k]
                    accumulator.cm += cms[k]
                    accumulator.nll += float(nll[k])
                    accumulator.brier += float(brier[k])
                    accumulator.bins += np.column_stack((bin_count[k], bin_confidence[k], bin_correct[k]))
                    accumulator.real_read_rows += int(real_counts[k])
                    if retained:
                        threshold_accumulator = thresholds.setdefault(identity, ThresholdAccumulator())
                        threshold_accumulator.total_support += cms[k].sum(axis=1)
                        threshold_accumulator.cm += np.asarray([item[k] for item in retained])
                if variant == "raw" and bootstrap and level in ["overall", "template", "template_role"] and local_real.any():
                    # This path keeps actual read IDs and each window's truth. A mixed-truth
                    # read remains one cluster, including when it crosses chunk boundaries.
                    local_ids = read_ids.to_numpy()[mask][local_real]
                    clusters = pd.DataFrame({"code": code[local_real], "read_id": local_ids,
                                             "cell": cell[local_real]}).groupby(["code", "read_id", "cell"], sort=False).size()
                    scopes = [json.dumps((level, key), separators=(",", ":")) for key in group_keys]
                    for (k, read_id, cell_id), occurrences in clusters.items():
                        cluster_batch[(scopes[k], read_id, int(cell_id))] += int(occurrences)
                    if len(cluster_batch) >= 20000:
                        flush_clusters()
        count += n
    return count


def consume_legacy_dataset(path, dataset, run, temperature, accumulators, group_info, thresholds,
                           metadata, cluster_batch, flush_clusters, bootstrap):
    count = 0
    with gzip.open(path, "rt", encoding="utf-8-sig", newline="") as stream:
        reader = csv.DictReader(stream)
        required = {"sample_index", "sample_id", "label", *[f"prob_{k}" for k in range(6)], *[f"calibrated_prob_{k}" for k in range(6)]}
        if not required.issubset(reader.fieldnames or []):
            raise ValueError(f"Prediction CSV fields missing: {path}")
        has_logits = [f"logit_{k}" in reader.fieldnames for k in range(6)]
        if any(has_logits) and not all(has_logits):
            raise ValueError(f"Incomplete saved logits: {path}")
        if temperature is not None and (not math.isfinite(float(temperature)) or float(temperature) <= 0):
            raise ValueError(f"Frozen calibration temperature is invalid: {run['run_id']}")
        for row in reader:
            if int(row["sample_index"]) != count or not row["sample_id"]:
                raise ValueError(f"Prediction sample order/identity invalid at {path}:{count}")
            truth = int(row["label"])
            if not 0 <= truth < 6:
                raise ValueError(f"Prediction truth outside six labels: {path}:{count}")
            for field, expected in [("model", run["model"]), ("seed", str(int(run["seed"]))),
                                    ("feature_scheme", run.get("feature_scheme", "full"))]:
                if field in row and row[field] != expected:
                    raise ValueError(f"Saved row {field} differs from evaluation plan: {path}:{count}")
            read_id, real_read = real_read_identity(row)
            for field, counts in metadata.items():
                if field == "read_id":
                    counts["known" if real_read else "unknown_or_proxy"] += 1
                else:
                    counts["known" if known(row.get(field)) else "unknown"] += 1
            groups = list(identity_groups(row, dataset))
            probabilities = {}
            for variant, prefix in [("raw", "prob_"), ("calibrated", "calibrated_prob_")]:
                p = np.asarray([float(row[prefix + str(k)]) for k in range(6)], dtype=float)
                if not np.isfinite(p).all() or np.any(p < 0) or np.any(p > 1) or abs(p.sum() - 1) > 2e-4:
                    raise ValueError(f"Invalid saved probabilities at {path}:{count}")
                probabilities[variant] = p
                pred = int(p.argmax())
                for level, key, values in groups:
                    identity = (variant, level, key)
                    if identity not in accumulators:
                        accumulators[identity] = Accumulator()
                        group_info[identity] = values
                    accumulators[identity].add(truth, pred, p, real_read)
                    if level in ["overall", "template", "template_role"]:
                        thresholds.setdefault(identity, ThresholdAccumulator()).add(truth, pred, float(p[pred]))
                        # Scalar temperature keeps argmax; store one copy of
                        # read counts and reuse the same draw for raw/calibrated.
                        if real_read and bootstrap and variant == "raw":
                            scope = json.dumps((level, key), separators=(",", ":"))
                            cluster_batch[(scope, read_id, truth * 6 + pred)] += 1
            if int(probabilities["raw"].argmax()) != int(probabilities["calibrated"].argmax()):
                raise ValueError("Scalar-temperature calibration changed argmax; inspect saved predictions")
            if all(has_logits):
                logits = np.asarray([float(row[f"logit_{k}"]) for k in range(6)], dtype=float)
                if not np.isfinite(logits).all():
                    raise ValueError(f"Nonfinite saved logits: {path}:{count}")
                for variant, temp in [("raw", 1.0), ("calibrated", temperature)]:
                    if temp is None:
                        continue
                    transformed = logits / float(temp)
                    expected = np.exp(transformed - transformed.max())
                    expected /= expected.sum()
                    if not np.allclose(probabilities[variant], expected, rtol=2e-7, atol=1e-10):
                        raise ValueError(f"Saved {variant} probabilities disagree with logits/frozen temperature: {path}:{count}")
            pred = int(probabilities["raw"].argmax())
            if "prediction" in row and int(row["prediction"]) != pred:
                raise ValueError(f"Saved prediction disagrees with probabilities: {path}:{count}")
            for field, variant in [("confidence", "raw"), ("calibrated_confidence", "calibrated")]:
                if field in row and not math.isclose(float(row[field]), float(probabilities[variant].max()), rel_tol=2e-7, abs_tol=1e-10):
                    raise ValueError(f"Saved {field} disagrees with probabilities: {path}:{count}")
            count += 1
            if len(cluster_batch) >= 20000:
                flush_clusters()
    return count


def analyze(output_root, bootstrap=200, analysis_seed=42, engine="chunked", chunk_size=50000):
    root = Path(output_root).expanduser().resolve()
    if int(bootstrap) != bootstrap or bootstrap < 0:
        raise ValueError("bootstrap must be a nonnegative integer")
    if engine not in ["legacy", "chunked"] or int(chunk_size) != chunk_size or chunk_size < 1:
        raise ValueError("engine must be legacy/chunked; chunk_size must be a positive integer")
    plan_path = root / "evaluation_plan.json"
    plan = json.loads(plan_path.read_text(encoding="utf-8"))
    runs, datasets = plan["runs"], plan["datasets"]
    if not runs or not datasets:
        raise ValueError("No planned model runs or datasets")
    run_ids = [run["run_id"] for run in runs]
    dataset_ids = [dataset["dataset_id"] for dataset in datasets]
    if len(set(run_ids)) != len(run_ids) or len(set(dataset_ids)) != len(dataset_ids):
        raise ValueError("Duplicate run_id or dataset_id in evaluation plan")
    missing, integrity, markers = [], [], {}
    for run in runs:
        for dataset in datasets:
            directory = root / "predictions" / run["run_id"] / dataset["dataset_id"]
            try:
                marker, verification = verify_prediction_marker(directory, plan, run, dataset)
                markers[(run["run_id"], dataset["dataset_id"])] = marker
                integrity.append(verification)
            except (ValueError, TypeError, KeyError, OSError) as error:
                missing.append({"run_id": run["run_id"], "dataset_id": dataset["dataset_id"], "reason": str(error)})
    coverage = {"planned_runs": len(runs), "planned_datasets": len(datasets), "expected_prediction_files": len(runs) * len(datasets),
                "missing_or_incompatible": missing, "observed_template_roles": sorted({(d.get("template", "unknown"), d.get("role", "unknown")) for d in datasets}),
                "input_coverage_as_recorded": plan.get("coverage", plan.get("discovery_audit", {})),
                "prediction_integrity": integrity,
                "status": "incomplete" if missing else "complete"}
    observed = {(str(d.get("template", "unknown")), str(d.get("role", "unknown")), str(d.get("file_strand", "unknown"))) for d in datasets}
    coverage["template_role_strand_presence"] = [{"template": f"XNA{number}", "role": role, "file_strand": strand,
        "present_in_plan": (f"XNA{number}", role, strand) in observed}
        for number in range(1, 21) for role in ["control", "target"] for strand in ["pos", "neg"]]
    coverage["coverage_note"] = "Presence is manifest coverage, not proof of source independence. Some designs may omit controls for XNA17-20; omissions stay explicit."
    json_dump(root / "analysis_coverage.json", coverage)
    if missing:
        raise ValueError(f"Analysis requires every planned prediction; {len(missing)} entries are incomplete. See analysis_coverage.json")
    work = Path(tempfile.mkdtemp(prefix=".analysis-working-", dir=root))
    rows, legacy_confusion_rows, threshold_rows, metadata_rows, prevalence_rows = [], [], [], [], []
    replicate_fields = GROUP_COLUMNS + ["replicate", "accuracy", "ncb_precision", "ncb_recall", "ncb_f1", "canonical_false_positive_rate", "target_recall_micro"]
    replicate_fields += [f"class_{k}_{metric}" for k in [4, 5] for metric in ["precision", "recall", "specificity", "fpr", "fdr", "f1"]]
    connection = None
    try:
        with ExitStack() as stack:
            replicate_writer = gzip_writer(stack, work / "read_cluster_bootstrap_replicates.csv.gz", replicate_fields)
            uncertainty_fields = ["read_cluster_ci_available", "uncertainty_method", "bootstrap_replicates", "read_cluster_count"]
            class_fields = GROUP_COLUMNS + list(counts_metrics(np.zeros((6, 6), dtype=int))[1][0])
            class_fields += [metric + suffix for metric in ["precision", "recall", "specificity", "fpr", "fdr", "f1"]
                             for suffix in ["_read_cluster_ci_low", "_read_cluster_ci_high", "_read_cluster_valid_replicates"]]
            class_writer = gzip_writer(stack, work / "per_class_metrics.csv.gz", class_fields + uncertainty_fields + ["wilson_scope"])
            confusion_writer = gzip_writer(stack, work / "confusion_matrices.csv.gz", GROUP_COLUMNS +
                ["true_label", "true_class", "predicted_label", "predicted_class", "count", "row_fraction"])
            calibration_writer = gzip_writer(stack, work / "calibration_bins.csv.gz", GROUP_COLUMNS +
                ["bin_left", "bin_right", "n", "mean_confidence", "accuracy"])
            for run_number, run in enumerate(runs):
                print(f'[ANALYZE] {run_number + 1}/{len(runs)} {run["run_id"]}', flush=True)
                accumulators, group_info, thresholds = {}, {}, {}
                database = work / "read_clusters.sqlite"
                connection = sqlite3.connect(database)
                connection.execute("PRAGMA journal_mode=OFF")
                connection.execute("PRAGMA synchronous=OFF")
                connection.execute("CREATE TABLE clusters(scope TEXT, read_id TEXT, cell INTEGER, n INTEGER, PRIMARY KEY(scope,read_id,cell)) WITHOUT ROWID")
                cluster_batch = Counter()
                metadata = {field: Counter() for field in ["read_id", "read_id_kind", "reference", "bam_strand", "run_id", "source_dataset", "template_group", "pair_id"]}
                def flush_clusters():
                    connection.executemany("INSERT INTO clusters(scope,read_id,cell,n) VALUES(?,?,?,?) ON CONFLICT(scope,read_id,cell) DO UPDATE SET n=n+excluded.n",
                                           [(*key, n) for key, n in cluster_batch.items()])
                    connection.commit()
                    cluster_batch.clear()
                for dataset in datasets:
                    path = root / "predictions" / run["run_id"] / dataset["dataset_id"] / "predictions.csv.gz"
                    before = path.stat()
                    marker = markers[(run["run_id"], dataset["dataset_id"])]
                    if (before.st_size, before.st_mtime_ns) != marker["_verified_prediction_stat"]:
                        raise ValueError(f"Prediction changed after integrity verification: {path}")
                    count = 0
                    temperature = run.get("temperature", marker.get("temperature"))
                    consume = consume_chunked_dataset if engine == "chunked" else consume_legacy_dataset
                    kwargs = {"chunk_size": chunk_size} if engine == "chunked" else {}
                    count = consume(path, dataset, run, temperature, accumulators, group_info, thresholds,
                                    metadata, cluster_batch, flush_clusters, bootstrap, **kwargs)
                    expected_n = dataset.get("n_samples", marker.get("n_samples"))
                    if count == 0 or (expected_n is not None and count != int(expected_n)):
                        raise ValueError(f"Prediction row count mismatch: {path}: {count}")
                    after = path.stat()
                    if (before.st_size, before.st_mtime_ns) != (after.st_size, after.st_mtime_ns):
                        raise ValueError(f"Prediction changed during analysis: {path}")
                flush_clusters()
                for identity, accumulator in accumulators.items():
                    variant, level, key = identity
                    base_row = {"training_run_id": run["run_id"], "model": run["model"], "seed": int(run["seed"]),
                                "feature_scheme": run.get("feature_scheme", "full"), "probability_type": variant,
                                "group_level": level, "group_key": key, **group_info[identity]}
                    metrics, classes = accumulator.metrics()
                    intervals, uncertainty = {}, {"read_cluster_ci_available": False, "uncertainty_method": "not computed for this subgroup", "bootstrap_replicates": 0}
                    if level in ["overall", "template", "template_role"]:
                        seed_hash = int(hashlib.sha256(json.dumps([analysis_seed, run["run_id"], level, key]).encode()).hexdigest()[:16], 16)
                        intervals, uncertainty = cluster_intervals(connection, json.dumps((level, key), separators=(",", ":")),
                            metrics["n"], accumulator.real_read_rows, bootstrap, np.random.default_rng(seed_hash), replicate_writer, base_row)
                    top_intervals = {k: v for k, v in intervals.items() if not k.startswith("class_")}
                    rows.append({**base_row, **metrics, **top_intervals, **uncertainty})
                    for item in classes:
                        current = {k.removeprefix(f'class_{item["class_id"]}_'): v for k, v in intervals.items() if k.startswith(f'class_{item["class_id"]}_')}
                        class_uncertainty = uncertainty if item["class_id"] >= 4 else {**uncertainty,
                            "read_cluster_ci_available": False, "uncertainty_method": "not computed: class-specific cluster intervals are restricted to X/Y",
                            "bootstrap_replicates": 0}
                        class_writer.writerow({**base_row, **item, **current, **class_uncertainty,
                                           "wilson_scope": "descriptive window interval; correlated windows are not independent"})
                    for i in range(6):
                        for j in range(6):
                            confusion_row = {**base_row, "true_label": i, "true_class": CLASS_NAMES[i],
                                "predicted_label": j, "predicted_class": CLASS_NAMES[j], "count": int(accumulator.cm[i, j]),
                                "row_fraction": ratio(accumulator.cm[i, j], accumulator.cm[i].sum())}
                            confusion_writer.writerow(confusion_row)
                            if run["model"] == "litemamba" and run.get("feature_scheme", "full") == "full" and variant == "raw" and level == "template_role" and group_info[identity].get("role") == "target":
                                legacy_confusion_rows.append(confusion_row)
                    for index, (n, confidence_sum, correct) in enumerate(accumulator.bins):
                        calibration_writer.writerow({**base_row, "bin_left": index / 15, "bin_right": (index + 1) / 15,
                            "n": int(n), "mean_confidence": ratio(confidence_sum, n), "accuracy": ratio(correct, n)})
                    if level in ["overall", "template"]:
                        for item in classes[4:]:
                            for prevalence in PREVALENCES:
                                tpr, fpr = item["recall"], item["fpr"]
                                ppv = ratio(prevalence * tpr, prevalence * tpr + (1 - prevalence) * fpr) if tpr is not None and fpr is not None else None
                                prevalence_rows.append({**base_row, "class_name": item["class_name"], "assumed_prevalence": prevalence,
                                    "observed_tpr": tpr, "observed_fpr": fpr, "hypothetical_precision": ppv,
                                    "hypothetical_fdr": 1 - ppv if ppv is not None else None,
                                    "scope": "hypothetical fixed class-vs-rest TPR/FPR and rest composition; not measured low-prevalence performance"})
                for identity, accumulator in thresholds.items():
                    variant, level, key = identity
                    identity_row = {"training_run_id": run["run_id"], "model": run["model"], "seed": int(run["seed"]),
                                    "feature_scheme": run.get("feature_scheme", "full"), "probability_type": variant,
                                    "group_level": level, "group_key": key, **group_info[identity]}
                    threshold_rows.extend({**identity_row, **row} for row in accumulator.rows())
                for field, counts in metadata.items():
                    metadata_rows.append({"training_run_id": run["run_id"], "model": run["model"], "seed": int(run["seed"]),
                                          "metadata_field": field, "known_rows": counts["known"],
                                          "unknown_or_proxy_rows": counts["unknown"] + counts["unknown_or_proxy"]})
                connection.close()
                connection = None
                database.unlink()
        for name, level in [("overall_metrics", "overall"), ("per_template_metrics", "template"),
                            ("per_template_role_metrics", "template_role"), ("per_template_strand_metrics", "template_strand"),
                            ("per_template_source_run_strand_metrics", "template_source_run_strand"), ("per_template_pair_metrics", "template_pair"),
                            ("per_reference_metrics", "reference"), ("per_source_metrics", "source"),
                            ("per_run_metrics", "run"), ("per_strand_metrics", "strand"),
                            ("per_motif_metrics", "motif"), ("per_cohort_metrics", "cohort")]:
            write_table(work, name, [row for row in rows if row["group_level"] == level],
                        GROUP_COLUMNS + list(counts_metrics(np.zeros((6, 6), dtype=int))[0]) + ["nll", "brier", "ece_15bins"])
        write_table(work, "confidence_thresholds", threshold_rows)
        write_table(work, "prevalence_scenarios", prevalence_rows)
        write_table(work, "metadata_coverage", metadata_rows)
        write_table(work, "template_coverage", coverage["template_role_strand_presence"])
        summaries = seed_summaries(rows, runs)
        summaries.to_csv(work / "seed_summary.csv", index=False)
        summaries[summaries.group_level.eq("template_role")].to_csv(work / "plot_template_summary.csv", index=False)
        figure_tables(work, summaries, legacy_confusion_rows)
        notes = {"plan_id": plan.get("plan_id"), "analysis_seed": analysis_seed, "bootstrap_replicates": bootstrap,
                 "class_order": CLASS_NAMES, "task": "six-class external window evaluation",
                 "completed_training_runs": len(runs), "analysis_is_offline": True,
                 "thresholds": THRESHOLDS, "threshold_selection": "fixed in code; no test-set threshold fitting",
                 "calibration_metrics": "NLL clips true-class probabilities at float64 tiny; Brier is the sum over all six squared class errors (range 0–2); ECE uses 15 fixed equal-width top-label bins.",
                 "template_pairing": "cohort/template design aliases combine control and target for descriptive PPV; do not establish matched same-run molecular controls",
                 "source_pairing": "template_source_run_strand combines roles within cohort/template/source_dataset/sequencing_run_id/bam_strand. Source/direction matching can only be discussed when all identity fields are known; unknown remains unknown. template_pair is emitted only for a supplied known pair_id and retains template_group. Missing canonical/NCB denominators remain undefined; design aliases alone are not verified experimental pairing.",
                 "development_analysis_benchmark": {"scope": "Synthetic CPU-only offline-analysis development test; not model inference speed or a guarantee for actual external datasets. The baseline preceded the two additional matching strata.",
                    "windows": 100000, "training_runs": 1, "motifs": 1024, "bootstrap": 0, "group_levels": 10,
                    "logits_and_sha_validated": True, "wall_seconds": 44.02, "peak_rss_mib": 99.13},
                 "uncertainty": "read-cluster bootstrap only with complete real read IDs; missing identity suppresses cluster CI; Wilson intervals are descriptive windows",
                 "scope": ["Filename template/strand declarations do not establish sequencing run identity or independent read/template holdout.",
                           "A source/strand prediction control and source-matched experiments are separate evidence; output tables alone cannot rule out confounding.",
                           "Current package evaluates saved new six-class models only; five-class tasks are not fabricated by dropping a six-class output.",
                           "Raw and calibrated are two probability versions of the same trained run, not independent repeated experiments.",
                           "Per-template missing classes yield undefined denominators; fixed-six-label macro F1 may be small when only two classes occur.",
                           "No models, thresholds or seeds are selected from external test performance."]}
        notes["analysis_engine"] = engine
        if plan.get("correction_provenance") is not None:
            notes["correction_provenance"] = plan["correction_provenance"]
            notes["correction_interpretation"] = "Old XNA13–16 target results are invalid and superseded by reference-sequence correction. Differences are a data-correction audit, not a valid model-performance improvement comparison. All model seeds and frozen weights remain included."
            json_dump(work / "reference_correction_summary.json", {"plan_id": plan.get("plan_id"),
                "correction_provenance": plan["correction_provenance"], "interpretation": notes["correction_interpretation"]})
        json_dump(work / "reviewer_coverage.json", notes)
        json_dump(work / "coverage_report.json", coverage)
        json_dump(work / "analysis_manifest.json", {"status": "complete", "plan_id": plan.get("plan_id"),
            "analysis_seed": analysis_seed, "bootstrap": bootstrap, "engine": engine,
            "chunk_size": chunk_size if engine == "chunked" else None,
            "evaluation_plan_sha256": file_sha256(plan_path), "analysis_code_sha256": file_sha256(Path(__file__)),
            "prediction_inputs": integrity, "correction_provenance": plan.get("correction_provenance"),
            "files": sorted(p.name for p in work.iterdir() if p.is_file())})
        destination = root / "analysis_data"
        if destination.exists():
            backup = root / ("analysis_data.previous_" + str(time.time_ns()))
            destination.rename(backup)
        work.rename(destination)
        print(f"[ANALYSIS COMPLETE] {destination}", flush=True)
        return destination
    except BaseException as error:
        if connection is not None:
            connection.close()
        try:
            (work / "read_clusters.sqlite").unlink(missing_ok=True)
        except OSError:
            pass
        try:
            json_dump(work / "analysis_error.json", {"status": "failed", "plan_id": plan.get("plan_id"), "error": f"{type(error).__name__}: {error}",
                "note": "Original predictions are unchanged; rerun offline analysis after resolving the error. This incomplete analysis working directory is disposable."})
        except OSError:
            pass
        raise


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output_root", required=True)
    parser.add_argument("--bootstrap", type=int, default=200)
    parser.add_argument("--analysis_seed", type=int, default=42)
    parser.add_argument("--engine", choices=["legacy", "chunked"], default="chunked")
    parser.add_argument("--chunk_size", type=int, default=50000)
    args = parser.parse_args()
    analyze(args.output_root, bootstrap=args.bootstrap, analysis_seed=args.analysis_seed, engine=args.engine, chunk_size=args.chunk_size)


if __name__ == "__main__":
    main()
