#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Five-class prediction analysis, figures and repeated-run summaries without PyTorch.

All decisions are argmax over A,T,C,G and the scenario-specific NCB. Calibration parameters must already
have been fitted on validation data by the trainer. Analysis never fits them.
Evaluation units are windows; read-cluster bootstrap keeps every sampled read's
windows and each window's own truth, including mixed-label matched reads.
"""
import argparse
import hashlib
import itertools
import json
import math
import re
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.sparse import coo_matrix
from sklearn.metrics import (average_precision_score, precision_recall_curve,
                             roc_auc_score, roc_curve)

VERSION = "1.0.0"
from config_6_all_6 import CLASS_NAMES, NUM_CLASSES as N_CLASSES, SCENARIO, NCB_NAME, SCOPE_NOTE
UNKNOWN = {"", "unknown", "nan", "none", "null", "na", "n/a", "<na>"}
GROUPS = {"motif": "per_motif.csv", "chrom": "per_reference.csv",
          "template_group": "per_template.csv", "bam_strand": "per_strand.csv",
          "source_dataset": "per_source.csv", "run_id": "per_run.csv"}
SUMMARY_METRICS = ("accuracy", "macro_f1", "weighted_f1", "balanced_accuracy",
                   "macro_precision", "macro_recall", "macro_specificity",
                   "macro_fpr", "macro_fdr", "macro_roc_auc", "macro_average_precision",
                   "micro_roc_auc", "micro_average_precision", "nll", "brier", "ece_15bins")
SUMMARY_METRICS += tuple("class_{}_{}".format(k, key) for k in range(N_CLASSES)
                         for key in ("precision", "recall", "specificity", "fpr", "fdr", "f1", "roc_auc", "average_precision"))


def _json_ready(value):
    if isinstance(value, dict):
        return {str(k): _json_ready(v) for k, v in value.items()}
    if isinstance(value, (list, tuple, np.ndarray)):
        return [_json_ready(v) for v in value]
    if isinstance(value, (np.integer,)):
        return int(value)
    if isinstance(value, (np.floating, float)):
        return float(value) if math.isfinite(float(value)) else None
    if isinstance(value, (np.bool_,)):
        return bool(value)
    if isinstance(value, Path):
        return str(value)
    return value


def _save_json(path, value):
    Path(path).write_text(json.dumps(_json_ready(value), ensure_ascii=False,
                                    indent=2, allow_nan=False) + "\n", encoding="utf-8")


def _ratio(a, b):
    return float(a / b) if b else None


def _mean_defined(values):
    values = [v for v in values if v is not None and math.isfinite(float(v))]
    return float(np.mean(values)) if values else None


def _labels(labels):
    values = np.asarray(labels)
    if values.ndim != 1:
        raise ValueError("Labels must be a one-dimensional vector.")
    try:
        numbers = values.astype(np.float64)
    except (TypeError, ValueError) as exc:
        raise ValueError("Labels must be integers 0..4 for the explicit scenario.") from exc
    if (not np.isfinite(numbers).all() or np.any(numbers != np.floor(numbers))
            or np.any((numbers < 0) | (numbers >= N_CLASSES))):
        raise ValueError("Labels must be finite exact integers 0..4 for the explicit scenario.")
    return numbers.astype(np.int64)


def _probabilities(probabilities, n=None):
    values = np.asarray(probabilities, dtype=np.float64)
    if values.ndim != 2 or values.shape[1] != N_CLASSES:
        raise ValueError("Probabilities must have shape N x 5.")
    if n is not None and len(values) != n:
        raise ValueError("Label and probability counts differ.")
    if not np.isfinite(values).all() or np.any((values < 0) | (values > 1)):
        raise ValueError("Probabilities must be finite and within [0,1].")
    if not np.allclose(values.sum(1), 1, rtol=1e-5, atol=1e-6):
        raise ValueError("Every probability row must sum to one.")
    return values


def wilson_interval(successes, total, z=1.959963984540054):
    """Descriptive independent-window Wilson interval, not a cluster interval."""
    if not total:
        return None, None
    p = successes / total
    d = 1 + z * z / total
    center = (p + z * z / (2 * total)) / d
    radius = z * math.sqrt(p * (1 - p) / total + z * z / (4 * total * total)) / d
    return max(0.0, center - radius), min(1.0, center + radius)


def _binary_counts(tn, fp, fn, tp):
    tn, fp, fn, tp = (int(v) for v in (tn, fp, fn, tp))
    n, support, negatives = tn + fp + fn + tp, tp + fn, tn + fp
    predicted = tp + fp
    row = {"n": n, "support": support, "negative_support": negatives,
           "predicted_positive": predicted, "tn": tn, "fp": fp, "fn": fn, "tp": tp,
           "precision": _ratio(tp, predicted), "recall": _ratio(tp, support),
           "specificity": _ratio(tn, negatives), "fpr": _ratio(fp, negatives),
           "fnr": _ratio(fn, support), "fdr": _ratio(fp, predicted),
           "npv": _ratio(tn, tn + fn), "f1": _ratio(2 * tp, 2 * tp + fp + fn)}
    row["sensitivity"] = row["recall"]
    proportions = {"precision": (tp, predicted), "recall": (tp, support),
                   "specificity": (tn, negatives), "fpr": (fp, negatives),
                   "fnr": (fn, support), "fdr": (fp, predicted), "npv": (tn, tn + fn)}
    for key, (numerator, denominator) in proportions.items():
        row[key + "_wilson_low"], row[key + "_wilson_high"] = wilson_interval(numerator, denominator)
    return row


def _from_confusion(matrix):
    matrix = np.asarray(matrix, dtype=np.int64).reshape(N_CLASSES, N_CLASSES)
    support = matrix.sum(1)
    n = int(matrix.sum())
    per_class = []
    for k in range(N_CLASSES):
        tp, fp, fn = matrix[k, k], matrix[:, k].sum() - matrix[k, k], support[k] - matrix[k, k]
        row = _binary_counts(n - tp - fp - fn, fp, fn, tp)
        row.update(class_id=k, class_name=CLASS_NAMES[k])
        per_class.append(row)
    # Fixed five-class macro F1 treats an absent and never predicted class as zero,
    # matching sklearn f1_score(labels=range(N_CLASSES), average='macro', zero_division=0).
    # Undefined one-vs-rest rates remain null in their class rows and strict macros.
    result = {"n": n, "support": n, "correct": int(np.trace(matrix)),
              "errors": int(n - np.trace(matrix)), "accuracy": _ratio(np.trace(matrix), n),
              "error_rate": _ratio(n - np.trace(matrix), n),
              "macro_f1": float(np.mean([r["f1"] or 0 for r in per_class])) if n else None,
              "weighted_f1": float(sum((r["f1"] or 0) * r["support"] for r in per_class) / n) if n else None,
              "balanced_accuracy": _mean_defined([r["recall"] for r in per_class]),
              "present_class_count": int(np.count_nonzero(support)),
              "class_support": support.tolist(), "confusion_matrix": matrix.tolist(),
              "per_class": per_class}
    for metric in ("precision", "recall", "specificity", "fpr", "fdr"):
        values = [r[metric] for r in per_class]
        result["macro_" + metric] = _mean_defined(values) if all(v is not None for v in values) else None
        result["defined_class_macro_" + metric] = _mean_defined(values)
    result["accuracy_wilson_low"], result["accuracy_wilson_high"] = wilson_interval(result["correct"], n)
    result["error_rate_wilson_low"], result["error_rate_wilson_high"] = wilson_interval(result["errors"], n)
    return result


def _calibration_table(y, p, bins=15):
    prediction = p.argmax(1)
    # Yield one observed vector at a time: do not retain six N-row copies.
    def specifications():
        yield "confidence", -1, p.max(1), (prediction == y).astype(float)
        for k in range(N_CLASSES):
            yield "class_probability", k, p[:, k], (y == k).astype(float)
    rows = []
    for mode, k, values, observed in specifications():
        indices = np.minimum((values * bins).astype(int), bins - 1)
        count = np.bincount(indices, minlength=bins)
        prob_sum = np.bincount(indices, weights=values, minlength=bins)
        obs_sum = np.bincount(indices, weights=observed, minlength=bins)
        for i in range(bins):
            mean, rate = _ratio(prob_sum[i], count[i]), _ratio(obs_sum[i], count[i])
            lo, hi = wilson_interval(obs_sum[i], count[i])
            rows.append({"mode": mode, "class_id": k, "class_name": CLASS_NAMES[k] if k >= 0 else "all",
                         "bin": i, "bin_left": i / bins, "bin_right": (i + 1) / bins,
                         "count": int(count[i]), "mean_probability": mean, "observed_rate": rate,
                         "observed_rate_wilson_low": lo, "observed_rate_wilson_high": hi,
                         "absolute_gap": abs(mean - rate) if count[i] else None})
    return pd.DataFrame(rows)


def multiclass_metrics(y_true, probabilities):
    """JSON-safe five-class window metrics; undefined rates/AUCs are null.

    macro_f1 is the fixed-five-class sklearn convention (zero for an absent class
    with no predictions). Strict macro ROC/AP require every class to be present;
    defined_class_macro_* explicitly averages available one-vs-rest classes only.
    Multiclass Brier is mean(sum_k (p_k - onehot(y)_k)^2), range [0,2].
    """
    y, p = _labels(y_true), _probabilities(probabilities, len(y_true))
    pred = p.argmax(1)
    matrix = np.bincount(y * N_CLASSES + pred, minlength=N_CLASSES ** 2).reshape(N_CLASSES, N_CLASSES)
    result = _from_confusion(matrix)
    auc, ap = [], []
    for k, row in enumerate(result["per_class"]):
        binary = y == k
        both = bool(binary.any() and (~binary).any())
        row["roc_auc"] = float(roc_auc_score(binary, p[:, k])) if both else None
        row["average_precision"] = float(average_precision_score(binary, p[:, k])) if both else None
        auc.append(row["roc_auc"])
        ap.append(row["average_precision"])
    for key, values in (("roc_auc", auc), ("average_precision", ap)):
        result["macro_" + key] = _mean_defined(values) if all(v is not None for v in values) else None
        result["defined_class_macro_" + key] = _mean_defined(values)
        result["weighted_" + key] = (sum(v * int(np.sum(y == k)) for k, v in enumerate(values) if v is not None) / len(y)
                                      if len(y) and all(values[k] is not None for k in np.unique(y)) else None)
    if len(y):
        target = np.eye(N_CLASSES, dtype=np.uint8)[y]
        result["micro_roc_auc"] = float(roc_auc_score(target.ravel(), p.ravel()))
        result["micro_average_precision"] = float(average_precision_score(target.ravel(), p.ravel()))
        result["nll"] = float(-np.log(np.clip(p[np.arange(len(y)), y], np.finfo(float).tiny, 1)).mean())
        result["brier"] = float((np.sum(p * p, axis=1) - 2 * p[np.arange(len(y)), y] + 1).mean())
        table = _calibration_table(y, p)
        confidence = table[table["mode"].eq("confidence")]
        result["ece_15bins"] = float((confidence["count"] * confidence["absolute_gap"].fillna(0)).sum() / len(y))
        result["classwise_ece_15bins"] = float((table.loc[table["class_id"].ge(0), "count"] *
                                                     table.loc[table["class_id"].ge(0), "absolute_gap"].fillna(0)).sum() / (len(y) * N_CLASSES))
    else:
        result.update({key: None for key in ("micro_roc_auc", "micro_average_precision", "nll", "brier", "ece_15bins", "classwise_ece_15bins")})
    result["auroc"], result["auprc"], result["ece"] = result["macro_roc_auc"], result["macro_average_precision"], result["ece_15bins"]
    return _json_ready(result)


def _known(series):
    return ~series.astype(str).str.strip().str.lower().isin(UNKNOWN)


def _prepare_predictions(path):
    # Ignore unused logits/embedding/provenance columns during analysis; the full
    # metadata-rich prediction CSV remains unchanged on disk. Parse probabilities
    # numerically rather than retaining millions of Python numeric strings.
    header = pd.read_csv(path, compression="infer", nrows=0).columns.tolist()
    identifiers = set(GROUPS) | {"read_id", "sample_id", "original_kmer", "read_id_kind",
        "read_identity_basis", "original_read_id", "reference_identity_known",
        "source_file", "pair_id", "sample_index", "kmer", "prediction", "label",
        "scenario", "class_name", "positive_source_sha256", "positive_data_record", "merge_input_role"}
    probability_columns = [c for c in header if re.fullmatch(r"(?:calibrated_)?prob_[0-9]+", c)]
    usecols = [c for c in header if c in identifiers or c in probability_columns]
    numeric = set(probability_columns)
    low_cardinality = set(GROUPS) | {"scenario", "class_name", "read_id_kind", "read_identity_basis",
        "reference_identity_known", "merge_input_role", "original_kmer", "kmer", "positive_source_sha256"}
    dtype = {c: ("float64" if c in numeric else "category" if c in low_cardinality else str) for c in usecols}
    frame = pd.read_csv(path, compression="infer", low_memory=False, usecols=usecols,
                        dtype=dtype, keep_default_na=False)
    if "label" not in frame or not len(frame):
        raise ValueError("Prediction CSV must contain label and at least one row.")
    frame["label"] = _labels(frame["label"].to_numpy())
    if "scenario" not in frame or not frame["scenario"].eq(SCENARIO).all():
        raise ValueError("Prediction scenario does not match CPLX_FIVECLASS_SCENARIO.")
    if "class_name" in frame and not frame["class_name"].eq(frame["label"].map(dict(enumerate(CLASS_NAMES)))).all():
        raise ValueError("Prediction class names do not match scenario-specific labels.")
    variants = {}
    for name, prefix in (("raw", "prob_"), ("calibrated", "calibrated_prob_")):
        if any(re.fullmatch(re.escape(prefix) + r"[0-9]+", str(c)) and int(str(c)[len(prefix):]) >= N_CLASSES for c in frame.columns):
            raise ValueError("Extra probability columns belong to another class definition: " + prefix)
        columns = [prefix + str(k) for k in range(N_CLASSES)]
        available = [c in frame for c in columns]
        if any(available) and not all(available):
            raise ValueError("Incomplete five-class probability columns: " + prefix)
        if all(available):
            variants[name] = _probabilities(frame[columns].to_numpy(dtype=float), len(frame))
    if "raw" not in variants:
        raise ValueError("CSV requires prob_0..prob_4; probabilities are never guessed from predictions.")
    columns = set(GROUPS) | {"read_id", "sample_id", "original_kmer", "read_id_kind", "read_identity_basis",
                            "original_read_id", "reference_identity_known", "source_file", "pair_id"}
    for column in columns:
        if column not in frame:
            frame[column] = "unknown"
        if isinstance(frame[column].dtype, pd.CategoricalDtype) and "unknown" not in frame[column].cat.categories:
            frame[column] = frame[column].cat.add_categories(["unknown"])
        frame.loc[~_known(frame[column]), column] = "unknown"
    if "sample_index" not in frame:
        frame["sample_index"] = np.arange(len(frame))
    if frame["sample_id"].eq("unknown").all():
        frame["sample_id"] = ["row:" + str(i) for i in range(len(frame))]
    elif frame["sample_id"].duplicated().any():
        raise ValueError("sample_id must be unique for matched-run analysis.")
    if frame["motif"].eq("unknown").any() and "kmer" in frame:
        frame.loc[frame["motif"].eq("unknown"), "motif"] = frame.loc[frame["motif"].eq("unknown"), "kmer"]
    missing = frame["read_id"].eq("unknown")
    frame["_analysis_read_id"] = frame["read_id"]
    frame.loc[missing, "_analysis_read_id"] = ["unresolved-row:" + str(i) for i in frame.index[missing]]
    proxy = frame["read_id_kind"].eq("positive_row_proxy")
    if missing.any():
        raise ValueError("Prediction rows require an explicit real read ID or a fully disclosed positive row proxy.")
    claimed_proxy = proxy | frame["read_id"].str.startswith("positive-row:")
    if claimed_proxy.any():
        part = frame.loc[claimed_proxy]
        required = {"positive_source_sha256", "positive_data_record", "merge_input_role",
                    "read_identity_basis", "reference_identity_known", "original_read_id"}
        if not required.issubset(part.columns):
            raise ValueError("Positive row proxies require complete original provenance flags.")
        digest = part["positive_source_sha256"].astype(str)
        ordinal = part["positive_data_record"].astype(str)
        valid = (part["label"].eq(4) & part["read_id_kind"].eq("positive_row_proxy")
            & part["merge_input_role"].eq("positive")
            & part["read_identity_basis"].eq("assumed_one_csv_row_per_positive_read")
            & part["reference_identity_known"].eq("false") & part["original_read_id"].eq("unknown")
            & part["source_dataset"].eq("XNA_replenishment") & part["chrom"].eq("unknown")
            & part["bam_strand"].eq("unknown") & part["run_id"].eq("unknown")
            & digest.str.fullmatch(r"[0-9a-f]{64}") & ordinal.str.fullmatch(r"[1-9][0-9]*")
            & part["read_id"].eq("positive-row:" + digest + ":" + ordinal))
        if not valid.all() or part["read_id"].duplicated().any():
            raise ValueError("Malformed, repeated or undisclosed positive row proxy in prediction CSV.")
    read_labels = frame.groupby("_analysis_read_id", sort=False)["label"].nunique()
    notes = {"unknown_read_id_rows": int(missing.sum()), "positive_row_proxy_rows": int(proxy.sum()),
             "positive_row_proxy_clusters": int(frame.loc[proxy, "_analysis_read_id"].nunique()),
             "real_read_id_rows": int((~proxy & ~missing).sum()),
             "mixed_truth_read_clusters": int(read_labels.gt(1).sum()),
             "identity_scope": "Real negative read IDs; positive original IDs when supplied, otherwise disclosed one-CSV-row-per-positive-read assumption. Proxy clusters do not establish original positive UUID independence."}
    # Numeric arrays are returned separately. Keeping the same probability
    # columns in the metadata frame would retain an unnecessary second Nx10
    # block. Provenance remains complete in the original prediction CSV.
    frame.drop(columns=probability_columns + [c for c in ("positive_source_sha256",
        "positive_data_record", "merge_input_role", "original_read_id", "source_file",
        "read_identity_basis", "reference_identity_known", "pair_id") if c in frame], inplace=True)
    return frame, variants, notes


def _flat_count_metrics(matrix):
    metrics = _from_confusion(matrix)
    row = {k: metrics[k] for k in ("accuracy", "macro_f1", "weighted_f1", "balanced_accuracy")}
    for class_row in metrics["per_class"]:
        row.update({"class_{}_{}".format(class_row["class_id"], key): class_row[key]
                    for key in ("precision", "recall", "specificity", "fpr", "fdr", "f1")})
    return row


def read_cluster_bootstrap(y_true, probabilities, read_ids, *, bootstrap=200, seed=42):
    """Resample read clusters, retaining all windows and their separate truths.

    Sufficient statistics avoid copying large signal/prediction tables. This
    bootstrap reports count-based metrics, NLL and Brier; AUC/ECE are point
    estimates with saved scores, not falsely labelled bootstrap intervals.
    """
    if int(bootstrap) != bootstrap or bootstrap < 0:
        raise ValueError("bootstrap must be a nonnegative integer.")
    y, p = _labels(y_true), _probabilities(probabilities, len(y_true))
    # Object pointers retain exact IDs without expanding every UUID/proxy into a
    # fixed-width UTF-32 array (which can itself exceed available RAM).
    ids = np.asarray(read_ids, dtype=object)
    if len(ids) != len(y):
        raise ValueError("read_ids and label counts differ.")
    codes, unique = pd.factorize(ids, sort=False)
    if not len(unique):
        return {"method": "read-cluster percentile bootstrap", "cluster_count": 0, "replicates": bootstrap, "intervals": {}}, pd.DataFrame()
    pred = p.argmax(1)
    row_idx = np.r_[codes, codes, codes, codes]
    col_idx = np.r_[y * N_CLASSES + pred, np.full(len(y), N_CLASSES ** 2), np.full(len(y), N_CLASSES ** 2 + 1), np.full(len(y), N_CLASSES ** 2 + 2)]
    nll = -np.log(np.clip(p[np.arange(len(y)), y], np.finfo(float).tiny, 1))
    brier = np.sum(p * p, axis=1) - 2 * p[np.arange(len(y)), y] + 1
    data = np.r_[np.ones(len(y)), np.ones(len(y)), nll, brier]
    counts = coo_matrix((data, (row_idx, col_idx)), shape=(len(unique), N_CLASSES ** 2 + 3)).tocsr().T.tocsr()
    rng, rows = np.random.default_rng(seed), []
    for iteration in range(int(bootstrap)):
        draws = rng.integers(0, len(unique), size=len(unique))
        weights = np.bincount(draws, minlength=len(unique))
        sampled = np.asarray(counts @ weights).ravel()
        metrics = _flat_count_metrics(np.rint(sampled[:N_CLASSES ** 2]).astype(np.int64).reshape(N_CLASSES, N_CLASSES))
        metrics.update(nll=_ratio(sampled[N_CLASSES ** 2 + 1], sampled[N_CLASSES ** 2]), brier=_ratio(sampled[N_CLASSES ** 2 + 2], sampled[N_CLASSES ** 2]))
        rows.append({"replicate": iteration, "sampled_window_count": int(sampled[N_CLASSES ** 2]), **metrics})
    replicates = pd.DataFrame(rows)
    intervals = {}
    for key in replicates:
        if key in ("replicate", "sampled_window_count"):
            continue
        values = pd.to_numeric(replicates[key], errors="coerce").dropna().to_numpy()
        low, high = np.quantile(values, [0.025, 0.975]) if len(values) else (None, None)
        intervals[key] = {"ci_low": low, "ci_high": high, "valid_replicates": len(values)}
    report = {"method": "Percentile bootstrap of read_id clusters; retain all windows and each window's truth.",
              "cluster_count": len(unique), "replicates": int(bootstrap), "seed": seed,
              "confidence_level": 0.95, "intervals": intervals,
              "scope": "Count metrics, NLL, Brier only. Does not represent variation across runs/templates or repeated training seeds."}
    return _json_ready(report), replicates


def _sequence_features(motif):
    motif = str(motif).upper()
    flank = motif[:3] + motif[4:] if len(motif) == 6 else ""
    canonical = [base for base in flank if base in "ATCG"]
    return {"flank_gc_fraction": _ratio(sum(base in "GC" for base in canonical), len(canonical)),
            "flank_gc_count": sum(base in "GC" for base in canonical),
            "known_flank_bases": len(canonical), "contains_unknown_flank": any(base not in "ATCG" for base in flank),
            "max_flank_homopolymer": max((len(list(group)) for _, group in itertools.groupby(flank)), default=0),
            **{"position_{}".format(i + 1): motif[i] if len(motif) == 6 else "unknown" for i in (0, 1, 2, 4, 5)}}


def _group_tables(frame, variants, column, *, bootstrap=0, seed=42):
    overall, classes, confusions = [], [], []
    y = frame["label"].to_numpy()
    predictions = {name: p.argmax(1) for name, p in variants.items()}
    for group_value, indices in frame.groupby(column, sort=True, dropna=False, observed=True).indices.items():
        indices = np.asarray(indices)
        for name, p in variants.items():
            matrix = np.bincount(y[indices] * N_CLASSES + predictions[name][indices], minlength=N_CLASSES ** 2).reshape(N_CLASSES, N_CLASSES)
            metrics = _from_confusion(matrix)
            keys = ("n", "correct", "errors", "accuracy", "error_rate", "macro_f1", "weighted_f1",
                    "balanced_accuracy", "present_class_count", "accuracy_wilson_low", "accuracy_wilson_high",
                    "error_rate_wilson_low", "error_rate_wilson_high")
            row = {column: group_value, "probability_type": name, **{key: metrics[key] for key in keys},
                   "unique_read_count": int(frame["_analysis_read_id"].iloc[indices].nunique()),
                   "identity_known": str(group_value).lower() not in UNKNOWN,
                   "interval_scope": "Descriptive window Wilson intervals; correlated windows are not independent.",
                   "precision_scope": "Conditional on this subgroup, not population PPV."}
            if column == "motif":
                row.update(_sequence_features(group_value))
            interval_report = {"intervals": {}}
            if bootstrap:
                interval_report, _ = read_cluster_bootstrap(y[indices], p[indices],
                    frame["_analysis_read_id"].iloc[indices].to_numpy(), bootstrap=bootstrap, seed=seed)
            row["read_cluster_bootstrap_replicates"] = int(bootstrap)
            row["read_cluster_interval_method"] = "read-cluster percentile bootstrap" if bootstrap else "not computed; rerun offline with --subgroup_bootstrap"
            for key in ("accuracy", "macro_f1", "weighted_f1", "balanced_accuracy"):
                for suffix in ("ci_low", "ci_high", "valid_replicates"):
                    row[key + "_read_cluster_" + suffix] = interval_report["intervals"].get(key, {}).get(suffix)
            overall.append(row)
            for class_row in metrics["per_class"]:
                class_row = {column: group_value, "probability_type": name, **class_row,
                             "read_cluster_bootstrap_replicates": int(bootstrap)}
                for key in ("precision", "recall", "specificity", "fpr", "fdr", "f1"):
                    for suffix in ("ci_low", "ci_high", "valid_replicates"):
                        class_row[key + "_read_cluster_" + suffix] = interval_report["intervals"].get(
                            "class_{}_{}".format(class_row["class_id"], key), {}).get(suffix)
                classes.append(class_row)
            confusions.extend({column: group_value, "probability_type": name, "true_label": i,
                               "true_name": CLASS_NAMES[i], "predicted_label": j, "predicted_name": CLASS_NAMES[j],
                               "count": int(matrix[i, j]), "row_fraction": _ratio(matrix[i, j], matrix[i].sum()),
                               "column_fraction": _ratio(matrix[i, j], matrix[:, j].sum())}
                              for i, j in itertools.product(range(N_CLASSES), repeat=2))
    return pd.DataFrame(overall), pd.DataFrame(classes), pd.DataFrame(confusions)


def _confidence_tables(y, p):
    rows, by_class = [], []
    confidence, prediction = p.max(1), p.argmax(1)
    for tau in sorted(set([0.0, 1 / N_CLASSES] + np.linspace(.2, .95, 16).tolist() + [.98, .99, .995, .999])):
        kept = confidence >= tau
        matrix = np.bincount(y[kept] * N_CLASSES + prediction[kept], minlength=N_CLASSES ** 2).reshape(N_CLASSES, N_CLASSES)
        metrics = _from_confusion(matrix)
        row = {"confidence_threshold": tau, "total_count": len(y), "retained_count": int(kept.sum()),
               "abstained_count": int((~kept).sum()), "coverage": _ratio(kept.sum(), len(y)),
               "accuracy_among_retained": metrics["accuracy"], "macro_f1_among_retained": metrics["macro_f1"],
               "overall_correct_fraction": _ratio(metrics["correct"], len(y))}
        row["coverage_wilson_low"], row["coverage_wilson_high"] = wilson_interval(kept.sum(), len(y))
        rows.append(row)
        for k, class_row in enumerate(metrics["per_class"]):
            by_class.append({"confidence_threshold": tau, **class_row,
                             "all_class_support": int(np.sum(y == k)),
                             "overall_recall": _ratio(class_row["tp"], np.sum(y == k)),
                             "scope": "Abstained windows are errors for overall recall; retained recall conditions on retention."})
    return pd.DataFrame(rows), pd.DataFrame(by_class)


def _prevalence_scenarios(per_class):
    rows = []
    for class_row in per_class:
        tpr, fpr = class_row["recall"], class_row["fpr"]
        for prior in (.0001, .001, .01, .05, .1, .5):
            ppv = _ratio(prior * tpr, prior * tpr + (1 - prior) * fpr) if tpr is not None and fpr is not None else None
            rows.append({"class_id": class_row["class_id"], "class_name": class_row["class_name"],
                         "assumed_class_prevalence": prior, "conditional_tpr": tpr, "conditional_fpr": fpr,
                         "expected_ppv": ppv, "expected_fdr": 1 - ppv if ppv is not None else None,
                         "assumption": "Hypothetical class-vs-rest scenario, unchanged conditional TPR/FPR and composition of the rest. Not five-class prior correction or measured low-frequency detection."})
    return pd.DataFrame(rows)


def _ncb_rows(y, prediction):
    rows = []
    for name, positive in ((NCB_NAME + "_vs_canonical_argmax", (4,)),):
        truth, call = np.isin(y, positive), np.isin(prediction, positive)
        row = _binary_counts(np.sum(~truth & ~call), np.sum(~truth & call), np.sum(truth & ~call), np.sum(truth & call))
        row.update(comparison=name, canonical_window_count=int(np.sum(y < 4)),
                   canonical_false_positive_calls=int(np.sum((y < 4) & call)),
                   canonical_false_positive_rate=_ratio(np.sum((y < 4) & call), np.sum(y < 4)),
                   scope="Window-level five-class argmax-derived NCB calls; all negatives are the four canonical classes.")
        rows.append(row)
    return pd.DataFrame(rows)


def analyze_predictions(predictions_path, output_dir, *, bootstrap=200, seed=42, plots=True,
                        make_plots=None, run_metadata=None, subgroup_bootstrap=0):
    """Export reproducible metric tables and figures from saved probabilities."""
    if make_plots is not None:
        plots = make_plots
    if int(bootstrap) != bootstrap or bootstrap < 0:
        raise ValueError("bootstrap must be a nonnegative integer.")
    if int(subgroup_bootstrap) != subgroup_bootstrap or subgroup_bootstrap < 0:
        raise ValueError("subgroup_bootstrap must be a nonnegative integer.")
    predictions_path = Path(predictions_path).expanduser().resolve()
    output_dir = Path(output_dir).expanduser().resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    frame, variants, notes = _prepare_predictions(predictions_path)
    y = frame["label"].to_numpy()
    files = []

    def save(name, table):
        if (output_dir / name).resolve() == predictions_path:
            raise ValueError("Analysis output must not overwrite input predictions.")
        table.to_csv(output_dir / name, index=False)
        files.append(name)

    # Headline metrics use every score. Curves retain up to 4097 actual points
    # including both endpoints; full probability/logit arrays and prediction CSV
    # remain available for exact offline regeneration without neural inference.
    curve_files = {"roc_curve": "roc_curve.csv", "pr_curve": "pr_curve.csv"}
    for filename in curve_files.values():
        (output_dir / filename).unlink(missing_ok=True)
    def save_curve(name, table):
        filename = curve_files[name]
        if len(table) > 4097:
            retained = np.unique(np.linspace(0, len(table) - 1, 4097, dtype=np.int64))
            table = table.iloc[retained]
        target = output_dir / filename
        table.to_csv(target, index=False, mode="a", header=not target.exists(), chunksize=100000)
        if filename not in files:
            files.append(filename)

    overall, uncertainty = {}, {}
    tables = {name: [] for name in ("per_class_metrics", "confusion_matrix", "roc_curve", "pr_curve", "calibration_bins",
                                   "confidence_threshold_scan", "confidence_per_class", "prevalence_scenarios",
                                   "ncb_calls", "bootstrap_replicates", "class_threshold_scan", "probability_histogram")}
    for name, p in variants.items():
        metrics = multiclass_metrics(y, p)
        overall[name] = metrics
        uncertainty[name], replicates = read_cluster_bootstrap(y, p, frame["_analysis_read_id"].to_numpy(), bootstrap=bootstrap, seed=seed)
        per_class = pd.DataFrame(metrics["per_class"])
        for key in ("precision", "recall", "specificity", "fpr", "fdr", "f1"):
            for suffix in ("ci_low", "ci_high", "valid_replicates"):
                per_class[key + "_read_cluster_" + suffix] = [uncertainty[name]["intervals"].get("class_{}_{}".format(k, key), {}).get(suffix) for k in range(N_CLASSES)]
        tables["per_class_metrics"].append(per_class.assign(probability_type=name))
        tables["bootstrap_replicates"].append(replicates.assign(probability_type=name))
        matrix = np.asarray(metrics["confusion_matrix"])
        tables["confusion_matrix"].append(pd.DataFrame([
            {"probability_type": name, "true_label": i, "true_name": CLASS_NAMES[i], "predicted_label": j,
             "predicted_name": CLASS_NAMES[j], "count": int(matrix[i, j]),
             "row_fraction": _ratio(matrix[i, j], matrix[i].sum()), "column_fraction": _ratio(matrix[i, j], matrix[:, j].sum())}
            for i, j in itertools.product(range(N_CLASSES), repeat=2)]))
        for k in range(N_CLASSES):
            binary = y == k
            if binary.any() and (~binary).any():
                fpr, tpr, thresholds = roc_curve(binary, p[:, k], drop_intermediate=False)
                save_curve("roc_curve", pd.DataFrame({"probability_type": name, "class_id": k, "class_name": CLASS_NAMES[k],
                                                         "fpr": fpr, "tpr": tpr, "threshold": thresholds}))
                precision, recall, thresholds = precision_recall_curve(binary, p[:, k])
                save_curve("pr_curve", pd.DataFrame({"probability_type": name, "class_id": k, "class_name": CLASS_NAMES[k],
                                                        "precision": precision, "recall": recall, "threshold": np.r_[thresholds, np.nan]}))
            negatives, positives = np.sort(p[~binary, k]), np.sort(p[binary, k])
            threshold_rows = []
            for tau in sorted(set(np.linspace(0, 1, 101).tolist() + [.99, .995, .999])):
                tp = len(positives) - np.searchsorted(positives, tau, side="left")
                fp = len(negatives) - np.searchsorted(negatives, tau, side="left")
                threshold_rows.append({"probability_type": name, "class_id": k, "class_name": CLASS_NAMES[k],
                                       "class_probability_threshold": tau,
                                       "decision_scope": "Independent one-vs-rest threshold; multiple classes can pass, distinct from five-class argmax.",
                                       **_binary_counts(len(negatives) - fp, fp, len(positives) - tp, tp)})
            tables["class_threshold_scan"].append(pd.DataFrame(threshold_rows))
            for true_class in range(N_CLASSES):
                counts, edges = np.histogram(p[y == true_class, k], bins=np.linspace(0, 1, 51))
                tables["probability_histogram"].append(pd.DataFrame({"probability_type": name, "score_class_id": k,
                    "true_class_id": true_class, "bin_left": edges[:-1], "bin_right": edges[1:], "count": counts}))
        micro_truth = np.eye(N_CLASSES, dtype=np.uint8)[y].ravel()
        micro_score = p.ravel()
        fpr, tpr, thresholds = roc_curve(micro_truth, micro_score, drop_intermediate=False)
        save_curve("roc_curve", pd.DataFrame({"probability_type": name, "class_id": -1, "class_name": "micro",
                                                 "fpr": fpr, "tpr": tpr, "threshold": thresholds}))
        precision, recall, thresholds = precision_recall_curve(micro_truth, micro_score)
        save_curve("pr_curve", pd.DataFrame({"probability_type": name, "class_id": -1, "class_name": "micro",
                                                "precision": precision, "recall": recall, "threshold": np.r_[thresholds, np.nan]}))
        tables["calibration_bins"].append(_calibration_table(y, p).assign(probability_type=name))
        confidence, confidence_class = _confidence_tables(y, p)
        tables["confidence_threshold_scan"].append(confidence.assign(probability_type=name))
        tables["confidence_per_class"].append(confidence_class.assign(probability_type=name))
        tables["prevalence_scenarios"].append(_prevalence_scenarios(metrics["per_class"]).assign(probability_type=name))
        tables["ncb_calls"].append(_ncb_rows(y, p.argmax(1)).assign(probability_type=name))
    for name, parts in tables.items():
        if name in curve_files and (output_dir / curve_files[name]).exists():
            continue
        # Header-only tables remain readable even when all class AUCs are undefined.
        columns = {"roc_curve": ["probability_type", "class_id", "class_name", "fpr", "tpr", "threshold"],
                   "pr_curve": ["probability_type", "class_id", "class_name", "precision", "recall", "threshold"],
                   "bootstrap_replicates": ["probability_type", "replicate", "accuracy", "macro_f1"]}.get(name, [])
        save(name + ".csv", pd.concat(parts, ignore_index=True) if parts else pd.DataFrame(columns=columns))
    coverage = []
    motif_table = None
    for column, filename in GROUPS.items():
        group_bootstrap = subgroup_bootstrap if column in ("motif", "chrom", "template_group") else 0
        table, class_table, confusion_table = _group_tables(frame, variants, column, bootstrap=group_bootstrap, seed=seed)
        save(filename, table)
        save(filename.replace(".csv", "_by_class.csv"), class_table)
        save(filename.replace(".csv", "_confusion.csv"), confusion_table)
        if column == "motif":
            motif_table = table
        known = _known(frame[column])
        coverage.append({"metadata_field": column, "known_rows": int(known.sum()), "unknown_rows": int((~known).sum()),
                         "known_unique_groups": int(frame.loc[known, column].nunique()),
                         "unknown_is_real_reference": False if column in ("chrom", "template_group") else None})
    save("metadata_coverage.csv", pd.DataFrame(coverage))
    ranked = motif_table.sort_values(["probability_type", "error_rate", "n"], ascending=[True, False, False]).copy()
    ranked.insert(0, "rank_within_probability_type", ranked.groupby("probability_type").cumcount() + 1)
    save("worst_motifs.csv", ranked)
    # Aggregate observed flank attributes with their actual window support.
    flank_lookup = {motif: str(_sequence_features(motif)["flank_gc_count"]) for motif in frame["motif"].unique()}
    flank = frame["motif"].map(flank_lookup).astype(str)
    frame["flank_gc_count"] = flank
    flank_metrics, _, _ = _group_tables(frame, variants, "flank_gc_count")
    save("per_flank_gc_count.csv", flank_metrics)
    read_summary = frame.groupby("_analysis_read_id", sort=False).agg(
        read_id=("read_id", "first"), window_count=("label", "size"), distinct_true_classes=("label", "nunique"),
        read_id_kind=("read_id_kind", "first"))
    save("read_cluster_summary.csv.gz", read_summary.reset_index().rename(columns={"_analysis_read_id": "analysis_read_id"}))
    sha = hashlib.sha256()
    with predictions_path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            sha.update(chunk)
    report = {"schema_version": VERSION, "task": "five_class_window_classification", "scenario": SCENARIO, "class_definition": dict(enumerate(CLASS_NAMES)),
              "predictions_path": str(predictions_path), "predictions_sha256": sha.hexdigest(),
              "decision": "argmax(prob_0..prob_4), ties choose smallest class ID", "row_count": len(frame),
              "analysis_read_count": int(frame["_analysis_read_id"].nunique()), "input_notes": notes,
              "subgroup_bootstrap": {"replicates": int(subgroup_bootstrap), "fields": ["motif", "chrom", "template_group"],
                                     "scope": "Each subgroup resamples its read clusters, retaining all subgroup windows and their own labels."},
              "overall": overall, "raw": overall["raw"], "calibrated": overall.get("calibrated"),
              "uncertainty": uncertainty, "metadata": run_metadata or {}, "files": files + ["metrics.json"],
              "analysis_memory_policy": "Numeric probability parsing; categorical subgroup metadata; bounded ROC/PR CSVs written per curve; rendering uses first/last points per 1/2000 x-bin only.",
              "curve_export_policy": {"max_points_per_curve": 4097, "selection": "Uniform indices of actual full-curve points, both endpoints retained; no interpolation.",
                  "headline_auc_ap": "Computed from all saved probabilities, never from thinned curve points.",
                  "exact_offline_regeneration": "Use retained predictions.csv.gz or full probabilities/logits arrays; no model training or neural inference is needed."},
              "notes": [
                  "Metrics evaluate windows/target bases, not a single class per read. Negative reads can contain multiple true classes; no averaging or majority vote collapses their truths.",
                  "Negative windows are grouped by real read ID. Positive row proxies, if present, retain the disclosed one-row-per-read design assumption and are never described as independently verified original UUIDs.",
                  "Calibration is evaluated from saved calibrated_prob_* only; analysis fits no parameters or thresholds on test data.",
                  "Strict macro ROC AUC/AP and strict macro rates are null when required class denominators are absent. Fixed-five-class macro F1 uses zero for a class with no true or predicted samples. Defined-class averages are labelled separately.",
                  "Overall and per-class read-cluster bootstrap intervals preserve within-read window dependence. Wilson intervals in subgroup tables are descriptive window intervals, not evidence of independent windows.",
                  "Cluster bootstrap intervals cover count-based metrics/NLL/Brier; AUC and ECE are point estimates. Run/template correlations require independent holdout experiments and are not covered by read bootstrap.",
                  "Unknown reference/template is a reporting category, not one actual biological reference; unknown data cannot establish reference/run/strand isolation.",
                  SCOPE_NOTE,
                  "Hypothetical per-class prevalence scenarios assume fixed class-vs-rest TPR/FPR and rest composition; they are neither full five-class prior correction nor measured rare-base detection.",
                  "Subgroup precision is conditional on that subgroup. Worst-motif rankings must be interpreted with saved support and uncertainty, especially for sparse groups."]}
    if "prediction" in frame:
        report["input_notes"]["supplied_prediction_disagrees_with_argmax_rows"] = int(np.sum(pd.to_numeric(frame["prediction"], errors="coerce").to_numpy() != variants["raw"].argmax(1)))
    if plots:
        report["plots"] = make_basic_plots(output_dir)
    _save_json(output_dir / "metrics.json", report)
    return _json_ready(report)


def make_basic_plots(output_dir):
    """Redraw exclusively from the CSV tables; no weights or PyTorch required."""
    import matplotlib
    matplotlib.use("Agg", force=True)
    import matplotlib.pyplot as plt
    output_dir, names = Path(output_dir), []

    def finish(fig, name):
        fig.tight_layout()
        for suffix in ("png", "pdf"):
            path = output_dir / (name + "." + suffix)
            fig.savefig(path, dpi=220, bbox_inches="tight")
            names.append(path.name)
        plt.close(fig)

    for filename, x, y, xlabel, ylabel, output in (
        ("roc_curve.csv", "fpr", "tpr", "False positive rate", "True positive rate", "roc_ovr"),
        ("pr_curve.csv", "recall", "precision", "Recall", "Precision", "pr_ovr")):
        # Rendering uses at most first/last point per x bin and curve. The
        # CSV contains bounded actual curve points; full scores are retained.
        points = {}
        row_number = 0
        for chunk in pd.read_csv(output_dir / filename, chunksize=100000):
            for identity, part in chunk.groupby(["probability_type", "class_name"], sort=False):
                bins = np.minimum(np.maximum((part[x].to_numpy() * 2000).astype(np.int32), 0), 2000)
                first = np.r_[True, bins[1:] != bins[:-1]]
                last = np.r_[bins[:-1] != bins[1:], True]
                for index in np.flatnonzero(first | last):
                    row = part.iloc[int(index)]
                    key = (str(identity[0]), str(identity[1]), int(bins[index]))
                    value = (int(part.index[index]), float(row[x]), float(row[y]))
                    if key not in points:
                        points[key] = [value, value]
                    else:
                        points[key][1] = value
            row_number += len(chunk)
        plotted = []
        for (variant, class_name, _), pair in points.items():
            for position, xx, yy in pair:
                plotted.append({"probability_type": variant, "class_name": class_name, "_position": position, x: xx, y: yy})
        table = pd.DataFrame(plotted).sort_values("_position") if plotted else pd.DataFrame(columns=["probability_type", "class_name", x, y])
        for variant, part in table.groupby("probability_type", sort=False):
            fig, ax = plt.subplots(figsize=(4.8, 4))
            for class_name, rows in part.groupby("class_name", sort=False):
                ax.plot(rows[x], rows[y], label=class_name, linewidth=1.2)
            if x == "fpr":
                ax.plot([0, 1], [0, 1], "--", color="0.7", linewidth=.8)
            ax.set(xlabel=xlabel, ylabel=ylabel, title=variant, xlim=(-.02, 1.02), ylim=(-.02, 1.02))
            ax.legend(frameon=False)
            finish(fig, output + "_" + variant)
    table = pd.read_csv(output_dir / "confusion_matrix.csv")
    for variant, part in table.groupby("probability_type", sort=False):
        for value, suffix in (("count", "counts"), ("row_fraction", "row_normalized"), ("column_fraction", "column_normalized")):
            matrix = part.pivot(index="true_label", columns="predicted_label", values=value).reindex(index=range(N_CLASSES), columns=range(N_CLASSES)).to_numpy()
            fig, ax = plt.subplots(figsize=(4.8, 4.2))
            im = ax.imshow(matrix, cmap="Blues", vmin=0, vmax=None if value == "count" else 1)
            for i, j in itertools.product(range(N_CLASSES), repeat=2):
                text = "NA" if not np.isfinite(matrix[i, j]) else str(int(matrix[i, j])) if value == "count" else "{:.2f}".format(matrix[i, j])
                ax.text(j, i, text, ha="center", va="center", fontsize=8,
                        color="white" if np.isfinite(matrix[i, j]) and matrix[i, j] > np.nanmax(matrix) / 2 else "black")
            ax.set(xticks=range(N_CLASSES), yticks=range(N_CLASSES), xticklabels=CLASS_NAMES, yticklabels=CLASS_NAMES,
                   xlabel="Predicted class", ylabel="True class", title=variant + " / " + suffix)
            fig.colorbar(im, ax=ax, fraction=.045)
            finish(fig, "confusion_" + suffix + "_" + variant)
    table = pd.read_csv(output_dir / "calibration_bins.csv")
    fig, ax = plt.subplots(figsize=(4.8, 4))
    for variant, part in table[table["mode"].eq("confidence") & table["count"].gt(0)].groupby("probability_type", sort=False):
        ax.plot(part["mean_probability"], part["observed_rate"], "o-", markersize=3, label=variant)
    ax.plot([0, 1], [0, 1], "--", color="0.7", linewidth=.8)
    ax.set(xlabel="Mean maximum-class confidence", ylabel="Observed accuracy", xlim=(0, 1), ylim=(0, 1))
    ax.legend(frameon=False)
    finish(fig, "confidence_calibration")
    for filename, x, y, xlabel, ylabel, output in (
        ("confidence_threshold_scan.csv", "coverage", "accuracy_among_retained", "Coverage", "Accuracy among retained", "confidence_coverage"),
        ("prevalence_scenarios.csv", "assumed_class_prevalence", "expected_ppv", "Hypothetical class prevalence", "Expected PPV", "prevalence_scenarios")):
        table = pd.read_csv(output_dir / filename)
        fig, ax = plt.subplots(figsize=(5.1, 4))
        groups = ["probability_type", "class_name"] if "class_name" in table else ["probability_type"]
        for identity, part in table.groupby(groups, sort=False):
            ax.plot(part[x], part[y], label=" / ".join(identity) if isinstance(identity, tuple) else str(identity))
        ax.set(xlabel=xlabel, ylabel=ylabel, ylim=(-.02, 1.02))
        if output == "prevalence_scenarios":
            ax.set_xscale("log")
            ax.set_title("Hypothetical class-vs-rest scenarios", fontsize=9)
        ax.legend(frameon=False, fontsize=6, ncol=2)
        finish(fig, output)
    for filename, column, output in (("per_motif.csv", "motif", "worst_motif_errors"), ("per_reference.csv", "chrom", "reference_errors")):
        table = pd.read_csv(output_dir / filename)
        table = table[table["probability_type"].eq("raw")].sort_values(["error_rate", "n"], ascending=[False, False]).head(30)
        if not len(table):
            continue
        fig, ax = plt.subplots(figsize=(max(5, len(table) * .3), 4))
        x = np.arange(len(table))
        ax.bar(x, table["error_rate"], color="#355D86")
        error = np.vstack([table["error_rate"] - table["error_rate_wilson_low"], table["error_rate_wilson_high"] - table["error_rate"]])
        ax.errorbar(x, table["error_rate"], yerr=np.maximum(error, 0), fmt="none", color=".25", capsize=2, linewidth=.7)
        ax.set(xticks=x, xticklabels=table[column], ylabel="Window error rate (descriptive Wilson interval)")
        ax.tick_params(axis="x", rotation=70, labelsize=7)
        finish(fig, output)
    return names


def _load_result(directory):
    directory = Path(directory)
    for filename in ("result.json", "results.json", "run_result.json"):
        if (directory / filename).is_file():
            return json.loads((directory / filename).read_text(encoding="utf-8"))
    return {}


def _content_hashes(value):
    if isinstance(value, dict):
        if isinstance(value.get("sha256"), str):
            return value["sha256"]
        return {k: _content_hashes(v) for k, v in sorted(value.items()) if k not in ("path", "mtime", "mtime_ns", "size", "cache_path")}
    if isinstance(value, list):
        return [_content_hashes(v) for v in value]
    return value


def _has_content_sha256(value):
    if isinstance(value, dict):
        return (isinstance(value.get("sha256"), str) and re.fullmatch(r"[0-9a-fA-F]{64}", value["sha256"]) is not None
                or any(_has_content_sha256(v) for v in value.values()))
    if isinstance(value, list):
        return any(_has_content_sha256(v) for v in value)
    return False


def _run_signature(directory, result):
    data = result.get("provenance", {}).get("input_fingerprints", result.get("input_fingerprints"))
    config = result.get("config", result.get("arguments", {}))
    if not isinstance(config, dict):
        config = {}
    for filename in ("resolved_config.json", "config.json"):
        if not config and (directory / filename).is_file():
            config = json.loads((directory / filename).read_text(encoding="utf-8"))
    if data is None:
        audit_path = directory / "data_audit.json"
        if audit_path.is_file():
            audit = json.loads(audit_path.read_text(encoding="utf-8"))
            data = audit.get("input_fingerprints", audit.get("fingerprints"))
    # No fingerprint evidence means no cross-run comparability claim.
    data = _content_hashes(data) if _has_content_sha256(data) else {"unverified_run": str(directory.resolve())}
    policy_fields = ("epochs", "patience", "batch_size", "accum_steps", "lr", "learning_rate", "weight_decay",
                     "label_smoothing", "seq_len", "raw_signal_len", "dwell_divisor", "quality_divisor", "monitor",
                     "feature_scheme", "target_index", "dropout", "classifier_dropout", "amp", "deterministic",
                     "scenario", "num_classes", "class_names")
    policy = {key: config.get(key, "unspecified") for key in policy_fields}
    source = result.get("provenance", {}).get("source_snapshot", config.get("source_snapshot", {})).get("files_sha256", {})
    policy["training_source_sha256"] = {key: source[key] for key in ("train_all.py", "model_benchmark_all.py", "IterDataset.py", "config_6_all_6.py") if key in source}
    if not config:
        policy["unverified_policy_run"] = str(directory.resolve())
    payload = json.dumps({"data": data, "policy": policy}, sort_keys=True, separators=(",", ":"))
    return hashlib.sha256(payload.encode()).hexdigest(), data, policy, config


def aggregate_runs(run_dirs, out_dir, *, split="test", bootstrap=200, seed=42):
    """Guarded multi-seed descriptive summaries and matched-read comparisons.

    Different data hashes/training policies remain separate comparison groups.
    Paired cluster bootstrap compares fixed models' saved predictions within a
    matched seed; it does not estimate training-seed variability.
    """
    if split not in ("train", "val", "validation", "test"):
        raise ValueError("Unknown split.")
    out_dir = Path(out_dir).expanduser().resolve()
    out_dir.mkdir(parents=True, exist_ok=True)
    rows, run_info, pair_rows, skips, efficiency = [], [], [], [], []
    if isinstance(run_dirs, (str, Path)):
        run_dirs = [run_dirs]
    roots = set()
    for value in run_dirs:
        directory = Path(value).expanduser().resolve()
        if directory.is_file() and directory.name in ("result.json", "results.json", "run_result.json"):
            roots.add(directory.parent)
        elif any((directory / name).is_file() for name in ("result.json", "results.json", "run_result.json")):
            roots.add(directory)
        elif directory.is_dir():
            for filename in ("result.json", "results.json", "run_result.json"):
                roots.update(p.parent for p in directory.rglob(filename))
    for directory in sorted(roots):
        result = _load_result(directory)
        signature, data, policy, config = _run_signature(directory, result)
        split_names = (split, "val" if split == "validation" else "validation" if split == "val" else split)
        paths = [directory / s / "analysis" / "metrics.json" for s in split_names] + [directory / "analysis" / s / "metrics.json" for s in split_names]
        found = next((p for p in paths if p.is_file()), None)
        if found is None:
            skips.append({"run_dir": str(directory), "reason": "No saved split analysis/metrics.json found."})
            continue
        metrics = json.loads(found.read_text(encoding="utf-8"))
        model = config.get("model_type", config.get("model", result.get("model_type", directory.name)))
        run_seed = config.get("seed", result.get("seed", "unknown"))
        model_setting = {"model": model, "model_kwargs": config.get("model_kwargs", "unspecified")}
        model_signature = hashlib.sha256(json.dumps(model_setting, sort_keys=True).encode()).hexdigest()
        for variant, values in metrics.get("overall", {}).items():
            flattened = dict(values)
            for class_row in values.get("per_class", []):
                flattened.update({"class_{}_{}".format(class_row["class_id"], key): value
                                  for key, value in class_row.items() if isinstance(value, (int, float)) or value is None})
            rows.append({"run_dir": str(directory), "comparison_group": signature, "model": model, "seed": run_seed,
                         "model_setting_signature": model_signature, "split": split, "probability_type": variant,
                         **{key: flattened.get(key) for key in SUMMARY_METRICS}})
        prediction_path = Path(metrics.get("predictions_path", ""))
        if not prediction_path.is_file():
            prediction_path = next((p for s in split_names for p in (directory / s / "predictions.csv.gz", directory / s / "predictions.csv") if p.is_file()), None)
        run_info.append({"directory": directory, "comparison_group": signature, "model": model, "seed": run_seed,
                         "model_setting_signature": model_signature, "prediction_path": prediction_path, "data": data, "policy": policy})
        benchmark_path = directory / "benchmark_isolated.json"
        if benchmark_path.is_file():
            benchmark = json.loads(benchmark_path.read_text(encoding="utf-8"))
            fields = ("total_parameters", "trainable_parameters", "batch_size", "warmup", "repeats", "forward_mean_batch_ms",
                      "forward_p50_batch_ms", "forward_p95_batch_ms", "amortized_sample_ms", "samples_per_second",
                      "device", "amp", "scope", "isolation", "host_peak_rss_bytes", "host_current_rss_bytes",
                      "host_process_tree_current_rss_bytes", "gpu_peak_allocated_bytes", "gpu_peak_reserved_bytes")
            efficiency.append({"run_dir": str(directory), "comparison_group": signature, "model": model, "seed": run_seed,
                               "model_setting_signature": model_signature, "source": "benchmark_isolated.json",
                               **{key: benchmark.get(key) for key in fields}})
    table = pd.DataFrame(rows)
    table.to_csv(out_dir / "runs.csv", index=False)
    summary_rows = []
    if len(table):
        duplicated = table.duplicated(["comparison_group", "model", "model_setting_signature", "seed", "probability_type"], keep=False)
        if duplicated.any():
            raise ValueError("Duplicate model/seed runs in one comparison group; repeated seeds are not independent replicates.")
        for identity, part in table.groupby(["comparison_group", "model", "model_setting_signature", "split", "probability_type"], sort=True):
            for metric in SUMMARY_METRICS:
                values = pd.to_numeric(part[metric], errors="coerce").dropna()
                summary_rows.append(dict(zip(("comparison_group", "model", "model_setting_signature", "split", "probability_type"), identity),
                    metric=metric, valid_seeds=len(values), planned_runs=len(part), mean=float(values.mean()) if len(values) else None,
                    std=float(values.std(ddof=1)) if len(values) > 1 else None,
                    min=float(values.min()) if len(values) else None, max=float(values.max()) if len(values) else None))
    pd.DataFrame(summary_rows).to_csv(out_dir / "mean_std.csv", index=False)
    for a, b in itertools.combinations(run_info, 2):
        if a["comparison_group"] != b["comparison_group"] or a["seed"] != b["seed"] or a["model"] == b["model"]:
            continue
        if a["prediction_path"] is None or b["prediction_path"] is None:
            continue
        fa, va, _ = _prepare_predictions(a["prediction_path"])
        fb, vb, _ = _prepare_predictions(b["prediction_path"])
        # Exact sample identity, truth and read grouping required; input-order differences allowed.
        if set(fa["sample_id"]) != set(fb["sample_id"]):
            skips.append({"models": [a["model"], b["model"]], "seed": a["seed"], "reason": "Unmatched sample IDs."})
            continue
        order = pd.Series(np.arange(len(fb)), index=fb["sample_id"]).loc[fa["sample_id"]].to_numpy()
        if not np.array_equal(fa["label"], fb["label"].to_numpy()[order]) or not np.array_equal(fa["read_id"], fb["read_id"].to_numpy()[order]):
            raise ValueError("Paired comparison label/read metadata conflict.")
        y = fa["label"].to_numpy()
        for variant in sorted(set(va) & set(vb)):
            ra, ba = read_cluster_bootstrap(y, va[variant], fa["_analysis_read_id"], bootstrap=bootstrap, seed=seed)
            rb, bb = read_cluster_bootstrap(y, vb[variant][order], fa["_analysis_read_id"], bootstrap=bootstrap, seed=seed)
            ma, mb = multiclass_metrics(y, va[variant]), multiclass_metrics(y, vb[variant][order])
            for metric in ("accuracy", "macro_f1", "weighted_f1", "balanced_accuracy", "nll", "brier"):
                values = (pd.to_numeric(ba[metric], errors="coerce") - pd.to_numeric(bb[metric], errors="coerce")).dropna() if len(ba) else []
                lo, hi = np.quantile(values, [.025, .975]) if len(values) else (None, None)
                pair_rows.append({"comparison_group": a["comparison_group"], "seed": a["seed"], "model_a": a["model"], "model_b": b["model"],
                                  "probability_type": variant, "metric": metric, "difference_a_minus_b": ma[metric] - mb[metric] if ma[metric] is not None and mb[metric] is not None else None,
                                  "read_cluster_ci_low": lo, "read_cluster_ci_high": hi, "valid_replicates": len(values),
                                  "scope": "Matched fixed-model read-cluster uncertainty within this training seed; not across-seed inference."})
    pd.DataFrame(pair_rows).to_csv(out_dir / "paired_model_comparisons.csv", index=False)
    pd.DataFrame(efficiency).to_csv(out_dir / "isolated_efficiency.csv", index=False)
    report = {"schema_version": VERSION, "split": split, "run_count": len(run_info),
              "comparison_group_count": len(set(r["comparison_group"] for r in run_info)),
              "comparison_groups": [{k: _json_ready(v) for k, v in r.items() if k != "prediction_path"} for r in run_info],
              "skipped": skips, "files": ["runs.csv", "mean_std.csv", "paired_model_comparisons.csv", "isolated_efficiency.csv"],
              "notes": ["Only matching content hashes and training policies share a comparison group; missing fingerprint evidence isolates a run.",
                        "Standard deviation uses ddof=1 and is null for one independent seed. Seeds and validation selection should be planned before reading test results.",
                        "Efficiency reads benchmark_isolated.json only; concurrent training timings are excluded. Isolation from other users' work must be checked by the operator.",
                        "Read-cluster paired differences hold training seed fixed and do not replace variation across seeds, runs or template holdouts."]}
    _save_json(out_dir / "aggregate_report.json", report)
    return _json_ready(report)


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest="command", required=True)
    analyze = sub.add_parser("analyze", help="Recompute metrics and figures from prediction CSV/CSV.gz.")
    analyze.add_argument("--predictions", required=True)
    analyze.add_argument("--output_dir", required=True)
    analyze.add_argument("--bootstrap", type=int, default=200)
    analyze.add_argument("--subgroup_bootstrap", type=int, default=0,
                         help="Optional read-cluster CIs for motif/reference/template groups. Can be added offline without neural retraining.")
    analyze.add_argument("--seed", type=int, default=42)
    analyze.add_argument("--no_plots", action="store_true")
    aggregate = sub.add_parser("aggregate", help="Guarded repeated-seed/model comparison.")
    aggregate.add_argument("--runs", nargs="+", required=True)
    aggregate.add_argument("--output_dir", required=True)
    aggregate.add_argument("--split", default="test", choices=("train", "val", "validation", "test"))
    aggregate.add_argument("--bootstrap", type=int, default=200)
    aggregate.add_argument("--seed", type=int, default=42)
    redraw = sub.add_parser("plots", help="Redraw all figures from saved CSV tables.")
    redraw.add_argument("--output_dir", required=True)
    args = parser.parse_args(argv)
    if args.command == "analyze":
        report = analyze_predictions(args.predictions, args.output_dir, bootstrap=args.bootstrap, seed=args.seed,
                                     plots=not args.no_plots, subgroup_bootstrap=args.subgroup_bootstrap)
        print(json.dumps({"output_dir": args.output_dir, "row_count": report["row_count"]}, ensure_ascii=False))
    elif args.command == "aggregate":
        report = aggregate_runs(args.runs, args.output_dir, split=args.split, bootstrap=args.bootstrap, seed=args.seed)
        print(json.dumps({"output_dir": args.output_dir, "run_count": report["run_count"]}, ensure_ascii=False))
    else:
        print(json.dumps({"plots": make_basic_plots(args.output_dir)}, ensure_ascii=False))


if __name__ == "__main__":
    main()
