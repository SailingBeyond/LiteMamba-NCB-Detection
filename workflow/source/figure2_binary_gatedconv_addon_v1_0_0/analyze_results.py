#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""二分类逐样本预测的独立分析与多 seed 汇总，不依赖 PyTorch。

本模块只读取已经产生的概率。温度等校准参数必须在 validation 上拟合，
这里不会在待评价数据上重新拟合。每张图均有对应的 CSV 数据表。

示例：
  python analyze_results.py analyze --predictions test_predictions.csv.gz --output_dir analysis/test
  python analyze_results.py aggregate --runs runs/seed_42 runs/seed_43 --output_dir runs/summary
"""
import argparse
import hashlib
import itertools
import json
import math
import re
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import average_precision_score, precision_recall_curve, roc_auc_score, roc_curve


VERSION = "1.0.0"
SPLIT_NAMES = ("train", "val", "test")
PROPORTIONS = {
    "accuracy": ("correct", "n"), "precision": ("tp", "predicted_positive"),
    "recall": ("tp", "positive_support"), "specificity": ("tn", "negative_support"),
    "fpr": ("fp", "negative_support"), "fnr": ("fn", "positive_support"),
    "fdr": ("fp", "predicted_positive"), "npv": ("tn", "predicted_negative"),
    "error_rate": ("errors", "n"),
}
SUMMARY_METRICS = ("accuracy", "balanced_accuracy", "precision", "recall", "specificity",
                   "fpr", "fnr", "fdr", "npv", "macro_f1", "f1", "mcc", "auroc", "auprc",
                   "nll", "brier", "ece_15bins")
UNKNOWN = {"", "unknown", "nan", "none", "null", "na", "n/a", "<na>"}


def _ratio(numerator, denominator):
    return float(numerator / denominator) if denominator else None


def _json_ready(value):
    if isinstance(value, dict):
        return {str(k): _json_ready(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [_json_ready(v) for v in value]
    if isinstance(value, np.ndarray):
        return _json_ready(value.tolist())
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
    with Path(path).open("w", encoding="utf-8") as stream:
        json.dump(_json_ready(value), stream, ensure_ascii=False, indent=2, allow_nan=False)
        stream.write("\n")


def _labels(y_true):
    values = np.asarray(y_true)
    if values.ndim != 1:
        values = values.reshape(-1)
    try:
        values = values.astype(np.float64)
    except (TypeError, ValueError) as exc:
        raise ValueError("label 必须为 0/1。") from exc
    if not np.all(np.isfinite(values)) or not np.all((values == 0) | (values == 1)):
        raise ValueError("label 必须为有限且精确等于 0 或 1 的数值。")
    return values.astype(np.int8)


def _positive_probabilities(probabilities, n=None):
    values = np.asarray(probabilities, dtype=np.float64)
    if values.ndim == 2 and values.shape[1] == 2:
        if not np.all(np.isfinite(values)) or np.any(values < 0) or np.any(values > 1):
            raise ValueError("概率必须是 [0,1] 内的有限数值。")
        if not np.allclose(values.sum(axis=1), 1.0, rtol=1e-5, atol=1e-6):
            raise ValueError("N×2 概率的每行之和必须为 1。")
        values = values[:, 1]
    elif values.ndim == 2 and values.shape[1] == 1:
        values = values[:, 0]
    elif values.ndim != 1:
        raise ValueError("probabilities 形状应为 N、N×1 或 N×2。")
    if n is not None and len(values) != n:
        raise ValueError("label 与概率的样本数量不一致。")
    if not np.all(np.isfinite(values)) or np.any(values < 0) or np.any(values > 1):
        raise ValueError("概率必须是 [0,1] 内的有限数值。")
    return values


def _from_counts(tn, fp, fn, tp):
    tn, fp, fn, tp = (int(v) for v in (tn, fp, fn, tp))
    n = tn + fp + fn + tp
    positives, negatives = tp + fn, tn + fp
    predicted_positive, predicted_negative = tp + fp, tn + fn
    recall, specificity = _ratio(tp, positives), _ratio(tn, negatives)
    f1 = _ratio(2 * tp, 2 * tp + fp + fn)
    negative_f1 = _ratio(2 * tn, 2 * tn + fp + fn)
    mcc_den = math.sqrt(float(predicted_positive) * predicted_negative * positives * negatives)
    result = {
        "n": n, "support": n, "positive_support": positives, "negative_support": negatives,
        "predicted_positive": predicted_positive, "predicted_negative": predicted_negative,
        "correct": tp + tn, "errors": fp + fn, "tn": tn, "fp": fp, "fn": fn, "tp": tp,
        "accuracy": _ratio(tp + tn, n), "error_rate": _ratio(fp + fn, n),
        "balanced_accuracy": (recall + specificity) / 2 if positives and negatives else None,
        "precision": _ratio(tp, predicted_positive), "recall": recall, "specificity": specificity,
        "tpr": recall, "sensitivity": recall,
        "fpr": _ratio(fp, negatives), "fnr": _ratio(fn, positives),
        "fdr": _ratio(fp, predicted_positive), "npv": _ratio(tn, predicted_negative),
        "f1": f1, "f1_positive": f1,
        "macro_f1": (f1 + negative_f1) / 2 if positives and negatives else None,
        "mcc": _ratio(tp * tn - fp * fn, mcc_den),
        "class_support": "both_classes" if positives and negatives else "positive_only" if positives else "negative_only" if negatives else "empty",
    }
    return result


def wilson_interval(successes, total, z=1.959963984540054):
    """独立 Bernoulli 计数的 Wilson 95% 区间；分母为 0 时返回 None。"""
    if not total:
        return None, None
    p = successes / total
    denominator = 1 + z * z / total
    center = (p + z * z / (2 * total)) / denominator
    radius = z * math.sqrt(p * (1 - p) / total + z * z / (4 * total * total)) / denominator
    return float(max(0, center - radius)), float(min(1, center + radius))


def _add_wilson(metrics):
    result = dict(metrics)
    for metric, (numerator, denominator) in PROPORTIONS.items():
        low, high = wilson_interval(metrics[numerator], metrics[denominator])
        result[metric + "_ci_low"], result[metric + "_ci_high"] = low, high
    return result


def _calibration_table(y, p, mode="positive_probability", bins=15, threshold=0.5):
    if mode == "confidence":
        values, observed = np.maximum(p, 1 - p), ((p >= threshold) == y).astype(float)
    else:
        values, observed = p, y.astype(float)
    indices = np.minimum((values * bins).astype(int), bins - 1)
    count = np.bincount(indices, minlength=bins)
    total_prob = np.bincount(indices, weights=values, minlength=bins)
    total_observed = np.bincount(indices, weights=observed, minlength=bins)
    rows = []
    for index in range(bins):
        mean_prob = _ratio(total_prob[index], count[index])
        empirical = _ratio(total_observed[index], count[index])
        lo, hi = wilson_interval(total_observed[index], count[index])
        rows.append({"mode": mode, "bin": index, "bin_left": index / bins,
                     "bin_right": (index + 1) / bins, "count": int(count[index]),
                     "mean_probability": mean_prob, "observed_rate": empirical,
                     "observed_rate_ci_low": lo, "observed_rate_ci_high": hi,
                     "absolute_gap": abs(mean_prob - empirical) if count[index] else None})
    return pd.DataFrame(rows)


def binary_metrics(y_true, probabilities, threshold=0.5):
    """二分类完整指标。probabilities 接受正类概率 N/N×1 或概率矩阵 N×2。

    决策为 p(class=1) >= threshold。缺少必要类别或分母为 0 的指标返回
    JSON null，而非伪造为 0。单类参考上的 precision 不能当作总体 PPV。
    """
    if not 0 <= threshold <= 1:
        raise ValueError("threshold 必须位于 [0,1]。")
    y = _labels(y_true)
    p = _positive_probabilities(probabilities, len(y))
    prediction = p >= threshold
    result = _from_counts(np.sum((y == 0) & ~prediction), np.sum((y == 0) & prediction),
                          np.sum((y == 1) & ~prediction), np.sum((y == 1) & prediction))
    both_classes = result["positive_support"] > 0 and result["negative_support"] > 0
    result["auroc"] = float(roc_auc_score(y, p)) if both_classes else None
    result["auprc"] = float(average_precision_score(y, p)) if both_classes else None
    result["roc_auc"] = result["auc"] = result["auroc"]
    result["average_precision"] = result["auprc"]
    if len(y):
        clipped = np.clip(p, np.finfo(np.float64).eps, 1 - np.finfo(np.float64).eps)
        result["nll"] = float(-np.mean(y * np.log(clipped) + (1 - y) * np.log1p(-clipped)))
        result["brier"] = float(np.mean((p - y) ** 2))
        positive_bins = _calibration_table(y, p)
        confidence_bins = _calibration_table(y, p, "confidence", threshold=threshold)
        result["ece_positive_probability_15bins"] = float((positive_bins["count"] * positive_bins["absolute_gap"].fillna(0)).sum() / len(y))
        result["ece_15bins"] = float((confidence_bins["count"] * confidence_bins["absolute_gap"].fillna(0)).sum() / len(y))
    else:
        result.update({"nll": None, "brier": None, "ece_positive_probability_15bins": None, "ece_15bins": None})
    result["ece"] = result["ece_15bins"]
    result["threshold"] = float(threshold)
    result["confusion_matrix"] = [[result["tn"], result["fp"]], [result["fn"], result["tp"]]]
    return _json_ready(result)


def _known(series):
    return ~series.astype(str).str.strip().str.lower().isin(UNKNOWN)


def _prepare_predictions(path):
    frame = pd.read_csv(path, compression="infer", low_memory=False, dtype={
        "read_id": str, "sample_id": str, "chrom": str, "pair_id": str,
        "template_group": str, "bam_strand": str, "motif": str, "original_kmer": str,
        "run_id": str, "source_dataset": str, "source_file": str,
    })
    if "label" not in frame or not len(frame):
        raise ValueError("预测表必须有 label 列，且至少包含一条记录。")
    y = _labels(frame["label"].to_numpy())
    frame["label"] = y
    variants = {}
    for name, prefix in (("raw", "prob_"), ("calibrated", "calibrated_prob_")):
        if prefix + "1" in frame:
            values = (frame[[prefix + "0", prefix + "1"]].to_numpy()
                      if prefix + "0" in frame else frame[prefix + "1"].to_numpy())
            variants[name] = _positive_probabilities(values, len(y))
        elif prefix + "0" in frame:
            variants[name] = _positive_probabilities(1 - pd.to_numeric(frame[prefix + "0"]).to_numpy(), len(y))
    if "raw" not in variants:
        raise ValueError("预测表需要 prob_1（可同时提供 prob_0）。不会从 prediction 或 confidence 猜测概率。")
    for column in ("sample_id", "read_id", "chrom", "pair_id", "template_group", "bam_strand",
                   "motif", "original_kmer", "run_id", "source_dataset", "source_file"):
        if column not in frame:
            frame[column] = "unknown"
        frame[column] = frame[column].fillna("unknown").astype(str)
        frame.loc[~_known(frame[column]), column] = "unknown"
    if "sample_index" not in frame:
        frame["sample_index"] = np.arange(len(frame))
    missing_read = frame["read_id"].eq("unknown")
    frame["_analysis_read_id"] = frame["read_id"]
    frame.loc[missing_read, "_analysis_read_id"] = ["__unresolved_row_{}".format(i) for i in frame.index[missing_read]]
    pair_from_reference = frame["chrom"].str.extract(r"^(?:PC|DNA|XNA)(\d+)$", expand=False)
    missing_pair = frame["pair_id"].eq("unknown")
    inferred_pair = missing_pair & pair_from_reference.notna()
    frame.loc[inferred_pair, "pair_id"] = pair_from_reference[inferred_pair].str.zfill(2)
    missing_pair = frame["pair_id"].eq("unknown") & frame["template_group"].ne("unknown")
    frame.loc[missing_pair, "pair_id"] = frame.loc[missing_pair, "template_group"]
    for name, probabilities in variants.items():
        frame["_p_" + name] = probabilities
    return frame, variants, {"unknown_read_id_rows": int(missing_read.sum()),
                             "pair_ids_derived_from_reference_rows": int(inferred_pair.sum())}


def _metric_row(y, p, threshold, read_ids=None):
    result = _add_wilson(binary_metrics(y, p, threshold))
    result.pop("confusion_matrix", None)
    if read_ids is not None:
        result["unique_read_count"] = int(pd.Series(read_ids).nunique())
        result["rows_share_read"] = bool(result["unique_read_count"] < result["n"])
    result["precision_scope_note"] = ("Within this group only; single-class groups cannot estimate overall PPV."
                                      if result["class_support"] != "both_classes" else "Within this group only.")
    return result


def _group_table(frame, group_column, variants, threshold):
    rows = []
    for value, group in frame.groupby(group_column, sort=True, dropna=False):
        for name in variants:
            metrics = _metric_row(group["label"].to_numpy(), group["_p_" + name].to_numpy(), threshold,
                                  group["_analysis_read_id"].to_numpy())
            rows.append({group_column: value, "probability_type": name, **metrics})
    return pd.DataFrame(rows)


def _sequence_features(motif):
    sequence = str(motif).upper()
    flanks = [sequence[:3], sequence[4:]] if len(sequence) == 6 else [sequence]
    canonical = "".join(c for part in flanks for c in part if c in "ATCG")
    longest = 0
    for part in flanks:
        for base, group in itertools.groupby(part):
            size = sum(1 for _ in group)
            if base in "ATCG":
                longest = max(longest, size)
    return {"sequence_length": len(sequence), "flank_gc_fraction": _ratio(sum(c in "GC" for c in canonical), len(canonical)),
            "flank_canonical_bases": len(canonical), "max_homopolymer_flank": longest,
            "left_flank": flanks[0] if len(sequence) == 6 else sequence,
            "right_flank": flanks[1] if len(sequence) == 6 else ""}


def _bootstrap_read_clusters(frame, p, threshold, bootstrap, seed):
    if bootstrap <= 0:
        return {"method": "read_id cluster bootstrap", "replicates": 0, "intervals": {}}, pd.DataFrame()
    y = frame["label"].to_numpy()
    prediction = p >= threshold
    codes, unique = pd.factorize(frame["_analysis_read_id"], sort=False)
    cluster_counts = np.column_stack([
        np.bincount(codes, weights=mask, minlength=len(unique))
        for mask in ((y == 0) & ~prediction, (y == 0) & prediction,
                     (y == 1) & ~prediction, (y == 1) & prediction)
    ])
    rng = np.random.default_rng(seed)
    rows = []
    keys = ("accuracy", "precision", "recall", "specificity", "fdr")
    for iteration in range(bootstrap):
        sample = rng.integers(0, len(unique), size=len(unique))
        counts = cluster_counts[sample].sum(axis=0)
        metrics = _from_counts(*counts)
        rows.append({"replicate": iteration, **{key: metrics[key] for key in keys}})
    replicates = pd.DataFrame(rows)
    intervals = {}
    for key in keys:
        values = replicates[key].dropna().to_numpy(dtype=float)
        bounds = np.quantile(values, [0.025, 0.975]).tolist() if len(values) else [None, None]
        intervals[key] = {"ci_low": bounds[0], "ci_high": bounds[1], "valid_replicates": int(len(values))}
    return {"method": "Percentile bootstrap: sample read_id clusters with replacement, retain all windows in each selected read.",
            "replicates": bootstrap, "seed": seed, "cluster_count": int(len(unique)),
            "intervals": intervals}, replicates


def _threshold_tables(y, p, threshold):
    prediction = p >= threshold
    confidence = np.maximum(p, 1 - p)
    confidence_rows = []
    thresholds = sorted(set([round(v, 2) for v in np.arange(0.5, 1.0, 0.05)] + [0.99, 0.995, 0.999]))
    all_positive, all_negative = int(np.sum(y == 1)), int(np.sum(y == 0))
    for tau in thresholds:
        retained = confidence >= tau
        metrics = _from_counts(np.sum(retained & (y == 0) & ~prediction), np.sum(retained & (y == 0) & prediction),
                               np.sum(retained & (y == 1) & ~prediction), np.sum(retained & (y == 1) & prediction))
        row = {"confidence_threshold": tau, "decision_threshold": threshold, "total_count": len(y),
               "retained_count": metrics["n"], "coverage": _ratio(metrics["n"], len(y)),
               "abstained_count": int(np.sum(~retained)),
               "abstained_positive_count": int(np.sum((~retained) & (y == 1))),
               "abstained_negative_count": int(np.sum((~retained) & (y == 0))),
               "all_positive_count": all_positive, "all_negative_count": all_negative,
               "precision_among_retained": metrics["precision"], "recall_among_retained": metrics["recall"],
               "overall_recall": _ratio(metrics["tp"], all_positive),
               "tn": metrics["tn"], "fp": metrics["fp"], "fn": metrics["fn"], "tp": metrics["tp"],
               "retained_positive_count": metrics["positive_support"], "retained_negative_count": metrics["negative_support"],
               "accuracy_among_retained": metrics["accuracy"], "specificity_among_retained": metrics["specificity"],
               "fdr_among_retained": metrics["fdr"]}
        row["coverage_ci_low"], row["coverage_ci_high"] = wilson_interval(metrics["n"], len(y))
        for metric, successes, denominator in (
            ("precision_among_retained", metrics["tp"], metrics["predicted_positive"]),
            ("recall_among_retained", metrics["tp"], metrics["positive_support"]),
            ("overall_recall", metrics["tp"], all_positive),
            ("accuracy_among_retained", metrics["correct"], metrics["n"]),
            ("specificity_among_retained", metrics["tn"], metrics["negative_support"]),
            ("fdr_among_retained", metrics["fp"], metrics["predicted_positive"]),
        ):
            row[metric + "_ci_low"], row[metric + "_ci_high"] = wilson_interval(successes, denominator)
        confidence_rows.append(row)
    negative, positive = np.sort(p[y == 0]), np.sort(p[y == 1])
    decision_rows = []
    for tau in sorted(set(np.linspace(0, 1, 201).tolist() + [0.99, 0.995, 0.999])):
        tp = len(positive) - np.searchsorted(positive, tau, side="left")
        fp = len(negative) - np.searchsorted(negative, tau, side="left")
        metrics = _add_wilson(_from_counts(len(negative) - fp, fp, len(positive) - tp, tp))
        decision_rows.append({"positive_probability_threshold": float(tau), **metrics})
    return pd.DataFrame(confidence_rows), pd.DataFrame(decision_rows)


def _prior_shift_table(metrics):
    rows = []
    tpr, fpr = metrics["recall"], metrics["fpr"]
    tpr_low, tpr_high = wilson_interval(metrics["tp"], metrics["positive_support"])
    fpr_low, fpr_high = wilson_interval(metrics["fp"], metrics["negative_support"])
    for prior in (0.0001, 0.001, 0.01, 0.05, 0.1, 0.5):
        ppv = npv = fdr = ppv_low = ppv_high = None
        if tpr is not None and fpr is not None:
            tp, fp, tn, fn = prior * tpr, (1 - prior) * fpr, (1 - prior) * (1 - fpr), prior * (1 - tpr)
            ppv, npv, fdr = _ratio(tp, tp + fp), _ratio(tn, tn + fn), _ratio(fp, tp + fp)
            ppv_low = _ratio(prior * tpr_low, prior * tpr_low + (1 - prior) * fpr_high)
            ppv_high = _ratio(prior * tpr_high, prior * tpr_high + (1 - prior) * fpr_low)
        rows.append({"assumed_positive_prevalence": prior, "conditional_tpr": tpr, "conditional_fpr": fpr,
                     "expected_ppv": ppv, "expected_npv": npv, "expected_fdr": fdr,
                     "ppv_wilson_bound_sensitivity_low": ppv_low, "ppv_wilson_bound_sensitivity_high": ppv_high,
                     "assumption": "Prior-shift scenario only: class-conditional TPR/FPR remain unchanged.",
                     "bound_note": "Sensitivity using marginal Wilson bounds; not a joint confidence interval or a measured low-prevalence experiment."})
    return pd.DataFrame(rows)


def _readlevel_predictions(frame, variants, threshold):
    group = frame.groupby("_analysis_read_id", sort=False)
    labels_per_read = group["label"].nunique()
    conflicting = labels_per_read[labels_per_read > 1].index
    conflict_rows = frame[frame["_analysis_read_id"].isin(conflicting)].drop(columns=[c for c in frame if c.startswith("_p_")])
    clean = frame[~frame["_analysis_read_id"].isin(conflicting)]
    groups = clean.groupby("_analysis_read_id", sort=False)
    output = groups.agg(label=("label", "first"), window_count=("label", "size"), read_id=("read_id", "first"))
    for column in ("chrom", "pair_id", "template_group", "bam_strand", "motif", "run_id", "source_dataset"):
        first, counts = groups[column].first(), groups[column].nunique()
        output[column] = first.where(counts == 1, "mixed")
    for name in variants:
        values = groups["_p_" + name].mean()
        prefix = "" if name == "raw" else "calibrated_"
        output[prefix + "prob_1"], output[prefix + "prob_0"] = values, 1 - values
        output[prefix + "prediction"] = (values >= threshold).astype(int)
    output.index.name = "analysis_group_id"
    return output.reset_index(), conflict_rows


def analyze_predictions(predictions_path, output_dir, *, bootstrap=200, seed=42, threshold=0.5,
                        plots=True, run_metadata=None):
    """读取预测，保存审稿所需的可重绘数据、分组指标、区间与可选图形。"""
    if int(bootstrap) != bootstrap or bootstrap < 0:
        raise ValueError("bootstrap 应为非负整数。")
    predictions_path = Path(predictions_path).expanduser().resolve()
    output_dir = Path(output_dir).expanduser().resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    frame, variants, input_notes = _prepare_predictions(predictions_path)
    y = frame["label"].to_numpy()
    files = []

    def save_table(name, table):
        target = output_dir / name
        if target.resolve() == predictions_path:
            raise ValueError("分析输出不能覆盖输入预测文件。")
        table.to_csv(target, index=False)
        files.append(name)

    overall, uncertainty = {}, {}
    curve_tables, pr_tables, calibration_tables, histogram_tables = [], [], [], []
    confidence_tables, decision_tables, prior_tables, confusion_rows, bootstrap_tables = [], [], [], [], []
    for name, p in variants.items():
        overall[name] = _add_wilson(binary_metrics(y, p, threshold))
        uncertainty[name], replicates = _bootstrap_read_clusters(frame, p, threshold, int(bootstrap), seed)
        if len(replicates):
            replicates.insert(0, "probability_type", name)
            bootstrap_tables.append(replicates)
        if overall[name]["class_support"] == "both_classes":
            fpr, tpr, cutoffs = roc_curve(y, p, drop_intermediate=False)
            curve_tables.append(pd.DataFrame({"probability_type": name, "fpr": fpr, "tpr": tpr, "threshold": cutoffs}))
            precision, recall, cutoffs = precision_recall_curve(y, p)
            pr_tables.append(pd.DataFrame({"probability_type": name, "recall": recall, "precision": precision,
                                           "threshold": np.r_[cutoffs, np.nan]}))
        for mode in ("positive_probability", "confidence"):
            table = _calibration_table(y, p, mode, threshold=threshold)
            table.insert(0, "probability_type", name)
            calibration_tables.append(table)
        edges = np.linspace(0, 1, 101)
        for label in (0, 1):
            values = p[y == label]
            counts, _ = np.histogram(values, bins=edges)
            histogram_tables.append(pd.DataFrame({"probability_type": name, "label": label,
                                                  "bin_left": edges[:-1], "bin_right": edges[1:],
                                                  "count": counts, "class_support": len(values),
                                                  "within_class_fraction": counts / len(values) if len(values) else np.nan}))
        confidence, decision = _threshold_tables(y, p, threshold)
        for table in (confidence, decision):
            table.insert(0, "probability_type", name)
        confidence_tables.append(confidence)
        decision_tables.append(decision)
        prior = _prior_shift_table(overall[name])
        prior.insert(0, "probability_type", name)
        prior_tables.append(prior)
        for true, predicted, key in ((0, 0, "tn"), (0, 1, "fp"), (1, 0, "fn"), (1, 1, "tp")):
            confusion_rows.append({"probability_type": name, "true_label": true, "predicted_label": predicted,
                                   "count": overall[name][key]})
    save_table("roc_curve.csv", pd.concat(curve_tables, ignore_index=True) if curve_tables else pd.DataFrame(columns=["probability_type", "fpr", "tpr", "threshold"]))
    save_table("pr_curve.csv", pd.concat(pr_tables, ignore_index=True) if pr_tables else pd.DataFrame(columns=["probability_type", "recall", "precision", "threshold"]))
    for name, tables in (("calibration_bins.csv", calibration_tables), ("probability_histogram.csv", histogram_tables),
                         ("confidence_threshold_scan.csv", confidence_tables), ("positive_threshold_scan.csv", decision_tables),
                         ("prevalence_prior_shift.csv", prior_tables)):
        save_table(name, pd.concat(tables, ignore_index=True))
    save_table("confusion_matrix.csv", pd.DataFrame(confusion_rows))
    save_table("bootstrap_replicates.csv", pd.concat(bootstrap_tables, ignore_index=True) if bootstrap_tables else pd.DataFrame(columns=["probability_type", "replicate", "accuracy", "precision", "recall", "specificity", "fdr"]))
    group_files = {"chrom": "per_reference.csv", "pair_id": "per_template_pair.csv", "motif": "per_motif.csv",
                   "bam_strand": "per_strand.csv", "source_dataset": "per_source.csv"}
    motif_table = None
    for column, name in group_files.items():
        table = _group_table(frame, column, variants, threshold)
        if column == "motif":
            features = pd.DataFrame([{"motif": motif, **_sequence_features(motif)} for motif in table["motif"].unique()])
            table = table.merge(features, on="motif", how="left", validate="many_to_one")
            motif_table = table
        save_table(name, table)
    run_frame = frame[frame["run_id"].ne("unknown")]
    save_table("per_run.csv", _group_table(run_frame, "run_id", variants, threshold) if len(run_frame)
               else pd.DataFrame(columns=["run_id", "probability_type", "n", "positive_support", "negative_support"]))
    bad_motifs = motif_table.sort_values(["probability_type", "error_rate", "n"], ascending=[True, False, False]).copy()
    bad_motifs.insert(0, "rank_within_probability_type", bad_motifs.groupby("probability_type").cumcount() + 1)
    save_table("bad_motifs.csv", bad_motifs)
    readlevel, conflicts = _readlevel_predictions(frame, variants, threshold)
    save_table("readlevel_predictions.csv.gz", readlevel)
    save_table("read_label_conflicts.csv", conflicts)
    read_metrics = {}
    if len(readlevel):
        for name in variants:
            prefix = "" if name == "raw" else "calibrated_"
            read_metrics[name] = _add_wilson(binary_metrics(readlevel["label"].to_numpy(), readlevel[prefix + "prob_1"].to_numpy(), threshold))
    _save_json(output_dir / "readlevel_metrics.json", {"aggregation": "Unweighted mean probability across windows of each read; conflicting-label reads excluded.",
                                                       "conflicting_read_count": int(conflicts["_analysis_read_id"].nunique()),
                                                       "overall": read_metrics})
    files.append("readlevel_metrics.json")
    unique_reads = int(frame["_analysis_read_id"].nunique())
    notes = [
        "Calibration is read from calibrated_prob_* columns only; no parameter is fitted on these predictions.",
        "Decision: prob_1 >= threshold. Confidence filtering uses max(prob_0, prob_1), not prob_1 alone.",
        "recall_among_retained conditions on retained positives; overall_recall divides retained TP by all original positives.",
        "AUROC, AUPRC, balanced accuracy and macro F1 are null when one true class is absent. Undefined denominators are null.",
        "Group precision is conditional on that group; single-class reference groups do not estimate population PPV.",
        "ECE uses 15 uniform bins of maximum class confidence versus correctness; positive-probability calibration ECE is also saved.",
        "Wilson intervals assume independent rows. Read-cluster bootstrap retains within-read dependence for overall count-based metrics.",
        "Read bootstrap does not address correlations across templates or sequencing runs; independent holdout experiments remain necessary.",
        "Prior-shift tables are hypothetical calculations assuming stable class-conditional TPR/FPR, not measured low-prevalence experiments.",
        "Unknown run/source identities remain unknown; no sequencing run is inferred from a filename.",
        "Motif rankings report every observed motif and its support; low-support rankings require the saved uncertainty intervals.",
    ]
    if unique_reads < len(frame):
        notes.append("Multiple rows share a read ID: per-row Wilson intervals are descriptive only; prefer read-cluster bootstrap for overall uncertainty.")
    if input_notes["unknown_read_id_rows"]:
        notes.append("Unknown read IDs use separate explicitly unresolved analysis groups; their actual dependence cannot be reconstructed.")
    if len(conflicts):
        notes.append("Some reads have conflicting true labels. They remain in window metrics but are excluded from readlevel averaging; inspect read_label_conflicts.csv.")
    report = {"schema_version": VERSION, "task": "binary classification", "predictions_path": str(predictions_path),
              "class_definition": {"0": "PC/DNA canonical control", "1": "XNA"},
              "decision_threshold": threshold, "row_count": len(frame), "analysis_read_count": unique_reads,
              "known_run_count": int(run_frame["run_id"].nunique()), "metadata": _json_ready(run_metadata or {}),
              "input_notes": input_notes, "overall": overall, "raw": overall["raw"],
              "calibrated": overall.get("calibrated"), "uncertainty": uncertainty, "notes": notes,
              "files": files + ["metrics.json"]}
    if "prediction" in frame:
        supplied = pd.to_numeric(frame["prediction"], errors="coerce").to_numpy()
        report["input_notes"]["supplied_prediction_disagrees_with_probability_threshold_rows"] = int(np.sum(supplied != (variants["raw"] >= threshold)))
    if plots:
        report["plots"] = make_basic_plots(output_dir)
    _save_json(output_dir / "metrics.json", report)
    return _json_ready(report)


def make_basic_plots(output_dir):
    """只读取 CSV 图源，可在没有模型与 PyTorch 的环境中重新绘图。"""
    import matplotlib
    matplotlib.use("Agg", force=True)
    import matplotlib.pyplot as plt
    output_dir = Path(output_dir)
    names = []
    colors = {"raw": "#2F5F8F", "calibrated": "#C16F3D"}

    def finish(fig, name):
        fig.tight_layout()
        for suffix in ("png", "pdf"):
            path = output_dir / (name + "." + suffix)
            fig.savefig(path, dpi=220, bbox_inches="tight")
            names.append(path.name)
        plt.close(fig)

    for filename, x, y, xlabel, ylabel, name in (
        ("roc_curve.csv", "fpr", "tpr", "False positive rate", "True positive rate", "roc_curve"),
        ("pr_curve.csv", "recall", "precision", "Recall", "Precision", "pr_curve"),
        ("confidence_threshold_scan.csv", "coverage", "accuracy_among_retained", "Coverage", "Accuracy among retained", "confidence_coverage"),
        ("prevalence_prior_shift.csv", "assumed_positive_prevalence", "expected_ppv", "Assumed positive prevalence", "Expected PPV", "prevalence_prior_shift"),
    ):
        table = pd.read_csv(output_dir / filename)
        if not len(table):
            continue
        fig, ax = plt.subplots(figsize=(4.5, 3.7))
        for variant, part in table.groupby("probability_type", sort=False):
            ax.plot(part[x], part[y], color=colors.get(variant), label=variant)
        if name == "roc_curve":
            ax.plot([0, 1], [0, 1], "--", color="0.7", linewidth=0.8)
        if name == "prevalence_prior_shift":
            ax.set_xscale("log")
            ax.set_title("Prior-shift scenario; stable TPR/FPR assumed", fontsize=9)
        ax.set(xlabel=xlabel, ylabel=ylabel, ylim=(-0.02, 1.02))
        ax.legend(frameon=False)
        finish(fig, name)
    table = pd.read_csv(output_dir / "calibration_bins.csv")
    fig, ax = plt.subplots(figsize=(4.5, 3.7))
    for variant, part in table[table["mode"].eq("positive_probability") & table["count"].gt(0)].groupby("probability_type", sort=False):
        ax.plot(part["mean_probability"], part["observed_rate"], "o-", markersize=3, label=variant, color=colors.get(variant))
    ax.plot([0, 1], [0, 1], "--", color="0.7", linewidth=0.8)
    ax.set(xlabel="Mean predicted positive probability", ylabel="Observed positive fraction", xlim=(-0.02, 1.02), ylim=(-0.02, 1.02))
    ax.legend(frameon=False)
    finish(fig, "calibration")
    table = pd.read_csv(output_dir / "confusion_matrix.csv")
    variants = list(table["probability_type"].unique())
    fig, axes = plt.subplots(1, len(variants), figsize=(3.6 * len(variants), 3.4), squeeze=False)
    for ax, variant in zip(axes[0], variants):
        part = table[table["probability_type"].eq(variant)]
        matrix = part.pivot(index="true_label", columns="predicted_label", values="count").reindex(index=[0, 1], columns=[0, 1]).to_numpy()
        ax.imshow(matrix, cmap="Blues")
        for i, j in itertools.product(range(2), repeat=2):
            ax.text(j, i, "{:,}".format(int(matrix[i, j])), ha="center", va="center", color="white" if matrix[i, j] > matrix.max() / 2 else "black")
        ax.set(xticks=[0, 1], yticks=[0, 1], xlabel="Predicted label", ylabel="True label", title=variant)
    finish(fig, "confusion_matrix")
    for filename, column, name in (("per_motif.csv", "motif", "motif_error_rates"), ("per_reference.csv", "chrom", "reference_error_rates")):
        table = pd.read_csv(output_dir / filename)
        table = table[table["probability_type"].eq("raw")].sort_values("error_rate", ascending=False).head(30)
        if not len(table):
            continue
        fig, ax = plt.subplots(figsize=(max(5, len(table) * 0.28), 3.7))
        positions = np.arange(len(table))
        ax.bar(positions, table["error_rate"], color=colors["raw"], width=0.7)
        low, high = table["error_rate_ci_low"].to_numpy(), table["error_rate_ci_high"].to_numpy()
        ax.errorbar(positions, table["error_rate"], yerr=np.vstack([table["error_rate"].to_numpy() - low, high - table["error_rate"].to_numpy()]), fmt="none", color="0.25", linewidth=0.7, capsize=2)
        ax.set(xticks=positions, xticklabels=table[column].astype(str), ylabel="Error rate", title="Raw probabilities; up to 30 groups, ordered by error")
        ax.tick_params(axis="x", rotation=70, labelsize=7)
        finish(fig, name)
    return names


def _variant_metrics(payload):
    if not isinstance(payload, dict):
        return {}
    if isinstance(payload.get("overall"), dict) and "raw" in payload["overall"]:
        return {name: value for name, value in payload["overall"].items() if isinstance(value, dict)}
    if isinstance(payload.get("raw"), dict):
        return {name: payload[name] for name in ("raw", "calibrated") if isinstance(payload.get(name), dict)}
    if "accuracy" in payload:
        return {"raw": payload}
    return {}


def _data_signature(result, run_directory):
    """只使用内容 SHA256，允许数据文件迁移；缺证据的 run 单独成组。"""
    fingerprints = result.get("provenance", {}).get("input_fingerprints", result.get("input_fingerprints"))
    hashes = []

    def content_only(value):
        if isinstance(value, dict):
            if value.get("sha256"):
                hashes.append(str(value["sha256"]))
                return {"sha256": value["sha256"], "size": value.get("size")}
            return {str(key): content_only(item) for key, item in value.items()}
        if isinstance(value, (list, tuple)):
            return [content_only(item) for item in value]
        return value

    canonical = content_only(fingerprints)
    if hashes:
        encoded = json.dumps(canonical, sort_keys=True, separators=(",", ":"), ensure_ascii=False)
        return hashlib.sha256(encoded.encode("utf-8")).hexdigest(), "input_content_sha256"
    # 不能证明数据一致的两个 run，既不合并为独立 seed，也不制造 paired delta。
    return "unverified:" + hashlib.sha256(str(run_directory).encode()).hexdigest(), "unverified_separate_run"


def _training_signature(result, run_directory):
    """同模型的 seed 汇总必须保持训练设置相同，忽略 seed/设备/文件路径。"""
    path = run_directory / "resolved_config.json"
    config = json.loads(path.read_text(encoding="utf-8")) if path.is_file() else result.get("resolved_config", result.get("config"))
    if not isinstance(config, dict):
        return "unknown", "unknown"
    keys = (
        "model_type", "model_kwargs", "raw_signal_len", "dwell_divisor", "quality_divisor",
        "feature_scheme", "feature_definition", "seq_len", "num_classes", "positive_name",
        "model_dim", "num_blocks", "state_dim", "target_index", "nhead", "num_layers",
        "gru_hidden", "dropout", "classifier_dropout", "ssm_bidirectional", "litemamba_variant", "readout",
        "epochs", "patience", "lr", "learning_rate", "batch_size", "accum_steps", "weight_decay",
        "label_smoothing", "clip_grad", "monitor", "amp", "optimizer", "scheduler", "trainer_version",
    )
    relevant = {key: config[key] for key in keys if key in config}
    if not relevant:
        return "unknown", "unknown"
    # 固定实现的优化器/调度器也写入签名定义；若未来显式配置，以该配置为准。
    relevant.setdefault("optimizer", {"name": "AdamW", "betas": [0.9, 0.999], "eps": 1e-8})
    relevant.setdefault("scheduler", {"name": "ReduceLROnPlateau", "factor": 0.5, "patience": 8, "min_lr": 1e-6})
    source = result.get("provenance", {}).get("source_snapshot", {}).get("files_sha256", {})
    relevant["training_source_sha256"] = {name: source[name] for name in ("train.py", "model_benchmark.py", "IterDataset.py") if name in source}
    encoded = json.dumps(relevant, sort_keys=True, separators=(",", ":"), ensure_ascii=False)
    return hashlib.sha256(encoded.encode("utf-8")).hexdigest(), "resolved_config" if path.is_file() else "embedded_config"


def aggregate_runs(run_dirs, out_dir, *, split="test"):
    """按预定 split 汇总全部 seed；不会按 test 选择最优 seed。"""
    if split not in SPLIT_NAMES:
        raise ValueError("split 应为 train、val 或 test。")
    if isinstance(run_dirs, (str, Path)):
        run_dirs = [run_dirs]
    roots = []
    for directory in run_dirs:
        path = Path(directory).expanduser().resolve()
        if path.name == "results.json" and path.is_file():
            roots.append(path.parent)
        elif (path / "results.json").is_file():
            roots.append(path)
        elif path.is_dir():
            roots.extend(p.parent for p in path.rglob("results.json"))
    roots = sorted(set(roots))
    output = Path(out_dir).expanduser().resolve()
    output.mkdir(parents=True, exist_ok=True)
    rows, skipped = [], []
    for directory in roots:
        try:
            result = json.loads((directory / "results.json").read_text(encoding="utf-8"))
            meta = {**result.get("metadata", {}), **result}
            if str(meta.get("status", "")).lower() in ("failed", "error", "incomplete", "running"):
                skipped.append({"run_dir": str(directory), "reason": "run status is " + str(meta.get("status"))})
                continue
            export_info = result.get("exports_by_split", {}).get(split, {})
            declared_analysis = export_info.get("analysis") if isinstance(export_info, dict) else None
            declared_path = Path(declared_analysis).expanduser() if declared_analysis else None
            if declared_path is not None and not declared_path.is_absolute():
                declared_path = directory / declared_path
            candidates = [directory / split / "analysis" / "metrics.json", directory / "analysis" / split / "metrics.json",
                          directory / (split + "_analysis") / "metrics.json", directory / split / "metrics.json"]
            if declared_path is not None:
                candidates.append(declared_path / "metrics.json")
            metric_file = next((p for p in candidates if p.is_file()), None)
            variants = _variant_metrics(json.loads(metric_file.read_text(encoding="utf-8"))) if metric_file else {}
            if not variants:
                variants = _variant_metrics(result.get("metrics_by_split", {}).get(split, {}))
            if not variants:
                variants = _variant_metrics(result.get(split + "_metrics", {}))
            if not variants:
                skipped.append({"run_dir": str(directory), "reason": "No explicitly named " + split + " metrics; no train/val fallback."})
                continue
            strand = str(meta.get("strand", "unknown"))
            dataset = meta.get("dataset_id", meta.get("dataset", strand))
            if isinstance(dataset, (dict, list)):
                dataset = json.dumps(dataset, sort_keys=True, ensure_ascii=False)
            feature_scheme = meta.get("feature_scheme", "unknown")
            if isinstance(feature_scheme, (dict, list)):
                feature_scheme = json.dumps(feature_scheme, sort_keys=True, ensure_ascii=False)
            model = str(meta.get("model_type", meta.get("model", "unknown")))
            variant, readout = str(meta.get("litemamba_variant", "unknown")), str(meta.get("readout", "unknown"))
            model_id = model + "|variant=" + variant + "|readout=" + readout
            data_signature, signature_status = _data_signature(result, directory)
            config_signature, config_status = _training_signature(result, directory)
            for probability_type, metrics in variants.items():
                row = {"run_dir": str(directory), "evaluation_split": split, "dataset": dataset, "strand": strand,
                       "feature_scheme": feature_scheme, "model_type": model, "litemamba_variant": variant,
                       "readout": readout, "model_id": model_id, "seed": meta.get("seed", "unknown"),
                       "data_signature": data_signature, "data_signature_status": signature_status,
                       "training_config_signature": config_signature, "training_config_status": config_status,
                       "probability_type": probability_type, "metric_file": str(metric_file) if metric_file else str(directory / "results.json")}
                row.update({key: metrics.get(key) for key in SUMMARY_METRICS})
                row.update({key: metrics.get(key) for key in ("n", "positive_support", "negative_support")})
                rows.append(row)
        except (OSError, ValueError, TypeError) as exc:
            skipped.append({"run_dir": str(directory), "reason": str(exc)})
    table = pd.DataFrame(rows)
    pd.DataFrame(skipped, columns=["run_dir", "reason"]).to_csv(output / "aggregation_skipped.csv", index=False)
    if not len(table):
        raise ValueError("没有找到可汇总的 {} 结果；详情见 aggregation_skipped.csv。".format(split))
    group_columns = ["evaluation_split", "dataset", "strand", "feature_scheme", "data_signature", "data_signature_status", "model_type", "litemamba_variant", "readout", "model_id", "training_config_signature", "probability_type"]
    duplicate_columns = group_columns + ["seed"]
    duplicates = table.duplicated(duplicate_columns, keep=False)
    if duplicates.any():
        table.loc[duplicates].to_csv(output / "duplicate_seed_runs.csv", index=False)
        raise ValueError("相同实验设置和 seed 出现多个结果，拒绝重复计为独立 seed；见 duplicate_seed_runs.csv。")
    table.to_csv(output / "run_metrics.csv", index=False)
    summary_rows = []
    for group, part in table.groupby(group_columns, dropna=False, sort=True):
        row = dict(zip(group_columns, group))
        row["seed_count"] = len(part)
        row["training_config_verified"] = bool(part["training_config_status"].ne("unknown").all())
        row["seeds"] = ";".join(part["seed"].astype(str))
        for metric in SUMMARY_METRICS:
            values = pd.to_numeric(part[metric], errors="coerce").dropna()
            row[metric + "_mean"] = float(values.mean()) if len(values) else None
            row[metric + "_std"] = float(values.std(ddof=1)) if len(values) > 1 else None
            row[metric + "_valid_seeds"] = len(values)
        summary_rows.append(row)
    pd.DataFrame(summary_rows).to_csv(output / "summary_by_model.csv", index=False)
    paired = []
    pairing_columns = ["evaluation_split", "dataset", "strand", "feature_scheme", "data_signature", "seed", "probability_type"]
    verified = table[table["data_signature_status"].eq("input_content_sha256")]
    for group, part in verified.groupby(pairing_columns, dropna=False, sort=True):
        if str(group[5]).strip().lower() in UNKNOWN:
            continue
        records = part.sort_values("model_id").to_dict("records")
        for left, right in itertools.combinations(records, 2):
            row = {**dict(zip(pairing_columns, group)), "model_a": left["model_id"], "model_b": right["model_id"],
                   "training_config_signature_a": left["training_config_signature"],
                   "training_config_signature_b": right["training_config_signature"],
                   "delta_definition": "model_a - model_b; same seed and split; observational paired difference, not a significance claim"}
            for metric in SUMMARY_METRICS:
                a, b = left.get(metric), right.get(metric)
                row[metric + "_delta"] = float(a - b) if a is not None and b is not None and pd.notna(a) and pd.notna(b) else None
            paired.append(row)
    pd.DataFrame(paired, columns=pairing_columns + ["model_a", "model_b", "training_config_signature_a", "training_config_signature_b", "delta_definition"] + [m + "_delta" for m in SUMMARY_METRICS]).to_csv(output / "paired_seed_deltas.csv", index=False)
    report = {"schema_version": VERSION, "evaluation_split": split, "included_run_count": len(roots) - len(skipped),
              "metric_rows": len(table), "skipped_run_count": len(skipped),
              "standard_deviation": "Sample standard deviation, ddof=1; null for fewer than two valid seeds.",
              "selection_policy": "All supplied completed runs with the requested split; no best-test-seed selection.",
              "training_config_grouping": "Within-model summaries also require the same training configuration signature. Seed, device and paths are excluded. Missing legacy configuration is explicitly unknown and cannot verify identical training settings.",
              "pairing_note": "Paired deltas require the same seed and the same content-SHA256 signature of input files, including split metadata. Runs missing fingerprints remain separate and are not paired. Read/identity provenance still depends on the supplied metadata.",
              "files": ["run_metrics.csv", "summary_by_model.csv", "paired_seed_deltas.csv", "aggregation_skipped.csv"]}
    _save_json(output / "aggregation.json", report)
    return report


def main(argv=None):
    argv = list(sys.argv[1:] if argv is None else argv)
    if argv and argv[0].startswith("--") and argv[0] not in ("--help", "-h"):
        argv.insert(0, "analyze")
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    subparsers = parser.add_subparsers(dest="command", required=True)
    analyze = subparsers.add_parser("analyze", help="从 CSV/CSV.gz 预测重新计算指标和图源")
    analyze.add_argument("--predictions", required=True)
    analyze.add_argument("--output_dir", required=True)
    analyze.add_argument("--bootstrap", type=int, default=200)
    analyze.add_argument("--seed", type=int, default=42)
    analyze.add_argument("--threshold", type=float, default=0.5)
    analyze.add_argument("--plots", action=argparse.BooleanOptionalAction, default=True)
    aggregate = subparsers.add_parser("aggregate", help="汇总全部独立 seed，默认只读取 test 结果")
    aggregate.add_argument("--runs", nargs="+", required=True)
    aggregate.add_argument("--output_dir", required=True)
    aggregate.add_argument("--split", choices=SPLIT_NAMES, default="test")
    redraw = subparsers.add_parser("plots", help="从已保存的 CSV 图源重新绘图")
    redraw.add_argument("--output_dir", required=True)
    args = parser.parse_args(argv)
    if args.command == "analyze":
        report = analyze_predictions(args.predictions, args.output_dir, bootstrap=args.bootstrap,
                                     seed=args.seed, threshold=args.threshold, plots=args.plots)
        print(json.dumps({"row_count": report["row_count"], "overall": report["overall"],
                          "output_dir": str(Path(args.output_dir).resolve())}, ensure_ascii=False, indent=2))
    elif args.command == "aggregate":
        print(json.dumps(aggregate_runs(args.runs, args.output_dir, split=args.split), ensure_ascii=False, indent=2))
    else:
        print(json.dumps({"plots": make_basic_plots(args.output_dir)}, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
