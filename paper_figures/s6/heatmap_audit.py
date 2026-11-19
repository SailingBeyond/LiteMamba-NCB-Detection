#!/usr/bin/env python3
"""热图最低值与测试集来源核查；不加载模型，不读取服务器原始特征。

热图显示逐背景准确率的三个训练 seed 等权均值。单 seed 最低值单独
报告，不能将它称为热图中最低的格子。测试集来源仅在绘图表与随包
冻结的完整逐背景计数逐项吻合时标记 verified；自定义表仍可正常绘图。
"""
from __future__ import annotations

import argparse
import csv
import json
import math
from pathlib import Path

import pandas as pd

from collect_results import SEEDS, read_frozen_snapshot


TASKS = ("6-class", "5X", "5Y")
TOLERANCE = 1e-12


def _json(value):
    return json.dumps(value, ensure_ascii=False, sort_keys=True)


def _write_csv(path, records, fields):
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8-sig", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(records)


def _integer(value):
    try:
        number = float(value)
        if math.isfinite(number) and number == int(number):
            return int(number)
    except (TypeError, ValueError, OverflowError):
        pass
    return None


def _matches_row(row, reference):
    return (reference is not None
            and _integer(row.n_samples) == reference[1]
            and math.isclose(float(row.accuracy), reference[0] / reference[1],
                             rel_tol=0, abs_tol=TOLERANCE))


def _count_for_row(row, frozen_counts):
    key = (str(row.task), int(row.seed), str(row.motif))
    reference = frozen_counts.get(key)
    if _matches_row(row, reference):
        return reference[0], "frozen_per_motif_integer_count"
    product = float(row.accuracy) * float(row.n_samples)
    rounded = round(product)
    if math.isclose(product, rounded, rel_tol=0, abs_tol=1e-7):
        return int(rounded), "recovered_integer_from_accuracy_times_n"
    return None, "not_integer_recoverable_from_plot_table"


def _frozen_evidence(frame):
    """核查冻结的元数据及当前绘图表；不以当前表的文件名推定来源。"""
    try:
        snapshot, manifest = read_frozen_snapshot()
    except Exception as error:
        return {}, {
            "status": "not_verified", "reason": str(error),
            "source": "bundled frozen result export; server feature files not read",
            "plot_frame_exactly_matches_frozen_per_motif": False,
            "runs": [], "tasks": {},
        }

    reference = {}
    specs = {}
    for run in snapshot["runs"]:
        spec = run["spec"]
        if spec["model"] != "GatedConv":
            continue
        task, seed = str(spec["task"]), int(spec["seed"])
        specs[(task, seed)] = run
        for motif in run.get("per_motif", []):
            reference[(task, seed, str(motif["motif"]))] = (
                int(motif["correct"]), int(motif["n"]))

    keys = [(str(row.task), int(row.seed), str(row.motif))
            for row in frame.itertuples(index=False)]
    frame_matches = (
        len(keys) == len(set(keys)) == len(reference)
        and set(keys) == set(reference)
        and all(_matches_row(row, reference.get(key))
                for row, key in zip(frame.itertuples(index=False), keys)))
    report = {
        "status": "not_verified",
        "source": "bundled frozen result export; server feature files not read",
        "snapshot_sha256": manifest["sha256"],
        "plot_frame_exactly_matches_frozen_per_motif": bool(frame_matches),
        "count_unit": "feature window records, excluding the header row",
        "interpretation_zh": "10%指各任务已选定数据的窗口记录比例，不是原始测序read数量比例。三个seed重复评估同一测试集，样本量不乘以三。",
        "runs": [], "tasks": {},
    }
    for task in TASKS:
        task_runs = []
        identities = []
        for seed in SEEDS:
            run = specs.get((task, seed))
            if run is None:
                continue
            dataset = run["spec"].get("dataset", {})
            splits = {}
            for split in ("train", "val", "test"):
                entry = dataset.get(split, {})
                feature, meta = entry.get("feature", {}), entry.get("metadata", {})
                feature_lines = _integer(feature.get("physical_lines"))
                metadata_lines = _integer(meta.get("physical_lines"))
                splits[split] = {
                    "n_windows": feature_lines - 1 if feature_lines is not None else None,
                    "feature_sha256": feature.get("sha256"),
                    "metadata_sha256": meta.get("sha256"),
                    "metadata_n_windows": metadata_lines - 1 if metadata_lines is not None else None,
                }
            counts = [splits[name]["n_windows"] for name in ("train", "val", "test")]
            total = sum(counts) if all(n is not None and n >= 0 for n in counts) else None
            test_n = splits["test"]["n_windows"]
            fraction = test_n / total if total and test_n is not None else None
            raw_n = int(run["raw_summary"]["n"])
            frozen_n = sum(int(m["n"]) for m in run.get("per_motif", []))
            selected = frame[frame.task.eq(task) & frame.seed.eq(seed)]
            plotted_n = _integer(selected.n_samples.sum())
            expected_keys = {key for key in reference if key[:2] == (task, seed)}
            actual_keys = {(str(row.task), int(row.seed), str(row.motif))
                           for row in selected.itertuples(index=False)}
            run_frame_match = (
                len(selected) == len(actual_keys) == len(expected_keys)
                and actual_keys == expected_keys
                and all(_matches_row(row, reference.get((task, seed, str(row.motif))))
                        for row in selected.itertuples(index=False)))
            ten_percent = fraction is not None and math.isclose(fraction, .1, rel_tol=0, abs_tol=1e-12)
            sums_match = test_n is not None and test_n == raw_n == frozen_n == plotted_n
            metadata_match = all(s["n_windows"] is not None and s["n_windows"] == s["metadata_n_windows"]
                                 for s in splits.values())
            row = {
                "task": task, "seed": int(seed), "splits": splits,
                "test_fraction_of_train_val_test": fraction,
                "test_is_ten_percent": bool(ten_percent),
                "raw_summary_n": raw_n, "frozen_per_motif_n_sum": frozen_n,
                "plot_per_motif_n_sum": plotted_n,
                "test_counts_all_equal": bool(sums_match),
                "feature_metadata_counts_equal": bool(metadata_match),
                "run_plot_rows_match_frozen": bool(run_frame_match),
            }
            row["status"] = "verified" if frame_matches and ten_percent and sums_match and metadata_match and run_frame_match else "not_verified"
            identities.append(tuple(splits["test"].get(k) for k in (
                "feature_sha256", "metadata_sha256", "n_windows")))
            task_runs.append(row)
        same_test = (len(identities) == len(SEEDS) and len(set(identities)) == 1
                     and all(value is not None for value in identities[0]))
        for row in task_runs:
            row["three_seeds_same_test"] = bool(same_test)
            if not same_test:
                row["status"] = "not_verified"
        task_status = (len(task_runs) == len(SEEDS)
                       and all(row["status"] == "verified" for row in task_runs))
        report["tasks"][task] = {"status": "verified" if task_status else "not_verified",
                                  "three_seeds_same_test": bool(same_test)}
        report["runs"].extend(task_runs)
    if frame_matches and all(report["tasks"][task]["status"] == "verified" for task in TASKS):
        report["status"] = "verified"
    else:
        report["reason"] = "Current plot table and/or split provenance differs from the complete frozen evidence; do not assert a verified 10% test source for custom data."
    return reference, report


def write_heatmap_audit(root, frame, aggregated):
    """写出两种最低值、并列背景与test来源；返回路径、summary及全局最低值。"""
    root = Path(root)
    out = root / "plot_data"
    out.mkdir(parents=True, exist_ok=True)
    frame = frame[frame.model.eq("GatedConv")].copy()
    frozen_counts, audit = _frozen_evidence(frame)
    summary, worst = [], []
    task_order = [task for task in TASKS if task in set(frame.task)]
    task_order += sorted(set(frame.task) - set(TASKS))
    for task in task_order:
        rows = frame[frame.task.eq(task)]
        means = aggregated[aggregated.task.eq(task) & aggregated.complete_seeds].dropna(subset=["accuracy_mean"])
        single_min = float(rows.accuracy.min())
        mean_min = float(means.accuracy_mean.min()) if len(means) else None
        single_rows = rows[(rows.accuracy - single_min).abs().le(TOLERANCE)].sort_values(["motif", "seed"])
        mean_motifs = sorted(means.loc[(means.accuracy_mean - mean_min).abs().le(TOLERANCE), "motif"].astype(str)) if mean_min is not None else []
        seed_n = {str(int(seed)): int(group.n_samples.sum()) for seed, group in rows.groupby("seed", sort=True)}
        single_details = [{"motif": str(row.motif), "seed": int(row.seed)}
                          for row in single_rows.itertuples(index=False)]
        record = {
            "task": str(task), "heatmap_mean_minimum": mean_min,
            "heatmap_mean_minimum_percent": mean_min * 100 if mean_min is not None else None,
            "mean_minimum_motifs": _json(mean_motifs),
            "mean_minimum_n_tied_backgrounds": len(mean_motifs),
            "single_seed_minimum": single_min, "single_seed_minimum_percent": single_min * 100,
            "single_seed_minimum_seed_and_motif": _json(single_details),
            "single_seed_minimum_n_tied_rows": len(single_details),
            "test_n_per_seed": _json(seed_n),
            "test_source_status": audit.get("tasks", {}).get(task, {}).get("status", "not_verified"),
            "heatmap_statistic": "equal-weight mean of three seed accuracies",
        }
        summary.append(record)
        groups = [("heatmap_mean_minimum", rows[rows.motif.isin(mean_motifs)]),
                  ("single_seed_minimum", single_rows)]
        mean_lookup = dict(zip(means.motif, means.accuracy_mean))
        for criterion, selected in groups:
            for row in selected.sort_values(["motif", "seed"]).itertuples(index=False):
                correct, source = _count_for_row(row, frozen_counts)
                worst.append({
                    "task": str(task), "criterion": criterion, "motif": str(row.motif),
                    "seed": int(row.seed), "accuracy": float(row.accuracy),
                    "n_correct": correct, "n_samples": _integer(row.n_samples),
                    "correct_count_source": source,
                    "heatmap_mean_accuracy": float(mean_lookup[row.motif]) if row.motif in mean_lookup else None,
                })
        mean_text = f"{mean_min:.8f} ({100 * mean_min:.4f}%)" if mean_min is not None else "无完整三seed均值"
        single_text = "; ".join(f"{x['motif']} seed={x['seed']}" for x in single_details)
        print(f"[HEATMAP MINIMUM] {task}: 热图三seed均值最低={mean_text}; 背景={','.join(mean_motifs)}")
        print(f"[SINGLE-SEED MINIMUM] {task}: 单seed最低={single_min:.8f} ({100 * single_min:.4f}%); {single_text}")
        print(f"[HEATMAP TEST] {task}: 每seed窗口数={_json(seed_n)}; 来源核查={record['test_source_status']}")

    summary_path = out / "heatmap_minimum_summary.csv"
    worst_path = out / "heatmap_worst_backgrounds.csv"
    audit_json_path = out / "heatmap_test_split_audit.json"
    audit_csv_path = out / "heatmap_test_split_audit.csv"
    summary_fields = ["task", "heatmap_mean_minimum", "heatmap_mean_minimum_percent", "mean_minimum_motifs",
                      "mean_minimum_n_tied_backgrounds", "single_seed_minimum", "single_seed_minimum_percent",
                      "single_seed_minimum_seed_and_motif", "single_seed_minimum_n_tied_rows", "test_n_per_seed",
                      "test_source_status", "heatmap_statistic"]
    _write_csv(summary_path, summary, summary_fields)
    _write_csv(worst_path, worst, ["task", "criterion", "motif", "seed", "accuracy", "n_correct", "n_samples",
                                  "correct_count_source", "heatmap_mean_accuracy"])
    audit_json_path.write_text(json.dumps(audit, ensure_ascii=False, indent=2, allow_nan=False) + "\n", encoding="utf-8")
    flat_runs = []
    for run in audit["runs"]:
        flat = {key: value for key, value in run.items() if key != "splits"}
        for split, evidence in run["splits"].items():
            flat.update({split + "_" + key: value for key, value in evidence.items()})
        flat_runs.append(flat)
    audit_fields = ["task", "seed", "status", "test_fraction_of_train_val_test", "test_is_ten_percent",
                    "raw_summary_n", "frozen_per_motif_n_sum", "plot_per_motif_n_sum", "test_counts_all_equal",
                    "feature_metadata_counts_equal", "run_plot_rows_match_frozen", "three_seeds_same_test"]
    audit_fields += [split + "_" + key for split in ("train", "val", "test") for key in (
        "n_windows", "feature_sha256", "metadata_sha256", "metadata_n_windows")]
    _write_csv(audit_csv_path, flat_runs, audit_fields)
    return {
        "paths": [str(path) for path in (summary_path, worst_path, audit_json_path, audit_csv_path)],
        "summary": summary,
        "global_single_seed_minimum": min((row["single_seed_minimum"] for row in summary), default=None),
        "global_mean_minimum": min((row["heatmap_mean_minimum"] for row in summary if row["heatmap_mean_minimum"] is not None), default=None),
        "test_source_status": audit["status"],
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--result-root", "--result_root", type=Path, required=True)
    args = parser.parse_args()
    from plot_common import aggregate_motifs, load_motifs
    frame = load_motifs(args.result_root)
    result = write_heatmap_audit(args.result_root, frame, aggregate_motifs(frame))
    print(json.dumps({key: value for key, value in result.items() if key != "summary"}, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
