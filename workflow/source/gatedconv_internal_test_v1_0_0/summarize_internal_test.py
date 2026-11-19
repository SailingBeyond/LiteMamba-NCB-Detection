#!/usr/bin/env python3
"""Summarize frozen internal-test predictions; this module never loads PyTorch."""
from __future__ import annotations

import argparse
import csv
from datetime import datetime, timezone
import html
import json
import math
from pathlib import Path
import statistics

import numpy as np


SEEDS = (42, 2024, 2025)
HISTORICAL_SEEDS = (42, 2024, 2025, 2026, 2027)
BASELINES = ("transformer", "convmixer", "bigru", "litemamba")
MODELS = ("gatedconv_target",) + BASELINES
CLASSES = ("A", "T", "C", "G", "X", "Y")
DISPLAY = {"gatedconv_target": "GatedConv", "transformer": "Transformer",
           "convmixer": "ConvMixer", "bigru": "BiGRU", "litemamba": "LiteMamba (SSM)"}
RATE_METRICS = ("accuracy", "macro_f1", "weighted_f1", "balanced_accuracy",
                "macro_precision", "macro_recall", "macro_roc_auc", "macro_average_precision",
                "micro_roc_auc", "micro_average_precision", "ece_15bins",
                "dna_to_xy_fpr", "xy_identity_recall", "xy_detection_recall")
METRICS = RATE_METRICS + ("nll", "brier")
LOWER_BETTER = {"dna_to_xy_fpr", "ece_15bins", "nll", "brier"}
CLASS_METRICS = ("precision", "recall", "specificity", "fpr", "fdr", "f1",
                 "roc_auc", "average_precision")


def read_json(path):
    return json.loads(Path(path).read_text(encoding="utf-8"))


def json_ready(value):
    if isinstance(value, dict):
        return {str(k): json_ready(v) for k, v in value.items()}
    if isinstance(value, (tuple, list)):
        return [json_ready(v) for v in value]
    if isinstance(value, np.generic):
        return json_ready(value.item())
    if isinstance(value, float) and not math.isfinite(value):
        return None
    return value


def atomic_json(path, value):
    path = Path(path)
    temporary = path.with_name(path.name + ".tmp")
    temporary.write_text(json.dumps(json_ready(value), ensure_ascii=False,
                                    indent=2, allow_nan=False) + "\n", encoding="utf-8")
    temporary.replace(path)


def write_csv(path, rows, fields=None):
    path = Path(path)
    if fields is None:
        fields = list(dict.fromkeys(key for row in rows for key in row))
    temporary = path.with_name(path.name + ".tmp")
    with temporary.open("w", encoding="utf-8", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)
    temporary.replace(path)


def ratio(a, b):
    return float(a / b) if b else None


def finite(value):
    return value is not None and math.isfinite(float(value))


def mean_sd(values):
    values = [float(v) for v in values if finite(v)]
    return {"n": len(values), "mean": statistics.mean(values) if values else None,
            "sd": statistics.stdev(values) if len(values) > 1 else None,
            "minimum": min(values) if values else None, "maximum": max(values) if values else None}


def recompute_confusion(metrics, where="metrics"):
    """Check saved scalar/class metrics against integer six-class counts."""
    try:
        raw = np.asarray(metrics["confusion_matrix"], dtype=np.float64)
    except (KeyError, TypeError, ValueError) as exc:
        raise ValueError(f"{where}: missing or invalid confusion matrix") from exc
    if (raw.shape != (6, 6) or not np.isfinite(raw).all() or np.any(raw < 0)
            or np.any(raw != np.floor(raw)) or np.any(raw > np.iinfo(np.int64).max)):
        raise ValueError(f"{where}: confusion matrix must contain 6x6 nonnegative integers")
    matrix = raw.astype(np.int64)
    support = matrix.sum(axis=1)
    n = int(support.sum())
    if n == 0:
        raise ValueError(f"{where}: empty evaluation")
    per_class = []
    saved_classes = metrics.get("per_class", [])
    saved_by_class = {}
    for i, row in enumerate(saved_classes):
        label = row.get("class_id", i)
        if label not in range(6) or label in saved_by_class:
            raise ValueError(f"{where}: repeated or invalid class id")
        saved_by_class[label] = row
    for k, name in enumerate(CLASSES):
        tp = int(matrix[k, k])
        fp = int(matrix[:, k].sum()) - tp
        fn = int(support[k]) - tp
        tn = n - tp - fp - fn
        row = {"class_id": k, "class_name": name, "support": int(support[k]),
               "tp": tp, "fp": fp, "fn": fn, "tn": tn,
               "precision": ratio(tp, tp + fp), "recall": ratio(tp, tp + fn),
               "specificity": ratio(tn, tn + fp), "fpr": ratio(fp, tn + fp),
               "fdr": ratio(fp, tp + fp), "f1": ratio(2 * tp, 2 * tp + fp + fn)}
        saved = saved_by_class.get(k, {})
        for key in ("tp", "fp", "fn", "tn", "support", "precision", "recall", "specificity", "fpr", "fdr", "f1"):
            if key in saved:
                check_number(saved[key], row[key], f"{where}/class_{k}/{key}")
        for key in ("roc_auc", "average_precision"):
            row[key] = float(saved[key]) if finite(saved.get(key)) else None
        per_class.append(row)
    derived = {"n": n, "support": n, "correct": int(np.trace(matrix)),
               "errors": int(n - np.trace(matrix)), "accuracy": float(np.trace(matrix) / n),
               "macro_f1": float(np.mean([row["f1"] or 0 for row in per_class])),
               "weighted_f1": float(sum((row["f1"] or 0) * row["support"] for row in per_class) / n),
               "balanced_accuracy": statistics.mean([row["recall"] for row in per_class if row["recall"] is not None]),
               "class_support": support.tolist(), "confusion_matrix": matrix.tolist(), "per_class": per_class}
    for key in ("precision", "recall"):
        values = [row[key] for row in per_class]
        derived["macro_" + key] = statistics.mean(values) if all(v is not None for v in values) else None
    for key in ("n", "support", "correct", "errors", "accuracy", "macro_f1", "weighted_f1", "balanced_accuracy",
                "macro_precision", "macro_recall"):
        if key in metrics:
            check_number(metrics[key], derived[key], f"{where}/{key}")
    if "class_support" in metrics and metrics["class_support"] != support.tolist():
        raise ValueError(f"{where}: saved class support differs from confusion matrix")
    for key in METRICS:
        if key not in derived:
            derived[key] = float(metrics[key]) if finite(metrics.get(key)) else None
    derived["dna_to_xy_fpr"] = ratio(int(matrix[:4, 4:].sum()), int(matrix[:4, :].sum()))
    derived["xy_identity_recall"] = ratio(int(matrix[4, 4] + matrix[5, 5]), int(matrix[4:, :].sum()))
    derived["xy_detection_recall"] = ratio(int(matrix[4:, 4:].sum()), int(matrix[4:, :].sum()))
    return derived


def check_number(saved, actual, where):
    if saved is None or actual is None:
        if saved is not None or actual is not None:
            raise ValueError(f"{where}: undefined value differs")
    elif not finite(saved) or not math.isclose(float(saved), float(actual), abs_tol=1e-10, rel_tol=1e-10):
        raise ValueError(f"{where}: saved metric differs from confusion matrix")


def test_metrics(record, where):
    split = record.get("metrics_by_split", {}).get("test")
    if split is None and isinstance(record.get("result"), dict):
        split = record["result"].get("metrics_by_split", {}).get("test")
    if split is None and isinstance(record.get("metrics"), dict):
        split = record["metrics"]
    if not isinstance(split, dict) or not all(kind in split for kind in ("raw", "calibrated")):
        raise ValueError(f"{where}: both raw and calibrated internal-test metrics are required")
    result = {kind: recompute_confusion(split[kind], f"{where}/{kind}") for kind in ("raw", "calibrated")}
    if result["raw"]["confusion_matrix"] != result["calibrated"]["confusion_matrix"]:
        raise ValueError(f"{where}: positive-temperature calibration changed argmax decisions")
    return result


def load_records(root, plan):
    records, baseline_index = [], {}
    planned = {int(item["seed"]): item for item in plan.get("runs", [])}
    if set(planned) != set(SEEDS) or len(plan.get("runs", [])) != 3:
        raise ValueError("The frozen evaluation plan must contain exactly seeds 42, 2024, 2025")
    for seed in SEEDS:
        folder = root / "runs" / "gatedconv_target" / f"seed_{seed}"
        complete = folder / "run_complete.json"
        path = complete if complete.is_file() else folder / "result.json"
        if not path.is_file():
            raise ValueError(f"Incomplete internal-test prediction: seed_{seed}")
        item = read_json(path)
        if item.get("status") not in (None, "complete", "test_complete"):
            raise ValueError(f"Incomplete run marker: {path}")
        if item.get("model") != "gatedconv_target" or item.get("seed") != seed:
            raise ValueError(f"Run marker model/seed identity differs: {path}")
        if item.get("plan_id") is not None and item["plan_id"] != plan.get("plan_id"):
            raise ValueError(f"Run marker belongs to a different evaluation plan: {path}")
        expected_hash = planned[seed].get("checkpoint_sha256")
        if expected_hash and item.get("checkpoint_sha256") != expected_hash:
            raise ValueError(f"Run checkpoint differs from frozen evaluation plan: {path}")
        records.append({"model": "gatedconv_target", "seed": seed,
                        "checkpoint_sha256": item.get("checkpoint_sha256"),
                        "metrics": test_metrics(item, str(path)), "source": str(path)})
    for item in plan.get("historical_baselines", []):
        model, seed = item.get("model"), item.get("seed")
        if model not in BASELINES or seed not in HISTORICAL_SEEDS:
            raise ValueError("Unexpected model/seed in historical baseline records")
        key = model, seed
        if key in baseline_index:
            raise ValueError(f"Duplicate historical baseline: {model}/seed_{seed}")
        value = {"model": model, "seed": seed, "checkpoint_sha256": item.get("checkpoint_sha256"),
                 "metrics": test_metrics(item, f"historical/{model}/{seed}"),
                 "source": item.get("run_dir", "frozen historical metadata")}
        baseline_index[key] = value
    required = {(model, seed) for model in BASELINES for seed in SEEDS}
    missing = required - set(baseline_index)
    if missing:
        raise ValueError("Missing same-seed historical baselines: " + str(sorted(missing)))
    records.extend(baseline_index[key] for key in sorted(required))
    expected_n = plan.get("dataset", {}).get("n_samples")
    expected_support = plan.get("dataset", {}).get("class_support")
    reference_support = records[0]["metrics"]["raw"]["class_support"]
    for item in records + list(baseline_index.values()):
        for kind, metric in item["metrics"].items():
            if expected_n is not None and metric["n"] != expected_n:
                raise ValueError(f"Different fixed-test sample count: {item['model']}/{item['seed']}/{kind}")
            if metric["class_support"] != reference_support:
                raise ValueError("Fixed internal-test class counts differ between models")
            if expected_support is not None and metric["class_support"] != expected_support:
                raise ValueError("Internal-test class counts differ from frozen dataset")
    complete_five = all((model, seed) in baseline_index for model in BASELINES for seed in HISTORICAL_SEEDS)
    historical = [baseline_index[(model, seed)] for model in BASELINES for seed in HISTORICAL_SEEDS] if complete_five else []
    return records, historical


def run_rows(records):
    rows = []
    for record in records:
        for kind, metrics in record["metrics"].items():
            rows.append({"model": record["model"], "seed": record["seed"], "split": "test",
                         "probability_kind": kind, "n_samples": metrics["n"],
                         "checkpoint_sha256": record["checkpoint_sha256"],
                         **{key: metrics.get(key) for key in METRICS}})
    return rows


def aggregate_rows(records):
    rows = []
    models = [model for model in MODELS if any(item["model"] == model for item in records)]
    for model in models:
        subset = [item for item in records if item["model"] == model]
        for kind in ("raw", "calibrated"):
            for metric in METRICS:
                rows.append({"model": model, "split": "test", "probability_kind": kind,
                             "metric": metric, "seeds": ",".join(str(item["seed"]) for item in subset),
                             "n_runs": len(subset), **mean_sd([item["metrics"][kind].get(metric) for item in subset]),
                             "unit": "fraction" if metric in RATE_METRICS else "native"})
    return rows


def class_rows(records):
    rows = []
    for record in records:
        for kind in ("raw", "calibrated"):
            for item in record["metrics"][kind]["per_class"]:
                rows.append({"model": record["model"], "seed": record["seed"],
                             "split": "test", "probability_kind": kind, **item})
    return rows


def aggregate_classes(rows):
    aggregate = []
    for model in MODELS:
        for kind in ("raw", "calibrated"):
            for label, name in enumerate(CLASSES):
                subset = [r for r in rows if (r["model"], r["probability_kind"], r["class_id"]) == (model, kind, label)]
                for metric in CLASS_METRICS:
                    aggregate.append({"model": model, "probability_kind": kind,
                                      "class_id": label, "class_name": name, "metric": metric,
                                      "n_runs": len(subset), "support_per_seed": subset[0]["support"] if subset else None,
                                      **mean_sd([r.get(metric) for r in subset])})
    return aggregate


def paired_rows(records):
    lookup = {(item["model"], item["seed"]): item for item in records}
    paired, aggregate = [], []
    for baseline in BASELINES:
        for kind in ("raw", "calibrated"):
            for metric in METRICS:
                values = []
                for seed in SEEDS:
                    gate = lookup[("gatedconv_target", seed)]["metrics"][kind].get(metric)
                    other = lookup[(baseline, seed)]["metrics"][kind].get(metric)
                    delta = float(gate - other) if finite(gate) and finite(other) else None
                    better = (delta < 0 if metric in LOWER_BETTER else delta > 0) if delta is not None else None
                    paired.append({"comparison": "GatedConv minus " + DISPLAY[baseline], "baseline": baseline,
                                   "seed": seed, "probability_kind": kind, "metric": metric,
                                   "gatedconv": gate, "baseline_value": other, "delta": delta,
                                   "delta_percentage_points": delta * 100 if delta is not None and metric in RATE_METRICS else None,
                                   "gatedconv_better": better,
                                   "preferred_direction": "lower" if metric in LOWER_BETTER else "higher"})
                    if delta is not None:
                        values.append(delta)
                stats = mean_sd(values)
                aggregate.append({"baseline": baseline, "probability_kind": kind, "metric": metric,
                                  "n_pairs": len(values), **stats,
                                  "mean_delta_percentage_points": stats["mean"] * 100 if stats["mean"] is not None and metric in RATE_METRICS else None,
                                  "sd_delta_percentage_points": stats["sd"] * 100 if stats["sd"] is not None and metric in RATE_METRICS else None,
                                  "n_gated_higher": sum(v > 0 for v in values), "n_equal": sum(v == 0 for v in values),
                                  "n_gated_lower": sum(v < 0 for v in values),
                                  "n_gated_better": sum(v < 0 if metric in LOWER_BETTER else v > 0 for v in values),
                                  "preferred_direction": "lower" if metric in LOWER_BETTER else "higher"})
    return paired, aggregate


def make_plot(directory, records):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    names = sorted(MODELS, key=lambda name: statistics.mean(
        item["metrics"]["raw"]["macro_f1"] for item in records if item["model"] == name), reverse=True)
    fig, axis = plt.subplots(figsize=(7.3, 4.1))
    all_values = []
    for y, name in enumerate(names):
        subset = sorted((item for item in records if item["model"] == name), key=lambda item: SEEDS.index(item["seed"]))
        values = [item["metrics"]["raw"]["macro_f1"] * 100 for item in subset]
        all_values.extend(values)
        color = "#117B92" if name == "gatedconv_target" else "#5C557A" if name == "transformer" else "#7F8A94"
        axis.scatter(values, np.asarray([-0.10, 0, 0.10]) + y, s=28, color=color, alpha=0.65, zorder=3)
        axis.errorbar(statistics.mean(values), y, xerr=statistics.stdev(values),
                      fmt="D", color=color, markersize=6, capsize=4, linewidth=1.5, zorder=4)
    axis.set_yticks(range(len(names)), [DISPLAY[name] for name in names])
    axis.invert_yaxis()
    span = max(all_values) - min(all_values)
    pad = max(0.08, span * 0.15)
    axis.set_xlim(min(all_values) - pad, max(all_values) + pad)
    axis.set_xlabel("Internal-test Macro-F1 (%)")
    axis.set_title("Same three seeds: 42, 2024, 2025", fontsize=12, loc="left")
    axis.grid(axis="x", color="#DFE5E8", linewidth=0.7)
    axis.set_axisbelow(True)
    axis.spines[["top", "right", "left"]].set_visible(False)
    axis.tick_params(axis="y", length=0)
    fig.text(0.99, 0.015, "Dots: individual seeds; diamond/error bar: mean ± sample SD.",
             ha="right", fontsize=8, color="#54616C")
    fig.tight_layout(rect=(0, 0.05, 1, 1))
    for suffix in ("png", "svg"):
        fig.savefig(directory / f"macro_f1_same3seeds.{suffix}", dpi=300, bbox_inches="tight")
    plt.close(fig)


def report_html(directory, plan, records, aggregate, paired, supplementary):
    def selected(model, key):
        return next(r for r in aggregate if r["model"] == model and r["probability_kind"] == "raw" and r["metric"] == key)
    def display_stat(model, key):
        row = selected(model, key)
        return "—" if row["mean"] is None else f"{row['mean'] * 100:.4f} ± {row['sd'] * 100:.4f}"
    order = sorted(MODELS, key=lambda name: selected(name, "macro_f1")["mean"], reverse=True)
    header = "<tr><th>模型</th><th>Macro-F1 (%)</th><th>Accuracy (%)</th><th>DNA→X/Y FPR (%)</th><th>X/Y 正确身份召回率 (%)</th></tr>"
    rows = "".join("<tr>" + f"<td>{html.escape(DISPLAY[name])}</td>" + "".join(
        f"<td>{display_stat(name, key)}</td>" for key in ("macro_f1", "accuracy", "dna_to_xy_fpr", "xy_identity_recall")) + "</tr>" for name in order)
    comparisons = [r for r in paired if r["probability_kind"] == "raw" and r["metric"] == "macro_f1"]
    pair_rows = "".join(f"<tr><td>GatedConv − {html.escape(DISPLAY[r['baseline']])}</td>"
                        f"<td>{r['mean_delta_percentage_points']:+.4f} ± {r['sd_delta_percentage_points']:.4f}</td>"
                        f"<td>{r['n_gated_better']}/{r['n_pairs']}</td></tr>" for r in comparisons)
    n = records[0]["metrics"]["raw"]["n"]
    counts = records[0]["metrics"]["raw"]["class_support"]
    supplementary_text = ("原四种模型的完整五种子结果另存于 historical_test_5seed_*.csv；"
                          "这部分不包含只有三个种子的 GatedConv，不与主表混合。" if supplementary else
                          "本次归档没有齐全的四模型×五种子元数据，因此没有生成不完整的五种子汇总。")
    text = f"""<!doctype html><html lang="zh-CN"><meta charset="utf-8"><title>GatedConv 固定内部测试评估</title>
<style>body{{font:16px/1.7 system-ui,sans-serif;color:#263441;max-width:1100px;margin:36px auto;padding:0 24px}}h1,h2{{color:#174E64}}table{{border-collapse:collapse;width:100%;margin:20px 0}}th,td{{border-bottom:1px solid #dce4e8;text-align:left;padding:9px 12px}}th{{background:#edf4f6}}.note{{background:#f0f5f7;padding:16px 20px;border-left:4px solid #117B92}}img{{max-width:850px;width:100%}}code{{font-size:90%}}small{{color:#586976}}</style>
<h1>GatedConv 固定内部测试评估</h1>
<p>本报告只分析内部独立测试集（test），不把 validation 或 Figure 4 外部测试结果混入主比较。
所有模型使用相同的固定种子 42、2024、2025，决策规则为 A/T/C/G/X/Y 六分类 argmax。</p>
<div class="note">每个模型每个种子评估 {n:,} 个窗口；六类样本量依次为 {', '.join(format(v, ',') for v in counts)}。
表中为三次运行的均值 ± 样本标准差。平均值排序和 3/3 同向变化不等于统计显著。</div>
<h2>相同三个种子的内部测试结果</h2><table>{header}{rows}</table>
<p>DNA→X/Y FPR 的分母为真实 A/T/C/G 窗口；X/Y 正确身份召回率要求 X 判为 X、Y 判为 Y。
它与“只要判成 X 或 Y 就算检出”的召回率不同。所有指标按窗口计数。</p>
<img src="macro_f1_same3seeds.png" alt="相同三个种子的内部测试 Macro-F1；散点为单种子，误差线为样本标准差">
<h2>按相同种子配对的差值</h2><table><tr><th>比较</th><th>Macro-F1 差值，百分点：均值 ± 样本标准差</th><th>GatedConv 更高的种子数</th></tr>{pair_rows}</table>
<p>{supplementary_text}</p>
<h2>解释范围与可复现信息</h2><ul>
<li>本次使用已经选定的三个检查点，只补测试预测；不训练、不重新选检查点、不在 test 上拟合温度或选择阈值。</li>
<li>raw 与 calibrated 是同一组训练运行的两种概率表示，不是两次独立实验。温度来自此前 validation；正温度不改变 argmax 分类结果。</li>
<li>原四模型的 test 已在先前分析中查看，本次 GatedConv 是返修过程中的后续固定评估；不把它描述成整项研究从未查看过的全新盲测。</li>
<li>沿用原始数据划分的独立性范围：阴性按真实 read ID 分组；阳性依赖每行对应一条独立 read 的实验设计与行代理。该报告不新增原始阳性 UUID 核验。</li>
<li>天然碱基与 X/Y 的数据来源及链方向混杂仍需独立分析；模型排名不能证明已排除这些因素。</li>
<li>逐窗口 logits、标签、原始及校准概率、读出表示、注意力与元数据保留在各 run 的 test 目录；后续作图或阈值分析可直接使用，无需重新预测。</li>
<li>各 run 的 analysis 目录保留原分析引擎生成的分组统计和 bootstrap 结果；跨种子标准差与读级 bootstrap 区间的含义不同。</li>
</ul><p>可下载分析表包括 <code>internal_test_3seed_runs.csv</code>、<code>internal_test_3seed_summary.csv</code>、
<code>paired_differences_3seed.csv</code>、<code>paired_summary_3seed.csv</code>、<code>per_class_3seed_runs.csv</code> 与 <code>per_class_3seed_summary.csv</code>。
比例在 CSV 中以 0–1 保存；配对表另有明确标注的百分点列。</p>
<small>冻结计划 ID：{html.escape(str(plan.get('plan_id', '')))}<br>生成时间：{datetime.now(timezone.utc).isoformat()}</small></html>"""
    temporary = directory / "report.html.tmp"
    temporary.write_text(text, encoding="utf-8")
    temporary.replace(directory / "report.html")


def summarize(output_dir):
    root = Path(output_dir).expanduser().resolve()
    plan = read_json(root / "evaluation_plan.json")
    records, historical = load_records(root, plan)
    directory = root / "analysis"
    directory.mkdir(parents=True, exist_ok=True)
    runs = run_rows(records)
    aggregate = aggregate_rows(records)
    classes = class_rows(records)
    paired, paired_aggregate = paired_rows(records)
    write_csv(directory / "internal_test_3seed_runs.csv", runs)
    write_csv(directory / "internal_test_3seed_summary.csv", aggregate)
    write_csv(directory / "per_class_3seed_runs.csv", classes)
    write_csv(directory / "per_class_3seed_summary.csv", aggregate_classes(classes))
    write_csv(directory / "paired_differences_3seed.csv", paired)
    write_csv(directory / "paired_summary_3seed.csv", paired_aggregate)
    if historical:
        write_csv(directory / "historical_test_5seed_runs.csv", run_rows(historical))
        write_csv(directory / "historical_test_5seed_summary.csv", aggregate_rows(historical))
    make_plot(directory, records)
    report_html(directory, plan, records, aggregate, paired_aggregate, bool(historical))
    result = {"status": "complete", "scope": "fixed_internal_test_only", "plan_id": plan.get("plan_id"),
              "primary_seeds": list(SEEDS), "primary_models": list(MODELS), "n_primary_training_runs": len(records),
              "probability_versions_are_independent_training_runs": False,
              "n_samples_per_run": records[0]["metrics"]["raw"]["n"],
              "class_support_per_run": records[0]["metrics"]["raw"]["class_support"],
              "confusion_matrices_validated": len(records) * 2 + (len(historical) - 12) * 2 if historical else len(records) * 2,
              "historical_five_seed_supplement": bool(historical),
              "historical_five_seed_models": list(BASELINES) if historical else [],
              "ranking_metric": "raw Macro-F1 mean across the same three seeds",
              "ranking": [r["model"] for r in sorted((r for r in aggregate if r["probability_kind"] == "raw" and r["metric"] == "macro_f1"), key=lambda r: r["mean"], reverse=True)],
              "statistical_significance_claimed": False,
              "created_at": datetime.now(timezone.utc).isoformat()}
    atomic_json(directory / "summary.json", result)
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output_dir", required=True)
    args = parser.parse_args()
    result = summarize(args.output_dir)
    print(json.dumps(result, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
