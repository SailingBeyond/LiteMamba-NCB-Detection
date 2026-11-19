#!/usr/bin/env python3
"""汇总固定 seed42 的训练/验证消融，并打包可上传的小结果文件。"""
from __future__ import annotations

import argparse
import csv
from datetime import datetime, timezone
import hashlib
import html
import json
import math
from pathlib import Path
import shutil
import tarfile


SCHEMES = ("no_raw", "no_context", "no_responses", "no_statistics")
CLASS_NAMES = ("A", "T", "C", "G", "X", "Y")
CONFIG_KEYS = (
    "epochs", "patience", "lr", "batch_size", "raw_signal_len", "dwell_divisor",
    "quality_divisor", "accum_steps", "weight_decay", "label_smoothing", "clip_grad",
    "monitor", "model_dim", "num_blocks", "state_dim", "litemamba_variant", "readout",
    "bootstrap", "no_plots", "no_embeddings", "embedding_dtype", "no_epoch_logits",
    "save_epoch_predictions", "staged_exports",
)
LABELS = {"full": "完整模型（已有）", "no_raw": "去除原始信号分支",
          "no_context": "去除碱基one-hot", "no_responses": "去除质量/比对响应",
          "no_statistics": "去除显式信号统计量"}


def read_json(path):
    with Path(path).open(encoding="utf-8") as stream:
        return json.load(stream)


def sha256(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(8 * 1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def save_json(path, value):
    path = Path(path)
    temporary = path.with_name(path.name + ".tmp")
    temporary.write_text(json.dumps(value, ensure_ascii=False, indent=2, allow_nan=False) + "\n", encoding="utf-8")
    temporary.replace(path)


def fingerprint_content(fingerprints):
    """用内容SHA与大小比较输入，路径和修改时间另作来源记录。"""
    result = {}
    for split in ("train", "val", "test"):
        result[split] = {}
        for kind in ("feature", "metadata"):
            item = fingerprints[split][kind]
            if not item.get("sha256"):
                raise ValueError("数据内容指纹缺少SHA：" + split + "/" + kind)
            result[split][kind] = {"sha256": item["sha256"], "size": item["size"]}
    return result


def read_history(run_dir, monitor):
    history = []
    with (run_dir / "epoch_metrics.jsonl").open(encoding="utf-8") as stream:
        for line in stream:
            if line.strip():
                history.append(json.loads(line))
    if not history:
        raise ValueError("缺少训练历史：" + str(run_dir))
    epochs = [int(row["epoch"]) for row in history]
    if epochs != list(range(1, len(epochs) + 1)):
        raise ValueError("训练历史epoch不是从1连续递增：" + str(run_dir))
    metric = "loss" if monitor == "val_loss" else "macro_f1"
    for row in history:
        if row.get("monitor", monitor) != monitor:
            raise ValueError("历史选模规则改变：" + str(run_dir))
        if not math.isfinite(float(row["val"][metric])):
            raise ValueError("验证指标无效：" + str(run_dir))
    # min/max遇到相同值时保留首次，和原训练器的严格改进规则一致。
    select = min if monitor == "val_loss" else max
    best = select(history, key=lambda row: row["val"][metric])
    return history, best


def metrics_from_confusion(matrix):
    if len(matrix) != 6 or any(len(row) != 6 for row in matrix):
        raise ValueError("混淆矩阵必须为固定A/T/C/G/X/Y的6×6")
    cm = []
    for row in matrix:
        values = [int(value) for value in row]
        if any(value < 0 or value != original for value, original in zip(values, row)):
            raise ValueError("混淆计数必须为非负整数")
        cm.append(values)
    total = sum(map(sum, cm))
    if total == 0:
        raise ValueError("验证集为空")
    classes = []
    for index, name in enumerate(CLASS_NAMES):
        tp = cm[index][index]
        support = sum(cm[index])
        predicted = sum(row[index] for row in cm)
        precision = tp / predicted if predicted else 0.0
        recall = tp / support if support else 0.0
        f1 = 2 * tp / (support + predicted) if support + predicted else 0.0
        classes.append(dict(class_id=index, class_name=name, support=support,
                            predicted_positive=predicted, tp=tp,
                            precision=precision, recall=recall, f1=f1))
    natural = sum(sum(row) for row in cm[:4])
    ncb = sum(sum(row) for row in cm[4:])
    false_ncb = sum(sum(row[4:]) for row in cm[:4])
    detected_ncb = sum(sum(row[4:]) for row in cm[4:])
    correct_identity = cm[4][4] + cm[5][5]
    wrong_identity = cm[4][5] + cm[5][4]
    result = dict(
        n=total, accuracy=sum(cm[i][i] for i in range(6)) / total,
        macro_f1=sum(row["f1"] for row in classes) / 6,
        balanced_accuracy=sum(row["recall"] for row in classes) / 6,
        canonical_false_positive_rate=false_ncb / natural if natural else None,
        any_ncb_precision=detected_ncb / (detected_ncb + false_ncb) if detected_ncb + false_ncb else None,
        any_ncb_recall=detected_ncb / ncb if ncb else None,
        correct_xy_identity_recall=correct_identity / ncb if ncb else None,
        xy_swap_rate=wrong_identity / ncb if ncb else None,
    )
    return result, classes


def check_configuration(config, baseline, scheme):
    if (config.get("model_type"), config.get("seed"), config.get("feature_scheme")) != ("litemamba", 42, scheme):
        raise ValueError("模型/seed/消融方案不符合固定诊断计划：" + scheme)
    for key in CONFIG_KEYS:
        if key not in config or key not in baseline or config[key] != baseline[key]:
            raise ValueError("诊断与full配置不同：" + scheme + "/" + key)
    if config.get("source_sha256") != baseline.get("source_sha256"):
        raise ValueError("诊断与full源码SHA不同：" + scheme)
    if fingerprint_content(config["input_fingerprints"]) != fingerprint_content(baseline["input_fingerprints"]):
        raise ValueError("诊断与full输入内容不同：" + scheme)


def check_no_test(run_dir):
    for relative in ("test_started.json", "test/export_complete.json", "test/predictions.csv.gz"):
        if (run_dir / relative).exists():
            raise ValueError("诊断目录出现测试推理标记/产物：" + str(run_dir / relative))
    result = read_json(run_dir / "results.json")
    if result.get("status") != "train_complete" or result.get("test_evaluated") is not False:
        raise ValueError("诊断任务尚未完成train/val或已经评价test：" + str(run_dir))
    if set(result.get("metrics_by_split", {})) - {"train", "val"}:
        raise ValueError("诊断结果出现train/val以外指标：" + str(run_dir))
    return result


def write_csv(path, rows):
    columns = list(dict.fromkeys(key for row in rows for key in row))
    with path.open("w", encoding="utf-8-sig", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=columns)
        writer.writeheader()
        writer.writerows(rows)


def format_percent(value):
    return "—" if value is None else "{:.4f}".format(100 * value)


def write_html(path, rows, monitor):
    cells = []
    for row in rows:
        cells.append("<tr>" + "".join("<td>" + html.escape(str(value)) + "</td>" for value in [
            LABELS[row["feature_scheme"]], row["selected_epoch"],
            format_percent(row["val_macro_f1"]),
            "{:+.4f}".format(row["delta_val_macro_f1_pp"]),
            format_percent(row["val_accuracy"]),
            format_percent(row["canonical_false_positive_rate"]),
            format_percent(row["correct_xy_identity_recall"]),
        ]) + "</tr>")
    document = """<!doctype html><html lang="zh-CN"><meta charset="utf-8">
<title>LiteMamba seed42 验证集诊断</title><style>
body{font-family:system-ui,'Microsoft YaHei',sans-serif;max-width:1200px;margin:40px auto;padding:0 24px;color:#173538;line-height:1.7}
table{border-collapse:collapse;width:100%;font-variant-numeric:tabular-nums}th,td{padding:10px 12px;text-align:right;border-bottom:1px solid #dce8e3}
th{background:#e8f1ec}th:first-child,td:first-child{text-align:left}h1{font-size:26px}.note{padding:14px 18px;background:#f3f7f5}
</style><h1>LiteMamba：固定seed42的四组输入消融</h1>
<p>只比较已有full和4个新训练任务的验证集结果。所有checkpoint沿用同一选择规则：<b>MONITOR</b>。
比较主指标是该checkpoint的验证集macro-F1，差值为“去除式 − full”的百分点。</p>
<table><thead><tr><th>方案</th><th>选中epoch</th><th>macro-F1 %</th><th>差值pp</th><th>准确率 %</th><th>天然误报 %</th><th>正确X/Y恢复 %</th></tr></thead><tbody>ROWS</tbody></table>
<p class="note">单个训练种子用于诊断，不能证明最终优越性。验证集比较不等于Figure4外部测试表现。
正差值可提示该输入组在本次训练下有负面作用，负差值可提示其有帮助；需要多种子确认，不自动据此修改架构。</p>
<p>训练历史和各类指标保存在同目录CSV。此报告未读取或比较baseline测试集指标，也未为4个诊断任务进行test或Figure4推理。
原引擎仍会读取test输入用于固定预处理缓存、内容指纹及read隔离审计。</p></html>"""
    path.write_text(document.replace("MONITOR", html.escape(monitor)).replace("ROWS", "".join(cells)), encoding="utf-8")


def summarize(baseline_run, diagnostic_dir):
    baseline_run = Path(baseline_run).expanduser().resolve()
    diagnostic_dir = Path(diagnostic_dir).expanduser().resolve()
    baseline = read_json(baseline_run / "queue_requested_config.json")
    if (baseline.get("model_type"), baseline.get("seed"), baseline.get("feature_scheme")) != ("litemamba", 42, "full"):
        raise ValueError("baseline必须为已有full/litemamba/seed_42")
    monitor = baseline["monitor"]
    if monitor not in ("val_loss", "val_macro_f1"):
        raise ValueError("不支持的checkpoint选择规则")
    out = diagnostic_dir / "diagnostic_summary"
    out.mkdir(parents=True, exist_ok=True)
    rows, class_rows, curves, source_files = [], [], [], []
    for scheme in ("full",) + SCHEMES:
        run_dir = baseline_run if scheme == "full" else diagnostic_dir / scheme / "litemamba" / "seed_42"
        config = read_json(run_dir / "queue_requested_config.json")
        check_configuration(config, baseline, scheme)
        result = None if scheme == "full" else check_no_test(run_dir)
        history, selected = read_history(run_dir, monitor)
        if result is not None and result.get("best_epoch") != selected["epoch"]:
            raise ValueError("保存的best_epoch与历史选择规则不一致：" + scheme)
        metric_path = run_dir / "val" / "analysis" / "metrics.json"
        exported = read_json(metric_path)
        raw = exported["overall"]["raw"]
        metric, per_class = metrics_from_confusion(raw["confusion_matrix"])
        for key in ("accuracy", "macro_f1", "balanced_accuracy"):
            if abs(float(raw[key]) - metric[key]) > 1e-9:
                raise ValueError("验证指标与混淆计数不一致：" + scheme + "/" + key)
        if int(raw["n"]) != metric["n"]:
            raise ValueError("验证样本数不一致：" + scheme)
        row = dict(feature_scheme=scheme, model_type="litemamba", seed=42,
                   baseline_reused=scheme == "full", monitor=monitor,
                   selected_epoch=selected["epoch"], epochs_completed=len(history),
                   val_loss=selected["val"]["loss"], val_macro_f1=metric.pop("macro_f1"),
                   val_accuracy=metric.pop("accuracy"), **metric,
                   selected_epoch_train_loss=selected["train"].get("loss"),
                   selected_epoch_train_accuracy=selected["train"].get("accuracy"),
                   val_nll=raw.get("nll"), val_brier=raw.get("brier"), val_ece=raw.get("ece_15bins"),
                   raw_validation_metrics_file=str(metric_path), run_dir=str(run_dir))
        rows.append(row)
        for item in per_class:
            class_rows.append(dict(feature_scheme=scheme, seed=42, **item))
        for epoch in history:
            record = {key: epoch.get(key) for key in ("epoch", "lr", "next_lr", "monitor", "monitor_value", "improved", "patience_counter")}
            record.update(feature_scheme=scheme, seed=42, checkpoint_selected=epoch["epoch"] == selected["epoch"])
            for split in ("train", "val"):
                for key, value in epoch[split].items():
                    if value is None or isinstance(value, (str, int, float, bool)):
                        record[split + "_" + key] = value
            curves.append(record)
        for relative in ("queue_requested_config.json", "resolved_config.json", "model_config.json", "environment.json",
                         "data_audit.json", "calibration.json", "metrics.csv", "epoch_metrics.jsonl", "val/analysis/metrics.json"):
            path = run_dir / relative
            if path.is_file():
                source_files.append((path, "source_metadata/" + scheme + "/" + relative))
        if scheme != "full":
            source_files.append((run_dir / "results.json", "source_metadata/" + scheme + "/results.json"))
    full = rows[0]
    for row in rows:
        row["delta_val_macro_f1_pp"] = 100 * (row["val_macro_f1"] - full["val_macro_f1"])
        row["delta_val_accuracy_pp"] = 100 * (row["val_accuracy"] - full["val_accuracy"])
        row["delta_canonical_fpr_pp"] = (None if row["canonical_false_positive_rate"] is None or full["canonical_false_positive_rate"] is None
                                          else 100 * (row["canonical_false_positive_rate"] - full["canonical_false_positive_rate"]))
    write_csv(out / "diagnostic_summary.csv", rows)
    write_csv(out / "diagnostic_per_class.csv", class_rows)
    write_csv(out / "diagnostic_learning_curves.csv", curves)
    evidence = []
    for source, relative in source_files:
        target = out / relative
        target.parent.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(source, target)
        evidence.append(dict(package_path=relative, source_path=str(source), size_bytes=source.stat().st_size, sha256=sha256(source)))
    for name in ("diagnostic_plan.json", "diagnostic_status.json"):
        source = diagnostic_dir / name
        if source.is_file():
            shutil.copyfile(source, out / name)
    report = dict(status="complete", diagnostic_only=True,
                  created_at=datetime.now(timezone.utc).isoformat(), seed=42,
                  baseline_retrained=False, primary_comparison_metric="raw validation macro-F1 at the validation-selected checkpoint",
                  checkpoint_selection_monitor=monitor,
                  difference_direction="ablation minus full; percentage points",
                  no_diagnostic_test_inference=True, baseline_test_metrics_read=False,
                  schemes=list(SCHEMES), all_configs_sources_and_input_content_match_full=True,
                  input_fingerprints=fingerprint_content(baseline["input_fingerprints"]),
                  model_source_sha256=baseline["source_sha256"], results=rows,
                  limitations=["Single training seed, diagnostic rather than confirmatory.",
                               "Validation comparison does not establish external Figure4 performance.",
                               "The original engine reads test inputs for fixed preprocessing, fingerprints and read isolation checks, without test model inference."],
                  source_files=evidence)
    save_json(out / "diagnostic_comparison.json", report)
    write_html(out / "diagnostic_report.html", rows, monitor)
    package = diagnostic_dir / "diagnostic_results_for_analysis_v1_0_0.tar.gz"
    temporary = package.with_name(package.name + ".tmp")
    with tarfile.open(temporary, "w:gz") as archive:
        archive.add(out, arcname="diagnostic_summary", recursive=True)
    temporary.replace(package)
    print("[SUMMARY] " + str(out / "diagnostic_summary.csv"), flush=True)
    print("[RESULT PACKAGE] " + str(package), flush=True)
    return report


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--baseline_run", type=Path, required=True)
    parser.add_argument("--diagnostic_dir", type=Path, required=True)
    args = parser.parse_args()
    summarize(args.baseline_run, args.diagnostic_dir)


if __name__ == "__main__":
    main()
