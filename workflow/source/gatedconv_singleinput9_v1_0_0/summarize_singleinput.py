#!/usr/bin/env python3
"""Summarize the fixed nine GatedConv single-input runs without loading weights.

The full server outputs remain untouched.  The optional core archive deliberately
excludes checkpoint tensors, prediction rows, embeddings, and feature caches.
"""
from __future__ import annotations

import argparse
import csv
import hashlib
import html
import io
import json
import math
import os
from pathlib import Path
import statistics
import tarfile
import tempfile
from datetime import datetime, timezone

VERSION = "1.0.0"
SCHEMES = ("raw_only", "context_only", "responses_only")
SEEDS = (42, 2024, 2025)
CLASS_NAMES = ("A", "T", "C", "G", "X", "Y")
METRICS = ("accuracy", "macro_f1", "X_recall", "X_f1", "Y_recall", "Y_f1",
           "canonical_false_positive_rate", "target_recall_micro", "ncb_recall",
           "ncb_precision", "nll", "brier", "ece_15bins")
LOWER = {"canonical_false_positive_rate", "nll", "brier", "ece_15bins"}
LABELS = {"accuracy": "准确率", "macro_f1": "六类 Macro-F1", "X_recall": "X 召回率",
          "X_f1": "X F1", "Y_recall": "Y 召回率", "Y_f1": "Y F1",
          "canonical_false_positive_rate": "DNA→X/Y 误检率", "target_recall_micro": "正确 X/Y 身份召回",
          "ncb_recall": "X/Y 合并检测召回", "ncb_precision": "X/Y 合并检测精确率",
          "nll": "NLL", "brier": "Brier", "ece_15bins": "ECE（15箱）"}
ARCHITECTURE = {"model_type": "litemamba", "litemamba_variant": "legacy_gatedconv",
                "readout": "target_attention", "model_dim": 96, "num_blocks": 4,
                "state_dim": 16, "target_index": 3}
COMPARABLE = ("raw_signal_len", "dwell_divisor", "quality_divisor", "epochs", "patience",
              "lr", "batch_size", "accum_steps", "weight_decay", "label_smoothing", "clip_grad", "monitor")


def read_json(path: Path, required=False):
    if not path.is_file():
        if required:
            raise ValueError("缺少文件：" + str(path))
        return None
    try:
        with path.open(encoding="utf-8") as handle:
            value = json.load(handle)
    except (OSError, json.JSONDecodeError) as exc:
        raise ValueError("无法完整读取 JSON：" + str(path)) from exc
    if not isinstance(value, dict):
        raise ValueError("JSON 顶层必须为对象：" + str(path))
    return value


def clean(value):
    if isinstance(value, float) and not math.isfinite(value):
        return None
    if isinstance(value, dict):
        return {str(k): clean(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [clean(v) for v in value]
    return value


def write_json(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + ".tmp")
    with temporary.open("w", encoding="utf-8") as handle:
        json.dump(clean(value), handle, ensure_ascii=False, indent=2, allow_nan=False)
        handle.write("\n")
        handle.flush()
        os.fsync(handle.fileno())
    os.replace(temporary, path)


def write_csv(path, rows, fields):
    path = Path(path)
    temporary = path.with_name(path.name + ".tmp")
    with temporary.open("w", encoding="utf-8-sig", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(clean(rows))
        handle.flush()
        os.fsync(handle.fileno())
    os.replace(temporary, path)


def fingerprint(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return {"sha256": digest.hexdigest(), "bytes": Path(path).stat().st_size}


def option(config, key):
    return config.get(key, config.get("model_kwargs", {}).get(key))


def input_identity(config):
    inputs = config.get("input_fingerprints", {})
    identity = {}
    for split in ("train", "val", "test"):
        current = inputs.get(split, {})
        identity[split] = {}
        for kind in ("feature", "metadata"):
            item = current.get(kind, {})
            if not item.get("sha256") or item.get("size") is None:
                raise ValueError("输入身份缺少 SHA256/size：" + split + "/" + kind)
            identity[split][kind] = {"sha256": item["sha256"], "size": item["size"]}
    return identity


def check_config(config, scheme, seed):
    issues = []
    for key, expected in ARCHITECTURE.items():
        if option(config, key) != expected:
            issues.append(key + " 必须为 " + str(expected))
    if option(config, "feature_scheme") != scheme:
        issues.append("feature_scheme 与计划不符")
    if config.get("seed") != seed:
        issues.append("seed 与计划不符")
    classes = config.get("class_names")
    order = tuple(classes.get(str(i)) for i in range(6)) if isinstance(classes, dict) else tuple(classes or ())
    if order != CLASS_NAMES:
        issues.append("六类顺序不是 A/T/C/G/X/Y")
    try:
        input_identity(config)
    except ValueError as exc:
        issues.append(str(exc))
    return issues


def comparable_config(config, reference):
    issues = []
    if input_identity(config) != input_identity(reference):
        issues.append("训练/验证/测试输入内容身份不同")
    for key in COMPARABLE:
        a, b = option(config, key), option(reference, key)
        if a is None or b is None or a != b:
            issues.append("训练/预处理协议不同或缺失：" + key)
    for name in ("model_benchmark_all.py", "IterDataset.py", "config_6_all_6.py"):
        left = config.get("source_snapshot", {}).get("files_sha256", {}).get(name)
        right = reference.get("source_snapshot", {}).get("files_sha256", {}).get(name)
        if not left or not right or left != right:
            issues.append("模型/数据源码身份不同或缺失：" + name)
    return issues


def metrics_at(run_dir, split, result):
    direct = read_json(run_dir / split / "prediction_metrics.json")
    if direct:
        embedded = result.get("metrics_by_split", {}).get(split)
        if embedded and direct != embedded:
            raise ValueError("导出指标与 results.json 不一致：" + str(run_dir / split))
        return direct, str(run_dir / split / "prediction_metrics.json")
    value = result.get("metrics_by_split", {}).get(split)
    if value:
        return value, str(run_dir / "results.json")
    if split == "val" and result.get("best_val_metrics"):
        return {"raw": result["best_val_metrics"]}, str(run_dir / "results.json")
    return {}, ""


def export_issues(run_dir, config, result):
    issues = []
    expected_hash = result.get('provenance', {}).get('checkpoint_sha256')
    if not expected_hash:
        issues.append('complete结果缺少最佳权重身份')
    for split in ('val', 'test'):
        info = read_json(run_dir / split / 'export_complete.json') or result.get('exports_by_split', {}).get(split)
        if not info:
            issues.append(split + ' 缺少完整导出标识')
            continue
        if info.get('checkpoint_sha256') != expected_hash:
            issues.append(split + ' 导出与最佳权重身份不同')
        identity = info.get('dataset_identity', {})
        if identity.get('input_fingerprints') != input_identity(config)[split]:
            issues.append(split + ' 导出与冻结数据身份不同')
        preprocessing = identity.get('preprocessing', {})
        if preprocessing.get('mask_index') != 3 or preprocessing.get('mask_base') != 'N':
            issues.append(split + ' 导出未证明第4位N预处理')
    return issues


def metric_values(metrics):
    values = {name: metrics.get(name) for name in METRICS}
    cm = metrics.get("confusion_matrix")
    if cm is not None:
        if not isinstance(cm, list) or len(cm) != 6 or any(not isinstance(row, list) or len(row) != 6 for row in cm):
            raise ValueError("混淆矩阵必须为 6×6")
        if any(not isinstance(v, (int, float)) or not math.isfinite(v) or v < 0 or int(v) != v for row in cm for v in row):
            raise ValueError("混淆矩阵必须为非负整数计数")
        support = [sum(row) for row in cm]
        total = sum(support)
        if metrics.get("n", total) != total:
            raise ValueError("混淆矩阵与样本数不一致")
        divide = lambda a, b: a / b if b else None
        for index, name in ((4, "X"), (5, "Y")):
            tp, predicted = cm[index][index], sum(row[index] for row in cm)
            values[name + "_recall"] = divide(tp, support[index])
            values[name + "_f1"] = divide(2 * tp, support[index] + predicted)
        values["canonical_false_positive_rate"] = divide(sum(sum(row[4:6]) for row in cm[:4]), sum(support[:4]))
        values["target_recall_micro"] = divide(cm[4][4] + cm[5][5], sum(support[4:6]))
        detected = sum(sum(row[4:6]) for row in cm[4:6])
        values["ncb_recall"] = divide(detected, sum(support[4:6]))
        values["ncb_precision"] = divide(detected, sum(sum(row[4:6]) for row in cm))
    return {key: float(value) if isinstance(value, (int, float)) and math.isfinite(value) else None for key, value in values.items()}


def rows_for(run_dir, result, scheme, seed, origin, splits=("val", "test")):
    rows = []
    for split in splits:
        versions, source = metrics_at(run_dir, split, result)
        for probability_type in ("raw", "calibrated"):
            if probability_type not in versions:
                continue
            for metric, value in metric_values(versions[probability_type]).items():
                rows.append({"scheme": scheme, "seed": seed, "split": split,
                             "probability_type": probability_type, "metric": metric,
                             "value": value, "origin": origin, "source": source})
    return rows


def baseline_paths(baseline_dir):
    if baseline_dir is None:
        return {}, {}
    baseline_dir = Path(baseline_dir)
    candidates, hashes = {}, {}
    plan = read_json(baseline_dir / "structure_plan.json") or {}
    for item in plan.get("reused_controls", []) + plan.get("tasks", []):
        if item.get("experiment") == "gatedconv_target" and item.get("seed") in SEEDS and item.get("run_dir"):
            seed = item["seed"]
            candidates[seed] = Path(item["run_dir"])
            if item.get("checkpoint_sha256"):
                hashes[seed] = item["checkpoint_sha256"]
    for seed in SEEDS:
        direct = baseline_dir / "full" / "gatedconv_target" / ("seed_" + str(seed))
        if direct.is_dir():
            candidates[seed] = direct
    return candidates, hashes


def summary_rows(rows):
    grouped = {}
    for row in rows:
        key = tuple(row[k] for k in ("scheme", "split", "probability_type", "metric"))
        grouped.setdefault(key, {})[row["seed"]] = row["value"]
    output = []
    for key, seeds in sorted(grouped.items()):
        values = [seeds[seed] for seed in SEEDS if seeds.get(seed) is not None]
        output.append(dict(zip(("scheme", "split", "probability_type", "metric"), key),
                           n_seeds=len(values), expected_seeds=3,
                           mean=statistics.mean(values) if values else None,
                           sample_sd=statistics.stdev(values) if len(values) >= 2 else None,
                           all_three_seeds_available=len(values) == 3,
                           preferred_direction="lower" if key[3] in LOWER else "higher"))
    return output


def paired_rows(rows):
    baseline = {(r["seed"], r["split"], r["probability_type"], r["metric"]): r["value"] for r in rows if r["scheme"] == "full"}
    output = []
    for row in rows:
        if row["scheme"] == "full":
            continue
        key = (row["seed"], row["split"], row["probability_type"], row["metric"])
        reference = baseline.get(key)
        value = row["value"]
        delta = value - reference if value is not None and reference is not None else None
        output.append({**row, "full_value": reference, "delta_single_minus_full": delta,
                       "delta_percentage_points": delta * 100 if delta is not None and row["metric"] not in ("nll", "brier") else None})
    return output


def paired_aggregate(rows):
    groups = {}
    for row in rows:
        key = tuple(row[k] for k in ("scheme", "split", "probability_type", "metric"))
        groups.setdefault(key, {})[row["seed"]] = row["delta_single_minus_full"]
    output = []
    for key, seeds in sorted(groups.items()):
        values = [seeds[s] for s in SEEDS if seeds.get(s) is not None]
        lower = key[3] in LOWER
        output.append(dict(zip(("scheme", "split", "probability_type", "metric"), key),
                           n_seed_pairs=len(values), expected_seed_pairs=3,
                           mean_delta=statistics.mean(values) if values else None,
                           sample_sd_delta=statistics.stdev(values) if len(values) >= 2 else None,
                           single_better_seed_count=sum(v < 0 if lower else v > 0 for v in values),
                           full_better_seed_count=sum(v > 0 if lower else v < 0 for v in values),
                           equal_seed_count=sum(v == 0 for v in values),
                           preferred_direction="lower" if lower else "higher"))
    return output


def report_html(statuses, aggregates, baseline_statuses):
    esc = lambda x: html.escape(str(x))
    def table(headings, rows):
        return '<div class="scroll"><table><thead><tr>' + ''.join('<th>' + esc(h) + '</th>' for h in headings) + '</tr></thead><tbody>' + ''.join('<tr>' + ''.join('<td>' + esc(v) + '</td>' for v in row) + '</tr>' for row in rows) + '</tbody></table></div>'
    def number(x, scale=100):
        return "—" if x is None else f"{x * scale:.4f}"
    complete = sum(row["status"] == "complete" for row in statuses)
    parts = ['<!doctype html><html lang="zh-CN"><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1"><title>GatedConv 单输入三种子实验</title><style>body{font:15px/1.65 system-ui,sans-serif;background:#f2f6f9;color:#253847}main{max-width:1200px;margin:auto;padding:32px;background:white}h1,h2{color:#12455b}.note{padding:14px;background:#edf5f7;border-left:4px solid #087e8b}.scroll{overflow:auto;margin:15px 0}table{border-collapse:collapse;width:100%;font-variant-numeric:tabular-nums}th,td{padding:8px 10px;white-space:nowrap;border-bottom:1px solid #dce5eb;text-align:left}th{background:#15475b;color:white}tr:nth-child(even){background:#f3f7fa}code{background:#edf1f4}</style><main>',
             '<h1>GatedConv 三种单输入 × 三个种子</h1>',
             '<p class="note">正式任务完成：' + str(complete) + '/9。固定种子42、2024、2025；输入组为 raw_only、context_only、responses_only。所有方案从头训练，完整输入 full 只复用为对照。</p>',
             '<p>单输入对照回答“仅用这一类输入能识别到什么程度”；删特征消融回答“去掉一类输入损失多少”，两者不能互相替代。验证集用于选模和温度校准，内部测试集仅在所有方案训练和导出完成后评价。此包尚不包含 Figure 4 外部预测结果，权重可供后续冻结外测。</p>',
             '<h2>全部九项任务状态</h2>', table(['输入','种子','状态','说明'], [(r['scheme'], r['seed'], r['status'], '; '.join(r['issues'])) for r in statuses]),
             '<h2>相同三种子的均值与样本标准差</h2><p>比例指标为%；NLL/Brier为原值。缺失值保留为空；不足三个有效种子的行不能作为完整三种子结论。SD不是置信区间，均值差异不自动代表统计显著。</p>']
    rows = []
    for r in aggregates:
        scale = 1 if r['metric'] in ('nll', 'brier') else 100
        rows.append((r['scheme'], r['split'], r['probability_type'], LABELS.get(r['metric'], r['metric']), '↓' if r['metric'] in LOWER else '↑', number(r['mean'], scale), number(r['sample_sd'], scale), str(r['n_seeds']) + '/3'))
    parts.append(table(['输入','集合','概率','指标','方向','均值','样本SD','有效种子'], rows))
    parts += ['<h2>完整输入对照的可比性</h2>', table(['种子','状态','说明'], [(r['seed'], r['status'], '; '.join(r['issues'])) for r in baseline_statuses]),
              '<p>只有架构、目标读出、类别顺序、数据内容及训练/预处理协议一致的完整输入记录才参与配对比较。内部测试结果若来自独立补评估目录，另核对来源运行、检查点哈希和测试输入身份。</p>',
              '<h2>文件入口</h2><ul><li><code>run_status.csv</code>：九项完整状态。</li><li><code>metrics_per_seed.csv</code>：原始与校准概率的逐种子指标。</li><li><code>mean_sd.csv</code>：各方案均值及样本SD。</li><li><code>paired_single_vs_full.csv</code>、<code>paired_mean_sd.csv</code>：单输入−完整输入的同种子差值。</li><li><code>summary.json</code>：状态、可比性和来源索引。</li></ul><p>完整概率、逐窗口预测、嵌入及检查点仍保留在服务器原目录。轻量结果包只携带配置、历史、指标和分组统计，便于上传分析。</p></main></html>']
    return '\n'.join(parts)


def collect_baselines(baseline_dir, baseline_test_dir, configs, package_sources):
    paths, hashes = baseline_paths(baseline_dir)
    rows, statuses = [], []
    for seed in SEEDS:
        status = {'seed': seed, 'status': 'unavailable', 'issues': []}
        path = paths.get(seed)
        try:
            config = read_json(path / 'resolved_config.json') if path else None
            result = read_json(path / 'results.json') if path else None
        except ValueError as exc:
            status.update(status='invalid', issues=[str(exc)])
            statuses.append(status)
            continue
        if not config or not result:
            status['issues'].append('未找到完整输入训练记录，未做零值填充')
            statuses.append(status)
            continue
        issues = check_config(config, 'full', seed)
        reference = configs.get(seed)
        if reference:
            issues += comparable_config(reference, config)
        else:
            issues.append('该种子没有有效单输入配置用于可比性核对')
        if issues:
            status.update(status='incompatible', issues=issues)
            statuses.append(status)
            continue
        status.update(status='comparable', run_dir=str(path))
        rows += rows_for(path, result, 'full', seed, 'existing_full_baseline', ('val', 'test'))
        package_sources.append(('baseline/seed_' + str(seed), path))
        if not any(r['seed'] == seed and r['split'] == 'test' and r['scheme'] == 'full' for r in rows):
            external = Path(baseline_test_dir) / 'runs/gatedconv_target' / ('seed_' + str(seed)) if baseline_test_dir else None
            completion = read_json(external / 'run_complete.json') if external else None
            if completion:
                tests = completion.get('dataset_identity', {}).get('input_fingerprints')
                expected = input_identity(config)['test']
                known_hash = hashes.get(seed) or result.get('provenance', {}).get('checkpoint_sha256')
                checks = [completion.get('status') == 'complete',
                          Path(completion.get('source_run_dir', '.')).resolve() == path.resolve(),
                          bool(known_hash) and known_hash == completion.get('checkpoint_sha256'),
                          tests == expected]
                if all(checks):
                    rows += rows_for(external, completion, 'full', seed, 'existing_full_internal_test', ('test',))
                    package_sources.append(('baseline_internal_test/seed_' + str(seed), external))
                else:
                    status['issues'].append('已有补测结果的运行/权重/输入身份不匹配，未并入test比较')
            else:
                status['issues'].append('没有可验证的full内部test，保留val比较；test差值为空')
        statuses.append(status)
    return rows, statuses


def make_archive(output_dir, sources, summary_dir, max_file_bytes=24 * 1024 * 1024, max_total_bytes=240 * 1024 * 1024):
    """Write a bounded, whitelist-only core archive, leaving server files untouched."""
    archive = output_dir / 'gatedconv_singleinput9_results_for_analysis_v1_0_0.tar.gz'
    temporary = archive.with_name(archive.name + '.tmp')
    selected, excluded, seen, total = [], [], set(), 0
    metadata_names = {'results.json', 'resolved_config.json', 'model_config.json', 'queue_requested_config.json',
                      'environment.json', 'data_audit.json', 'calibration.json', 'architecture.txt', 'history.csv', 'metrics.csv',
                      'epoch_metrics.jsonl', 'run_complete.json', 'benchmark_isolated.json', 'benchmark.json',
                      'prediction_metrics.json', 'export_complete.json', 'resource_profile.json'}
    for prefix, directory in sources + [('summary', summary_dir)]:
        if not directory.is_dir():
            continue
        for source in sorted(directory.rglob('*')):
            if not source.is_file() or source.is_symlink():
                continue
            relative = source.relative_to(directory)
            if source.resolve() in seen:
                continue
            seen.add(source.resolve())
            accepted = prefix == 'summary' or source.name in metadata_names or ('analysis' in relative.parts and source.suffix in ('.csv', '.json')) or ('source_snapshot' in relative.parts and source.suffix in ('.py', '.json'))
            if any(part in {'feature_cache','previous_exports','epochs','epoch_predictions','__pycache__'} for part in relative.parts):
                accepted = False
            if source.suffix in ('.pt','.pth','.npy','.npz') or source.name.endswith('predictions.csv.gz'):
                accepted = False
            size = source.stat().st_size
            if not accepted:
                continue
            if size > max_file_bytes or total + size > max_total_bytes:
                excluded.append({'source':str(source),'bytes':size,'reason':'core package size bound; full file retained on server'})
                continue
            destination = 'gatedconv_singleinput9_core/' + prefix + '/' + relative.as_posix()
            selected.append((source, destination, fingerprint(source)))
            total += size
    manifest = {'version':VERSION, 'created_at':datetime.now(timezone.utc).isoformat(),
                'weights_or_prediction_arrays_included':False, 'uncompressed_bytes':total,
                'files':[{'path':dest,'source':str(src),**identity} for src,dest,identity in selected], 'excluded_large_files':excluded}
    payload = json.dumps(manifest,ensure_ascii=False,indent=2,allow_nan=False).encode('utf-8')
    with tarfile.open(temporary,'w:gz') as tar:
        for source,destination,identity in selected:
            if source.stat().st_size != identity['bytes']:
                raise ValueError('打包期间文件大小改变：' + str(source))
            tar.add(source,arcname=destination,recursive=False)
        entry = tarfile.TarInfo('gatedconv_singleinput9_core/package_manifest.json')
        entry.size=len(payload)
        tar.addfile(entry,io.BytesIO(payload))
    os.replace(temporary,archive)
    write_json(output_dir/'core_package_manifest.json',manifest)
    return archive


def summarize(output_dir, baseline_dir=None, baseline_test_dir=None, package=False):
    output_dir = Path(output_dir).expanduser().resolve()
    target = output_dir/'singleinput_summary'
    target.mkdir(parents=True,exist_ok=True)
    statuses, rows, configs, package_sources = [], [], {}, []
    reference_identity = None
    reference_config = None
    for scheme in SCHEMES:
        for seed in SEEDS:
            run_dir = output_dir/scheme/'litemamba'/('seed_'+str(seed))
            item = {'scheme':scheme,'seed':seed,'run_dir':str(run_dir),'status':'missing','issues':[]}
            try:
                config = read_json(run_dir/'resolved_config.json')
                result = read_json(run_dir/'results.json')
            except ValueError as exc:
                item.update(status='invalid',issues=[str(exc)])
                statuses.append(item)
                continue
            if config and result:
                item['issues'] += check_config(config,scheme,seed)
                if not item['issues']:
                    identity = input_identity(config)
                    if reference_identity is None:
                        reference_identity = identity
                        reference_config = config
                    elif identity != reference_identity:
                        item['issues'].append('九项任务之间的数据内容身份不同')
                    elif reference_config:
                        item['issues'] += comparable_config(config, reference_config)
                if result.get('model_type') != 'litemamba' or result.get('seed') != seed or result.get('feature_scheme') != scheme:
                    item['issues'].append('results任务身份与计划不一致')
                item['status'] = result.get('status','unknown')
                if item['status']=='complete' and result.get('test_evaluated') is not True:
                    item['issues'].append('complete结果未确认test_evaluated')
                if item['status']=='complete' and not item['issues']:
                    item['issues'] += export_issues(run_dir, config, result)
                if item['issues']:
                    item['status']='invalid'
                else:
                    configs.setdefault(seed,config)
                    if item['status']=='complete':
                        candidate_rows=rows_for(run_dir,result,scheme,seed,'new_singleinput')
                        needed={(s,p,m) for s in ('val','test') for p in ('raw','calibrated') for m in ('accuracy','macro_f1')}
                        present={(r['split'],r['probability_type'],r['metric']) for r in candidate_rows if r['value'] is not None}
                        if not needed.issubset(present):
                            item['status']='invalid'
                            item['issues'].append('缺少完整val/test原始与校准核心指标')
                        else:
                            rows += candidate_rows
                package_sources.append(('runs/'+scheme+'/seed_'+str(seed),run_dir))
            else:
                item['issues'].append('训练尚未开始或配置/结果尚未完整产出')
            statuses.append(item)
    if baseline_dir and baseline_test_dir is None:
        baseline_test_dir=Path(baseline_dir).parent/'gatedconv_internal_test_v1_0_0'
    baseline_rows, baseline_statuses=collect_baselines(baseline_dir,baseline_test_dir,configs,package_sources)
    rows+=baseline_rows
    aggregated=summary_rows(rows)
    pairs=paired_rows(rows)
    paired_summary=paired_aggregate(pairs)
    done=sum(row['status']=='complete' for row in statuses)
    summary={'version':VERSION,'created_at':datetime.now(timezone.utc).isoformat(),'fixed_seeds':list(SEEDS),
             'feature_schemes':list(SCHEMES),'expected_new_training_runs':9,'completed_new_training_runs':done,
             'status':'complete' if done==9 else 'incomplete','all_formal_tasks_reported':len(statuses)==9,
             'scope':'single-input controls, same architecture/data/protocol; final internal test after training; Figure4 not run here',
             'runs':statuses,'full_baseline':baseline_statuses,'mean_sd':aggregated,'paired_mean_sd':paired_summary,
             'missing_values_policy':'undefined/unavailable retained as null; never replaced with zero',
             'uncertainty':'sample SD of fixed seeds; not a confidence interval or significance claim'}
    write_json(target/'summary.json',summary)
    write_csv(target/'run_status.csv',statuses,['scheme','seed','run_dir','status','issues'])
    write_csv(target/'metrics_per_seed.csv',rows,['scheme','seed','split','probability_type','metric','value','origin','source'])
    write_csv(target/'mean_sd.csv',aggregated,['scheme','split','probability_type','metric','n_seeds','expected_seeds','mean','sample_sd','all_three_seeds_available','preferred_direction'])
    write_csv(target/'paired_single_vs_full.csv',pairs,['scheme','seed','split','probability_type','metric','value','full_value','delta_single_minus_full','delta_percentage_points','origin','source'])
    write_csv(target/'paired_mean_sd.csv',paired_summary,['scheme','split','probability_type','metric','n_seed_pairs','expected_seed_pairs','mean_delta','sample_sd_delta','single_better_seed_count','full_better_seed_count','equal_seed_count','preferred_direction'])
    (target/'analysis_report.html').write_text(report_html(statuses,aggregated,baseline_statuses),encoding='utf-8')
    if package:
        # Top-level queue and launch metadata are small and part of provenance.
        package_sources.append(('queue',output_dir))
        # make_archive whitelists; avoid recursively duplicating all nine runs.
        queue_stage=Path(tempfile.mkdtemp(prefix='.singleinput_core_queue_',dir=output_dir))
        try:
            import shutil
            for path in output_dir.iterdir():
                if path.is_file() and path.suffix in ('.json','.csv') and path.name!='core_package_manifest.json':
                    if path.stat().st_size<24*1024*1024:
                        shutil.copy2(path,queue_stage/path.name)
            package_sources=[item for item in package_sources if item[0]!='queue']+[('queue',queue_stage)]
            archive=make_archive(output_dir,package_sources,target)
        finally:
            shutil.rmtree(queue_stage,ignore_errors=True)
        summary['result_package']=str(archive)
    return summary


def main(argv=None):
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output_dir',required=True)
    parser.add_argument('--baseline_dir')
    parser.add_argument('--baseline_test_dir')
    parser.add_argument('--package',action='store_true')
    parser.add_argument('--allow_incomplete',action='store_true',help='仍完整列出失败/缺失任务，用于中断诊断；不标记全部完成')
    args=parser.parse_args(argv)
    try:
        result=summarize(args.output_dir,args.baseline_dir,args.baseline_test_dir,args.package)
    except (ValueError,OSError) as exc:
        print('[SUMMARY ERROR] '+str(exc))
        return 2
    print('[SUMMARY] '+result['status']+' '+str(result['completed_new_training_runs'])+'/9')
    if result.get('result_package'):
        print('[RESULT PACKAGE] '+result['result_package'])
    return 0 if result['status']=='complete' or args.allow_incomplete else 2


if __name__=='__main__':
    raise SystemExit(main())
