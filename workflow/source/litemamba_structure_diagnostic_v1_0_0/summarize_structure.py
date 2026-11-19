#!/usr/bin/env python3
"""核验两项固定seed42结构诊断，并与已有full的raw验证结果比较。"""
from __future__ import annotations

import argparse
from datetime import datetime, timezone
import html
import json
import math
from pathlib import Path
import re
import shutil
import tarfile
import tempfile

from legacy_summary import (
    CLASS_NAMES, CONFIG_KEYS, fingerprint_content, metrics_from_confusion,
    read_history, read_json, save_json, sha256, write_csv,
)

SEED = 42
ORDER = ('full', 'gatedconv_target', 'ssm_global_only')
IDENTITIES = {
    'full': ('selective_ssm', 'target_attention'),
    'gatedconv_target': ('legacy_gatedconv', 'target_attention'),
    'ssm_global_only': ('selective_ssm', 'global_attention_only'),
}
LABELS = {'full': '已有full', 'gatedconv_target': '门控卷积 + 第四位读出',
          'ssm_global_only': 'Selective SSM + 仅全局attention读出'}
DIFFERENT_KEYS = ('litemamba_variant', 'readout')
SOURCE_FILES = ('train_all.py', 'model_benchmark_all.py', 'config_6_all_6.py', 'IterDataset.py',
                'analyze_results.py', 'run_all_4_models.py', 'resource_scheduler.py', 'cuda_device_probe.py')
METRICS = ('val_macro_f1', 'val_accuracy', 'balanced_accuracy', 'canonical_false_positive_rate',
           'any_ncb_precision', 'any_ncb_recall', 'correct_xy_identity_recall', 'xy_swap_rate',
           'val_nll', 'val_brier', 'val_ece')
METADATA = ('queue_requested_config.json', 'resolved_config.json', 'model_config.json',
            'environment.json', 'data_audit.json', 'calibration.json', 'metrics.csv',
            'epoch_metrics.jsonl', 'val/analysis/metrics.json')


def finite(value, name, lower=None, upper=None):
    if isinstance(value, bool) or not isinstance(value, (int, float)) or not math.isfinite(value):
        raise ValueError('数值无效：' + name)
    if (lower is not None and value < lower) or (upper is not None and value > upper):
        raise ValueError('数值超出范围：' + name)
    return float(value)


def digest_value(value, name):
    if not isinstance(value, str) or re.fullmatch('[0-9a-f]{64}', value) is None:
        raise ValueError('缺少有效SHA256：' + name)
    return value


def plan_sources(plan):
    tasks = plan.get('tasks', [])
    if len(tasks) != 2 or {t.get('experiment') for t in tasks} != set(ORDER[1:]):
        raise ValueError('任务必须恰为gatedconv_target和ssm_global_only各一项')
    sources = [dict(experiment='full', run_dir=plan['baseline_run'], origin='existing_full',
                    litemamba_variant='selective_ssm', readout='target_attention')]
    for name in ORDER[1:]:
        task = next(t for t in tasks if t['experiment'] == name)
        if tuple(task.get(k) for k in DIFFERENT_KEYS) != IDENTITIES[name]:
            raise ValueError('结构任务身份与预设不同：' + name)
        if task.get('seed', SEED) != SEED:
            raise ValueError('结构任务种子必须固定为42')
        sources.append(dict(experiment=name, run_dir=task['run_dir'], origin='new_training',
                            litemamba_variant=task['litemamba_variant'], readout=task['readout']))
    directories = [Path(s['run_dir']).expanduser().resolve() for s in sources]
    if len(set(directories)) != 3 or any(not d.is_dir() for d in directories):
        raise ValueError('三个运行目录必须存在且互不相同')
    return sources


def validate_configuration(config, baseline, source, plan):
    name = source['experiment']
    if (config.get('model_type'), config.get('seed'), config.get('feature_scheme')) != ('litemamba', SEED, 'full'):
        raise ValueError('模型/种子/输入方案不同：' + name)
    if tuple(config.get(k) for k in DIFFERENT_KEYS) != IDENTITIES[name]:
        raise ValueError('结构配置与任务身份不同：' + name)
    if name != 'full' and config.get('structure_experiment') != name:
        raise ValueError('队列structure_experiment与任务身份不同：' + name)
    for key in CONFIG_KEYS:
        if key in DIFFERENT_KEYS:
            continue
        if key not in config or key not in baseline or config[key] != baseline[key]:
            raise ValueError('结构之外的训练配置发生变化：' + name + '/' + key)
        if key in plan.get('scientific_configuration', {}) and config[key] != plan['scientific_configuration'][key]:
            raise ValueError('训练配置不符合冻结计划：' + name + '/' + key)
    if config.get('monitor') != 'val_loss' or config.get('external_manifest'):
        raise ValueError('必须保留val_loss选模且不使用external_manifest')
    expected_source = plan['source_sha256'] if name == 'full' else plan['patched_source_sha256']
    if set(expected_source) != set(SOURCE_FILES) or config.get('source_sha256') != expected_source:
        raise ValueError('源码SHA与计划不同：' + name)
    for filename, value in expected_source.items():
        digest_value(value, filename)
    expected_input = fingerprint_content(plan['expected_input_fingerprints'])
    if fingerprint_content(config['input_fingerprints']) != expected_input:
        raise ValueError('输入内容与计划不同：' + name)


def no_test_result(run, source):
    for relative in ('test_started.json', 'test_evaluated.json'):
        if (run / relative).exists():
            raise ValueError('新结构任务出现test推理标记：' + str(run))
    test_dir = run / 'test'
    if test_dir.is_dir() and any(p.is_file() for p in test_dir.rglob('*')):
        raise ValueError('新结构任务出现test产物：' + str(run))
    result = read_json(run / 'results.json')
    if result.get('status') != 'train_complete' or result.get('test_evaluated') is not False:
        raise ValueError('结构任务尚未完成train/val或已经评价test：' + str(run))
    if (result.get('model_type'), result.get('seed'), result.get('feature_scheme')) != ('litemamba', SEED, 'full'):
        raise ValueError('结构任务results身份不同')
    if tuple(result.get(k) for k in DIFFERENT_KEYS) != IDENTITIES[source['experiment']]:
        raise ValueError('结构任务results结构身份不同')
    for key in ('metrics_by_split', 'exports_by_split'):
        if set(result.get(key, {})) - {'train', 'val'}:
            raise ValueError('结构任务出现train/val之外的结果：' + key)
    return result


def validate_provenance(run, source, plan, config, resolved, result):
    name = source['experiment']
    expected_sources = config['source_sha256']
    snapshots = [resolved.get('source_snapshot', {}).get('files_sha256', {})]
    if result is not None:
        snapshots.append(result.get('provenance', {}).get('source_snapshot', {}).get('files_sha256', {}))
        if fingerprint_content(result['provenance']['input_fingerprints']) != fingerprint_content(plan['expected_input_fingerprints']):
            raise ValueError('results输入来源与计划不同：' + name)
    for snapshot in snapshots:
        for filename, expected in expected_sources.items():
            if snapshot.get(filename) != expected:
                raise ValueError('源码快照与请求SHA不同：' + name + '/' + filename)
    calibration = read_json(run / 'calibration.json')
    checkpoint_sha = digest_value(calibration.get('checkpoint_sha256'), name + '/calibration')
    if calibration.get('fitted_on') != 'val':
        raise ValueError('校准不是在验证集拟合：' + name)
    if result is not None:
        if result['provenance'].get('checkpoint_sha256') != checkpoint_sha:
            raise ValueError('results与calibration检查点SHA不同：' + name)
        for split in ('train', 'val'):
            if result.get('exports_by_split', {}).get(split, {}).get('checkpoint_sha256') != checkpoint_sha:
                raise ValueError('导出检查点SHA不同或导出缺失：' + name + '/' + split)
    return checkpoint_sha


def read_run(source, baseline, plan):
    name = source['experiment']
    run = Path(source['run_dir']).expanduser().resolve()
    config = read_json(run / 'queue_requested_config.json')
    validate_configuration(config, baseline, source, plan)
    resolved = read_json(run / 'resolved_config.json')
    for key in CONFIG_KEYS:
        # staged_exports只属于队列参数，其实际对应defer_train_export。
        if key != 'staged_exports' and resolved.get(key) != config[key]:
            raise ValueError('resolved与请求配置不同：' + name + '/' + key)
    if (resolved.get('model_type'), resolved.get('seed'), resolved.get('feature_scheme')) != ('litemamba', SEED, 'full'):
        raise ValueError('resolved模型身份不同：' + name)
    if resolved.get('external_manifest'):
        raise ValueError('resolved不允许external_manifest')
    if fingerprint_content(resolved['input_fingerprints']) != fingerprint_content(plan['expected_input_fingerprints']):
        raise ValueError('resolved输入与计划不同：' + name)
    model = read_json(run / 'model_config.json')
    if tuple(model.get(k) for k in DIFFERENT_KEYS) != IDENTITIES[name]:
        raise ValueError('model_config结构身份不同：' + name)
    if model.get('class_names') != list(CLASS_NAMES) or model.get('target_index') != 3:
        raise ValueError('类别顺序或第四位目标索引不同：' + name)
    base_model = read_json(Path(plan['baseline_run']).expanduser() / 'model_config.json')
    for key in ('num_classes', 'seq_len', 'raw_signal_len', 'model_dim', 'target_index',
                'dropout', 'classifier_dropout', 'feature_scheme', 'num_blocks', 'state_dim'):
        if key not in model or model[key] != base_model.get(key):
            raise ValueError('预设结构之外的模型参数不同：' + name + '/' + key)
    result = None if name == 'full' else no_test_result(run, source)
    checkpoint_sha = validate_provenance(run, source, plan, config, resolved, result)
    history, selected = read_history(run, 'val_loss')
    for epoch in history:
        for split in ('train', 'val'):
            finite(epoch[split]['loss'], 'history/' + split + '/loss', 0)
            for key in ('accuracy', 'macro_f1'):
                finite(epoch[split][key], 'history/' + split + '/' + key, 0, 1)
    if result is not None and result.get('best_epoch') != selected['epoch']:
        raise ValueError('best_epoch不符合val_loss选模')
    metrics = read_json(run / 'val/analysis/metrics.json')
    if metrics.get('class_definition') != {str(i): name for i, name in enumerate(CLASS_NAMES)}:
        raise ValueError('验证指标的六类定义不同')
    raw = metrics['overall']['raw']
    recomputed, classes = metrics_from_confusion(raw['confusion_matrix'])
    for key in ('accuracy', 'macro_f1', 'balanced_accuracy'):
        value = finite(raw[key], 'raw/' + key, 0, 1)
        if abs(value - recomputed[key]) > 1e-9:
            raise ValueError('raw验证指标与混淆矩阵不一致：' + name + '/' + key)
    if type(raw.get('n')) is not int or raw['n'] != recomputed['n']:
        raise ValueError('验证窗口数量不一致')
    if abs(selected['val']['macro_f1'] - recomputed['macro_f1']) > 1e-9:
        raise ValueError('导出不是loss选中检查点的raw指标')
    if selected['val'].get('confusion_matrix') is not None and selected['val']['confusion_matrix'] != raw['confusion_matrix']:
        raise ValueError('导出混淆矩阵与loss选中检查点不同')
    record = dict(experiment=name, label=LABELS[name], model_type='litemamba', seed=SEED, feature_scheme='full',
                  litemamba_variant=source['litemamba_variant'], readout=source['readout'],
                  origin=source['origin'], monitor='val_loss', selected_epoch=int(selected['epoch']),
                  epochs_completed=len(history), val_loss=selected['val']['loss'],
                  val_macro_f1=recomputed.pop('macro_f1'), val_accuracy=recomputed.pop('accuracy'),
                  **recomputed, val_nll=finite(raw['nll'], 'nll', 0),
                  val_brier=finite(raw['brier'], 'brier', 0), val_ece=finite(raw['ece_15bins'], 'ece', 0, 1),
                  checkpoint_sha256=checkpoint_sha, run_dir=str(run))
    for key in METRICS:
        finite(record[key], name + '/' + key)
    class_rows = [dict(experiment=name, seed=SEED, **item) for item in classes]
    curves = []
    for epoch in history:
        row = dict(experiment=name, seed=SEED, epoch=epoch['epoch'], checkpoint_selected=epoch['epoch'] == selected['epoch'])
        for key in ('lr', 'next_lr', 'monitor', 'monitor_value', 'improved', 'patience_counter'):
            row[key] = epoch.get(key)
        for split in ('train', 'val'):
            for key, value in epoch[split].items():
                if value is None or isinstance(value, (str, int, float, bool)):
                    row[split + '_' + key] = value
        curves.append(row)
    files = [(run / relative, 'source_metadata/' + name + '/' + relative)
             for relative in METADATA if (run / relative).is_file()]
    if result is not None:
        files.append((run / 'results.json', 'source_metadata/' + name + '/results.json'))
    return record, class_rows, curves, tuple(c['support'] for c in classes), files


def validate_equivalence_report(output, plan):
    """正式汇总复用与当前冻结计划完全对应的等价性报告。"""
    path = output / 'equivalence_report.json'
    if not path.is_file():
        raise ValueError('缺少equivalence_report.json，正式汇总需要已通过的full等价性预检')
    report = read_json(path)
    if report.get('status') != 'passed':
        raise ValueError('full等价性报告状态不是passed')
    if report.get('original_source_sha256') != plan['source_sha256'] or report.get('patched_source_sha256') != plan['patched_source_sha256']:
        raise ValueError('等价性报告的原/新源码SHA与计划不同')
    if 'baseline_checkpoint_sha256' in plan and report.get('baseline_checkpoint_sha256') != plan['baseline_checkpoint_sha256']:
        raise ValueError('等价性报告的baseline检查点SHA与计划不同')
    return report


def parameter_records(report, rows):
    """只使用已记录计数；未知值留空，不由结构名称猜测。"""
    counts = report.get('parameter_counts', report.get('models', {}))
    if isinstance(counts, list):
        counts = {r.get('experiment', r.get('name')): r for r in counts}
    records = []
    for row in rows:
        name = row['experiment']
        item = counts.get(name, {}) if isinstance(counts, dict) else {}
        if isinstance(item, (int, float)) and not isinstance(item, bool):
            item = {'total_parameters': item}
        record = dict(experiment=name, source='equivalence_report.json' if item else 'unknown',
                      total_parameters=None, trainable_parameters=None,
                      inactive_classifier_parameters=None, classifier_input_width=None,
                      classifier_data_active_input_width=None)
        aliases = {'total_parameters': ('total_parameters', 'total', 'parameters'),
                   'trainable_parameters': ('trainable_parameters', 'trainable'),
                   'inactive_classifier_parameters': ('inactive_classifier_parameters', 'inactive_target_classifier_parameters',
                                                      'classifier_data_inactive_input_weight_count'),
                   'classifier_input_width': ('classifier_input_width',),
                   'classifier_data_active_input_width': ('classifier_data_active_input_width',)}
        for target, keys in aliases.items():
            for key in keys:
                if key in item:
                    value = item[key]
                    if type(value) is not int or value < 0:
                        raise ValueError('参数计数必须为非负整数：' + name + '/' + key)
                    record[target] = value
                    break
        records.append(record)
    return records


def write_report(path, rows):
    table = []
    for r in rows:
        values = [r['label'], str(r['selected_epoch']), f"{100*r['val_macro_f1']:.4f}",
                  f"{r['delta_val_macro_f1_pp']:+.4f}", f"{100*r['canonical_false_positive_rate']:.4f}",
                  str(r.get('total_parameters')) if r.get('total_parameters') is not None else '未记录']
        table.append('<tr>' + ''.join('<td>' + html.escape(v) + '</td>' for v in values) + '</tr>')
    document = '''<!doctype html><html lang="zh-CN"><meta charset="utf-8"><meta name="viewport" content="width=device-width, initial-scale=1"><title>LiteMamba结构诊断</title><style>body{font:16px/1.8 system-ui,'Microsoft YaHei',sans-serif;max-width:1150px;margin:35px auto;padding:0 24px;color:#263e42}table{width:100%;border-collapse:collapse;font-variant-numeric:tabular-nums}td,th{padding:10px;border-bottom:1px solid #dce5e0;text-align:right}td:first-child,th:first-child{text-align:left}th{background:#e8f1ec}.note{background:#f3f6f2;padding:14px}</style><h1>LiteMamba：两项固定seed42的结构诊断</h1><p>full复用已有结果，两项新任务从头训练；全部保留full输入、相同数据划分及科学训练参数。按val_loss选检查点，比较该检查点的raw验证指标。</p><table><tr><th>结构方案</th><th>最佳epoch</th><th>Macro-F1 %</th><th>相对full差值 pp</th><th>天然误报 %</th><th>总参数</th></tr>ROWS</table><p>差值为结构对照−已有full。天然误报率为A/T/C/G验证窗口中预测为X/Y的比例，不是Figure4外部误报率。</p><p class="note">这是固定seed42、固定验证划分的探索性结构诊断，没有多种子标准差、置信区间或显著性P值。结果不能证明跨模型优越性、外部Figure4提升或方法创新性；如果作为正式结构证据，需按预设规则补重复确认，不按结果筛选种子。</p><p>门控卷积版本替换selective SSM块，参数和运算并不完全匹配，应结合参数统计解释。global_attention_only保留全局attention池化，将第四位直接表示分支置零，分类器宽度保持不变；名义参数量相同不代表有效分类器输入完全相同。</p><p>本汇总不读取已有full的test指标，也没有对新任务进行test或Figure4推理。原训练引擎预检仍读取test输入用于固定缓存、内容指纹及read隔离审计。</p><p>逐任务指标、每类指标、完整学习曲线、参数记录、计划和来源元数据均在同目录保存；大权重、概率、embedding和epoch logits保留在各run_dir中，本包不复制、不删除它们。</p></html>'''
    path.write_text(document.replace('ROWS', ''.join(table)), encoding='utf-8')


def summarize(plan: dict, output_dir: Path) -> dict:
    output = Path(output_dir).expanduser().resolve()
    output.mkdir(parents=True, exist_ok=True)
    sources = plan_sources(plan)
    equivalence_report = validate_equivalence_report(output, plan)
    baseline_dir = Path(plan['baseline_run']).expanduser().resolve()
    baseline = read_json(baseline_dir / 'queue_requested_config.json')
    for relative, digest in plan.get('baseline_file_sha256', {}).items():
        if Path(relative).is_absolute() or '..' in Path(relative).parts:
            raise ValueError('基线文件相对路径非法')
        if sha256(baseline_dir / relative) != digest:
            raise ValueError('基线元数据相对冻结计划发生变化：' + relative)
    rows, classes, curves, files = [], [], [], []
    support = None
    for source in sources:
        row, class_rows, history, current_support, source_files = read_run(source, baseline, plan)
        if support is not None and support != current_support:
            raise ValueError('三个运行的验证类别数量不同')
        support = current_support
        rows.append(row)
        classes.extend(class_rows)
        curves.extend(history)
        files.extend(source_files)
    reference = rows[0]
    for row in rows:
        for metric in METRICS:
            suffix = '' if metric in ('val_nll', 'val_brier') else '_pp'
            factor = 1 if metric in ('val_nll', 'val_brier') else 100
            row['delta_' + metric + suffix] = factor*(row[metric]-reference[metric])
    parameters = parameter_records(equivalence_report, rows)
    for row, count in zip(rows, parameters):
        row.update({k: v for k, v in count.items() if k not in ('experiment', 'source')})
    with tempfile.TemporaryDirectory(prefix='.structure_summary_', dir=output) as temp:
        stage = Path(temp) / 'structure_summary'
        stage.mkdir()
        write_csv(stage / 'structure_runs.csv', rows)
        write_csv(stage / 'structure_per_class.csv', classes)
        write_csv(stage / 'structure_learning_curves.csv', curves)
        write_csv(stage / 'structure_parameters.csv', parameters)
        evidence = []
        for source, relative in files:
            if source.stat().st_size > 64*1024*1024:
                raise ValueError('元数据异常过大，拒绝打包：' + str(source))
            target = stage / relative
            target.parent.mkdir(parents=True, exist_ok=True)
            shutil.copyfile(source, target)
            evidence.append(dict(package_path=relative, source_path=str(source), size_bytes=target.stat().st_size, sha256=sha256(target)))
        save_json(stage / 'structure_plan.json', plan)
        for name in ('structure_status.json', 'equivalence_report.json'):
            if (output / name).is_file():
                shutil.copyfile(output / name, stage / name)
        comparison = dict(status='complete', version='1.0.0', created_at=datetime.now(timezone.utc).isoformat(),
            fixed_seed=SEED, task_count=3, new_training_count=2, baseline_reused=True,
            checkpoint_selection='val_loss', metric_scope='raw validation at loss-selected checkpoint',
            difference_direction='structure control minus full; percentage points for probability metrics',
            test_inference=False, full_test_metrics_read=False, statistical_significance_test_performed=False,
            all_configuration_and_input_checks_passed=True, per_run=rows, parameter_records=parameters,
            original_source_sha256=plan['source_sha256'], patched_source_sha256=plan['patched_source_sha256'],
            input_fingerprints=fingerprint_content(plan['expected_input_fingerprints']), source_files=evidence,
            limitations=['Single training seed on one fixed validation split; exploratory structural diagnostic.',
                         'Gated convolution and selective SSM differ in capacity and computation.',
                         'Zeroing the direct target branch preserves nominal classifier width but changes effective input.',
                         'No model-family ranking, external Figure4 improvement, or statistical significance follows.'])
        save_json(stage / 'structure_comparison.json', comparison)
        write_report(stage / 'structure_report.html', rows)
        (stage / 'README.txt').write_text('打开structure_report.html查看；本包只含固定seed42结构诊断的验证指标与来源元数据。\n权重、概率、嵌入、epoch logits等大文件保留在原run_dir中。\n', encoding='utf-8')
        package = output / 'structure_results_for_analysis_v1_0_0.tar.gz'
        temporary_package = package.with_name(package.name + '.tmp')
        with tarfile.open(temporary_package, 'w:gz') as archive:
            archive.add(stage, arcname='structure_summary')
        destination = output / 'structure_summary'
        previous = Path(temp) / 'previous_summary'
        if destination.exists():
            destination.rename(previous)
        try:
            stage.rename(destination)
        except BaseException:
            if previous.exists():
                previous.rename(destination)
            raise
        temporary_package.replace(package)
    print('[SUMMARY] ' + str(output / 'structure_summary/structure_runs.csv'), flush=True)
    print('[RESULT PACKAGE] ' + str(package), flush=True)
    return comparison


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--plan', required=True, type=Path)
    parser.add_argument('--output_dir', type=Path)
    args = parser.parse_args()
    plan = read_json(args.plan)
    summarize(plan, args.output_dir or Path(plan['output_dir']))


if __name__ == '__main__':
    main()
