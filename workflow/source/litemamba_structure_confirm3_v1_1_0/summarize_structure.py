#!/usr/bin/env python3
"""严格核验三种子结构确认，汇总九项raw验证结果和同seed差值。"""
from __future__ import annotations

import argparse
from datetime import datetime, timezone
import html
import json
import math
from pathlib import Path
import re
import shutil
import statistics
import tarfile
import tempfile

from legacy_summary import (
    CLASS_NAMES, CONFIG_KEYS, fingerprint_content, metrics_from_confusion,
    read_history, read_json, save_json, sha256, write_csv,
)

SEEDS = (42, 2024, 2025)
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
    if plan.get('validation_only') is not True:
        raise ValueError('确认计划必须为validation_only')
    if plan.get('seeds') != list(SEEDS) or plan.get('new_seeds') != list(SEEDS[1:]):
        raise ValueError('必须固定全部seed42/2024/2025，不得筛选种子')
    sources = []
    groups = (('baselines', 'existing_full', {('full', s) for s in SEEDS}),
              ('reused_controls', 'existing_structure_seed42', {(n, 42) for n in ORDER[1:]}),
              ('tasks', 'new_training', {(n, s) for n in ORDER[1:] for s in SEEDS[1:]}))
    for key, origin, expected in groups:
        items = plan.get(key, [])
        identities = [(t.get('experiment'), t.get('seed')) for t in items]
        if len(identities) != len(expected) or set(identities) != expected:
            raise ValueError('冻结任务身份不完整或重复：' + key)
        for item in items:
            source = dict(item, origin=origin)
            name = source['experiment']
            for field, expected_value in zip(DIFFERENT_KEYS, IDENTITIES[name]):
                if field in source and source[field] != expected_value:
                    raise ValueError('结构任务身份与预设不同：' + name)
                source[field] = expected_value
            sources.append(source)
    sources.sort(key=lambda s: (ORDER.index(s['experiment']), SEEDS.index(s['seed'])))
    directories = [Path(s['run_dir']).expanduser().resolve() for s in sources]
    if len(set(directories)) != 9 or any(not d.is_dir() for d in directories):
        raise ValueError('九个运行目录必须存在且互不相同')
    return sources


def validate_frozen_source(source):
    run = Path(source['run_dir']).expanduser().resolve()
    frozen = source.get('file_sha256', {})
    if source['origin'] != 'new_training' and not frozen:
        raise ValueError('复用结果缺少冻结元数据SHA：' + str(run))
    for relative, digest in frozen.items():
        if Path(relative).is_absolute() or '..' in Path(relative).parts:
            raise ValueError('元数据相对路径非法')
        digest_value(digest, relative)
        if sha256(run / relative) != digest:
            raise ValueError('复用结果相对冻结计划发生变化：' + str(run / relative))


def validate_configuration(config, baseline, source, plan):
    name = source['experiment']
    if (config.get('model_type'), config.get('seed'), config.get('feature_scheme')) != ('litemamba', source['seed'], 'full'):
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
    if name == 'full':
        expected_source = plan['source_sha256']
    elif source['origin'] == 'existing_structure_seed42':
        expected_source = plan['prior_patched_source_sha256']
    else:
        expected_source = plan['patched_source_sha256']
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
    if (result.get('model_type'), result.get('seed'), result.get('feature_scheme')) != ('litemamba', source['seed'], 'full'):
        raise ValueError('结构任务results身份不同')
    if tuple(result.get(k) for k in DIFFERENT_KEYS) != IDENTITIES[source['experiment']]:
        raise ValueError('结构任务results结构身份不同')
    for key in ('metrics_by_split', 'exports_by_split'):
        if set(result.get(key, {})) != {'train', 'val'}:
            raise ValueError('结构任务train/val导出缺失或包含其他split：' + key)
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
    if source.get('checkpoint_sha256') is not None and checkpoint_sha != source['checkpoint_sha256']:
        raise ValueError('检查点与冻结来源不同：' + str(run))
    checkpoints = [Path(source['checkpoint'])] if source.get('checkpoint') else [
        run / relative for relative in ('best_checkpoint.pt', 'checkpoints/best_checkpoint.pt', 'checkpoints/best.pt')]
    checkpoint = next((p for p in checkpoints if p.is_file()), None)
    if checkpoint is None or sha256(checkpoint) != checkpoint_sha:
        raise ValueError('实际best checkpoint缺失或SHA不符：' + str(run))
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
    validate_frozen_source(source)
    config = read_json(run / 'queue_requested_config.json')
    validate_configuration(config, baseline, source, plan)
    resolved = read_json(run / 'resolved_config.json')
    for key in CONFIG_KEYS:
        # staged_exports只属于队列参数，其实际对应defer_train_export。
        if key != 'staged_exports' and resolved.get(key) != config[key]:
            raise ValueError('resolved与请求配置不同：' + name + '/' + key)
    if (resolved.get('model_type'), resolved.get('seed'), resolved.get('feature_scheme')) != ('litemamba', source['seed'], 'full'):
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
    cm = raw['confusion_matrix']
    error_counts = dict(total_errors=sum(sum(r) for r in cm)-sum(cm[i][i] for i in range(6)),
        canonical_internal=sum(sum(r[:4]) for r in cm[:4])-sum(cm[i][i] for i in range(4)),
        canonical_to_xy=sum(sum(r[4:]) for r in cm[:4]),
        xy_to_canonical=sum(sum(r[:4]) for r in cm[4:]), xy_swap=cm[4][5]+cm[5][4],
        a_g_swap=cm[0][3]+cm[3][0])
    record = dict(experiment=name, label=LABELS[name], model_type='litemamba', seed=source['seed'], feature_scheme='full',
                  litemamba_variant=source['litemamba_variant'], readout=source['readout'],
                  origin=source['origin'], monitor='val_loss', selected_epoch=int(selected['epoch']),
                  epochs_completed=len(history), val_loss=selected['val']['loss'],
                  val_macro_f1=recomputed.pop('macro_f1'), val_accuracy=recomputed.pop('accuracy'),
                  **recomputed, val_nll=finite(raw['nll'], 'nll', 0),
                  val_brier=finite(raw['brier'], 'brier', 0), val_ece=finite(raw['ece_15bins'], 'ece', 0, 1),
                  checkpoint_sha256=checkpoint_sha, run_dir=str(run), **error_counts)
    for key in METRICS:
        finite(record[key], name + '/' + key)
    class_rows = [dict(experiment=name, seed=source['seed'], **item) for item in classes]
    curves = []
    for epoch in history:
        row = dict(experiment=name, seed=source['seed'], epoch=epoch['epoch'], checkpoint_selected=epoch['epoch'] == selected['epoch'])
        for key in ('lr', 'next_lr', 'monitor', 'monitor_value', 'improved', 'patience_counter'):
            row[key] = epoch.get(key)
        for split in ('train', 'val'):
            for key, value in epoch[split].items():
                if value is None or isinstance(value, (str, int, float, bool)):
                    row[split + '_' + key] = value
        curves.append(row)
    prefix = 'source_metadata/' + name + '/seed_' + str(source['seed']) + '/'
    files = [(run / relative, prefix + relative)
             for relative in METADATA if (run / relative).is_file()]
    if result is not None:
        files.append((run / 'results.json', prefix + 'results.json'))
    calibrated = metrics.get('overall', {}).get('calibrated', {})
    calibration = read_json(run / 'calibration.json')
    calibration_row = dict(experiment=name, seed=source['seed'], fitted_on='val',
        temperature=finite(calibration.get('temperature'), 'temperature', 0),
        raw_nll=record['val_nll'], raw_brier=record['val_brier'], raw_ece=record['val_ece'],
        calibrated_nll=finite(calibrated.get('nll'), 'calibrated/nll', 0),
        calibrated_brier=finite(calibrated.get('brier'), 'calibrated/brier', 0),
        calibrated_ece=finite(calibrated.get('ece_15bins'), 'calibrated/ece', 0, 1),
        interpretation='temperature fitted and evaluated on the same validation set; descriptive fit, not independent calibration evaluation')
    return record, class_rows, curves, tuple(c['support'] for c in classes), files, calibration_row


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


def aggregate_rows(rows):
    records = []
    for name in ORDER:
        group = [r for r in rows if r['experiment'] == name]
        if sorted(r['seed'] for r in group) != sorted(SEEDS):
            raise ValueError('汇总必须包含每个结构的全部三个种子')
        record = dict(experiment=name, label=LABELS[name], n_seeds=3,
                      seeds=','.join(map(str, SEEDS)), validation_windows_per_seed=group[0]['n'])
        for metric in METRICS:
            values = [r[metric] for r in group]
            record[metric + '_mean'] = statistics.mean(values)
            record[metric + '_sd'] = statistics.stdev(values)
            record[metric + '_min'] = min(values)
            record[metric + '_max'] = max(values)
        records.append(record)
    return records


def paired_comparison(rows):
    lookup = {(r['experiment'], r['seed']): r for r in rows}
    differences = []
    lower_better = {'canonical_false_positive_rate', 'xy_swap_rate', 'val_nll', 'val_brier', 'val_ece'}
    for name in ORDER[1:]:
        for seed in SEEDS:
            reference, current = lookup['full', seed], lookup[name, seed]
            row = dict(experiment=name, seed=seed, reference='full', direction='control minus same-seed full')
            for metric in METRICS:
                suffix, factor = ('', 1) if metric in ('val_nll', 'val_brier') else ('_pp', 100)
                row['delta_' + metric + suffix] = factor*(current[metric]-reference[metric])
            differences.append(row)
    aggregate = []
    for name in ORDER[1:]:
        group = [r for r in differences if r['experiment'] == name]
        for metric in METRICS:
            suffix, factor = ('', 1) if metric in ('val_nll', 'val_brier') else ('_pp', 100)
            values = [r['delta_' + metric + suffix] for r in group]
            lower = metric in lower_better
            better = sum(v < 0 if lower else v > 0 for v in values)
            equal = sum(v == 0 for v in values)
            aggregate.append(dict(experiment=name, metric=metric, n_pairs=3,
                difference_unit='absolute metric units' if factor == 1 else 'percentage points',
                preferred_direction='lower' if lower else 'higher',
                mean_difference=statistics.mean(values), sd_difference=statistics.stdev(values),
                min_difference=min(values), max_difference=max(values),
                n_better=better, n_equal=equal, n_worse=3-better-equal))
    return differences, aggregate


def per_class_summaries(classes):
    lookup = {(r['experiment'], r['seed'], r['class_name']): r for r in classes}
    aggregates, pairs, pair_aggregates = [], [], []
    for name in ORDER:
        for base in CLASS_NAMES:
            group = [lookup[name, seed, base] for seed in SEEDS]
            row = dict(experiment=name, class_name=base, n_seeds=3,
                       support_per_seed=group[0]['support'])
            for metric in ('precision', 'recall', 'f1'):
                values = [r[metric] for r in group]
                row[metric+'_mean'] = statistics.mean(values)
                row[metric+'_sd'] = statistics.stdev(values)
            aggregates.append(row)
            if name == 'full':
                continue
            for seed in SEEDS:
                pair = dict(experiment=name, class_name=base, seed=seed)
                for metric in ('precision', 'recall', 'f1'):
                    pair['delta_'+metric+'_pp'] = 100*(lookup[name,seed,base][metric]-lookup['full',seed,base][metric])
                pairs.append(pair)
            for metric in ('precision', 'recall', 'f1'):
                values = [100*(lookup[name,seed,base][metric]-lookup['full',seed,base][metric]) for seed in SEEDS]
                pair_aggregates.append(dict(experiment=name, class_name=base, metric=metric,
                    n_pairs=3, mean_difference_pp=statistics.mean(values), sd_difference_pp=statistics.stdev(values),
                    min_difference_pp=min(values), max_difference_pp=max(values)))
    return aggregates, pairs, pair_aggregates


def comparison_svg(rows):
    """仅用标准库画出各seed原值；不把窗口数当独立重复数。"""
    values = [100*r['val_macro_f1'] for r in rows]
    minimum, maximum = min(values), max(values)
    padding = max(.12, (maximum-minimum)*.12)
    low, high = minimum-padding, maximum+padding
    def x(value):
        return 285+600*(value-low)/(high-low)
    parts = ['<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 1040 315" role="img" aria-label="三种子验证Macro-F1">',
             '<rect width="1040" height="315" fill="white"/>',
             '<text x="30" y="28" font-size="18" fill="#263e42">Validation macro-F1: every seed shown</text>']
    for step in range(6):
        v = low+(high-low)*step/5
        px = x(v)
        parts.append(f'<path d="M {px:.2f} 42 V 238" stroke="#e3eae5"/>')
        parts.append(f'<text x="{px:.2f}" y="263" text-anchor="middle" font-size="13">{v:.3f}</text>')
    colors = {42:'#205f50',2024:'#5989a3',2025:'#9a515c'}
    english = {'full':'SSM + target (full)','gatedconv_target':'Gated conv + target','ssm_global_only':'SSM + global only'}
    for i, name in enumerate(ORDER):
        y = 77+67*i
        group = [r for r in rows if r['experiment'] == name]
        parts.append(f'<text x="268" y="{y+5}" text-anchor="end" font-size="15">{html.escape(english[name])}</text>')
        for j, r in enumerate(group):
            px, py = x(100*r['val_macro_f1']), y+(j-1)*12
            parts.append(f'<circle cx="{px:.2f}" cy="{py}" r="5" fill="{colors[r["seed"]]}"><title>seed{r["seed"]}: {100*r["val_macro_f1"]:.5f}%</title></circle>')
        mean = 100*statistics.mean(r['val_macro_f1'] for r in group)
        parts.append(f'<path d="M {x(mean):.2f} {y-20} V {y+20}" stroke="#263e42" stroke-width="2"/>')
        parts.append(f'<text x="905" y="{y+4}" font-size="13">mean {mean:.4f}%</text>')
    parts.append('<text x="585" y="290" text-anchor="middle" font-size="13">Macro-F1 (%) - horizontal axis truncated to show small differences</text>')
    for i, seed in enumerate(SEEDS):
        px = 320+i*170
        parts.append(f'<circle cx="{px}" cy="308" r="4" fill="{colors[seed]}"/><text x="{px+11}" y="312" font-size="12">seed{seed}</text>')
    parts.append('<text x="840" y="312" font-size="12">vertical bar: mean</text></svg>')
    return ''.join(parts)


def write_report(path, rows, aggregate, paired):
    table = []
    for r in rows:
        values = [r['label'], str(r['seed']), r['origin'], str(r['selected_epoch']), f"{100*r['val_macro_f1']:.4f}",
                  f"{r['delta_val_macro_f1_pp']:+.4f}", f"{100*r['canonical_false_positive_rate']:.4f}",
                  str(r.get('total_parameters')) if r.get('total_parameters') is not None else '未记录']
        table.append('<tr>' + ''.join('<td>' + html.escape(v) + '</td>' for v in values) + '</tr>')
    overview = []
    for r in aggregate:
        delta = next((p for p in paired if p['experiment']==r['experiment'] and p['metric']=='val_macro_f1'), None)
        values = [r['label'], '3', f"{100*r['val_macro_f1_mean']:.4f} ± {100*r['val_macro_f1_sd']:.4f}",
            '—' if delta is None else f"{delta['mean_difference']:+.4f} ± {delta['sd_difference']:.4f}",
            '—' if delta is None else f"{delta['n_better']}/3", f"{100*r['canonical_false_positive_rate_mean']:.4f} ± {100*r['canonical_false_positive_rate_sd']:.4f}"]
        overview.append('<tr>'+''.join('<td>'+html.escape(v)+'</td>' for v in values)+'</tr>')
    document = '''<!doctype html><html lang="zh-CN"><meta charset="utf-8"><meta name="viewport" content="width=device-width, initial-scale=1"><title>LiteMamba三种子结构确认</title><style>body{font:16px/1.8 system-ui,'Microsoft YaHei',sans-serif;max-width:1280px;margin:35px auto;padding:0 24px;color:#263e42}h1{font-size:28px}h2{color:#205f50;font-size:22px}table{width:100%;border-collapse:collapse;font-variant-numeric:tabular-nums;font-size:14px}td,th{padding:10px;border-bottom:1px solid #dce5e0;text-align:right}td:first-child,th:first-child{text-align:left}th{background:#e8f1ec}.note{background:#f3f6f2;padding:14px}.scroll{overflow-x:auto}img{width:100%;height:auto}</style><h1>LiteMamba：三种子结构确认</h1><p>固定seed42、2024、2025。复用3项full和2项seed42结构结果，仅新增4项训练；保留full输入、同一数据划分及训练参数，按val_loss选择checkpoint。主比较使用该checkpoint的raw验证指标。</p><h2>三个种子的完整汇总</h2><div class="scroll"><table><tr><th>结构方案</th><th>种子数</th><th>Macro-F1 %，均值±SD</th><th>同seed差值pp，均值±SD</th><th>F1优于full种子数</th><th>天然误报 %，均值±SD</th></tr>AGGREGATE</table></div><img src="structure_macro_f1.svg" alt="每个种子及均值的验证Macro-F1"><p>SD为3个训练种子的样本标准差（ddof=1）；同seed差值为对照减相同种子的full。天然误报率指本验证集A/T/C/G窗口预测为X/Y的比例，不是Figure4外部DNA误报率。三次训练共用同一验证划分，不能视为三次独立数据集验证。</p><h2>逐seed原值：不筛选结果</h2><div class="scroll"><table><tr><th>结构方案</th><th>seed</th><th>来源</th><th>最佳epoch</th><th>Macro-F1 %</th><th>同seed差值pp</th><th>天然误报 %</th><th>名义参数</th></tr>RUNS</table></div><h2>解释与论文边界</h2><p class="note">本报告保留全部3个种子；没有把验证窗口数作为独立重复数，没有计算样本级P值、置信区间或自动宣称统计显著性。方向一致和均值差异是描述性证据，不自动证明跨数据集稳定优势、其他模型排名或新颖性。</p><p>门控卷积整体替换selective SSM，参数与运算不同，并非严格等参数的单一递推项删除。若门控卷积更好，必须如实报告；该版本不包含SSM，不能将其解释为Mamba机制创新。global-only保留全局attention与第四位参与上下文计算，只将第四位直接分支置零；12,288个第一层分类器权重没有数据梯度，名义参数相同不等于有效容量相同。</p><p>Raw与温度校准后结果分别保存在structure_calibration.csv。温度在同一验证集拟合，校准后同验证集指标仅描述拟合效果，不是独立校准评估；主结构比较仍使用raw结果。F1、天然误报、正确X/Y身份召回与校准指标可能存在取舍，应同时报告，不能只挑一个有利指标。</p><p>本汇总不读取已有full的test指标，也不新增test或Figure4推理；原引擎仍读取test输入做固定缓存、内容指纹和read隔离审计。已观察过的外部数据不能重新称为未见测试集。</p><p>完整学习曲线、各类结果、错误分解、同seed差值、参数统计与冻结来源均保存在同目录。原始权重、概率、嵌入和epoch logits保留在各run_dir，不复制、不删除；复用文件不会写回。三种子确认用于决定是否保留或简化骨干，不是持续调参直到获得有利结果。</p><h2>主要文件</h2><p>structure_runs.csv：9项原值；structure_aggregate.csv：每结构均值与样本SD；structure_paired_differences.csv和structure_paired_aggregate.csv：6项同seed差值及汇总；structure_per_class*.csv：逐类原值、聚合及差值；structure_learning_curves.csv：完整训练记录；structure_errors.csv：误差分解；structure_calibration.csv：raw及同val温度校准；structure_comparison.json：完整来源和审计边界。</p></html>'''
    path.write_text(document.replace('AGGREGATE',''.join(overview)).replace('RUNS', ''.join(table)), encoding='utf-8')


def summarize(plan: dict, output_dir: Path) -> dict:
    output = Path(output_dir).expanduser().resolve()
    output.mkdir(parents=True, exist_ok=True)
    sources = plan_sources(plan)
    equivalence_report = validate_equivalence_report(output, plan)
    baseline = read_json(Path(plan['baseline_run']).expanduser().resolve() / 'queue_requested_config.json')
    rows, classes, curves, files, calibration = [], [], [], [], []
    support = None
    for source in sources:
        row, class_rows, history, current_support, source_files, calibrated = read_run(source, baseline, plan)
        if support is not None and support != current_support:
            raise ValueError('九个运行的验证类别数量不同')
        support = current_support
        rows.append(row)
        classes.extend(class_rows)
        curves.extend(history)
        files.extend(source_files)
        calibration.append(calibrated)
    references = {r['seed']:r for r in rows if r['experiment']=='full'}
    for row in rows:
        reference = references[row['seed']]
        for metric in METRICS:
            suffix = '' if metric in ('val_nll', 'val_brier') else '_pp'
            factor = 1 if metric in ('val_nll', 'val_brier') else 100
            row['delta_' + metric + suffix] = factor*(row[metric]-reference[metric])
        for metric in ('total_errors','canonical_internal','canonical_to_xy','xy_to_canonical','xy_swap','a_g_swap'):
            row['delta_' + metric] = row[metric]-reference[metric]
    parameters = parameter_records(equivalence_report, [next(r for r in rows if r['experiment']==name) for name in ORDER])
    counts = {r['experiment']:r for r in parameters}
    for row in rows:
        count = counts[row['experiment']]
        row.update({k: v for k, v in count.items() if k not in ('experiment', 'source')})
    aggregate = aggregate_rows(rows)
    differences, paired = paired_comparison(rows)
    class_aggregate, class_pairs, class_paired_aggregate = per_class_summaries(classes)
    previous = Path(plan['previous_structure_dir'])
    for relative, digest in plan.get('previous_structure_file_sha256', {}).items():
        if Path(relative).is_absolute() or '..' in Path(relative).parts:
            raise ValueError('旧结构来源相对路径非法')
        if sha256(previous / relative) != digest:
            raise ValueError('旧结构计划/等价报告/汇总发生变化：' + relative)
        files.append((previous / relative,'previous_structure_metadata/'+relative))
    with tempfile.TemporaryDirectory(prefix='.structure_summary_', dir=output) as temp:
        stage = Path(temp) / 'structure_summary'
        stage.mkdir()
        write_csv(stage / 'structure_runs.csv', rows)
        write_csv(stage / 'structure_per_class.csv', classes)
        write_csv(stage / 'structure_learning_curves.csv', curves)
        write_csv(stage / 'structure_parameters.csv', parameters)
        write_csv(stage / 'structure_aggregate.csv', aggregate)
        write_csv(stage / 'structure_paired_differences.csv', differences)
        write_csv(stage / 'structure_paired_aggregate.csv', paired)
        write_csv(stage / 'structure_per_class_aggregate.csv', class_aggregate)
        write_csv(stage / 'structure_per_class_paired_differences.csv', class_pairs)
        write_csv(stage / 'structure_per_class_paired_aggregate.csv', class_paired_aggregate)
        write_csv(stage / 'structure_calibration.csv', calibration)
        error_keys = ('total_errors','canonical_internal','canonical_to_xy','xy_to_canonical','xy_swap','a_g_swap')
        write_csv(stage / 'structure_errors.csv', [dict(experiment=r['experiment'],seed=r['seed'],
            **{key:r[key] for metric in error_keys for key in (metric,'delta_'+metric)}) for r in rows])
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
        for source in sources:
            validate_frozen_source(source)
        comparison = dict(status='complete', version='1.1.0', created_at=datetime.now(timezone.utc).isoformat(),
            seeds=list(SEEDS), task_count=9, new_training_count=4, reused_full_count=3, reused_structure_count=2,
            checkpoint_selection='val_loss', metric_scope='raw validation at loss-selected checkpoint',
            difference_direction='structure control minus full; percentage points for probability metrics',
            test_inference=False, full_test_metrics_read=False, statistical_significance_test_performed=False,
            all_configuration_and_input_checks_passed=True, per_run=rows, aggregate=aggregate,
            paired_differences=differences, paired_aggregate=paired, parameter_records=parameters,
            original_source_sha256=plan['source_sha256'], patched_source_sha256=plan['patched_source_sha256'],
            input_fingerprints=fingerprint_content(plan['expected_input_fingerprints']), source_files=evidence,
            prior_patched_source_sha256=plan['prior_patched_source_sha256'],
            uncertainty_unit='sample standard deviation across3 fixed training seeds on one fixed validation split',
            validation_predictions_reinferred=False, full_weights_reloaded_by_summary=False,
            limitations=['Three fixed training seeds on one fixed validation split; not independent data replications.',
                         'Gated convolution and selective SSM differ in capacity and computation.',
                         'Zeroing the direct target branch preserves nominal classifier width but changes effective input.',
                         'No model-family ranking, external Figure4 improvement, or statistical significance follows.'])
        save_json(stage / 'structure_comparison.json', comparison)
        (stage / 'structure_macro_f1.svg').write_text(comparison_svg(rows),encoding='utf-8')
        write_report(stage / 'structure_report.html', rows, aggregate, paired)
        (stage / 'README.txt').write_text('打开structure_report.html查看；固定全部seed42/2024/2025，9项验证结果（5旧+4新）。\n差值均为对照减相同seed的full。SD为3个训练种子的样本标准差，非窗口级不确定性；不做显著性检验。\n权重、概率、嵌入、epoch logits等大文件保留在原run_dir中。\n', encoding='utf-8')
        package = output / 'structure_confirm3_results_for_analysis_v1_1_0.tar.gz'
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
