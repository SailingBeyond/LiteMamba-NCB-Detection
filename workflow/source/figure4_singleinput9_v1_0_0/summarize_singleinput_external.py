#!/usr/bin/env python3
"""Summarize all frozen Figure 4 single-input runs; never load or fit a model."""
from __future__ import annotations

import argparse
from datetime import datetime, timezone
import hashlib
import html
import io
import json
import math
import os
from pathlib import Path
import tarfile
import tempfile

import numpy as np
import pandas as pd

SEEDS = (42, 2024, 2025)
SCHEMES = ('full', 'raw_only', 'context_only', 'responses_only')
MODEL = 'gatedconv_target'
CLASSES = ('A', 'T', 'C', 'G', 'X', 'Y')
LABELS = {'full': 'Full', 'raw_only': 'Raw signal only', 'context_only': 'Context only', 'responses_only': 'Responses only'}
COLORS = {'full': '#205f50', 'raw_only': '#507ba4', 'context_only': '#9984ac', 'responses_only': '#c4834c'}
VERSION = '1.0.0'
PACKAGE_NAME = 'figure4_singleinput9_results_core_v1_0_0.tar.gz'
TABLES = {'overall': 'overall_metrics.csv', 'cohort': 'per_cohort_metrics.csv',
          'template': 'per_template_metrics.csv', 'template_role': 'per_template_role_metrics.csv'}
METRICS = {'accuracy': ('higher', 'fraction'), 'macro_f1': ('higher', 'fraction'),
           'balanced_accuracy': ('higher', 'fraction'), 'canonical_false_positive_rate': ('lower', 'fraction'),
           'correct_xy_identity_recall': ('higher', 'fraction'), 'any_xy_recall': ('higher', 'fraction'),
           'xy_swap_rate': ('lower', 'fraction'), 'xy_to_canonical_rate': ('lower', 'fraction'),
           'canonical_accuracy': ('higher', 'fraction'), 'nll': ('lower', 'metric_units'),
           'brier': ('lower', 'metric_units'), 'ece_15bins': ('lower', 'fraction')}
REQUIRED_ANALYSIS = ('analysis_manifest.json', 'reviewer_coverage.json', 'coverage_report.json',
    'overall_metrics.csv', 'per_cohort_metrics.csv', 'per_template_metrics.csv',
    'per_template_role_metrics.csv', 'per_template_strand_metrics.csv',
    'per_template_source_run_strand_metrics.csv', 'per_template_pair_metrics.csv',
    'per_reference_metrics.csv', 'per_source_metrics.csv', 'per_run_metrics.csv',
    'per_strand_metrics.csv', 'confidence_thresholds.csv', 'prevalence_scenarios.csv',
    'metadata_coverage.csv', 'template_coverage.csv', 'seed_summary.csv',
    'per_class_metrics.csv.gz', 'confusion_matrices.csv.gz', 'calibration_bins.csv.gz',
    'read_cluster_bootstrap_replicates.csv.gz')


def read_json(path):
    return json.loads(Path(path).read_text(encoding='utf-8'))


def sha256(path):
    digest = hashlib.sha256()
    with Path(path).open('rb') as stream:
        for chunk in iter(lambda: stream.read(8 * 1024 * 1024), b''):
            digest.update(chunk)
    return digest.hexdigest()


def save_json(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + '.tmp-' + str(os.getpid()))
    temporary.write_text(json.dumps(value, ensure_ascii=False, indent=2, allow_nan=False) + '\n', encoding='utf-8')
    temporary.replace(path)


def write_csv(path, frame):
    frame.to_csv(path, index=False, encoding='utf-8-sig')


def canonical_group(value):
    decoded = json.loads(str(value))
    if not isinstance(decoded, dict):
        raise ValueError('group_key必须为JSON对象')
    return json.dumps(decoded, sort_keys=True, separators=(',', ':'), ensure_ascii=False)


def numeric(value, name, missing=False):
    if value is None or value == '' or pd.isna(value):
        if missing:
            return np.nan
        raise ValueError('缺少数值：' + name)
    number = float(value)
    if not math.isfinite(number):
        raise ValueError('非有限数值：' + name)
    return number


def exact_integer(value, name, minimum=0):
    number = numeric(value, name)
    if number != int(number) or number < minimum:
        raise ValueError('非负整数不合法：' + name)
    return int(number)


def close(actual, expected, where):
    if not np.isclose(actual, numeric(expected, where), atol=1e-9, rtol=0):
        raise ValueError('整数混淆矩阵复算不一致：' + where)


def validate_plan(root):
    root = Path(root)
    plan = read_json(root / 'evaluation_plan.json')
    manifest = read_json(root / 'analysis_data/analysis_manifest.json')
    if manifest.get('status') != 'complete' or manifest.get('plan_id') != plan.get('plan_id'):
        raise ValueError('标准分析未完整完成或评估计划不同')
    if manifest.get('evaluation_plan_sha256') != sha256(root / 'evaluation_plan.json'):
        raise ValueError('标准分析记录的计划SHA不同')
    expected = {(scheme, seed) for scheme in SCHEMES for seed in SEEDS}
    runs = plan.get('runs', [])
    if len(runs) != 12 or {(r.get('feature_scheme'), r.get('seed')) for r in runs} != expected:
        raise ValueError('需要完整4种输入×3固定种子，共12个模型')
    lookup = {}
    for run in runs:
        key = run['feature_scheme'], run['seed']
        expected_id = f'{key[0]}/{MODEL}/seed_{key[1]}'
        if run.get('model') != MODEL or run.get('run_id') != expected_id:
            raise ValueError('运行身份不同：' + str(run.get('run_id')))
        lookup[key] = expected_id
    datasets = plan.get('datasets', [])
    identifiers = [d['dataset_id'] for d in datasets]
    expected_design = {(f'XNA{i}', role, strand) for i in range(1, 21)
        for role in (('control', 'target') if i <= 16 else ('target',)) for strand in ('pos', 'neg')}
    actual_design = [(d.get('template'), d.get('role'), d.get('file_strand')) for d in datasets]
    if len(identifiers) != 72 or len(set(identifiers)) != 72 or set(actual_design) != expected_design:
        raise ValueError('需要全部72个修正后的模板/角色/方向文件，不接受筛选子集')
    inputs = manifest.get('prediction_inputs', [])
    expected_pairs = {(run_id, dataset) for run_id in lookup.values() for dataset in identifiers}
    observed = [(item.get('run_id'), item.get('dataset_id')) for item in inputs]
    if len(observed) != 864 or set(observed) != expected_pairs:
        raise ValueError('标准分析的864项逐文件预测核验记录不完整或重复')
    if any(item.get('prediction_sha256_verified') is not True for item in inputs):
        raise ValueError('存在尚未完整核验SHA的逐窗口预测')
    return plan, manifest, lookup


def check_identity(frame, lookup, where):
    required = {'model', 'seed', 'feature_scheme', 'probability_type', 'training_run_id', 'group_key'}
    if not required.issubset(frame.columns):
        raise ValueError(where + '缺少列：' + ','.join(sorted(required - set(frame.columns))))
    if set(frame['model']) != {MODEL} or set(frame['seed']) != set(SEEDS):
        raise ValueError(where + '模型/种子集合不同')
    if set(frame['feature_scheme']) != set(SCHEMES) or set(frame['probability_type']) != {'raw', 'calibrated'}:
        raise ValueError(where + '输入/概率版本集合不完整')
    for row in frame[['feature_scheme', 'seed', 'training_run_id']].drop_duplicates().itertuples(index=False):
        if lookup.get((row.feature_scheme, int(row.seed))) != row.training_run_id:
            raise ValueError(where + '训练运行身份不匹配')


def read_standard_tables(root, lookup):
    tables, sources = {}, {}
    plan = read_json(root / 'evaluation_plan.json')
    scope_keys = {'overall': (), 'cohort': ('cohort',), 'template': ('cohort', 'template'),
        'template_role': ('cohort', 'template', 'role')}
    sources['analysis_data/analysis_manifest.json'] = sha256(root / 'analysis_data/analysis_manifest.json')
    for scope, filename in TABLES.items():
        relative = 'analysis_data/' + filename
        frame = pd.read_csv(root / relative, keep_default_na=False)
        check_identity(frame, lookup, filename)
        frame['seed'] = frame['seed'].astype(int)
        frame['group_key'] = frame['group_key'].map(canonical_group)
        if set(frame.group_level) != {scope}:
            raise ValueError(filename + '分组层级不同')
        keys = ['feature_scheme', 'seed', 'probability_type', 'group_key']
        if frame.duplicated(keys).any():
            raise ValueError(filename + '重复运行/分组')
        expected = {(s, seed, p) for s in SCHEMES for seed in SEEDS for p in ('raw', 'calibrated')}
        expected_groups = {}
        for dataset in plan['datasets']:
            group = json.dumps({key: dataset.get(key, 'unknown') for key in scope_keys[scope]}, sort_keys=True, separators=(',', ':'), ensure_ascii=False)
            expected_groups[group] = expected_groups.get(group, 0) + exact_integer(dataset['n_samples'], 'planned n_samples', 1)
        if set(frame.group_key) != set(expected_groups):
            raise ValueError(filename + '没有完整覆盖冻结计划中的全部分组')
        for _, part in frame.groupby('group_key', sort=False):
            if len(part) != 24 or set(zip(part.feature_scheme, part.seed, part.probability_type)) != expected:
                raise ValueError(filename + '某分组缺少输入/种子/概率版本')
            expected_n = expected_groups[part.group_key.iloc[0]]
            if any(exact_integer(n, 'group n') != expected_n for n in part.n):
                raise ValueError(filename + '分组窗口数与72份冻结输入计划不一致')
        tables[scope], sources[relative] = frame, sha256(root / relative)
    for name in ('confusion_matrices.csv.gz', 'per_class_metrics.csv.gz'):
        sources['analysis_data/' + name] = sha256(root / 'analysis_data' / name)
    return tables, sources


def read_confusions(root, lookup):
    chunks = []
    for chunk in pd.read_csv(root / 'analysis_data/confusion_matrices.csv.gz', chunksize=100000, keep_default_na=False):
        chosen = chunk[chunk.group_level.isin(TABLES)]
        if len(chosen):
            chunks.append(chosen)
    if not chunks:
        raise ValueError('缺少混淆矩阵')
    frame = pd.concat(chunks, ignore_index=True)
    check_identity(frame, lookup, 'confusion_matrices')
    frame['group_key'] = frame['group_key'].map(canonical_group)
    matrices = {}
    keys = ['group_level', 'group_key', 'feature_scheme', 'seed', 'probability_type']
    for identity, part in frame.groupby(keys, sort=False):
        if len(part) != 36 or part.duplicated(['true_label', 'predicted_label']).any():
            raise ValueError('需要无重复的完整6×6混淆矩阵')
        cm = np.zeros((6, 6), dtype=np.int64)
        for row in part.itertuples(index=False):
            true, pred = exact_integer(row.true_label, 'true_label'), exact_integer(row.predicted_label, 'predicted_label')
            if true > 5 or pred > 5:
                raise ValueError('混淆矩阵类别越界')
            cm[true, pred] = exact_integer(row.count, 'count')
        matrices[identity] = cm
    for identity, cm in matrices.items():
        if identity[-1] == 'raw':
            calibrated = matrices.get(identity[:-1] + ('calibrated',))
            if calibrated is None or not np.array_equal(calibrated, cm):
                raise ValueError('正温度校准前后argmax混淆计数不同')
    supports = {}
    for identity, cm in matrices.items():
        key = identity[:2]
        if key in supports and not np.array_equal(supports[key], cm.sum(1)):
            raise ValueError('不同输入/种子/概率版本的六类真值计数不同')
        supports[key] = cm.sum(1)
    plan = read_json(root / 'evaluation_plan.json')
    if all(isinstance(d.get('label_counts'), dict) and d['label_counts'] for d in plan['datasets']):
        scope_keys = {'overall': (), 'cohort': ('cohort',), 'template': ('cohort', 'template'),
            'template_role': ('cohort', 'template', 'role')}
        planned = {}
        for scope, fields in scope_keys.items():
            for dataset in plan['datasets']:
                key = scope, json.dumps({f: dataset.get(f, 'unknown') for f in fields}, sort_keys=True, separators=(',', ':'), ensure_ascii=False)
                expected = planned.setdefault(key, np.zeros(6, dtype=np.int64))
                total = 0
                for label, count in dataset['label_counts'].items():
                    label = exact_integer(label, 'planned label')
                    if label > 5:
                        raise ValueError('冻结计划真值标签越界')
                    count = exact_integer(count, 'planned label count')
                    expected[label] += count; total += count
                if total != dataset['n_samples']:
                    raise ValueError('冻结计划类别计数与窗口数不同')
        for key, actual in supports.items():
            if key not in planned or not np.array_equal(actual, planned[key]):
                raise ValueError('六类真值支持数与72份冻结输入的类别计数不同')
    return matrices


def label_baselines(matrices):
    baselines, predictions = [], []
    for identity, cm in matrices.items():
        scope, group, scheme, seed, probability = identity
        if probability != 'raw':
            continue
        support, predicted = cm.sum(1), cm.sum(0)
        n = int(cm.sum())
        if scheme == 'full' and seed == SEEDS[0]:
            baselines.append(dict(scope=scope, group_key=group, n=n,
                majority_class_names='|'.join(CLASSES[i] for i in range(6) if support[i] == max(support)),
                majority_class_accuracy=float(max(support) / n), uniform_sixclass_expected_accuracy=1 / 6,
                **{'true_support_' + CLASSES[i]: int(support[i]) for i in range(6)},
                interpretation='Descriptive truth prevalence; not an externally fitted predictor or evaluation threshold.'))
        predictions.append(dict(scope=scope, group_key=group, feature_scheme=scheme, seed=seed, n=n,
            **{'predicted_support_' + CLASSES[i]: int(predicted[i]) for i in range(6)},
            n_predicted_classes=int((predicted > 0).sum())))
    return pd.DataFrame(baselines), pd.DataFrame(predictions)


def derive_rows(tables, matrices):
    rows, residuals, checks = [], [], 0
    for scope, frame in tables.items():
        for source in frame.to_dict('records'):
            identity = scope, source['group_key'], source['feature_scheme'], source['seed'], source['probability_type']
            if identity not in matrices:
                raise ValueError('缺少标准表对应混淆矩阵：' + str(identity))
            cm = matrices[identity]
            support, predicted = cm.sum(1), cm.sum(0)
            n, canonical, xy = int(cm.sum()), int(support[:4].sum()), int(support[4:].sum())
            if n <= 0:
                raise ValueError('分组无样本')
            correct, false_xy = int(np.trace(cm)), int(cm[:4, 4:].sum())
            identity_xy = int(cm[4, 4] + cm[5, 5])
            detected_xy, swap, leaked = int(cm[4:, 4:].sum()), int(cm[4, 5] + cm[5, 4]), int(cm[4:, :4].sum())
            canonical_correct = int(sum(cm[i, i] for i in range(4)))
            denominator = support + predicted
            f1 = np.divide(2 * np.diag(cm), denominator, out=np.zeros(6), where=denominator > 0)
            recall = np.divide(np.diag(cm), support, out=np.zeros(6), where=support > 0)
            defined = support > 0
            values = {'accuracy': (correct / n, correct, n), 'macro_f1': (float(f1.mean()), np.nan, np.nan),
                'balanced_accuracy': (float(recall[defined].mean()), np.nan, np.nan),
                'canonical_false_positive_rate': (false_xy / canonical if canonical else np.nan, false_xy, canonical),
                'correct_xy_identity_recall': (identity_xy / xy if xy else np.nan, identity_xy, xy),
                'any_xy_recall': (detected_xy / xy if xy else np.nan, detected_xy, xy),
                'xy_swap_rate': (swap / xy if xy else np.nan, swap, xy),
                'xy_to_canonical_rate': (leaked / xy if xy else np.nan, leaked, xy),
                'canonical_accuracy': (canonical_correct / canonical if canonical else np.nan, canonical_correct, canonical)}
            for metric in ('nll', 'brier', 'ece_15bins'):
                values[metric] = numeric(source[metric], metric), np.nan, n
            for metric, actual in [('n', n), ('correct', correct), ('canonical_support', canonical), ('target_support', xy)]:
                close(actual, source[metric], metric); checks += 1
            for metric in ('accuracy', 'macro_f1', 'balanced_accuracy'):
                close(values[metric][0], source[metric], metric); checks += 1
            if canonical:
                close(values['canonical_false_positive_rate'][0], source['canonical_false_positive_rate'], 'canonical_fpr'); checks += 1
            if xy:
                close(values['correct_xy_identity_recall'][0], source['target_recall_micro'], 'xy_identity'); checks += 1
            meta = {key: source.get(key, '') for key in ('model', 'seed', 'feature_scheme', 'probability_type', 'group_key', 'cohort', 'template', 'role')}
            meta.update(scope=scope, n=n, canonical_support=canonical, xy_support=xy, present_class_count=int(defined.sum()))
            for metric, (value, numerator, total) in values.items():
                preferred, unit = METRICS[metric]
                rows.append(dict(meta, metric=metric, value=value, numerator=numerator, denominator=total,
                                 unit=unit, preferred_direction=preferred))
            if scope == 'template_role' and source.get('role') == 'target' and source.get('cohort') == 'XNA17-20':
                if xy != n:
                    raise ValueError('XNA17-20 target含非X/Y真值')
                for label, count in [(CLASSES[i], int(cm[4:, i].sum())) for i in range(4)] + [('X/Y identity swap', swap)]:
                    residuals.append(dict(meta, error_type=label, error_count=count, denominator=xy, error_per_10000=count / xy * 10000))
    values = pd.DataFrame(rows)
    for _, group in values.groupby(['scope', 'group_key']):
        for key in ('n', 'canonical_support', 'xy_support', 'present_class_count'):
            if group[key].nunique() != 1:
                raise ValueError('不同输入/种子/概率版本的分母不同：' + key)
    return values, pd.DataFrame(residuals), checks


def aggregate_values(rows):
    keys = ['scope', 'group_key', 'cohort', 'template', 'role', 'probability_type', 'metric', 'feature_scheme']
    result = []
    for identity, group in rows.groupby(keys, dropna=False, sort=False):
        if len(group) != 3 or set(group.seed) != set(SEEDS):
            raise ValueError('每种输入必须完整包含3个固定种子')
        values = pd.to_numeric(group.value)
        defined = int(values.notna().sum())
        if defined not in (0, 3):
            raise ValueError('不按部分可用种子汇总')
        result.append(dict(zip(keys, identity), n_seeds=3, n_defined_seeds=defined,
            mean=values.mean() if defined else np.nan, sample_sd=values.std(ddof=1) if defined else np.nan,
            minimum=values.min() if defined else np.nan, maximum=values.max() if defined else np.nan,
            n=int(group.n.iloc[0]), canonical_support=int(group.canonical_support.iloc[0]),
            xy_support=int(group.xy_support.iloc[0]), unit=group.unit.iloc[0], preferred_direction=group.preferred_direction.iloc[0]))
    return pd.DataFrame(result)


def paired_values(rows):
    keys = ['scope', 'group_key', 'cohort', 'template', 'role', 'probability_type', 'metric', 'seed']
    pairs = []
    for identity, group in rows.groupby(keys, dropna=False, sort=False):
        if len(group) != 4 or set(group.feature_scheme) != set(SCHEMES):
            raise ValueError('同seed配对必须完整包含4种输入')
        lookup = group.set_index('feature_scheme')
        full = lookup.loc['full']
        for scheme in SCHEMES[1:]:
            single = lookup.loc[scheme]
            factor = 100 if full.unit == 'fraction' else 1
            defined = not pd.isna(full.value)
            if pd.isna(full.value) != pd.isna(single.value):
                raise ValueError('同seed指标只在一侧有定义')
            pairs.append(dict(zip(keys, identity), comparator_scheme=scheme, full_value=full.value,
                single_value=single.value, difference=(full.value - single.value) * factor if defined else np.nan,
                difference_unit='percentage points' if factor == 100 else 'metric units',
                preferred_direction=full.preferred_direction, n=int(full.n), denominator=full.denominator,
                numerator_full=full.numerator, numerator_single=single.numerator))
    pairs = pd.DataFrame(pairs)
    aggregate = []
    group_keys = keys[:-1] + ['comparator_scheme']
    for identity, group in pairs.groupby(group_keys, dropna=False, sort=False):
        if len(group) != 3 or set(group.seed) != set(SEEDS):
            raise ValueError('配对汇总需要3个固定种子')
        values = pd.to_numeric(group.difference)
        defined = int(values.notna().sum())
        if defined not in (0, 3):
            raise ValueError('配对汇总不能选择部分种子')
        direction = group.preferred_direction.iloc[0]
        better = values.lt(0) if direction == 'lower' else values.gt(0)
        equal = values.eq(0)
        aggregate.append(dict(zip(group_keys, identity), n_pairs=3, n_defined_pairs=defined,
            mean_difference=values.mean() if defined else np.nan, sample_sd=values.std(ddof=1) if defined else np.nan,
            minimum=values.min() if defined else np.nan, maximum=values.max() if defined else np.nan,
            n_full_better=int(better.sum()), n_equal=int(equal.sum()), n_full_worse=int(defined - better.sum() - equal.sum()),
            difference_unit=group.difference_unit.iloc[0], preferred_direction=direction, n=int(group.n.iloc[0]),
            denominator=group.denominator.iloc[0]))
    return pairs, pd.DataFrame(aggregate)


def read_classes(root, lookup, matrices):
    frame = pd.read_csv(root / 'analysis_data/per_class_metrics.csv.gz', keep_default_na=False)
    frame = frame[frame.group_level.isin(TABLES)].copy()
    check_identity(frame, lookup, 'per_class_metrics')
    frame['group_key'] = frame['group_key'].map(canonical_group)
    keys = ['group_level', 'group_key', 'feature_scheme', 'seed', 'probability_type', 'class_id']
    if frame.duplicated(keys).any():
        raise ValueError('每类指标重复')
    values, checks = [], 0
    meta_keys = ['group_level', 'group_key', 'cohort', 'template', 'role', 'seed', 'probability_type', 'feature_scheme', 'class_id', 'class_name']
    for source in frame.to_dict('records'):
        ident = tuple(source[k] for k in keys[:-1])
        cm = matrices.get(ident)
        if cm is None:
            raise ValueError('每类指标缺少对应混淆矩阵')
        k = exact_integer(source['class_id'], 'class_id')
        if k > 5 or source['class_name'] != CLASSES[k]:
            raise ValueError('每类标签定义不一致')
        support, predicted, tp = int(cm[k].sum()), int(cm[:, k].sum()), int(cm[k, k])
        total = int(cm.sum()); fp, fn = predicted - tp, support - tp
        tn = total - tp - fp - fn
        for name, actual in [('support', support), ('predicted_support', predicted), ('tp', tp)]:
            close(actual, source[name], name); checks += 1
        fractions = {'precision': (tp, predicted), 'recall': (tp, support),
            'f1': (2 * tp, support + predicted), 'specificity': (tn, tn + fp),
            'fpr': (fp, tn + fp), 'fdr': (fp, predicted)}
        for metric in ('precision', 'recall', 'f1', 'specificity', 'fpr', 'fdr'):
            value = numeric(source[metric], metric, missing=True)
            numerator, denominator = fractions[metric]
            if denominator:
                close(numerator / denominator, value, 'class ' + metric)
            elif pd.notna(value):
                raise ValueError('无分母的每类指标应保留未定义：' + metric)
            checks += 1
            values.append(dict({k: source.get(k, '') for k in meta_keys}, metric=metric, value=value,
                support=support, predicted_support=predicted, tp=tp))
    long = pd.DataFrame(values)
    expected_count = len(matrices) * 6
    if len(frame) != expected_count:
        raise ValueError('每个分组必须完整包含6类')
    group_keys = [k for k in meta_keys if k not in ('seed', 'feature_scheme')] + ['metric']
    class_aggregate, pairs = [], []
    for identity, group in long.groupby(group_keys + ['feature_scheme'], dropna=False, sort=False):
        if len(group) != 3 or set(group.seed) != set(SEEDS):
            raise ValueError('每类汇总缺少种子')
        vals = pd.to_numeric(group.value); count = int(vals.notna().sum())
        class_aggregate.append(dict(zip(group_keys + ['feature_scheme'], identity), n_seeds=3, n_defined_seeds=count,
            mean=vals.mean() if count else np.nan, sample_sd=vals.std(ddof=1) if count > 1 else np.nan,
            support=int(group.support.iloc[0]), scope_note='Undefined precision or FDR stays missing; defined-seed count is explicit.'))
    for identity, group in long.groupby(group_keys + ['seed'], dropna=False, sort=False):
        if len(group) != 4 or set(group.feature_scheme) != set(SCHEMES):
            raise ValueError('每类配对缺少输入方案')
        lookup_rows = group.set_index('feature_scheme'); full = lookup_rows.loc['full']
        for scheme in SCHEMES[1:]:
            single = lookup_rows.loc[scheme]
            if single.support != full.support:
                raise ValueError('每类分母不同')
            valid = pd.notna(full.value) and pd.notna(single.value)
            pairs.append(dict(zip(group_keys + ['seed'], identity), comparator_scheme=scheme,
                full=full.value, single=single.value, difference_pp=(full.value - single.value) * 100 if valid else np.nan,
                support=int(full.support)))
    pairs = pd.DataFrame(pairs)
    pair_aggregate = []
    for identity, group in pairs.groupby(group_keys + ['comparator_scheme'], dropna=False, sort=False):
        vals = pd.to_numeric(group.difference_pp); count = int(vals.notna().sum())
        pair_aggregate.append(dict(zip(group_keys + ['comparator_scheme'], identity), n_pairs=3, n_defined_pairs=count,
            mean_difference_pp=vals.mean() if count else np.nan, sample_sd_pp=vals.std(ddof=1) if count > 1 else np.nan))
    return frame, pd.DataFrame(class_aggregate), pairs, pd.DataFrame(pair_aggregate), checks


def render_report(destination, aggregate, paired, checks):
    rows = []
    overall = aggregate[(aggregate.scope == 'overall') & (aggregate.probability_type == 'raw')]
    for metric, label in [('accuracy', '准确率'), ('macro_f1', '固定六类Macro-F1'),
        ('canonical_false_positive_rate', '天然窗口误报X/Y'), ('correct_xy_identity_recall', '正确X/Y身份召回'),
        ('any_xy_recall', '任意X/Y检出召回'), ('xy_swap_rate', 'X/Y互换率')]:
        subset = overall[overall.metric == metric].set_index('feature_scheme')
        cells = [html.escape(label)]
        for scheme in SCHEMES:
            record = subset.loc[scheme]
            cells.append('未定义' if pd.isna(record['mean']) else f'{100 * record["mean"]:.4f} ± {100 * record.sample_sd:.4f}%')
        rows.append('<tr>' + ''.join('<td>' + value + '</td>' for value in cells) + '</tr>')
    differences = paired[(paired.scope == 'overall') & (paired.probability_type == 'raw') & (paired.metric == 'macro_f1')]
    difference_rows = ''.join(f'<tr><td>{html.escape(LABELS[r.comparator_scheme])}</td><td>{r.mean_difference:+.4f} ± {r.sample_sd:.4f}</td><td>{r.n_full_better}/3</td></tr>' for r in differences.itertuples(index=False))
    document = '''<!doctype html><html lang="zh-CN"><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1"><title>Figure 4 单输入外部评价</title><style>body{font:16px/1.75 system-ui,"Microsoft YaHei",sans-serif;max-width:1240px;margin:30px auto;padding:0 22px;color:#263e42}h1{font-size:28px}h2{font-size:22px;color:#205f50;margin-top:30px}table{width:100%;border-collapse:collapse;font-size:14px;font-variant-numeric:tabular-nums}th,td{padding:9px;border-bottom:1px solid #dce5e0;text-align:right}th{background:#e8f1ec}td:first-child,th:first-child{text-align:left}.note{background:#f5f1e9;padding:16px}.small{font-size:14px;color:#62776d}img{width:100%;height:auto;margin:18px 0}</style><h1>Figure 4：完整输入与三种单输入的外部评价</h1><p>使用固定seed42、2024、2025。9个单输入模型均已分别训练，本次只加载冻结检查点；3个完整输入GatedConv的已有外部预测经核验后复用。全部12组在相同72份修正后的数据上比较，不重新训练、不拟合外部温度、不优化外部阈值。</p><h2>总体raw结果：三种子均值 ± 样本SD</h2><table><tr><th>指标</th><th>Full</th><th>Raw only</th><th>Context only</th><th>Responses only</th></tr>ROWS</table><p class="small">总体按实际窗口加权；同时阅读cohort、模板和角色分层结果。各模型使用同一批窗口，种子重复不会增加独立生物学样本数。样本SD描述3次训练的变化，并非独立样本泛化置信区间。</p><h2>同seed的完整输入减单输入：Macro-F1</h2><table><tr><th>单输入对照</th><th>平均差值 ± SD（百分点）</th><th>Full更好的种子数</th></tr>DIFFERENCES</table><img src="../figures_singleinput/01_overall_inputs.png" alt="完整和单输入总体指标"><img src="../figures_singleinput/02_same_seed_differences.png" alt="相同种子完整输入减单输入"><h2>模板与天然对照</h2><img src="../figures_singleinput/03_targets_by_template.png" alt="各XNA目标的正确身份召回"><img src="../figures_singleinput/04_controls_by_template.png" alt="各DNA天然对照的误报"><h2>解读规则</h2><ul><li>正确X/Y身份召回要求X判为X、Y判为Y；任意X/Y检出也计入X与Y互换。天然误报的分母是真值A/T/C/G窗口，不能与目标召回混用。</li><li>context_only只输入第4位已置N的序列编码。内部平衡测试的1/6准确率结论不直接迁移到类别或motif分布不同的外部数据；外部表按实际真值计算。外部类别偏斜可能使始终预测一种类别也获得较高准确率，不能把这一点解释为序列输入具有强识别能力。label_prevalence_baselines.csv保存各组多数类准确率及六类真值计数，predicted_class_counts.csv记录各模型预测类别分布。</li><li>单输入结果用于检验信息来源的独立识别能力。Full与raw_only的差异是加入其他输入的联合收益，不能称为“原始信号的必要贡献”；必要性需要相应no_raw等移除实验。</li><li>固定六类Macro-F1包含缺失类为0的项；不同模板可能只含部分真值类别，不能把模板Macro-F1当成同难度排名。缺少分母的召回/FPR保留为空。</li><li>正温度校准只改变概率，不改变argmax。raw为主要分类比较，calibrated概率的NLL、Brier、ECE等另存；温度来自各自内部validation，没有外部拟合。</li><li>可用真实read的聚类bootstrap及窗口描述性Wilson区间沿用标准分析。这里的配对差值按训练种子计算，没有由边际表制造样本级P值或联合read bootstrap。</li><li>XNA/DNA1–12每read1窗口、13–16每read2窗口、XNA17–20每read4窗口；13号目标及对照允许1碱基重叠。可核验范围及实际偏离见窗口审计，缺少的真实ID/坐标不被合成。</li><li>这是已观察外部数据上的后续输入比较，支持范围应与实际结果对应；它不能单独排除来源/批次/参考模板混杂，也不证明所有审稿要求均已完成或某架构优于Transformer。</li></ul><h2>保存与重画</h2><p><code>singleinput_data/metric_values.csv</code>保存逐seed计数、分母及率；<code>metric_aggregate.csv</code>保存均值/SD；<code>paired_differences.csv</code>和<code>paired_aggregate.csv</code>为Full减相同seed单输入；class_*表保存每类结果。原标准<code>analysis_data</code>还包含方向/来源/参考分层、固定阈值、校准及read聚类区间。兼容Figure 4 A/B/C/D仅展示full GatedConv；四种输入比较应使用figures_singleinput及singleinput_data。</p><p>率差值单位为百分点，NLL/Brier为原指标单位；FPR、NLL等越低越好。所有比较完整保留，无选择种子或模板。混淆矩阵/计数完成 CHECKS 项一致性核验。逐窗口CSV、logits、labels、embedding、attention、缓存和检查点保留在服务器；核心包不包含这些大型数组。</p><p>离线重画：<code>python3 summarize_singleinput_external.py --output_root OUT --plot_only</code>。这不会调用模型。核心包优先保证全部12模型的完成记录、配置、紧凑统计与图，motif明细按单独限额收录并明确列出未收录项。</p></html>'''
    document = document.replace('ROWS', ''.join(rows)).replace('DIFFERENCES', difference_rows).replace('CHECKS', str(checks))
    (destination / 'singleinput_report.html').write_text(document, encoding='utf-8')


def export_figure(fig, folder, name):
    for extension in ('png', 'svg'):
        target = folder / (name + '.' + extension)
        temporary = target.with_name(target.name + '.tmp-' + str(os.getpid()))
        fig.savefig(temporary, format=extension, dpi=250, bbox_inches='tight', facecolor='white')
        if extension == 'png' and not temporary.read_bytes().endswith(b'\x00\x00\x00\x00IEND\xaeB`\x82'):
            raise RuntimeError('PNG输出不完整')
        temporary.replace(target)


def make_plots(root):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    folder, destination = root / 'singleinput_data', root / 'figures_singleinput'
    destination.mkdir(exist_ok=True)
    values = pd.read_csv(folder / 'metric_values.csv')
    aggregate = pd.read_csv(folder / 'metric_aggregate.csv')
    pairs = pd.read_csv(folder / 'paired_differences.csv')
    plt.rcParams.update({'font.family': 'DejaVu Sans', 'font.size': 10, 'axes.spines.top': False,
        'axes.spines.right': False, 'text.color': '#263e42', 'svg.fonttype': 'none'})
    markers = {42: 'o', 2024: '^', 2025: 's'}
    metrics = [('macro_f1', 'Six-class macro-F1 (%)'), ('correct_xy_identity_recall', 'Correct X/Y identity (%)'),
               ('canonical_false_positive_rate', 'Canonical false X/Y (%)')]
    fig, axes = plt.subplots(1, 3, figsize=(13.7, 4.4))
    for ax, (metric, label) in zip(axes, metrics):
        for i, scheme in enumerate(SCHEMES):
            subset = values[(values.scope == 'overall') & (values.probability_type == 'raw') & (values.metric == metric) & (values.feature_scheme == scheme)]
            samples = pd.to_numeric(subset.value)
            ax.errorbar(i, samples.mean() * 100, yerr=samples.std(ddof=1) * 100, fmt='D', color=COLORS[scheme], capsize=3, markersize=6)
            for offset, seed in zip((-.14, 0, .14), SEEDS):
                ax.scatter(i + offset, float(subset.loc[subset.seed == seed, 'value'].iloc[0]) * 100,
                    color=COLORS[scheme], marker=markers[seed], s=29, zorder=3)
        ax.set_xticks(range(4), ['Full', 'Raw', 'Context', 'Responses'], rotation=25, ha='right')
        ax.set_ylabel(label); ax.grid(axis='y', alpha=.15); ax.set_axisbelow(True)
        top = ax.get_ylim()[1]
        ax.set_ylim(0, min(100, max(1, top)))
    fig.suptitle('Figure 4: all fixed seeds, raw argmax', x=.07, ha='left', fontsize=14)
    handles = [plt.Line2D([], [], marker=markers[s], ls='', color='#526b61', label='seed ' + str(s)) for s in SEEDS]
    handles.append(plt.Line2D([], [], marker='D', ls='-', color='#526b61', label='mean +/- sample SD'))
    fig.legend(handles=handles, loc='lower center', bbox_to_anchor=(.5, .045), ncol=4, frameon=False, fontsize=9)
    fig.text(.5, .01, 'Rate axes are bounded at 0-100%; any SD extending beyond a boundary is clipped. Complete SD is retained in CSV.', ha='center', fontsize=8)
    fig.subplots_adjust(left=.07, right=.985, bottom=.27, top=.86, wspace=.35)
    export_figure(fig, destination, '01_overall_inputs'); plt.close(fig)
    fig, axes = plt.subplots(1, 3, figsize=(13.7, 4.2))
    for ax, (metric, label) in zip(axes, metrics):
        for i, scheme in enumerate(SCHEMES[1:]):
            subset = pairs[(pairs.scope == 'overall') & (pairs.probability_type == 'raw') & (pairs.metric == metric) & (pairs.comparator_scheme == scheme)]
            samples = pd.to_numeric(subset.difference)
            ax.errorbar(samples.mean(), i, xerr=samples.std(ddof=1), fmt='D', color=COLORS[scheme], capsize=3)
            for offset, seed in zip((-.13, 0, .13), SEEDS):
                ax.scatter(float(subset.loc[subset.seed == seed, 'difference'].iloc[0]), i + offset,
                    color=COLORS[scheme], marker=markers[seed], s=28)
        ax.axvline(0, color='#819087', ls='--', lw=1)
        ax.set_yticks(range(3), ['Raw', 'Context', 'Responses']); ax.set_ylim(2.6, -.6)
        ax.set_title(label.replace(' (%)', ''), loc='left'); ax.set_xlabel('Full minus single input (pp)')
        ax.grid(axis='x', alpha=.15)
    fig.suptitle('Same-seed differences; FPR lower is better', x=.07, ha='left', fontsize=14)
    fig.subplots_adjust(left=.07, right=.985, bottom=.19, top=.8, wspace=.42)
    export_figure(fig, destination, '02_same_seed_differences'); plt.close(fig)
    for number, role, metric, name, ylabel in [(20, 'target', 'correct_xy_identity_recall', '03_targets_by_template', 'Correct X/Y identity recall (%)'),
        (16, 'control', 'canonical_false_positive_rate', '04_controls_by_template', 'Canonical false X/Y calls (%)')]:
        data = aggregate[(aggregate.scope == 'template_role') & (aggregate.probability_type == 'raw') &
            (aggregate.role == role) & (aggregate.metric == metric)].copy()
        data['template_number'] = data.template.str.extract(r'(\d+)').astype(int)
        fig, axes = plt.subplots(2, 2, figsize=(13.5, 7.8), sharex=True, sharey=True)
        for ax, scheme in zip(axes.flat, SCHEMES):
            sub = data[data.feature_scheme == scheme].sort_values('template_number')
            ax.errorbar(sub.template_number, sub['mean'] * 100, yerr=sub.sample_sd * 100, color=COLORS[scheme],
                marker='o', markersize=4, lw=1.25, capsize=2)
            ax.set_title(LABELS[scheme], loc='left'); ax.grid(axis='y', alpha=.15)
            ax.set_xticks(range(1, number + 1), [('DNA' if role == 'control' else 'XNA') + str(n) for n in range(1, number + 1)], rotation=55, ha='right')
            ax.axvline(12.5, color='#ccd5cf', ls='--', lw=1)
            if number == 20:
                ax.axvline(16.5, color='#ccd5cf', ls='--', lw=1)
        for ax in axes[:, 0]:
            ax.set_ylabel(ylabel)
        top = min(100, max(1, axes.flat[0].get_ylim()[1]))
        for ax in axes.flat:
            ax.set_ylim(0, top)
        fig.suptitle('All three fixed seeds; mean +/- sample SD; shared y scale', x=.07, ha='left', fontsize=14)
        fig.text(.5, .015, 'Rate axes bounded at 0-100%; SD extending beyond a boundary is clipped. Complete SD is retained in CSV.', ha='center', fontsize=8)
        fig.subplots_adjust(left=.07, right=.985, bottom=.15, top=.91, wspace=.12, hspace=.4)
        export_figure(fig, destination, name); plt.close(fig)
        write_csv(destination / (name + '.csv'), data)
    save_json(destination / 'plot_manifest.json', {'status': 'complete', 'formats': ['png', 'svg'],
        'summary_manifest_sha256': sha256(folder / 'singleinput_manifest.json'), 'code_sha256': sha256(__file__)})


def package_results(root, optional_budget_bytes=32 * 1024 * 1024, essential_limit_bytes=384 * 1024 * 1024):
    """Never truncate essential records; optional motifs have an independent budget."""
    root = Path(root)
    plan, _, _ = validate_plan(root)
    summary = read_json(root / 'singleinput_data/singleinput_manifest.json')
    if summary.get('status') != 'complete' or summary.get('plan_id') != plan.get('plan_id'):
        raise ValueError('单输入汇总未完成或计划不同')
    for name, expected in summary.get('source_sha256', {}).items():
        if sha256(root / name) != expected:
            raise ValueError('汇总后标准分析文件改变：' + name)
    files, optional, omitted = {}, [], []
    def add(path, name=None, required=False):
        path = Path(path)
        if not path.is_file():
            if required:
                raise ValueError('核心文件缺失：' + str(path))
            return
        relative = name or path.relative_to(root).as_posix()
        if Path(relative).is_absolute() or '..' in Path(relative).parts:
            raise ValueError('非法归档路径')
        if relative in files and files[relative] != path:
            raise ValueError('归档路径冲突')
        files[relative] = path
    for name in ('evaluation_plan.json', 'queue_state.json', 'reuse_manifest.json', 'external_identity_audit.json',
        'external_window_design_audit.json', 'comparison_source_provenance.json', 'singleinput_source_provenance.json', 'singleinput_registry.json',
        'analysis_coverage.json', 'commands.jsonl', 'resource_events.jsonl'):
        add(root / name, required=name in ('evaluation_plan.json', 'reuse_manifest.json', 'external_identity_audit.json', 'external_window_design_audit.json'))
    for name in REQUIRED_ANALYSIS:
        add(root / 'analysis_data' / name, required=True)
    for pattern in ('figure4*.csv', 'figure4*.csv.gz'):
        for path in sorted((root / 'analysis_data').glob(pattern)):
            add(path)
    for folder in ('singleinput_data', 'figures_singleinput', 'source_provenance', 'source_evaluation_metadata'):
        for path in sorted((root / folder).rglob('*')):
            if path.is_file() and path.suffix.lower() not in ('.sqlite', '.db', '.pt', '.npy', '.npz'):
                add(path)
    dataset_ids = {d['dataset_id'] for d in plan['datasets']}
    if len(plan['runs']) != 12 or len(dataset_ids) != 72:
        raise ValueError('核心归档需要12模型/72文件')
    metadata_coverage = []
    for run in plan['runs']:
        run_id = run['run_id']; base = root / 'predictions' / run_id
        marker = read_json(base / 'run_complete.json')
        if marker.get('status') != 'complete' or marker.get('plan_id') != plan['plan_id'] or marker.get('run_id') != run_id or set(marker.get('datasets', [])) != dataset_ids:
            raise ValueError('运行完成记录不完整：' + run_id)
        add(base / 'run_complete.json', required=True)
        for dataset in plan['datasets']:
            path = base / dataset['dataset_id'] / 'completion.json'; value = read_json(path)
            for key, expected in [('status', 'complete'), ('plan_id', plan['plan_id']), ('run_id', run_id), ('dataset_id', dataset['dataset_id']), ('n_samples', dataset['n_samples'])]:
                if value.get(key) != expected:
                    raise ValueError('逐文件完成记录不同：' + str(path) + '/' + key)
            add(path, required=True)
        source = Path(run['run_dir'])
        required_metadata = ('resolved_config.json', 'model_config.json', 'calibration.json')
        for name, field in [('resolved_config.json', 'resolved_config_sha256'), ('calibration.json', 'calibration_sha256')]:
            if sha256(source / name) != run.get(field):
                raise ValueError('冻结训练元数据改变：' + run_id + '/' + name)
        for name in ('queue_requested_config.json', *required_metadata, 'results.json', 'data_audit.json',
            'environment.json', 'epoch_metrics.jsonl', 'metrics.csv', 'source_snapshot/manifest.json'):
            add(source / name, 'training_run_metadata/' + run_id + '/' + name, required=name in required_metadata)
        metadata_coverage.append({'run_id': run_id, 'checkpoint_sha256': run.get('checkpoint_sha256'),
            'resolved_config_sha256': run.get('resolved_config_sha256'), 'calibration_sha256': run.get('calibration_sha256'),
            'frozen_metadata_sha256': run.get('frozen_metadata_sha256', {}),
            'checkpoint_file_included': False, 'required_metadata_complete': True})
    for path in sorted((root / 'resource_profiles').rglob('*.json')):
        add(path)
    for path in sorted((root / 'logs').rglob('invocation.json')):
        add(path)
    here = Path(__file__).resolve().parent
    for path in sorted(here.glob('*.py')):
        add(path, 'evaluation_code/' + path.name)
    for relative in ('vendor/model_benchmark_all.py', 'vendor/IterDataset.py', 'vendor/resource_scheduler.py', 'vendor/cuda_device_probe.py', 'CODE_MANIFEST.json'):
        add(here / relative, 'evaluation_code/' + relative)
    for path in sorted((here / 'reference').rglob('*.json')):
        add(path, 'evaluation_code/' + path.relative_to(here).as_posix())
    essential_names = sorted(files)
    essential_size = sum(path.stat().st_size for path in files.values())
    if essential_size > essential_limit_bytes:
        raise ValueError(f'核心文件超过显式预算({essential_size}字节)，未删减任何后续模型；请检查异常文件后重新打包')
    # Large motif tables may be useful, but can never consume core coverage budget.
    optional_candidates = sorted((root / 'analysis_data').glob('*motif*'))
    used = 0
    for path in optional_candidates:
        if not path.is_file() or path.name in REQUIRED_ANALYSIS:
            continue
        size = path.stat().st_size
        if used + size <= optional_budget_bytes:
            add(path); optional.append(path.relative_to(root).as_posix()); used += size
        else:
            omitted.append({'path': path.relative_to(root).as_posix(), 'size_bytes': size,
                'reason': 'optional motif budget; all core records already reserved'})
    records = {}
    for name, path in sorted(files.items()):
        before = path.stat(); digest = sha256(path); after = path.stat()
        if (before.st_size, before.st_mtime_ns) != (after.st_size, after.st_mtime_ns):
            raise ValueError('打包时文件变化：' + name)
        records[name] = {'size_bytes': after.st_size, 'mtime_ns': after.st_mtime_ns, 'sha256': digest,
            'priority': 'essential' if name in essential_names else 'optional'}
    package_manifest = {'status': 'complete', 'package_version': VERSION, 'plan_id': plan['plan_id'],
        'planned_runs': 12, 'planned_datasets': 72, 'completion_records': 864,
        'essential_count': len(essential_names), 'essential_bytes': essential_size, 'optional_bytes': used,
        'essential_files': essential_names, 'optional_files': optional, 'omitted_optional_files': omitted,
        'run_metadata_coverage': metadata_coverage, 'files': records,
        'excluded': 'Per-window CSV/arrays, caches, checkpoints, audit databases and training inputs remain on server.',
        'policy': 'No essential file is budget-truncated; all12 runs are mandatory before any optional motif table.'}
    destination = root / PACKAGE_NAME
    temporary = destination.with_name(destination.name + '.partial-' + str(os.getpid()))
    try:
        with tarfile.open(temporary, 'w:gz', compresslevel=5, dereference=True) as archive:
            for name in essential_names + optional:
                path, record = files[name], records[name]; before = path.stat()
                if (before.st_size, before.st_mtime_ns) != (record['size_bytes'], record['mtime_ns']):
                    raise ValueError('打包前文件改变：' + name)
                archive.add(path, arcname=name, recursive=False)
                after = path.stat()
                if (before.st_size, before.st_mtime_ns) != (after.st_size, after.st_mtime_ns):
                    raise ValueError('打包期间文件改变：' + name)
            data = (json.dumps(package_manifest, ensure_ascii=False, indent=2, allow_nan=False) + '\n').encode()
            item = tarfile.TarInfo('RESULT_PACKAGE_MANIFEST.json'); item.size = len(data)
            archive.addfile(item, io.BytesIO(data))
        # Stream-read every member and check hashes before publishing the archive.
        with tarfile.open(temporary, 'r:gz') as archive:
            names = [member.name for member in archive.getmembers()]
            if len(names) != len(records) + 1 or set(names) != set(records) | {'RESULT_PACKAGE_MANIFEST.json'}:
                raise ValueError('归档包含文件数/路径不一致')
            for name, expected in records.items():
                stream = archive.extractfile(name); digest = hashlib.sha256(); size = 0
                if stream is None:
                    raise ValueError('归档成员不可读')
                for chunk in iter(lambda: stream.read(8 * 1024 * 1024), b''):
                    digest.update(chunk); size += len(chunk)
                if digest.hexdigest() != expected['sha256'] or size != expected['size_bytes']:
                    raise ValueError('归档成员SHA/大小不同：' + name)
        temporary.replace(destination)
    except BaseException:
        temporary.unlink(missing_ok=True)
        raise
    print('[RESULT PACKAGE] ' + str(destination), flush=True)
    return destination


def summarize(output_root, plot_only=False, no_plots=False, package=True):
    root = Path(output_root).expanduser().resolve()
    plan, analysis, lookup = validate_plan(root)
    tables, fingerprints = read_standard_tables(root, lookup)
    destination = root / 'singleinput_data'
    if plot_only:
        saved = read_json(destination / 'singleinput_manifest.json')
        if saved.get('status') != 'complete' or saved.get('plan_id') != plan['plan_id'] or saved.get('source_sha256') != fingerprints:
            raise ValueError('已有汇总与计划/标准表不同，请先完整分析')
    else:
        matrices = read_confusions(root, lookup)
        values, residuals, checks = derive_rows(tables, matrices)
        aggregate = aggregate_values(values); pairs, paired = paired_values(values)
        classes, class_aggregate, class_pairs, class_pair_aggregate, class_checks = read_classes(root, lookup, matrices)
        baselines, predicted = label_baselines(matrices)
        checks += class_checks
        with tempfile.TemporaryDirectory(prefix='.singleinput-summary-', dir=root) as temporary:
            stage = Path(temporary) / 'singleinput_data'; stage.mkdir()
            for name, frame in [('metric_values.csv', values), ('metric_aggregate.csv', aggregate),
                ('paired_differences.csv', pairs), ('paired_aggregate.csv', paired), ('residual_errors.csv', residuals),
                ('class_values.csv', classes), ('class_aggregate.csv', class_aggregate),
                ('class_paired_differences.csv', class_pairs), ('class_paired_aggregate.csv', class_pair_aggregate),
                ('label_prevalence_baselines.csv', baselines), ('predicted_class_counts.csv', predicted)]:
                write_csv(stage / name, frame)
            for scope in TABLES:
                write_csv(stage / (scope + '_values.csv'), values[values.scope == scope])
                write_csv(stage / (scope + '_aggregate.csv'), aggregate[aggregate.scope == scope])
                write_csv(stage / (scope + '_paired_differences.csv'), pairs[pairs.scope == scope])
                write_csv(stage / (scope + '_paired_aggregate.csv'), paired[paired.scope == scope])
            render_report(stage, aggregate, paired, checks)
            save_json(stage / 'singleinput_manifest.json', {'status': 'complete', 'version': VERSION, 'plan_id': plan['plan_id'],
                'seeds': list(SEEDS), 'feature_schemes': list(SCHEMES), 'model': MODEL,
                'source_sha256': fingerprints, 'code_sha256': sha256(__file__),
                'counts': {'runs': 12, 'datasets': 72, 'prediction_files': 864, 'metric_rows': len(values),
                    'paired_rows': len(pairs), 'integer_confusion_and_class_checks': checks},
                'difference_direction': 'Full minus same-seed single input; fractions reported as percentage points',
                'uncertainty': 'Sample SD of three fixed training seeds; standard read-cluster CIs remain in analysis_data; no joint pairwise read bootstrap.',
                'external_scope': 'Follow-up input comparison on previously observed external data; no training/external fitting or selection.',
                'created_at': datetime.now(timezone.utc).isoformat()})
            old = Path(temporary) / 'old_singleinput_data'
            if destination.exists():
                destination.rename(old)
            try:
                stage.rename(destination)
            except BaseException:
                if old.exists():
                    old.rename(destination)
                raise
    if not no_plots:
        make_plots(root)
    archive = package_results(root) if package else destination
    print('[SINGLEINPUT ANALYSIS COMPLETE] ' + str(destination / 'singleinput_report.html'), flush=True)
    return archive


make_core_archive = package_results


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output_root', required=True, type=Path)
    parser.add_argument('--plot_only', action='store_true')
    parser.add_argument('--no_plots', action='store_true')
    parser.add_argument('--package_only', action='store_true', help='仅按现有完整汇总重新打包，不调用模型或重算分析')
    args = parser.parse_args(argv)
    if args.plot_only and args.no_plots:
        parser.error('plot_only不能同时no_plots')
    if args.package_only:
        if args.plot_only or args.no_plots:
            parser.error('package_only不能与plot_only/no_plots同时使用')
        package_results(args.output_root.expanduser().resolve())
    else:
        summarize(args.output_root, plot_only=args.plot_only, no_plots=args.no_plots)
    return 0


if __name__ == '__main__':
    try:
        raise SystemExit(main())
    except Exception as error:
        print('[ERROR] ' + type(error).__name__ + ': ' + str(error), file=__import__('sys').stderr)
        raise SystemExit(1)
