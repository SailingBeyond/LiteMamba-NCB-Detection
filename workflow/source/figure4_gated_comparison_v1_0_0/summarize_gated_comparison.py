#!/usr/bin/env python3
"""汇总固定三种子的 Figure4 门控卷积对照，不加载模型或调整阈值。"""
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
import shutil
import sys
import tarfile
import tempfile

import numpy as np
import pandas as pd

SEEDS = (42, 2024, 2025)
MODELS = ('litemamba', 'gatedconv_target')
LABELS = {'litemamba': 'SSM full', 'gatedconv_target': 'Gated convolution'}
COLORS = {'litemamba': '#205f50', 'gatedconv_target': '#5989a3'}
CLASSES = ('A', 'T', 'C', 'G', 'X', 'Y')
VERSION = '1.0.0'
PACKAGE_NAME = 'figure4_gated_comparison_results_for_analysis_v1_0_0.tar.gz'
TABLES = {
    'overall': 'overall_metrics.csv',
    'cohort': 'per_cohort_metrics.csv',
    'template': 'per_template_metrics.csv',
    'template_role': 'per_template_role_metrics.csv',
}
METRICS = {
    'accuracy': ('higher', 'fraction'),
    'macro_f1': ('higher', 'fraction'),
    'balanced_accuracy': ('higher', 'fraction'),
    'canonical_false_positive_rate': ('lower', 'fraction'),
    'correct_xy_identity_recall': ('higher', 'fraction'),
    'any_xy_recall': ('higher', 'fraction'),
    'xy_swap_rate': ('lower', 'fraction'),
    'xy_to_canonical_rate': ('lower', 'fraction'),
    'nll': ('lower', 'metric_units'),
    'brier': ('lower', 'metric_units'),
    'ece_15bins': ('lower', 'fraction'),
}


def read_json(path):
    return json.loads(Path(path).read_text(encoding='utf-8'))


def sha256(path):
    value = hashlib.sha256()
    with Path(path).open('rb') as stream:
        for chunk in iter(lambda: stream.read(8*1024*1024), b''):
            value.update(chunk)
    return value.hexdigest()


def save_json(path, value):
    path = Path(path)
    temporary = path.with_name(path.name+'.tmp-'+str(os.getpid()))
    temporary.write_text(json.dumps(value, ensure_ascii=False, indent=2, allow_nan=False)+'\n', encoding='utf-8')
    temporary.replace(path)


def write_csv(path, frame):
    frame.to_csv(path, index=False, encoding='utf-8-sig')


def canonical_group(value):
    decoded = json.loads(str(value))
    if not isinstance(decoded, dict):
        raise ValueError('group_key必须是JSON对象')
    return json.dumps(decoded, sort_keys=True, separators=(',', ':'), ensure_ascii=False)


def validate_plan(root):
    plan = read_json(root/'evaluation_plan.json')
    manifest = read_json(root/'analysis_data/analysis_manifest.json')
    if manifest.get('status') != 'complete' or manifest.get('plan_id') != plan.get('plan_id'):
        raise ValueError('标准分析尚未完整完成或不属于当前评估计划；先运行phase analyze')
    if manifest.get('evaluation_plan_sha256') != sha256(root/'evaluation_plan.json'):
        raise ValueError('标准分析记录的evaluation_plan SHA与当前计划不同')
    expected = {(model, seed) for model in MODELS for seed in SEEDS}
    records = [(run.get('model'), run.get('seed')) for run in plan.get('runs', [])]
    if len(records) != 6 or set(records) != expected:
        raise ValueError('比较必须包含full和门控卷积各自的全部seed42/2024/2025，共6项')
    if any(run.get('feature_scheme') != 'full' for run in plan['runs']):
        raise ValueError('本比较只接受full特征方案')
    lookup = {(run['model'], run['seed']): run['run_id'] for run in plan['runs']}
    for key, run_id in lookup.items():
        if run_id != 'full/'+key[0]+'/seed_'+str(key[1]):
            raise ValueError('运行身份或目录含义不同：'+str(run_id))
    dataset_ids = [d['dataset_id'] for d in plan.get('datasets', [])]
    if not dataset_ids or len(dataset_ids) != len(set(dataset_ids)):
        raise ValueError('冻结数据集合为空或dataset_id重复')
    # 标准分析已经核验完整逐窗口输出；此处核验其完成记录覆盖范围。
    inputs = manifest.get('prediction_inputs', [])
    if inputs:
        expected_pairs = {(run_id, dataset_id) for run_id in lookup.values() for dataset_id in dataset_ids}
        observed = [(item.get('run_id'), item.get('dataset_id')) for item in inputs]
        if len(observed) != len(expected_pairs) or set(observed) != expected_pairs:
            raise ValueError('标准分析的预测文件集合不完整或重复')
        if any(item.get('prediction_sha256_verified') is not True for item in inputs):
            raise ValueError('存在未核验完整SHA的逐窗口预测，拒绝作为本次完整比较')
    return plan, manifest, lookup


def check_identity(frame, lookup, where):
    required = {'model', 'seed', 'feature_scheme', 'probability_type', 'training_run_id', 'group_key'}
    if not required.issubset(frame.columns):
        raise ValueError(where+'缺少列：'+','.join(sorted(required-set(frame.columns))))
    if set(frame['model']) != set(MODELS) or set(frame['seed']) != set(SEEDS):
        raise ValueError(where+'模型/种子集合不同，不允许事后筛选或补成重复')
    if set(frame['feature_scheme']) != {'full'} or set(frame['probability_type']) != {'raw', 'calibrated'}:
        raise ValueError(where+'特征方案或概率类型不完整')
    for row in frame[['model', 'seed', 'training_run_id']].drop_duplicates().itertuples(index=False):
        if lookup.get((row.model, int(row.seed))) != row.training_run_id:
            raise ValueError(where+'训练运行身份不匹配')


def read_standard_tables(root, lookup):
    tables = {}
    files = ['analysis_data/analysis_manifest.json', 'analysis_data/confusion_matrices.csv.gz',
             'analysis_data/per_class_metrics.csv.gz']
    for scope, filename in TABLES.items():
        path = root/'analysis_data'/filename
        frame = pd.read_csv(path, keep_default_na=False)
        check_identity(frame, lookup, filename)
        frame['seed'] = frame['seed'].astype(int)
        frame['group_key'] = frame['group_key'].map(canonical_group)
        if set(frame.group_level) != {scope}:
            raise ValueError(filename+'分组层级不同')
        keys = ['model', 'seed', 'probability_type', 'group_key']
        if frame.duplicated(keys).any():
            raise ValueError(filename+'存在重复运行/分组')
        # 每个分组必须完整包含2个结构×3个种子×2种概率。
        for _, part in frame.groupby('group_key', sort=False):
            if len(part) != 12 or set(zip(part.model, part.seed, part.probability_type)) != {
                    (model, seed, probability) for model in MODELS for seed in SEEDS for probability in ('raw', 'calibrated')}:
                raise ValueError(filename+'某分组缺少种子或概率类型')
        tables[scope] = frame
        files.append('analysis_data/'+filename)
    return tables, {name: sha256(root/name) for name in files}


def numeric(value, name, allow_missing=False):
    if value == '' or value is None or pd.isna(value):
        if allow_missing:
            return np.nan
        raise ValueError('缺少数值：'+name)
    number = float(value)
    if not math.isfinite(number):
        raise ValueError('非有限数值：'+name)
    return number


def close(actual, expected, name):
    if not np.isclose(actual, numeric(expected, name), atol=1e-9, rtol=0):
        raise ValueError('整数混淆矩阵复算与标准表不同：'+name)


def read_confusions(root, lookup):
    chunks = []
    for chunk in pd.read_csv(root/'analysis_data/confusion_matrices.csv.gz', chunksize=100000, keep_default_na=False):
        chosen = chunk[chunk.group_level.isin(TABLES)]
        if len(chosen):
            chunks.append(chosen)
    if not chunks:
        raise ValueError('缺少所需混淆矩阵')
    frame = pd.concat(chunks, ignore_index=True)
    check_identity(frame, lookup, 'confusion_matrices')
    frame['group_key'] = frame['group_key'].map(canonical_group)
    keys = ['group_level', 'group_key', 'model', 'seed', 'probability_type']
    matrices = {}
    for identity, part in frame.groupby(keys, sort=False):
        if len(part) != 36 or part.duplicated(['true_label', 'predicted_label']).any():
            raise ValueError('混淆矩阵必须是无重复的完整6×6')
        cm = np.zeros((6, 6), dtype=np.int64)
        for row in part.itertuples(index=False):
            true, pred, count = int(row.true_label), int(row.predicted_label), int(row.count)
            if not 0 <= true < 6 or not 0 <= pred < 6 or count < 0 or count != row.count:
                raise ValueError('混淆矩阵索引/计数不合法')
            cm[true, pred] = count
        matrices[identity] = cm
    return matrices


def derive_rows(tables, matrices):
    rows, residuals = [], []
    for scope, frame in tables.items():
        for source in frame.to_dict('records'):
            identity = (scope, source['group_key'], source['model'], source['seed'], source['probability_type'])
            if identity not in matrices:
                raise ValueError('缺少标准表对应的混淆矩阵：'+str(identity))
            cm = matrices[identity]
            support, predicted = cm.sum(1), cm.sum(0)
            n, canonical, xy = int(cm.sum()), int(support[:4].sum()), int(support[4:].sum())
            if n <= 0:
                raise ValueError('比较分组不能为空')
            correct, false_xy = int(np.trace(cm)), int(cm[:4, 4:].sum())
            xy_identity = int(cm[4, 4]+cm[5, 5])
            detected_xy, swap, leaked = int(cm[4:, 4:].sum()), int(cm[4, 5]+cm[5, 4]), int(cm[4:, :4].sum())
            denominator = support+predicted
            f1 = np.divide(2*np.diag(cm), denominator, out=np.zeros(6), where=denominator>0)
            recalls = np.divide(np.diag(cm), support, out=np.zeros(6), where=support>0)
            defined = support > 0
            values = {'accuracy': (correct/n, correct, n), 'macro_f1': (float(f1.mean()), np.nan, np.nan),
                'balanced_accuracy': (float(recalls[defined].mean()), np.nan, np.nan),
                'canonical_false_positive_rate': (false_xy/canonical if canonical else np.nan, false_xy, canonical),
                'correct_xy_identity_recall': (xy_identity/xy if xy else np.nan, xy_identity, xy),
                'any_xy_recall': (detected_xy/xy if xy else np.nan, detected_xy, xy),
                'xy_swap_rate': (swap/xy if xy else np.nan, swap, xy),
                'xy_to_canonical_rate': (leaked/xy if xy else np.nan, leaked, xy)}
            for metric in ('nll', 'brier', 'ece_15bins'):
                values[metric] = (numeric(source[metric], metric), np.nan, n)
            close(n, source['n'], 'n'); close(correct, source['correct'], 'correct')
            for metric in ('accuracy', 'macro_f1', 'balanced_accuracy'):
                close(values[metric][0], source[metric], metric)
            close(canonical, source['canonical_support'], 'canonical_support')
            close(xy, source['target_support'], 'target_support')
            if canonical:
                close(values['canonical_false_positive_rate'][0], source['canonical_false_positive_rate'], 'canonical_fpr')
            if xy:
                close(values['correct_xy_identity_recall'][0], source['target_recall_micro'], 'correct_xy_identity_recall')
            meta = {key: source.get(key, '') for key in ('model', 'seed', 'probability_type', 'group_key', 'cohort', 'template', 'role')}
            meta.update(scope=scope, n=n, canonical_support=canonical, xy_support=xy, present_class_count=int(defined.sum()))
            for metric, (value, numerator, total) in values.items():
                preferred, unit = METRICS[metric]
                rows.append(dict(meta, metric=metric, value=value, numerator=numerator, denominator=total,
                                 unit=unit, preferred_direction=preferred))
            if scope == 'template_role' and source.get('role') == 'target' and source.get('cohort') == 'XNA17-20':
                if xy != n:
                    raise ValueError('XNA17-20 target中出现非X/Y真值；不能把它们悄悄排除')
                for label, count in [(CLASSES[i], int(cm[4:, i].sum())) for i in range(4)]+[('X/Y identity swap', swap)]:
                    residuals.append(dict(meta, error_type=label, error_count=count, denominator=xy,
                                          error_per_10000=count/xy*10000))
    # 同一数据组的支持数必须逐seed、逐结构、逐概率一致。
    result = pd.DataFrame(rows)
    for _, group in result.groupby(['scope', 'group_key']):
        for key in ('n', 'canonical_support', 'xy_support', 'present_class_count'):
            if group[key].nunique() != 1:
                raise ValueError('比较双方或种子间的分母不同：'+key)
    return result, pd.DataFrame(residuals)


def aggregate_values(rows):
    keys = ['scope', 'group_key', 'cohort', 'template', 'role', 'probability_type', 'metric', 'model']
    output = []
    for identity, group in rows.groupby(keys, dropna=False, sort=False):
        if set(group.seed) != set(SEEDS) or len(group) != 3:
            raise ValueError('每结构分组必须完整包含3个seed')
        values = group.value.astype(float)
        count = int(values.notna().sum())
        if count not in (0, 3):
            raise ValueError('同一指标仅在部分种子定义，不能按可用种子选择性汇总')
        record = dict(zip(keys, identity))
        record.update(n_seeds=3, n_defined_seeds=count, mean=values.mean() if count else np.nan,
                      sample_sd=values.std(ddof=1) if count else np.nan,
                      minimum=values.min() if count else np.nan, maximum=values.max() if count else np.nan,
                      n=int(group.n.iloc[0]), canonical_support=int(group.canonical_support.iloc[0]),
                      xy_support=int(group.xy_support.iloc[0]), unit=group.unit.iloc[0],
                      preferred_direction=group.preferred_direction.iloc[0])
        output.append(record)
    return pd.DataFrame(output)


def paired_values(rows):
    keys = ['scope', 'group_key', 'cohort', 'template', 'role', 'probability_type', 'metric', 'seed']
    pairs = []
    for identity, group in rows.groupby(keys, dropna=False, sort=False):
        if set(group.model) != set(MODELS) or len(group) != 2:
            raise ValueError('同seed配对必须恰好包含full和门控卷积')
        lookup = group.set_index('model')
        full, gate = lookup.loc['litemamba'], lookup.loc['gatedconv_target']
        factor = 100 if full.unit == 'fraction' else 1
        missing = pd.isna(full.value) and pd.isna(gate.value)
        if pd.isna(full.value) != pd.isna(gate.value):
            raise ValueError('同seed指标仅在一侧可用')
        pairs.append(dict(zip(keys, identity), reference_value=full.value, gated_value=gate.value,
            difference=(gate.value-full.value)*factor if not missing else np.nan,
            difference_unit='percentage points' if factor==100 else 'metric units',
            preferred_direction=full.preferred_direction, n=int(full.n),
            denominator=full.denominator, numerator_full=full.numerator, numerator_gated=gate.numerator))
    pairs = pd.DataFrame(pairs)
    group_keys = keys[:-1]
    summaries = []
    for identity, group in pairs.groupby(group_keys, dropna=False, sort=False):
        if set(group.seed) != set(SEEDS) or len(group) != 3:
            raise ValueError('配对统计缺少种子或有重复种子')
        values = group.difference.astype(float)
        count = int(values.notna().sum())
        if count not in (0,3):
            raise ValueError('不能仅用部分seed计算配对统计')
        preferred = group.preferred_direction.iloc[0]
        better = (values<0) if preferred=='lower' else (values>0)
        equal = values.eq(0)
        summaries.append(dict(zip(group_keys, identity), n_pairs=3, n_defined_pairs=count,
            mean_difference=values.mean() if count else np.nan, sample_sd=values.std(ddof=1) if count else np.nan,
            minimum=values.min() if count else np.nan, maximum=values.max() if count else np.nan,
            n_better=int(better.sum()) if count else 0, n_equal=int(equal.sum()) if count else 0,
            n_worse=int(count-better.sum()-equal.sum()) if count else 0,
            difference_unit=group.difference_unit.iloc[0], preferred_direction=preferred,
            n=int(group.n.iloc[0]), denominator=group.denominator.iloc[0]))
    return pairs, pd.DataFrame(summaries)


def read_per_class(root, lookup):
    frame = pd.read_csv(root/'analysis_data/per_class_metrics.csv.gz', keep_default_na=False)
    frame = frame[frame.group_level.isin(['overall', 'cohort', 'template_role'])].copy()
    check_identity(frame, lookup, 'per_class_metrics')
    frame['group_key'] = frame['group_key'].map(canonical_group)
    keys = ['group_level', 'group_key', 'model', 'seed', 'probability_type', 'class_id']
    if frame.duplicated(keys).any():
        raise ValueError('每类指标重复')
    pairs = []
    group_keys = ['group_level', 'group_key', 'cohort', 'template', 'role', 'seed', 'probability_type', 'class_id', 'class_name']
    for identity, group in frame.groupby(group_keys, dropna=False, sort=False):
        if set(group.model)!=set(MODELS) or len(group)!=2:
            raise ValueError('每类指标缺少结构或种子')
        full, gate = group.set_index('model').loc[list(MODELS)].to_dict('records')
        if numeric(full['support'], 'support') != numeric(gate['support'], 'support'):
            raise ValueError('每类指标分母不一致')
        for metric in ['precision', 'recall', 'f1', 'specificity', 'fpr', 'fdr']:
            a = numeric(full[metric], metric, allow_missing=True)
            b = numeric(gate[metric], metric, allow_missing=True)
            pairs.append(dict(zip(group_keys, identity), metric=metric, full=a, gated=b,
                difference_pp=(b-a)*100 if math.isfinite(a) and math.isfinite(b) else np.nan,
                support=int(full['support']), tp_full=int(full['tp']), tp_gated=int(gate['tp']),
                predicted_support_full=int(full['predicted_support']), predicted_support_gated=int(gate['predicted_support'])))
    pairs = pd.DataFrame(pairs)
    summaries = []
    keys = [k for k in group_keys if k!='seed']+['metric']
    for identity, group in pairs.groupby(keys, dropna=False, sort=False):
        if set(group.seed)!=set(SEEDS) or len(group)!=3:
            raise ValueError('每类配对统计缺少3个固定种子')
        values=group.difference_pp
        count=int(values.notna().sum())
        summaries.append(dict(zip(keys, identity), n_pairs=3,n_defined_pairs=count,
            mean_difference_pp=values.mean() if count else np.nan,
            sample_sd_pp=values.std(ddof=1) if count>1 else np.nan,
            scope_note='class-specific undefined denominators remain missing; n_defined_pairs is explicit'))
    return frame, pairs, pd.DataFrame(summaries)


def render_report(folder, aggregate, paired):
    def cell(value, unit='fraction'):
        if pd.isna(value):return '未定义'
        return f'{100*value:.4f}%' if unit=='fraction' else f'{value:.6f}'
    headline=[]
    for metric,label in [('macro_f1','总体Macro-F1'),('canonical_false_positive_rate','天然窗口误报X/Y'),
                         ('correct_xy_identity_recall','正确X/Y身份召回'),('xy_swap_rate','X/Y互换率'),('nll','NLL')]:
        part=aggregate[(aggregate.scope=='overall')&(aggregate.probability_type=='raw')&(aggregate.metric==metric)].set_index('model')
        delta=paired[(paired.scope=='overall')&(paired.probability_type=='raw')&(paired.metric==metric)].iloc[0]
        values=[]
        for model in MODELS:
            r=part.loc[model];factor=100 if r.unit=='fraction' else 1
            values.append(f'{factor*r["mean"]:.4f} ± {factor*r.sample_sd:.4f}'+('%' if factor==100 else ''))
        headline.append('<tr><td>'+label+'</td><td>'+values[0]+'</td><td>'+values[1]+'</td><td>'+f'{delta.mean_difference:+.4f} ± {delta.sample_sd:.4f}'+'</td><td>'+str(int(delta.n_better))+'/3</td></tr>')
    doc='''<!doctype html><html lang="zh-CN"><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1"><title>Figure4固定三种子结构对照</title><style>body{font:16px/1.8 system-ui,"Microsoft YaHei",sans-serif;max-width:1240px;margin:32px auto;padding:0 24px;color:#263e42}h1{font-size:28px}h2{font-size:22px;color:#205f50;margin-top:30px}table{width:100%;border-collapse:collapse;font-size:14px;font-variant-numeric:tabular-nums}th,td{padding:10px;border-bottom:1px solid #dce5e0;text-align:right}th{background:#e8f1ec}td:first-child,th:first-child{text-align:left}.note{background:#f5f1e9;padding:16px}.small{font-size:14px;color:#62776d}img{width:100%;height:auto;display:block;margin:20px 0}</style><h1>Figure4：SSM full与门控卷积的固定三种子比较</h1><p>本报告使用全部seed42、2024、2025，模型权重与validation温度冻结。full预测经来源核验后复用，门控卷积只做新预测；没有训练、更新权重、拟合外部温度或优化阈值。</p><p class="note">这是已经观察过外部数据上的后续结构比较。它检验内部验证集的结构差异是否转化为外部取舍，不能重新称为从未见过的盲测，也不能据此继续循环改结构直到某一方获胜。所有结果完整保留，报告不预设胜者。</p><h2>总体raw结果</h2><table><tr><th>指标</th><th>SSM full，均值±SD</th><th>门控卷积，均值±SD</th><th>配对变化，均值±SD</th><th>门控更好种子数</th></tr>HEADLINE</table><p class="small">SD是3个训练种子的样本标准差。差值为门控减相同seed的full；率的差值单位为百分点，NLL/Brier为原始指标单位。天然误报越低越好，正确身份召回越高越好。种子重复没有增加独立窗口数；SD不等于独立样本泛化置信区间。</p><img src="../figures_comparison/01_overall_and_paired_changes.png" alt="总体表现与同seed差值"><h2>各天然对照与目标模板</h2><p>天然对照误报使用control角色窗口中真值A/T/C/G被预测为X/Y的比例。目标识别使用target角色中真值X/Y被预测为正确X或Y的比例，分母及计数均保存在逐seed表。pos/neg为文件分组，不推断未提供的BAM方向或分子配对。</p><img src="../figures_comparison/02_control_false_calls_by_template.png" alt="天然对照误报"><img src="../figures_comparison/03_target_identity_recall_by_template.png" alt="目标身份召回"><h2>XNA17–20残余错误</h2><p>残差分为误判A、T、C、G与X/Y身份互换，分母为相应模板真值X/Y窗口数。堆叠高度为各错误率的三seed均值；总错误点与误差线描述同三seed的变化，不把均值分母伪装成更多独立样本。</p><img src="../figures_comparison/04_xna17_20_residual_errors.png" alt="XNA17–20残余错误"><h2>解释范围</h2><ul><li>raw和validation温度校准后结果分别保存，不是两次独立训练。固定正温度不会改变argmax分类；校准只改变概率，温度不在外部数据重新拟合。</li><li>总体Macro-F1按固定六类计算，缺失类别F1为0；模板/角色可能不含全部六类，不能把不同类别覆盖的模板Macro-F1直接当作同一难度排名。</li><li>已有标准分析的窗口Wilson区间和可用真实read聚类bootstrap保留。这里仅作同seed结构差值，没有从边际表计算样本级P值、McNemar或跨结构联合read bootstrap。</li><li>未知read ID、坐标、实验批次和配对身份仍保持未知。XNA1–12/DNA1–12每read1个窗口，XNA13–16/DNA13–16每read2个窗口，XNA17–20每read4个窗口属于声明的实验设计；XNA13及DNA13允许两个窗口存在1碱基重叠。相应可核验范围见外部窗口审计。</li><li>不含其他模型的新预测，也不自动宣称LiteMamba优于Transformer或具有算法新颖性。是否更换主模型还涉及命名、该骨干自己的消融和其他任务证据。</li></ul><h2>文件与后续画图</h2><p>comparison_data/metric_values.csv保存完整分母与逐seed率；metric_aggregate.csv保存每结构均值/SD；paired_differences.csv与paired_aggregate.csv保存同seed差值；class_*保存每类结果；residual_errors.csv保存残差计数。标准analysis_data全部保留原格式。旧figure4A/B/C兼容表仍只针对litemamba full，不能作为门控卷积结果；两结构比较应使用本目录及figures_comparison。</p><p>完整逐窗口CSV、logits、labels、embedding、attention和checkpoint仍保留在服务器，结果小包只包含统计、图和溯源。重画可运行<code>python3 summarize_gated_comparison.py --output_root OUT --plot_only</code>，不调用模型。</p></html>'''
    doc=doc.replace('<h2>各天然对照与目标模板</h2>',
        '<p>总体指标按窗口汇总；不同cohort的窗口数可能差别很大，尤其XNA17–20可占较大权重。应同时检查cohort和template结果，不能凭总体均值宣布所有模板都更好。</p><h2>各天然对照与目标模板</h2>')
    (folder/'comparison_report.html').write_text(doc.replace('HEADLINE',''.join(headline)),encoding='utf-8')


def export_figure(fig, folder, name, formats):
    for extension in formats:
        if extension=='pdf':
            from matplotlib.backends.backend_pdf import PdfPages
            buffer=io.BytesIO()
            with PdfPages(buffer) as pages:pages.savefig(fig,bbox_inches='tight',facecolor='white')
            data=buffer.getvalue()
            if not data.rstrip().endswith(b'%%EOF'):raise RuntimeError('PDF未完整写入')
        elif extension=='png':
            buffer=io.BytesIO()
            fig.savefig(buffer,format='png',dpi=300,bbox_inches='tight',facecolor='white')
            data=buffer.getvalue()
            if not data.endswith(b'\x00\x00\x00\x00IEND\xaeB`\x82'):
                raise RuntimeError('PNG未完整写入')
        else:raise ValueError('未知图片格式')
        path=folder/(name+'.'+extension)
        temporary=path.with_name(path.name+'.tmp-'+str(os.getpid()))
        temporary.write_bytes(data)
        temporary.replace(path)


def make_plots(root, formats=('png','pdf')):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    folder=root/'comparison_data';out=root/'figures_comparison';out.mkdir(exist_ok=True)
    rows=pd.read_csv(folder/'metric_values.csv',keep_default_na=False)
    pairs=pd.read_csv(folder/'paired_differences.csv',keep_default_na=False)
    aggregate=pd.read_csv(folder/'metric_aggregate.csv',keep_default_na=False)
    residual=pd.read_csv(folder/'residual_errors.csv',keep_default_na=False)
    for frame in [rows,pairs,aggregate,residual]:
        if 'seed' in frame:frame['seed']=frame.seed.astype(int)
    plt.rcParams.update({'font.family':'DejaVu Sans','font.size':10,'axes.spines.top':False,
        'axes.spines.right':False,'axes.titleweight':'semibold','text.color':'#263e42','pdf.fonttype':42})
    markers={42:'o',2024:'^',2025:'s'}
    fig,axes=plt.subplots(1,2,figsize=(12.8,4.4))
    for i,model in enumerate(MODELS):
        part=rows[(rows.scope=='overall')&(rows.probability_type=='raw')&(rows.model==model)&(rows.metric=='macro_f1')]
        values=pd.to_numeric(part.value)
        for offset,seed in zip([-.15,0,.15],SEEDS):
            value=100*float(part.loc[part.seed==seed,'value'].iloc[0])
            axes[0].scatter(value,i+offset,marker=markers[seed],color=COLORS[model],s=45,zorder=3)
        axes[0].errorbar(100*values.mean(),i+.3,xerr=100*values.std(ddof=1),fmt='D',color=COLORS[model],capsize=3)
    axes[0].set_yticks(range(2),[LABELS[m] for m in MODELS]);axes[0].set_ylim(1.7,-.6)
    axes[0].set_xlabel('External macro-F1 (%)');axes[0].set_title('A  All three fixed seeds',loc='left',pad=14)
    main=[('macro_f1','Macro-F1'),('canonical_false_positive_rate','Canonical false X/Y'),('correct_xy_identity_recall','Correct X/Y identity')]
    for i,(metric,label) in enumerate(main):
        part=pairs[(pairs.scope=='overall')&(pairs.probability_type=='raw')&(pairs.metric==metric)]
        for offset,seed in zip([-.16,0,.16],SEEDS):
            value=float(part.loc[part.seed==seed,'difference'].iloc[0])
            axes[1].scatter(value,i+offset,marker=markers[seed],color='#5989a3',s=45)
        values=pd.to_numeric(part.difference)
        axes[1].errorbar(values.mean(),i+.32,xerr=values.std(ddof=1),fmt='D',color='#205f50',capsize=3)
    axes[1].set_yticks(range(3),[label for _,label in main]);axes[1].set_ylim(2.7,-.6)
    axes[1].axvline(0,color='#819087',ls='--',lw=1)
    axes[1].set_xlabel('Gated minus same-seed full (percentage points)')
    axes[1].set_title('B  Same-seed differences; FPR lower is better',loc='left',pad=14)
    for ax in axes:ax.grid(axis='x',alpha=.15);ax.set_axisbelow(True)
    handles=[plt.Line2D([],[],marker=markers[s],ls='',color='#526b61',label='seed'+str(s)) for s in SEEDS]
    handles.append(plt.Line2D([],[],marker='D',ls='-',color='#526b61',label='mean +/- sample SD'))
    fig.legend(handles=handles,loc='lower center',ncol=4,frameon=False,bbox_to_anchor=(.5,.005),fontsize=9)
    fig.subplots_adjust(left=.14,right=.98,top=.83,bottom=.24,wspace=.6)
    export_figure(fig,out,'01_overall_and_paired_changes',formats);plt.close(fig)
    write_csv(out/'01_overall_and_paired_changes.csv',pairs[pairs.scope=='overall'])
    for number,role,metric,title,filename in [(16,'control','canonical_false_positive_rate','Natural controls: false X/Y calls','02_control_false_calls_by_template'),
            (20,'target','correct_xy_identity_recall','Targets: correct X/Y identity recall','03_target_identity_recall_by_template')]:
        data=rows[(rows.scope=='template_role')&(rows.probability_type=='raw')&(rows.role==role)&(rows.metric==metric)].copy()
        data['template_number']=data.template.str.extract(r'(\d+)').astype(int)
        data=data[data.template_number.between(1,number)]
        fig,ax=plt.subplots(figsize=(12.8,4.3))
        for model,offset in [('litemamba',-.12),('gatedconv_target',.12)]:
            part=data[data.model==model]
            grouped=part.groupby('template_number').value.apply(lambda s:pd.to_numeric(s).mean()*100)
            sd=part.groupby('template_number').value.apply(lambda s:pd.to_numeric(s).std(ddof=1)*100)
            x=grouped.index.to_numpy()+offset
            ax.errorbar(x,grouped.to_numpy(),yerr=sd.to_numpy(),fmt='o-',color=COLORS[model],capsize=2,
                        markersize=4,lw=1.3,label=LABELS[model])
            for seed in SEEDS:
                each=part[part.seed==seed].sort_values('template_number')
                ax.scatter(each.template_number+offset,pd.to_numeric(each.value)*100,marker=markers[seed],
                           s=18,color=COLORS[model],alpha=.45)
        ax.set_xticks(range(1,number+1),[('DNA' if role=='control' else 'XNA')+str(i) for i in range(1,number+1)],rotation=45,ha='right')
        ax.set_ylabel('False X/Y calls (%)' if role=='control' else 'Correct identity recall (%)')
        ax.set_title(title+'; all seeds, mean +/- sample SD',loc='left',pad=13)
        ax.grid(axis='y',alpha=.15);ax.set_axisbelow(True)
        ax.legend(frameon=False,loc='upper center',bbox_to_anchor=(.5,-.24),ncol=2,fontsize=9)
        ax.axvline(12.5,color='#c7d2cc',ls='--',lw=1)
        if number==20:ax.axvline(16.5,color='#c7d2cc',ls='--',lw=1)
        fig.subplots_adjust(left=.075,right=.985,top=.86,bottom=.26)
        fig.text(.5,.025,'Three seeds share the same external windows. Raw argmax; frozen validation temperature is reported separately.',ha='center',fontsize=8.6,color='#647773')
        export_figure(fig,out,filename,formats);plt.close(fig);write_csv(out/(filename+'.csv'),data)
    residual=residual[residual.probability_type=='raw'].copy()
    residual['template_number']=residual.template.str.extract(r'(\d+)').astype(int)
    error_types=['A','T','C','G','X/Y identity swap']
    palette=['#ccd8d1','#9db5a7','#779b8a','#436f5b','#a98aa5']
    fig,ax=plt.subplots(figsize=(11.8,4.6));positions=[];labels=[]
    for i,template in enumerate(range(17,21)):
        for j,model in enumerate(MODELS):
            x=i*2.4+j*.85;positions.append(x);labels.append('XNA'+str(template)+'\n'+('full' if j==0 else 'gated'))
            subset=residual[(residual.template_number==template)&(residual.model==model)]
            bottom=0.
            for error,color in zip(error_types,palette):
                part=subset[subset.error_type==error]
                mean=pd.to_numeric(part.error_per_10000).mean()
                ax.bar(x,mean,bottom=bottom,width=.7,color=color,edgecolor='white',lw=.7,hatch='//' if j==0 else None,
                       label=error if i==0 and j==0 else None)
                bottom+=mean
            total=subset.groupby('seed').error_per_10000.sum()
            ax.errorbar(x,total.mean(),yerr=total.std(ddof=1),fmt='D',color='#263e42',capsize=3,markersize=4,zorder=5)
    ax.set_xticks(positions,labels);ax.set_ylabel('Incorrect calls per 10,000 true X/Y windows')
    ax.set_title('XNA17-20 residual errors: same denominators and all three seeds',loc='left',pad=14)
    ax.grid(axis='y',alpha=.15);ax.set_axisbelow(True);ax.legend(ncol=5,frameon=False,loc='upper center',bbox_to_anchor=(.5,-.16),fontsize=9)
    fig.subplots_adjust(left=.09,right=.985,top=.85,bottom=.27)
    fig.text(.5,.018,'Stack: mean error components; diamond +/- bar: total error mean +/- sample SD. Hatched bars denote full.',ha='center',fontsize=8.7,color='#647773')
    export_figure(fig,out,'04_xna17_20_residual_errors',formats);plt.close(fig)
    write_csv(out/'04_xna17_20_residual_errors.csv',residual)
    save_json(out/'comparison_plot_manifest.json',dict(status='complete',formats=list(formats),
        comparison_manifest_sha256=sha256(folder/'comparison_manifest.json'),plot_code_sha256=sha256(__file__),
        figures=['01_overall_and_paired_changes','02_control_false_calls_by_template','03_target_identity_recall_by_template','04_xna17_20_residual_errors']))


def package_results(root):
    plan=read_json(root/'evaluation_plan.json')
    manifest=read_json(root/'comparison_data/comparison_manifest.json')
    if manifest.get('status')!='complete' or manifest.get('plan_id')!=plan.get('plan_id'):
        raise ValueError('结构比较尚未完成或计划不一致')
    files={}
    def add(path,name=None):
        path=Path(path)
        if path.is_file():
            archive_name=name or path.relative_to(root).as_posix()
            if Path(archive_name).is_absolute() or '..' in Path(archive_name).parts:raise ValueError('非法归档路径')
            files[archive_name]=path
    for name in ['evaluation_plan.json','reuse_manifest.json','comparison_registry.json','external_window_design_audit.json',
        'external_identity_audit.json','corrected_split_audit.json','reference_correction_provenance.json',
        'comparison_provenance.json','comparison_source_provenance.json','queue_state.json','commands.jsonl','resource_events.jsonl','discovered_inputs.csv']:
        add(root/name)
    for item in plan.get('comparison_provenance',{}).get('source_metadata_files',[]):
        relative=Path(item['local_copy'])
        if relative.is_absolute() or '..' in relative.parts:raise ValueError('非法来源副本路径')
        path=root/relative
        if not path.is_file() or sha256(path)!=item['sha256']:
            raise ValueError('旧参考修正/确认来源副本缺失或内容改变：'+str(path))
    for folder in ['analysis_data','comparison_data','figures_comparison','figures','source_provenance','source_evaluation_metadata']:
        for path in sorted((root/folder).rglob('*')):
            if path.is_file() and path.suffix.lower() not in ['.sqlite','.db']:
                add(path)
    dataset_ids={d['dataset_id'] for d in plan['datasets']}
    for run in plan['runs']:
        run_id=run['run_id'];base=root/'predictions'/run_id
        completed=read_json(base/'run_complete.json')
        if completed.get('status')!='complete' or completed.get('plan_id')!=plan['plan_id'] or set(completed.get('datasets',[]))!=dataset_ids:
            raise ValueError('运行完成记录与完整数据集合不同：'+run_id)
        add(base/'run_complete.json')
        for dataset in plan['datasets']:
            marker=base/dataset['dataset_id']/'completion.json';value=read_json(marker)
            for key,expected in [('status','complete'),('plan_id',plan['plan_id']),('run_id',run_id),('dataset_id',dataset['dataset_id']),('n_samples',dataset['n_samples'])]:
                if value.get(key)!=expected:raise ValueError('逐文件完成记录不同：'+str(marker)+'/'+key)
            add(marker)
        training=Path(run['run_dir'])
        for name in ['queue_requested_config.json','resolved_config.json','calibration.json','model_config.json',
                     'results.json','data_audit.json','environment.json','epoch_metrics.jsonl','metrics.csv']:
            add(training/name,'training_run_metadata/'+run_id+'/'+name)
    for path in sorted((root/'resource_profiles').rglob('*.json')):add(path)
    for path in sorted((root/'logs').rglob('invocation.json')):add(path)
    records={}
    for name,path in sorted(files.items()):
        before=path.stat()
        # 不把意外的大型逐窗口文件当成元数据打包。
        if name.startswith('training_run_metadata/') and before.st_size>64*1024*1024:
            raise ValueError('训练元数据异常过大：'+str(path))
        digest=sha256(path);after=path.stat()
        if (before.st_size,before.st_mtime_ns)!=(after.st_size,after.st_mtime_ns):raise ValueError('打包时文件变化')
        records[name]=dict(size_bytes=after.st_size,mtime_ns=after.st_mtime_ns,sha256=digest)
    package_manifest=dict(status='complete',package_version=VERSION,plan_id=plan['plan_id'],
        comparison='fixed full versus gated convolution; seeds42/2024/2025; no training or external fitting',
        included='All standard tables, paired comparison tables, figures, source/reuse/window audits, completion records and small training metadata.',
        excluded='Per-window predictions/arrays, feature cache, audit databases, checkpoints and training inputs remain on server.',
        files=records)
    destination=root/PACKAGE_NAME;temporary=destination.with_name(destination.name+'.partial-'+str(os.getpid()))
    try:
        with tarfile.open(temporary,'w:gz',compresslevel=5,dereference=True) as archive:
            for name,path in sorted(files.items()):
                before=path.stat();record=records[name]
                if (before.st_size,before.st_mtime_ns)!=(record['size_bytes'],record['mtime_ns']):raise ValueError('打包前文件变化')
                archive.add(path,arcname=name,recursive=False)
                after=path.stat()
                if (before.st_size,before.st_mtime_ns)!=(after.st_size,after.st_mtime_ns):raise ValueError('打包期间文件变化')
            data=(json.dumps(package_manifest,ensure_ascii=False,indent=2,allow_nan=False)+'\n').encode()
            item=tarfile.TarInfo('RESULT_PACKAGE_MANIFEST.json');item.size=len(data)
            archive.addfile(item,io.BytesIO(data))
        temporary.replace(destination)
    except BaseException:
        temporary.unlink(missing_ok=True);raise
    print('[RESULT PACKAGE] '+str(destination),flush=True)
    return destination


def summarize(output_root, no_plots=False, plot_only=False, formats=('png','pdf'), package=True):
    root=Path(output_root).expanduser().resolve()
    plan,analysis,lookup=validate_plan(root)
    tables,fingerprints=read_standard_tables(root,lookup)
    destination=root/'comparison_data'
    if plot_only:
        saved=read_json(destination/'comparison_manifest.json')
        if saved.get('plan_id')!=plan['plan_id'] or saved.get('source_sha256')!=fingerprints:
            raise ValueError('已有comparison_data与当前计划/标准表不同；先完整analyze后再重绘')
        if saved.get('status')!='complete':raise ValueError('比较表未完整完成')
    else:
        matrices=read_confusions(root,lookup)
        values,residual=derive_rows(tables,matrices)
        aggregate=aggregate_values(values);pairs,paired=paired_values(values)
        classes,class_pairs,class_aggregate=read_per_class(root,lookup)
        with tempfile.TemporaryDirectory(prefix='.gated_comparison_',dir=root) as temporary:
            stage=Path(temporary)/'comparison_data';stage.mkdir()
            for name,frame in [('metric_values.csv',values),('metric_aggregate.csv',aggregate),
                ('paired_differences.csv',pairs),('paired_aggregate.csv',paired),('residual_errors.csv',residual),
                ('class_values.csv',classes),('class_paired_differences.csv',class_pairs),('class_paired_aggregate.csv',class_aggregate)]:
                write_csv(stage/name,frame)
            for scope in TABLES:
                write_csv(stage/(scope+'_paired_differences.csv'),pairs[pairs.scope==scope])
                write_csv(stage/(scope+'_paired_aggregate.csv'),paired[paired.scope==scope])
            render_report(stage,aggregate,paired)
            save_json(stage/'comparison_manifest.json',dict(status='complete',version=VERSION,plan_id=plan['plan_id'],
                seeds=list(SEEDS),models=list(MODELS),source_sha256=fingerprints,code_sha256=sha256(__file__),
                metric_scope='Raw argmax primary; calibrated probabilities kept separate, temperature fitted only on training validation.',
                difference_direction='gated minus same-seed full',uncertainty='sample SD across3 fixed training seeds; no sample-level significance test',
                external_evaluation='follow-up comparison on already observed external data; not a new blind test',
                legacy_figure4_abc_tables='full-only compatibility outputs; use comparison_data for two-structure comparisons',
                counts=dict(runs=6,metric_rows=len(values),paired_rows=len(pairs),paired_aggregate_rows=len(paired)),
                created_at=datetime.now(timezone.utc).isoformat()))
            old=Path(temporary)/'old_comparison'
            if destination.exists():destination.rename(old)
            try:stage.rename(destination)
            except BaseException:
                if old.exists():old.rename(destination)
                raise
    if not no_plots:make_plots(root,formats=formats)
    result=package_results(root) if package else destination
    print('[COMPARISON COMPLETE] '+str(destination/'comparison_report.html'),flush=True)
    return result


def main(argv=None):
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output_root',required=True,type=Path)
    parser.add_argument('--phase',choices=('all','plot'),default='all')
    parser.add_argument('--plot_only',action='store_true')
    parser.add_argument('--no_plots',action='store_true')
    parser.add_argument('--formats',nargs='+',choices=('png','pdf'),default=['png','pdf'])
    args=parser.parse_args(argv)
    if args.no_plots and (args.plot_only or args.phase=='plot'):parser.error('plot模式不能同时no_plots')
    summarize(args.output_root,no_plots=args.no_plots,plot_only=args.plot_only or args.phase=='plot',formats=tuple(args.formats))
    return 0


if __name__=='__main__':
    try:raise SystemExit(main())
    except Exception as error:
        print('[ERROR] '+type(error).__name__+': '+str(error),file=sys.stderr)
        raise SystemExit(1)
