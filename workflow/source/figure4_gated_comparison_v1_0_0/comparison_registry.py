#!/usr/bin/env python3
"""只读注册固定三种子的full和门控卷积检查点，不导入PyTorch。"""
from __future__ import annotations

import math
from pathlib import Path

from figure4_common import CLASSES, HERE, identity_hash, read_json, sha256

SEEDS = (42, 2024, 2025)
EXPERIMENTS = ('full', 'gatedconv_target')
SOURCE_MANIFEST = HERE / 'COMPARISON_TRAINING_SOURCE_MANIFEST.json'
SCIENTIFIC_KEYS = (
    'epochs', 'patience', 'lr', 'batch_size', 'raw_signal_len', 'dwell_divisor',
    'quality_divisor', 'accum_steps', 'weight_decay', 'label_smoothing', 'clip_grad',
    'monitor', 'model_dim', 'num_blocks', 'state_dim', 'bootstrap', 'no_plots',
    'no_embeddings', 'embedding_dtype', 'no_epoch_logits', 'save_epoch_predictions',
    'staged_exports',
)
REQUIRED_METADATA = (
    'queue_requested_config.json', 'resolved_config.json', 'model_config.json',
    'environment.json', 'data_audit.json', 'calibration.json', 'epoch_metrics.jsonl',
    'val/analysis/metrics.json',
)


def _fail(condition, message):
    if not condition:
        raise ValueError(message)


def _content(fingerprints):
    return {split: {kind: {'sha256': fingerprints[split][kind]['sha256'],
                         'size': fingerprints[split][kind]['size']}
                   for kind in ('feature', 'metadata')} for split in ('train', 'val', 'test')}


def _class_names(value):
    if isinstance(value, dict):
        return [value.get(str(i), value.get(i)) for i in range(6)]
    return value


def _selection(run, reference):
    """复核原val_loss所选轮次，不读取full的测试结果。"""
    import json
    history = [json.loads(line) for line in (run/'epoch_metrics.jsonl').read_text(encoding='utf-8').splitlines() if line.strip()]
    _fail(bool(history) and [item['epoch'] for item in history] == list(range(1, len(history)+1)),
          '训练历史缺失或轮次不连续：' + str(run))
    for item in history:
        _fail(item.get('monitor', 'val_loss') == 'val_loss' and math.isfinite(float(item['val']['loss'])),
              '验证选模规则或损失无效：' + str(run))
    best = min(history, key=lambda item: item['val']['loss'])
    metrics = read_json(run/'val/analysis/metrics.json')
    raw = metrics['overall']['raw']
    matrix = raw['confusion_matrix']
    _fail(len(matrix) == 6 and all(len(row) == 6 for row in matrix), '混淆矩阵不是6×6')
    _fail(all(type(v) is int and v >= 0 for row in matrix for v in row), '混淆计数不是非负整数')
    support = [sum(row) for row in matrix]
    predicted = [sum(row[i] for row in matrix) for i in range(6)]
    _fail(all(v > 0 for v in support), '验证集缺少类别')
    macro = sum(2*matrix[i][i]/(support[i]+predicted[i]) for i in range(6))/6
    _fail(_class_names(metrics.get('class_definition')) == CLASSES, '验证指标类别定义不同')
    _fail(sum(support) == raw['n'] and abs(raw['macro_f1']-macro) < 1e-9,
          '验证指标与混淆计数不符')
    _fail(best['epoch'] == reference['selected_epoch'] and
          abs(float(best['val']['macro_f1'])-macro) < 1e-9 and
          abs(float(reference['val_macro_f1'])-macro) < 1e-9,
          '验证指标不是冻结的val_loss检查点：' + str(run))
    return best['epoch']


def _relative(path, root):
    try:
        relative = Path(path).relative_to(Path(root))
    except ValueError as error:
        raise ValueError('冻结检查点或任务目录不位于记录的运行目录内') from error
    _fail(not relative.is_absolute() and '..' not in relative.parts, '冻结相对路径非法')
    return relative


def _metadata_hashes(comparison, experiment, seed, directory):
    prefix = f'source_metadata/{experiment}/seed_{seed}/'
    frozen = {}
    for item in comparison.get('source_files', []):
        packaged = item.get('package_path', '')
        if not packaged.startswith(prefix):
            continue
        relative = packaged[len(prefix):]
        _fail(relative and not Path(relative).is_absolute() and '..' not in Path(relative).parts,
              '比较记录中的元数据路径非法')
        _fail(relative not in frozen, '比较记录中的元数据重复')
        target = directory/relative
        _fail(target.is_file() and sha256(target) == item['sha256'] and target.stat().st_size == item['size_bytes'],
              '归档来源与原运行文件不同：' + str(target))
        frozen[relative] = item['sha256']
    _fail(set(REQUIRED_METADATA).issubset(frozen), '冻结比较缺少原运行元数据：' + str(directory))
    if experiment != 'full':
        _fail('results.json' in frozen, '门控运行缺少冻结完成记录')
    return frozen


def _verify_source(snapshot, expected, label):
    _fail(all(snapshot.get(name) == value for name, value in expected.items()), '训练源码快照不同：' + label)


def discover_comparison_runs(confirmation_dir, baseline_root=None):
    """返回旧Figure4兼容的(root,runs,preprocessing)，固定注册6个模型。"""
    confirmation = Path(confirmation_dir).expanduser().resolve()
    plan_path = confirmation/'structure_plan.json'
    comparison_path = confirmation/'structure_summary/structure_comparison.json'
    plan, comparison = read_json(plan_path), read_json(comparison_path)
    expected = read_json(SOURCE_MANIFEST)
    _fail(plan.get('version') == '1.1.0' and plan.get('validation_only') is True and
          plan.get('seeds') == list(SEEDS) and plan.get('new_seeds') == [2024, 2025],
          '只接受已完成的固定3种子结构确认计划')
    _fail(comparison.get('status') == 'complete' and comparison.get('seeds') == list(SEEDS) and
          comparison.get('task_count') == 9 and comparison.get('test_inference') is False,
          '结构确认汇总未完整完成或种子集合不同')
    for key in ('source_sha256', 'prior_patched_source_sha256', 'patched_source_sha256'):
        manifest_key = 'original_source_sha256' if key == 'source_sha256' else key
        _fail(plan.get(key) == expected[manifest_key], '结构确认源码与受支持版本不同：' + key)
        comparison_key = 'original_source_sha256' if key == 'source_sha256' else key
        _fail(comparison.get(comparison_key) == plan[key], '结构确认计划与汇总源码不同：' + key)
    _fail(expected['prior_patched_source_sha256']['model_benchmark_all.py'] == expected['patched_source_sha256']['model_benchmark_all.py'],
          '两轮门控卷积的模型源码不同，不能合并比较')
    baselines = plan.get('baselines', [])
    _fail(len(baselines) == 3 and {(r.get('experiment'), r.get('seed')) for r in baselines} == {('full', s) for s in SEEDS},
          '必须完整包含三个full基线')
    old_sources, new_sources = plan.get('reused_controls', []), plan.get('tasks', [])
    controls = ('gatedconv_target', 'ssm_global_only')
    _fail(len(old_sources) == 2 and {(r.get('experiment'), r.get('seed')) for r in old_sources} == {(e, 42) for e in controls},
          '上轮结构来源身份不完整')
    _fail(len(new_sources) == 4 and {(r.get('experiment'), r.get('seed')) for r in new_sources} == {(e, s) for e in controls for s in SEEDS[1:]},
          '新增结构来源身份不完整')
    old = [r for r in old_sources if r.get('experiment') == 'gatedconv_target']
    new = [r for r in new_sources if r.get('experiment') == 'gatedconv_target']
    _fail(len(old) == 1 and old[0].get('seed') == 42 and len(new) == 2 and {r.get('seed') for r in new} == {2024, 2025},
          '门控来源必须是旧seed42和新增2024/2025')
    entries = comparison.get('per_run', [])
    _fail(len(entries) == 9 and {(r['experiment'], r['seed']) for r in entries} ==
          {(e, s) for e in EXPERIMENTS+('ssm_global_only',) for s in SEEDS},
          '结构确认汇总缺失或重复运行身份')
    lookup = {(row['experiment'], row['seed']): row for row in entries}
    sources = {(r['experiment'], r['seed']): r for r in baselines+old+new}
    root = Path(baseline_root).expanduser().resolve() if baseline_root else Path(plan['baseline_root']).expanduser().resolve()
    baseline_reference = read_json(root/'seed_42/queue_requested_config.json')
    resolved_reference = read_json(root/'seed_42/resolved_config.json')
    kwargs_reference = resolved_reference['model_kwargs']
    selection_sources = {str(plan_path): sha256(plan_path), str(comparison_path): sha256(comparison_path),
                         str(SOURCE_MANIFEST.resolve()): sha256(SOURCE_MANIFEST)}
    runs, preprocessing = [], None
    for experiment in EXPERIMENTS:
        for seed in SEEDS:
            source, reference = sources[experiment, seed], lookup[experiment, seed]
            if experiment == 'full':
                directory = root/('seed_'+str(seed))
                checkpoint = directory/_relative(source['checkpoint'], source['run_dir'])
            elif seed == 42:
                directory = Path(source['run_dir']).expanduser().resolve()
                checkpoint = Path(source['checkpoint']).expanduser().resolve()
            else:
                directory = confirmation/_relative(source['run_dir'], plan['output_dir'])
                checkpoint = directory/'best_checkpoint.pt'
            _fail(directory.is_dir(), '运行目录不存在：' + str(directory))
            metadata = _metadata_hashes(comparison, experiment, seed, directory)
            for name, value in source.get('file_sha256', {}).items():
                _fail(metadata.get(name) == value, '结构计划与归档元数据不同：' + name)
            config_path, calibration_path = directory/'resolved_config.json', directory/'calibration.json'
            requested = read_json(directory/'queue_requested_config.json')
            config, model_config, calibration = read_json(config_path), read_json(directory/'model_config.json'), read_json(calibration_path)
            variant = 'selective_ssm' if experiment == 'full' else 'legacy_gatedconv'
            source_key = 'source_sha256' if experiment == 'full' else 'prior_patched_source_sha256' if seed == 42 else 'patched_source_sha256'
            source_hashes = plan[source_key]
            for current in (requested, config):
                _fail((current.get('model_type'), current.get('seed'), current.get('feature_scheme')) == ('litemamba', seed, 'full'),
                      '模型/种子/全量特征身份不同：' + str(directory))
                _fail(current.get('litemamba_variant') == variant and current.get('readout') == 'target_attention' and
                      current.get('monitor') == 'val_loss' and not current.get('external_manifest'),
                      '模型结构或选模规则与预设不同：' + str(directory))
                _fail(_content(current['input_fingerprints']) == _content(plan['expected_input_fingerprints']),
                      '训练数据内容与结构确认计划不同')
            for key in SCIENTIFIC_KEYS:
                _fail(requested.get(key) == baseline_reference.get(key), '训练科学配置不同：' + key)
                if key != 'staged_exports':
                    _fail(config.get(key) == requested.get(key), '请求与实际配置不同：' + key)
            _fail(requested.get('source_sha256') == source_hashes, '请求训练源码版本不同')
            _verify_source(config.get('source_snapshot', {}).get('files_sha256', {}), source_hashes, str(directory))
            if experiment != 'full':
                _fail(requested.get('structure_experiment') == experiment, '门控结构别名缺失或不同')
            _fail(_class_names(config.get('class_names')) == CLASSES and _class_names(model_config.get('class_names')) == CLASSES,
                  '类别顺序必须为A,T,C,G,X,Y')
            _fail(config.get('target_index') == 3 and model_config.get('target_index') == 3 and
                  model_config.get('model_type') == 'litemamba' and
                  model_config.get('seq_len') == 6 and model_config.get('num_classes') == 6 and
                  model_config.get('readout') == 'target_attention' and model_config.get('litemamba_variant') == variant,
                  '只支持第4位直接读出的新版六分类结构')
            _fail(model_config.get('one_hot_base_order') == ['A', 'C', 'T', 'G', 'N'], 'one-hot碱基顺序不同')
            kwargs = dict(config['model_kwargs'])
            _fail(set(kwargs) == set(kwargs_reference) and all(kwargs[k] == kwargs_reference[k]
                  for k in kwargs if k != 'litemamba_variant') and kwargs.get('litemamba_variant') == variant,
                  '除预设骨干外的模型构造参数发生变化')
            for key in ('model_dim', 'num_blocks', 'state_dim', 'target_index', 'dropout',
                        'classifier_dropout', 'feature_scheme', 'readout', 'litemamba_variant', 'ssm_bidirectional'):
                _fail(key in kwargs and model_config.get(key) == kwargs[key], '模型记录与构造参数不同：' + key)
            _fail(model_config.get('raw_signal_len') == config['raw_signal_len'], '模型信号长度与实际配置不同')
            _fail(config.get('amp') == resolved_reference.get('amp'), '训练AMP策略不同')
            current = {'seq_len': 6, 'raw_signal_len': int(config['raw_signal_len']),
                'dwell_divisor': float(config['dwell_divisor']), 'quality_divisor': float(config['quality_divisor']),
                'mask_index': 3, 'mask_base': 'N', 'one_hot_base_order': ['A', 'C', 'T', 'G', 'N'],
                'class_names': CLASSES, 'signal_rectification': 'symmetric_zero_pad_or_floor_linspace_indices',
                'signal_round_decimals': 4}
            _fail(preprocessing is None or preprocessing == current, '模型之间的输入预处理不同')
            preprocessing = current
            digest = sha256(checkpoint)
            _fail(digest == reference['checkpoint_sha256'] == calibration.get('checkpoint_sha256'),
                  '实际checkpoint与确认汇总/温度指纹不同：' + str(directory))
            if source.get('checkpoint_sha256'):
                _fail(digest == source['checkpoint_sha256'], '实际checkpoint与确认计划不同')
            _fail(calibration.get('fitted_on') == 'val' and math.isfinite(float(calibration.get('temperature', 0))) and
                  float(calibration.get('temperature', 0)) > 0, '温度不是仅在val拟合的正数')
            selected = _selection(directory, reference)
            if experiment != 'full':
                result = read_json(directory/'results.json')
                _fail(result.get('status') == 'train_complete' and result.get('test_evaluated') is False and
                      result.get('best_epoch') == selected, '门控训练或验证导出尚未完成')
                _fail((result.get('model_type'), result.get('seed'), result.get('feature_scheme'), result.get('litemamba_variant'), result.get('readout')) ==
                      ('litemamba', seed, 'full', variant, 'target_attention'), '门控完成记录身份不同')
                _fail(set(result.get('exports_by_split', {})) == {'train', 'val'} and
                      set(result.get('metrics_by_split', {})) == {'train', 'val'}, '门控训练结果包含其他split或缺失train/val')
                _fail(result['provenance'].get('checkpoint_sha256') == digest and
                      _content(result['provenance']['input_fingerprints']) == _content(plan['expected_input_fingerprints']),
                      '门控完成记录来源不同')
                _verify_source(result['provenance'].get('source_snapshot', {}).get('files_sha256', {}), source_hashes, str(directory))
                _fail(all(result['exports_by_split'][split].get('checkpoint_sha256') == digest for split in ('train', 'val')),
                      '门控导出与检查点不同')
                _fail(not any((directory/name).exists() for name in ('test_started.json', 'test_evaluated.json')) and
                      not ((directory/'test').exists() and any((directory/'test').rglob('*'))), '门控训练目录出现test产物')
            runtime = HERE/('vendor_full' if experiment == 'full' else 'vendor')
            runtime_hashes = {name: sha256(runtime/name) for name in ('model_benchmark_all.py', 'IterDataset.py')}
            _fail(all(source_hashes.get(name) == value for name, value in runtime_hashes.items()),
                  '评估运行时与训练模型/数据源码不匹配：' + experiment)
            alias = 'litemamba' if experiment == 'full' else 'gatedconv_target'
            profile_key = identity_hash({'model': alias, 'kwargs': kwargs, 'preprocessing': preprocessing})[:20]
            runs.append({'run_id': f'full/{alias}/seed_{seed}', 'model': alias, 'actual_model_type': 'litemamba',
                'structure_experiment': experiment, 'seed': seed, 'feature_scheme': 'full', 'profile_key': profile_key,
                'run_dir': str(directory), 'checkpoint': str(checkpoint), 'checkpoint_sha256': digest,
                'resolved_config': str(config_path), 'resolved_config_sha256': sha256(config_path),
                'calibration': str(calibration_path), 'calibration_sha256': sha256(calibration_path),
                'temperature': float(calibration['temperature']), 'model_kwargs': kwargs,
                'runtime_dir': str(runtime.resolve()), 'runtime_sha256': runtime_hashes,
                'training_input_fingerprints': config['input_fingerprints'],
                'training_source_sha256': source_hashes, 'frozen_metadata_sha256': metadata,
                'selection_sources': selection_sources, 'checkpoint_selected_epoch': selected})
    _fail(len({r['run_id'] for r in runs}) == 6 and len({r['run_dir'] for r in runs}) == 6, '模型注册身份或目录重复')
    return root.parent.parent, runs, preprocessing
