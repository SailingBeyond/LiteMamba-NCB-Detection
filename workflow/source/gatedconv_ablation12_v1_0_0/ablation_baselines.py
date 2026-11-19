#!/usr/bin/env python3
"""Read-only identity checks for the three frozen final GatedConv full baselines.

No checkpoint is loaded and no inference is performed. The optional archive-only
mode verifies recorded identities when weight bytes are unavailable locally.
"""
from __future__ import annotations
from contextlib import contextmanager, ExitStack
import fcntl
import hashlib
import json
import math
from pathlib import Path

SEEDS = (42, 2024, 2025)
CLASSES = ['A', 'T', 'C', 'G', 'X', 'Y']
BASE_ORDER = ['A', 'C', 'T', 'G', 'N']
REQUIRED_TRAIN_METADATA = ('queue_requested_config.json', 'resolved_config.json',
    'model_config.json', 'results.json', 'calibration.json', 'epoch_metrics.jsonl',
    'source_snapshot/manifest.json', 'environment.json', 'data_audit.json',
    'val/export_complete.json', 'val/analysis/metrics.json')
REQUIRED_TEST_METADATA = ('run_complete.json', 'test/export_complete.json', 'test/analysis/metrics.json')


def _fail(condition, message):
    if not condition:
        raise ValueError(message)


def read_json(path):
    return json.loads(Path(path).read_text(encoding='utf-8'))


def sha256(path):
    value = hashlib.sha256()
    with Path(path).open('rb') as stream:
        for chunk in iter(lambda: stream.read(8 * 1024 * 1024), b''):
            value.update(chunk)
    return value.hexdigest()


def _classes(value):
    return [value.get(str(i), value.get(i)) for i in range(6)] if isinstance(value, dict) else value


def _content(pair):
    return {kind: {'sha256': pair[kind]['sha256'],
                   'size': pair[kind].get('size', pair[kind].get('size_bytes'))}
            for kind in ('feature', 'metadata')}


def _same_inputs(actual, expected):
    try:
        return all(_content(actual[s]) == _content(expected[s]) for s in ('train', 'val', 'test'))
    except (KeyError, TypeError):
        return False


def _metrics(document, expected_n, where):
    _fail(_classes(document.get('class_definition')) == CLASSES,
          '完整输入指标类别顺序不同：' + where)
    _fail(document.get('row_count') == expected_n, '完整输入指标行数不同：' + where)
    variants = document.get('overall', {})
    for variant in ('raw', 'calibrated'):
        item = variants.get(variant, {})
        matrix = item.get('confusion_matrix')
        _fail(isinstance(matrix, list) and len(matrix) == 6 and
              all(isinstance(row, list) and len(row) == 6 for row in matrix),
              '完整输入混淆矩阵缺失或不是6×6：' + where)
        _fail(all(type(v) is int and v >= 0 for row in matrix for v in row),
              '完整输入混淆矩阵存在非法计数：' + where)
        support = [sum(row) for row in matrix]
        predicted = [sum(row[i] for row in matrix) for i in range(6)]
        correct = sum(matrix[i][i] for i in range(6))
        macro = sum(2 * matrix[i][i] / (support[i] + predicted[i])
                    if support[i] + predicted[i] else 0. for i in range(6)) / 6
        _fail(sum(support) == expected_n == item.get('n') and all(v > 0 for v in support),
              '完整输入指标支持数不同或缺类：' + where)
        _fail(item.get('correct') == correct and
              math.isclose(float(item.get('accuracy', -1)), correct / expected_n, abs_tol=1e-10) and
              math.isclose(float(item.get('macro_f1', -1)), macro, abs_tol=1e-10),
              '完整输入指标与整数混淆矩阵不一致：' + where)
    _fail(variants['raw']['confusion_matrix'] == variants['calibrated']['confusion_matrix'],
          '正温度前后完整输入argmax不同：' + where)
    return variants


def _validate_export(marker, split, n, checkpoint_sha256, temperature, reference):
    _fail(marker.get('split') == split and marker.get('n_samples') == n and
          marker.get('checkpoint_sha256') == checkpoint_sha256 and marker.get('temperature') == temperature,
          '完整输入导出记录的split/行数/权重/温度不同：' + split)
    identity = marker.get('dataset_identity', {})
    _fail(_content(identity.get('input_fingerprints', {})) == _content(reference['input_fingerprints'][split]),
          '完整输入导出不是固定内部' + split + '数据')
    pre = identity.get('preprocessing', {})
    expected = {'seq_len': 6, 'mask_index': 3, 'mask_base': 'N',
                'raw_signal_len': reference['scientific_configuration']['raw_signal_len'],
                'dwell_divisor': reference['scientific_configuration']['dwell_divisor'],
                'quality_divisor': reference['scientific_configuration']['quality_divisor']}
    _fail(all(pre.get(k) == v for k, v in expected.items()) and
          pre.get('base_order') == BASE_ORDER and _classes(pre.get('class_names')) == CLASSES,
          '完整输入导出预处理或第4位N规则不同：' + split)
    return marker.get('metrics', {})


def _choose_run(reference_record, baseline_dir, plan):
    seed = reference_record['seed']
    suffix = Path('full') / 'gatedconv_target' / ('seed_' + str(seed))
    candidates = [baseline_dir / suffix]
    rows = plan.get('reused_controls', []) + plan.get('tasks', [])
    for item in rows:
        if item.get('experiment') != 'gatedconv_target' or item.get('seed') != seed or not item.get('run_dir'):
            continue
        recorded = Path(item['run_dir']).expanduser()
        if plan.get('output_dir'):
            try:
                candidates.append(baseline_dir / recorded.relative_to(Path(plan['output_dir'])))
            except ValueError:
                pass
        candidates.append(recorded)
    original = Path(reference_record['run_dir']).expanduser()
    # Seed42 may live in the earlier sibling structural experiment.
    if len(original.parents) >= 3:
        candidates.append(baseline_dir.parent / original.parents[2].name / suffix)
    candidates.append(original)
    for candidate in candidates:
        if candidate.is_dir():
            return candidate.resolve()
    raise FileNotFoundError('缺少最终GatedConv完整输入 seed_' + str(seed) + '；不能用SSM基线替代')


@contextmanager
def baseline_read_locks(baseline_dir, baseline_test_dir):
    """Reuse only when the known structure and test entrypoints are inactive."""
    locks = [(Path(baseline_dir) / 'structure_wrapper.lock', '结构确认队列'),
             (Path(baseline_test_dir) / 'evaluation.lock', '已有内部测试队列')]
    with ExitStack() as stack:
        for path, label in locks:
            if not path.is_file():
                continue
            handle = stack.enter_context(path.open('r', encoding='utf-8'))
            try:
                fcntl.flock(handle.fileno(), fcntl.LOCK_SH | fcntl.LOCK_NB)
            except BlockingIOError as exc:
                raise RuntimeError(label + '仍在运行，不能同时复用其结果：' + str(path)) from exc
        yield


def verify_baselines(reference, baseline_dir, baseline_test_dir, require_checkpoints=True):
    """Return three full GatedConv baseline records; never silently omit a seed.

    `require_checkpoints=False` is only for auditing an existing compact archive;
    existing checkpoint bytes, if present, are still checked. The training launch
    uses the default True, so all3 best checkpoint bytes must exist and match.
    """
    baseline_dir = Path(baseline_dir).expanduser().resolve()
    baseline_test_dir = Path(baseline_test_dir).expanduser().resolve()
    records = reference.get('baseline_records', [])
    _fail(reference.get('seeds') == list(SEEDS) and len(records) == 3 and
          {r.get('seed') for r in records} == set(SEEDS), '完整输入基线须保留预设三个种子')
    plan_path = baseline_dir / 'structure_plan.json'
    plan = read_json(plan_path) if plan_path.is_file() else {}
    output = []
    with baseline_read_locks(baseline_dir, baseline_test_dir):
        for record in sorted(records, key=lambda x: SEEDS.index(x['seed'])):
            seed = record['seed']
            run = _choose_run(record, baseline_dir, plan)
            test_run = baseline_test_dir / 'runs' / 'gatedconv_target' / ('seed_' + str(seed))
            for name in REQUIRED_TRAIN_METADATA:
                if not (run / name).is_file():
                    raise FileNotFoundError('完整输入基线核心记录缺失：' + str(run / name))
            for name in REQUIRED_TEST_METADATA:
                if not (test_run / name).is_file():
                    raise FileNotFoundError('完整输入内部独立test缺失：' + str(test_run / name))
            metadata = [run / name for name in REQUIRED_TRAIN_METADATA] + [test_run / name for name in REQUIRED_TEST_METADATA]
            before_hashes = {str(path): sha256(path) for path in metadata}
            config, requested, model_config, result, calibration = [read_json(run / name) for name in
                ('resolved_config.json', 'queue_requested_config.json', 'model_config.json', 'results.json', 'calibration.json')]
            _fail(sha256(run / 'model_config.json') == record.get('model_config_sha256'),
                  '完整输入模型配置SHA不同：' + str(run))
            for current in (config, requested, result):
                _fail((current.get('model_type'), current.get('seed'), current.get('feature_scheme')) == ('litemamba', seed, 'full') and
                      current.get('litemamba_variant') == 'legacy_gatedconv' and current.get('readout') == 'target_attention',
                      '不是最终GatedConv full，或模型/种子/方案不同：' + str(run))
            _fail(_classes(config.get('class_names')) == CLASSES and _classes(model_config.get('class_names')) == CLASSES and
                  model_config.get('one_hot_base_order') == BASE_ORDER and model_config.get('seq_len') == 6 and
                  model_config.get('num_classes') == 6 and config.get('target_index') == model_config.get('target_index') == 3,
                  '完整输入类别顺序、6mer或第4位目标不同：' + str(run))
            science = reference['scientific_configuration']
            for key, expected in science.items():
                if key == 'staged_exports':
                    continue  # queue scheduling field; resolved checkpoints do not store it.
                _fail(config.get(key) == expected and requested.get(key) == expected,
                      '完整输入训练科学设置不同：' + key)
            _fail(config.get('checkpoint') is None and config.get('monitor') == 'val_loss' and
                  not config.get('external_manifest'), '完整输入初始化/选模不是固定validation loss')
            kwargs = config.get('model_kwargs', {})
            for key, expected in {**reference['model_defaults'], **{k: science[k] for k in
                    ('model_dim', 'num_blocks', 'state_dim', 'litemamba_variant', 'readout')}, 'feature_scheme': 'full'}.items():
                _fail(kwargs.get(key) == expected, '完整输入网络参数不同：' + key)
            for current in (config, requested):
                _fail(_same_inputs(current.get('input_fingerprints', {}), reference['input_fingerprints']),
                      '完整输入训练/验证/test输入内容不同：' + str(run))
            source = config.get('source_snapshot', {}).get('files_sha256', {})
            expected_source = record.get('source_sha256', {})
            manifest_source = read_json(run / 'source_snapshot/manifest.json').get('files_sha256', {})
            _fail(bool(expected_source) and source == expected_source == manifest_source,
                  '完整输入实际训练源码谱系不符：' + str(run))
            for name, expected in reference['frozen_sources'].items():
                _fail(source.get(name) == expected, '完整输入与新实验的模型/预处理源码不同：' + name)
            checkpoint = run / 'best_checkpoint.pt'
            expected_checkpoint = record['checkpoint_sha256']
            if require_checkpoints or checkpoint.is_file():
                _fail(checkpoint.is_file() and sha256(checkpoint) == expected_checkpoint,
                      '完整输入检查点缺失或SHA不同：' + str(checkpoint))
            _fail(result.get('status') in ('train_complete', 'complete') and
                  result.get('provenance', {}).get('checkpoint_sha256') == expected_checkpoint and
                  _same_inputs(result.get('provenance', {}).get('input_fingerprints', {}), reference['input_fingerprints']),
                  '完整输入训练完成记录或检查点谱系不同：' + str(run))
            temperature = float(calibration.get('temperature', float('nan')))
            _fail(calibration.get('fitted_on') == 'val' and math.isfinite(temperature) and temperature > 0 and
                  calibration.get('checkpoint_sha256') == expected_checkpoint and calibration.get('n_samples') == reference['input_rows']['val'],
                  '完整输入温度并非本检查点的validation校准：' + str(run))
            history = [json.loads(line) for line in (run / 'epoch_metrics.jsonl').read_text(encoding='utf-8').splitlines() if line.strip()]
            _fail(bool(history) and [x.get('epoch') for x in history] == list(range(1, len(history) + 1)) and
                  all(x.get('monitor', 'val_loss') == 'val_loss' and math.isfinite(float(x['val']['loss'])) for x in history),
                  '完整输入验证历史不完整或选模规则不同：' + str(run))
            selected = min(history, key=lambda x: float(x['val']['loss']))
            _fail(selected['epoch'] == result.get('best_epoch'), '完整输入不是validation loss所选检查点')
            val_metrics = read_json(run / 'val/analysis/metrics.json')
            val = _metrics(val_metrics, reference['input_rows']['val'], 'val/seed_' + str(seed))
            _fail(math.isclose(float(selected['val']['macro_f1']), float(val['raw']['macro_f1']), abs_tol=1e-9),
                  '完整输入最佳epoch与val导出指标不符')
            val_export = read_json(run / 'val/export_complete.json')
            exported = _validate_export(val_export, 'val', reference['input_rows']['val'], expected_checkpoint, temperature, reference)
            _fail(exported.get('raw', {}).get('confusion_matrix') == val['raw']['confusion_matrix'], '完整输入val导出与分析混淆计数不同')
            completion = read_json(test_run / 'run_complete.json')
            _fail((completion.get('status'), completion.get('model'), completion.get('seed'), completion.get('feature_scheme')) ==
                  ('complete', 'gatedconv_target', seed, 'full') and completion.get('test_evaluated') is True and
                  completion.get('checkpoint_sha256') == expected_checkpoint and completion.get('temperature') == temperature and
                  completion.get('n_samples') == reference['input_rows']['test'], '完整输入内部test完成身份不同')
            allowed_sources = {str(run.resolve()), str(Path(record['run_dir']).expanduser().resolve())}
            _fail(str(Path(completion.get('source_run_dir', '.')).expanduser().resolve()) in allowed_sources,
                  '完整输入test来自另一训练运行')
            _validate_export({**completion, 'split': 'test', 'metrics': completion.get('metrics_by_split', {}).get('test', {})},
                             'test', reference['input_rows']['test'], expected_checkpoint, temperature, reference)
            test_metrics = read_json(test_run / 'test/analysis/metrics.json')
            test = _metrics(test_metrics, reference['input_rows']['test'], 'test/seed_' + str(seed))
            test_export = read_json(test_run / 'test/export_complete.json')
            exported_test = _validate_export(test_export, 'test', reference['input_rows']['test'], expected_checkpoint, temperature, reference)
            _fail(exported_test.get('raw', {}).get('confusion_matrix') == test['raw']['confusion_matrix'] ==
                  completion.get('metrics_by_split', {}).get('test', {}).get('raw', {}).get('confusion_matrix'),
                  '完整输入test导出、完成记录与分析混淆计数不同')
            _fail(all(sha256(path) == value for path, value in before_hashes.items()), '完整输入来源在核验期间被修改')
            output.append({'seed': seed, 'scheme': 'full', 'run_dir': str(run), 'checkpoint': str(checkpoint),
                'checkpoint_sha256': expected_checkpoint, 'checkpoint_bytes_verified': checkpoint.is_file(),
                'selected_epoch': selected['epoch'], 'temperature': temperature, 'config': config,
                'calibration': calibration, 'val_metrics_path': str(run / 'val/analysis/metrics.json'),
                'test_metrics_path': str(test_run / 'test/analysis/metrics.json'),
                'val_export_path': str(run / 'val/export_complete.json'),
                'test_export_path': str(test_run / 'test/export_complete.json'),
                'test_run_dir': str(test_run), 'internal_test_dir': str(test_run),
                'val_metrics': val_metrics, 'test_metrics': test_metrics,
                'metadata_sha256': before_hashes, 'use': 'read_only_comparison_never_initialization'})
    return output
