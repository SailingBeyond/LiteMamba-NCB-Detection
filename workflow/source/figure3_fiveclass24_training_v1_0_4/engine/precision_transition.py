"""Audited, restartable metadata adoption for the known v1.0.2 FP16 failure.

No model or optimizer tensor is changed. Original bytes are archived before any
run file is replaced, and the original selection history remains identifiable.
"""
from __future__ import annotations

import copy
from datetime import datetime, timezone
import hashlib
import json
import math
import os
from pathlib import Path
import shutil

POLICY = 'fp32_compute_preserved_scaler_v1'
MIGRATION_NAME = 'v1_0_2_to_v1_0_4_fp32_compute'


def _require(value, message):
    if not value:
        raise ValueError(message)


def _load(path):
    return json.loads(Path(path).read_text(encoding='utf-8'))


def _sha(path):
    digest = hashlib.sha256()
    with Path(path).open('rb') as stream:
        for block in iter(lambda: stream.read(4 * 1024 ** 2), b''):
            digest.update(block)
    return digest.hexdigest()


def _write(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temp = path.with_name(path.name + '.writing.' + str(os.getpid()))
    with temp.open('w', encoding='utf-8') as stream:
        json.dump(value, stream, ensure_ascii=False, indent=2, allow_nan=False)
        stream.write('\n'); stream.flush(); os.fsync(stream.fileno())
    os.replace(temp, path)


def _copy(source, target):
    target = Path(target)
    target.parent.mkdir(parents=True, exist_ok=True)
    temp = target.with_name(target.name + '.writing.' + str(os.getpid()))
    with Path(source).open('rb') as src, temp.open('wb') as dst:
        shutil.copyfileobj(src, dst, length=4 * 1024 ** 2)
        dst.flush(); os.fsync(dst.fileno())
    os.replace(temp, target)


def _torch_load(path):
    import torch
    try:
        return torch.load(path, map_location='cpu', weights_only=False)
    except TypeError:
        return torch.load(path, map_location='cpu')


def _torch_save(path, value):
    import torch
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temp = path.with_name(path.name + '.writing.' + str(os.getpid()))
    with temp.open('wb') as stream:
        torch.save(value, stream); stream.flush(); os.fsync(stream.fileno())
    os.replace(temp, path)


def _equal(a, b):
    """Value equality, treating historical NaN metric sentinels as preserved."""
    import numpy as np
    import torch
    if torch.is_tensor(a) or torch.is_tensor(b):
        return (torch.is_tensor(a) and torch.is_tensor(b) and a.dtype == b.dtype
                and a.shape == b.shape and torch.equal(a.cpu(), b.cpu()))
    if isinstance(a, np.ndarray) or isinstance(b, np.ndarray):
        return (isinstance(a, np.ndarray) and isinstance(b, np.ndarray) and a.dtype == b.dtype
                and np.array_equal(a, b, equal_nan=True))
    if isinstance(a, dict) or isinstance(b, dict):
        return isinstance(a, dict) and isinstance(b, dict) and a.keys() == b.keys() and all(_equal(a[k], b[k]) for k in a)
    if isinstance(a, (list, tuple)) or isinstance(b, (list, tuple)):
        return type(a) is type(b) and len(a) == len(b) and all(_equal(x, y) for x, y in zip(a, b))
    if isinstance(a, (float, np.floating)) and isinstance(b, (float, np.floating)):
        return (math.isnan(float(a)) and math.isnan(float(b))) or a == b
    return a == b


def pending(root):
    path = Path(root) / 'migration' / MIGRATION_NAME / 'transaction.json'
    return path.is_file() and _load(path).get('status') != 'complete'


def _check_gate(root, new_engine):
    path = Path(root) / 'precision_fix_validation' / 'gate.json'
    _require(path.is_file(), '缺少异常批次的全FP32修复验证；先执行自动gate')
    gate = _load(path)
    _require(gate.get('status') == 'PASS' and gate.get('numerical_precision_policy') == POLICY,
             '异常批次的全FP32修复验证尚未通过')
    _require(gate.get('engine_sha256') == new_engine, '修复gate不属于当前完整引擎源码')
    return _sha(path)


def _validate_old_plan(previous, current, reference):
    _require(previous.get('version') == '1.0.2', '只支持已核对的1.0.2训练前/训练中状态迁移')
    _require(previous.get('source_code_sha256') == reference['source_code_sha256'], '原1.0.2方案源码SHA不符，拒绝迁移')
    comparable = copy.deepcopy(current)
    comparable['version'] = '1.0.2'
    comparable['source_code_sha256'] = previous['source_code_sha256']
    comparable['amp_policy'] = 'CUDA AMP enabled; CPU FP32'
    comparable.pop('numerical_precision_policy', None)
    _require(previous == comparable, '原1.0.2输入、任务、种子或科学配置与当前方案不符，拒绝迁移')


def _validate_no_test_or_exports(root):
    root = Path(root)
    for name in ('evaluation_plan.json', 'test_started.json', 'evaluation_complete.json'):
        _require(not (root / name).exists(), '模型已冻结或test已开始，不能更改本次训练精度方案：' + name)
    for run in (root / 'scenarios').glob('*/runs/*/seed_*'):
        result = _load(run / 'results.json') if (run / 'results.json').is_file() else {}
        _require(not result.get('test_evaluated') and result.get('status') not in ('weights_complete', 'train_complete', 'complete'),
                 '已有完成权重/导出的run，不能在本迁移中静默混用：' + str(run))
        for child in ('val', 'test', 'calibration.json'):
            _require(not (run / child).exists(), '已有校准/预测导出，不能静默混用精度：' + str(run / child))


def _adopt_config(config, snapshot, transition):
    new = copy.deepcopy(config)
    new['source_snapshot'] = copy.deepcopy(snapshot)
    new['numerical_precision_policy'] = POLICY
    new['precision_transition'] = copy.deepcopy(transition)
    new['trainer_version'] = '1.0.4-fiveclass'
    return new


def _safe_path(root, relative):
    part = Path(relative)
    _require(not part.is_absolute() and '..' not in part.parts, '非法迁移路径')
    return Path(root) / part


def _apply(root, archive, transaction, current):
    _require(transaction['current_plan'] == current, '待完成迁移不属于当前源码/配置，拒绝继续')
    for entry in transaction['entries']:
        target = _safe_path(root, entry['relative_path'])
        actual = _sha(target) if target.is_file() else None
        _require(actual in (entry['before_sha256'], entry['after_sha256']),
                 '迁移期间目标文件被另行修改，拒绝覆盖：' + str(target))
        source = _safe_path(archive / 'staged', entry['relative_path'])
        _require(_sha(source) == entry['after_sha256'], '迁移暂存文件损坏')
        if actual != entry['after_sha256']:
            _copy(source, target)
    transaction['status'] = 'complete'
    transaction['completed_at'] = datetime.now(timezone.utc).isoformat()
    _write(archive / 'transaction.json', transaction)
    print('[MIGRATE] 已备份原方案/源码/检查点；采用全FP32计算策略，原权重及Adam/scaler/RNG/history不变', flush=True)


def migrate(root, package_root, previous, current):
    """Called with the original pipeline lock held; safely retries partial commit."""
    root, package_root = Path(root), Path(package_root)
    _require(not (root / 'migration' / 'v1_0_2_to_v1_0_3_fp32_logits' / 'transaction.json').exists(),
             '检测到已开始的1.0.3迁移；本包仅处理验证失败而未迁移的原1.0.2状态，请保留结果核查')
    archive = root / 'migration' / MIGRATION_NAME
    transaction_path = archive / 'transaction.json'
    new_engine = {path.name: _sha(path) for path in sorted((package_root / 'engine').glob('*.py'))}
    gate_sha = _check_gate(root, new_engine)
    if transaction_path.is_file():
        transaction = _load(transaction_path)
        _require(transaction['gate_engine_sha256'] == new_engine, '迁移记录引擎身份改变')
        if transaction.get('status') == 'complete':
            _require(previous == current, '迁移已完成但当前方案又改变，拒绝重复迁移')
            return
        _apply(root, archive, transaction, current)
        return
    reference = _load(package_root / 'reference' / 'v1_0_2_migration.json')
    _validate_old_plan(previous, current, reference)
    _validate_no_test_or_exports(root)
    old_engine = reference['engine_sha256']
    archive.mkdir(parents=True, exist_ok=True)
    # Incomplete staging can only be discarded before transaction.json exists;
    # no original target is replaced until that durable record is committed.
    for name in ('backup', 'staged'):
        if (archive / name).exists():
            shutil.rmtree(archive / name)
        (archive / name).mkdir()
    entries, run_records = [], []

    def stage(relative, writer):
        target = _safe_path(root, relative)
        backup = _safe_path(archive / 'backup', relative)
        before = _sha(target) if target.is_file() else None
        if before is not None:
            _copy(target, backup)
            _require(_sha(backup) == before, '原文件备份SHA不符')
        staged = _safe_path(archive / 'staged', relative)
        staged.parent.mkdir(parents=True, exist_ok=True)
        writer(staged)
        entries.append({'relative_path': str(relative), 'before_sha256': before,
                        'after_sha256': _sha(staged), 'backup_exists': before is not None})

    for run in sorted((root / 'scenarios').glob('*/runs/*/seed_*')):
        last_path = run / 'last_checkpoint.pt'
        if not last_path.is_file():
            _require(not any(run.glob('*.pt')), '有权重却没有last，拒绝静默重新初始化：' + str(run))
            continue
        config_path = run / 'resolved_config.json'
        config = _load(config_path)
        parts = run.relative_to(root).parts
        _require(len(parts) == 5 and parts[0] == 'scenarios' and parts[2] == 'runs', '未知run布局')
        scenario, model, seed = parts[1], parts[3], int(parts[4].removeprefix('seed_'))
        _require(scenario in current['fixed_scenarios'] and model in current['fixed_models']
                 and seed in current['fixed_seeds'], 'run不属于冻结24项')
        _require(config.get('scenario') == scenario and config.get('seed') == seed,
                 'run目录与scenario/seed配置不符')
        for key, value in previous['model_configurations'][model].items():
            _require(config.get(key) == value, '原run科学设置与冻结方案不符：' + key)
        prepared = _load(root / 'scenarios' / scenario / 'prepared_data' / 'prepare_manifest.json')
        for split, filename in (('train', 'train'), ('val', 'validation'), ('test', 'test')):
            for field, ending in (('feature', '.csv'), ('metadata', '_meta.tsv')):
                _require(config['input_fingerprints'][split][field]['sha256'] == prepared['files'][filename + ending]['sha256'],
                         '原run输入指纹与冻结预备数据不符')
        _require(config['source_snapshot']['files_sha256'] == old_engine, '部分run并非冻结1.0.2引擎：' + str(run))
        _require(config.get('numerical_precision_policy') in (None, 'legacy_amp_all_layers'), '部分run已用其他精度方案')
        for name, digest in old_engine.items():
            _require(_sha(run / 'source_snapshot' / name) == digest, '原run源码快照SHA不符：' + name)
        last = _torch_load(last_path)
        _require(last.get('resolved_config', {}).get('source_snapshot', {}).get('files_sha256') == old_engine,
                 'last checkpoint源码与原run配置不符')
        epochs = int(last['epoch'])
        names = [name for name in ('last_checkpoint.pt', 'best_checkpoint.pt', 'best_state_dict.pt') if (run / name).is_file()]
        _require('best_checkpoint.pt' in names and 'best_state_dict.pt' in names, '部分run缺少best检查点，需先核对已提交轮次')
        old_hashes = {name: _sha(run / name) for name in names}
        transition = {'kind': MIGRATION_NAME, 'numerical_precision_policy': POLICY,
                      'from_source_snapshot': copy.deepcopy(config['source_snapshot']),
                      'legacy_checkpoint_sha256': old_hashes,
                      'legacy_history_through_epoch': epochs,
                      'next_training_epoch': epochs + 1,
                      'historical_monitor_values_precision': 'legacy AMP; preserved, not recomputed with full FP32',
                      'new_validation_and_test_precision': POLICY,
                      'new_forward_backward_compute_dtype': 'float32',
                      'saved_amp_flag_semantics': 'Saved GradScaler state retained for continuity; FP16 autocast is disabled under this policy',
                      'scope': 'Numerical repair only; not bitwise replay across the boundary. All original parameter/Adam/scaler/RNG/history values preserved.',
                      'backup_directory': str(archive / 'backup' / run.relative_to(root)),
                      'gate_sha256': gate_sha}
        snapshot = {'files_sha256': new_engine, 'git_revision': None}
        for name in names:
            relative = run.relative_to(root) / name
            if name == 'best_state_dict.pt':
                stage(relative, lambda p, source=run / name: _copy(source, p))
                continue
            original = _torch_load(run / name)
            _require(original['resolved_config']['source_snapshot']['files_sha256'] == old_engine,
                     '检查点源码并非冻结1.0.2：' + str(run / name))
            for key in ('model_kwargs', 'model_type', 'seed', 'monitor', 'dwell_divisor', 'quality_divisor',
                        'raw_signal_len', 'batch_size', 'accum_steps', 'lr', 'weight_decay', 'label_smoothing',
                        'clip_grad', 'patience', 'amp', 'class_names', 'scenario', 'num_classes'):
                _require(original['resolved_config'].get(key) == config.get(key),
                         '原检查点和run配置不一致：' + key)
            adopted = copy.deepcopy(original)
            adopted['resolved_config'] = _adopt_config(original['resolved_config'], snapshot, transition)
            _require(_equal({k:v for k,v in original.items() if k!='resolved_config'},
                            {k:v for k,v in adopted.items() if k!='resolved_config'}), '迁移意外改变训练状态')
            stage(relative, lambda p, value=adopted: _torch_save(p, value))
            loaded = _torch_load(archive / 'staged' / relative)
            _require(_equal(loaded, adopted), '暂存检查点重读后状态不一致')
            del original, adopted, loaded
        stage(run.relative_to(root) / 'resolved_config.json',
              lambda p, value=_adopt_config(config, snapshot, transition): _write(p, value))
        if (run / 'model_config.json').is_file():
            model_config = _load(run / 'model_config.json')
            model_config['numerical_precision_policy'] = POLICY
            stage(run.relative_to(root) / 'model_config.json', lambda p, value=model_config: _write(p, value))
        # This is an explicit provenance snapshot adoption, not a silent source
        # guard bypass. The exact six historical files remain in backup.
        for path in sorted((package_root / 'engine').glob('*.py')):
            stage(run.relative_to(root) / 'source_snapshot' / path.name,
                  lambda p, source=path: _copy(source, p))
        stage(run.relative_to(root) / 'source_snapshot' / 'manifest.json', lambda p, value=snapshot: _write(p, value))
        stage(run.relative_to(root) / 'precision_transition.json', lambda p, value=transition: _write(p, value))
        run_records.append({'relative_run': str(run.relative_to(root)), 'legacy_last_epoch': epochs,
                            'next_epoch': epochs + 1, 'legacy_checkpoint_sha256': old_hashes,
                            'tensor_optimizer_rng_history_values_unchanged': True})
    # Commit plan last. A partially committed transaction cannot start training;
    # the next invocation finishes it before any model is launched.
    stage(Path('experiment_plan.json'), lambda p: _write(p, current))
    transaction = {'status': 'prepared', 'from_version': '1.0.2', 'to_version': '1.0.4',
                   'created_at': datetime.now(timezone.utc).isoformat(),
                   'numerical_precision_policy': POLICY, 'gate_sha256': gate_sha,
                   'gate_engine_sha256': new_engine, 'previous_plan': previous, 'current_plan': current,
                   'entries': entries, 'runs': run_records,
                   'historical_monitor_values_preserved_not_recomputed': True,
                   'no_model_optimizer_rng_or_history_value_change': True}
    _write(transaction_path, transaction)
    _apply(root, archive, transaction, current)
