#!/usr/bin/env python3
"""Figure 3 两项五分类：四模型、三个固定种子，先训练验证、再统一冻结测试。"""
from __future__ import annotations
import argparse
from datetime import datetime, timezone
import fcntl
import hashlib
import json
import math
import os
from pathlib import Path
import shutil
import signal
import subprocess
import sys
import time

VERSION = '1.0.4'
NUMERICAL_PRECISION_POLICY = 'fp32_compute_preserved_scaler_v1'
HERE = Path(__file__).resolve().parent
ENGINE = HERE / 'engine'
REFERENCE_DIR = HERE / 'reference'
SCENARIOS = ('X', 'Y')
MODELS = ('gatedconv', 'transformer', 'convmixer', 'bigru')
SEEDS = (42, 2024, 2025)
CLASS_NAMES = {'X': ['A', 'T', 'C', 'G', 'X'], 'Y': ['A', 'T', 'C', 'G', 'Y']}
DEFAULT_SOURCE = '/path/to/revision-inputs/CPLX-XNA-replenishment/CPLX-XNA-replenishment/fast5_single_guppy/fiveclass_overlap_read_split_721'
DEFAULT_OUTPUT = '/path/to/revision-results/figure3_fiveclass24_results_v1_0_0'
SCIENTIFIC = {
    'litemamba_variant': 'legacy_gatedconv',
    'readout': 'target_attention', 'feature_scheme': 'full', 'model_dim': 96,
    'num_blocks': 4, 'state_dim': 16, 'target_index': 3, 'dropout': 0.1,
    'classifier_dropout': 0.2, 'ssm_bidirectional': 1, 'nhead': 4,
    'num_layers': 2, 'gru_hidden': 64, 'epochs': 200, 'patience': 30,
    'lr': 0.001, 'batch_size': 256, 'raw_signal_len': 30,
    'dwell_divisor': 60.0, 'quality_divisor': 50.0, 'accum_steps': 1,
    'weight_decay': 0.0001, 'label_smoothing': 0.0, 'clip_grad': 1.0,
    'monitor': 'val_loss', 'bootstrap': 200, 'embedding_dtype': 'float32',
    'checkpoint_every': 10,
}
OUTPUT_POLICY = 'compact'
OUTPUT_CONTRACT = {
    'policy': OUTPUT_POLICY,
    'retained_prediction_splits': ['val', 'test'],
    'retained': ['all_epoch_training_and_validation_metrics', 'best_checkpoint',
                 'last_checkpoint_with_optimizer_and_rng', 'best_state_dict',
                 'validation_temperature', 'validation_and_test_float32_logits',
                 'validation_and_test_raw_and_calibrated_probabilities',
                 'validation_and_test_prediction_csv_and_analysis'],
    'omitted': ['training_split_prediction_export', 'per_epoch_validation_logits',
                'embeddings', 'attention_arrays', 'periodic_epoch_checkpoints'],
    'scientific_training_and_model_configuration_unchanged': True,
}
GIB = 2 ** 30
STOP = False
CHILDREN = {}
LOCK_FD = None
VERIFIED_EXPORT_FILES = {}


def now(): return datetime.now(timezone.utc).isoformat()
def load_json(path): return json.loads(Path(path).read_text(encoding='utf-8'))
def sha256(path):
    digest = hashlib.sha256()
    with Path(path).open('rb') as stream:
        for block in iter(lambda: stream.read(8 * 1024 ** 2), b''): digest.update(block)
    return digest.hexdigest()


def write_json(path, data):
    path = Path(path); path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + '.writing.' + str(os.getpid()))
    try:
        with temporary.open('w', encoding='utf-8') as stream:
            json.dump(data, stream, ensure_ascii=False, indent=2, allow_nan=False)
            stream.write('\n'); stream.flush(); os.fsync(stream.fileno())
        os.replace(temporary, path)
    finally: temporary.unlink(missing_ok=True)


def scenario_root(args, scenario):
    if scenario not in SCENARIOS: raise ValueError('未知五分类任务：' + str(scenario))
    return args.output_dir / 'scenarios' / scenario


def run_path(args, scenario, model, seed):
    if model not in MODELS or seed not in SEEDS:
        raise ValueError('model/seed不在固定24项清单中')
    return scenario_root(args, scenario) / 'runs' / model / ('seed_' + str(seed))

def phase_jobs(phase):
    if phase not in ('prepare', 'train', 'freeze', 'evaluate', 'all', 'summary'):
        raise ValueError('未知阶段')
    return [(scenario, model, seed) for seed in SEEDS for model in MODELS for scenario in SCENARIOS]


def model_configuration(model):
    if model not in MODELS: raise ValueError('未知模型：' + str(model))
    return dict(SCIENTIFIC, model_type='litemamba' if model == 'gatedconv' else model,
                checkpoint_every=0, output_policy=OUTPUT_POLICY)

def process_token(pid):
    try:
        text = Path('/proc/%d/stat' % pid).read_text(); tail = text[text.rfind(')') + 2:].split()
        return None if tail[0] == 'Z' else tail[19]
    except (OSError, IndexError): return None


def live_job(record):
    return bool(record.get('pid') and record.get('process_start_token')
                and process_token(record['pid']) == record['process_start_token'])


def request_stop(sig, frame):
    global STOP
    STOP = True
    for process in list(CHILDREN.values()):
        if process.poll() is None: process.terminate()


def parser():
    result = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    result.add_argument('--version', action='version', version='Figure3 fiveclass24 ' + VERSION)
    result.add_argument('--source_dir', default=DEFAULT_SOURCE)
    result.add_argument('--output_dir', default=DEFAULT_OUTPUT)
    result.add_argument('--phase', choices=('prepare', 'train', 'freeze', 'evaluate', 'all', 'summary'), default='all')
    result.add_argument('--resume', action='store_true')
    result.add_argument('--precision_probe_dir', type=Path, default=None,
                        help='可选：已捕获fault_state.pt和failed_batch.npz的诊断attempt目录')
    result.add_argument('--device', default='auto')
    result.add_argument('--cpu_threads', type=int, default=4)
    result.add_argument('--workers', type=int, default=0)
    result.add_argument('--max_parallel', type=int, choices=(1, 2, 3, 4), default=1,
                        help='权重训练硬上限；按两个任务实测资源及当前空闲量自动减少；导出/测试串行')
    result.add_argument('--disk_reserve_gib', type=float, default=8)
    result.add_argument('--gpu_reserve_gib', type=float, default=2)
    result.add_argument('--ram_reserve_gib', type=float, default=3)
    result.add_argument('--reserve_fraction', type=float, default=.10)
    result.add_argument('--memory_safety_factor', type=float, default=1.35)
    result.add_argument('--poll_seconds', type=float, default=3)
    result.add_argument('--dry_run', action='store_true', help='只打印固定阶段；不读取数据或启动子进程')
    return result


def validate_args(args):
    args.source_dir = Path(args.source_dir).expanduser().resolve()
    args.output_dir = Path(args.output_dir or DEFAULT_OUTPUT).expanduser().resolve()
    if args.output_dir == args.source_dir or args.output_dir in args.source_dir.parents:
        raise ValueError('输出必须是独立子目录，不能覆盖源数据目录或其父目录')
    if args.output_dir == HERE or HERE in args.output_dir.parents or args.output_dir in HERE.parents:
        raise ValueError('结果目录不能位于代码目录内或覆盖代码目录')
    if args.device not in ('auto', 'cpu', 'cuda'):
        if not args.device.startswith('cuda:') or not args.device[5:].isdigit(): raise ValueError('device仅支持auto/cpu/cuda/cuda:N')
    values = [args.disk_reserve_gib, args.gpu_reserve_gib, args.ram_reserve_gib,
              args.reserve_fraction, args.memory_safety_factor, args.poll_seconds]
    if not all(math.isfinite(value) for value in values): raise ValueError('资源参数须为有限数')
    if args.cpu_threads < 1 or args.workers < 0 or args.disk_reserve_gib <= 0 or min(args.gpu_reserve_gib, args.ram_reserve_gib) < 0:
        raise ValueError('线程/磁盘余量须为正数，workers及内存余量不能负')
    if not 0 <= args.reserve_fraction < 1 or args.memory_safety_factor < 1 or args.poll_seconds <= 0:
        raise ValueError('资源比例/系数/间隔无效')
    return args


def source_identity():
    expected = load_json(REFERENCE_DIR / 'engine_identity.json')
    for name, digest in expected['engine_sha256'].items():
        if sha256(ENGINE / name) != digest: raise ValueError('随包冻结引擎发生变化：' + name)
    files = [HERE / name for name in ('run_fiveclass24.py', 'prepare_data.py', 'summarize_fiveclass24.py',
                                     'validate_precision_fix.py', 'validate_precision_worker.py')]
    files += sorted(ENGINE.glob('*.py')) + sorted(REFERENCE_DIR.rglob('*'))
    return {str(path.relative_to(HERE)): sha256(path) for path in files
            if path.is_file() and '__pycache__' not in path.parts and path.suffix not in ('.pyc', '.pyo')}

def make_plan(args, prepared):
    return {
        'version': VERSION, 'num_classes': 5, 'fixed_scenarios': list(SCENARIOS),
        'fixed_models': list(MODELS), 'model_names': {'gatedconv': 'GatedConv', 'transformer': 'Transformer', 'convmixer': 'ConvMixer', 'bigru': 'BiGRU'},
        'class_names': CLASS_NAMES, 'fixed_seeds': list(SEEDS),
        'scientific_configuration': dict(SCIENTIFIC),
        'output_policy': OUTPUT_POLICY, 'output_contract': OUTPUT_CONTRACT,
        'model_configurations': {model: model_configuration(model) for model in MODELS},
        'initialization': '24 independent seeded random initializations; no pretrained weights',
        'training_order': 'seed, architecture, task; dynamically admitted concurrent training; serial exports and tests',
        'test_policy': 'All 24 checkpoints and validation temperatures frozen before any test prediction; no test-dependent selection.',
        'source_dir': str(args.source_dir), 'output_dir': str(args.output_dir),
        'prepared_manifest_sha256': {scenario: sha256(prepared[scenario]['manifest_path']) for scenario in SCENARIOS},
        'source_code_sha256': source_identity(),
        'planned_test_rows': {scenario: prepared[scenario]['rows_by_split']['test'] for scenario in SCENARIOS},
        'scope': 'Figure3 broad-context five-class tasks from the user-prepared datasets; not the 77-group same-source directional supplementary control. Real-read and disclosed positive-row-proxy identity scopes remain separate.',
        'target_mask_index': 3, 'seq_len': 6,
        'amp_policy': 'Full FP32 forward/backward; original CUDA GradScaler state retained; no FP16 autocast',
        'numerical_precision_policy': NUMERICAL_PRECISION_POLICY,
    }

def commit_plan(args, prepared):
    path = args.output_dir / 'experiment_plan.json'; plan = make_plan(args, prepared)
    from engine.precision_transition import pending, migrate
    if path.exists():
        previous = load_json(path)
        if pending(args.output_dir) or previous.get('version') == '1.0.2':
            if not args.resume: raise ValueError('1.0.2数值精度修复迁移必须使用--resume')
            migrate(args.output_dir, HERE, previous, plan)
        elif previous.get('version') == '1.0.3':
            raise ValueError('本包用于1.0.3验证失败且原计划仍为1.0.2的状态；已迁移的1.0.3结果需单独核查，保留全部结果，不会覆盖')
        elif previous != plan:
            migrate_unstarted_v101_plan(args, prepared, previous, plan)
    else: write_json(path, plan)
    return plan


def migrate_unstarted_v101_plan(args, prepared, previous, current):
    """Upgrade only the reported v1.0.1 stop before any child was launched.

    Both historical code identity and the prepared manifests remain checked.
    There is deliberately no route that silently mixes trained model versions.
    """
    error = '已有方案不属于可迁移的1.0.1训练前状态；不能混入不同源码或产物，请使用新输出目录'
    if not args.resume or previous.get('version') != '1.0.1': raise ValueError(error)
    reference = load_json(REFERENCE_DIR / 'v1_0_1_migration.json')
    expected_source = reference.get('source_code_sha256', {})
    recorded_source = previous.get('source_code_sha256')
    if not expected_source or not isinstance(recorded_source, dict): raise ValueError(error)
    meaningful_source = {}
    for name, digest in recorded_source.items():
        part = Path(name)
        if part.is_absolute() or '..' in part.parts: raise ValueError(error)
        # Old coordinator incidentally included generated reference bytecode in
        # its source list. It is not source code and is never migrated/executed.
        if '__pycache__' in part.parts and part.suffix in ('.pyc', '.pyo'):
            if not isinstance(digest, str) or len(digest) != 64 or any(c not in '0123456789abcdefABCDEF' for c in digest):
                raise ValueError(error)
            continue
        meaningful_source[name] = digest
    if meaningful_source != expected_source: raise ValueError(error + '（原代码SHA不符）')
    comparable = dict(current)
    comparable['version'] = '1.0.1'
    comparable['source_code_sha256'] = recorded_source
    comparable.pop('output_policy', None); comparable.pop('output_contract', None)
    comparable['model_configurations'] = {
        model: {key:value for key,value in dict(config, checkpoint_every=10).items() if key!='output_policy'}
        for model,config in current['model_configurations'].items()}
    if previous != comparable: raise ValueError(error + '（输入或科学方案不符）')
    forbidden = ['evaluation_plan.json', 'test_started.json', 'evaluation_complete.json',
                 'resource_admission.json', 'stage_status.json']
    if any((args.output_dir / name).exists() for name in forbidden): raise ValueError(error)
    if any((args.output_dir / 'job_records').glob('*')): raise ValueError(error + '（已有子进程记录）')
    for scenario in SCENARIOS:
        folder = scenario_root(args, scenario)
        if any((folder / 'runs').rglob('*')) or any((folder / 'preflight').rglob('*')):
            raise ValueError(error + '（已有模型或预检产物）')
        if any((folder / 'resource_profiles').rglob('*')):
            raise ValueError(error + '（已有探测产物）')
        if previous.get('prepared_manifest_sha256', {}).get(scenario) != sha256(prepared[scenario]['manifest_path']):
            raise ValueError(error + '（准备清单SHA不符）')
    if any(args.output_dir.rglob('*.pt')) or any(args.output_dir.rglob('results.json')):
        raise ValueError(error + '（已有模型权重或结果）')
    archive = args.output_dir / 'migration' / 'v1_0_1_pretraining'
    archive.mkdir(parents=True, exist_ok=True)
    archived_plan = archive / 'experiment_plan.json'
    original_path = args.output_dir / 'experiment_plan.json'
    original_bytes = original_path.read_bytes()
    if archived_plan.exists():
        if archived_plan.read_bytes() != original_bytes: raise ValueError('已有迁移归档与原方案不同，拒绝覆盖')
    else:
        # Copy the exact historical bytes, rather than rewriting their JSON.
        temporary = archived_plan.with_suffix('.json.writing')
        with temporary.open('wb') as stream:
            stream.write(original_bytes); stream.flush(); os.fsync(stream.fileno())
        os.replace(temporary, archived_plan)
    record = {'from_version': '1.0.1', 'to_version': VERSION,
              'scope': 'Prepared read indexes only; no child, model checkpoint or test existed.',
              'previous_experiment_plan_sha256': sha256(archived_plan),
              'historical_source_code_sha256': recorded_source,
              'prepared_manifest_sha256': previous['prepared_manifest_sha256'],
              'scientific_configuration_unchanged': True,
              'output_policy': OUTPUT_POLICY, 'status': 'validated_before_plan_commit'}
    write_json(archive / 'migration.json', record)
    write_json(original_path, current)
    record.update(status='complete', completed_at=now(),
                  current_experiment_plan_sha256=sha256(original_path))
    write_json(archive / 'migration.json', record)
    print('[MIGRATE] 已保留1.0.1原方案及SHA；复用已核验read索引，切换compact输出，未重新抽样', flush=True)


def shared_cache_present(args, scenario, prepared):
    """预检标记不能替代缓存本体；缺失/截断缓存仍计入重建预算。"""
    manifest_path=scenario_root(args,scenario)/'preflight'/'data_manifest.json'
    if not manifest_path.is_file():return False
    try:
        manifest=load_json(manifest_path)
        for split,source_split in [('train','train'),('val','validation'),('test','test')]:
            folder=Path(manifest['manifest_paths'][split]).parent
            n=int(prepared['rows_by_split'][source_split])
            sizes={'merged.npy':n*6*13*4,'raw.npy':n*6*SCIENTIFIC['raw_signal_len']*4,
                   'labels.npy':n*8,'metadata_offsets.npy':n*8}
            if not (folder/'manifest.jsonl').is_file() or not (folder/'audit.json').is_file():return False
            if any(not (folder/name).is_file() or (folder/name).stat().st_size < size for name,size in sizes.items()):return False
        return True
    except (OSError,ValueError,KeyError,TypeError):return False


def check_disk(args, jobs=(), include_prepare=False, prepared=None, include_test=False):
    budget = 0
    breakdown = {'shared_feature_cache': 0, 'remaining_training_exports_and_checkpoints': 0,
                 'remaining_epoch_validation_logits': 0, 'remaining_test_exports': 0,
                 'remaining_analysis_tables_and_working_files': 0}
    # Both val/test retain lossless float32 scores, calibrated probabilities and
    # the metadata-rich gzip CSV. Predictions are int64. Embeddings/attention
    # and per-epoch numeric snapshots are intentionally not generated.
    per_window = 5 * 4 * 3 + 8 + 8 + 1000
    if prepared is not None:
        for scenario in SCENARIOS:
            rec = prepared[scenario]
            if not shared_cache_present(args,scenario,rec):
                breakdown['shared_feature_cache'] += int(rec['rows']) * 3000
        for scenario, model, seed in jobs:
            rec = prepared[scenario]; status = training_status(run_path(args, scenario, model, seed))
            if status not in ('train_complete', 'complete'):
                count = int(rec['rows_by_split']['validation'])
                breakdown['remaining_training_exports_and_checkpoints'] += count * per_window + 150 * 1024 ** 2
                breakdown['remaining_analysis_tables_and_working_files'] += count * 128
        if include_test:
            for scenario, model, seed in phase_jobs('evaluate'):
                if not load_json_or_empty(run_path(args, scenario, model, seed) / 'results.json').get('test_evaluated'):
                    count = int(prepared[scenario]['rows_by_split']['test'])
                    breakdown['remaining_test_exports'] += count * per_window
                    breakdown['remaining_analysis_tables_and_working_files'] += count * 128
    budget = sum(breakdown.values())
    needed = budget + args.disk_reserve_gib * GIB; free = shutil.disk_usage(args.output_dir).free
    write_json(args.output_dir / 'disk_budget.json', {'created_at': now(), 'free_bytes': free,
        'conservative_additional_bytes': budget, 'reserve_bytes': args.disk_reserve_gib * GIB,
        'breakdown_bytes': breakdown, 'source_feature_copy_bytes': 0, 'output_policy': OUTPUT_POLICY,
        'bytes_per_export_window': {'embedding': 0, 'logits_raw_calibrated_probabilities': 5 * 4 * 3, 'label': 8, 'prediction': 8, 'attention': 0, 'compressed_prediction_csv_estimate': 1000, 'analysis_tables_and_working_files_estimate': 128},
        'retained_prediction_splits': ['val', 'test'], 'epoch_validation_logits_retained': False,
        'note': 'Compact outputs retain validation/test scores and analysis plus all epoch metric logs and best/last checkpoints. No train inference, embeddings, attention or per-epoch logits. Cache/CSV/analysis estimates remain conservative; current disk guard is retained.'})
    if free < needed:
        raise OSError(28, '输出盘剩余%.2f GiB，当前阶段保守预算%.2f GiB（含%.2f GiB余量）；清理/换输出盘后使用--resume' %
                      (free / GIB, needed / GIB, args.disk_reserve_gib))
    print('[DISK] free=%.2f GiB current_stage_budget=%.2f GiB (source CSV is not copied)' % (free / GIB, needed / GIB), flush=True)


def load_json_or_empty(path):
    return load_json(path) if Path(path).is_file() else {}

def child_command(args, prepared, scenario, model, seed, mode, run_dir=None):
    run_dir = Path(run_dir or run_path(args, scenario, model, seed))
    command = [sys.executable, '-u', str(ENGINE / 'train_all.py'), '--mode', mode,
               '--data_dir', str(prepared[scenario]['data_dir']), '--cache_dir', str(scenario_root(args, scenario) / 'feature_cache'),
               '--output_dir', str(run_dir), '--device', args.device, '--cpu_threads', str(args.cpu_threads),
               '--workers', str(args.workers), '--concurrent_training', '--output_policy', OUTPUT_POLICY]
    if mode in ('train', 'preflight', 'profile'):
        for key, value in model_configuration(model).items():
            if key != 'output_policy': command += ['--' + key, str(value)]
        command += ['--seed', str(seed), '--defer_test', '--defer_train_export']
    if mode == 'profile': command += ['--profile_steps', '8']
    if mode == 'finalize': command += ['--defer_test', '--defer_train_export']
    return command

def orphan_check(args):
    for path in (args.output_dir / 'job_records').glob('*.json'):
        record = load_json(path)
        if record.get('state') == 'running' and live_job(record):
            raise RuntimeError('此前子进程仍在运行，禁止重复启动：pid=%s log=%s' % (record['pid'], record.get('log')))


def child_environment(args, scenario):
    if scenario not in SCENARIOS: raise ValueError('子进程缺少明确scenario')
    return dict(os.environ, OMP_NUM_THREADS=str(args.cpu_threads), MKL_NUM_THREADS=str(args.cpu_threads),
                PYTHONUNBUFFERED='1', CPLX_FIVECLASS_SCENARIO=scenario,
                FIVECLASS_PREPARED_MANIFEST=str(scenario_root(args,scenario)/'prepared_data'/'prepare_manifest.json'))


def start_child(args, command, name, scenario):
    if STOP: raise InterruptedError('收到停止请求，未启动新任务')
    stamp = datetime.now().strftime('%Y%m%d_%H%M%S_%f')
    log = args.output_dir / 'logs' / ('%s_%s.log' % (name, stamp)); log.parent.mkdir(exist_ok=True)
    record_path = args.output_dir / 'job_records' / ('%s_%s.json' % (name, stamp))
    record = {'state': 'launching', 'name': name, 'scenario': scenario, 'command': command,
              'log': str(log), 'started_at': now(), 'scenario_environment': scenario}
    write_json(record_path, record)
    with log.open('wb') as stream:
        process = subprocess.Popen(command, stdout=stream, stderr=subprocess.STDOUT, stdin=subprocess.DEVNULL,
                                   env=child_environment(args, scenario), pass_fds=(() if LOCK_FD is None else (LOCK_FD,)))
        CHILDREN[process.pid] = process
        record.update(state='running', pid=process.pid, process_start_token=process_token(process.pid))
        try: write_json(record_path, record)
        except BaseException:
            process.terminate(); process.wait(); CHILDREN.pop(process.pid, None); raise
    print('[LAUNCH] %s pid=%d log=%s' % (name, process.pid, log), flush=True)
    return process, record_path, record


def finish_child(handle):
    process, record_path, record = handle; code = process.wait(); CHILDREN.pop(process.pid, None)
    record.update(state='complete' if code == 0 else 'failed', returncode=code, finished_at=now())
    write_json(record_path, record); return code


def progress_message(handle):
    record = handle[2]; clock = time.monotonic()
    if clock - record.get('_last_progress_clock', 0) < 30: return
    record['_last_progress_clock'] = clock
    try:
        with Path(record['log']).open('rb') as stream:
            stream.seek(0, 2); stream.seek(max(0, stream.tell() - 4096))
            lines = stream.read().decode('utf-8', errors='replace').splitlines()
        latest = next((line for line in reversed(lines) if line.strip()), 'initializing')
    except OSError: latest = 'initializing'
    print('[RUNNING] %s pid=%d | %s' % (record['name'], handle[0].pid, latest[-230:]), flush=True)


def wait_child(args, handle):
    process = handle[0]
    while process.poll() is None:
        progress_message(handle)
        if STOP: process.terminate()
        if shutil.disk_usage(args.output_dir).free < args.disk_reserve_gib * GIB:
            process.terminate(); finish_child(handle)
            raise OSError(28, '剩余空间低于预留值；已停止子进程，清理后加--resume恢复最近完整epoch')
        time.sleep(min(args.poll_seconds, 3))
    code = finish_child(handle)
    if code: raise RuntimeError('%s退出码%d；日志：%s；保留任务状态，使用--resume继续' % (handle[2]['name'], code, handle[2]['log']))
    if STOP: raise InterruptedError('已停止；用相同命令加--resume恢复')


def run_child(args, command, name, scenario):
    wait_child(args, start_child(args, command, name, scenario))


def correct_scope_notes(run_dir, scenario):
    """仅保存范围说明；不改变模型或任何统计数值。"""
    write_json(Path(run_dir) / 'fiveclass_scope.json', {
        'scenario': scenario, 'class_names': CLASS_NAMES[scenario],
        'scope': 'Figure3 independently balanced X or Y five-class main task; label4 is task-specific.',
        'limitations': ['Not the strict77 same-source direction-matched control.',
                        'Read identity/proxy and source/template limits are disclosed in data_audit.json.',
                        'No template/run/batch independence inferred from row partitions.']})

def preflight(args, prepared, scenario):
    directory = scenario_root(args, scenario) / 'preflight'
    run_child(args, child_command(args, prepared, scenario, 'gatedconv', 42, 'preflight', directory), 'preflight_' + scenario, scenario)
    if not (directory / 'preflight.json').exists() or load_json(directory / 'preflight.json').get('status') != 'complete':
        raise RuntimeError('预检未完整完成：' + scenario)
    audit = load_json(directory / 'data_audit.json')
    if audit.get('status') != 'PASS' or any(audit['read_overlap_counts'].values()): raise ValueError('read/proxy跨集合审计失败：' + scenario)
    correct_scope_notes(directory, scenario)
    print('[PREFLIGHT PASS] %s five classes; real/proxy identity scope disclosed; no test inference' % scenario, flush=True)

def measure_profile(args, prepared, scenario, model):
    directory = scenario_root(args, scenario) / 'resource_profiles' / model / datetime.now().strftime('%Y%m%d_%H%M%S_%f')
    run_child(args, child_command(args, prepared, scenario, model, 42, 'profile', directory), 'resource_profile_' + scenario + '_' + model, scenario)
    profile = load_json(directory / 'resource_profile.json')
    if profile.get('actual_optimizer_steps', 0) < 8:
        raise ValueError('资源profile未测满8次真实AdamW更新')
    if profile.get('model_type') != model_configuration(model)['model_type']:
        raise ValueError('资源profile模型不符')
    if model == 'gatedconv' and profile.get('model_kwargs', {}).get('litemamba_variant') != 'legacy_gatedconv':
        raise ValueError('资源profile不是最终GatedConv')
    if _class_names(profile.get('class_names')) != CLASS_NAMES[scenario]: raise ValueError('资源profile类别与任务不符')
    correct_scope_notes(directory, scenario); return profile

def combined_profile(profiles):
    expected = {scenario + '/' + model for scenario in SCENARIOS for model in MODELS}
    if set(profiles) != expected: raise ValueError('两个任务的四种架构均须完成资源实测')
    devices = {profile['device'] for profile in profiles.values()}
    if len(devices) != 1: raise ValueError('资源profile逻辑设备不同')
    return {'device': devices.pop(),
            'host_admission_estimate_bytes': max(p.get('host_admission_estimate_bytes', 0) for p in profiles.values()),
            'gpu_admission_estimate_bytes': max(p.get('gpu_admission_estimate_bytes', 0) for p in profiles.values()),
            'scenario_model_profiles': profiles}

def memory_snapshot(device):
    memory = {}
    for line in Path('/proc/meminfo').read_text().splitlines():
        key, value = line.split(':', 1); memory[key] = int(value.split()[0]) * 1024
    result = {'host_available': memory.get('MemAvailable', memory.get('MemFree', 0)),
              'host_total': memory['MemTotal'], 'gpu_free': 0, 'gpu_total': 0}
    if device.startswith('cuda'):
        import torch
        free, total = torch.cuda.mem_get_info(torch.device(device)); result.update(gpu_free=free, gpu_total=total)
    return result


def admission(args, profile, wanted, active_count=0, capacity=None):
    snapshot = memory_snapshot(profile['device'])
    cpus = len(os.sched_getaffinity(0)) if hasattr(os, 'sched_getaffinity') else (os.cpu_count() or 1)
    cpu_slots = max(1, cpus // max(1, args.cpu_threads + args.workers))
    host_need = max(profile.get('host_admission_estimate_bytes', 0), 512 * 1024 ** 2) * args.memory_safety_factor
    host_free = max(0, snapshot['host_available'] - max(args.ram_reserve_gib * GIB, snapshot['host_total'] * args.reserve_fraction))
    slots = min(wanted, max(0, args.max_parallel - active_count), max(0, cpu_slots - active_count), int(host_free // host_need))
    gpu_need = 0
    if profile['device'].startswith('cuda'):
        gpu_need = max(profile.get('gpu_admission_estimate_bytes', 0), 512 * 1024 ** 2) * args.memory_safety_factor
        gpu_free = max(0, snapshot['gpu_free'] - max(args.gpu_reserve_gib * GIB, snapshot['gpu_total'] * args.reserve_fraction))
        slots = min(slots, int(gpu_free // gpu_need))
    else: slots = min(slots, max(0, 1 - active_count))
    if capacity is not None: slots = min(slots, max(0, capacity - active_count))
    write_json(args.output_dir / 'resource_admission.json', {
        'created_at': now(), 'wanted': wanted, 'active_count': active_count, 'admitted_additional': slots,
        'initial_capacity': capacity, 'snapshot': snapshot, 'profile_estimate_host_bytes': host_need,
        'profile_estimate_gpu_bytes': gpu_need, 'cpu_slots': cpu_slots,
        'scope': 'Current free-memory admission using larger scenario profile; initial cap reserves for starting children. Logical CUDA only; no physical GPU mapping guessed.'})
    if slots < 1 and active_count == 0:
        raise RuntimeError('当前空闲GPU/RAM不足以按固定设置启动一个作业；稍后--resume重试，不自动修改batch或科学设置')
    return slots


def training_status(run_dir):
    path = Path(run_dir) / 'results.json'; return load_json(path).get('status') if path.exists() else None


def training_action(args, prepared, scenario, model, seed):
    run_dir = run_path(args, scenario, model, seed); status = training_status(run_dir)
    if status in ('train_complete', 'complete'):
        if not args.resume: raise ValueError('已有完成结果；复用请加--resume：%s/%s/seed%d' % (scenario, model, seed))
        verify_trained_run(run_dir, allow_test=False)
        print('[REUSE] %s/%s/seed%d %s' % (scenario, model, seed, status), flush=True); return None
    if status == 'weights_complete':
        if not args.resume: raise ValueError('已有权重阶段，请加--resume继续导出')
        return None
    command = child_command(args, prepared, scenario, model, seed, 'train')
    if run_dir.exists() and any(run_dir.iterdir()):
        if not args.resume: raise ValueError('已有不完整产物；用--resume继续：' + str(run_dir))
        if (run_dir / 'test').exists() or (args.output_dir / 'test_started.json').exists(): raise ValueError('测试开始后不能恢复权重训练')
        if (run_dir / 'last_checkpoint.pt').exists(): command += ['--resume', 'auto']
        else:
            if any(run_dir.glob('*.pt')): raise ValueError('缺少last checkpoint但仍存在权重；拒绝静默随机重启')
            archived = args.output_dir / 'failed_before_checkpoint' / ('%s_%s_seed%d_%s' % (scenario, model, seed, datetime.now().strftime('%Y%m%d_%H%M%S_%f')))
            archived.parent.mkdir(exist_ok=True); shutil.move(str(run_dir), str(archived))
            print('[ARCHIVE] no committed epoch; identical seeded restart: ' + str(archived), flush=True)
    return start_child(args, command, 'weights_%s_%s_seed%d' % (scenario, model, seed), scenario)

def _class_names(value):
    if isinstance(value, dict): return [value[str(i)] if str(i) in value else value[i] for i in range(len(value))]
    return list(value) if isinstance(value, (list, tuple)) else []


def verify_export_files(directory, marker, *, output_policy=None):
    """验证完整导出；同一队列进程仅在文件stat改变后重复读取大数组。"""
    policy = output_policy or marker.get('output_policy', 'full')
    if policy not in ('full', 'compact'): raise ValueError('导出保存策略无效')
    required={'predictions.csv.gz','logits.npy','labels.npy','probabilities.npy','calibrated_probabilities.npy',
              'prediction_metrics.json','analysis/metrics.json'}
    if policy=='full': required.update(('embeddings.npy','attention.npy'))
    if output_policy=='compact' and marker.get('output_policy')!='compact':
        raise ValueError('导出保存策略与冻结compact方案不同')
    records=marker.get('file_integrity')
    if not isinstance(records,dict) or not required.issubset(records):
        raise ValueError('缺少完整导出SHA清单：'+str(directory))
    for name,record in records.items():
        part=Path(name)
        if part.is_absolute() or '..' in part.parts:raise ValueError('非法导出清单路径')
        path=Path(directory)/part
        if not path.is_file():raise ValueError('缺少导出文件：'+str(path))
        stat=path.stat()
        if stat.st_size!=record['size_bytes']:raise ValueError('导出文件大小改变：'+str(path))
        key=(str(path.resolve()),stat.st_size,stat.st_mtime_ns,stat.st_ctime_ns)
        digest=VERIFIED_EXPORT_FILES.get(key)
        if digest is None:
            digest=sha256(path);VERIFIED_EXPORT_FILES[key]=digest
        if digest!=record['sha256']:raise ValueError('导出文件SHA改变：'+str(path))


def verify_trained_run(run_dir, allow_test=False):
    run_dir = Path(run_dir); model = run_dir.parent.name; scenario_dir = run_dir.parent.parent.parent; scenario = scenario_dir.name
    if scenario not in SCENARIOS or model not in MODELS or scenario_dir.parent.name != 'scenarios':
        raise ValueError('运行目录不是固定scenario/model布局')
    output = scenario_dir.parent.parent; plan = load_json(output / 'experiment_plan.json'); result = load_json(run_dir / 'results.json')
    valid_status = ('train_complete', 'complete') if allow_test else ('train_complete',)
    if result.get('status') not in valid_status or (not allow_test and result.get('test_evaluated')):
        raise ValueError('运行未完成train/val，或test状态不符：' + str(run_dir))
    config = load_json(run_dir / 'resolved_config.json'); seed = int(run_dir.name.removeprefix('seed_'))
    if (config.get('numerical_precision_policy') != NUMERICAL_PRECISION_POLICY
            or plan.get('numerical_precision_policy') != NUMERICAL_PRECISION_POLICY):
        raise ValueError('模型/冻结方案数值精度策略不一致')
    if seed not in SEEDS or result.get('seed') != seed or config.get('seed') != seed: raise ValueError('运行seed与目录不一致')
    if config.get('scenario') != scenario or result.get('scenario') != scenario: raise ValueError('运行scenario或label4身份不一致')
    if _class_names(config.get('class_names')) != CLASS_NAMES[scenario] or _class_names(result.get('class_names')) != CLASS_NAMES[scenario]:
        raise ValueError('五类标签定义不一致')
    expected_config = plan['model_configurations'][model]
    for key, value in expected_config.items():
        if config.get(key) != value: raise ValueError('已训练科学设置与计划不同：' + key)
    if result.get('model_type') != expected_config['model_type']: raise ValueError('架构与固定模型不一致')
    if plan.get('output_policy')!=OUTPUT_POLICY or config.get('output_policy')!=OUTPUT_POLICY or result.get('output_policy')!=OUTPUT_POLICY:
        raise ValueError('模型结果保存策略与compact方案不一致')
    if not all(config.get(key) is True for key in ('no_epoch_logits','no_embeddings','no_attention')):
        raise ValueError('compact结果仍请求保留已省略的大型产物')
    if config.get('model_kwargs', {}).get('target_index') != 3 or config.get('num_classes', 5) != 5:
        raise ValueError('目标位点或分类输出不符')
    snapshot = config.get('source_snapshot', {}).get('files_sha256', {})
    for name, digest in snapshot.items():
        if plan['source_code_sha256'].get('engine/' + name) != digest: raise ValueError('训练源码快照与冻结引擎不一致：' + name)
    if not {'train_all.py','model_benchmark_all.py','IterDataset.py','config_6_all_6.py'}.issubset(snapshot): raise ValueError('缺少训练源码快照')
    manifest_path = scenario_dir / 'prepared_data' / 'prepare_manifest.json'; manifest = load_json(manifest_path)
    if sha256(manifest_path) != plan['prepared_manifest_sha256'][scenario]: raise ValueError('预备输入清单改变')
    for split, filename in (('train','train'),('val','validation'),('test','test')):
        for field, name in (('feature',filename+'.csv'),('metadata',filename+'_meta.tsv')):
            if config['input_fingerprints'][split][field]['sha256'] != manifest['files'][name]['sha256']:
                raise ValueError('输入SHA与固定数据不同：' + scenario + '/' + name)
    checkpoint = sha256(run_dir / 'best_checkpoint.pt'); calibration = load_json(run_dir / 'calibration.json')
    if result['provenance']['checkpoint_sha256'] != checkpoint or calibration['checkpoint_sha256'] != checkpoint:
        raise ValueError('checkpoint/calibration/results SHA不一致')
    temperature = calibration.get('temperature')
    if calibration.get('fitted_on') != 'val' or not isinstance(temperature,(int,float)) or not math.isfinite(temperature) or temperature <= 0:
        raise ValueError('校准不是validation拟合')
    # Full export hashes are checked by engine finalize/evaluate reuse; coordinator checks the fixed identities below.
    for split in ('val',):
        marker = load_json(run_dir / split / 'export_complete.json')
        if marker['checkpoint_sha256'] != checkpoint or marker['temperature'] != temperature: raise ValueError('导出模型身份不一致')
        verify_export_files(run_dir/split,marker,output_policy=OUTPUT_POLICY)
    return {'scenario':scenario,'model':model,'seed':seed,'best_epoch':result['best_epoch'],'checkpoint_sha256':checkpoint,
            'state_dict_sha256':sha256(run_dir/'best_state_dict.pt'),'calibration_sha256':sha256(run_dir/'calibration.json'),
            'temperature':temperature,'resolved_config_sha256':sha256(run_dir/'resolved_config.json')}

def train_jobs(args, prepared, jobs, profile=None):
    if (args.output_dir/'evaluation_plan.json').exists() or (args.output_dir/'test_started.json').exists():
        raise ValueError('24模型已冻结或测试已开始，训练阶段已结束')
    pending=list(jobs); active=[]; initial_capacity=None; failure=None
    while pending or active:
        if STOP: raise InterruptedError('收到停止请求')
        while pending and training_status(run_path(args,*pending[0])) in ('train_complete','complete','weights_complete'):
            scenario,model,seed=pending.pop(0);training_action(args,prepared,scenario,model,seed)
        if pending:
            if profile is None: raise RuntimeError('待训练任务缺少实测profile')
            slots=admission(args,profile,len(pending),len(active),initial_capacity)
            if initial_capacity is None:
                initial_capacity=slots;print('[RESOURCE] initial_capacity=%d max_parallel=%d device=%s' % (slots,args.max_parallel,profile['device']),flush=True)
            for _ in range(slots):
                scenario,model,seed=pending.pop(0);handle=training_action(args,prepared,scenario,model,seed)
                if handle: active.append((scenario,model,seed,handle))
        for scenario,model,seed,handle in list(active):
            progress_message(handle)
            if handle[0].poll() is not None:
                code=finish_child(handle);active.remove((scenario,model,seed,handle))
                if code: failure=RuntimeError('%s/%s/seed%d训练退出码%d；日志%s；检查后--resume' % (scenario,model,seed,code,handle[2]['log']))
        if shutil.disk_usage(args.output_dir).free < args.disk_reserve_gib*GIB:failure=OSError(28,'空间低于预留；清理后--resume')
        if failure:
            for *_,handle in active:
                if handle[0].poll() is None:handle[0].terminate()
            for *_,handle in active:finish_child(handle)
            raise failure
        if active:time.sleep(min(args.poll_seconds,3))
    for scenario,model,seed in jobs:
        run_dir=run_path(args,scenario,model,seed)
        if training_status(run_dir)=='weights_complete':
            run_child(args,child_command(args,prepared,scenario,model,seed,'finalize'),'finalize_%s_%s_seed%d' % (scenario,model,seed),scenario)
        correct_scope_notes(run_dir,scenario);verify_trained_run(run_dir,allow_test=False)

def freeze_models(args):
    records=[verify_trained_run(run_path(args,s,m,seed),allow_test=(args.output_dir/'test_started.json').exists()) for s,m,seed in phase_jobs('freeze')]
    plan={'version':VERSION,'fixed_scenarios':list(SCENARIOS),'fixed_models':list(MODELS),'fixed_seeds':list(SEEDS),'models':records,
          'experiment_plan_sha256':sha256(args.output_dir/'experiment_plan.json'),
          'prepared_manifest_sha256':{s:sha256(scenario_root(args,s)/'prepared_data'/'prepare_manifest.json') for s in SCENARIOS},
          'decision':'Separate five-class argmax; raw and validation-temperature calibrated probabilities',
          'scope':'All 24 seeded models fixed before first test inference; no test threshold/model selection'}
    path=args.output_dir/'evaluation_plan.json'
    if path.exists():
        previous=load_json(path)
        if {k:v for k,v in previous.items() if k!='frozen_at'}!=plan:raise ValueError('冻结模型/温度/数据改变，拒绝覆盖')
    else:
        if (args.output_dir/'test_started.json').exists():raise ValueError('test已开始但冻结清单缺失')
        write_json(path,dict(plan,frozen_at=now()))
    print('[FROZEN] all 24 checkpoints and validation temperatures; no test prediction',flush=True)

def evaluate_frozen(args, prepared):
    plan_path=args.output_dir/'evaluation_plan.json'
    if not plan_path.exists():raise ValueError('请先完成24项训练并freeze')
    freeze_models(args);marker=args.output_dir/'test_started.json';identity=sha256(plan_path)
    if marker.exists():
        if load_json(marker)['evaluation_plan_sha256']!=identity:raise ValueError('test开始后冻结清单改变')
    else:write_json(marker,{'started_at':now(),'evaluation_plan_sha256':identity,'policy':'All24 fixed before any test prediction'})
    frozen_records={(v['scenario'],v['model'],v['seed']):v for v in load_json(plan_path)['models']}
    for scenario,model,seed in phase_jobs('evaluate'):
        run_dir=run_path(args,scenario,model,seed)
        prior=load_json_or_empty(run_dir/'results.json')
        if prior.get('test_evaluated'):
            if prior.get('status')!='complete':raise ValueError('已有test标记但运行状态不完整：'+str(run_dir))
            exported=load_json(run_dir/'test/export_complete.json');record=frozen_records[(scenario,model,seed)]
            if exported.get('checkpoint_sha256')!=record['checkpoint_sha256'] or exported.get('temperature')!=record['temperature']:
                raise ValueError('已有test导出身份与冻结模型不一致')
            verify_export_files(run_dir/'test',exported,output_policy=OUTPUT_POLICY)
            print('[REUSE TEST] %s/%s/seed%d verified; no inference' % (scenario,model,seed),flush=True)
            continue
        run_child(args,child_command(args,prepared,scenario,model,seed,'evaluate'),'evaluate_%s_%s_seed%d' % (scenario,model,seed),scenario)
        correct_scope_notes(run_dir,scenario);result=load_json(run_dir/'results.json')
        if not result.get('test_evaluated') or result.get('status')!='complete':raise RuntimeError('test评价未完成：'+str(run_dir))
    write_json(args.output_dir/'evaluation_complete.json',{'status':'complete','completed_at':now(),'evaluation_plan_sha256':identity,
               'fixed_scenarios':list(SCENARIOS),'fixed_models':list(MODELS),'seeds':list(SEEDS)})

def _summary(args):
    from summarize_fiveclass24 import summarize_and_package
    result=summarize_and_package(args.output_dir,HERE);print('[PACKAGED] '+str(result['core_path']),flush=True);return result

def run(args):
    global LOCK_FD
    args=validate_args(args)
    if args.dry_run:
        print(json.dumps({'phase':args.phase,'jobs':[{'scenario':s,'model':m,'seed':seed} for s,m,seed in phase_jobs(args.phase)],
            'class_names':CLASS_NAMES,'model_configurations':{m:model_configuration(m) for m in MODELS},
            'output_policy':OUTPUT_POLICY,'output_contract':OUTPUT_CONTRACT,
            'test_inference':args.phase in ('evaluate','all'),'source_dir':str(args.source_dir),'output_dir':str(args.output_dir),
            'all24_frozen_before_test':True,'no_data_read_or_process_started':True},ensure_ascii=False,indent=2));return
    args.output_dir.mkdir(parents=True,exist_ok=True)
    with (args.output_dir/'.pipeline.lock').open('a') as lock:
        try:fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
        except BlockingIOError:raise RuntimeError('同一输出目录已有任务运行')
        LOCK_FD=lock.fileno();orphan_check(args)
        if args.phase=='summary':return _summary(args)
        from validate_precision_fix import run_gate
        run_gate(args, lock_fd=LOCK_FD)
        frozen=(args.output_dir/'evaluation_plan.json').exists() or (args.output_dir/'test_started.json').exists()
        if args.phase=='train' and frozen:raise ValueError('24模型已冻结，不允许再训练')
        check_disk(args)
        from prepare_data import prepare_inputs, audit_cross_task_read_splits
        prepared={scenario:prepare_inputs(args.source_dir,scenario_root(args,scenario),scenario,resume=args.resume,
                  disk_reserve_gib=args.disk_reserve_gib,reference_dir=REFERENCE_DIR) for scenario in SCENARIOS}
        audit_cross_task_read_splits(prepared,args.output_dir)
        commit_plan(args,prepared)
        print('[VERSION] Figure3 fiveclass24 %s phase=%s' % (VERSION,args.phase),flush=True)
        training_requested=args.phase=='train' or (args.phase=='all' and not frozen)
        if args.phase=='prepare' or training_requested:
            jobs=phase_jobs('train') if training_requested else []
            check_disk(args,jobs,prepared=prepared,include_test=args.phase=='all')
            for scenario in SCENARIOS:preflight(args,prepared,scenario)
            if training_requested:
                needing=any(training_status(run_path(args,*job)) not in ('train_complete','complete','weights_complete') for job in jobs)
                profile=combined_profile({s+'/'+m:measure_profile(args,prepared,s,m) for s in SCENARIOS for m in MODELS}) if needing else None
                train_jobs(args,prepared,jobs,profile)
        if args.phase in ('freeze','all'):freeze_models(args)
        if args.phase in ('evaluate','all'):
            check_disk(args,prepared=prepared,include_test=True);evaluate_frozen(args,prepared)
        write_json(args.output_dir/'stage_status.json',{'phase':args.phase,'status':'complete','finished_at':now(),
            'test_evaluated':(args.output_dir/'evaluation_complete.json').exists()})
        result=_summary(args);print('[FULL PASS] phase=%s output=%s' % (args.phase,args.output_dir),flush=True);return result

def main(argv=None):
    for sig in (signal.SIGINT, signal.SIGTERM): signal.signal(sig, request_stop)
    return run(parser().parse_args(argv))


if __name__ == '__main__':
    try: main()
    except SystemExit: raise
    except BaseException as error:
        for process in list(CHILDREN.values()):
            if process.poll() is None: process.terminate()
        for process in list(CHILDREN.values()):
            try: process.wait(timeout=30)
            except subprocess.TimeoutExpired: process.kill(); process.wait()
        print('[ERROR] %s: %s' % (type(error).__name__, error), file=sys.stderr, flush=True)
        sys.exit(2 if isinstance(error, (InterruptedError, KeyboardInterrupt)) else 1)
