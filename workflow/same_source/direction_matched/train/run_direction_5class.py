#!/usr/bin/env python3
"""同来源方向匹配五分类：固定最终 GatedConv；默认只训练/验证两个 seed42。"""
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

VERSION = '1.0.0'
HERE = Path(__file__).resolve().parent
ENGINE = HERE / 'engine'
REFERENCE_DIR = HERE / 'reference'
SCENARIOS = ('X_plus', 'Y_minus')
SEEDS = (42, 2024, 2025)
CLASS_NAMES = {'X_plus': ['A', 'T', 'C', 'G', 'X'], 'Y_minus': ['A', 'T', 'C', 'G', 'Y']}
DEFAULT_SOURCE = 'cplx_all_direction_matched_5class_v1_0_0'
OUTPUT_NAME = 'gatedconv_direction_5class_v1_0_0'
SCIENTIFIC = {
    'model_type': 'litemamba', 'litemamba_variant': 'legacy_gatedconv',
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
GIB = 2 ** 30
STOP = False
CHILDREN = {}
LOCK_FD = None


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


def run_path(args, scenario, seed):
    if seed not in SEEDS: raise ValueError('seed不在冻结清单中')
    return scenario_root(args, scenario) / 'runs' / ('seed_' + str(seed))


def phase_jobs(phase):
    seeds = SEEDS[:1] if phase == 'pilot' else SEEDS[1:] if phase == 'confirm' else SEEDS
    return [(scenario, seed) for seed in seeds for scenario in SCENARIOS]


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
    result.add_argument('--version', action='version', version='CPLX direction-matched fiveclass GatedConv ' + VERSION)
    result.add_argument('--source_dir', default=DEFAULT_SOURCE)
    result.add_argument('--output_dir')
    result.add_argument('--phase', choices=('prepare', 'pilot', 'confirm', 'freeze', 'evaluate', 'summary'), default='pilot')
    result.add_argument('--resume', action='store_true')
    result.add_argument('--device', default='auto')
    result.add_argument('--cpu_threads', type=int, default=4)
    result.add_argument('--workers', type=int, default=0)
    result.add_argument('--max_parallel', type=int, choices=(1, 2, 3, 4), default=4,
                        help='权重训练硬上限；按两个任务实测资源及当前空闲量自动减少；导出/测试串行')
    result.add_argument('--disk_reserve_gib', type=float, default=4)
    result.add_argument('--gpu_reserve_gib', type=float, default=2)
    result.add_argument('--ram_reserve_gib', type=float, default=3)
    result.add_argument('--reserve_fraction', type=float, default=.10)
    result.add_argument('--memory_safety_factor', type=float, default=1.35)
    result.add_argument('--poll_seconds', type=float, default=3)
    result.add_argument('--dry_run', action='store_true', help='只打印固定阶段；不读取数据或启动子进程')
    return result


def validate_args(args):
    args.source_dir = Path(args.source_dir).expanduser().resolve()
    args.output_dir = Path(args.output_dir or args.source_dir / OUTPUT_NAME).expanduser().resolve()
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
    files = [HERE / name for name in ('run_direction_5class.py', 'prepare_data.py', 'summarize_direction_5class.py', 'reference6.py')]
    files += sorted(ENGINE.glob('*.py')) + sorted(REFERENCE_DIR.rglob('*'))
    return {(str(path.relative_to(HERE)) if path.is_relative_to(HERE)
             else 'reference/' + str(path.relative_to(REFERENCE_DIR))): sha256(path)
            for path in files if path.is_file()}


def make_plan(args, prepared):
    summary = load_json(REFERENCE_DIR / 'extract_core' / 'extraction_summary.json')
    return {
        'version': VERSION, 'model_name': 'GatedConv', 'num_classes': 5,
        'fixed_scenarios': list(SCENARIOS), 'class_names': CLASS_NAMES,
        'fixed_seeds': list(SEEDS), 'scientific_configuration': dict(SCIENTIFIC),
        'initialization': 'independent seeded random initialization; no pretrained weights',
        'training_order': 'both scenario seed42 train/val pilots, then both scenario seeds2024/2025; fixed settings',
        'test_policy': 'No test prediction before all six checkpoints and validation temperatures are frozen; no test-dependent seed or architecture selection.',
        'source_dir': str(args.source_dir), 'output_dir': str(args.output_dir),
        'prepared_manifest_sha256': {scenario: sha256(prepared[scenario]['manifest_path']) for scenario in SCENARIOS},
        'source_code_sha256': source_identity(),
        'source_inputs': {scenario: {kind: summary['scenarios'][scenario][kind]['sha256'] for kind in ('features', 'metadata')} for scenario in SCENARIOS},
        'planned_test_rows': {scenario: summary['scenarios'][scenario]['rows_by_split']['test'] for scenario in SCENARIOS},
        'batch_id': 'CPLX-XNA-all', 'physical_run_id': 'unknown',
        'scope': 'Supplementary same-source, BAM-reference-assigned, direction/context-matched five-class read holdout. Natural/NCB positions differ; not template/run/batch holdout or within-direction X-vs-Y discrimination.',
        'target_mask_index': 3, 'seq_len': 6,
        'amp_policy': 'CUDA AMP enabled; CPU FP32, as in frozen final GatedConv training',
    }


def commit_plan(args, prepared):
    path = args.output_dir / 'experiment_plan.json'; plan = make_plan(args, prepared)
    if path.exists():
        if load_json(path) != plan: raise ValueError('输入、源码或固定实验方案改变；不能混入已有结果目录')
    else: write_json(path, plan)
    return plan


def check_disk(args, jobs=(), include_prepare=False):
    summary = load_json(REFERENCE_DIR / 'extract_core' / 'extraction_summary.json'); budget = 0
    for scenario, seed in jobs:
        if training_status(run_path(args, scenario, seed)) in ('train_complete', 'complete'): continue
        rec = summary['scenarios'][scenario]
        budget += rec['rows'] * 7000 + rec['rows_by_split']['validation'] * 40 * SCIENTIFIC['epochs'] + 150 * 1024 ** 2
    if include_prepare: budget += summary['rows'] * 3500
    needed = budget + args.disk_reserve_gib * GIB; free = shutil.disk_usage(args.output_dir).free
    if free < needed:
        raise OSError(28, '输出盘剩余%.2f GiB，当前阶段保守预算%.2f GiB（含%.2f GiB余量）；清理后使用--resume' %
                      (free / GIB, needed / GIB, args.disk_reserve_gib))
    print('[DISK] free=%.2f GiB current_stage_budget=%.2f GiB' % (free / GIB, needed / GIB), flush=True)


def child_command(args, prepared, scenario, seed, mode, run_dir=None):
    run_dir = Path(run_dir or run_path(args, scenario, seed))
    command = [sys.executable, '-u', str(ENGINE / 'train_all.py'), '--mode', mode,
               '--data_dir', str(prepared[scenario]['data_dir']), '--cache_dir', str(scenario_root(args, scenario) / 'feature_cache'),
               '--output_dir', str(run_dir), '--device', args.device, '--cpu_threads', str(args.cpu_threads),
               '--workers', str(args.workers), '--concurrent_training']
    if mode in ('train', 'preflight', 'profile'):
        for key, value in SCIENTIFIC.items(): command += ['--' + key, str(value)]
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
                PYTHONUNBUFFERED='1', CPLX_FIVECLASS_SCENARIO=scenario)


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
    """保存实验范围；不后处理更改任何分类和统计数值。"""
    write_json(Path(run_dir) / 'direction_matched_scope.json', {
        'scenario': scenario, 'class_names': CLASS_NAMES[scenario], 'source_dataset': 'CPLX-XNA-all',
        'real_read_ids_all_classes': True, 'physical_run_id': 'unknown',
        'scope': 'Same-source, same-direction, BAM-reference-assigned/context-matched internal read holdout.',
        'limitations': ['Natural and NCB positions differ.', 'Not physical-template verification or template/run/batch holdout.',
                        'This task contains one NCB type; no within-direction X-vs-Y discrimination claim.',
                        'The original six-class test was already inspected; supplementary sensitivity evidence.']})


def preflight(args, prepared, scenario):
    directory = scenario_root(args, scenario) / 'preflight'
    run_child(args, child_command(args, prepared, scenario, 42, 'preflight', directory), 'preflight_' + scenario, scenario)
    if not (directory / 'preflight.json').exists() or load_json(directory / 'preflight.json').get('status') != 'complete':
        raise RuntimeError('预检未完整完成：' + scenario)
    audit = load_json(directory / 'data_audit.json')
    if audit.get('status') != 'PASS' or any(audit['read_overlap_counts'].values()): raise ValueError('真实read跨集合审计失败：' + scenario)
    if any(record.get('positive_row_proxy_rows', 0) for record in audit['split_summary'].values()): raise ValueError('本实验不允许阳性行代理')
    correct_scope_notes(directory, scenario)
    print('[PREFLIGHT PASS] %s real reads, five classes; no test inference' % scenario, flush=True)


def measure_profile(args, prepared, scenario):
    directory = scenario_root(args, scenario) / 'resource_profiles' / datetime.now().strftime('%Y%m%d_%H%M%S_%f')
    run_child(args, child_command(args, prepared, scenario, 42, 'profile', directory), 'resource_profile_' + scenario, scenario)
    profile = load_json(directory / 'resource_profile.json')
    if profile.get('model_kwargs', {}).get('litemamba_variant') != 'legacy_gatedconv' or profile.get('actual_optimizer_steps', 0) < 8:
        raise ValueError('资源profile未测满8次真实AdamW更新或模型不匹配')
    if _class_names(profile.get('class_names')) != CLASS_NAMES[scenario]: raise ValueError('资源profile类别与任务不符')
    correct_scope_notes(directory, scenario); return profile


def combined_profile(profiles):
    if set(profiles) != set(SCENARIOS): raise ValueError('两个任务均须完成资源实测')
    devices = {profile['device'] for profile in profiles.values()}
    if len(devices) != 1: raise ValueError('两个任务的逻辑设备不同')
    return {'device': devices.pop(),
            'host_admission_estimate_bytes': max(p.get('host_admission_estimate_bytes', 0) for p in profiles.values()),
            'gpu_admission_estimate_bytes': max(p.get('gpu_admission_estimate_bytes', 0) for p in profiles.values()),
            'scenario_profiles': profiles}


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


def training_action(args, prepared, scenario, seed):
    run_dir = run_path(args, scenario, seed); status = training_status(run_dir)
    if status in ('train_complete', 'complete'):
        if not args.resume: raise ValueError('已有完成结果；复用请加--resume：%s/seed%d' % (scenario, seed))
        verify_trained_run(run_dir, allow_test=False)
        print('[REUSE] %s/seed%d %s' % (scenario, seed, status), flush=True); return None
    if status == 'weights_complete':
        if not args.resume: raise ValueError('已有权重阶段，请加--resume继续导出')
        return None
    command = child_command(args, prepared, scenario, seed, 'train')
    if run_dir.exists() and any(run_dir.iterdir()):
        if not args.resume: raise ValueError('已有不完整产物；用--resume继续：' + str(run_dir))
        if (run_dir / 'test').exists() or (args.output_dir / 'test_started.json').exists(): raise ValueError('测试开始后不能恢复权重训练')
        if (run_dir / 'last_checkpoint.pt').exists(): command += ['--resume', 'auto']
        else:
            if any(run_dir.glob('*.pt')): raise ValueError('缺少last checkpoint但仍存在权重；拒绝静默随机重启，请恢复完整检查点')
            archived = args.output_dir / 'failed_before_checkpoint' / ('%s_seed%d_%s' % (scenario, seed, datetime.now().strftime('%Y%m%d_%H%M%S_%f')))
            archived.parent.mkdir(exist_ok=True); shutil.move(str(run_dir), str(archived))
            print('[ARCHIVE] no committed epoch; identical seeded restart: ' + str(archived), flush=True)
    return start_child(args, command, 'weights_%s_seed%d' % (scenario, seed), scenario)


def _class_names(value):
    if isinstance(value, dict): return [value[str(i)] if str(i) in value else value[i] for i in range(len(value))]
    return list(value) if isinstance(value, (list, tuple)) else []


def verify_trained_run(run_dir, allow_test=False):
    run_dir = Path(run_dir); scenario_dir = run_dir.parent.parent; scenario = scenario_dir.name
    if scenario not in SCENARIOS or scenario_dir.parent.name != 'scenarios': raise ValueError('运行目录不是固定scenario布局')
    output = scenario_dir.parent.parent; plan = load_json(output / 'experiment_plan.json'); result = load_json(run_dir / 'results.json')
    valid_status = ('train_complete', 'complete') if allow_test else ('train_complete',)
    if result.get('status') not in valid_status or (not allow_test and result.get('test_evaluated')):
        raise ValueError('运行未完成train/val，或test状态不符合阶段：' + str(run_dir))
    config = load_json(run_dir / 'resolved_config.json'); seed = int(run_dir.name.removeprefix('seed_'))
    if seed not in SEEDS or result.get('seed') != seed or config.get('seed') != seed: raise ValueError('运行seed与固定目录不一致')
    if config.get('scenario') != scenario or result.get('scenario') != scenario: raise ValueError('运行scenario或label4身份不一致')
    if _class_names(config.get('class_names')) != CLASS_NAMES[scenario] or _class_names(result.get('class_names')) != CLASS_NAMES[scenario]:
        raise ValueError('运行五类标签定义不一致')
    for key, value in plan['scientific_configuration'].items():
        if config.get(key) != value: raise ValueError('已训练科学设置与计划不同：' + key)
    if result.get('litemamba_variant') != 'legacy_gatedconv' or result.get('readout') != 'target_attention' or result.get('feature_scheme') != 'full':
        raise ValueError('已有运行不是最终完整输入GatedConv')
    if config.get('model_kwargs', {}).get('target_index') != 3: raise ValueError('运行目标位点不是第四位')
    if config.get('num_classes', 5) != 5: raise ValueError('分类输出不是五类')
    snapshot = config.get('source_snapshot', {}).get('files_sha256', {})
    for name, digest in snapshot.items():
        if plan['source_code_sha256'].get('engine/' + name) != digest: raise ValueError('训练源码快照与冻结引擎不一致：' + name)
    if not {'train_all.py', 'model_benchmark_all.py', 'IterDataset.py', 'config_6_all_6.py'}.issubset(snapshot): raise ValueError('缺少完整训练源码快照')
    prepared = scenario_dir / 'prepared_data'; manifest = load_json(prepared / 'prepare_manifest.json')
    if sha256(prepared / 'prepare_manifest.json') != plan['prepared_manifest_sha256'][scenario]: raise ValueError('已训练预备输入清单改变')
    for split, filename in (('train', 'train'), ('val', 'validation'), ('test', 'test')):
        for field, name in (('feature', filename + '.csv'), ('metadata', filename + '_meta.tsv')):
            expected = manifest['files'][name]['sha256']
            if config['input_fingerprints'][split][field]['sha256'] != expected or sha256(prepared / name) != expected:
                raise ValueError('已训练输入SHA与固定数据不同：' + scenario + '/' + name)
    checkpoint = sha256(run_dir / 'best_checkpoint.pt'); calibration = load_json(run_dir / 'calibration.json')
    if result['provenance']['checkpoint_sha256'] != checkpoint or calibration['checkpoint_sha256'] != checkpoint: raise ValueError('checkpoint/calibration/results SHA不一致')
    temperature = calibration.get('temperature')
    if calibration.get('fitted_on') != 'val' or not isinstance(temperature, (int, float)) or not math.isfinite(temperature) or temperature <= 0:
        raise ValueError('校准参数未明确在validation拟合')
    for split in ('train', 'val'):
        marker = load_json(run_dir / split / 'export_complete.json')
        if marker['checkpoint_sha256'] != checkpoint or marker['temperature'] != temperature: raise ValueError('train/val导出模型身份不一致')
        for name in ('predictions.csv.gz', 'logits.npy', 'labels.npy', 'probabilities.npy', 'calibrated_probabilities.npy',
                     'embeddings.npy', 'attention.npy', 'prediction_metrics.json', 'analysis/metrics.json'):
            if not (run_dir / split / name).is_file(): raise ValueError('缺少完整导出文件：' + str(run_dir / split / name))
    return {'scenario': scenario, 'seed': seed, 'best_epoch': result['best_epoch'], 'checkpoint_sha256': checkpoint,
            'state_dict_sha256': sha256(run_dir / 'best_state_dict.pt'), 'calibration_sha256': sha256(run_dir / 'calibration.json'),
            'temperature': temperature, 'resolved_config_sha256': sha256(run_dir / 'resolved_config.json')}


def train_jobs(args, prepared, jobs, profile=None):
    if (args.output_dir / 'evaluation_plan.json').exists() or (args.output_dir / 'test_started.json').exists(): raise ValueError('六模型已冻结或测试已开始，训练阶段已结束')
    pending = list(jobs); active = []; initial_capacity = None; failure = None
    while pending or active:
        if STOP: raise InterruptedError('收到停止请求')
        while pending and training_status(run_path(args, *pending[0])) in ('train_complete', 'complete', 'weights_complete'):
            scenario, seed = pending.pop(0); training_action(args, prepared, scenario, seed)
        if pending:
            if profile is None: raise RuntimeError('待训练任务缺少实测资源profile')
            slots = admission(args, profile, len(pending), len(active), initial_capacity)
            if initial_capacity is None:
                initial_capacity = slots
                print('[RESOURCE] initial_capacity=%d max_parallel=%d device=%s' % (initial_capacity, args.max_parallel, profile['device']), flush=True)
            for _ in range(slots):
                scenario, seed = pending.pop(0); handle = training_action(args, prepared, scenario, seed)
                if handle: active.append((scenario, seed, handle))
        for scenario, seed, handle in list(active):
            progress_message(handle)
            if handle[0].poll() is not None:
                code = finish_child(handle); active.remove((scenario, seed, handle))
                if code: failure = RuntimeError('%s/seed%d训练退出码%d；日志%s；检查后--resume' % (scenario, seed, code, handle[2]['log']))
        if shutil.disk_usage(args.output_dir).free < args.disk_reserve_gib * GIB: failure = OSError(28, '空间低于预留值；清理后--resume恢复最近完整epoch')
        if failure:
            for _, _, handle in active:
                if handle[0].poll() is None: handle[0].terminate()
            for _, _, handle in active: finish_child(handle)
            raise failure
        if active: time.sleep(min(args.poll_seconds, 3))
    for scenario, seed in jobs:
        run_dir = run_path(args, scenario, seed)
        if training_status(run_dir) == 'weights_complete':
            run_child(args, child_command(args, prepared, scenario, seed, 'finalize'), 'finalize_%s_seed%d' % (scenario, seed), scenario)
        correct_scope_notes(run_dir, scenario); verify_trained_run(run_dir, allow_test=False)


def freeze_models(args):
    records = [verify_trained_run(run_path(args, scenario, seed), allow_test=(args.output_dir / 'test_started.json').exists()) for scenario, seed in phase_jobs('freeze')]
    plan = {'version': VERSION, 'fixed_scenarios': list(SCENARIOS), 'fixed_seeds': list(SEEDS), 'models': records,
            'experiment_plan_sha256': sha256(args.output_dir / 'experiment_plan.json'),
            'prepared_manifest_sha256': {scenario: sha256(scenario_root(args, scenario) / 'prepared_data' / 'prepare_manifest.json') for scenario in SCENARIOS},
            'decision': 'Separate five-class argmax; raw and validation-temperature calibrated probabilities',
            'scope': 'All six independent seeded models fixed before first test inference; no test threshold/model selection'}
    path = args.output_dir / 'evaluation_plan.json'
    if path.exists():
        previous = load_json(path)
        if {key: value for key, value in previous.items() if key != 'frozen_at'} != plan: raise ValueError('冻结模型/温度/数据已改变，拒绝覆盖evaluation_plan')
    else:
        if (args.output_dir / 'test_started.json').exists(): raise ValueError('test已开始但冻结清单缺失')
        write_json(path, dict(plan, frozen_at=now()))
    print('[FROZEN] all 6 checkpoints and validation temperatures; freeze makes no test prediction', flush=True)


def evaluate_frozen(args, prepared):
    plan_path = args.output_dir / 'evaluation_plan.json'
    if not plan_path.exists(): raise ValueError('请先完成两个任务的3种子训练，再显式运行--phase freeze')
    freeze_models(args); marker = args.output_dir / 'test_started.json'; identity = sha256(plan_path)
    if marker.exists():
        if load_json(marker)['evaluation_plan_sha256'] != identity: raise ValueError('test开始后冻结清单改变')
    else: write_json(marker, {'started_at': now(), 'evaluation_plan_sha256': identity, 'policy': 'All six checkpoints fixed before first test prediction'})
    for scenario, seed in phase_jobs('evaluate'):
        run_dir = run_path(args, scenario, seed)
        run_child(args, child_command(args, prepared, scenario, seed, 'evaluate'), 'evaluate_%s_seed%d' % (scenario, seed), scenario)
        correct_scope_notes(run_dir, scenario); result = load_json(run_dir / 'results.json')
        if not result.get('test_evaluated') or result.get('status') != 'complete': raise RuntimeError('test评价未完成：%s/seed%d' % (scenario, seed))
    write_json(args.output_dir / 'evaluation_complete.json', {'status': 'complete', 'completed_at': now(), 'evaluation_plan_sha256': identity,
               'fixed_scenarios': list(SCENARIOS), 'seeds': list(SEEDS)})


def _summary(args):
    from summarize_direction_5class import summarize_and_package
    result = summarize_and_package(args.output_dir, HERE); print('[PACKAGED] ' + str(result['core_path']), flush=True); return result


def run(args):
    global LOCK_FD
    args = validate_args(args)
    if args.dry_run:
        print(json.dumps({'phase': args.phase, 'jobs': [{'scenario': s, 'seed': seed} for s, seed in phase_jobs(args.phase)],
                          'class_names': CLASS_NAMES, 'scientific_configuration': SCIENTIFIC, 'test_inference': args.phase == 'evaluate',
                          'source_dir': str(args.source_dir), 'output_dir': str(args.output_dir), 'no_data_read_or_process_started': True}, ensure_ascii=False, indent=2)); return
    args.output_dir.mkdir(parents=True, exist_ok=True)
    with (args.output_dir / '.pipeline.lock').open('a') as lock:
        try: fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError: raise RuntimeError('同一输出目录已有任务运行')
        LOCK_FD = lock.fileno(); orphan_check(args)
        if args.phase == 'summary': return _summary(args)
        if args.phase in ('pilot', 'confirm') and ((args.output_dir / 'evaluation_plan.json').exists() or (args.output_dir / 'test_started.json').exists()):
            raise ValueError('六模型已冻结或测试已开始，不允许重新进入训练阶段')
        jobs = phase_jobs(args.phase) if args.phase in ('pilot', 'confirm') else []
        check_disk(args, jobs, include_prepare=any(not (scenario_root(args, s) / 'prepared_data').exists() for s in SCENARIOS))
        from prepare_data import prepare_inputs
        prepared = {scenario: prepare_inputs(args.source_dir, scenario_root(args, scenario), scenario, resume=args.resume,
                    disk_reserve_gib=args.disk_reserve_gib, reference_dir=REFERENCE_DIR) for scenario in SCENARIOS}
        commit_plan(args, prepared)
        print('[VERSION] CPLX direction-matched fiveclass GatedConv %s phase=%s' % (VERSION, args.phase), flush=True)
        if args.phase in ('prepare', 'pilot', 'confirm'):
            for scenario in SCENARIOS: preflight(args, prepared, scenario)
            if args.phase != 'prepare':
                if args.phase == 'confirm':
                    for scenario in SCENARIOS: verify_trained_run(run_path(args, scenario, 42), allow_test=False)
                needing_weights = any(training_status(run_path(args, scenario, seed)) not in ('train_complete', 'complete', 'weights_complete') for scenario, seed in jobs)
                profile = combined_profile({scenario: measure_profile(args, prepared, scenario) for scenario in SCENARIOS}) if needing_weights else None
                train_jobs(args, prepared, jobs, profile)
        elif args.phase == 'freeze': freeze_models(args)
        elif args.phase == 'evaluate': evaluate_frozen(args, prepared)
        write_json(args.output_dir / 'stage_status.json', {'phase': args.phase, 'status': 'complete', 'finished_at': now(),
                   'test_evaluated': (args.output_dir / 'evaluation_complete.json').exists()})
        result = _summary(args)
        print('[FULL PASS] phase=%s output=%s' % (args.phase, args.output_dir), flush=True)
        if args.phase == 'pilot': print('[NEXT] both seed42 train/validation pilots complete; no test prediction. Upload the pilot core for review.', flush=True)
        return result


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
