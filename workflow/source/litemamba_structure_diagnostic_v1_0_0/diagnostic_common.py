#!/usr/bin/env python3
"""独立启动 LiteMamba seed42 四项诊断；复用原引擎，仅训练/验证，不评价 test。"""
from __future__ import annotations
import argparse
from datetime import datetime, timezone
import fcntl
import hashlib
import json
import os
from pathlib import Path
import shlex
import shutil
import signal
import subprocess
import sys
import time
VERSION = '1.0.0'
GIB = 1024 ** 3
DEFAULT_ENGINE = '/path/to/revision-code/sixclass_training_auto_v1_0_4'
DEFAULT_DATA = '/path/to/revision-inputs/CPLX-XNA-replenishment/CPLX-XNA-replenishment/fast5_single_guppy/sixclass_overlap_read_split_721'
SCHEMES = ('no_raw', 'no_context', 'no_responses', 'no_statistics')
SOURCE_FILES = (
    'train_all.py',
    'model_benchmark_all.py',
    'config_6_all_6.py',
    'IterDataset.py',
    'analyze_results.py',
    'run_all_4_models.py',
    'resource_scheduler.py',
    'cuda_device_probe.py',
)
SCALAR_KEYS = (
    'epochs',
    'patience',
    'lr',
    'batch_size',
    'raw_signal_len',
    'dwell_divisor',
    'quality_divisor',
    'accum_steps',
    'weight_decay',
    'label_smoothing',
    'clip_grad',
    'monitor',
    'model_dim',
    'num_blocks',
    'state_dim',
    'litemamba_variant',
    'readout',
    'bootstrap',
    'embedding_dtype',
)
BOOL_KEYS = ('no_plots', 'no_embeddings', 'no_epoch_logits', 'save_epoch_predictions')
FIXED_ENGINE_DEFAULTS = {
    'target_index': 3,
    'dropout': 0.1,
    'classifier_dropout': 0.2,
    'ssm_bidirectional': 1,
    'nhead': 4,
    'num_layers': 2,
    'gru_hidden': 64,
}
RESOURCE_BUDGET = {
    'gpu_reserve_gib': 2,
    'ram_reserve_gib': 3,
    'reserve_fraction': 0.1,
    'memory_safety_factor': 1.35,
    'train_ram_extra_gib': 1.0,
    'oom_retries': 1,
}
INPUT_NAMES = {
    'train': {'feature': 'train.csv', 'metadata': 'train_meta.tsv'},
    'val': {'feature': 'validation.csv', 'metadata': 'validation_meta.tsv'},
    'test': {'feature': 'test.csv', 'metadata': 'test_meta.tsv'},
}


def now():
    return datetime.now(timezone.utc).isoformat()


def load_json(path):
    return json.loads(Path(path).read_text(encoding='utf-8'))


def save_json(path, value):
    path = Path(path)
    temporary = path.with_name(path.name + '.tmp.' + str(os.getpid()))
    temporary.write_text(
        json.dumps(value, ensure_ascii=False, indent=2, allow_nan=False) + '\n',
        encoding='utf-8',
    )
    temporary.replace(path)


def sha256(path):
    h = hashlib.sha256()
    with Path(path).open('rb') as stream:
        for block in iter(lambda: stream.read(8 * 1024 * 1024), b''):
            h.update(block)
    return h.hexdigest()


def validate_sources(engine_dir, requested_config):
    expected = requested_config.get('source_sha256', {})
    result = {}
    for name in SOURCE_FILES:
        if name not in expected:
            raise ValueError('baseline缺少源代码SHA：' + name)
        path = Path(engine_dir) / name
        if not path.is_file():
            raise FileNotFoundError(path)
        result[name] = sha256(path)
        if result[name] != expected[name]:
            raise ValueError('源代码SHA不匹配，拒绝启动：' + name)
    return result


def validate_data(data_dir, expected_fingerprints, verify_sha=True):
    """test文件只作内容指纹核对，不解析标签、信号或test指标。"""
    result = {}
    for split, names in INPUT_NAMES.items():
        result[split] = {}
        for kind, name in names.items():
            expected = expected_fingerprints.get(split, {}).get(kind)
            if not expected or len(str(expected.get('sha256', ''))) != 64:
                raise ValueError('baseline缺少六输入SHA：' + split + '/' + kind)
            path = Path(data_dir) / name
            entry = {
                'path': str(path),
                'expected_sha256': expected['sha256'],
                'sha256_verified': False,
                'audit_role': 'fingerprint_only_no_test_evaluation' if split == 'test' else 'training_or_validation_input',
            }
            if path.is_file():
                before = path.stat()
                entry.update(size_bytes=before.st_size, mtime_ns=before.st_mtime_ns)
                if verify_sha:
                    print('[VERIFY SHA] ' + split + '/' + kind + ' ' + str(path), flush=True)
                    size = expected.get('size', expected.get('size_bytes'))
                    if size is not None and before.st_size != size:
                        raise ValueError('数据大小不匹配：' + str(path))
                    actual = sha256(path)
                    after = path.stat()
                    if (before.st_size, before.st_mtime_ns) != (after.st_size, after.st_mtime_ns):
                        raise RuntimeError('核验期间数据变化：' + str(path))
                    if actual != expected['sha256']:
                        raise ValueError('数据SHA不匹配：' + str(path))
                    entry.update(sha256=actual, sha256_verified=True)
            elif verify_sha:
                raise FileNotFoundError(path)
            else:
                entry['note'] = 'dry_run未读取/核验大输入，真实启动会核验'
            result[split][kind] = entry
    return result


def capacity_limit(baseline_run, requested):
    """旧容量仅作为硬上限，原引擎仍逐方案实测并继续降档。"""
    baseline = Path(baseline_run)
    main_root = baseline.parents[2]
    results_root = main_root.parent
    candidates = []
    evidence = []
    for path, fields in [
        (results_root / 'capacity_selection.json', ('selected_max_parallel', 'current_max_parallel')),
        (main_root / 'resource_runtime_state.json', ('max_parallel',)),
    ]:
        if not path.is_file():
            continue
        data = load_json(path)
        for field in fields:
            value = data.get(field)
            if isinstance(value, int) and (not isinstance(value, bool)) and (value >= 1):
                candidates.append(value)
                evidence.append({'path': str(path), 'field': field, 'value': value})
    if not candidates:
        candidates = [2]
        evidence.append({'reason': '旧容量缺失，保守默认仅作上限', 'value': 2})
    cap = min(int(requested), 4, *candidates)
    if cap < 1:
        raise ValueError('max_parallel必须至少为1')
    return (cap, evidence)


def load_baseline(baseline_run):
    baseline = Path(baseline_run)
    requested = load_json(baseline / 'queue_requested_config.json')
    resolved = load_json(baseline / 'resolved_config.json')
    for config in [requested, resolved]:
        if config.get('model_type') != 'litemamba' or config.get('seed') != 42 or config.get('feature_scheme') != 'full':
            raise ValueError('baseline必须为full/litemamba/seed42')
        if config.get('monitor') != 'val_loss':
            raise ValueError('baseline monitor必须保留val_loss，不能改成val_macro_f1')
    for key in SCALAR_KEYS + BOOL_KEYS:
        if key not in requested:
            raise ValueError('baseline请求配置缺少：' + key)
        if key in resolved and requested[key] != resolved[key]:
            raise ValueError('baseline requested/resolved配置冲突：' + key)
    # 原queue没有这些CLI参数，先确认原baseline确实使用引擎默认。
    for label, configuration in [
        ('resolved_config', resolved),
        ('model_kwargs', resolved.get('model_kwargs', {})),
    ]:
        for key, expected in FIXED_ENGINE_DEFAULTS.items():
            if key in configuration and configuration[key] != expected:
                raise ValueError('baseline与原引擎不可覆盖默认不一致：' + label + '/' + key)
    if requested.get('external_manifest') or resolved.get('external_manifest'):
        raise ValueError('诊断基线不能带external_manifest')
    if requested['no_embeddings'] or requested['no_epoch_logits']:
        raise ValueError('此包保留完整embedding/epoch logits，不能使用关闭导出的baseline')
    if not resolved.get('cache_dir'):
        raise ValueError('baseline resolved_config缺少cache_dir')
    for split in INPUT_NAMES:
        for kind in INPUT_NAMES[split]:
            a = requested.get('input_fingerprints', {}).get(split, {}).get(kind, {}).get('sha256')
            b = resolved.get('input_fingerprints', {}).get(split, {}).get(kind, {}).get('sha256')
            if not a or a != b:
                raise ValueError('baseline requested/resolved输入SHA冲突：' + split + '/' + kind)
    return (requested, resolved)


def build_command(args, requested, resolved, cap):
    command = [
        sys.executable,
        '-u',
        str(Path(args.engine_dir) / 'run_all_4_models.py'),
        '--data_dir',
        str(args.data_dir),
        '--output_dir',
        str(args.output_dir),
        '--cache_dir',
        str(resolved['cache_dir']),
        '--models',
        'litemamba',
        '--seeds',
        '42',
        '--feature_schemes',
        *SCHEMES,
        '--phase',
        'train',
        '--staged_exports',
        '--no_isolated_benchmark',
        '--profile_steps',
        '8',
        '--max_parallel',
        str(cap),
        '--device',
        args.device,
        '--workers',
        str(args.workers if args.workers is not None else resolved.get('workers', 0)),
        '--cpu_threads',
        str(args.cpu_threads if args.cpu_threads is not None else resolved.get('cpu_threads', 4)),
    ]
    for key in SCALAR_KEYS:
        command += ['--' + key, str(requested[key])]
    for key in BOOL_KEYS:
        if requested[key]:
            command.append('--' + key)
    for key, value in RESOURCE_BUDGET.items():
        command += ['--' + key, str(value)]
    if args.resume:
        command.append('--resume_queue')
    return command


def build_plan(args):
    args.engine_dir = Path(args.engine_dir).expanduser().resolve()
    initial_data = Path(args.data_dir or DEFAULT_DATA).expanduser().resolve()
    args.baseline_run = Path(args.baseline_run).expanduser().resolve() if args.baseline_run else initial_data / 'sixclass_auto_v1_0_4_results/main_20/full/litemamba/seed_42'
    requested, resolved = load_baseline(args.baseline_run)
    args.data_dir = Path(args.data_dir or requested.get('data_dir') or DEFAULT_DATA).expanduser().resolve()
    args.output_dir = Path(args.output_dir).expanduser().resolve() if args.output_dir else args.data_dir / 'sixclass_auto_v1_0_4_results/litemamba_diagnostic_seed42_v1_0_0'
    if args.output_dir == args.data_dir or args.output_dir in args.data_dir.parents or args.output_dir == args.baseline_run or (args.output_dir in args.baseline_run.parents):
        raise ValueError('诊断输出不得覆盖数据或baseline目录')
    metrics = args.baseline_run / 'val/analysis/metrics.json'
    if not metrics.is_file():
        raise FileNotFoundError('baseline验证指标文件缺失：' + str(metrics))
    sources = validate_sources(args.engine_dir, requested)
    fingerprints = validate_data(args.data_dir, requested['input_fingerprints'], not args.dry_run)
    cap, evidence = capacity_limit(args.baseline_run, args.max_parallel)
    command = build_command(args, requested, resolved, cap)
    baseline_files = {name: sha256(args.baseline_run / name) for name in ['queue_requested_config.json', 'resolved_config.json', 'val/analysis/metrics.json']}
    scientific = {key: requested[key] for key in SCALAR_KEYS + BOOL_KEYS}
    scientific.update(model_type='litemamba', seed=42)
    return dict(
        version=VERSION,
        purpose='single-seed validation diagnostic; not final five-seed ablation',
        validation_only=True,
        baseline_run=str(args.baseline_run),
        engine_dir=str(args.engine_dir),
        data_dir=str(args.data_dir),
        output_dir=str(args.output_dir),
        cache_dir=str(resolved['cache_dir']),
        fixed_model='litemamba',
        fixed_seed=42,
        feature_schemes=list(SCHEMES),
        tasks=[{'run_id': s + '/litemamba/seed_42', 'model': 'litemamba', 'seed': 42, 'feature_scheme': s} for s in SCHEMES],
        scientific_configuration=scientific,
        baseline_file_sha256=baseline_files,
        baseline_validation_metrics_file=str(metrics),
        baseline_validation_metrics_sha256=baseline_files['val/analysis/metrics.json'],
        source_sha256=sources,
        expected_input_fingerprints=requested['input_fingerprints'],
        data_verification=fingerprints,
        phase='train',
        staged_exports=True,
        no_isolated_benchmark=True,
        external_manifest=None,
        concurrency=dict(
            requested=args.max_parallel,
            effective_hard_cap=cap,
            evidence=evidence,
            note='仅上限，不保证各方案同时运行；原引擎逐方案profile后按GPU/RAM/CPU预算准入并可降档',
        ),
        disk_policy=dict(
            startup_free_gib=args.startup_free_gib,
            disk_reserve_gib=args.disk_reserve_gib,
            low_space_action='SIGTERM queue process group, preserve checkpoints, no automatic retry',
        ),
        argv=command,
        command=shlex.join(command),
        dry_run=args.dry_run,
        created_at=now(),
    )


def plan_identity(plan):
    return {key: plan[key] for key in ['baseline_run', 'engine_dir', 'data_dir', 'output_dir', 'cache_dir', 'tasks', 'scientific_configuration', 'baseline_file_sha256', 'source_sha256', 'expected_input_fingerprints']}


def disk_paths(plan):
    paths = []
    devices = set()
    for value in [plan['output_dir'], plan['cache_dir']]:
        path = Path(value)
        while not path.exists():
            path = path.parent
        device = path.stat().st_dev
        if device not in devices:
            paths.append(path)
            devices.add(device)
    return paths


def disk_free_gib(paths):
    return {str(p): shutil.disk_usage(p).free / GIB for p in paths}


def process_start_ticks(pid):
    try:
        return Path('/proc/' + str(pid) + '/stat').read_text().rsplit(')', 1)[1].split()[19]
    except (OSError, IndexError):
        return None


def terminate_recorded_groups(output):
    """只向此诊断commands记录且PID/startticks仍匹配的独立trainer组发TERM。"""
    commands = Path(output) / 'commands.jsonl'
    if not commands.is_file():
        return
    seen = set()
    with commands.open(encoding='utf-8') as stream:
        for line in stream:
            try:
                item = json.loads(line)
            except ValueError:
                continue
            pid = item.get('pid')
            ticks = item.get('process_start_ticks')
            if item.get('event') != 'queue_child_started':
                continue
            if not isinstance(pid, int) or isinstance(pid, bool) or pid <= 1 or (ticks is None):
                continue
            identity = (pid, str(ticks))
            if identity in seen or str(ticks) != str(process_start_ticks(pid)):
                continue
            seen.add(identity)
            try:
                if os.getpgid(pid) == pid:
                    os.killpg(pid, signal.SIGTERM)
            except ProcessLookupError:
                continue


def stop_queue(process, output):
    """先让原queue正常清理；即使queue已退出，也检查其独立session残留。"""
    if process.poll() is None:
        try:
            os.killpg(process.pid, signal.SIGTERM)
        except ProcessLookupError:
            pass
        try:
            process.wait(timeout=45)
        except subprocess.TimeoutExpired:
            pass
    # queue已退出也可能遗留独立session；只清理本目录且PID/ticks仍匹配的记录。
    terminate_recorded_groups(output)
    if process.poll() is None:
        try:
            process.wait(timeout=10)
        except subprocess.TimeoutExpired:
            try:
                os.killpg(process.pid, signal.SIGKILL)
            except ProcessLookupError:
                pass
            process.wait(timeout=10)


def verify_validation_only(output):
    states = []
    for scheme in SCHEMES:
        run = Path(output) / scheme / 'litemamba/seed_42'
        if (run / 'test_started.json').exists() or (run / 'test_evaluated.json').exists() or ((run / 'test').is_dir() and any((run / 'test').iterdir())):
            raise RuntimeError('检测到test开始/输出，拒绝诊断汇总：' + str(run))
        result = load_json(run / 'results.json')
        if result.get('test_evaluated') is not False or result.get('status') != 'train_complete' or 'test' in result.get('exports_by_split', {}):
            raise RuntimeError('任务不是仅训练验证完成：' + str(run))
        if not (run / 'val/analysis/metrics.json').is_file():
            raise RuntimeError('验证指标未完整导出：' + str(run))
        states.append({'run_id': scheme + '/litemamba/seed_42', 'status': 'train_complete', 'test_evaluated': False})
    return states


def parse_args(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--engine_dir', default=DEFAULT_ENGINE)
    parser.add_argument('--data_dir')
    parser.add_argument('--baseline_run')
    parser.add_argument('--output_dir')
    parser.add_argument('--max_parallel', type=int, default=4)
    parser.add_argument('--device', default='auto')
    parser.add_argument('--resume', action='store_true')
    parser.add_argument('--dry_run', action='store_true')
    parser.add_argument('--disk_reserve_gib', type=float, default=2.0)
    parser.add_argument('--startup_free_gib', type=float, default=30.0)
    parser.add_argument('--workers', type=int)
    parser.add_argument('--cpu_threads', type=int)
    args = parser.parse_args(argv)
    if args.max_parallel < 1 or args.disk_reserve_gib < 0 or args.startup_free_gib < args.disk_reserve_gib:
        parser.error('并发至少1，空间阈值非负且启动阈值不得小于运行保留值')
    if args.workers is not None and args.workers < 0 or (args.cpu_threads is not None and args.cpu_threads < 1):
        parser.error('workers>=0, cpu_threads>=1')
    return args


def main(argv=None):
    args = parse_args(argv)
    plan = build_plan(args)
    output = args.output_dir
    output.mkdir(parents=True, exist_ok=True)
    with (output / 'diagnostic_wrapper.lock').open('a+') as lock:
        try:
            fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError:
            raise RuntimeError('此诊断目录已有wrapper运行，请勿重复提交')
        lock.seek(0)
        lock.truncate()
        lock.write(json.dumps({'pid': os.getpid(), 'at': now()}))
        lock.flush()
        plan_path = output / 'diagnostic_plan.json'
        if plan_path.is_file() and plan_identity(load_json(plan_path)) != plan_identity(plan):
            raise RuntimeError('已有诊断计划的配置/数据/源码/基线不同，拒绝覆盖或resume')
        if (output / 'queue_manifest.json').exists() and (not args.resume) and (not args.dry_run):
            raise RuntimeError('已有队列，请使用--resume继续未完成任务')
        stamp = datetime.now(timezone.utc).strftime('%Y%m%d_%H%M%S_%f') + '_' + str(os.getpid())
        logdir = output / 'logs'
        logdir.mkdir(exist_ok=True)
        logpath = logdir / ('diagnostic_' + stamp + '.log')
        plan['invocation_log'] = str(logpath)
        save_json(plan_path, plan)
        with logpath.open('x', encoding='utf-8', buffering=1) as log:
            latest_temporary = logdir / ('diagnostic_latest_' + str(os.getpid()) + '.tmp')
            latest_temporary.symlink_to(logpath.name)
            latest_temporary.replace(logdir / 'diagnostic_latest.log')
            log.write(json.dumps({'at': now(), 'wrapper_pid': os.getpid(), 'argv': plan['argv'], 'dry_run': args.dry_run}, ensure_ascii=False) + '\n')
            print('[PLAN] ' + str(plan_path), flush=True)
            print('[LOG] ' + str(logpath), flush=True)
            print('[FOLLOW] tail -F ' + shlex.quote(str(logdir / 'diagnostic_latest.log')), flush=True)
            print('[COMMAND] ' + plan['command'], flush=True)
            if args.dry_run:
                print('[DRY RUN] 仅元数据和8个源码SHA已校验；六个大输入SHA未核验，没有启动训练。', flush=True)
                return 0
            summary_script = Path(__file__).with_name('summarize_diagnostic.py')
            if not summary_script.is_file():
                raise FileNotFoundError('缺少同包汇总脚本：' + str(summary_script))
            paths = disk_paths(plan)
            free = disk_free_gib(paths)
            if min(free.values()) < args.startup_free_gib:
                raise RuntimeError('启动空闲空间不足：' + json.dumps(free) + ' GiB；需要至少' + str(args.startup_free_gib))
            process = None
            status = {
                'status': 'running',
                'started_at': now(),
                'log': str(logpath),
                'validation_only': True,
            }
            try:
                process = subprocess.Popen(
                    plan['argv'],
                    cwd=str(args.engine_dir),
                    stdout=log,
                    stderr=subprocess.STDOUT,
                    start_new_session=True,
                )
                status['queue_pid'] = process.pid
                save_json(output / 'diagnostic_status.json', status)
                while process.poll() is None:
                    free = disk_free_gib(paths)
                    if min(free.values()) < args.disk_reserve_gib:
                        status.update(
                            status='stopped_low_disk',
                            free_gib=free,
                            note='已TERM本队列，未自动重试；清理空间后同命令加--resume',
                        )
                        stop_queue(process, output)
                        save_json(output / 'diagnostic_status.json', status)
                        raise RuntimeError('运行空闲空间低于保留阈值，已安全停止；可--resume')
                    time.sleep(2)
                if process.returncode:
                    raise RuntimeError('原队列失败code=' + str(process.returncode) + '，保留现场可--resume；日志：' + str(logpath))
                states = verify_validation_only(output)
                for name, digest in plan['baseline_file_sha256'].items():
                    if sha256(args.baseline_run / name) != digest:
                        raise RuntimeError('运行期间baseline元数据/验证指标改变：' + name)
                status.update(status='training_validation_complete', runs=states, finished_at=now())
                save_json(output / 'diagnostic_status.json', status)
                summary = [
                    sys.executable,
                    str(summary_script),
                    '--baseline_run',
                    str(args.baseline_run),
                    '--diagnostic_dir',
                    str(output),
                ]
                completed = subprocess.run(summary, stdout=log, stderr=subprocess.STDOUT)
                if completed.returncode:
                    raise RuntimeError('诊断汇总失败；训练无需重跑，查看日志：' + str(logpath))
                status.update(status='complete', summary_finished_at=now())
                save_json(output / 'diagnostic_status.json', status)
                print(
                    '[COMPLETE] ' + str(output / 'diagnostic_results_for_analysis_v1_0_0.tar.gz'),
                    flush=True,
                )
                return 0
            except BaseException as error:
                if process is not None:
                    stop_queue(process, output)
                if status.get('status') != 'stopped_low_disk':
                    status.update(
                        status='interrupted' if isinstance(error, KeyboardInterrupt) else 'failed',
                        error=str(error),
                        finished_at=now(),
                    )
                save_json(output / 'diagnostic_status.json', status)
                raise


def handle_sigterm(signum, frame):
    raise KeyboardInterrupt('收到SIGTERM')
if __name__ == '__main__':
    signal.signal(signal.SIGTERM, handle_sigterm)
    try:
        raise SystemExit(main())
    except KeyboardInterrupt as error:
        print('已中断；保持配置不变可加--resume。' + str(error), file=sys.stderr)
        raise SystemExit(130)
    except Exception as error:
        print('ERROR: ' + type(error).__name__ + ': ' + str(error), file=sys.stderr)
        raise SystemExit(1)
