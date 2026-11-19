#!/usr/bin/env python3
"""固定9次 GatedConv 单输入训练；调用随包冻结引擎，不覆盖已有实验。"""
from __future__ import annotations

import argparse
import contextlib
from datetime import datetime, timezone
import fcntl
import hashlib
import json
import math
import os
from pathlib import Path
import signal
import shutil
import subprocess
import sys
import time

VERSION = '1.0.0'
HERE = Path(__file__).resolve().parent
ENGINE = HERE / 'engine'
REFERENCE = HERE / 'reference' / 'experiment_reference.json'
DEFAULT_DATA = '/path/to/revision-inputs/CPLX-XNA-replenishment/CPLX-XNA-replenishment/fast5_single_guppy/sixclass_overlap_read_split_721'
RESULTS_NAME = 'sixclass_auto_v1_0_4_results'
OUTPUT_NAME = 'gatedconv_singleinput9_v1_0_0'
SCHEMES = ('raw_only', 'context_only', 'responses_only')
SEEDS = (42, 2024, 2025)
INPUT_FILES = {'train': ('train.csv', 'train_meta.tsv'),
               'val': ('validation.csv', 'validation_meta.tsv'),
               'test': ('test.csv', 'test_meta.tsv')}
GIB = 1024 ** 3


def now():
    return datetime.now(timezone.utc).isoformat()


def read_json(path):
    return json.loads(Path(path).read_text(encoding='utf-8'))


def write_json(path, data):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + '.tmp.' + str(os.getpid()))
    temporary.write_text(json.dumps(data, ensure_ascii=False, indent=2, allow_nan=False) + '\n', encoding='utf-8')
    temporary.replace(path)


def sha256(path):
    digest = hashlib.sha256()
    with Path(path).open('rb') as stream:
        for block in iter(lambda: stream.read(8 * 1024 ** 2), b''):
            digest.update(block)
    return digest.hexdigest()


def parse_args(argv=None):
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    p.add_argument('--data_dir', default=DEFAULT_DATA)
    p.add_argument('--output_dir')
    p.add_argument('--cache_dir')
    p.add_argument('--baseline_dir', help='已有三种子结构确认结果根目录；只读，用于完整输入对照汇总')
    p.add_argument('--baseline_test_dir', help='已有 GatedConv 内部测试结果根目录；只读')
    p.add_argument('--resume', action='store_true', help='配置和输入不变时跳过完整任务，恢复中断任务')
    p.add_argument('--device', default='auto')
    p.add_argument('--gpu_ids', type=int, nargs='+')
    p.add_argument('--max_parallel', type=int, choices=(1, 2, 3, 4), default=4)
    p.add_argument('--cpu_threads', type=int, default=4)
    p.add_argument('--cpu_budget', type=int)
    p.add_argument('--workers', type=int, default=0)
    p.add_argument('--gpu_reserve_gib', type=float, default=2)
    p.add_argument('--ram_reserve_gib', type=float, default=3)
    p.add_argument('--reserve_fraction', type=float, default=.10)
    p.add_argument('--memory_safety_factor', type=float, default=1.35)
    p.add_argument('--train_ram_extra_gib', type=float, default=1)
    p.add_argument('--disk_reserve_gib', type=float, default=8)
    p.add_argument('--disk_safety_factor', type=float, default=1.20)
    p.add_argument('--poll_seconds', type=float, default=3)
    p.add_argument('--dry_run', action='store_true', help='只打印固定9任务和命令，不读取大数据、不启动进程')
    p.add_argument('--check_only', action='store_true', help='核验环境、输入SHA和空间，不开始训练')
    p.add_argument('--summary_only', action='store_true', help='只重新汇总已有结果及核心上传包，不训练/推理')
    args = p.parse_args(argv)
    if args.cpu_threads < 1 or args.workers < 0:
        p.error('cpu_threads须>=1，workers须>=0')
    if args.cpu_budget is not None and args.cpu_budget < args.cpu_threads + args.workers:
        p.error('cpu_budget不足以启动一个作业')
    numeric = (args.gpu_reserve_gib, args.ram_reserve_gib, args.train_ram_extra_gib,
               args.disk_reserve_gib, args.reserve_fraction, args.memory_safety_factor,
               args.disk_safety_factor, args.poll_seconds)
    if any(not math.isfinite(x) for x in numeric):
        p.error('资源数值必须有限')
    if min(args.gpu_reserve_gib, args.ram_reserve_gib, args.train_ram_extra_gib) < 0:
        p.error('内存保留值不能为负')
    if args.disk_reserve_gib <= 0 or args.disk_safety_factor < 1 or args.memory_safety_factor < 1:
        p.error('磁盘余量须>0，安全系数须>=1')
    if not 0 <= args.reserve_fraction < 1 or args.poll_seconds <= 0:
        p.error('reserve_fraction须在[0,1)，poll_seconds须>0')
    args.data_dir = Path(args.data_dir).expanduser().resolve()
    results = args.data_dir / RESULTS_NAME
    args.output_dir = Path(args.output_dir or results / OUTPUT_NAME).expanduser().resolve()
    args.cache_dir = Path(args.cache_dir or results / 'feature_cache').expanduser().resolve()
    args.baseline_dir = Path(args.baseline_dir or results / 'litemamba_structure_confirm3_v1_1_0').expanduser().resolve()
    args.baseline_test_dir = Path(args.baseline_test_dir or results / 'gatedconv_internal_test_v1_0_0').expanduser().resolve()
    if args.output_dir == args.data_dir or args.output_dir in args.data_dir.parents or args.output_dir == results:
        p.error('output_dir须为独立实验子目录，不能使用数据目录/结果总根目录')
    if (args.output_dir == args.cache_dir or args.output_dir in args.cache_dir.parents
            or args.cache_dir in args.output_dir.parents):
        p.error('输出不能等于或包含共享缓存；缓存应放在输出之外')
    for baseline in (args.baseline_dir, args.baseline_test_dir):
        if args.output_dir == baseline or args.output_dir in baseline.parents or baseline in args.output_dir.parents:
            p.error('输出不能与已有完整输入对照/内部测试目录重叠')
    return args


def source_contract(reference):
    actual = {}
    for name, expected in reference['frozen_sources'].items():
        actual[name] = sha256(ENGINE / name)
        if actual[name] != expected:
            raise ValueError('最终 GatedConv 模型/预处理源码与冻结参考不一致：' + name)
    science = reference['scientific_configuration']
    if (reference['seeds'] != list(SEEDS) or reference['schemes'] != list(SCHEMES)
            or science['litemamba_variant'] != 'legacy_gatedconv'
            or science['readout'] != 'target_attention'
            or reference['model_defaults']['target_index'] != 3):
        raise ValueError('固定9任务/模型身份不符合本包约定')
    return actual


def build_command(args, reference):
    s = reference['scientific_configuration']
    command = [sys.executable, '-u', str(ENGINE / 'run_all_4_models.py'),
               '--data_dir', str(args.data_dir), '--output_dir', str(args.output_dir),
               '--cache_dir', str(args.cache_dir), '--models', 'litemamba',
               '--seeds', *map(str, SEEDS), '--feature_schemes', *SCHEMES,
               '--phase', 'all', '--staged_exports', '--no_isolated_benchmark',
               '--profile_steps', '8', '--oom_retries', '2', '--device', args.device]
    for key in ('epochs', 'patience', 'lr', 'batch_size', 'raw_signal_len', 'dwell_divisor',
                'quality_divisor', 'accum_steps', 'weight_decay', 'label_smoothing',
                'clip_grad', 'monitor', 'model_dim', 'num_blocks', 'state_dim',
                'litemamba_variant', 'readout', 'bootstrap', 'embedding_dtype'):
        command += ['--' + key, str(s[key])]
    for key in ('no_plots', 'no_embeddings', 'no_epoch_logits', 'save_epoch_predictions'):
        if s[key]:
            command.append('--' + key)
    for key in ('max_parallel', 'cpu_threads', 'workers', 'gpu_reserve_gib',
                'ram_reserve_gib', 'reserve_fraction', 'memory_safety_factor',
                'train_ram_extra_gib', 'poll_seconds'):
        command += ['--' + key, str(getattr(args, key))]
    if args.cpu_budget is not None:
        command += ['--cpu_budget', str(args.cpu_budget)]
    if args.gpu_ids:
        command += ['--gpu_ids', *map(str, args.gpu_ids)]
    if args.resume:
        command.append('--resume_queue')
    return command


def validate_runtime(args):
    script = ('import json,torch,numpy,pandas,scipy,sklearn,matplotlib; '
              'print(json.dumps({"torch":torch.__version__,"numpy":numpy.__version__,'
              '"cuda_available":torch.cuda.is_available(),"cuda_visible_devices":'
              '__import__("os").environ.get("CUDA_VISIBLE_DEVICES")}))')
    process = subprocess.run([sys.executable, '-c', script], capture_output=True, text=True)
    if process.returncode:
        raise RuntimeError('当前Python缺少训练依赖；请使用原 pytorch2.0.1 环境。\n' + process.stderr[-3000:])
    info = json.loads(process.stdout.strip().splitlines()[-1])
    if args.device != 'cpu' and not info['cuda_available']:
        raise RuntimeError('当前环境未发现可用CUDA，未启动CPU长训练；请激活原GPU环境。CPU验证须显式--device cpu。')
    version = tuple(int(x) for x in info['torch'].split('+')[0].split('.')[:2])
    if version < (2, 0):
        raise RuntimeError('需要PyTorch>=2.0.1')
    if version < (2, 4) and int(info['numpy'].split('.')[0]) >= 2:
        raise RuntimeError('旧PyTorch环境请保留numpy<2，避免张量桥接不兼容')
    return info


def validate_inputs(args, reference):
    inputs = {}
    for split, names in INPUT_FILES.items():
        inputs[split] = {}
        for kind, name in zip(('feature', 'metadata'), names):
            expected = reference['input_fingerprints'][split][kind]
            path = args.data_dir / name
            before = path.stat()
            if before.st_size != expected.get('size', expected.get('size_bytes')):
                raise ValueError('输入大小与已完成GatedConv不一致：' + str(path))
            print('[VERIFY SHA] ' + split + '/' + kind + ' ' + str(path), flush=True)
            actual = sha256(path)
            after = path.stat()
            if (before.st_size, before.st_mtime_ns) != (after.st_size, after.st_mtime_ns):
                raise RuntimeError('核验期间输入被修改：' + str(path))
            if actual != expected['sha256']:
                raise ValueError('输入SHA与已有完整输入模型不一致：' + str(path))
            inputs[split][kind] = {'path': str(path), 'sha256': actual,
                                   'size': before.st_size, 'mtime_ns': before.st_mtime_ns}
    return inputs


def existing_baseline_evidence(reference):
    evidence = []
    for record in reference['baseline_records']:
        directory = Path(record['run_dir'])
        checkpoint = directory / 'best_checkpoint.pt'
        entry = {'seed': record['seed'], 'run_dir': str(directory), 'expected_checkpoint_sha256': record['checkpoint_sha256'],
                 'server_checkpoint_available': checkpoint.is_file(), 'use': 'read_only_full_comparison_never_initialization'}
        if checkpoint.is_file():
            actual = sha256(checkpoint)
            if actual != record['checkpoint_sha256']:
                raise ValueError('已有完整输入checkpoint与冻结记录不同：' + str(checkpoint))
            entry['checkpoint_sha256'] = actual
        evidence.append(entry)
    return evidence


def _read_committed_epoch(run):
    path = run / 'epoch_metrics.jsonl'
    if not (run / 'last_checkpoint.pt').is_file() or not path.is_file():
        return 0
    try:
        rows = [json.loads(x) for x in path.read_text(encoding='utf-8').splitlines() if x.strip()]
        # JSON may contain one uncommitted epoch if interrupted just before last;
        # allow for that boundary conservatively in the remaining-space budget.
        return max(0, int(rows[-1]['epoch']) - 1) if rows else 0
    except (OSError, ValueError, KeyError, TypeError):
        return 0


def disk_estimate(args, reference, verified_inputs=None):
    """Budget future bytes, never delete existing arrays/history/checkpoints."""
    rows = reference['input_rows']
    s = reference['scientific_configuration']
    parts = []
    for scheme in SCHEMES:
        for seed in SEEDS:
            run = args.output_dir / scheme / 'litemamba' / ('seed_' + str(seed))
            result = read_json(run / 'results.json') if (run / 'results.json').is_file() else {}
            weights_done = result.get('status') in ('weights_complete', 'train_complete', 'complete')
            remaining_epochs = 0 if weights_done else max(0, s['epochs'] - _read_committed_epoch(run))
            epoch_bytes = 0 if s['no_epoch_logits'] else remaining_epochs * rows['val'] * (6 * 4 + 8 + 8)
            # Includes embeddings, attention, labels/logits/raw+cal probs,
            # compressed metadata-rich CSV, plots, analyses and write temporaries.
            export_bytes = sum(n * 1800 for split, n in rows.items()
                               if not (run / split / 'export_complete.json').is_file())
            checkpoint_bytes = 0 if weights_done else (remaining_epochs // 10 + 4) * 6 * 1024 ** 2
            parts.append({'scheme': scheme, 'seed': seed, 'remaining_epochs_upper_bound': remaining_epochs,
                          'epoch_arrays_bytes_uncompressed_bound': epoch_bytes,
                          'remaining_export_budget_bytes': export_bytes,
                          'checkpoint_budget_bytes': checkpoint_bytes})
    cache_missing = list(rows)
    if verified_inputs and args.cache_dir.is_dir():
        for path in args.cache_dir.glob('*/audit.json'):
            try:
                audit = read_json(path)
                prep = audit.get('preprocessing', {})
                if (prep.get('raw_signal_len') != s['raw_signal_len'] or prep.get('mask_index') != 3
                        or prep.get('dwell_divisor') != s['dwell_divisor']
                        or prep.get('quality_divisor') != s['quality_divisor']):
                    continue
                for split in list(cache_missing):
                    fp = audit.get('input_fingerprints', {})
                    if all(all(fp.get(k, {}).get(field) == verified_inputs[split][k][field]
                               for field in ('path', 'sha256', 'size', 'mtime_ns'))
                           for k in ('feature', 'metadata')):
                        required = ['merged.npy', 'raw.npy', 'labels.npy', 'metadata_offsets.npy', 'manifest.jsonl']
                        if all((path.parent / name).is_file() for name in required):
                            cache_missing.remove(split)
            except (OSError, ValueError, TypeError):
                continue
    cache_bytes = sum(rows[split] * 3600 for split in cache_missing)
    output_bytes = math.ceil(sum(p['epoch_arrays_bytes_uncompressed_bound'] + p['remaining_export_budget_bytes']
                                 + p['checkpoint_budget_bytes'] for p in parts) * args.disk_safety_factor)
    return {'scope': 'remaining training/internal exports only; excludes future Figure4 predictions',
            'basis': 'uncompressed epoch array bound + conservative 1800 bytes/window export budget + checkpoints; safety factor applied',
            'runs': parts, 'shared_cache_missing_splits': cache_missing,
            'cache_budget_bytes': math.ceil(cache_bytes * args.disk_safety_factor),
            'output_budget_bytes': output_bytes, 'reserve_bytes': math.ceil(args.disk_reserve_gib * GIB)}


def _existing_ancestor(path):
    path = Path(path)
    while not path.exists():
        path = path.parent
    return path


def disk_requirements(args, estimate):
    requirements = {}
    for target, budget in [(args.output_dir, estimate['output_budget_bytes']),
                           (args.cache_dir, estimate['cache_budget_bytes'])]:
        probe = _existing_ancestor(target)
        device = probe.stat().st_dev
        entry = requirements.setdefault(device, {'probe': str(probe), 'targets': [], 'additional_budget_bytes': 0})
        entry['targets'].append(str(target))
        entry['additional_budget_bytes'] += budget
    result = []
    for entry in requirements.values():
        entry['free_bytes'] = shutil.disk_usage(entry['probe']).free
        entry['reserve_bytes'] = estimate['reserve_bytes']
        entry['required_free_bytes'] = entry['additional_budget_bytes'] + entry['reserve_bytes']
        result.append(entry)
    return result


def check_disk(args, estimate):
    requirements = disk_requirements(args, estimate)
    for entry in requirements:
        print('[DISK] ' + entry['probe'] + ' 可用 {:.2f} GiB；后续预算 {:.2f} GiB + 保留 {:.2f} GiB'.format(
            entry['free_bytes'] / GIB, entry['additional_budget_bytes'] / GIB, entry['reserve_bytes'] / GIB), flush=True)
        if entry['free_bytes'] < entry['required_free_bytes']:
            raise OSError('磁盘余量不足以覆盖本次剩余任务预算；尚未启动训练。清理空间后使用同一命令重试。')
    return requirements


@contextlib.contextmanager
def output_lock(output):
    output.mkdir(parents=True, exist_ok=True)
    with (output / '.singleinput.lock').open('a+') as stream:
        try:
            fcntl.flock(stream, fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError as exc:
            raise RuntimeError('同一输出目录已有启动器运行；未修改其日志/PID') from exc
        stream.seek(0)
        stream.truncate()
        stream.write(json.dumps({'pid': os.getpid(), 'started_at': now()}))
        stream.flush()
        yield


def package_source_hashes():
    files = sorted(HERE.glob('*.py')) + sorted(ENGINE.glob('*.py'))
    return {str(path.relative_to(HERE)): sha256(path) for path in files}


def immutable_plan(args, reference):
    return {'version': VERSION, 'seeds': list(SEEDS), 'schemes': list(SCHEMES),
            'scientific_configuration': reference['scientific_configuration'],
            'expected_input_sha256': {split: {kind: f['sha256'] for kind, f in pair.items()}
                                      for split, pair in reference['input_fingerprints'].items()},
            'source_sha256': package_source_hashes(),
            'data_dir': str(args.data_dir), 'cache_dir': str(args.cache_dir),
            'initialization': 'fresh seeded initialization; no full checkpoint warmstart',
            'evaluation_policy': 'all9 weights complete -> serial train/val export -> freeze9 hashes -> serial test',
            'external_evaluation': 'not_started_by_this_package; frozen checkpoints and configs retained for Figure4'}


def run_supervised(command, args, log_path):
    environment = dict(os.environ, PYTHONUNBUFFERED='1', OMP_NUM_THREADS=str(args.cpu_threads),
                       OPENBLAS_NUM_THREADS=str(args.cpu_threads), MKL_NUM_THREADS=str(args.cpu_threads),
                       NUMEXPR_NUM_THREADS=str(args.cpu_threads))
    print('[QUEUE] ' + str(log_path), flush=True)
    with log_path.open('a', encoding='utf-8', buffering=1) as log:
        process = subprocess.Popen(command, cwd=str(ENGINE), stdout=log, stderr=subprocess.STDOUT,
                                   env=environment, start_new_session=True)
        write_json(args.output_dir / 'active_queue.json', {'pid': process.pid, 'log': str(log_path), 'started_at': now()})
        try:
            while process.poll() is None:
                for path in (args.output_dir, args.cache_dir):
                    free = shutil.disk_usage(_existing_ancestor(path)).free
                    if free < args.disk_reserve_gib * GIB:
                        raise OSError('剩余磁盘触及保留线，停止本队列；清理后--resume恢复：' + str(path))
                time.sleep(min(args.poll_seconds, 5))
            if process.returncode:
                raise RuntimeError('训练/评估队列退出码={}；查看日志：{}'.format(process.returncode, log_path))
        except BaseException:
            if process.poll() is None:
                process.terminate()  # queue SIGTERM handler terminates owned child groups
                try:
                    process.wait(timeout=45)
                except subprocess.TimeoutExpired:
                    os.killpg(process.pid, signal.SIGKILL)
                    process.wait()
            raise
        finally:
            write_json(args.output_dir / 'active_queue.json', {'pid': process.pid, 'returncode': process.returncode,
                                                              'log': str(log_path), 'finished_at': now()})


def summarize(args):
    command = [sys.executable, str(HERE / 'summarize_singleinput.py'), '--output_dir', str(args.output_dir),
               '--baseline_dir', str(args.baseline_dir), '--baseline_test_dir', str(args.baseline_test_dir), '--package']
    subprocess.run(command, cwd=str(HERE), check=True)


def main(argv=None):
    args = parse_args(argv)
    reference = read_json(REFERENCE)
    source_contract(reference)
    for item in reference['baseline_records']:
        baseline = Path(item['run_dir']).expanduser().resolve()
        if args.output_dir == baseline or args.output_dir in baseline.parents or baseline in args.output_dir.parents:
            raise ValueError('输出不能与冻结完整输入训练目录重叠：' + str(baseline))
    command = build_command(args, reference)
    if args.dry_run:
        print(json.dumps({'version': VERSION, 'new_training_runs': 9,
                          'experiments': [{'scheme': scheme, 'seed': seed} for scheme in SCHEMES for seed in SEEDS],
                          'command': command, 'output_dir': str(args.output_dir),
                          'baseline_comparison_only': True, 'disk_estimate_unverified': disk_estimate(args, reference),
                          'warning': 'No runtime/data SHA/cache validity check or process launch in dry_run'},
                         ensure_ascii=False, indent=2))
        return 0
    if args.summary_only:
        with output_lock(args.output_dir):
            summarize(args)
        return 0
    with output_lock(args.output_dir):
        logs = args.output_dir / 'logs'
        logs.mkdir(exist_ok=True)
        write_json(logs / 'singleinput9.pid.json', {'pid': os.getpid(), 'started_at': now()})
        print('[VERSION] GatedConv single-input9 ' + VERSION + '；固定3输入×3种子，从头训练', flush=True)
        runtime = validate_runtime(args)
        inputs = validate_inputs(args, reference)
        baseline = existing_baseline_evidence(reference)
        estimate = disk_estimate(args, reference, inputs)
        disk = check_disk(args, estimate)
        checks = {'checked_at': now(), 'runtime': runtime, 'inputs': inputs, 'baseline_evidence': baseline,
                  'disk_estimate': estimate, 'disk_checks': disk, 'model_preprocessing_source_check': 'pass'}
        write_json(args.output_dir / 'launch_checks.json', checks)
        if args.check_only:
            print('[CHECK PASS] 环境、输入SHA、冻结模型/预处理和空间预算通过；尚未训练。', flush=True)
            return 0
        plan = immutable_plan(args, reference)
        path = args.output_dir / 'singleinput_plan.json'
        if path.is_file():
            if read_json(path) != plan:
                raise ValueError('原启动计划与当前代码/数据/科学配置不一致；不会改写旧实验，请恢复原包后续跑')
            if not args.resume:
                raise ValueError('此目录已有启动计划，请使用 --resume 续跑；完整任务会自动跳过')
        elif (args.output_dir / 'queue_plan.json').exists():
            raise ValueError('输出已有其他队列计划，不能接管；请使用新的独立目录')
        write_json(path, plan)
        stamp = datetime.now(timezone.utc).strftime('%Y%m%d_%H%M%S_%f')
        queue_log = logs / ('queue_' + stamp + '.log')
        latest = logs / 'queue_latest.log'
        temporary_link = logs / ('.queue_latest.' + str(os.getpid()))
        temporary_link.symlink_to(queue_log.name)
        temporary_link.replace(latest)
        status_path = args.output_dir / 'singleinput_status.json'
        write_json(status_path, {'status': 'running', 'started_at': now(), 'command': command, 'queue_log': str(queue_log)})
        try:
            run_supervised(command, args, queue_log)
            summarize(args)
            write_json(status_path, {'status': 'complete', 'finished_at': now(), 'new_training_runs': 9,
                                     'internal_test_complete': True, 'figure4_external_evaluated': False})
        except BaseException as exc:
            write_json(status_path, {'status': 'interrupted' if isinstance(exc, KeyboardInterrupt) else 'failed',
                                     'finished_at': now(), 'error': type(exc).__name__ + ': ' + str(exc),
                                     'resume_command': 'bash run_singleinput9.sh --resume'})
            raise
        print('[FULL PASS] 9次单输入训练、内部冻结测试与汇总完成；Figure4外测尚未启动。', flush=True)
        print('[RESULT PACKAGE] ' + str(args.output_dir / 'gatedconv_singleinput9_results_for_analysis_v1_0_0.tar.gz'), flush=True)
    return 0


if __name__ == '__main__':
    def interrupt(signum, frame):
        raise KeyboardInterrupt('收到信号 ' + str(signum))
    signal.signal(signal.SIGTERM, interrupt)
    try:
        raise SystemExit(main())
    except KeyboardInterrupt as exc:
        print('[INTERRUPTED] ' + str(exc) + '；保持原包和输入后 --resume。', file=sys.stderr)
        raise SystemExit(130)
    except Exception as exc:
        print('[ERROR] ' + type(exc).__name__ + ': ' + str(exc), file=sys.stderr)
        raise SystemExit(1)
