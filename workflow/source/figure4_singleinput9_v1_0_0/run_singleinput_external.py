#!/usr/bin/env python3
"""使用九个冻结单输入检查点，完成 Figure4 外部评价并复用三个完整输入 GatedConv。"""
from __future__ import annotations

import argparse
from contextlib import contextmanager, ExitStack
from datetime import datetime, timezone
import fcntl
import json
import math
import os
import re
import shutil
from pathlib import Path
import signal
import sys
import time

import run_figure4 as queue
from figure4_common import sha256, verify_run

HERE = Path(__file__).resolve().parent
RESULTS = Path(queue.DEFAULT_RESULTS).parent
DEFAULT_PREVIOUS = RESULTS / 'figure4_gated_comparison_v1_0_0'
DEFAULT_TRAINING = RESULTS / 'gatedconv_singleinput9_v1_0_0'
OUTPUT_NAME = 'figure4_singleinput9_v1_0_0'
RESULT_ARCHIVE = 'figure4_singleinput9_results_core_v1_0_0.tar.gz'
SEEDS = (42, 2024, 2025)
SCHEMES = ('full', 'raw_only', 'context_only', 'responses_only')
PREDICTION_FIELDS = ('batch_size', 'workers', 'cpu_threads', 'embedding_dtype', 'no_embeddings', 'no_attention')


def inherit_prediction_settings(args, explicitly_set):
    """沿用旧 full 预测设置，避免复用后混合不同导出精度。"""
    path = args.previous_output / 'evaluation_plan.json'
    if not path.is_file():
        raise FileNotFoundError('缺少已完成的修正参考 Figure4 计划：' + str(path))
    previous = queue.read_json(path)
    settings = previous.get('prediction_settings', {})
    if settings.get('amp_policy') != 'training_checkpoint_config':
        raise ValueError('旧计划 AMP 策略不符合冻结 checkpoint 配置')
    for name in PREDICTION_FIELDS:
        if name not in settings:
            raise ValueError('旧预测设置缺少：' + name)
        if name in explicitly_set and getattr(args, name) != settings[name]:
            raise ValueError('必须沿用旧预测设置 --' + name + '=' + str(settings[name]))
        setattr(args, name, settings[name])
    if len(previous.get('datasets', [])) != 72:
        raise ValueError('当前确认入口需要上一轮完整的72个冻结数据文件')
    return previous


def resolve_paths(args):
    for name in ('previous_output', 'training_dir', 'output_root'):
        value = getattr(args, name)
        if value is not None:
            setattr(args, name, Path(value).expanduser().resolve())
    if args.output_root is None:
        args.output_root = args.previous_output.parent / OUTPUT_NAME
    protected = [args.previous_output, args.training_dir, HERE]
    for directory in protected:
        if args.output_root == directory or args.output_root in directory.parents or directory in args.output_root.parents:
            raise ValueError('新输出必须与旧结果、训练来源及代码目录独立：' + str(args.output_root))
    # 默认训练 main_20 也不得被新预测输出嵌入或覆盖。
    training_root = args.training_dir.parent / 'main_20'
    if args.output_root == training_root or args.output_root in training_root.parents or training_root in args.output_root.parents:
        raise ValueError('新输出不能覆盖或嵌入原训练 main_20 目录')
    return args


def parse_args(argv=None):
    raw = list(sys.argv[1:] if argv is None else argv)
    explicitly_set = {token.split('=', 1)[0][2:] for token in raw if token.startswith('--')}
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--version', action='version', version='Figure4 single-input external 1.0.0')
    parser.add_argument('--previous_output', type=Path, default=DEFAULT_PREVIOUS)
    parser.add_argument('--training_dir', type=Path, default=DEFAULT_TRAINING)
    parser.add_argument('--output_root', '--output_dir', dest='output_root', type=Path)
    parser.add_argument('--reuse_mode', choices=('symlink', 'hardlink', 'copy'), default='symlink')
    parser.add_argument('--phase', choices=('all', 'predict', 'analyze', 'plot'), default='all')
    parser.add_argument('--resume', action='store_true')
    parser.add_argument('--check_only', action='store_true', help='前台只读预检；不训练、不预测、不生成正式评估计划')
    parser.add_argument('--max_parallel', type=int, choices=range(1, 10), default=4)
    parser.add_argument('--device', default='auto')
    parser.add_argument('--batch_size', type=int)
    parser.add_argument('--workers', type=int)
    parser.add_argument('--cpu_threads', type=int)
    parser.add_argument('--embedding_dtype', choices=('float16', 'float32'))
    parser.add_argument('--no_embeddings', action='store_true', default=None)
    parser.add_argument('--no_attention', action='store_true', default=None)
    parser.add_argument('--cpu_parallel_limit', type=int, default=2)
    parser.add_argument('--cpu_budget', type=int)
    parser.add_argument('--gpu_reserve_gib', type=float, default=2)
    parser.add_argument('--ram_reserve_gib', type=float, default=3)
    parser.add_argument('--reserve_fraction', type=float, default=.10)
    parser.add_argument('--memory_safety_factor', type=float, default=1.35)
    parser.add_argument('--disk_reserve_gib', type=float, default=8)
    parser.add_argument('--disk_bytes_per_sample', type=int, default=1536)
    parser.add_argument('--profile_batches', type=int, default=8)
    parser.add_argument('--profile_min_seconds', type=float, default=5)
    parser.add_argument('--poll_seconds', type=float, default=2)
    parser.add_argument('--resource_wait_seconds', type=float, default=3600)
    parser.add_argument('--oom_retries', type=int, choices=(0, 1), default=1)
    parser.add_argument('--bootstrap', type=int, default=200)
    parser.add_argument('--analysis_seed', type=int, default=42)
    parser.add_argument('--launcher_log', type=Path, help=argparse.SUPPRESS)
    parser.add_argument('--pid_file', type=Path, help=argparse.SUPPRESS)
    parser.add_argument('--latest_log', type=Path, help=argparse.SUPPRESS)
    args = parser.parse_args(raw)
    resolve_paths(args)
    inherit_prediction_settings(args, explicitly_set)
    if args.device not in ('auto', 'cpu', 'cuda') and not re.fullmatch(r'cuda:\d+', args.device):
        parser.error('device 仅支持 auto、cpu、cuda 或 cuda:N')
    if args.workers != 0:
        parser.error('流式 mmap 入口要求沿用 workers=0')
    positive = ('batch_size', 'cpu_threads', 'cpu_parallel_limit', 'disk_bytes_per_sample', 'profile_batches')
    if any(getattr(args, name) <= 0 for name in positive) or args.bootstrap < 0:
        parser.error('计数参数必须为正，bootstrap >= 0')
    if args.cpu_budget is not None and args.cpu_budget < args.cpu_threads + args.workers:
        parser.error('cpu_budget 至少能容纳一个预测进程')
    finite = ('gpu_reserve_gib', 'ram_reserve_gib', 'reserve_fraction', 'memory_safety_factor',
              'disk_reserve_gib', 'profile_min_seconds', 'poll_seconds', 'resource_wait_seconds')
    if not all(math.isfinite(getattr(args, name)) for name in finite):
        parser.error('资源设置必须为有限数值')
    if min(args.gpu_reserve_gib, args.ram_reserve_gib, args.disk_reserve_gib) < 0:
        parser.error('保留空间不能为负')
    if args.memory_safety_factor < 1 or not 0 <= args.reserve_fraction < 1:
        parser.error('安全系数至少1，reserve_fraction位于[0,1)')
    if min(args.profile_min_seconds, args.poll_seconds, args.resource_wait_seconds) <= 0:
        parser.error('探测、轮询与等待秒数必须大于0')
    return args


def prepare_command(args):
    command = [sys.executable, '-u', str(HERE/'prepare_singleinput.py'),
               '--previous_output', str(args.previous_output), '--training_dir', str(args.training_dir),
               '--output_root', str(args.output_root), '--reuse_mode', args.reuse_mode,
               '--disk_reserve_gib', str(args.disk_reserve_gib), '--parent_queue_pid', str(os.getpid())]
    if args.resume:
        command.append('--resume')
    if args.check_only:
        command.append('--check_only')
    return command


def run_serial(command, log_path, args, stage, state, state_path, commands_path, run_id):
    """长时间准备/分析时也监测输出盘，失败只终止本次子进程。"""
    child = queue.start_process(command, log_path, args, stage, state, state_path, commands_path, run_id)
    try:
        while child['process'].poll() is None:
            queue.check_disk(args)
            time.sleep(min(args.poll_seconds, 2))
        code = queue.finish_process(child, state, state_path, commands_path)
    except BaseException:
        queue.terminate_owned(child['process'])
        child['handle'].close()
        raise
    if code:
        raise RuntimeError(stage + '失败，code=' + str(code) + '，日志：' + str(log_path))


def validate_plan(plan, args):
    """分析/续跑沿用固定六组模型和来源，不隐式扩展评估范围。"""
    from prepare_singleinput import scientific_plan, code_fingerprints
    expected = {(scheme+'/gatedconv_target/seed_'+str(seed), 'gatedconv_target', seed, scheme)
                for scheme in SCHEMES for seed in SEEDS}
    runs = plan.get('runs', [])
    actual = {(r.get('run_id'), r.get('model'), r.get('seed'), r.get('feature_scheme')) for r in runs}
    if len(runs) != 12 or actual != expected or plan.get('version') != 'singleinput-external-1.0.0':
        raise ValueError('评估计划必须为四种输入方案×三个固定种子的12项 GatedConv 结果')
    if len(plan.get('datasets', [])) != 72 or len({d['dataset_id'] for d in plan['datasets']}) != 72:
        raise ValueError('评估计划必须包含72个不同的冻结数据文件')
    if 'replacement_dataset_ids' in plan:
        raise ValueError('本次必须评价全部72文件，不能只跑更正过的8份文件')
    if plan.get('prediction_settings') != queue.prediction_settings(args):
        raise ValueError('冻结计划与预测导出设置不符')
    from figure4_common import identity_hash
    if plan.get('plan_id') != identity_hash(scientific_plan(plan)):
        raise ValueError('冻结计划身份哈希不匹配')
    provenance = plan.get('comparison_provenance', {})
    for key in ('previous_output', 'training_dir'):
        if Path(provenance.get(key, '')).resolve() != getattr(args, key):
            raise ValueError('计划与本次来源参数不同：' + key)
    if provenance.get('reuse_mode') != args.reuse_mode:
        raise ValueError('计划与本次reuse_mode不同')
    source_files = provenance.get('source_metadata_files')
    if not isinstance(source_files, list) or not source_files:
        raise ValueError('计划缺少冻结来源元数据清单')
    for item in source_files:
        if sha256(item['path']) != item['sha256']:
            raise ValueError('冻结来源元数据改变：' + str(item['path']))
    if plan.get('comparison_code_sha256') != code_fingerprints():
        raise ValueError('冻结后评价代码改变；保留原包续跑或另建输出目录')
    for run in runs:
        verify_run(run)
    return plan


def pending_runs(args, plan):
    pending = []
    for run in plan['runs']:
        if queue.run_is_complete(args, run, plan):
            continue
        if run.get('feature_scheme') == 'full':
            raise RuntimeError('旧full复用不完整，停止而不重新推理full：' + run['run_id'])
        pending.append(run)
    return pending


def summary_command(args, plot_only=False):
    command = [sys.executable, '-u', str(HERE/'summarize_singleinput_external.py'), '--output_root', str(args.output_root)]
    if plot_only:
        command.append('--plot_only')
    return command


@contextmanager
def source_read_locks(args):
    """只读持有旧来源共享锁，防止正常旧队列边复用边改写。"""
    entries = [(args.previous_output/'.figure4_queue.lock', '旧 Figure4 队列'),
               (args.training_dir/'.singleinput.lock', '单输入训练启动器'),
               (args.training_dir/'.queue.lock', '单输入训练队列')]
    with ExitStack() as stack:
        for path, label in entries:
            if not path.is_file():
                continue
            handle = stack.enter_context(path.open('r', encoding='utf-8'))
            try:
                fcntl.flock(handle.fileno(), fcntl.LOCK_SH | fcntl.LOCK_NB)
            except BlockingIOError as error:
                raise RuntimeError(label+'仍在运行，请等待其退出后再比较：'+str(path)) from error
        yield


def preflight(args):
    import subprocess
    # Confirm imports only. GPU probing and genuine prediction profiles happen in the formal queue.
    import torch, numpy, pandas, scipy, sklearn, matplotlib
    print('[ENV] Python='+sys.version.split()[0]+' PyTorch='+str(torch.__version__)+
          ' NumPy='+str(numpy.__version__), flush=True)
    if args.device.startswith('cuda') and not torch.cuda.is_available():
        raise RuntimeError('请求 CUDA，但当前环境没有可用 CUDA')
    with source_read_locks(args):
        subprocess.run(prepare_command(args), check=True, env=queue.environment(args))
    previous = queue.read_json(args.previous_output/'evaluation_plan.json')
    frozen = args.output_root/'evaluation_plan.json'
    if frozen.is_file():
        plan = validate_plan(queue.read_json(frozen), args)
        required = queue.estimate_remaining_bytes(args, plan, pending_runs(args, plan))
    else:
        n = sum(int(d['n_samples']) for d in previous['datasets']) * 9
        per_sample = args.disk_bytes_per_sample
        if not args.no_embeddings:
            per_sample += 192*(4 if args.embedding_dtype == 'float32' else 2)
        if not args.no_attention:
            per_sample += 24
        required = n*per_sample
    target = args.output_root
    while not target.exists():
        target = target.parent
    available = shutil.disk_usage(target).free
    print('[DISK CHECK] 尚未完成预测的保守估算 {:.2f} GiB；另保留 {:.2f} GiB；当前可用 {:.2f} GiB'.format(
        required/queue.GIB, args.disk_reserve_gib, available/queue.GIB), flush=True)
    if available < required + math.ceil(args.disk_reserve_gib*queue.GIB):
        raise OSError(28, '预检空间不足；请释放空间或使用有足够空间的新 --output_dir')
    print('[CHECK PASS] 来源、9个检查点、完整输入复用记录和72份外部数据预检通过；尚未预测。', flush=True)
    return 0


def main(argv=None):
    args = parse_args(argv)
    print('[VERSION] Figure4 single-input external 1.0.0；9项单输入预测+3项full复用，无训练', flush=True)
    if args.check_only:
        return preflight(args)
    args.output_root.mkdir(parents=True, exist_ok=True)
    (args.output_root/'logs').mkdir(exist_ok=True)
    state_path = args.output_root/'queue_state.json'
    commands_path = args.output_root/'commands.jsonl'
    invocation = datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%S_%f_')+str(os.getpid())
    with queue.output_lock(args.output_root), source_read_locks(args):
        queue.validate_jsonl(commands_path)
        queue.validate_jsonl(args.output_root/'resource_events.jsonl')
        previous = queue.read_json(state_path) if state_path.is_file() else {}
        queue.assert_no_live_children(previous)
        if previous and args.phase in ('all', 'predict') and not args.resume:
            raise RuntimeError('此新输出目录已有队列；保持参数不变并加--resume')
        queue.publish_launcher(args)
        old_cap = int(previous.get('effective_max_parallel', args.max_parallel))
        effective = min(args.max_parallel, old_cap) if args.resume else args.max_parallel
        state = dict(version='figure4-singleinput-v1.0.0', invocation_id=invocation,
            status='running', phase=args.phase, started_at=queue.utc_now(), effective_max_parallel=effective,
            requested_max_parallel=args.max_parallel, active_children=[], runs=previous.get('runs', {}),
            history=list(previous.get('history', [])))
        if previous:
            state['history'].append({key:previous.get(key) for key in ('invocation_id','started_at','finished_at','status','error')})
        queue.write_json(state_path, state)
        queue.write_json(args.output_root/'logs'/invocation/'invocation.json',
            dict(invocation_id=invocation, started_at=state['started_at'], python_executable=sys.executable,
                 python_version=sys.version, cuda_visible_devices=os.environ.get('CUDA_VISIBLE_DEVICES'),
                 arguments={key:str(value) if isinstance(value,Path) else value for key,value in vars(args).items()},
                 prediction_settings=queue.prediction_settings(args)))
        try:
            queue.check_disk(args)
            if args.phase in ('all','predict'):
                run_serial(prepare_command(args), args.output_root/'logs'/invocation/'prepare.log', args,
                    'prepare', state, state_path, commands_path, 'prepare')
            plan_path = args.output_root/'evaluation_plan.json'
            if not plan_path.is_file():
                raise FileNotFoundError('没有冻结评估计划，请先运行all/predict：' + str(plan_path))
            plan = validate_plan(queue.read_json(plan_path), args)
            state['plan_id'] = plan['plan_id']
            queue.write_json(state_path, state)
            if args.phase in ('all','predict'):
                pending = pending_runs(args, plan)
                pending_ids = {run['run_id'] for run in pending}
                for run in plan['runs']:
                    if run['run_id'] not in pending_ids:
                        state['runs'][run['run_id']] = dict(status='skipped_matching_completed')
                required = queue.estimate_remaining_bytes(args, plan, pending)
                queue.check_disk(args, required)
                state['remaining_output_estimate_bytes'] = required
                print('[DISK] 剩余预测预计新增{:.2f} GiB，另保留{:.2f} GiB'.format(required/queue.GIB,args.disk_reserve_gib),flush=True)
                if pending:
                    mapping, hardware = queue.discover_devices(args)
                    state['hardware'] = hardware
                    queue.write_json(state_path,state)
                    profiles = queue.profile_runs(args, plan, pending, mapping, invocation, state, state_path, commands_path)
                    queue.predict_queue(args, plan, pending, profiles, mapping, invocation, state, state_path, commands_path)
                # 预测退出后重新核验完成文件，防止误将只有进程退出视为成功。
                if pending_runs(args, plan):
                    raise RuntimeError('仍有单输入文件未完成，禁止进入汇总')
                state['prediction_status'] = 'complete'
                queue.write_json(state_path,state)
            if args.phase in ('all','analyze'):
                if args.phase == 'analyze' and pending_runs(args, plan):
                    raise RuntimeError('预测产物尚未完整，先用--resume继续预测，再重做分析')
                command = [sys.executable,'-u',str(HERE/'figure4_analyze.py'),'--output_root',str(args.output_root),
                           '--bootstrap',str(args.bootstrap),'--analysis_seed',str(args.analysis_seed)]
                run_serial(command,args.output_root/'logs'/invocation/'analysis.log',args,
                    'analyze',state,state_path,commands_path,'analysis')
                run_serial(summary_command(args),args.output_root/'logs'/invocation/'comparison.log',args,
                    'comparison',state,state_path,commands_path,'comparison')
            elif args.phase == 'plot':
                run_serial(summary_command(args,plot_only=True),args.output_root/'logs'/invocation/'comparison_plot.log',args,
                    'plot',state,state_path,commands_path,'comparison_plot')
            state.update(status='complete',finished_at=queue.utc_now(),active_children=[])
            if args.phase in ('all','analyze','plot'):
                package=args.output_root/RESULT_ARCHIVE
                if not package.is_file():
                    raise RuntimeError('汇总未生成预期结果包：'+str(package))
                state['result_archive']=str(package)
                queue.write_json(state_path,state)
                from summarize_singleinput_external import make_core_archive
                make_core_archive(args.output_root)
                print('[RESULT PACKAGE] '+str(package),flush=True)
                print('[FULL PASS] Figure4九个单输入模型外部评价、四方案比较及打包完成：'+str(args.output_root),flush=True)
            else:
                queue.write_json(state_path,state)
                print('[PREDICTION PASS] 全部12组冻结结果已完成；运行--phase analyze生成比较图和结果包。',flush=True)
            return 0
        except BaseException as error:
            state.update(status='interrupted' if isinstance(error,KeyboardInterrupt) else 'failed',
                error=type(error).__name__+': '+str(error),finished_at=queue.utc_now())
            try:queue.write_json(state_path,state)
            except OSError:pass
            raise


if __name__ == '__main__':
    def interrupted(signum, frame):
        raise KeyboardInterrupt('收到信号 '+str(signum))
    signal.signal(signal.SIGTERM, interrupted)
    try:
        raise SystemExit(main())
    except KeyboardInterrupt as error:
        print('[INTERRUPTED] '+str(error)+'；保留现场，使用--resume。',file=sys.stderr)
        raise SystemExit(130)
    except Exception as error:
        print('[ERROR] '+type(error).__name__+': '+str(error),file=sys.stderr)
        raise SystemExit(1)
