#!/usr/bin/env python3
"""冻结三个已有 GatedConv 检查点，补充固定内部测试及同种子比较。"""
from __future__ import annotations

import argparse
from datetime import datetime, timezone
import json
import math
import os
from pathlib import Path
import signal
import shutil
import subprocess
import sys
import tarfile
import time

from common import (DEFAULT_DATA_DIR, DEFAULT_RESULTS_DIR, SEEDS, atomic_json,
                    file_lock, prepare_plan, read_json, sha256, timestamp, verify_plan)
from evaluate_worker import validate_prediction_complete, validate_run_complete

HERE = Path(__file__).resolve().parent
GIB = 1024 ** 3
ARCHIVE = 'gatedconv_internal_test_results_for_analysis_v1_0_0.tar.gz'


def arguments(argv=None):
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--data_dir', default=DEFAULT_DATA_DIR)
    p.add_argument('--results_dir', default=DEFAULT_RESULTS_DIR)
    p.add_argument('--output_dir')
    p.add_argument('--cache_dir')
    for seed in SEEDS:
        p.add_argument('--seed' + str(seed) + '_dir')
    p.add_argument('--device', default='auto')
    p.add_argument('--max_parallel', type=int, choices=(1, 2, 3), default=3)
    p.add_argument('--cpu_threads', type=int, default=4)
    p.add_argument('--gpu_reserve_gib', type=float, default=2.0)
    p.add_argument('--ram_reserve_gib', type=float, default=3.0)
    p.add_argument('--disk_reserve_gib', type=float, default=8.0)
    p.add_argument('--reserve_fraction', type=float, default=.10)
    p.add_argument('--memory_safety_factor', type=float, default=1.35)
    p.add_argument('--resource_wait_seconds', type=float, default=1800)
    p.add_argument('--bootstrap', type=int, default=200)
    p.add_argument('--phase', choices=('all', 'predict', 'analyze'), default='all')
    p.add_argument('--resume', action='store_true')
    p.add_argument('--launcher_log', help=argparse.SUPPRESS)
    args = p.parse_args(argv)
    args.data_dir = str(Path(args.data_dir).expanduser().resolve())
    args.results_dir = str(Path(args.results_dir).expanduser().resolve())
    args.output_dir = str(Path(args.output_dir or Path(args.results_dir)/'gatedconv_internal_test_v1_0_0').expanduser().resolve())
    output = Path(args.output_dir)
    protected = [Path(args.results_dir)/'main_20']
    for seed in SEEDS:
        override = getattr(args,'seed'+str(seed)+'_dir')
        folder = 'litemamba_structure_seed42_v1_0_0' if seed == 42 else 'litemamba_structure_confirm3_v1_1_0'
        protected.append(Path(override).expanduser().resolve() if override else Path(args.results_dir)/folder/'full'/'gatedconv_target'/('seed_'+str(seed)))
    for source in protected:
        if output == source or output in source.parents or source in output.parents:
            p.error('新输出目录必须与原训练运行目录独立：'+str(source))
    for source in (HERE,Path(args.data_dir),Path(args.results_dir)):
        if output == source or output in source.parents:
            p.error('新输出不能覆盖代码、数据或结果根目录：'+str(source))
    if HERE in output.parents:
        p.error('新输出请放在训练结果根目录下的独立文件夹')
    if args.cpu_threads < 1 or args.bootstrap < 0:
        p.error('cpu_threads 必须正数，bootstrap 必须非负')
    names = ('gpu_reserve_gib', 'ram_reserve_gib', 'disk_reserve_gib', 'reserve_fraction',
             'memory_safety_factor', 'resource_wait_seconds')
    if not all(math.isfinite(getattr(args, name)) for name in names):
        p.error('资源参数必须是有限数值')
    if min(args.gpu_reserve_gib, args.ram_reserve_gib, args.disk_reserve_gib) < 0 or not 0 <= args.reserve_fraction < 1 or args.memory_safety_factor < 1 or args.resource_wait_seconds <= 0:
        p.error('资源预留非负、安全系数至少1、等待时间正数、reserve_fraction 在 [0,1)')
    return args


def disk_check(args, required=0):
    free = shutil.disk_usage(args.output_dir).free
    minimum = int(args.disk_reserve_gib * GIB) + int(required)
    if free < minimum:
        raise OSError(f'磁盘剩余 {free/GIB:.2f} GiB，当前阶段需要 {required/GIB:.2f} GiB 并保留 {args.disk_reserve_gib:.2f} GiB；清理空间后使用 --resume')


def worker_command(args, mode, seed=None, profile_out=None):
    cmd = [sys.executable, '-u', str(HERE/'evaluate_worker.py'), '--mode', mode,
           '--output_dir', args.output_dir, '--device', args.device,
           '--cpu_threads', str(args.cpu_threads), '--bootstrap', str(args.bootstrap)]
    if seed is not None:
        cmd += ['--seed', str(seed)]
    if profile_out is not None:
        cmd += ['--profile_out', str(profile_out)]
    return cmd


def snapshot(args):
    result = subprocess.run(worker_command(args, 'snapshot'), capture_output=True, text=True, timeout=60)
    if result.returncode:
        raise RuntimeError('设备检查失败：' + result.stderr[-3000:])
    for line in reversed(result.stdout.splitlines()):
        try:
            value = json.loads(line)
            if isinstance(value, dict) and 'chosen_device' in value:
                return value
        except json.JSONDecodeError:
            continue
    raise RuntimeError('设备检查未返回可用的资源快照')


def start_child(command, log_path):
    log_path.parent.mkdir(parents=True, exist_ok=True)
    handle = log_path.open('a', encoding='utf-8')
    environment = dict(os.environ, PYTHONUNBUFFERED='1', MPLBACKEND='Agg')
    process = subprocess.Popen(command, stdout=handle, stderr=subprocess.STDOUT,
                               env=environment, start_new_session=True)
    return {'process': process, 'handle': handle, 'log': str(log_path), 'command': command}


def terminate_child(child):
    process = child['process']
    if process.poll() is None:
        os.killpg(process.pid, signal.SIGTERM)
        try:
            process.wait(timeout=10)
        except subprocess.TimeoutExpired:
            os.killpg(process.pid, signal.SIGKILL)
            process.wait(timeout=10)
    child['handle'].close()


def previous_child_alive(item, args):
    """避免将已回收给无关程序的 PID 误判为本次评估子进程。"""
    pid = int(item['pid'])
    try:
        os.kill(pid,0)
    except ProcessLookupError:
        return False
    try:
        tokens = Path('/proc')/str(pid)/'cmdline'
        command = tokens.read_bytes().split(b'\0')
        return (str(HERE/'evaluate_worker.py').encode() in command and
                args.output_dir.encode() in command)
    except OSError:
        return True


def serial(command, log, args, stage):
    child = start_child(command, log)
    print(f'[{stage}] pid={child["process"].pid} log={log}', flush=True)
    try:
        while child['process'].poll() is None:
            disk_check(args)
            time.sleep(1)
        code = child['process'].returncode
    except BaseException:
        terminate_child(child)
        raise
    child['handle'].close()
    if code:
        raise RuntimeError(f'{stage} 失败，退出码 {code}；日志：{log}')


def admission(profile, resource, active_count, args):
    """对活动任务再次扣除完整预算；无需猜测逻辑/物理 GPU 编号。"""
    factor = args.memory_safety_factor
    host = resource['host']
    reserve_ram = max(args.ram_reserve_gib * GIB, host['total_bytes'] * args.reserve_fraction)
    host_need = int(profile['host_admission_estimate_bytes'] * factor)
    # OS page cache 及已运行进程可能已计入 available；再扣完整预算偏保守。
    shared = int(profile.get('shared_cache_bytes', 0))
    if host['available_bytes'] - reserve_ram - shared - active_count * host_need < host_need:
        return False, 'RAM'
    cpu_count = int(resource.get('available_cpu_slots', os.cpu_count() or 1))
    if (active_count + 1) * args.cpu_threads > cpu_count:
        return False, 'CPU'
    if args.device.startswith('cuda'):
        gpu = resource['gpu']
        reserve_gpu = max(args.gpu_reserve_gib * GIB, gpu['total_bytes'] * args.reserve_fraction)
        gpu_need = int(profile['gpu_admission_estimate_bytes'] * factor)
        if gpu['free_bytes'] - reserve_gpu - active_count * gpu_need < gpu_need:
            return False, 'GPU'
    return True, 'admitted'


def wait_analysis_memory(args, sample_count):
    """按原分析器保守行预算等待内存，避免分析阶段与别的作业抢满RAM。"""
    sys.path.insert(0,str(HERE/'vendor'))
    from resource_scheduler import memory_snapshot
    need = int(sample_count)*8192 + GIB
    started = time.monotonic()
    last_report = 0.0
    while True:
        host = memory_snapshot()
        reserve = max(args.ram_reserve_gib*GIB,host['total_bytes']*args.reserve_fraction)
        if host['available_bytes']-reserve >= need:
            return
        disk_check(args)
        if time.monotonic()-started >= args.resource_wait_seconds:
            raise RuntimeError('离线分析等待内存超时；释放RAM后使用 --phase analyze --resume')
        if time.monotonic()-last_report >= 30:
            print(f'[WAIT RAM] 分析预算 {need/GIB:.2f} GiB，另保留 {reserve/GIB:.2f} GiB',flush=True)
            last_report = time.monotonic()
        time.sleep(2)


def predict(args, plan, logs, state):
    pending = [seed for seed in SEEDS if not validate_prediction_complete(args.output_dir, seed, plan)]
    if not pending:
        print('[SKIP] 三个种子的完整预测均已保存', flush=True)
        return
    profile_path = logs/'resource_profile.json'
    serial(worker_command(args, 'profile', pending[0], profile_path), logs/'profile.log', args, 'PROFILE')
    profile = read_json(profile_path)
    profile['shared_cache_bytes'] = sum(p.stat().st_size for p in Path(plan['dataset']['cache_path']).iterdir() if p.is_file())
    cap = min(args.max_parallel, int(state.get('effective_max_parallel', args.max_parallel)))
    if args.device == 'cpu':
        cap = min(cap, 2)
    state['effective_max_parallel'] = cap
    active, retries = {}, {}
    waited_at = time.monotonic()
    last_wait_report = 0.0
    try:
        while pending or active:
            disk_check(args)
            for seed, child in list(active.items()):
                code = child['process'].poll()
                if code is None:
                    continue
                child['handle'].close()
                del active[seed]
                if code == 0 and validate_prediction_complete(args.output_dir, seed, plan, strict=True):
                    print(f'[PREDICT DONE] seed={seed}', flush=True)
                else:
                    message = Path(child['log']).read_text(encoding='utf-8', errors='replace')[-20000:]
                    oom = any(token in message.lower() for token in ('cuda out of memory', 'torch.cuda.outofmemoryerror', 'cannot allocate memory'))
                    if oom and retries.get(seed, 0) == 0:
                        retries[seed] = 1
                        cap = max(1, cap-1)
                        state['effective_max_parallel'] = cap
                        pending.insert(0, seed)
                        print(f'[OOM] seed={seed}，并行上限降至 {cap}，同一权重/批大小重试一次', flush=True)
                    else:
                        raise RuntimeError(f'预测失败 seed={seed} code={code}；日志：{child["log"]}')
                waited_at = time.monotonic()
            launched = False
            if pending and len(active) < cap:
                resource = snapshot(args)
                allowed, reason = admission(profile, resource, len(active), args)
                if allowed:
                    seed = pending.pop(0)
                    attempt = retries.get(seed, 0)+1
                    child = start_child(worker_command(args, 'predict', seed), logs/f'predict_seed_{seed}_attempt_{attempt}.log')
                    active[seed] = child
                    print(f'[PREDICT] seed={seed} pid={child["process"].pid} 并行={len(active)}/{cap} log={child["log"]}', flush=True)
                    launched = True
                    waited_at = time.monotonic()
                elif not active:
                    if time.monotonic()-waited_at > args.resource_wait_seconds:
                        raise RuntimeError('等待资源超时：'+reason+'；释放资源后使用 --resume')
                    if time.monotonic()-last_wait_report >= 30:
                        print('[WAIT] 当前 '+reason+' 预算不足，等待资源释放', flush=True)
                        last_wait_report = time.monotonic()
            state['active_children'] = [{'seed':seed,'pid':child['process'].pid,'log':child['log']} for seed,child in active.items()]
            state['updated_at'] = timestamp()
            atomic_json(Path(args.output_dir)/'queue_state.json', state)
            if not launched:
                time.sleep(1)
    finally:
        for child in active.values():
            terminate_child(child)


def package_results(output):
    """打包统计、审计、日志和源码；完整逐样本输出保留服务器。"""
    output = Path(output)
    archive = output/ARCHIVE
    temporary = archive.with_name(archive.name+'.partial')
    include = []
    for folder in ('analysis', 'source_metadata', 'logs', 'code_snapshot'):
        include += [p for p in (output/folder).rglob('*') if p.is_file()]
    for name in ('evaluation_plan.json', 'queue_state.json', 'resource_snapshot.json'):
        if (output/name).is_file():
            include.append(output/name)
    for seed in SEEDS:
        root = output/'runs'/'gatedconv_target'/f'seed_{seed}'
        include += [p for p in root.rglob('*') if p.is_file() and
                    (p.suffix == '.json' or ('analysis' in p.relative_to(root).parts and p.suffix in ('.csv','.svg','.png','.html')))]
    # latest.log 是服务器绝对路径符号链接；小包只收录可独立打开的普通文件。
    include = [p for p in set(include) if not p.is_symlink()]
    manifest = {str(p.relative_to(output)): {'sha256':sha256(p),'size_bytes':p.stat().st_size} for p in sorted(include)}
    atomic_json(output/'result_package_manifest.json', {'files':manifest,'purpose':'analysis-only small package; all per-sample arrays and predictions.csv.gz remain on server'})
    with tarfile.open(temporary, 'w:gz') as tar:
        for path in sorted(set(include)):
            tar.add(path, arcname=str(path.relative_to(output)), recursive=False)
        tar.add(output/'result_package_manifest.json', arcname='result_package_manifest.json')
    temporary.replace(archive)
    return archive


def main(argv=None):
    args = arguments(argv)
    output = Path(args.output_dir)
    output.mkdir(parents=True, exist_ok=True)
    def interrupted(signum, frame):
        raise KeyboardInterrupt('收到终止信号 '+str(signum))
    signal.signal(signal.SIGTERM, interrupted)
    with file_lock(output/'evaluation.lock'):
        prior = read_json(output/'queue_state.json') if (output/'queue_state.json').is_file() else {}
        if prior and not args.resume:
            raise RuntimeError('已有队列记录；保持原参数并加 --resume')
        # 锁文件不会因上次中断残留而阻止续跑。父进程意外退出时不接管仍活跃子进程。
        for item in prior.get('active_children', []):
            if previous_child_alive(item,args):
                raise RuntimeError(f'上次记录的子进程 PID={item["pid"]} 仍存在；请先确认旧任务已结束，避免重复预测')
        log_dir = output/'logs'/datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%S_%f')
        log_dir.mkdir(parents=True)
        if args.launcher_log:
            (output/'logs'/'latest.log').unlink(missing_ok=True)
            (output/'logs'/'latest.log').symlink_to(Path(args.launcher_log).resolve())
        (output/'logs'/'evaluation.pid').write_text(str(os.getpid())+'\n')
        state = {'status':'running','started_at':timestamp(),'pid':os.getpid(),'active_children':[],
                 'effective_max_parallel':prior.get('effective_max_parallel',args.max_parallel),'phase':args.phase}
        atomic_json(output/'queue_state.json',state)
        print(f'[ACCEPTED] pid={os.getpid()} output={output}',flush=True)
        try:
            # 新缓存、三个float32嵌入及预测/分析输出预留4GiB，另加磁盘余量。
            disk_check(args, 4*GIB if not (output/'evaluation_plan.json').exists() else 0)
            if args.phase == 'analyze' and (output/'evaluation_plan.json').exists():
                # 离线分析无需打开 GPU；仍保留计划原来的预测设备身份。
                saved = read_json(output/'evaluation_plan.json')
                selected = saved['prediction_settings']['device']
                host_args = argparse.Namespace(**vars(args))
                host_args.device = 'cpu'
                resource = snapshot(host_args)
                args.device = selected
            else:
                resource = snapshot(args)
                args.device = resource['chosen_device']
            atomic_json(output/'resource_snapshot.json',resource)
            print('[DEVICE] '+args.device+'；固定三个种子42/2024/2025，使用已有权重',flush=True)
            plan = prepare_plan(args)
            verify_plan(plan)
            snapshot_dir = output/'code_snapshot'
            for name in plan['code_manifest']:
                destination = snapshot_dir/name
                destination.parent.mkdir(parents=True,exist_ok=True)
                if not destination.exists():
                    shutil.copyfile(HERE/name,destination)
            if args.phase in ('all','predict'):
                predict(args,plan,log_dir,state)
            if args.phase in ('all','analyze'):
                for seed in SEEDS:
                    if not validate_prediction_complete(args.output_dir,seed,plan,strict=True):
                        raise RuntimeError('缺少完整预测：seed='+str(seed)+'；先使用 --phase all --resume')
                    if validate_run_complete(args.output_dir,seed,plan):
                        print('[ANALYZE SKIP] seed='+str(seed),flush=True)
                        continue
                    # 单独串行分析避免三个pandas/cluster bootstrap进程叠加内存。
                    wait_analysis_memory(args,plan['dataset']['n_samples'])
                    serial(worker_command(args,'analyze',seed),log_dir/f'analysis_seed_{seed}.log',args,'ANALYZE')
                serial([sys.executable,'-u',str(HERE/'summarize_internal_test.py'),'--output_dir',args.output_dir],log_dir/'summary.log',args,'SUMMARY')
                if not all(validate_run_complete(args.output_dir,seed,plan,strict=True) for seed in SEEDS):
                    raise RuntimeError('最终完成记录校验失败')
                state.update(status='complete',completed_at=timestamp(),active_children=[])
                atomic_json(output/'queue_state.json',state)
                disk_check(args)
                archive = package_results(output)
                print('[RESULT PACKAGE] '+str(archive),flush=True)
                print('[FULL PASS] 三个GatedConv内部测试、串行分析和同种子汇总全部完成',flush=True)
            else:
                state.update(status='prediction_complete',active_children=[],completed_at=timestamp())
                atomic_json(output/'queue_state.json',state)
                print('[PREDICTION PASS] 完整预测已保存；使用 --phase analyze --resume 完成统计',flush=True)
        except BaseException as error:
            state.update(status='interrupted' if isinstance(error,KeyboardInterrupt) else 'failed',error=str(error),active_children=[],updated_at=timestamp())
            atomic_json(output/'queue_state.json',state)
            raise


if __name__ == '__main__':
    try:
        main()
    except (Exception,KeyboardInterrupt) as exc:
        print('[ERROR] '+type(exc).__name__+': '+str(exc),file=sys.stderr,flush=True)
        sys.exit(1)
