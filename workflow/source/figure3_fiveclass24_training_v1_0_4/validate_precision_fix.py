"""Run a bounded real-fault regression gate before any precision migration."""
from __future__ import annotations
import hashlib
import json
import shutil
import os
from datetime import datetime
from pathlib import Path
import subprocess
import sys

POLICY = 'fp32_compute_preserved_scaler_v1'
HERE = Path(__file__).resolve().parent


def digest(path):
    value = hashlib.sha256()
    with Path(path).open('rb') as stream:
        for chunk in iter(lambda: stream.read(4 * 1024 ** 2), b''): value.update(chunk)
    return value.hexdigest()


def run_gate(args, lock_fd=None):
    target = args.output_dir / 'precision_fix_validation'
    target.mkdir(exist_ok=True)
    reference = json.loads((HERE / 'reference/precision_fault_reference.json').read_text())
    engine_hashes = {path.name: digest(path) for path in sorted((HERE / 'engine').glob('*.py'))}
    identity = {'engine_sha256': engine_hashes,
                'validation_source_sha256': {name: digest(HERE / name) for name in
                                            ('validate_precision_fix.py', 'validate_precision_worker.py')},
                'reference_artifacts': reference['reference_artifacts'],
                'numerical_precision_policy': POLICY,
                'requested_device': args.device}
    key = hashlib.sha256(json.dumps(identity, sort_keys=True).encode()).hexdigest()
    saved = target / 'gate.json'
    if saved.is_file():
        old = json.loads(saved.read_text())
        if old.get('status') == 'PASS' and old.get('gate_key') == key:
            print('[PRECISION GATE REUSE] 已验证相同源码和故障批次；不重复验证', flush=True)
            return old
    if saved.is_file():
        archive = target / 'previous_gates' / (datetime.now().strftime('%Y%m%d_%H%M%S_%f') + '_' + str(os.getpid()))
        archive.mkdir(parents=True, exist_ok=False)
        shutil.copy2(saved, archive / 'gate.json')
        if (target / 'gate_request.json').is_file(): shutil.copy2(target / 'gate_request.json', archive / 'gate_request.json')
    explicit = getattr(args, 'precision_probe_dir', None)
    if explicit:
        candidates = [Path(explicit).expanduser().resolve()]
    else:
        candidates = sorted(Path(reference['default_probe_root']).glob('attempt_*'), reverse=True)
        candidates += sorted((args.output_dir.parent / 'figure3_fiveclass_nonfinite_probe_v1_0_0').glob('attempt_*'), reverse=True)
    chosen = None
    for directory in dict.fromkeys(candidates):
        if all((directory / name).is_file() and (directory / name).stat().st_size == record['size']
               and digest(directory / name) == record['sha256'] for name, record in reference['reference_artifacts'].items()):
            chosen = directory
            break
    if chosen is None:
        raise ValueError('未找到已捕获故障快照。请保留原诊断attempt目录，或加 --precision_probe_dir 指向同时含fault_state.pt和failed_batch.npz的目录；不会先迁移或启动训练。')
    request = {**identity, 'gate_key': key, 'probe_dir': str(chosen),
               'device': args.device, 'cpu_threads': args.cpu_threads}
    request_path = target / 'gate_request.json'
    request_path.write_text(json.dumps(request, ensure_ascii=False, indent=2) + '\n')
    print('[PRECISION GATE] 只验证已捕获故障批次的全FP32前向和反向；不重放整轮', flush=True)
    command = [sys.executable, '-u', str(HERE / 'validate_precision_worker.py'), str(request_path), str(saved)]
    result = subprocess.run(command, pass_fds=(() if lock_fd is None else (lock_fd,)))
    report = json.loads(saved.read_text()) if saved.exists() else {}
    if result.returncode or report.get('status') != 'PASS' or report.get('gate_key') != key:
        raise RuntimeError('故障批次修复验证未通过，原训练检查点尚未迁移；详见 ' + str(saved))
    print('[PRECISION GATE PASS] 修复后前向有限，已验证完整FP32有限反向与更新', flush=True)
    return report
