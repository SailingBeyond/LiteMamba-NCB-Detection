#!/usr/bin/env python3
"""Validate unchanged-architecture full-FP32 compute on the captured batch."""
from __future__ import annotations
import argparse
import copy
import contextlib
import hashlib
import json
import math
import os
from pathlib import Path
import sys
import traceback

sys.dont_write_bytecode = True
HERE = Path(__file__).resolve().parent


def finite_state(value):
    import torch
    if torch.is_tensor(value): return bool(torch.isfinite(value).all().item())
    if isinstance(value, dict): return all(finite_state(v) for v in value.values())
    if isinstance(value, (list, tuple)): return all(finite_state(v) for v in value)
    if isinstance(value, float): return math.isfinite(value)
    return True


def forward_summary(logits, loss):
    import torch
    finite = torch.isfinite(logits)
    return {'dtype': str(logits.dtype), 'all_logits_finite': bool(finite.all().item()),
            'nonfinite_logits': int((~finite).sum().item()),
            'min_finite_logit': float(logits[finite].min().item()) if bool(finite.any().item()) else None,
            'max_finite_logit': float(logits[finite].max().item()) if bool(finite.any().item()) else None,
            'loss_finite': bool(torch.isfinite(loss).item()),
            'loss_sum': float(loss.item()) if bool(torch.isfinite(loss).item()) else str(float(loss.item()))}


def main():
    request_path, report_path = map(Path, sys.argv[1:3])
    request = json.loads(request_path.read_text())
    report = {'status': 'FAIL', 'gate_key': request['gate_key'],
              'numerical_precision_policy': request['numerical_precision_policy'],
              'engine_sha256': request['engine_sha256'],
              'reference_artifacts': request['reference_artifacts'],
              'scope': 'Captured training batch only; independently loaded models; no official checkpoint writes or test/validation inference'}
    try:
        os.environ['CPLX_FIVECLASS_SCENARIO'] = 'X'
        os.environ['PYTHONDONTWRITEBYTECODE'] = '1'
        sys.path.insert(0, str(HERE / 'engine'))
        import numpy as np
        import torch
        import torch.nn.functional as F
        import train_all as engine
        reference = json.loads((HERE / 'reference/precision_fault_reference.json').read_text())
        probe = Path(request['probe_dir'])
        for name, record in request['reference_artifacts'].items():
            if hashlib.sha256((probe / name).read_bytes()).hexdigest() != record['sha256']:
                raise ValueError('故障快照SHA改变：' + name)
        fault = engine.torch_load(probe / 'fault_state.pt', 'cpu')
        options = argparse.Namespace(device=request['device'], amp=None, cpu_threads=request['cpu_threads'])
        device = engine.select_device(options)
        engine.set_seed(42)
        with np.load(probe / 'failed_batch.npz', allow_pickle=False) as batch:
            inputs = tuple(torch.from_numpy(batch['input_%d' % i].copy()).to(device) for i in range(3))
            labels = torch.from_numpy(batch['labels'].copy()).long().to(device)
        if not all(finite_state(v) for v in inputs) or not bool(((labels >= 0) & (labels < 5)).all().item()):
            raise ValueError('参考故障批次输入或标签非法')
        fixed = engine.build_model('litemamba', num_classes=5, seq_len=6, raw_signal_len=30,
                                   **reference['checkpoint_identity']['model_kwargs']).to(device)
        fixed.load_state_dict(fault['model_state_dict'], strict=True)
        fixed.train()
        legacy = copy.deepcopy(fixed)
        old_layer = legacy.cls[-1]
        replacement = torch.nn.Linear(old_layer.in_features, old_layer.out_features, bias=old_layer.bias is not None).to(device)
        replacement.load_state_dict(old_layer.state_dict())
        legacy.cls[-1] = replacement
        legacy.train()
        engine.restore_rng(fault['rng_state'])
        legacy_context = torch.autocast(device_type='cuda', dtype=torch.float16) if device.type == 'cuda' else contextlib.nullcontext()
        with legacy_context:
            legacy_logits, _ = legacy(inputs)
            legacy_loss = F.cross_entropy(legacy_logits, labels, reduction='sum')
        report['legacy'] = forward_summary(legacy_logits, legacy_loss)
        report['device'] = str(device)
        report['cuda_amp_gate'] = bool(device.type == 'cuda' and options.amp)
        report['cuda_amp_gate_scope'] = 'Legacy reference only; repaired computation uses FP32 throughout'
        report['actual_repaired_autocast_enabled'] = False
        report['saved_scaler_enabled'] = bool(options.amp)
        report['environment'] = engine.environment(device)
        del legacy, legacy_logits, legacy_loss
        if report['cuda_amp_gate'] and report['legacy']['all_logits_finite']:
            raise RuntimeError('原AMP故障在当前环境未复现；先核对环境，未开始迁移')
        report['backward_trials'] = []
        scaler_state = copy.deepcopy(fault['scaler_state_dict'])
        successful = False
        for trial in range(8):
            candidate = copy.deepcopy(fixed)
            candidate.train()
            optimizer = torch.optim.AdamW(candidate.parameters(), lr=0.001, weight_decay=.0001)
            optimizer.load_state_dict(copy.deepcopy(fault['optimizer_state_dict']))
            try: scaler = torch.amp.GradScaler('cuda', enabled=options.amp)
            except AttributeError: scaler = torch.cuda.amp.GradScaler(enabled=options.amp)
            if options.amp: scaler.load_state_dict(scaler_state)
            optimizer.zero_grad(set_to_none=True)
            engine.restore_rng(fault['rng_state'])
            activation_dtypes = {}
            def record_dtype(_module, _inputs, output):
                def visit(value):
                    if torch.is_tensor(value) and value.is_floating_point(): activation_dtypes[str(value.dtype)] = activation_dtypes.get(str(value.dtype), 0) + 1
                    elif isinstance(value, (list, tuple)):
                        for child in value: visit(child)
                    elif isinstance(value, dict):
                        for child in value.values(): visit(child)
                visit(output)
            handles = [module.register_forward_hook(record_dtype) for module in candidate.modules() if not any(module.children())]
            with engine.autocast_context(options, device):
                logits, _ = candidate(inputs)
                loss = F.cross_entropy(logits, labels, reduction='sum')
            for handle in handles: handle.remove()
            if any(name != 'torch.float32' for name in activation_dtypes): raise RuntimeError('全FP32策略中仍观测到低精度浮点激活')
            summary = forward_summary(logits, loss)
            if not summary['all_logits_finite'] or not summary['loss_finite'] or logits.dtype != torch.float32:
                raise RuntimeError('全模型FP32修复仍有非有限前向或输出不是float32')
            scaler.scale(loss).backward()
            scaler.unscale_(optimizer)
            for parameter in candidate.parameters():
                if parameter.grad is not None: parameter.grad.div_(labels.numel())
            gradients_finite = all(finite_state(p.grad) for p in candidate.parameters() if p.grad is not None)
            norm = torch.nn.utils.clip_grad_norm_(candidate.parameters(), 1.0)
            scale_before = float(scaler.get_scale())
            scaler.step(optimizer)
            scaler.update()
            scale_after = float(scaler.get_scale())
            skipped = bool(scaler.is_enabled() and scale_after < scale_before)
            state_finite = finite_state(candidate.state_dict()) and finite_state(optimizer.state_dict())
            trial_report = {'trial': trial + 1, 'forward': summary,
                            'activation_dtypes': activation_dtypes,
                            'unscaled_normalized_gradients_finite': gradients_finite,
                            'gradient_norm': float(norm.item()) if bool(torch.isfinite(norm).item()) else str(float(norm.item())),
                            'scale_before': scale_before, 'scale_after': scale_after,
                            'AMP_skipped': skipped, 'model_and_Adam_finite_after_step': state_finite}
            report['backward_trials'].append(trial_report)
            print('[FAULT BATCH FP32] trial=%d output=%s finite_loss=%s finite_grad=%s AMP_skip=%s' %
                  (trial + 1, str(logits.dtype), summary['loss_finite'], gradients_finite, skipped), flush=True)
            if not state_finite: raise RuntimeError('全模型修复的诊断更新后状态非有限')
            if gradients_finite and not skipped and bool(torch.isfinite(norm).item()):
                successful = True
                report['fixed'] = summary
                break
            if not options.amp: break
            # Diagnostic-only backoff on an independent copy of the captured
            # state. The original training scaler/checkpoint is never changed.
            scaler_state = copy.deepcopy(scaler.state_dict())
            del candidate, optimizer, scaler, logits, loss
        if not successful: raise RuntimeError('故障批次未在有限次独立诊断中通过有限反向，停止迁移')
        report['status'] = 'PASS'
        report['actual_training_scaler_modified'] = False
        report['CPU_limitation'] = None if report['cuda_amp_gate'] else 'CPU FP32 validation cannot establish CUDA FP16 fault reproduction.'
    except Exception as error:
        report['error'] = type(error).__name__ + ': ' + str(error)
        report['traceback'] = traceback.format_exc()
    temporary = report_path.with_suffix('.writing')
    temporary.write_text(json.dumps(report, indent=2, ensure_ascii=False, allow_nan=False) + '\n')
    temporary.replace(report_path)
    if report['status'] != 'PASS':
        print('[PRECISION GATE ERROR] ' + report.get('error', 'unknown'), flush=True)
        return 1
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
