#!/usr/bin/env python3
"""在CPU上验证旧full与本包full等价，并核验两个结构对照的定义。"""
from __future__ import annotations

import argparse
from datetime import datetime, timezone
import hashlib
import importlib.util
import json
from pathlib import Path
import sys


SOURCE_FILES = ('train_all.py', 'model_benchmark_all.py', 'config_6_all_6.py',
                'IterDataset.py', 'analyze_results.py', 'run_all_4_models.py',
                'resource_scheduler.py', 'cuda_device_probe.py')


def sha256(path):
    digest = hashlib.sha256()
    with Path(path).open('rb') as stream:
        for block in iter(lambda: stream.read(8 * 1024 * 1024), b''):
            digest.update(block)
    return digest.hexdigest()


def read_json(path):
    return json.loads(Path(path).read_text(encoding='utf-8'))


def save_json(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + '.tmp')
    temporary.write_text(json.dumps(value, ensure_ascii=False, indent=2, allow_nan=False) + '\n', encoding='utf-8')
    temporary.replace(path)


def load_module(path, name):
    specification = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(specification)
    sys.modules[name] = module
    specification.loader.exec_module(module)
    return module


def compare_state(torch, first, second, label):
    a, b = first.state_dict(), second.state_dict()
    if list(a) != list(b):
        raise AssertionError(label + ': state_dict keys differ')
    for key in a:
        if a[key].shape != b[key].shape or not torch.equal(a[key], b[key]):
            raise AssertionError(label + ': state tensor differs: ' + key)
    return len(a)


def compare_outputs(torch, first, second, label):
    a_logits, a_embedding, a_aux = first
    b_logits, b_embedding, b_aux = second
    for name, a, b in [('logits', a_logits, b_logits), ('embedding', a_embedding, b_embedding),
                       ('attention', a_aux['attention'], b_aux['attention'])]:
        if a.shape != b.shape or not torch.equal(a, b):
            raise AssertionError(label + ': ' + name + ' differs')
        if not torch.isfinite(a).all():
            raise AssertionError(label + ': nonfinite ' + name)


def inputs(torch, raw_len):
    generator = torch.Generator().manual_seed(20261023)
    sequence = torch.nn.functional.one_hot(torch.randint(0, 4, (3, 6), generator=generator), 5).float()
    sequence[:, 3, :] = 0
    sequence[:, 3, 4] = 1
    scalars = torch.randn(3, 6, 8, generator=generator)
    merged = torch.cat((sequence, scalars), dim=-1)
    raw = torch.randn(3, 6, raw_len, generator=generator)
    return sequence, merged, raw


def cloned_inputs(values):
    return tuple(value.detach().clone().requires_grad_(True) for value in values)


def parameter_counts(model):
    config = model.model_config
    dim = int(config['model_dim'])
    return {
        'total': sum(parameter.numel() for parameter in model.parameters()),
        'trainable': sum(parameter.numel() for parameter in model.parameters() if parameter.requires_grad),
        'classifier_input_width': 2 * dim,
        'classifier_data_active_input_width': config.get('classifier_data_active_input_width', 2 * dim),
        'classifier_data_inactive_input_weight_count': config.get('classifier_data_inactive_input_weight_count', 0),
        'note': config.get('parameter_count_note', 'All classifier input columns receive data-dependent embeddings.'),
    }


def torch_load(torch, path):
    try:
        return torch.load(path, map_location='cpu', weights_only=False)
    except TypeError:
        return torch.load(path, map_location='cpu')


def run_probe(original_engine, baseline_run):
    import torch

    original_engine = Path(original_engine).expanduser().resolve()
    baseline_run = Path(baseline_run).expanduser().resolve()
    patched_engine = Path(__file__).resolve().parent
    requested = read_json(baseline_run / 'queue_requested_config.json')
    resolved = read_json(baseline_run / 'resolved_config.json')
    if (requested.get('model_type'), requested.get('seed'), requested.get('feature_scheme')) != ('litemamba', 42, 'full'):
        raise ValueError('baseline must be full/litemamba/seed42')
    options = dict(resolved['model_kwargs'])
    if options.get('litemamba_variant') != 'selective_ssm' or options.get('readout') != 'target_attention':
        raise ValueError('baseline must use selective_ssm and target_attention')
    if options.get('target_index') != 3:
        raise ValueError('baseline target_index must be the fourth position, zero-based 3')
    original_sha = {name: sha256(original_engine / name) for name in SOURCE_FILES}
    if original_sha != requested.get('source_sha256'):
        raise ValueError('original engine source hashes do not match the baseline')
    patched_sha = {name: sha256(patched_engine / name) for name in SOURCE_FILES}
    checkpoint_candidates = [baseline_run / relative for relative in
                             ('best_checkpoint.pt', 'checkpoints/best_checkpoint.pt', 'checkpoints/best.pt')]
    checkpoint_path = next((path for path in checkpoint_candidates if path.is_file()), None)
    if checkpoint_path is None:
        raise FileNotFoundError('baseline best checkpoint is required for strict compatibility validation: ' + str(baseline_run))
    torch.set_num_threads(1)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    old = load_module(original_engine / 'model_benchmark_all.py', 'structure_original_models')
    new = load_module(patched_engine / 'model_benchmark_all.py', 'structure_patched_models')
    raw_len = int(resolved['raw_signal_len'])
    construct = dict(num_classes=6, seq_len=6, raw_signal_len=raw_len, **options)
    torch.manual_seed(42)
    old_full = old.build_model('litemamba', **construct)
    torch.manual_seed(42)
    new_full = new.build_model('litemamba', **construct)
    state_tensor_count = compare_state(torch, old_full, new_full, 'initialization')
    values = inputs(torch, raw_len)
    old_full.eval()
    new_full.eval()
    with torch.no_grad():
        compare_outputs(torch, old_full(values, return_aux=True), new_full(values, return_aux=True), 'eval forward')

    old_full.train()
    new_full.train()
    old_values, new_values = cloned_inputs(values), cloned_inputs(values)
    torch.manual_seed(20261023)
    a = old_full(old_values, return_aux=True)
    torch.manual_seed(20261023)
    b = new_full(new_values, return_aux=True)
    compare_outputs(torch, a, b, 'train forward')
    target = torch.arange(3)
    loss_a = torch.nn.functional.cross_entropy(a[0], target)
    loss_b = torch.nn.functional.cross_entropy(b[0], target)
    loss_a.backward()
    loss_b.backward()
    old_parameters, new_parameters = dict(old_full.named_parameters()), dict(new_full.named_parameters())
    for key, parameter in old_parameters.items():
        a_grad, b_grad = parameter.grad, new_parameters[key].grad
        if (a_grad is None) != (b_grad is None):
            raise AssertionError('gradient presence differs: ' + key)
        if a_grad is not None and (not torch.equal(a_grad, b_grad) or not torch.isfinite(a_grad).all()):
            raise AssertionError('gradient differs/nonfinite: ' + key)
    for index, (a_value, b_value) in enumerate(zip(old_values, new_values)):
        if (a_value.grad is None) != (b_value.grad is None):
            raise AssertionError('input gradient presence differs: ' + str(index))
        if a_value.grad is not None and not torch.equal(a_value.grad, b_value.grad):
            raise AssertionError('input gradient differs: ' + str(index))
    compare_state(torch, old_full, new_full, 'post-train-forward buffers')

    checkpoint = torch_load(torch, checkpoint_path)
    if checkpoint.get('resolved_config', {}).get('model_kwargs') != options:
        raise ValueError('checkpoint model options differ from the baseline resolved config')
    state = checkpoint['model_state_dict']
    old_full.load_state_dict(state, strict=True)
    new_full.load_state_dict(state, strict=True)
    compare_state(torch, old_full, new_full, 'loaded baseline checkpoint')
    old_full.eval()
    new_full.eval()
    with torch.no_grad():
        compare_outputs(torch, old_full(values, return_aux=True), new_full(values, return_aux=True), 'baseline checkpoint forward')

    global_options = dict(options, readout='global_attention_only')
    gated_options = dict(options, litemamba_variant='legacy_gatedconv')
    torch.manual_seed(42)
    global_model = new.build_model('litemamba', num_classes=6, seq_len=6, raw_signal_len=raw_len, **global_options)
    torch.manual_seed(42)
    gated_model = new.build_model('litemamba', num_classes=6, seq_len=6, raw_signal_len=raw_len, **gated_options)
    dim = int(options['model_dim'])
    generator = torch.Generator().manual_seed(20261024)
    tokens = torch.randn(3, 6, dim, generator=generator, requires_grad=True)
    embedding, attention = global_model.readout(tokens)
    expected_pool = (attention.unsqueeze(-1) * tokens).sum(dim=1)
    if not torch.equal(embedding[:, :dim], expected_pool) or torch.count_nonzero(embedding[:, dim:]).item() != 0:
        raise AssertionError('global-only readout must equal [attention_pool, zeros]')
    global_model.readout.target_index = 5
    alternative, _ = global_model.readout(tokens)
    global_model.readout.target_index = 3
    if not torch.equal(embedding, alternative):
        raise AssertionError('global-only direct branch depends on the selected target position')
    if embedding.shape != (3, 2 * dim):
        raise AssertionError('global-only classifier width changed')
    embedding[:, dim:].sum().backward(retain_graph=True)
    if tokens.grad is None or torch.count_nonzero(tokens.grad).item() != 0:
        raise AssertionError('zero branch has a token gradient')
    global_model.zero_grad(set_to_none=True)
    global_model.train()
    global_logits, _, _ = global_model(values, return_aux=True)
    torch.nn.functional.cross_entropy(global_logits, target).backward()
    head_grad = global_model.cls[0].weight.grad
    if head_grad is None or torch.count_nonzero(head_grad[:, dim:]).item() != 0:
        raise AssertionError('inactive classifier half must have zero data gradient')
    if not any(isinstance(module, new.SelectiveStateSpaceMixer) for module in global_model.modules()):
        raise AssertionError('global-only task lost the selective SSM')
    if any(isinstance(module, new.SelectiveStateSpaceMixer) for module in gated_model.modules()):
        raise AssertionError('gated-convolution control unexpectedly contains a selective SSM')
    with torch.no_grad():
        gated_logits, gated_embedding, gated_aux = gated_model(values, return_aux=True)
    if gated_logits.shape != (3, 6) or gated_embedding.shape != (3, 2 * dim) or not torch.isfinite(gated_logits).all():
        raise AssertionError('gated control forward is invalid')
    checks = {key: True for key in [
        'original_source_matches_baseline', 'same_seed_initial_state_dict_exact',
        'eval_forward_exact', 'train_forward_exact_with_fixed_dropout_seed',
        'parameter_and_input_gradients_exact', 'post_forward_buffers_exact',
        'baseline_checkpoint_strict_load', 'baseline_checkpoint_forward_exact',
        'global_only_pool_and_zero_branch', 'global_only_direct_target_index_invariant',
        'zero_branch_has_zero_data_gradient', 'inactive_classifier_columns_zero_data_gradient',
        'gated_control_has_no_ssm', 'global_control_retains_ssm', 'control_output_shapes_finite',
    ]}
    return {
        'status': 'passed', 'version': '1.0.0', 'created_at': datetime.now(timezone.utc).isoformat(),
        'scope': 'CPU float32 synthetic inputs only; no train/val/test data or model evaluation metrics read',
        'torch': torch.__version__, 'cpu_threads': torch.get_num_threads(), 'seed': 42,
        'comparison_rule': 'torch.equal for every compared tensor; strict=True checkpoint loading',
        'original_engine': str(original_engine), 'patched_engine': str(patched_engine),
        'baseline_run': str(baseline_run), 'baseline_checkpoint': str(checkpoint_path),
        'baseline_checkpoint_sha256': sha256(checkpoint_path),
        'baseline_checkpoint_epoch': checkpoint.get('epoch'),
        'original_source_sha256': original_sha, 'patched_source_sha256': patched_sha,
        'state_tensor_count': state_tensor_count, 'checks': checks,
        'parameter_counts': {'full': parameter_counts(new_full), 'gatedconv_target': parameter_counts(gated_model),
                             'ssm_global_only': parameter_counts(global_model)},
        'control_model_configs': {'gatedconv_target': gated_model.model_config, 'ssm_global_only': global_model.model_config},
        'notes': [
            'Real baseline weights are loaded only into equivalent full models to validate compatibility. New controls are trained from scratch by the queue.',
            'The fourth token still participates in contextual encoding and global attention; only its direct readout branch is zeroed.',
            'Global-only keeps nominal classifier width and parameter count, but its inactive input columns have zero data gradient; effective capacity is not identical.',
            'The gated-convolution control has different parameter count and operations; it is not an exact parameter-matched causal intervention.',
            'CPU equivalence checks do not establish bitwise reproducibility of all GPU kernels or cross-platform training.',
        ],
    }


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--original_engine', required=True, type=Path)
    parser.add_argument('--baseline_run', required=True, type=Path)
    parser.add_argument('--output_json', required=True, type=Path)
    args = parser.parse_args(argv)
    try:
        result = run_probe(args.original_engine, args.baseline_run)
    except BaseException as error:
        failure = {'status': 'failed', 'created_at': datetime.now(timezone.utc).isoformat(),
                   'error_type': type(error).__name__, 'error': str(error)}
        try:
            save_json(args.output_json, failure)
        except OSError:
            pass
        raise
    save_json(args.output_json, result)
    print(json.dumps({'status': result['status'], 'output_json': str(args.output_json.resolve()),
                      'parameter_counts': result['parameter_counts']}, ensure_ascii=False), flush=True)
    return 0


if __name__ == '__main__':
    try:
        raise SystemExit(main())
    except Exception as error:
        print('ERROR: ' + type(error).__name__ + ': ' + str(error), file=sys.stderr, flush=True)
        raise SystemExit(1)
