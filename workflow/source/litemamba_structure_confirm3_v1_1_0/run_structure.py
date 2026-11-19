#!/usr/bin/env python3
"""补两项结构对照的2024/2025，复用五项旧结果，仅训练及验证。"""
from __future__ import annotations

import argparse
from datetime import datetime, timezone
import fcntl
import json
import math
import os
from pathlib import Path
import shlex
import signal
import subprocess
import sys
import time

import diagnostic_common as common
import legacy_summary as legacy

VERSION = '1.1.0'
SEEDS = (42, 2024, 2025)
NEW_SEEDS = (2024, 2025)
HERE = Path(__file__).resolve().parent
PATCHED_ENGINE = HERE / 'engine'
EXPERIMENTS = (
    ('gatedconv_target', 'legacy_gatedconv', 'target_attention'),
    ('ssm_global_only', 'selective_ssm', 'global_attention_only'),
)
INTENTIONAL_SOURCE_CHANGES = {'model_benchmark_all.py', 'train_all.py', 'run_all_4_models.py'}
BASELINE_METADATA = ('queue_requested_config.json', 'resolved_config.json', 'model_config.json',
                     'environment.json', 'data_audit.json', 'calibration.json',
                     'epoch_metrics.jsonl', 'val/analysis/metrics.json')
LAUNCHER_FILES = ('run_structure.py', 'run_structure.sh', 'summarize_structure.py',
                  'diagnostic_common.py', 'legacy_summary.py', 'engine/equivalence_probe.py',
                  'ENGINE_SOURCE_MANIFEST.json')


def find_best_checkpoint(run):
    for relative in ('best_checkpoint.pt', 'checkpoints/best_checkpoint.pt', 'checkpoints/best.pt'):
        path = Path(run) / relative
        if path.is_file():
            return path
    raise FileNotFoundError('缺少原full的best checkpoint，等价性检查需要此文件：' + str(run))


def validate_patched_sources(original_sha):
    manifest = common.load_json(HERE / 'ENGINE_SOURCE_MANIFEST.json')
    if manifest['original_source_sha256'] != original_sha:
        raise ValueError('原引擎与本包基于的源码版本不同，拒绝混用')
    actual = {name: common.sha256(PATCHED_ENGINE / name) for name in common.SOURCE_FILES}
    if actual != manifest['patched_source_sha256']:
        raise ValueError('本包engine源码指纹不同，请使用完整未修改的压缩包')
    changed = {name for name in actual if actual[name] != original_sha[name]}
    if changed - INTENTIONAL_SOURCE_CHANGES:
        raise ValueError('发现非预期的引擎修改：' + ','.join(sorted(changed)))
    return actual


def load_full_configuration(run, seed):
    """读取不同seed的旧full；不改写原记录，也不读取其test指标。"""
    requested = common.load_json(Path(run) / 'queue_requested_config.json')
    resolved = common.load_json(Path(run) / 'resolved_config.json')
    for config in (requested, resolved):
        if (config.get('model_type'), config.get('seed'), config.get('feature_scheme')) != ('litemamba', seed, 'full'):
            raise ValueError('full模型、seed或特征方案不同：' + str(run))
        if config.get('monitor') != 'val_loss' or config.get('external_manifest'):
            raise ValueError('full必须按val_loss选模且不能有external_manifest：' + str(run))
    for key in common.SCALAR_KEYS + common.BOOL_KEYS:
        if key not in requested:
            raise ValueError('full请求配置缺少：' + key)
        if key in resolved and resolved[key] != requested[key]:
            raise ValueError('full请求和resolved配置不同：' + key)
    for config in (resolved, resolved.get('model_kwargs', {})):
        for key, expected in common.FIXED_ENGINE_DEFAULTS.items():
            if key in config and config[key] != expected:
                raise ValueError('full使用了不同的固定模型参数：' + key)
    if requested['no_embeddings'] or requested['no_epoch_logits'] or not resolved.get('cache_dir'):
        raise ValueError('full必须保存embedding和epoch logits并记录共享cache')
    if legacy.fingerprint_content(requested['input_fingerprints']) != legacy.fingerprint_content(resolved['input_fingerprints']):
        raise ValueError('full请求与resolved输入指纹不符')
    return requested, resolved


def validate_baseline(run, seed=42):
    requested, resolved = load_full_configuration(run, seed)
    if (requested.get('litemamba_variant'),requested.get('readout')) != ('selective_ssm','target_attention'):
        raise ValueError('full基线必须是selective_ssm + target_attention')
    if requested.get('staged_exports') is not True:
        raise ValueError('full基线必须与当前相同，使用staged_exports')
    for name in BASELINE_METADATA:
        if not (Path(run) / name).is_file():
            raise FileNotFoundError(Path(run) / name)
    model = common.load_json(Path(run) / 'model_config.json')
    if (model.get('model_type'),model.get('litemamba_variant'),model.get('readout')) != ('litemamba','selective_ssm','target_attention'):
        raise ValueError('full model_config的模型结构与预设不符')
    if model.get('target_index') != 3 or model.get('class_names') != list(legacy.CLASS_NAMES):
        raise ValueError('full model_config必须保留第四位索引3及原六类顺序')
    history, selected = legacy.read_history(Path(run), 'val_loss')
    metrics = common.load_json(Path(run) / 'val/analysis/metrics.json')
    if metrics.get('class_definition') != {str(i):name for i,name in enumerate(legacy.CLASS_NAMES)}:
        raise ValueError('full验证指标的类别定义不同')
    raw = metrics['overall']['raw']
    computed, classes = legacy.metrics_from_confusion(raw['confusion_matrix'])
    if any(c['support'] <= 0 for c in classes):
        raise ValueError('full验证集缺少某一类别')
    for key in ('macro_f1', 'accuracy', 'balanced_accuracy'):
        value = float(raw[key])
        if not math.isfinite(value) or abs(value - computed[key]) > 1e-9:
            raise ValueError('full验证指标与混淆矩阵不同：' + key)
    if raw['n'] != computed['n'] or abs(float(selected['val']['macro_f1']) - computed['macro_f1']) > 1e-9:
        raise ValueError('full验证指标与val_loss选中epoch不符')
    checkpoint = find_best_checkpoint(run)
    checkpoint_sha = common.sha256(checkpoint)
    calibration = common.load_json(Path(run) / 'calibration.json')
    if calibration.get('checkpoint_sha256') != checkpoint_sha:
        raise ValueError('full best checkpoint与已有校准记录的SHA不同')
    hashes = {name: common.sha256(Path(run) / name) for name in BASELINE_METADATA}
    return requested, resolved, checkpoint, checkpoint_sha, hashes


def same_scientific_configuration(config, reference, name, structural=False):
    """种子允许不同，除此以外只允许预先指定的结构差异。"""
    for key in legacy.CONFIG_KEYS:
        if structural and key in ('litemamba_variant', 'readout'):
            continue
        if key not in config or key not in reference or config[key] != reference[key]:
            raise ValueError('科学配置与full基线不同：' + name + '/' + key)
    if 'input_fingerprints' in config and legacy.fingerprint_content(config['input_fingerprints']) != legacy.fingerprint_content(reference['input_fingerprints']):
        raise ValueError('输入内容与full基线不同：' + name)


def frozen_run(run, seed, experiment, checkpoint, checkpoint_sha, hashes):
    return dict(experiment=experiment, seed=seed, run_dir=str(run), checkpoint=str(checkpoint),
                checkpoint_sha256=checkpoint_sha, file_sha256=hashes)


def validation_support(run):
    raw = common.load_json(Path(run)/'val/analysis/metrics.json')['overall']['raw']
    _, classes = legacy.metrics_from_confusion(raw['confusion_matrix'])
    return tuple(row['support'] for row in classes)


def validate_previous_structure(args, full_requested, original_sha, patched_sha, baseline_record):
    """对上轮计划、完成汇总、两项seed42及实际best权重做交叉核对。"""
    previous = args.previous_structure_dir
    relatives = ('structure_plan.json', 'equivalence_report.json', 'structure_summary/structure_comparison.json')
    documents = {name: common.load_json(previous / name) for name in relatives}
    old_plan = documents['structure_plan.json']
    report = documents['equivalence_report.json']
    comparison = documents['structure_summary/structure_comparison.json']
    manifest = common.load_json(HERE / 'ENGINE_SOURCE_MANIFEST.json')
    prior_sha = manifest.get('prior_patched_source_sha256', patched_sha)
    if old_plan.get('version') != '1.0.0' or old_plan.get('seed') != 42 or old_plan.get('validation_only') is not True:
        raise ValueError('上轮必须为已完成的v1.0.0固定seed42结构诊断')
    if old_plan.get('source_sha256') != original_sha or old_plan.get('patched_source_sha256') != prior_sha:
        raise ValueError('上轮结构引擎SHA与本包兼容历史不同')
    if report.get('status') != 'passed' or report.get('original_source_sha256') != original_sha or report.get('patched_source_sha256') != prior_sha:
        raise ValueError('上轮结构等价报告未通过或源码不符')
    if report.get('baseline_checkpoint_sha256') != baseline_record['checkpoint_sha256']:
        raise ValueError('上轮等价检查使用的full/42与当前基线不同')
    if old_plan.get('baseline_checkpoint_sha256') != baseline_record['checkpoint_sha256']:
        raise ValueError('上轮计划与当前full/42 checkpoint不同')
    if Path(old_plan['baseline_run']).resolve() != Path(baseline_record['run_dir']).resolve():
        raise ValueError('上轮full/42运行目录与当前不同')
    for relative, digest in old_plan.get('baseline_file_sha256', {}).items():
        if baseline_record['file_sha256'].get(relative) != digest:
            raise ValueError('上轮冻结full/42元数据与当前不同：' + relative)
    same_scientific_configuration(old_plan['scientific_configuration'], full_requested, 'previous_structure_plan')
    if legacy.fingerprint_content(old_plan['expected_input_fingerprints']) != legacy.fingerprint_content(full_requested['input_fingerprints']):
        raise ValueError('上轮结构输入内容不同')
    if comparison.get('status') != 'complete' or comparison.get('test_inference') is not False or comparison.get('fixed_seed') != 42:
        raise ValueError('上轮结构汇总必须完整且没有新test推理')
    if comparison.get('original_source_sha256') != original_sha or comparison.get('patched_source_sha256') != prior_sha:
        raise ValueError('上轮comparison源码与计划不符')
    tasks = old_plan.get('tasks', [])
    if len(tasks) != 2 or {(t.get('experiment'),t.get('seed')) for t in tasks} != {(e[0],42) for e in EXPERIMENTS}:
        raise ValueError('上轮结构计划任务身份不符')
    records = []
    for name, variant, readout in EXPERIMENTS:
        task = next(t for t in tasks if t['experiment'] == name)
        run = previous / 'full' / name / 'seed_42'
        if Path(task['run_dir']).resolve() != run.resolve():
            raise ValueError('上轮任务目录与指定previous_structure_dir不符：' + name)
        if (task.get('litemamba_variant'),task.get('readout')) != (variant,readout):
            raise ValueError('上轮结构任务定义不符：' + name)
        requested = common.load_json(run / 'queue_requested_config.json')
        resolved = common.load_json(run / 'resolved_config.json')
        result = common.load_json(run / 'results.json')
        for config in (requested, resolved, result):
            if (config.get('model_type'),config.get('seed'),config.get('feature_scheme')) != ('litemamba',42,'full'):
                raise ValueError('上轮结构模型/seed/输入身份不同：' + name)
            if (config.get('litemamba_variant'),config.get('readout')) != (variant,readout):
                raise ValueError('上轮结构身份不同：' + name)
        same_scientific_configuration(requested,full_requested,name,structural=True)
        if requested.get('structure_experiment') != name or requested.get('source_sha256') != prior_sha:
            raise ValueError('上轮结构队列身份或源码不同：' + name)
        for key in legacy.CONFIG_KEYS:
            if key != 'staged_exports' and resolved.get(key) != requested[key]:
                raise ValueError('上轮结构resolved科学配置不同：' + name + '/' + key)
        if requested.get('external_manifest') or resolved.get('external_manifest'):
            raise ValueError('上轮结构不能有external_manifest')
        if legacy.fingerprint_content(resolved['input_fingerprints']) != legacy.fingerprint_content(full_requested['input_fingerprints']):
            raise ValueError('上轮结构resolved输入不同：' + name)
        model = common.load_json(run/'model_config.json')
        full_model = common.load_json(Path(baseline_record['run_dir'])/'model_config.json')
        if (model.get('model_type'),model.get('litemamba_variant'),model.get('readout')) != ('litemamba',variant,readout):
            raise ValueError('上轮结构model_config身份不符：'+name)
        for key in ('num_classes','seq_len','raw_signal_len','model_dim','target_index','dropout','classifier_dropout','feature_scheme','num_blocks','state_dim','class_names'):
            if model.get(key) != full_model.get(key):
                raise ValueError('上轮结构额外模型参数不同：'+name+'/'+key)
        if validation_support(run) != validation_support(baseline_record['run_dir']):
            raise ValueError('上轮结构验证集各类样本数不同：'+name)
        if result.get('status') != 'train_complete' or result.get('test_evaluated') is not False:
            raise ValueError('上轮结构尚未完成train/val或已经评价test：' + name)
        for relative in ('test_started.json', 'test_evaluated.json'):
            if (run / relative).exists():
                raise ValueError('上轮结构发现test标记：' + name)
        if (run/'test').is_dir() and any(p.is_file() for p in (run/'test').rglob('*')):
            raise ValueError('上轮结构发现test产物：' + name)
        for key in ('metrics_by_split','exports_by_split'):
            if set(result.get(key,{})) != {'train','val'}:
                raise ValueError('上轮结构train/val导出缺失或包含其他split：' + name)
        checkpoint = find_best_checkpoint(run)
        checkpoint_sha = common.sha256(checkpoint)
        calibration = common.load_json(run/'calibration.json')
        if calibration.get('checkpoint_sha256') != checkpoint_sha or calibration.get('fitted_on') != 'val':
            raise ValueError('上轮结构校准checkpoint或拟合split不符：' + name)
        provenance = result.get('provenance',{})
        if provenance.get('checkpoint_sha256') != checkpoint_sha or legacy.fingerprint_content(provenance['input_fingerprints']) != legacy.fingerprint_content(full_requested['input_fingerprints']):
            raise ValueError('上轮结构results来源不符：' + name)
        for snapshot in (resolved.get('source_snapshot',{}),provenance.get('source_snapshot',{})):
            if any(snapshot.get('files_sha256',{}).get(k) != v for k,v in prior_sha.items()):
                raise ValueError('上轮结构源码快照不符：' + name)
        if any(result['exports_by_split'][split].get('checkpoint_sha256') != checkpoint_sha for split in ('train','val')):
            raise ValueError('上轮结构导出checkpoint不符：' + name)
        _, selected = legacy.read_history(run,'val_loss')
        metrics = common.load_json(run/'val/analysis/metrics.json')
        raw = metrics['overall']['raw']
        computed, classes = legacy.metrics_from_confusion(raw['confusion_matrix'])
        if metrics.get('class_definition') != {str(i):base for i,base in enumerate(legacy.CLASS_NAMES)}:
            raise ValueError('上轮结构类别定义不符')
        if result.get('best_epoch') != selected['epoch'] or abs(selected['val']['macro_f1']-computed['macro_f1'])>1e-9:
            raise ValueError('上轮结构选模与导出不同：' + name)
        if raw.get('n') != computed['n'] or any(c['support']<=0 for c in classes):
            raise ValueError('上轮结构验证样本数量或类别缺失')
        for key in ('macro_f1','accuracy','balanced_accuracy'):
            if not math.isfinite(float(raw[key])) or abs(float(raw[key])-computed[key])>1e-9:
                raise ValueError('上轮结构raw指标不符：' + name+'/'+key)
        row = next((r for r in comparison.get('per_run',[]) if r.get('experiment')==name and r.get('seed')==42),None)
        if row is None or row.get('checkpoint_sha256')!=checkpoint_sha or abs(row['val_macro_f1']-computed['macro_f1'])>1e-9:
            raise ValueError('上轮结构汇总与原运行不同：' + name)
        hashes = {relative:common.sha256(run/relative) for relative in BASELINE_METADATA+('results.json',)}
        for evidence in comparison.get('source_files',[]):
            prefix = 'source_metadata/'+name+'/'
            relative = evidence.get('package_path','')
            if relative.startswith(prefix) and relative[len(prefix):] in hashes and hashes[relative[len(prefix):]] != evidence['sha256']:
                raise ValueError('上轮归档元数据与当前运行不同：' + name+'/'+relative)
        records.append(frozen_run(run,42,name,checkpoint,checkpoint_sha,hashes))
    frozen_files = {name:common.sha256(previous/name) for name in relatives}
    return records, prior_sha, frozen_files


def resolve_paths(args):
    args.engine_dir = Path(args.engine_dir).expanduser().resolve()
    args.data_dir = Path(args.data_dir).expanduser().resolve()
    results = args.data_dir / 'sixclass_auto_v1_0_4_results'
    args.baseline_root = Path(args.baseline_root).expanduser().resolve() if args.baseline_root else results / 'main_20/full/litemamba'
    args.baseline_run = args.baseline_root / 'seed_42'
    args.previous_structure_dir = Path(args.previous_structure_dir).expanduser().resolve() if args.previous_structure_dir else results / 'litemamba_structure_seed42_v1_0_0'
    args.output_dir = Path(args.output_dir).expanduser().resolve() if args.output_dir else results / 'litemamba_structure_confirm3_v1_1_0'
    for protected in (args.data_dir, args.engine_dir, args.baseline_root, args.previous_structure_dir, HERE):
        if args.output_dir == protected or args.output_dir in protected.parents:
            raise ValueError('输出目录不能覆盖数据、引擎、基线或代码包：' + str(args.output_dir))
    for protected in (args.engine_dir, args.baseline_root, args.previous_structure_dir, HERE, args.baseline_run.parents[2]):
        if protected in args.output_dir.parents:
            raise ValueError('输出目录不能位于旧运行或代码包内部：' + str(args.output_dir))


def build_command(args, requested, resolved, cap):
    command = common.build_command(args, requested, resolved, cap)
    command[2] = str(PATCHED_ENGINE / 'run_all_4_models.py')
    index = command.index('--models')
    command[index + 1:index + 2] = [x[0] for x in EXPERIMENTS]
    begin = command.index('--seeds')
    end = command.index('--feature_schemes', begin)
    command[begin + 1:end] = [str(seed) for seed in NEW_SEEDS]
    begin = command.index('--feature_schemes')
    end = command.index('--phase', begin)
    command[begin + 1:end] = ['full']
    command.append('--structure_diagnostic')
    return command


def build_plan(args):
    resolve_paths(args)
    requested, resolved, checkpoint, checkpoint_sha, hashes = validate_baseline(args.baseline_run,42)
    original_sha = common.validate_sources(args.engine_dir, requested)
    patched_sha = validate_patched_sources(original_sha)
    baselines = [frozen_run(args.baseline_run,42,'full',checkpoint,checkpoint_sha,hashes)]
    baseline_source = requested['source_sha256']
    for seed in NEW_SEEDS:
        run = args.baseline_root / ('seed_'+str(seed))
        candidate, candidate_resolved, weights, weights_sha, metadata = validate_baseline(run,seed)
        same_scientific_configuration(candidate,requested,'full/seed_'+str(seed))
        if validation_support(run) != validation_support(args.baseline_run):
            raise ValueError('不同seed的full验证集各类样本数不同：'+str(seed))
        if candidate.get('source_sha256') != baseline_source:
            raise ValueError('不同seed的full源码SHA不同：' + str(seed))
        for key in ('num_classes','seq_len','raw_signal_len','model_dim','target_index','dropout','classifier_dropout','feature_scheme','num_blocks','state_dim'):
            if common.load_json(run/'model_config.json').get(key) != common.load_json(args.baseline_run/'model_config.json').get(key):
                raise ValueError('不同seed的full模型参数不同：'+str(seed)+'/'+key)
        baselines.append(frozen_run(run,seed,'full',weights,weights_sha,metadata))
    reused_controls,prior_sha,previous_files = validate_previous_structure(args,requested,original_sha,patched_sha,baselines[0])
    verification = common.validate_data(args.data_dir, requested['input_fingerprints'], not (args.dry_run or args.summary_only))
    cap, evidence = common.capacity_limit(args.baseline_run, args.max_parallel)
    cap = min(2, cap)
    command = build_command(args, requested, resolved, cap)
    tasks = [dict(experiment=name, model_type='litemamba', seed=seed, feature_scheme='full',
                  litemamba_variant=variant, readout=readout, run_id='full/' + name + '/seed_'+str(seed),
                  run_dir=str(args.output_dir / 'full' / name / ('seed_'+str(seed)))) for seed in NEW_SEEDS for name, variant, readout in EXPERIMENTS]
    scientific = {key: requested[key] for key in common.SCALAR_KEYS + common.BOOL_KEYS}
    scientific.update(model_type='litemamba', seeds=list(SEEDS), feature_scheme='full', staged_exports=True)
    return dict(version=VERSION, purpose='uniform three-seed confirmation of both prespecified structural controls',
                validation_only=True, seeds=list(SEEDS),new_seeds=list(NEW_SEEDS), tasks=tasks, baseline_run=str(args.baseline_run),
                baseline_root=str(args.baseline_root),baselines=baselines,reused_controls=reused_controls,
                previous_structure_dir=str(args.previous_structure_dir), previous_structure_file_sha256=previous_files,
                baseline_checkpoint=str(checkpoint), baseline_checkpoint_sha256=checkpoint_sha,
                baseline_file_sha256=hashes, engine_dir=str(args.engine_dir), patched_engine_dir=str(PATCHED_ENGINE),
                data_dir=str(args.data_dir), output_dir=str(args.output_dir), cache_dir=str(Path(resolved['cache_dir']).resolve()),
                source_sha256=original_sha, prior_patched_source_sha256=prior_sha, patched_source_sha256=patched_sha,
                launcher_source_sha256={name:common.sha256(HERE / name) for name in LAUNCHER_FILES},
                scientific_configuration=scientific, expected_input_fingerprints=requested['input_fingerprints'],
                data_verification=verification, checkpoint_selection='val_loss',
                initialization='each new candidate trained from scratch with seed2024/2025; baseline weights used only for equivalence checks',
                phase='train', staged_exports=True, no_isolated_benchmark=True,
                concurrency=dict(requested=args.max_parallel,effective_hard_cap=cap,evidence=evidence,
                                 admission='one shared ResourceController across all four new candidates'),
                disk_policy=dict(startup_free_gib=args.startup_free_gib,disk_reserve_gib=args.disk_reserve_gib),
                intentional_differences=dict(gatedconv_target='legacy gated convolution replaces selective SSM; parameter/computation differences reported',
                                            ssm_global_only='zero fourth-position direct branch; retain pooled attention and nominal classifier width'),
                argv=command, command=shlex.join(command), dry_run=args.dry_run,created_at=common.now())


def plan_identity(plan):
    keys = ('version','seeds','new_seeds','tasks','baselines','reused_controls','baseline_root','previous_structure_dir','previous_structure_file_sha256',
            'baseline_run','baseline_checkpoint','baseline_checkpoint_sha256',
            'baseline_file_sha256','engine_dir','patched_engine_dir','data_dir','output_dir','cache_dir',
            'source_sha256','prior_patched_source_sha256','patched_source_sha256','launcher_source_sha256','scientific_configuration',
            'expected_input_fingerprints','initialization','intentional_differences')
    return {key:plan[key] for key in keys}


def verify_baseline_unchanged(plan):
    for record in plan['baselines']+plan['reused_controls']:
        run = Path(record['run_dir'])
        for name,digest in record['file_sha256'].items():
            if common.sha256(run/name) != digest:
                raise RuntimeError('运行期间复用元数据改变：'+str(run/name))
        if common.sha256(record['checkpoint']) != record['checkpoint_sha256']:
            raise RuntimeError('运行期间复用checkpoint改变：'+str(run))
    for name,digest in plan['previous_structure_file_sha256'].items():
        if common.sha256(Path(plan['previous_structure_dir'])/name) != digest:
            raise RuntimeError('运行期间上轮结构归档改变：'+name)


def equivalence_command(plan, output_json):
    return [sys.executable, '-u', str(PATCHED_ENGINE / 'equivalence_probe.py'),
            '--original_engine', plan['engine_dir'], '--baseline_run', plan['baseline_run'],
            '--output_json', str(output_json)]


def run_equivalence(plan, logpath, report_path):
    """在服务器当前PyTorch环境用CPU核验full默认路径和旧checkpoint。"""
    with Path(logpath).open('a',encoding='utf-8',buffering=1) as log:
        completed = subprocess.run(equivalence_command(plan, report_path), stdout=log, stderr=subprocess.STDOUT,
                                   cwd=str(PATCHED_ENGINE), timeout=180)
    if completed.returncode:
        raise RuntimeError('full等价性预检失败，未开始训练；查看日志：' + str(logpath))
    report = common.load_json(report_path)
    if report.get('status') != 'passed':
        raise RuntimeError('full等价性预检没有passed状态，未开始训练')
    if report.get('baseline_checkpoint_sha256') != plan['baseline_checkpoint_sha256']:
        raise RuntimeError('等价性报告的baseline checkpoint指纹不同')
    if report.get('original_source_sha256') != plan['source_sha256'] or report.get('patched_source_sha256') != plan['patched_source_sha256']:
        raise RuntimeError('等价性报告的源码指纹不同')
    return report


def check_free_space(plan, threshold):
    free = common.disk_free_gib(common.disk_paths(plan))
    if min(free.values()) < threshold:
        raise RuntimeError('输出/cache空闲空间不足：' + json.dumps(free) + ' GiB，需要至少' + str(threshold))
    return free


def parse_args(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--engine_dir',default=common.DEFAULT_ENGINE,help='原始未修改的v1.0.4引擎')
    parser.add_argument('--data_dir',default=common.DEFAULT_DATA)
    parser.add_argument('--baseline_root',help='含seed_42、seed_2024、seed_2025的旧full/litemamba目录')
    parser.add_argument('--previous_structure_dir',help='已完成的litemamba_structure_seed42_v1_0_0结果目录')
    parser.add_argument('--output_dir')
    parser.add_argument('--max_parallel',type=int,choices=(1,2),default=2)
    parser.add_argument('--device',default='auto')
    parser.add_argument('--workers',type=int)
    parser.add_argument('--cpu_threads',type=int)
    parser.add_argument('--resume',action='store_true')
    parser.add_argument('--dry_run',action='store_true')
    parser.add_argument('--summary_only',action='store_true')
    parser.add_argument('--startup_free_gib',type=float,default=30.0)
    parser.add_argument('--disk_reserve_gib',type=float,default=8.0)
    args = parser.parse_args(argv)
    if args.dry_run and args.summary_only:
        parser.error('dry_run与summary_only不能同时使用')
    if not all(math.isfinite(v) for v in (args.startup_free_gib,args.disk_reserve_gib)) or args.disk_reserve_gib < 0 or args.startup_free_gib < args.disk_reserve_gib:
        parser.error('空间阈值应为有限非负值，启动阈值不小于保留值')
    if (args.workers is not None and args.workers < 0) or (args.cpu_threads is not None and args.cpu_threads < 1):
        parser.error('workers>=0，cpu_threads>=1')
    return args


def main(argv=None):
    args = parse_args(argv)
    resolve_paths(args)
    output = args.output_dir
    output.mkdir(parents=True,exist_ok=True)
    with (output / 'structure_wrapper.lock').open('a+') as lock:
        try:
            fcntl.flock(lock,fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError:
            raise RuntimeError('此结构诊断目录已有程序运行，请勿重复提交')
        lock.seek(0)
        lock.truncate()
        lock.write(json.dumps(dict(pid=os.getpid(),at=common.now())))
        lock.flush()
        plan = build_plan(args)
        plan_path = output / 'structure_plan.json'
        previous = common.load_json(plan_path) if plan_path.is_file() else None
        if previous is not None and plan_identity(previous) != plan_identity(plan):
            raise RuntimeError('冻结计划的科学配置、源码、数据或基线不同，拒绝覆盖')
        if (output / 'queue_manifest.json').exists() and not (args.resume or args.dry_run or args.summary_only):
            raise RuntimeError('已有队列，请保持相同参数并加--resume继续')
        if args.summary_only and previous is None:
            raise RuntimeError('没有已有正式计划，不能summary_only')
        logdir = output / 'logs'
        logdir.mkdir(exist_ok=True)
        stamp = datetime.now(timezone.utc).strftime('%Y%m%d_%H%M%S_%f') + '_' + str(os.getpid())
        logpath = logdir / ('structure_' + stamp + '.log')
        with logpath.open('x',encoding='utf-8',buffering=1) as log:
            temporary = logdir / ('latest_' + str(os.getpid()) + '.tmp')
            temporary.symlink_to(logpath.name)
            temporary.replace(logdir / 'structure_latest.log')
            print('[PLAN] ' + str(plan_path),flush=True)
            print('[REUSE] full的42/2024/2025，加两项结构对照seed42；五个原运行目录只读',flush=True)
            print('[NEW] gatedconv_target、ssm_global_only各2024/2025，共4项from scratch',flush=True)
            print('[LOG] ' + str(logpath),flush=True)
            print('[FOLLOW] tail -F ' + shlex.quote(str(logdir / 'structure_latest.log')),flush=True)
            log.write(json.dumps(dict(at=common.now(),argv=plan['argv'],dry_run=args.dry_run),ensure_ascii=False)+'\n')
            if args.dry_run:
                common.save_json(output / ('structure_plan_dry_run_' + stamp + '.json'),plan)
                print('[DRY RUN] 核对元数据、旧checkpoint及源码指纹；未读取六个大输入、未执行PyTorch等价性检查或训练。',flush=True)
                print('[COMMAND] ' + plan['command'],flush=True)
                return 0
            if previous is None:
                common.save_json(plan_path,plan)
            else:
                common.save_json(logdir / ('invocation_' + stamp + '.json'),plan)
            check_free_space(plan,args.disk_reserve_gib if args.summary_only else args.startup_free_gib)
            status_path = output / 'structure_status.json'
            status = dict(status='running',started_at=common.now(),log=str(logpath),validation_only=True,
                          reused_count=5,new_count=4,summary_only=args.summary_only)
            common.save_json(status_path,status)
            process = None
            try:
                if not args.summary_only:
                    report_path = logdir / ('equivalence_' + stamp + '.json')
                    print('[EQUIVALENCE] CPU检查原/新full的初始化、forward、梯度和旧checkpoint严格加载',flush=True)
                    report = run_equivalence(plan,logpath,report_path)
                    common.save_json(output / 'equivalence_report.json',report)
                    # 等价性子进程使用独立append句柄，随后共享queue句柄从最新末尾写入。
                    log.seek(0,os.SEEK_END)
                    print('[EQUIVALENCE PASS] full默认计算一致；开始共享资源队列',flush=True)
                    process = subprocess.Popen(plan['argv'],cwd=str(PATCHED_ENGINE),stdout=log,
                                               stderr=subprocess.STDOUT,start_new_session=True)
                    status['queue_pid'] = process.pid
                    common.save_json(status_path,status)
                    while process.poll() is None:
                        free = common.disk_free_gib(common.disk_paths(plan))
                        if min(free.values()) < args.disk_reserve_gib:
                            status.update(status='stopped_low_disk',free_gib=free,
                                          note='清理空间后使用原命令加--resume；保留checkpoints及全部输出')
                            common.stop_queue(process,output)
                            raise RuntimeError('低磁盘保护已停止本队列，可清理后resume')
                        time.sleep(2)
                    if process.returncode:
                        raise RuntimeError('结构队列失败code=' + str(process.returncode) + '，保留现场可--resume；日志：' + str(logpath))
                verify_baseline_unchanged(plan)
                status.update(status='training_validation_complete',training_finished_at=common.now())
                common.save_json(status_path,status)
                check_free_space(plan,args.disk_reserve_gib)
                from summarize_structure import summarize
                result = summarize(plan,output)
                status.update(status='complete',finished_at=common.now(),summary=result)
                common.save_json(status_path,status)
                print('[FULL PASS] 4项追加训练及三结构×三seed验证汇总全部完成，无新增test推理。',flush=True)
                print('[RESULT PACKAGE] ' + str(output / 'structure_confirm3_results_for_analysis_v1_1_0.tar.gz'),flush=True)
                return 0
            except BaseException as error:
                if process is not None:
                    common.stop_queue(process,output)
                if status.get('status') != 'stopped_low_disk':
                    status.update(status='interrupted' if isinstance(error,KeyboardInterrupt) else 'failed',error=str(error))
                status['stopped_at'] = common.now()
                try:
                    common.save_json(status_path,status)
                except OSError as status_error:
                    print('状态文件未能写入，保留原错误：' + str(status_error),file=sys.stderr,flush=True)
                raise


if __name__ == '__main__':
    signal.signal(signal.SIGTERM,common.handle_sigterm)
    try:
        raise SystemExit(main())
    except KeyboardInterrupt as error:
        print('已中断，保持配置不变可加--resume：' + str(error),file=sys.stderr)
        raise SystemExit(130)
    except Exception as error:
        print('ERROR: ' + type(error).__name__ + ': ' + str(error),file=sys.stderr)
        raise SystemExit(1)
