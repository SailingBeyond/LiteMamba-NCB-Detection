#!/usr/bin/env python3
"""Prepare a fixed all-nine Figure 4 single-input evaluation and reuse full outputs."""
from __future__ import annotations
import argparse
from contextlib import contextmanager, ExitStack
import copy
from datetime import datetime, timezone
import fcntl
import json
import math
from pathlib import Path

from figure4_common import (HERE, VENDOR, atomic_json, ensure_disk, identity_hash,
    read_json, sha256, validate_completion, verify_run)
from prepare_figure4_correction import _reuse_one, correction_write_lock, previous_read_lock, SETTING_KEYS
from prepare_gated_comparison import (CACHE_FILES, DECLARATIONS, bind_cache,
    cache_matches, fingerprint_content, logical_dataset, preserve_source_metadata,
    scientific_plan as prior_scientific_plan, verify_sources)
from singleinput_registry import (MODEL, RUNTIME_SHA256, SCHEMES, SEEDS,
    content, discover_singleinput_runs)

RESULTS=Path('/path/to/revision-inputs/CPLX-XNA-replenishment/CPLX-XNA-replenishment/fast5_single_guppy/sixclass_overlap_read_split_721/sixclass_auto_v1_0_4_results')
DEFAULT_PREVIOUS=RESULTS/'figure4_gated_comparison_v1_0_0'
DEFAULT_TRAINING=RESULTS/'gatedconv_singleinput9_v1_0_0'
DEFAULT_OUTPUT=RESULTS/'figure4_singleinput9_v1_0_0'
VERSION='singleinput-external-1.0.0'
CODE_FILES=('figure4_common.py','figure4_worker.py','figure4_analyze.py','run_figure4.py',
    'run_singleinput_external.py','prepare_singleinput.py','singleinput_registry.py',
    'summarize_singleinput_external.py','prepare_figure4.py','prepare_figure4_correction.py',
    'prepare_gated_comparison.py','audit_corrected_split.py',
    'vendor/IterDataset.py','vendor/model_benchmark_all.py',
    'vendor/resource_scheduler.py','vendor/cuda_device_probe.py')


def code_fingerprints():
    return {name:sha256(HERE/name) for name in CODE_FILES}


def scientific_plan(plan):
    keys=('version','task','training_results','runs','preprocessing','models','seeds',
        'feature_schemes','prediction_settings','selection_policy','comparison_provenance',
        'comparison_code_sha256','external_identity_audit','external_window_design_audit',
        'inherited_reference_correction','missing_template_roles')
    return {**{key:plan[key] for key in keys},'datasets':[logical_dataset(d) for d in plan['datasets']]}


def validate_previous_plan(plan):
    if plan.get('version')!='gated-comparison-1.0.0' or plan.get('task')!='6-class':
        raise ValueError('必须使用已完成的 figure4_gated_comparison_v1_0_0，不能用旧SSM或更正前结果替代')
    if identity_hash(prior_scientific_plan(plan))!=plan.get('plan_id'):
        raise ValueError('旧GatedConv Figure4计划内容与冻结科学配置SHA不一致')
    correction=plan.get('inherited_reference_correction',{})
    if correction.get('kind')!='reference_sequence_correction':
        raise ValueError('旧计划缺少XNA13–16参考修正的完整来源')
    expected={(f'XNA{i}',role,strand) for i in range(1,21)
        for role in (('target','control') if i<=16 else ('target',)) for strand in ('pos','neg')}
    observed=[(d.get('template'),d.get('role'),d.get('file_strand')) for d in plan.get('datasets',[])]
    ids=[d.get('dataset_id') for d in plan.get('datasets',[])]
    if len(observed)!=72 or set(observed)!=expected or len(set(ids))!=72:
        raise ValueError('Figure4必须保留全部72个模板/角色/方向文件，不允许筛选子集')
    corrected={d['dataset_id'] for d in plan['datasets'] if d.get('template') in {'XNA13','XNA14','XNA15','XNA16'} and d.get('role')=='target'}
    replacement=correction.get('replacement_map',{})
    mapped=set(replacement.values()) if isinstance(replacement,dict) else {item.get('new_dataset_id') for item in replacement}
    # Gated comparison inherits the original correction map; values contain the
    # corrected dataset IDs. The old comparison scientific hash binds all72.
    if not corrected.issubset(mapped):
        raise ValueError('修正来源没有完整覆盖XNA13–16 target')
    for d in plan['datasets']:
        if (Path(d['dataset_id']).name!=d['dataset_id'] or d.get('status')!='complete'
            or int(d.get('n_samples',0))<1 or set(d.get('outputs',{}))!=set(CACHE_FILES)
            or d.get('identity',{}).get('preprocessing')!=plan.get('preprocessing')):
            raise ValueError('旧Figure4缓存清单无效：'+str(d.get('dataset_id')))
    runs=[r for r in plan.get('runs',[]) if r.get('model')==MODEL]
    if len(runs)!=3 or {(r.get('feature_scheme'),r.get('seed')) for r in runs}!={('full',seed) for seed in SEEDS}:
        raise ValueError('旧结果必须有全部三个完整输入 GatedConv，不能混入SSM full')
    for run in runs:
        if run.get('run_id')!=f'full/{MODEL}/seed_{run["seed"]}' or run.get('actual_model_type')!='litemamba':
            raise ValueError('旧完整输入GatedConv运行身份不同')
        kwargs=run.get('model_kwargs',{})
        if any(kwargs.get(k)!=v for k,v in {'feature_scheme':'full','litemamba_variant':'legacy_gatedconv','readout':'target_attention','target_index':3}.items()):
            raise ValueError('旧模型不是最终第4位GatedConv full')
        if run.get('runtime_sha256')!=RUNTIME_SHA256:
            raise ValueError('旧full模型运行时与已确认GatedConv不同')
    if plan.get('external_window_design_audit',{}).get('status')=='known_deviation':
        raise ValueError('旧窗口审计存在已知越界，不能作为相同外部评价继承')
    return plan


def inherited_prediction_settings(previous, plan):
    settings=copy.deepcopy(plan.get('prediction_settings',{}))
    if set(settings)!=SETTING_KEYS or settings.get('workers')!=0 or settings.get('amp_policy')!='training_checkpoint_config':
        raise ValueError('旧预测设置缺失或workers/AMP规则不受支持')
    if (settings.get('embedding_dtype') not in ('float16','float32') or
        type(settings.get('batch_size')) is not int or settings['batch_size']<1 or
        type(settings.get('cpu_threads')) is not int or settings['cpu_threads']<1):
        raise ValueError('旧预测batch/导出精度配置无效')
    ids={d['dataset_id'] for d in plan['datasets']}
    for run in [r for r in plan['runs'] if r.get('model')==MODEL]:
        path=Path(previous)/'predictions'/run['run_id']/'run_complete.json'
        marker=read_json(path)
        if (marker.get('status')!='complete' or marker.get('plan_id')!=plan['plan_id'] or
            marker.get('run_id')!=run['run_id'] or marker.get('prediction_settings')!=settings or
            set(marker.get('datasets',[]))!=ids or len(marker.get('datasets',[]))!=72):
            raise ValueError('旧full未完整完成或预测设置不同：'+str(path))
    return settings


def _verify_cache_contents(dataset):
    for name in CACHE_FILES:
        expected=dataset['outputs'][name]
        path=Path(dataset['cache_dir'])/name
        if path.stat().st_size!=expected['size_bytes'] or sha256(path)!=expected['sha256']:
            raise ValueError('外部缓存不同于已冻结Figure4内容：'+str(path))


def _full_runs(plan):
    runs=copy.deepcopy([r for r in plan['runs'] if r.get('model')==MODEL])
    for r in runs:
        # Same source bytes, new self-contained runtime location. Prediction
        # outputs are reused; no full forward pass is submitted by this entry.
        r['runtime_dir']=str(VENDOR)
        verify_run(r)
    return sorted(runs,key=lambda r:SEEDS.index(r['seed']))


def _source_records(previous,training,runs):
    files=[(previous/'evaluation_plan.json','previous/evaluation_plan.json')]
    for name in ('external_identity_audit.json','external_window_design_audit.json','reuse_manifest.json','comparison_source_provenance.json'):
        if (previous/name).is_file():files.append((previous/name,'previous/'+name))
    for name in ('singleinput_plan.json','singleinput_status.json','queue_plan.json','evaluation_plan.json','queue_manifest.json'):
        files.append((training/name,'training/'+name))
    for run in runs:
        if run['feature_scheme']=='full':continue
        for name in run['frozen_metadata_sha256']:
            files.append((Path(run['run_dir'])/name,'runs/'+run['training_run_id']+'/'+name))
    return [dict(path=str(path),sha256=sha256(path),local_copy='source_provenance/'+local) for path,local in files]


@contextmanager
def training_read_locks(root):
    with ExitStack() as stack:
        for name in ('.singleinput.lock','.queue.lock'):
            path=Path(root)/name
            if not path.is_file():continue
            handle=stack.enter_context(path.open('r',encoding='utf-8'))
            try:fcntl.flock(handle.fileno(),fcntl.LOCK_SH|fcntl.LOCK_NB)
            except BlockingIOError as exc:raise RuntimeError('单输入训练队列仍在运行，请等待其退出：'+str(path)) from exc
        yield


def _boundary(previous,training,output):
    for protected in (previous,training,HERE):
        if output==protected or output in protected.parents or protected in output.parents:
            raise ValueError('新输出必须与旧Figure4、单输入训练和代码目录互不包含：'+str(output))


def _inherited_audits(previous,old,training):
    basis={'kind':'same_frozen_external_contents_and_internal_data',
        'previous_output':str(previous),'previous_plan_id':old['plan_id'],
        'external_artifacts':'All72 source and cache artifacts are SHA256 verified; rows and coordinates are unchanged.',
        'internal_artifacts':'All12 runs have identical train/val/test feature and metadata content SHA256.',
        'interpretation':'Previous automated verification limitations remain; no biological read identity or run identity is synthesized.'}
    identity=copy.deepcopy(old.get('external_identity_audit',{}))
    window=copy.deepcopy(old.get('external_window_design_audit',{}))
    if not identity or not window:raise ValueError('旧Figure4缺少read身份/窗口设计审计')
    identity['inherited_without_row_modification']=basis
    window['inherited_without_row_modification']=basis
    return identity,window


def _prepare_locked(args):
    previous=Path(args.previous_output).expanduser().resolve()
    training=Path(args.training_dir).expanduser().resolve()
    output=Path(args.output_root).expanduser().resolve()
    _boundary(previous,training,output)
    check_only=bool(getattr(args,'check_only',False))
    existing=read_json(output/'evaluation_plan.json') if (output/'evaluation_plan.json').is_file() else None
    if existing is not None and not args.resume and not check_only:
        raise ValueError('本目录已有冻结Figure4计划，请保持参数不变并加--resume')
    with previous_read_lock(previous),training_read_locks(training):
        old=validate_previous_plan(read_json(previous/'evaluation_plan.json'))
        settings=inherited_prediction_settings(previous,old)
        full=_full_runs(old)
        new,preprocessing=discover_singleinput_runs(training,old)
        if preprocessing!=old['preprocessing']:raise ValueError('单输入预处理与旧Figure4不同')
        runs=full+new
        sources=_source_records(previous,training,runs)
        provenance={'kind':'fixed_singleinput_external_evaluation','previous_output':str(previous),
            'training_dir':str(training),'previous_plan_id':old['plan_id'],
            'previous_plan_sha256':sha256(previous/'evaluation_plan.json'),
            'reuse_mode':args.reuse_mode,'source_metadata_files':sources,
            'reused_full_runs':3,'new_singleinput_runs':9,
            'evaluation_role':'Follow-up on previously observed, corrected Figure4 data; fixed9 checkpoints, no training, external calibration, threshold fitting or seed selection.'}
        code=code_fingerprints()
        if existing and (existing.get('comparison_provenance')!=provenance or existing.get('comparison_code_sha256')!=code):
            raise ValueError('冻结后的来源/预测代码改变，不能混合续跑；请使用独立新目录')
        reserve=float(args.disk_reserve_gib)
        prior={d['dataset_id']:d for d in existing['datasets']} if existing else {}
        datasets=[];cache_records=[]
        for index,old_dataset in enumerate(old['datasets'],1):
            if check_only:
                current=verify_sources(old_dataset)
                if not cache_matches(old_dataset,current):
                    raise ValueError('只读预检发现缓存需重建；正式运行将尝试按冻结内容重建：'+old_dataset['dataset_id'])
                dataset=copy.deepcopy(old_dataset);action='verified_existing_cache'
            else:
                dataset,action=bind_cache(old_dataset,preprocessing,output,reserve,prior.get(old_dataset['dataset_id']))
            _verify_cache_contents(dataset)
            if logical_dataset(dataset)!=logical_dataset(old_dataset):raise ValueError('缓存或源数据内容与旧冻结计划不同')
            datasets.append(dataset);cache_records.append(dict(dataset_id=dataset['dataset_id'],action=action,cache_dir=dataset['cache_dir']))
            print(f'[VERIFY EXTERNAL] {index}/72 '+dataset['dataset_id'],flush=True)
        identity_audit,window_audit=_inherited_audits(previous,old,training)
        if check_only:
            for run in full:
                for dataset in datasets:
                    folder=previous/'predictions'/run['run_id']/dataset['dataset_id']
                    if not validate_completion(folder,old['plan_id'],run['run_id'],dataset['dataset_id'],settings,dataset['n_samples']):
                        raise ValueError('完整输入旧预测产物缺失或SHA不一致：'+str(folder))
                    marker=read_json(folder/'completion.json')
                    if marker.get('checkpoint_sha256')!=run['checkpoint_sha256'] or marker.get('temperature')!=run['temperature']:
                        raise ValueError('旧full预测checkpoint/温度与模型不同')
            print('[CHECK PASS] 9个单输入检查点、72个冻结数据文件/缓存、3个full的216份预测完整核验；无新推理。',flush=True)
            return {'status':'check_complete','new_prediction_runs':9,'reused_full_runs':3,'datasets':72,'prediction_settings':settings}
        core=dict(version=VERSION,task='6-class',training_results=str(training),runs=runs,datasets=datasets,
            preprocessing=preprocessing,models=[MODEL],seeds=list(SEEDS),feature_schemes=['full',*SCHEMES],
            prediction_settings=settings,selection_policy='Prespecified all9 single-input checkpoints and3 full GatedConv baselines; validation-loss checkpoints and validation-only temperatures; no new training.',
            comparison_provenance=provenance,comparison_code_sha256=code,external_identity_audit=identity_audit,
            external_window_design_audit=window_audit,missing_template_roles=[],inherited_reference_correction=old['inherited_reference_correction'])
        core['plan_id']=identity_hash(scientific_plan(core))
        sizes=[sum(d['outputs'][name]['size_bytes'] for name in CACHE_FILES[:3]) for d in datasets]
        core.update(total_cache_bytes=sum(sizes),max_file_cache_bytes=max(sizes),shared_cache_budget_bytes=max(sizes),
            cache_budget_scope='All72 external files; read-only shared cache, one mmap file per prediction process',
            created_at=existing.get('created_at') if existing else datetime.now(timezone.utc).isoformat())
        if existing is not None and existing.get('plan_id')!=core['plan_id']:
            raise ValueError('数据、掩码、种子或科学配置改变，不能续跑')
        if any(sha256(item['path'])!=item['sha256'] for item in sources):raise RuntimeError('准备期间来源记录发生变化')
        preserve_source_metadata(sources,output)
        atomic_json(output/'evaluation_plan.json',core)
        atomic_json(output/'external_identity_audit.json',identity_audit)
        atomic_json(output/'external_window_design_audit.json',window_audit)
        reused=[]
        for run in full:
            for dataset in datasets:
                _reuse_one(previous,output,old,core,run,dataset,args.reuse_mode,reserve)
                path=previous/'predictions'/run['run_id']/dataset['dataset_id']/'completion.json'
                marker=read_json(path)
                reused.append(dict(run_id=run['run_id'],dataset_id=dataset['dataset_id'],source_completion=str(path),source_completion_sha256=sha256(path),outputs=marker['outputs']))
            atomic_json(output/'predictions'/run['run_id']/'run_complete.json',dict(status='complete',plan_id=core['plan_id'],
                run_id=run['run_id'],prediction_settings=settings,datasets=[d['dataset_id'] for d in datasets],
                completed_at=datetime.now(timezone.utc).isoformat(),reused_without_new_inference=True))
            print('[REUSE] '+run['run_id']+'：72个逐文件输出已核验复用',flush=True)
        atomic_json(output/'reuse_manifest.json',dict(status='complete',plan_id=core['plan_id'],previous_plan_id=old['plan_id'],
            reuse_mode=args.reuse_mode,reused_model_runs=[r['run_id'] for r in full],new_prediction_model_runs=[r['run_id'] for r in new],
            reused_file_predictions=len(reused),new_file_predictions=9*72,cache_records=cache_records,reused_exports=reused,
            all_source_outputs_size_and_sha_verified=True,old_directories_modified=False))
        atomic_json(output/'singleinput_source_provenance.json',provenance)
        print('[PLAN] 9个单输入检查点新增648份预测；3个full复用216份预测；不训练。',flush=True)
        return core


def prepare(args):
    if getattr(args,'check_only',False):return _prepare_locked(args)
    output=Path(args.output_root).expanduser().resolve();output.mkdir(parents=True,exist_ok=True)
    with correction_write_lock(output,getattr(args,'parent_queue_pid',None)):
        return _prepare_locked(args)


def parse_args(argv=None):
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--previous_output',type=Path,default=DEFAULT_PREVIOUS)
    p.add_argument('--training_dir',type=Path,default=DEFAULT_TRAINING)
    p.add_argument('--output_root','--output_dir',dest='output_root',type=Path,default=DEFAULT_OUTPUT)
    p.add_argument('--reuse_mode',choices=('symlink','hardlink','copy'),default='symlink')
    p.add_argument('--disk_reserve_gib',type=float,default=8)
    p.add_argument('--resume',action='store_true')
    p.add_argument('--check_only',action='store_true')
    p.add_argument('--parent_queue_pid',type=int)
    a=p.parse_args(argv)
    if not math.isfinite(a.disk_reserve_gib) or a.disk_reserve_gib<0:p.error('disk_reserve_gib须为有限非负数')
    return a


if __name__=='__main__':
    try:prepare(parse_args())
    except Exception as error:
        import sys
        print('[ERROR] '+type(error).__name__+': '+str(error),file=sys.stderr)
        raise SystemExit(1)
