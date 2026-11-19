#!/usr/bin/env python3
"""继承修正后的Figure4数据，复用三个full，仅为三个门控卷积模型准备预测。"""
from __future__ import annotations

import argparse
from collections import Counter, defaultdict
import copy
import csv
from datetime import datetime, timezone
import json
import math
from pathlib import Path
import shutil
import sqlite3

from figure4_common import atomic_json, ensure_disk, fingerprint, identity_hash, known, read_json, sha256, verify_run
from prepare_figure4 import cache_valid, independence_audit, prepare_cache
from prepare_figure4_correction import (_reuse_one, _validate_old_plan,
    correction_write_lock, previous_read_lock, SETTING_KEYS)

HERE = Path(__file__).resolve().parent
RESULTS = '/path/to/revision-inputs/CPLX-XNA-replenishment/CPLX-XNA-replenishment/fast5_single_guppy/sixclass_overlap_read_split_721/sixclass_auto_v1_0_4_results'
DEFAULT_PREVIOUS = RESULTS + '/figure4_external_xna13_16_corrected_v1_1_0'
DEFAULT_CONFIRMATION = RESULTS + '/litemamba_structure_confirm3_v1_1_0'
DEFAULT_OUTPUT = RESULTS + '/figure4_gated_comparison_v1_0_0'
SEEDS = (42, 2024, 2025)
DECLARATIONS = ('feature_csv', 'metadata_tsv', 'reference', 'file_strand', 'source_dataset',
                'run_id', 'template_group', 'pair_id', 'template', 'cohort', 'role')
CACHE_FILES = ('merged.npy', 'raw.npy', 'labels.npy', 'metadata.jsonl')
CODE_FILES = ('figure4_common.py', 'figure4_worker.py', 'prepare_gated_comparison.py',
              'prepare_figure4.py', 'prepare_figure4_correction.py', 'comparison_registry.py',
              'run_figure4.py', 'run_gated_comparison.py', 'vendor/resource_scheduler.py',
              'vendor/cuda_device_probe.py', 'vendor/IterDataset.py', 'vendor/model_benchmark_all.py',
              'vendor_full/IterDataset.py', 'vendor_full/model_benchmark_all.py')


def code_fingerprints():
    return {name: sha256(HERE / name) for name in CODE_FILES}


def fingerprint_content(value):
    if isinstance(value, dict):
        if 'sha256' in value:
            return {'sha256': value['sha256'], 'size_bytes': value.get('size_bytes', value.get('size'))}
        return {key: fingerprint_content(child) for key, child in value.items()}
    if isinstance(value, list):
        return [fingerprint_content(child) for child in value]
    return value


def validate_dataset_design(plan):
    _validate_old_plan(plan)
    if plan.get('correction_provenance', {}).get('kind') != 'reference_sequence_correction':
        raise ValueError('必须使用已修正XNA13–16参考的v1.1.0结果目录，不能使用更正前数据')
    expected = {(f'XNA{i}', role, strand) for i in range(1, 21)
                for role in (('target', 'control') if i <= 16 else ('target',))
                for strand in ('pos', 'neg')}
    actual = [(d.get('template'), d.get('role'), d.get('file_strand')) for d in plan['datasets']]
    if len(actual) != 72 or set(actual) != expected or len(set(actual)) != len(actual):
        raise ValueError('修正后的Figure4计划必须包含全部72个模板/角色/方向文件，不接受筛选子集')
    replaced = set(plan.get('replacement_dataset_ids', []))
    corrected = {d['dataset_id'] for d in plan['datasets']
                 if d['template'] in {'XNA13','XNA14','XNA15','XNA16'} and d['role']=='target'}
    if not corrected.issubset(replaced):
        raise ValueError('XNA13–16 target没有完整记录在已修正数据范围内')
    for dataset in plan['datasets']:
        if dataset.get('status') != 'complete' or int(dataset.get('n_samples', 0)) < 1:
            raise ValueError('旧数据缓存清单未完整：' + dataset['dataset_id'])
        if set(dataset.get('outputs', {})) != set(CACHE_FILES):
            raise ValueError('旧数据缓存清单不完整：' + dataset['dataset_id'])
        if dataset.get('identity', {}).get('preprocessing') != plan['preprocessing']:
            raise ValueError('旧数据缓存与冻结预处理不符：' + dataset['dataset_id'])


def inherit_settings(previous, old_plan, full_runs):
    settings = copy.deepcopy(old_plan.get('prediction_settings'))
    if not isinstance(settings, dict) or set(settings) != SETTING_KEYS:
        raise ValueError('旧计划缺少完整一致的预测设置')
    if settings['workers'] != 0 or settings['amp_policy'] != 'training_checkpoint_config':
        raise ValueError('旧预测worker/AMP设置不受当前入口支持')
    if (settings['embedding_dtype'] not in ('float16','float32')
            or type(settings['batch_size']) is not int or settings['batch_size'] < 1
            or type(settings['cpu_threads']) is not int or settings['cpu_threads'] < 1):
        raise ValueError('旧预测设置无效')
    dataset_ids = {d['dataset_id'] for d in old_plan['datasets']}
    old_runs = {r['run_id']:r for r in old_plan['runs']}
    for run in full_runs:
        old = old_runs.get(run['run_id'])
        if old is None:
            raise ValueError('旧Figure4没有相同seed的full：' + run['run_id'])
        for key in ('model','seed','feature_scheme','checkpoint_sha256','resolved_config_sha256',
                    'calibration_sha256','temperature','model_kwargs','runtime_sha256'):
            if old.get(key) != run.get(key):
                raise ValueError('旧full预测与本次冻结模型不一致：' + run['run_id'] + '/' + key)
        if fingerprint_content(old.get('training_input_fingerprints')) != fingerprint_content(run.get('training_input_fingerprints')):
            raise ValueError('旧full训练输入与本次模型不同：' + run['run_id'])
        marker_path = previous/'predictions'/run['run_id']/'run_complete.json'
        marker = read_json(marker_path)
        if (marker.get('status')!='complete' or marker.get('plan_id')!=old_plan['plan_id']
                or marker.get('run_id')!=run['run_id'] or marker.get('prediction_settings')!=settings
                or set(marker.get('datasets',[]))!=dataset_ids):
            raise ValueError('旧full运行完成记录不完整或身份不同：' + str(marker_path))
    return settings


def verify_sources(dataset):
    current = {}
    for kind in ('feature','metadata'):
        expected = dataset['identity'].get(kind)
        declared = dataset.get('feature_csv' if kind=='feature' else 'metadata_tsv')
        if expected is None:
            if declared:
                raise ValueError('旧输入metadata声明与冻结指纹不同：' + dataset['dataset_id'])
            current[kind] = None
            continue
        if not declared or Path(declared).resolve() != Path(expected['path']).resolve():
            raise ValueError('旧输入路径声明与冻结指纹不同：' + dataset['dataset_id'])
        actual = fingerprint(expected['path'])
        if fingerprint_content(actual) != fingerprint_content(expected):
            raise ValueError('冻结后的Figure4源文件内容改变：' + expected['path'])
        current[kind] = actual
    return current


def cache_matches(dataset, current):
    for kind in ('feature','metadata'):
        if dataset['identity'].get(kind) != current[kind]:
            return False
    return cache_valid(Path(dataset['cache_dir']), dataset['identity'])


def bind_cache(old, preprocessing, output, reserve, existing=None):
    current = verify_sources(old)
    for candidate in (existing, old):
        if candidate is None:
            continue
        try:
            complete = cache_matches(candidate, current)
        except (OSError,KeyError,ValueError):
            complete = False
        if complete:
            result = copy.deepcopy(candidate)
            result['dataset_id'] = old['dataset_id']
            return result, 'verified_existing_cache'
    item = copy.deepcopy(old['identity']['declarations'])
    if any(item.get(k) != old.get(k) for k in DECLARATIONS):
        raise ValueError('旧数据declarations与记录不符：' + old['dataset_id'])
    rebuilt = prepare_cache(item, preprocessing, output, reserve)
    if rebuilt['n_samples'] != old['n_samples'] or rebuilt['label_counts'] != old['label_counts']:
        raise ValueError('重建缓存的样本数/真值数量与旧冻结缓存不同：' + old['dataset_id'])
    for name in CACHE_FILES:
        if fingerprint_content(rebuilt['outputs'][name]) != fingerprint_content(old['outputs'][name]):
            raise ValueError('重建缓存的内容与旧Figure4输入不完全相同：' + old['dataset_id'] + '/' + name)
    rebuilt['cache_generated_dataset_id'] = rebuilt['dataset_id']
    rebuilt['dataset_id'] = old['dataset_id']
    rebuilt['cache_rebuild_basis'] = 'all four cache artifact content hashes match the old frozen plan'
    return rebuilt, 'rebuilt_identical_cache'


def logical_dataset(dataset):
    keys = ('dataset_id','template','cohort','role','reference','file_strand','feature_csv','metadata_tsv',
            'source_dataset','run_id','template_group','pair_id','n_samples','label_counts')
    value = {key:dataset.get(key) for key in keys}
    value['input_contents'] = {kind:fingerprint_content(dataset['identity'].get(kind)) for kind in ('feature','metadata')}
    value['preprocessing'] = dataset['identity']['preprocessing']
    value['cache_contents'] = {name:fingerprint_content(dataset['outputs'][name]) for name in CACHE_FILES}
    return value


def scientific_plan(plan):
    keys = ('version','task','training_results','runs','preprocessing','models','seeds','feature_schemes',
            'prediction_settings','selection_policy','comparison_provenance','comparison_code_sha256')
    return {**{key:plan[key] for key in keys}, 'datasets':[logical_dataset(d) for d in plan['datasets']]}


def coordinate(metadata):
    chrom = str(metadata.get('chrom') or metadata.get('reference') or '').strip()
    start, end = metadata.get('kmer_ref_start'), metadata.get('kmer_ref_end')
    if not known(chrom) or not known(start) or not known(end):
        return chrom, None, None, 'missing_explicit_window_coordinates'
    try:
        start, end = int(start), int(end)
    except (ValueError, TypeError):
        return chrom, None, None, 'invalid_coordinates'
    if start < 0 or end-start+1 != 6:
        return chrom, start, end, 'invalid_coordinates'
    return chrom, start, end, 'valid'


def window_design_audit(datasets, output):
    """按真实read核验1/2/4窗口；13号模板和对照允许1bp交叠。"""
    database = output/'window_design_audit.sqlite'
    connection = sqlite3.connect(database)
    connection.execute('DROP TABLE IF EXISTS windows')
    connection.execute('CREATE TABLE windows(template TEXT, role TEXT, rid TEXT, chrom TEXT, s INTEGER, e INTEGER, valid INTEGER)')
    scopes = {}
    for d in datasets:
        key = (d['template'],d['role'])
        if key not in scopes:
            number = int(d['template'][3:])
            scopes[key] = dict(template=d['template'],role=d['role'],expected_max_windows_per_read=1 if number<=12 else 2 if number<=16 else 4,
                allowed_overlap_bp=1 if number==13 else 0,rows=0,real_id_rows=0,unknown_id_rows=0,
                valid_coordinate_rows=0,missing_coordinate_rows=0,invalid_coordinate_rows=0,
                above_expected_reads=0,below_expected_reads=0,allowed_overlap_pairs=0,excess_overlap_pairs=0)
        state = scopes[key]
        batch = []
        with (Path(d['cache_dir'])/'metadata.jsonl').open(encoding='utf-8') as stream:
            for line in stream:
                row = json.loads(line); state['rows'] += 1
                if row.get('read_id_kind') != 'sequencing_read_id' or not known(row.get('read_id')):
                    state['unknown_id_rows'] += 1
                    continue
                state['real_id_rows'] += 1
                chrom,start,end,status = coordinate(row.get('original_metadata',{}))
                state['valid_coordinate_rows' if status=='valid' else 'invalid_coordinate_rows' if status=='invalid_coordinates' else 'missing_coordinate_rows'] += 1
                batch.append((key[0],key[1],row['read_id'],chrom,start,end,int(status=='valid')))
                if len(batch)>=10000:
                    connection.executemany('INSERT INTO windows VALUES(?,?,?,?,?,?,?)',batch);batch.clear()
        if batch:connection.executemany('INSERT INTO windows VALUES(?,?,?,?,?,?,?)',batch)
    connection.commit()
    connection.execute('CREATE INDEX by_read ON windows(template,role,rid)')
    examples = []
    for t,r,rid,n in connection.execute('SELECT template,role,rid,COUNT(*) FROM windows GROUP BY template,role,rid'):
        state=scopes[t,r]; maximum=state['expected_max_windows_per_read']
        state['above_expected_reads'] += int(n>maximum);state['below_expected_reads'] += int(n<maximum)
        if n>maximum and len(examples)<20:examples.append(dict(template=t,role=r,read_id=rid,issue='above_expected_windows',observed=n))
    previous, intervals = None, []
    for t,r,rid,chrom,start,end in connection.execute('SELECT template,role,rid,chrom,s,e FROM windows WHERE valid=1 ORDER BY template,role,rid,chrom,s,e'):
        key=(t,r,rid,chrom)
        if key!=previous:previous,intervals=key,[]
        # 保留仍可能与当前6mer相交的窗口，无需按整条read存储任意长列表。
        intervals=[(s,e) for s,e in intervals if e>=start]
        for a,b in intervals:
            overlap=min(b,end)-max(a,start)+1
            if overlap>0:
                state=scopes[t,r]
                state['allowed_overlap_pairs' if overlap<=state['allowed_overlap_bp'] else 'excess_overlap_pairs'] += 1
                if overlap>state['allowed_overlap_bp'] and len(examples)<20:
                    examples.append(dict(template=t,role=r,read_id=rid,issue='excess_window_overlap',overlap_bp=overlap))
        intervals.append((start,end))
    connection.close()
    rows=sorted(scopes.values(),key=lambda r:(int(r['template'][3:]),r['role']))
    for state in rows:
        failure=state['above_expected_reads'] or state['invalid_coordinate_rows'] or state['excess_overlap_pairs']
        complete=state['real_id_rows']==state['rows'] and state['valid_coordinate_rows']==state['rows'] and state['rows']>0
        state['status']='known_deviation' if failure else 'verified_on_observed_metadata' if complete else 'not_fully_verifiable'
    report=dict(status='known_deviation' if any(r['status']=='known_deviation' for r in rows) else 'verified_on_observed_metadata' if all(r['status']=='verified_on_observed_metadata' for r in rows) else 'not_fully_verifiable',
        policy=dict(real_read_key=['template','role','read_id'],aggregate_across_pos_neg=True,
                    expected_max_windows={'1-12':1,'13-16':2,'17-20':4},allowed_overlap_bp={'template13_target_and_control':1,'all_others':0},
                    missing_identity_or_coordinates='Keep experimental design declaration; absence limits automated verification and does not imply non-independence.',
                    below_expected='Descriptive; filtering or incomplete reads may contribute fewer windows. No rows are removed or deduplicated.',
                    coordinates='Only explicit inclusive kmer_ref_start/kmer_ref_end of length6 plus chrom/reference; no guessed or shifted coordinates.'),
        per_template=rows,examples=examples)
    atomic_json(output/'external_window_design_audit.json',report)
    if report['status']=='known_deviation':
        raise ValueError('已知read窗口数量/坐标超过声明规则，详见external_window_design_audit.json；未修改或丢弃数据')
    return report


def metadata_sources(previous, confirmation):
    files=[previous/'evaluation_plan.json',confirmation/'structure_plan.json',confirmation/'structure_summary/structure_comparison.json']
    for name in ('corrected_split_audit.json','reference_correction_provenance.json','external_identity_audit.json'):
        if (previous/name).is_file():files.append(previous/name)
    return [dict(path=str(path),sha256=sha256(path),
                 local_copy='source_provenance/'+('previous_output' if path.is_relative_to(previous) else 'confirmation')
                            +'/'+str(path.relative_to(previous if path.is_relative_to(previous) else confirmation)))
            for path in files]


def preserve_source_metadata(files, output):
    for item in files:
        target=output/item['local_copy'];target.parent.mkdir(parents=True,exist_ok=True)
        temporary=target.with_name(target.name+'.tmp')
        shutil.copyfile(item['path'],temporary)
        if sha256(temporary)!=item['sha256']:
            raise RuntimeError('复制来源元数据时内容发生变化：'+item['path'])
        temporary.replace(target)


def _prepare_locked(args):
    from comparison_registry import discover_comparison_runs
    previous=Path(args.previous_output).expanduser().resolve()
    confirmation=Path(args.confirmation_dir).expanduser().resolve()
    output=Path(args.output_root).expanduser().resolve()
    for protected in (previous,confirmation,HERE):
        if output==protected or output in protected.parents or protected in output.parents:
            raise ValueError('新输出须独立于旧结果、训练结果和代码目录：'+str(output))
    existing=read_json(output/'evaluation_plan.json') if (output/'evaluation_plan.json').is_file() else None
    if existing is not None and not args.resume:
        raise ValueError('已有确认评估计划，请保留参数并加--resume')
    with previous_read_lock(previous):
        old_plan=read_json(previous/'evaluation_plan.json')
        validate_dataset_design(old_plan)
        root,runs,preprocessing=discover_comparison_runs(confirmation,getattr(args,'baseline_root',None))
        if preprocessing!=old_plan['preprocessing']:
            raise ValueError('新旧模型的预处理与旧Figure4不完全一致')
        if output==root or output in root.parents or root in output.parents:
            raise ValueError('新输出不能覆盖或嵌入旧训练目录')
        expected={(model,seed) for model in ('litemamba','gatedconv_target') for seed in SEEDS}
        if len(runs)!=6 or {(r['model'],r['seed']) for r in runs}!=expected:
            raise ValueError('模型注册必须为固定3seed的full和门控卷积，共6项')
        for run in runs:verify_run(run)
        full=[r for r in runs if r['model']=='litemamba']
        settings=inherit_settings(previous,old_plan,full)
        sources=metadata_sources(previous,confirmation)
        provenance=dict(kind='frozen_structure_external_comparison',previous_output=str(previous),confirmation_dir=str(confirmation),
            baseline_root=str(Path(args.baseline_root).expanduser().resolve()) if getattr(args,'baseline_root',None) else None,
            reuse_mode=args.reuse_mode,previous_plan_id=old_plan['plan_id'],previous_plan_sha256=sha256(previous/'evaluation_plan.json'),
            confirmation_plan_sha256=sha256(confirmation/'structure_plan.json'),
            confirmation_comparison_sha256=sha256(confirmation/'structure_summary/structure_comparison.json'),source_metadata_files=sources,
            evaluation_role='Follow-up comparison on previously observed external data; no retraining, external calibration or threshold fitting.')
        code=code_fingerprints()
        if existing is not None and (existing.get('comparison_provenance')!=provenance or existing.get('comparison_code_sha256')!=code):
            raise ValueError('冻结后的来源或预测处理代码改变，请使用新输出目录')
        reserve=float(args.disk_reserve_gib)
        prior_caches={d['dataset_id']:d for d in existing['datasets']} if existing else {}
        datasets=[]; cache_records=[]
        for old in old_plan['datasets']:
            dataset,action=bind_cache(old,preprocessing,output,reserve,prior_caches.get(old['dataset_id']))
            datasets.append(dataset);cache_records.append(dict(dataset_id=old['dataset_id'],action=action,cache_dir=dataset['cache_dir']))
        audit=independence_audit(datasets,runs,output)
        audit['inherited_audit_for_same_external_contents']=old_plan.get('external_identity_audit',{})
        audit['experimenter_read_window_design']='XNA/DNA1–12: one window/read; XNA/DNA13–16: two; XNA17–20: four. Template13 target/control may overlap by one base.'
        atomic_json(output/'external_identity_audit.json',audit)
        window_audit=window_design_audit(datasets,output)
        core=dict(version='gated-comparison-1.0.0',task='6-class',training_results=str(root),runs=runs,datasets=datasets,
            preprocessing=preprocessing,models=['litemamba','gatedconv_target'],seeds=list(SEEDS),feature_schemes=['full'],
            prediction_settings=settings,selection_policy='All three prespecified seeds of both frozen structures; validation-loss checkpoints and validation-only temperatures. No new training.',
            comparison_provenance=provenance,comparison_code_sha256=code,external_identity_audit=audit,
            external_window_design_audit=window_audit,missing_template_roles=[],
            inherited_reference_correction=old_plan['correction_provenance'])
        core['plan_id']=identity_hash(scientific_plan(core))
        sizes=[sum(d['outputs'][name]['size_bytes'] for name in CACHE_FILES[:3]) for d in datasets]
        core.update(total_cache_bytes=sum(sizes),max_file_cache_bytes=max(sizes),shared_cache_budget_bytes=max(sizes),
                    cache_budget_scope='All72 external files; one mmap file per prediction process',created_at=datetime.now(timezone.utc).isoformat())
        if existing is not None and existing.get('plan_id')!=core['plan_id']:
            raise ValueError('已有计划的模型、数据或科学设置改变，不能混合续跑')
        if any(sha256(item['path'])!=item['sha256'] for item in sources):
            raise RuntimeError('准备期间旧计划或确认来源记录发生变化')
        preserve_source_metadata(sources,output)
        atomic_json(output/'evaluation_plan.json',core)
        reused=[]
        for run in full:
            for dataset in datasets:
                _reuse_one(previous,output,old_plan,core,run,dataset,args.reuse_mode,reserve)
                old_folder=previous/'predictions'/run['run_id']/dataset['dataset_id']
                old_marker=read_json(old_folder/'completion.json')
                reused.append(dict(run_id=run['run_id'],dataset_id=dataset['dataset_id'],source_completion=str(old_folder/'completion.json'),
                    source_completion_sha256=sha256(old_folder/'completion.json'),outputs=old_marker['outputs']))
            atomic_json(output/'predictions'/run['run_id']/'run_complete.json',dict(status='complete',plan_id=core['plan_id'],
                run_id=run['run_id'],prediction_settings=settings,datasets=[d['dataset_id'] for d in datasets],
                completed_at=datetime.now(timezone.utc).isoformat(),reused_without_new_inference=True))
            print('[REUSE] '+run['run_id']+'：72个逐文件输出已核验并只读复用',flush=True)
        atomic_json(output/'reuse_manifest.json',dict(status='complete',plan_id=core['plan_id'],previous_plan_id=old_plan['plan_id'],
            reuse_mode=args.reuse_mode,reused_model_runs=[r['run_id'] for r in full],
            new_prediction_model_runs=[r['run_id'] for r in runs if r['model']=='gatedconv_target'],
            reused_file_predictions=len(reused),new_file_predictions=3*len(datasets),cache_records=cache_records,reused_exports=reused,
            all_source_outputs_size_and_sha_verified=True,old_directories_modified=False))
        atomic_json(output/'comparison_source_provenance.json',provenance)
        print('[COMPARISON PLAN] 3项full复用216个逐文件预测；3项门控卷积新增216个逐文件预测；无训练。',flush=True)
        return core


def prepare(args):
    output=Path(args.output_root).expanduser().resolve()
    output.mkdir(parents=True,exist_ok=True)
    with correction_write_lock(output,getattr(args,'parent_queue_pid',None)):
        return _prepare_locked(args)


prepare_comparison=prepare


def parse_args(argv=None):
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--previous_output',type=Path,default=Path(DEFAULT_PREVIOUS))
    parser.add_argument('--confirmation_dir',type=Path,default=Path(DEFAULT_CONFIRMATION))
    parser.add_argument('--baseline_root',type=Path)
    parser.add_argument('--output_root','--output_dir',dest='output_root',type=Path,default=Path(DEFAULT_OUTPUT))
    parser.add_argument('--reuse_mode',choices=('symlink','hardlink','copy'),default='symlink')
    parser.add_argument('--disk_reserve_gib',type=float,default=8.0)
    parser.add_argument('--resume',action='store_true')
    parser.add_argument('--parent_queue_pid',type=int)
    args=parser.parse_args(argv)
    if not math.isfinite(args.disk_reserve_gib) or args.disk_reserve_gib<0:parser.error('disk_reserve_gib须为有限非负数')
    return args


if __name__=='__main__':
    prepare(parse_args())
