#!/usr/bin/env python3
"""CPLX 同来源六分类：固定最终 GatedConv；默认仅 seed42、train/validation。"""
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
import shutil
import signal
import subprocess
import sys
import time

VERSION='1.0.0'
HERE=Path(__file__).resolve().parent
ENGINE=HERE/'engine'
REFERENCE_DIR=HERE/'reference'
SEEDS=(42,2024,2025)
DEFAULT_SOURCE='cplx_all_same_source_6class_v1_0_0'
OUTPUT_NAME='gatedconv_same_source_v1_0_0'
SCIENTIFIC={
    'model_type':'litemamba','litemamba_variant':'legacy_gatedconv',
    'readout':'target_attention','feature_scheme':'full','model_dim':96,
    'num_blocks':4,'state_dim':16,'target_index':3,'dropout':0.1,
    'classifier_dropout':0.2,'ssm_bidirectional':1,'nhead':4,
    'num_layers':2,'gru_hidden':64,'epochs':200,'patience':30,
    'lr':0.001,'batch_size':256,'raw_signal_len':30,
    'dwell_divisor':60.0,'quality_divisor':50.0,'accum_steps':1,
    'weight_decay':0.0001,'label_smoothing':0.0,'clip_grad':1.0,
    'monitor':'val_loss','bootstrap':200,'embedding_dtype':'float32',
    'checkpoint_every':10,
}
GIB=2**30
STOP=False
CHILDREN={}
LOCK_FD=None

def now(): return datetime.now(timezone.utc).isoformat()
def load_json(path): return json.loads(Path(path).read_text(encoding='utf-8'))
def sha256(path):
    h=hashlib.sha256()
    with Path(path).open('rb') as f:
        for b in iter(lambda:f.read(8*1024**2),b''):h.update(b)
    return h.hexdigest()
def write_json(path,data):
    path=Path(path);path.parent.mkdir(parents=True,exist_ok=True)
    tmp=path.with_name(path.name+'.writing.'+str(os.getpid()))
    try:
        with tmp.open('w',encoding='utf-8') as f:
            json.dump(data,f,ensure_ascii=False,indent=2,allow_nan=False)
            f.write('\n');f.flush();os.fsync(f.fileno())
        os.replace(tmp,path)
    finally:
        if tmp.exists():tmp.unlink()
def process_token(pid):
    try:
        # comm can contain spaces; fields after its final ')' start at field 3.
        s=Path('/proc/%d/stat'%pid).read_text();tail=s[s.rfind(')')+2:].split()
        if tail[0]=='Z':return None
        return tail[19]
    except (OSError,IndexError):return None
def live_job(record):
    return bool(record.get('pid') and record.get('process_start_token') and
                process_token(record['pid'])==record['process_start_token'])
def request_stop(sig,frame):
    global STOP
    STOP=True
    for p in list(CHILDREN.values()):
        if p.poll() is None:p.terminate()

def parser():
    p=argparse.ArgumentParser(description=__doc__,formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    p.add_argument('--version',action='version',version='CPLX same-source GatedConv '+VERSION)
    p.add_argument('--source_dir',default=DEFAULT_SOURCE)
    p.add_argument('--output_dir')
    p.add_argument('--phase',choices=('prepare','pilot','confirm','freeze','evaluate','summary'),default='pilot')
    p.add_argument('--resume',action='store_true')
    p.add_argument('--device',default='auto')
    p.add_argument('--cpu_threads',type=int,default=4)
    p.add_argument('--workers',type=int,default=0)
    p.add_argument('--max_parallel',type=int,choices=(1,2),default=2,help='仅confirm阶段的两次权重训练；导出/评价仍串行')
    p.add_argument('--disk_reserve_gib',type=float,default=4)
    p.add_argument('--gpu_reserve_gib',type=float,default=2)
    p.add_argument('--ram_reserve_gib',type=float,default=3)
    p.add_argument('--reserve_fraction',type=float,default=.10)
    p.add_argument('--memory_safety_factor',type=float,default=1.35)
    p.add_argument('--poll_seconds',type=float,default=3)
    p.add_argument('--dry_run',action='store_true',help='打印固定阶段，不读取数据或启动训练')
    return p

def validate_args(args):
    args.source_dir=Path(args.source_dir).expanduser().resolve()
    args.output_dir=Path(args.output_dir or args.source_dir/OUTPUT_NAME).expanduser().resolve()
    if args.output_dir==args.source_dir or args.output_dir in args.source_dir.parents:
        raise ValueError('输出必须是独立子目录，不能覆盖源数据目录或其父目录')
    if args.output_dir==HERE or HERE in args.output_dir.parents or args.output_dir in HERE.parents:
        raise ValueError('结果目录不能位于代码目录内或覆盖代码目录')
    if args.device not in ('auto','cpu','cuda'):
        if not args.device.startswith('cuda:') or not args.device[5:].isdigit():raise ValueError('device仅支持auto/cpu/cuda/cuda:N')
    values=[args.disk_reserve_gib,args.gpu_reserve_gib,args.ram_reserve_gib,args.reserve_fraction,args.memory_safety_factor,args.poll_seconds]
    if not all(math.isfinite(x) for x in values):raise ValueError('资源参数须为有限数')
    if args.cpu_threads<1 or args.workers<0 or args.disk_reserve_gib<=0 or min(args.gpu_reserve_gib,args.ram_reserve_gib)<0:
        raise ValueError('线程/磁盘余量须为正数，workers及内存余量不能负')
    if not 0<=args.reserve_fraction<1 or args.memory_safety_factor<1 or args.poll_seconds<=0:raise ValueError('资源比例/系数/间隔无效')
    return args

def source_identity():
    expected=load_json(REFERENCE_DIR/'engine_identity.json')
    for name,digest in expected['engine_sha256'].items():
        if sha256(ENGINE/name)!=digest:raise ValueError('随包冻结引擎发生变化：'+name)
    files=[HERE/x for x in ('run_same_source.py','prepare_data.py','summarize_same_source.py','reference6.py')]
    files+=sorted(ENGINE.glob('*.py'))+sorted(REFERENCE_DIR.glob('*'))
    return {(str(p.relative_to(HERE)) if p.is_relative_to(HERE) else 'reference/'+p.name):sha256(p) for p in files if p.is_file()}

def make_plan(args,prepared):
    return {'version':VERSION,'model_name':'GatedConv','class_names':['A','T','C','G','X','Y'],
        'fixed_seeds':list(SEEDS),'scientific_configuration':dict(SCIENTIFIC),
        'initialization':'independent seeded random initialization; no pretrained weights',
        'training_order':'seed42 pilot train/val, then fixed 2024 and 2025; no test-dependent setting changes',
        'test_policy':'no model prediction on test before all three best checkpoints and validation temperatures are frozen',
        'source_dir':str(args.source_dir),'output_dir':str(args.output_dir),
        'prepared_manifest_sha256':sha256(prepared['manifest_path']),
        'source_code_sha256':source_identity(),
        'source_features_sha256':load_json(REFERENCE_DIR/'extraction_summary.json')['features']['sha256'],
        'source_metadata_sha256':load_json(REFERENCE_DIR/'extraction_summary.json')['metadata']['sha256'],
        'scope':'same CPLX batch and pooled common contexts; read holdout, not batch/template holdout; NCB/natural positions and X/Y direction association remain',
        'batch_id':'CPLX-XNA-all','physical_run_id':'unknown','planned_test_rows':load_json(REFERENCE_DIR/'extraction_summary.json')['rows_by_split']['test'],
        'target_mask_index':3,'seq_len':6,'amp_policy':'CUDA AMP enabled; CPU FP32, as in the frozen trainer'}

def commit_plan(args,prepared):
    path=args.output_dir/'experiment_plan.json';plan=make_plan(args,prepared)
    if path.exists():
        if load_json(path)!=plan:raise ValueError('输入、源码或固定实验方案改变；不能混入已有结果目录')
    else:write_json(path,plan)
    return plan

def check_disk(args,remaining_runs=0,include_prepare=False):
    # Conservative storage budget, not a measured final size. Do not reserve for
    # all three jobs when the user is only launching the seed42 pilot.
    n=load_json(REFERENCE_DIR/'extraction_summary.json')['rows']
    val=load_json(REFERENCE_DIR/'extraction_summary.json')['rows_by_split']['validation']
    budget=remaining_runs*(n*7000+val*40*SCIENTIFIC['epochs']+150*1024**2)
    if include_prepare:budget+=n*3500
    needed=budget+args.disk_reserve_gib*GIB
    free=shutil.disk_usage(args.output_dir).free
    if free<needed:raise OSError(28,'输出盘剩余%.2f GiB，当前阶段保守预算%.2f GiB（含%.2f GiB余量）；清理后使用--resume'%(free/GIB,needed/GIB,args.disk_reserve_gib))
    print('[DISK] free=%.2f GiB current_stage_budget=%.2f GiB'%(free/GIB,needed/GIB),flush=True)

def child_command(args,prepared,seed,mode,run_dir=None):
    run_dir=Path(run_dir or args.output_dir/'runs'/('seed_'+str(seed)))
    cmd=[sys.executable,'-u',str(ENGINE/'train_all.py'),'--mode',mode,
         '--data_dir',str(prepared['data_dir']),'--cache_dir',str(args.output_dir/'feature_cache'),
         '--output_dir',str(run_dir),'--device',args.device,'--cpu_threads',str(args.cpu_threads),
         '--workers',str(args.workers),'--concurrent_training']
    if mode in ('train','preflight','profile'):
        for k,v in SCIENTIFIC.items():cmd+=['--'+k,str(v)]
        cmd+=['--seed',str(seed),'--defer_test','--defer_train_export']
    if mode=='profile':cmd+=['--profile_steps','8']
    if mode=='finalize':cmd+=['--defer_test','--defer_train_export']
    return cmd

def orphan_check(args):
    for f in (args.output_dir/'job_records').glob('*.json'):
        r=load_json(f)
        if r.get('state')=='running' and live_job(r):
            raise RuntimeError('此前子进程仍在运行，禁止重复启动：pid=%s log=%s'%(r['pid'],r.get('log')))

def start_child(args,command,name):
    if STOP:raise InterruptedError('收到停止请求，未启动新任务')
    out=args.output_dir;stamp=datetime.now().strftime('%Y%m%d_%H%M%S_%f')
    log=out/'logs'/('%s_%s.log'%(name,stamp));log.parent.mkdir(exist_ok=True)
    record=out/'job_records'/('%s_%s.json'%(name,stamp))
    data={'state':'launching','name':name,'command':command,'log':str(log),'started_at':now()}
    write_json(record,data)
    f=log.open('wb')
    try:
        env=dict(os.environ,OMP_NUM_THREADS=str(args.cpu_threads),MKL_NUM_THREADS=str(args.cpu_threads),PYTHONUNBUFFERED='1')
        p=subprocess.Popen(command,stdout=f,stderr=subprocess.STDOUT,stdin=subprocess.DEVNULL,env=env,
                           pass_fds=(() if LOCK_FD is None else (LOCK_FD,)))
        CHILDREN[p.pid]=p
        data.update(state='running',pid=p.pid,process_start_token=process_token(p.pid))
        try:write_json(record,data)
        except BaseException:
            p.terminate();p.wait();CHILDREN.pop(p.pid,None);raise
    finally:f.close()
    print('[LAUNCH] %s pid=%d log=%s'%(name,p.pid,log),flush=True)
    return p,record,data

def finish_child(handle):
    p,record,data=handle;code=p.wait();CHILDREN.pop(p.pid,None)
    data.update(state='complete' if code==0 else 'failed',returncode=code,finished_at=now())
    write_json(record,data)
    return code

def progress_message(handle):
    data=handle[2];clock=time.monotonic()
    if clock-data.get('_last_progress_clock',0)<30:return
    data['_last_progress_clock']=clock
    try:
        with Path(data['log']).open('rb') as stream:
            stream.seek(0,2);stream.seek(max(0,stream.tell()-4096))
            lines=stream.read().decode('utf-8',errors='replace').splitlines()
        latest=next((x for x in reversed(lines) if x.strip()),'initializing')
    except OSError:latest='initializing'
    print('[RUNNING] %s pid=%d | %s'%(data['name'],handle[0].pid,latest[-230:]),flush=True)


def wait_child(args,handle):
    p=handle[0]
    while p.poll() is None:
        progress_message(handle)
        if STOP:p.terminate()
        if shutil.disk_usage(args.output_dir).free<args.disk_reserve_gib*GIB:
            p.terminate();finish_child(handle)
            raise OSError(28,'剩余空间低于预留值；已停止子进程，清理后加--resume恢复最近完整epoch')
        time.sleep(min(args.poll_seconds,3))
    code=finish_child(handle)
    if code:raise RuntimeError('%s退出码%d；日志：%s；保留任务状态，使用--resume继续'%(handle[2]['name'],code,handle[2]['log']))
    if STOP:raise InterruptedError('已停止；用相同命令加--resume恢复')

def run_child(args,command,name):
    h=start_child(args,command,name);wait_child(args,h)

def correct_scope_notes(run_dir):
    """Correct historical explanatory prose, retaining every original override.
    No labels, probabilities, metrics, feature arrays or checkpoint are changed.
    The immutable original engine is retained to preserve numerical equivalence.
    """
    run_dir=Path(run_dir);history_path=run_dir/'same_source_scope_overrides.json'
    history=load_json(history_path) if history_path.exists() else {'version':VERSION,'overrides':[]}
    same_source='All six classes are extracted from CPLX-XNA-all; source matching does not eliminate position effects or the X/Y strand association. This is read-held-out, not batch-held-out.'
    for p in list(run_dir.rglob('data_audit.json'))+list(run_dir.rglob('audit.json'))+list(run_dir.rglob('metrics.json')):
        if 'source_snapshot' in p.parts:continue
        d=load_json(p);changes={}
        if 'source_confound_note' in d:changes['source_confound_note']=same_source
        if d.get('positive_row_proxy_rows')==0 and 'positive_read_identity_note' in d:
            changes['positive_read_identity_note']='All classes use real row-aligned extraction read IDs; no positive row proxies are used.'
        if 'input_notes' in d and d['input_notes'].get('positive_row_proxy_rows',0)==0:
            old=d['input_notes'].get('identity_scope')
            new='Provided real sequencing read IDs for all six classes; windows from each read are retained together in read-cluster analysis.'
            if old!=new:
                history['overrides'].append({'path':str(p.relative_to(run_dir)),'field':'input_notes.identity_scope','original':old,'replacement':new})
                d['input_notes']['identity_scope']=new
        for k,v in changes.items():
            if d.get(k)!=v:
                history['overrides'].append({'path':str(p.relative_to(run_dir)),'field':k,'original':d.get(k),'replacement':v});d[k]=v
        if isinstance(d.get('notes'),list):
            changed=[]
            for i,text in enumerate(d['notes']):
                if isinstance(text,str) and ('E. coli' in text or 'E.coli' in text):
                    history['overrides'].append({'path':str(p.relative_to(run_dir)),'field':'notes[%d]'%i,'original':text,'replacement':same_source});text=same_source
                changed.append(text)
            d['notes']=changed
        write_json(p,d)
    write_json(history_path,history)
    write_json(run_dir/'same_source_scope.json',{'source_dataset':'CPLX-XNA-all','real_read_ids_all_classes':True,
        'physical_run_id':'unknown','note':same_source,'engine_numerical_code_modified':False})

def preflight(args,prepared):
    path=args.output_dir/'preflight'/'preflight.json'
    # The trainer validates every scalar/signal row, creates shared read-only
    # caches and rejects read overlap. No prediction or fitted test statistic.
    run_child(args,child_command(args,prepared,42,'preflight',args.output_dir/'preflight'),'preflight')
    if not path.exists() or load_json(path).get('status')!='complete':raise RuntimeError('预检未完整完成')
    correct_scope_notes(args.output_dir/'preflight')
    audit=load_json(args.output_dir/'preflight'/'data_audit.json')
    if audit.get('status')!='PASS' or any(audit['read_overlap_counts'].values()):raise ValueError('真实read跨集合审计失败')
    if any(v.get('positive_row_proxy_rows',0) for v in audit['split_summary'].values()):raise ValueError('本实验不允许使用阳性行代理')
    print('[PREFLIGHT PASS] real read IDs in all classes; no test inference',flush=True)

def measure_profile(args,prepared):
    directory=args.output_dir/'resource_profiles'/datetime.now().strftime('%Y%m%d_%H%M%S_%f');path=directory/'resource_profile.json'
    run_child(args,child_command(args,prepared,42,'profile',directory),'resource_profile')
    r=load_json(path)
    if r.get('model_kwargs',{}).get('litemamba_variant')!='legacy_gatedconv' or r.get('actual_optimizer_steps',0)<8:
        raise ValueError('资源profile未测满8次真实AdamW更新或模型不匹配')
    correct_scope_notes(directory)
    return r

def memory_snapshot(device):
    mem={}
    for line in Path('/proc/meminfo').read_text().splitlines():
        k,v=line.split(':',1);mem[k]=int(v.split()[0])*1024
    result={'host_available':mem.get('MemAvailable',mem.get('MemFree',0)),'host_total':mem['MemTotal'],'gpu_free':0,'gpu_total':0}
    if device.startswith('cuda'):
        import torch
        free,total=torch.cuda.mem_get_info(torch.device(device));result.update(gpu_free=free,gpu_total=total)
    return result

def admission(args,profile,wanted):
    snap=memory_snapshot(profile['device']);cpu=os.cpu_count() or 1
    cpu_slots=max(1,cpu//max(1,args.cpu_threads+args.workers))
    host_need=max(profile.get('host_admission_estimate_bytes',0),512*1024**2)*args.memory_safety_factor
    host_free=max(0,snap['host_available']-max(args.ram_reserve_gib*GIB,snap['host_total']*args.reserve_fraction))
    n=min(wanted,args.max_parallel,cpu_slots,int(host_free//host_need))
    if profile['device'].startswith('cuda'):
        gpu_need=max(profile.get('gpu_admission_estimate_bytes',0),512*1024**2)*args.memory_safety_factor
        gpu_free=max(0,snap['gpu_free']-max(args.gpu_reserve_gib*GIB,snap['gpu_total']*args.reserve_fraction))
        n=min(n,int(gpu_free//gpu_need))
    else:n=min(n,1)
    report={'created_at':now(),'wanted':wanted,'admitted':n,'snapshot':snap,'profile_estimate_host_bytes':host_need,'cpu_slots':cpu_slots,
        'scope':'pre-launch conservative admission; sampled conditions can change; no physical/logical GPU mapping is guessed'}
    write_json(args.output_dir/'resource_admission.json',report)
    if n<1:raise RuntimeError('当前空闲GPU/RAM不足以按固定设置启动一个作业；稍后用--resume重试，不自动更改batch或科学设置')
    print('[RESOURCE] admitted=%d wanted=%d device=%s'%(n,wanted,profile['device']),flush=True)
    return n

def training_status(run_dir):
    p=Path(run_dir)/'results.json'
    return load_json(p).get('status') if p.exists() else None

def training_action(args,prepared,seed):
    run=args.output_dir/'runs'/('seed_'+str(seed));status=training_status(run)
    if status in ('train_complete','complete'):
        if not args.resume:raise ValueError('seed%d已有完成结果；复用请加--resume'%seed)
        verify_trained_run(run,allow_test=True);print('[REUSE] seed%d %s'%(seed,status),flush=True);return None
    if status=='weights_complete':
        if not args.resume:raise ValueError('已有权重阶段，请加--resume继续导出')
        return None
    cmd=child_command(args,prepared,seed,'train')
    if run.exists() and any(run.iterdir()):
        if not args.resume:raise ValueError('seed%d目录已有产物；用--resume继续'%seed)
        if (run/'test').exists() or (args.output_dir/'test_started.json').exists():raise ValueError('测试开始后不能恢复权重训练')
        if (run/'last_checkpoint.pt').exists():cmd+=['--resume','auto']
        else:
            if any(run.glob('*.pt')):
                raise ValueError('缺少last checkpoint但仍存在其他权重；拒绝冒充首次训练随机重启，请恢复完整检查点')
            # Preserve a failed import/preparation attempt in full. No model has
            # committed an epoch, so an identical seeded restart is appropriate.
            archive=args.output_dir/'failed_before_checkpoint'/('seed_%d_%s'%(seed,datetime.now().strftime('%Y%m%d_%H%M%S_%f')))
            archive.parent.mkdir(exist_ok=True);shutil.move(str(run),str(archive))
            print('[ARCHIVE] no committed epoch; identical restart from random seed: '+str(archive),flush=True)
    return start_child(args,cmd,'weights_seed_'+str(seed))

def verify_trained_run(run,allow_test=False):
    run=Path(run);r=load_json(run/'results.json')
    if r.get('status') not in (('train_complete','complete') if allow_test else ('train_complete',)):
        raise ValueError('运行未完成train/val，或test状态不符合阶段：'+str(run))
    if not allow_test and r.get('test_evaluated'):raise ValueError('冻结前发现test已评价')
    output=run.parent.parent
    plan=load_json(output/'experiment_plan.json')
    cfg=load_json(run/'resolved_config.json')
    expected_seed=int(run.name.removeprefix('seed_'))
    if r.get('seed')!=expected_seed or cfg.get('seed')!=expected_seed:
        raise ValueError('运行seed与固定目录不一致')
    for key,value in plan['scientific_configuration'].items():
        if cfg.get(key)!=value:raise ValueError('已训练运行的科学设置与计划不同：'+key)
    if r.get('litemamba_variant')!='legacy_gatedconv' or r.get('readout')!='target_attention' or r.get('feature_scheme')!='full':
        raise ValueError('已有运行不是最终完整输入GatedConv')
    if cfg.get('model_kwargs',{}).get('target_index')!=3:raise ValueError('运行目标位点不是第四位')
    snapshot=cfg.get('source_snapshot',{}).get('files_sha256',{})
    for name,digest in snapshot.items():
        if plan['source_code_sha256'].get('engine/'+name)!=digest:
            raise ValueError('训练源码快照与冻结引擎不一致：'+name)
    if not {'train_all.py','model_benchmark_all.py','IterDataset.py','config_6_all_6.py'}.issubset(snapshot):
        raise ValueError('缺少完整训练源码快照')
    pm=load_json(output/'prepared_data'/'prepare_manifest.json')
    for split,filename in [('train','train'),('val','validation'),('test','test')]:
        for field,name in [('feature',filename+'.csv'),('metadata',filename+'_meta.tsv')]:
            expected=pm['files'][name]['sha256']
            if cfg['input_fingerprints'][split][field]['sha256']!=expected:
                raise ValueError('已训练输入SHA与本次同来源数据不同：'+split+'/'+field)
            if sha256(output/'prepared_data'/name)!=expected:raise ValueError('准备好的输入文件发生变化：'+name)
    checkpoint=sha256(run/'best_checkpoint.pt');cal=load_json(run/'calibration.json')
    if r['provenance']['checkpoint_sha256']!=checkpoint or cal['checkpoint_sha256']!=checkpoint:raise ValueError('checkpoint/calibration/results SHA不一致')
    for split in ('train','val'):
        marker=load_json(run/split/'export_complete.json')
        if marker['checkpoint_sha256']!=checkpoint or marker['temperature']!=cal['temperature']:raise ValueError('train/val导出模型身份不一致')
        for name in ('predictions.csv.gz','logits.npy','labels.npy','probabilities.npy','calibrated_probabilities.npy','embeddings.npy','attention.npy','prediction_metrics.json','analysis/metrics.json'):
            if not (run/split/name).is_file():raise ValueError('缺少完整导出文件：'+str(run/split/name))
    return {'seed':r['seed'],'best_epoch':r['best_epoch'],'checkpoint_sha256':checkpoint,
        'state_dict_sha256':sha256(run/'best_state_dict.pt'),'calibration_sha256':sha256(run/'calibration.json'),
        'temperature':cal['temperature'],'resolved_config_sha256':sha256(run/'resolved_config.json')}

def train_seeds(args,prepared,seeds,profile):
    if (args.output_dir/'evaluation_plan.json').exists():raise ValueError('三模型已冻结；训练阶段已结束')
    pending=list(seeds);active=[];failure=None
    while pending or active:
        if STOP:raise InterruptedError('收到停止请求')
        if not active and pending:
            slots=admission(args,profile,len(pending))
            for _ in range(slots):
                seed=pending.pop(0);h=training_action(args,prepared,seed)
                if h:active.append((seed,h))
        for seed,h in list(active):
            progress_message(h)
            if h[0].poll() is not None:
                code=finish_child(h);active.remove((seed,h))
                if code:failure=RuntimeError('seed%d训练退出码%d，日志%s；清理/检查后--resume'%(seed,code,h[2]['log']))
            elif shutil.disk_usage(args.output_dir).free<args.disk_reserve_gib*GIB:
                for _,other in active:
                    if other[0].poll() is None:other[0].terminate()
                failure=OSError(28,'空间低于预留值；恢复最近完整epoch需--resume')
        if failure:
            for _,h in active:finish_child(h)
            raise failure
        if active:time.sleep(min(args.poll_seconds,3))
    # Large probability/embedding/statistical exports run one process at a time.
    for seed in seeds:
        run=args.output_dir/'runs'/('seed_'+str(seed))
        if training_status(run)=='weights_complete':run_child(args,child_command(args,prepared,seed,'finalize'),'finalize_seed_'+str(seed))
        correct_scope_notes(run);verify_trained_run(run,allow_test=False)

def freeze_models(args):
    records=[verify_trained_run(args.output_dir/'runs'/('seed_'+str(s)),allow_test=(args.output_dir/'test_started.json').exists()) for s in SEEDS]
    plan={'version':VERSION,'fixed_seeds':list(SEEDS),'models':records,
          'experiment_plan_sha256':sha256(args.output_dir/'experiment_plan.json'),
          'prepared_manifest_sha256':sha256(args.output_dir/'prepared_data'/'prepare_manifest.json'),
          'decision':'six-class argmax; raw and validation-temperature calibrated probabilities',
          'scope':'internal real-read-held-out same-source six-class experiment; no threshold or model tuning on test'}
    p=args.output_dir/'evaluation_plan.json'
    if p.exists():
        old=load_json(p)
        comparable={k:v for k,v in old.items() if k!='frozen_at'}
        if comparable!=plan:raise ValueError('冻结模型/温度/数据已改变；拒绝覆盖evaluation_plan')
    else:
        if (args.output_dir/'test_started.json').exists():raise ValueError('test已开始但冻结清单缺失')
        write_json(p,dict(plan,frozen_at=now()))
    print('[FROZEN] all 3 checkpoints and validation temperatures; test not invoked by freeze',flush=True)

def evaluate_frozen(args,prepared):
    p=args.output_dir/'evaluation_plan.json'
    if not p.exists():raise ValueError('请先完成3种子训练，再显式运行--phase freeze')
    freeze_models(args)
    marker=args.output_dir/'test_started.json';identity=sha256(p)
    if marker.exists():
        if load_json(marker)['evaluation_plan_sha256']!=identity:raise ValueError('test开始后冻结清单改变')
    else:write_json(marker,{'started_at':now(),'evaluation_plan_sha256':identity,'policy':'all three checkpoints fixed before first test inference'})
    for seed in SEEDS:
        run=args.output_dir/'runs'/('seed_'+str(seed))
        run_child(args,child_command(args,prepared,seed,'evaluate'),'evaluate_seed_'+str(seed))
        correct_scope_notes(run)
        r=load_json(run/'results.json')
        if not r.get('test_evaluated') or r.get('status')!='complete':raise RuntimeError('test评价未完成')
    write_json(args.output_dir/'evaluation_complete.json',{'status':'complete','completed_at':now(),'evaluation_plan_sha256':identity,'seeds':list(SEEDS)})

def run(args):
    global LOCK_FD
    args=validate_args(args)
    if args.dry_run:
        print(json.dumps({'phase':args.phase,'seeds':list(SEEDS[:1] if args.phase=='pilot' else SEEDS[1:] if args.phase=='confirm' else SEEDS),
            'scientific_configuration':SCIENTIFIC,'test_inference':args.phase=='evaluate','source_dir':str(args.source_dir),
            'output_dir':str(args.output_dir),'no_data_read_or_process_started':True},ensure_ascii=False,indent=2));return
    args.output_dir.mkdir(parents=True,exist_ok=True)
    with (args.output_dir/'.pipeline.lock').open('a') as lock:
        try:fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
        except BlockingIOError:raise RuntimeError('同一输出目录已有任务运行')
        LOCK_FD=lock.fileno()
        orphan_check(args)
        if args.phase=='summary':
            from summarize_same_source import summarize_and_package
            result=summarize_and_package(args.output_dir,HERE);print('[PACKAGED] '+str(result['core_path']),flush=True);return
        from prepare_data import prepare_inputs
        wanted=1 if args.phase=='pilot' else 2 if args.phase=='confirm' else 0
        check_disk(args,wanted,include_prepare=not (args.output_dir/'prepared_data').exists())
        prepared=prepare_inputs(args.source_dir,args.output_dir,resume=args.resume,
            disk_reserve_gib=args.disk_reserve_gib,reference_dir=REFERENCE_DIR)
        commit_plan(args,prepared)
        print('[VERSION] CPLX same-source GatedConv '+VERSION+' phase='+args.phase,flush=True)
        if args.phase in ('prepare','pilot','confirm'):
            preflight(args,prepared)
            if args.phase!='prepare':
                if args.phase=='confirm':verify_trained_run(args.output_dir/'runs'/'seed_42',allow_test=False)
                profile=measure_profile(args,prepared)
                train_seeds(args,prepared,SEEDS[:1] if args.phase=='pilot' else SEEDS[1:],profile)
        elif args.phase=='freeze':freeze_models(args)
        elif args.phase=='evaluate':evaluate_frozen(args,prepared)
        write_json(args.output_dir/'stage_status.json',{'phase':args.phase,'status':'complete','finished_at':now(),
            'test_evaluated':(args.output_dir/'evaluation_complete.json').exists()})
        from summarize_same_source import summarize_and_package
        result=summarize_and_package(args.output_dir,HERE)
        print('[PACKAGED] '+str(result['core_path']),flush=True)
        print('[FULL PASS] phase=%s output=%s'%(args.phase,args.output_dir),flush=True)
        if args.phase=='pilot':print('[NEXT] seed42 train/validation complete; test has not been inferred. Upload the pilot core for review.',flush=True)

def main(argv=None):
    for s in (signal.SIGINT,signal.SIGTERM):signal.signal(s,request_stop)
    return run(parser().parse_args(argv))

if __name__=='__main__':
    try:main()
    except SystemExit:
        raise
    except BaseException as e:
        for p in list(CHILDREN.values()):
            if p.poll() is None:p.terminate()
        for p in list(CHILDREN.values()):
            try:p.wait(timeout=30)
            except subprocess.TimeoutExpired:p.kill();p.wait()
        print('[ERROR] %s: %s'%(type(e).__name__,e),file=sys.stderr,flush=True)
        sys.exit(2 if isinstance(e,(InterruptedError,KeyboardInterrupt)) else 1)
