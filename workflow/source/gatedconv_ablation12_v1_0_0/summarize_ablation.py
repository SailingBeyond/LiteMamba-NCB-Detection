#!/usr/bin/env python3
"""Summarize all twelve frozen GatedConv ablations; never load or fit a model."""
from __future__ import annotations
import argparse,csv,hashlib,html,io,json,math,os,shutil,statistics,tarfile,tempfile
from datetime import datetime,timezone
from pathlib import Path

HERE=Path(__file__).resolve().parent
REFERENCE_FILE=HERE/'reference/experiment_reference.json'
VERSION='1.0.0'
SCHEMES=('no_raw','no_context','no_responses','no_statistics')
SEEDS=(42,2024,2025)
CLASS_NAMES=('A','T','C','G','X','Y')
PACKAGE_NAME='gatedconv_ablation12_results_core_v1_0_0.tar.gz'
SUMMARY_NAME='ablation_summary'
ARCHITECTURE={'model_type':'litemamba','litemamba_variant':'legacy_gatedconv','readout':'target_attention','model_dim':96,'num_blocks':4,'state_dim':16,'target_index':3}
SCIENCE=('epochs','patience','lr','batch_size','raw_signal_len','dwell_divisor','quality_divisor','accum_steps','weight_decay','label_smoothing','clip_grad','monitor')
METRICS=('accuracy','macro_f1','balanced_accuracy','X_recall','X_f1','Y_recall','Y_f1','canonical_false_positive_rate','target_recall_micro','ncb_recall','ncb_precision','xy_swap_rate','xy_to_canonical_rate','nll','brier','ece_15bins')
LOWER={'canonical_false_positive_rate','xy_swap_rate','xy_to_canonical_rate','nll','brier','ece_15bins'}
REQUIRED_RUN=('queue_requested_config.json','resolved_config.json','model_config.json','results.json','calibration.json','epoch_metrics.jsonl','source_snapshot/manifest.json','environment.json','data_audit.json')
CORE_ANALYSIS={'metrics.json','confusion_matrix.csv','per_class_metrics.csv','ncb_calls.csv','confidence_threshold_scan.csv','confidence_per_class.csv','class_threshold_scan.csv','prevalence_scenarios.csv','calibration_bins.csv','bootstrap_replicates.csv','probability_histogram.csv','metadata_coverage.csv','per_flank_gc_count.csv','per_motif.csv','worst_motifs.csv'}
GROUP_CORE_PREFIX=('per_chrom','per_reference','per_template_group','per_source','per_run','per_bam','per_strand','per_file_strand')


def read_json(path,required=True):
 path=Path(path)
 if not path.is_file():
  if required:raise ValueError('缺少文件：'+str(path))
  return None
 value=json.loads(path.read_text(encoding='utf-8'))
 if not isinstance(value,dict):raise ValueError('JSON须为对象：'+str(path))
 return value

def clean(v):
 if isinstance(v,float) and not math.isfinite(v):return None
 if isinstance(v,dict):return {str(k):clean(x) for k,x in v.items()}
 if isinstance(v,(list,tuple)):return [clean(x) for x in v]
 return v

def write_json(path,value):
 path=Path(path);path.parent.mkdir(parents=True,exist_ok=True);tmp=path.with_name(path.name+'.tmp-'+str(os.getpid()))
 tmp.write_text(json.dumps(clean(value),ensure_ascii=False,indent=2,allow_nan=False)+'\n',encoding='utf-8');tmp.replace(path)

def write_csv(path,rows,fields=None):
 path=Path(path);path.parent.mkdir(parents=True,exist_ok=True)
 if fields is None:fields=list(dict.fromkeys(k for row in rows for k in row))
 tmp=path.with_name(path.name+'.tmp-'+str(os.getpid()))
 with tmp.open('w',encoding='utf-8-sig',newline='') as f:
  writer=csv.DictWriter(f,fieldnames=fields,extrasaction='ignore');writer.writeheader();writer.writerows(clean(rows))
 tmp.replace(path)

def fingerprint(path):
 p=Path(path);h=hashlib.sha256();before=p.stat()
 with p.open('rb') as f:
  for block in iter(lambda:f.read(8*1024*1024),b''):h.update(block)
 after=p.stat()
 if (before.st_size,before.st_mtime_ns)!=(after.st_size,after.st_mtime_ns):raise ValueError('读取期间文件改变：'+str(p))
 return {'sha256':h.hexdigest(),'size_bytes':after.st_size,'mtime_ns':after.st_mtime_ns}

def option(c,k):return c.get(k,c.get('model_kwargs',{}).get(k))
def classes(x):return tuple(x.get(str(i),x.get(i)) for i in range(6)) if isinstance(x,dict) else tuple(x or ())
def require(ok,message):
 if not ok:raise ValueError(message)

def input_identity(config):
 out={}
 for split in ('train','val','test'):
  out[split]={}
  for kind in ('feature','metadata'):
   x=config['input_fingerprints'][split][kind];size=x.get('size',x.get('size_bytes'))
   require(bool(x.get('sha256')) and type(size) is int and size>=0,'输入缺少SHA/size')
   out[split][kind]={'sha256':x['sha256'],'size':size}
 return out

def check_config(config,scheme,seed,reference):
 for k,v in ARCHITECTURE.items():require(option(config,k)==v,'最终GatedConv结构不一致：'+k)
 require(option(config,'feature_scheme')==scheme and config.get('seed')==seed,'输入方案或种子不同')
 require(classes(config.get('class_names'))==CLASS_NAMES,'类别顺序须为A/T/C/G/X/Y')
 kwargs=config.get('model_kwargs',{})
 require(kwargs.get('feature_scheme')==scheme and kwargs.get('litemamba_variant')=='legacy_gatedconv' and kwargs.get('readout')=='target_attention' and kwargs.get('target_index')==3,'实际model_kwargs屏蔽/读出不同')
 expected=input_identity({'input_fingerprints':reference['input_fingerprints']})
 require(input_identity(config)==expected,'训练/验证/测试内容身份不同')
 for key in SCIENCE:require(option(config,key)==reference['scientific_configuration'][key],'训练/预处理协议不同：'+key)
 require(not config.get('external_manifest') and config.get('checkpoint') is None,'不得外部选模或从full权重初始化')
 source=config.get('source_snapshot',{}).get('files_sha256',{})
 for name,digest in reference['frozen_sources'].items():require(source.get(name)==digest,'实际模型/数据源码不同：'+name)


def check_requested(config,scheme,seed,reference):
 require((config.get('model_type'),config.get('feature_scheme'),config.get('seed'))==('litemamba',scheme,seed),'请求任务身份不同')
 for k,v in ARCHITECTURE.items():
  if k=='target_index' and k not in config:continue
  require(option(config,k)==v,'请求最终GatedConv结构不同：'+k)
 require(input_identity(config)==input_identity({'input_fingerprints':reference['input_fingerprints']}),'请求数据内容不同')
 for key in SCIENCE:require(option(config,key)==reference['scientific_configuration'][key],'请求训练协议不同：'+key)
 source=config.get('source_sha256',config.get('source_snapshot',{}).get('files_sha256',{}))
 for name,digest in reference['frozen_sources'].items():require(source.get(name)==digest,'请求模型/数据源码不同：'+name)
 require(not config.get('external_manifest') and config.get('checkpoint') is None,'请求不得外部选模或full初始化')


def ratio(a,b):return a/b if b else None

def count_metrics(saved):
 cm=saved.get('confusion_matrix');require(isinstance(cm,list) and len(cm)==6 and all(isinstance(row,list) and len(row)==6 for row in cm),'需要6×6混淆矩阵')
 require(all(type(x) is int and x>=0 for row in cm for x in row),'混淆矩阵须为非负整数')
 support=[sum(row) for row in cm];pred=[sum(row[k] for row in cm) for k in range(6)];n=sum(support);require(n>0,'混淆矩阵为空')
 require(saved.get('n',n)==n,'样本数与混淆计数不符')
 f1=[ratio(2*cm[k][k],support[k]+pred[k]) for k in range(6)];rec=[ratio(cm[k][k],support[k]) for k in range(6)]
 can=sum(support[:4]);xy=sum(support[4:]);det=sum(sum(row[4:]) for row in cm[4:]);false=sum(sum(row[4:]) for row in cm[:4])
 vals={'accuracy':sum(cm[k][k] for k in range(6))/n,'macro_f1':sum(x or 0 for x in f1)/6,'balanced_accuracy':statistics.mean(x for x in rec if x is not None),'X_recall':rec[4],'X_f1':f1[4],'Y_recall':rec[5],'Y_f1':f1[5],'canonical_false_positive_rate':ratio(false,can),'target_recall_micro':ratio(cm[4][4]+cm[5][5],xy),'ncb_recall':ratio(det,xy),'ncb_precision':ratio(det,det+false),'xy_swap_rate':ratio(cm[4][5]+cm[5][4],xy),'xy_to_canonical_rate':ratio(sum(sum(row[:4]) for row in cm[4:]),xy)}
 for k in ('accuracy','macro_f1','balanced_accuracy'):
  if saved.get(k) is not None:require(math.isclose(float(saved[k]),vals[k],rel_tol=1e-9,abs_tol=1e-10),'保存指标与整数混淆矩阵不符：'+k)
 for k in ('nll','brier','ece_15bins'):
  x=saved.get(k);require(isinstance(x,(int,float)) and math.isfinite(x),'缺少有限概率指标：'+k);vals[k]=float(x)
 cls=[]
 for k,name in enumerate(CLASS_NAMES):
  tp=cm[k][k];fp=pred[k]-tp;fn=support[k]-tp;tn=n-tp-fp-fn
  cls.append({'class_id':k,'class_name':name,'support':support[k],'predicted_support':pred[k],'tp':tp,'fp':fp,'fn':fn,'tn':tn,'precision':ratio(tp,pred[k]),'recall':rec[k],'specificity':ratio(tn,tn+fp),'fpr':ratio(fp,tn+fp),'fdr':ratio(fp,tp+fp),'f1':f1[k]})
 return vals,cls,cm,n


def metric_versions(path,result=None,split=None):
 d=read_json(path)
 versions=d.get('overall',d)
 require(all(k in versions and isinstance(versions[k],dict) for k in ('raw','calibrated')),'缺少raw/calibrated指标：'+str(path))
 if result and split:
  embedded=result.get('metrics_by_split',{}).get(split)
  if embedded:
   for v in ('raw','calibrated'):
    require(embedded.get(v,{}).get('confusion_matrix')==versions[v].get('confusion_matrix'),'results与逐集合指标混淆计数不同')
 return versions


def audit_new_run(run,scheme,seed,reference,freeze):
 config=read_json(run/'resolved_config.json');result=read_json(run/'results.json')
 for rel in REQUIRED_RUN:read_json(run/rel) if rel.endswith('.json') else require((run/rel).is_file(),'缺少训练历史')
 check_config(config,scheme,seed,reference);check_requested(read_json(run/'queue_requested_config.json'),scheme,seed,reference)
 mc=read_json(run/'model_config.json')
 for k,v in [('model_type','litemamba'),('feature_scheme',scheme),('litemamba_variant','legacy_gatedconv'),('readout','target_attention'),('target_index',3),('seq_len',6),('num_classes',6)]:require(mc.get(k)==v,'model_config不同：'+k)
 require(classes(mc.get('class_names'))==CLASS_NAMES,'model_config类别顺序不同')
 require(mc.get('one_hot_base_order')==['A','C','T','G','N'],'model_config one-hot顺序不同')
 for k in ('model_dim','num_blocks','state_dim'):
  require(mc.get(k)==ARCHITECTURE[k],'model_config结构不同：'+k)
 require(result.get('status')=='complete' and result.get('test_evaluated') is True,'训练/内部测试尚未完成')
 require((result.get('model_type'),result.get('feature_scheme'),result.get('seed'))==('litemamba',scheme,seed),'results任务身份不同')
 ck=result.get('provenance',{}).get('checkpoint_sha256');require(bool(ck),'结果缺少检查点SHA')
 rid=f'{scheme}/litemamba/seed_{seed}';require(freeze.get(rid,{}).get('checkpoint_sha256')==ck,'内部测试前冻结检查点SHA不同')
 calibration=read_json(run/'calibration.json');t=calibration.get('temperature')
 require(calibration.get('fitted_on')=='val' and isinstance(t,(int,float)) and math.isfinite(t) and t>0 and calibration.get('checkpoint_sha256')==ck,'温度不是本检查点的validation-only校准')
 if 'n_samples' in calibration:require(calibration['n_samples']==reference['input_rows']['val'],'温度拟合样本数不是validation数量')
 history=[json.loads(line) for line in (run/'epoch_metrics.jsonl').read_text().splitlines() if line.strip()]
 require(history and [x['epoch'] for x in history]==list(range(1,len(history)+1)),'训练历史缺失或不连续')
 require(all(x.get('monitor','val_loss')=='val_loss' and math.isfinite(float(x['val']['loss'])) for x in history),'选模规则不是有限val_loss')
 best=min(history,key=lambda x:x['val']['loss']);require(best['epoch']==result.get('best_epoch'),'检查点不属于val_loss最优轮次')
 versions={};ident=input_identity(config)
 for split in ('train','val','test'):
  ep=read_json(run/split/'export_complete.json');require(ep.get('checkpoint_sha256')==ck,'导出检查点不同：'+split)
  require(ep.get('dataset_identity',{}).get('input_fingerprints')==ident[split],'导出数据身份不同：'+split)
  prep=ep.get('dataset_identity',{}).get('preprocessing',{});require(prep.get('mask_index')==3 and prep.get('mask_base')=='N','导出未证明第四位N：'+split)
  if 'temperature' in ep:require(ep['temperature']==t,'导出温度与validation不同：'+split)
  versions[split]=metric_versions(run/split/'prediction_metrics.json',result,split)
  for variant in ('raw','calibrated'):
   _,_,_,n=count_metrics(versions[split][variant]);require(n==reference['input_rows'][split],'导出窗口数量不同：'+split)
  require(versions[split]['raw']['confusion_matrix']==versions[split]['calibrated']['confusion_matrix'],'正温度改变了分类混淆计数')
  if split in ('val','test'):
   analysis=metric_versions(run/split/'analysis/metrics.json')
   for variant in ('raw','calibrated'):require(analysis[variant]['confusion_matrix']==versions[split][variant]['confusion_matrix'],'逐窗口分析与导出混淆矩阵不同')
 require(math.isclose(best['val']['macro_f1'],versions['val']['raw']['macro_f1'],rel_tol=1e-9,abs_tol=1e-9),'最佳轮次验证F1与导出不同')
 return {'scheme':scheme,'seed':seed,'run_dir':str(run),'config':config,'result':result,'checkpoint_sha256':ck,'temperature':t,'selected_epoch':best['epoch'],'history':history,'versions':versions,'origin':'new_ablation'}


def queue_freeze(output):
 expected={f'{s}/litemamba/seed_{seed}' for s in SCHEMES for seed in SEEDS}
 values={}
 for name in ('queue_plan.json','queue_manifest.json','evaluation_plan.json'):
  x=read_json(output/name);entries=x.get('experiments',[])
  require(len(entries)==12 and {r.get('run_id') for r in entries}==expected,'整队12项计划/冻结记录缺失或重复：'+name)
  values[name]=x
 q=values['queue_manifest.json'];require(q.get('status')=='complete' and not q.get('failed_runs') and q.get('policy')=='train_all_defer_test_then_freeze_all_then_evaluate','整队尚未完成或训练/冻结/测试顺序不同')
 f=values['evaluation_plan.json'];require(f.get('policy')=='all_predeclared_training_succeeded_before_any_test;validation_only_selection','冻结计划未证明validation-only选模')
 return {x['run_id']:x for x in f['experiments']}


def _rows(record):
 rows=[];classes_out=[];confusions=[]
 for split,versions in record['versions'].items():
  for variant in ('raw','calibrated'):
   vals,cl,cm,n=count_metrics(versions[variant]);base={'scheme':record['scheme'],'seed':record['seed'],'split':split,'probability_type':variant,'origin':record['origin'],'checkpoint_sha256':record['checkpoint_sha256'],'n':n}
   rows.extend({**base,'metric':m,'value':vals[m],'unit':'metric_units' if m in ('nll','brier') else 'fraction','preferred_direction':'lower' if m in LOWER else 'higher'} for m in METRICS)
   classes_out.extend({**base,**c} for c in cl)
   confusions.extend({**base,'true_label':i,'predicted_label':j,'count':cm[i][j]} for i in range(6) for j in range(6))
 return rows,classes_out,confusions


def aggregate(rows,keys,value='value'):
 groups={}
 for row in rows:groups.setdefault(tuple(row[k] for k in keys),{})[row['seed']]=row[value]
 out=[]
 for key,by_seed in sorted(groups.items()):
  vals=[by_seed[s] for s in SEEDS if by_seed.get(s) is not None]
  out.append({**dict(zip(keys,key)),'expected_seeds':3,'n_defined_seeds':len(vals),'mean':statistics.mean(vals) if vals else None,'sample_sd':statistics.stdev(vals) if len(vals)>1 else None})
 return out


def paired_rows(rows):
 baseline={(r['seed'],r['split'],r['probability_type'],r['metric']):r['value'] for r in rows if r['scheme']=='full'};pairs=[]
 for r in rows:
  if r['scheme']=='full' or r['split']=='train':continue
  full=baseline.get((r['seed'],r['split'],r['probability_type'],r['metric']));value=r['value'];delta=full-value if full is not None and value is not None else None
  scale=1 if r['metric'] in ('nll','brier') else 100
  pairs.append({**r,'full_value':full,'ablation_value':value,'full_minus_ablation':delta,'difference':delta*scale if delta is not None else None,'difference_unit':'metric units' if scale==1 else 'percentage points','full_better':None if delta is None else delta<0 if r['metric'] in LOWER else delta>0})
 keys=['scheme','split','probability_type','metric'];agg=aggregate(pairs,keys,'difference')
 for row in agg:
  group=[r for r in pairs if all(r[k]==row[k] for k in keys)]
  row.update(n_pairs=len(group),n_full_better=sum(x['full_better'] is True for x in group),n_equal=sum(x['full_minus_ablation']==0 for x in group),difference_unit=group[0]['difference_unit'],preferred_direction=group[0]['preferred_direction'])
 return pairs,agg


def learning_history(records):
 epochs=[];diagnostics=[]
 for r in records:
  if r['scheme']=='full':continue
  attempt=success=skips=nonfinite=0;known=False
  for h in r['history']:
   tr=h.get('train',{});val=h.get('val',{});row={'scheme':r['scheme'],'seed':r['seed'],'epoch':h['epoch'],'train_loss':tr.get('loss'),'val_loss':val.get('loss'),'train_accuracy':tr.get('accuracy'),'val_accuracy':val.get('accuracy'),'train_macro_f1':tr.get('macro_f1'),'val_macro_f1':val.get('macro_f1'),'lr':h.get('lr'),'monitor':h.get('monitor','val_loss')}
   for k in ('optimizer_step_attempts','optimizer_steps_successful','optimizer_steps_skipped_amp','optimizer_update_count_method','grad_norm_count','grad_norm_finite_count','grad_norm_nonfinite_count','mean_grad_norm_finite','max_grad_norm_finite','amp_scale_start','amp_scale_end','amp_scale_min','amp_scale_max'):row[k]=tr.get(k)
   epochs.append(row)
   if tr.get('optimizer_step_attempts') is not None:
    known=True;attempt+=tr['optimizer_step_attempts'];success+=tr.get('optimizer_steps_successful',0);skips+=tr.get('optimizer_steps_skipped_amp',0);nonfinite+=tr.get('grad_norm_nonfinite_count',0)
  diagnostics.append({'scheme':r['scheme'],'seed':r['seed'],'epochs':len(r['history']),'selected_epoch':r['selected_epoch'],'diagnostics_available':known,'optimizer_step_attempts':attempt if known else None,'optimizer_steps_successful':success if known else None,'optimizer_steps_skipped_amp':skips if known else None,'grad_norm_nonfinite_count':nonfinite if known else None,'skipped_attempt_fraction':ratio(skips,attempt) if known else None,'count_method':'GradScaler scale backoff inference when AMP enabled; not optimizer hooks','all_train_val_losses_finite':all(math.isfinite(float(h[k]['loss'])) for h in r['history'] for k in ('train','val'))})
 return epochs,diagnostics


def make_report(records,aggregates,pairs,statuses):
 def table(headers,rows):return '<table><tr>'+''.join('<th>'+html.escape(x)+'</th>' for x in headers)+'</tr>'+''.join('<tr>'+''.join('<td>'+html.escape(str(x))+'</td>' for x in row)+'</tr>' for row in rows)+'</table>'
 def number(x):return '—' if x is None else f'{100*x:.3f}'
 selected=[]
 for s in ('full',)+SCHEMES:
  row=[s]
  for m in ('accuracy','macro_f1','target_recall_micro','canonical_false_positive_rate'):
   item=next((x for x in aggregates if x['scheme']==s and x['split']=='test' and x['probability_type']=='raw' and x['metric']==m),None)
   row.append('—' if item is None else number(item['mean'])+' ± '+number(item['sample_sd']))
  selected.append(row)
 body='''<!doctype html><html lang="zh-CN"><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1"><title>GatedConv 去除组消融</title><style>body{font:16px/1.7 system-ui,sans-serif;color:#293e36;max-width:1250px;margin:28px auto;padding:0 24px}h1,h2{color:#235d52}table{border-collapse:collapse;font-size:14px;width:100%;margin:20px 0}td,th{padding:8px;border-bottom:1px solid #dce5dd;text-align:right}td:first-child,th:first-child{text-align:left}th{background:#edf4ef}.note{padding:16px;background:#fbf3e7}img{width:100%;height:auto}</style><h1>最终 GatedConv：四个去除组 × 三种子</h1><p>新增no_raw、no_context、no_responses、no_statistics各3次，从各自种子的随机初始化训练。复用固定full GatedConv三个种子；没有从full权重热启动。内部test仅在12项训练和train/val导出完成后冻结检查点再评价。</p><h2>固定内部测试：三种子均值 ± 样本SD（%）</h2>TABLE<p class="note">完整输入−去除组为同种子配对差值。比例差值用百分点，NLL/Brier用原指标单位；误报、互换率和损失越低越好。均值或SD不是样本级显著性检验。undefined分母保留为空，不替换为0。Raw/calibrated是同一模型的两种概率，不是独立重复。</p><h2>任务核验</h2>STATUS<h2>解释范围</h2><p>去除组实验检验其他特征仍然存在时，某一输入组的增量作用；单输入对照检验仅该组能做到什么程度，两者不能替代。raw片段可能含幅值、持续时间、补零及来源相关信息，结果不能单独排除来源、模板或链方向混杂。state_dim为兼容字段；实际架构是门控卷积，没有选择性状态空间递推。</p><p>本包只训练与评估固定内部数据，尚未自动执行Figure4外部预测。若某去除组效果更好，仍完整保存并如实比较，不据内部test再改配置。本次属于在既有已观察内部测试集上的补充固定评估，不额外声明为新获盲测。</p><h2>保存与重画</h2><p>metrics_per_seed.csv和class_metrics.csv保存可复算指标；mean_sd.csv保存三种子均值与SD；paired_full_minus_ablation.csv、paired_mean_sd.csv保存Full减去除组配对差值；confusion_counts.csv为整数计数；learning_curves.csv和numerical_diagnostics.csv保留每轮曲线与数值记录。完整预测、概率、embedding、attention及权重留在服务器。</p><p>核心结果包首先保证全部12新运行及3个full对照元数据完整，再按单独预算收录motif明细；不会截断最后一个运行。ROC/PR完整曲线、per-read大表和预测大数组留在服务器，不需要重新训练或推理即可后续作图。</p></html>'''
 return body.replace('TABLE',table(['方案','准确率','Macro-F1','正确X/Y身份召回','天然→X/Y误报'],selected)).replace('STATUS',table(['输入组','种子','状态','原因'],[(x['scheme'],x['seed'],x['status'],'; '.join(x['issues'])) for x in statuses]))


def _baseline_records(reference,baseline_dir,baseline_test_dir):
 from ablation_baselines import verify_baselines
 raw=verify_baselines(reference,baseline_dir,baseline_test_dir,require_checkpoints=False)
 records=[]
 require(len(raw)==3 and {x['seed'] for x in raw}==set(SEEDS),'需要3个可比的完整输入对照')
 for r in raw:
  versions={}
  for split in ('val','test'):
   data=r[split+'_metrics'];versions[split]=data.get('overall',data)
   require(all(k in versions[split] for k in ('raw','calibrated')),'full缺少原始和校准指标')
   for variant in ('raw','calibrated'):count_metrics(versions[split][variant])
  records.append({**r,'scheme':'full','versions':versions,'origin':'reused_full','history':[],'result':read_json(Path(r['run_dir'])/'results.json')})
 return records


def summarize(output_dir,baseline_dir=None,baseline_test_dir=None,package=False):
 root=Path(output_dir).expanduser().resolve();reference=read_json(REFERENCE_FILE)
 if baseline_dir is None:baseline_dir=root.parent/'litemamba_structure_confirm3_v1_1_0'
 if baseline_test_dir is None:baseline_test_dir=root.parent/'gatedconv_internal_test_v1_0_0'
 target=root/SUMMARY_NAME;target.mkdir(parents=True,exist_ok=True)
 statuses=[];records=[];issues=[]
 try:freeze=queue_freeze(root)
 except (ValueError,KeyError,OSError,TypeError) as exc:freeze={};issues.append(str(exc))
 for scheme in SCHEMES:
  for seed in SEEDS:
   run=root/scheme/'litemamba'/f'seed_{seed}';status={'scheme':scheme,'seed':seed,'run_dir':str(run),'status':'invalid','issues':[]}
   try:
    record=audit_new_run(run,scheme,seed,reference,freeze);records.append(record);status.update(status='complete',checkpoint_sha256=record['checkpoint_sha256'],selected_epoch=record['selected_epoch'])
   except (ValueError,KeyError,OSError,TypeError,OverflowError) as exc:status['issues'].append(str(exc))
   statuses.append(status)
 baselines=[]
 try:baselines=_baseline_records(reference,baseline_dir,baseline_test_dir);records.extend(baselines)
 except (ValueError,KeyError,OSError,TypeError) as exc:issues.append('Full对照：'+str(exc))
 done=sum(x['status']=='complete' for x in statuses);complete=done==12 and len(baselines)==3 and not issues
 rows=[];classrows=[];cms=[]
 for record in records:
  rr,cc,mm=_rows(record);rows+=rr;classrows+=cc;cms+=mm
 agg=aggregate(rows,['scheme','split','probability_type','metric']);pairs,pagg=paired_rows(rows);curves,diag=learning_history(records)
 runtime_rows=[{'scheme':r['scheme'],'seed':r['seed'],'origin':r['origin'],**r.get('result',{}).get('runtime',{})} for r in records]
 # per-class rates in a long form; no ambiguous multi-row CSV headers.
 cr=[]
 for r in classrows:
  for m in ('precision','recall','specificity','fpr','fdr','f1'):cr.append({**{k:r[k] for k in ('scheme','seed','split','probability_type','class_name')},'metric':m,'value':r[m]})
 cagg=aggregate(cr,['scheme','split','probability_type','class_name','metric'])
 summary={'version':VERSION,'status':'complete' if complete else 'incomplete','created_at':datetime.now(timezone.utc).isoformat(),'fixed_seeds':list(SEEDS),'feature_schemes':list(SCHEMES),'expected_new_training_runs':12,'new_training_runs':12,'completed_new_training_runs':done,'reused_full_runs':len(baselines),'all_formal_tasks_reported':len(statuses)==12,'scope':'leave-one-feature-group-out controls, frozen final GatedConv; internal supplementary evaluation; Figure4 not automatically run','runs':statuses,'issues':issues,'full_baseline':[{'seed':r['seed'],'status':'comparable','run_dir':str(r['run_dir']),'test_run_dir':str(r['test_run_dir']),'checkpoint_sha256':r['checkpoint_sha256'],'temperature':r['temperature']} for r in baselines],'mean_sd':agg,'paired_mean_sd':pagg,'missing_values_policy':'undefined or unavailable remains null; macro-F1 fixed6 absent contribution is zero','uncertainty':'sample SD of three training seeds, not a confidence interval or a significance test','baseline_dir':str(Path(baseline_dir).resolve()),'baseline_test_dir':str(Path(baseline_test_dir).resolve())}
 write_json(target/'summary.json',summary)
 for name,data in [('run_status.csv',statuses),('metrics_per_seed.csv',rows),('mean_sd.csv',agg),('class_metrics.csv',classrows),('class_mean_sd.csv',cagg),('confusion_counts.csv',cms),('paired_full_minus_ablation.csv',pairs),('paired_mean_sd.csv',pagg),('learning_curves.csv',curves),('numerical_diagnostics.csv',diag),('runtime_per_seed.csv',runtime_rows)]:write_csv(target/name,data)
 (target/'analysis_report.html').write_text(make_report(records,agg,pagg,statuses),encoding='utf-8')
 if package:
  require(complete,'12项训练/测试或3个Full对照不完整，拒绝发布完整核心包')
  summary['result_package']=str(make_core_archive(root,baseline_dir,baseline_test_dir,_checked=(summary,records)))
 return summary


def make_core_archive(output_dir,baseline_dir=None,baseline_test_dir=None,optional_budget_bytes=32*1024*1024,essential_limit_bytes=384*1024*1024,_checked=None):
 root=Path(output_dir).expanduser().resolve()
 if _checked is None:
  summary=summarize(root,baseline_dir,baseline_test_dir,package=False)
  require(summary['status']=='complete','缺少完整12项训练/测试或3个Full对照；不会发布部分结果包')
  reference=read_json(REFERENCE_FILE);freeze=queue_freeze(root)
  records=[audit_new_run(root/s/'litemamba'/f'seed_{seed}',s,seed,reference,freeze) for s in SCHEMES for seed in SEEDS]
  records+=_baseline_records(reference,summary['baseline_dir'],summary['baseline_test_dir'])
 else:summary,records=_checked
 require(len(records)==15,'核心包须包括12新运行和3个Full对照')
 selected={};optional=[];omitted=[];source_index=[]
 def add(path,name,required=False):
  path=Path(path)
  if not path.is_file():
   if required:raise ValueError('核心文件缺失：'+str(path))
   return
  require(not Path(name).is_absolute() and '..' not in Path(name).parts,'非法归档路径')
  if name in selected:
   previous=fingerprint(selected[name]);current=fingerprint(path)
   require(previous['sha256']==current['sha256'] and previous['size_bytes']==current['size_bytes'],'归档路径内容冲突：'+name)
   return
  selected[name]=path
 for name in ('ablation_plan.json','ablation_status.json','queue_plan.json','queue_manifest.json','evaluation_plan.json'):
  add(root/name,'meta/'+name,True)
 for name in ('launch_checks.json','active_queue.json','resource_runtime_state.json','commands.jsonl','resource_events.jsonl','baseline_preflight.json'):
  add(root/name,'meta/'+name)
 for folder in ('resource_profiles','preflight'):
  for path in sorted((root/folder).rglob('*')):
   if path.is_file() and path.suffix in ('.json','.jsonl'):
    add(path,'resource_records/'+path.relative_to(root).as_posix())
 for path in sorted((root/SUMMARY_NAME).rglob('*')):
  if path.is_file() and path.suffix.lower() in ('.json','.csv','.html','.png','.svg'):add(path,'summary/'+path.relative_to(root/SUMMARY_NAME).as_posix())
 for record in records:
  scheme,seed=record['scheme'],record['seed'];directory=Path(record['run_dir']);prefix=('baseline' if scheme=='full' else 'runs/'+scheme)+f'/seed_{seed}'
  required=REQUIRED_RUN if scheme!='full' else ('resolved_config.json','model_config.json','calibration.json','results.json','epoch_metrics.jsonl','source_snapshot/manifest.json')
  for name in required:add(directory/name,prefix+'/'+name,True)
  for name in ('queue_requested_config.json','environment.json','data_audit.json','architecture.txt','metrics.csv','history.csv'):add(directory/name,prefix+'/'+name)
  source_files=[]
  for source in sorted((directory/'source_snapshot').rglob('*.py')):
   info=fingerprint(source);name='shared_source/'+info['sha256']+'/'+source.name;add(source,name);source_files.append({'original_relative':source.relative_to(directory).as_posix(),'package_path':name,'sha256':info['sha256'],'size_bytes':info['size_bytes']})
  source_index.append({'scheme':scheme,'seed':seed,'files':source_files})
  splits=('train','val','test') if scheme!='full' else ('val',)
  for split in splits:
   source=directory/split
   add(source/'export_complete.json',prefix+'/'+split+'/export_complete.json',True)
   add(source/'prediction_metrics.json',prefix+'/'+split+'/prediction_metrics.json',True)
   for path in sorted((source/'analysis').glob('*')):
    if not path.is_file():continue
    name=prefix+'/'+split+'/analysis/'+path.name
    if path.name in CORE_ANALYSIS or path.name.startswith(GROUP_CORE_PREFIX):add(path,name)
    elif 'motif' in path.name and path.suffix in ('.csv','.json'):optional.append((path,name))
    elif path.suffix in ('.csv','.gz','.npy','.npz'):omitted.append({'path':name,'size_bytes':path.stat().st_size,'reason':'large curve/read/prediction detail retained on server; not required for core coverage'})
   if split in ('val','test'):
    for name in ('metrics.json','confusion_matrix.csv','per_class_metrics.csv'):add(source/'analysis'/name,prefix+'/'+split+'/analysis/'+name,True)
  if scheme=='full':
   side=Path(record['test_run_dir']);bp=f'baseline_internal_test/seed_{seed}'
   add(side/'run_complete.json',bp+'/run_complete.json',True)
   add(side/'test/prediction_metrics.json',bp+'/test/prediction_metrics.json',True)
   # Some internal-test wrappers export to test/; preserve the verified supplied paths directly.
   for field,label in [('test_metrics_path','test/analysis/metrics.json'),('test_export_path','test/export_complete.json')]:
    add(Path(record[field]),bp+'/'+label,True)
   for path in sorted((Path(record['test_metrics_path']).parent).glob('*')):
    if path.is_file() and path.name in CORE_ANALYSIS:add(path,bp+'/test/analysis/'+path.name)
 for f in sorted(HERE.glob('*.py')):add(f,'evaluation_code/'+f.name)
 for f in sorted((HERE/'engine').glob('*.py')):add(f,'evaluation_code/engine/'+f.name)
 for f in (REFERENCE_FILE,HERE/'README_中文.md',HERE/'VALIDATION.md',HERE/'CODE_MANIFEST.json'):
  add(f,'reference/'+f.name)
 essential=list(selected);total=sum(p.stat().st_size for p in selected.values());require(total<=essential_limit_bytes,'核心文件超过预算；未删减任何后续任务，请检查异常文件')
 used=0
 for path,name in optional:
  size=path.stat().st_size
  if used+size<=optional_budget_bytes:add(path,name);used+=size
  else:omitted.append({'path':name,'size_bytes':size,'reason':'separate optional motif budget; all15 run core records already reserved'})
 entries={name:{**fingerprint(path),'priority':'essential' if name in essential else 'optional'} for name,path in selected.items()}
 manifest={'status':'complete','package_version':VERSION,'new_training_runs':12,'reused_full_runs':3,'seeds':list(SEEDS),'schemes':list(SCHEMES),'essential_count':len(essential),'essential_bytes':total,'optional_bytes':used,'optional_budget_bytes':optional_budget_bytes,'files':entries,'omitted_optional_or_heavy_files':omitted,'source_code_index':source_index,'policy':'All12 new runs and3 frozen full controls are mandatory. No essential record is budget-truncated; optional motif records cannot consume core coverage.','weights_or_prediction_arrays_included':False,'classification':'Supplementary internal fixed test evaluation; Figure4 not run by this package.'}
 archive=root/PACKAGE_NAME;temp=archive.with_name(archive.name+'.partial-'+str(os.getpid()))
 try:
  with tarfile.open(temp,'w:gz',compresslevel=5,dereference=True) as t:
   for name,path in selected.items():
    require(fingerprint(path)=={k:v for k,v in entries[name].items() if k!='priority'},'打包前源文件改变：'+str(path));t.add(path,arcname=name,recursive=False)
   payload=(json.dumps(clean(manifest),ensure_ascii=False,indent=2,allow_nan=False)+'\n').encode();info=tarfile.TarInfo('RESULT_PACKAGE_MANIFEST.json');info.size=len(payload);t.addfile(info,io.BytesIO(payload))
  with tarfile.open(temp,'r:gz') as t:
   names=t.getnames();require(len(names)==len(entries)+1 and set(names)==set(entries)|{'RESULT_PACKAGE_MANIFEST.json'},'归档成员不完整或重复')
   for member in t:
    if member.name=='RESULT_PACKAGE_MANIFEST.json':continue
    h=hashlib.sha256();n=0
    stream=t.extractfile(member);require(stream is not None,'归档成员不可读')
    for block in iter(lambda:stream.read(8*1024*1024),b''):
     h.update(block);n+=len(block)
    require(n==entries[member.name]['size_bytes'] and h.hexdigest()==entries[member.name]['sha256'],'归档内容SHA不符：'+member.name)
  temp.replace(archive)
 except BaseException:temp.unlink(missing_ok=True);raise
 write_json(root/'core_package_manifest.json',manifest)
 print('[RESULT PACKAGE] '+str(archive),flush=True)
 return archive


def main(argv=None):
 p=argparse.ArgumentParser(description=__doc__);p.add_argument('--output_dir',required=True);p.add_argument('--baseline_dir');p.add_argument('--baseline_test_dir');p.add_argument('--package',action='store_true');p.add_argument('--package_only',action='store_true');a=p.parse_args(argv)
 try:
  if a.package_only:make_core_archive(a.output_dir,a.baseline_dir,a.baseline_test_dir);return 0
  result=summarize(a.output_dir,a.baseline_dir,a.baseline_test_dir,a.package)
  print('[SUMMARY] '+result['status']+' '+str(result['completed_new_training_runs'])+'/12')
  return 0 if result['status']=='complete' else 2
 except (ValueError,OSError,KeyError,TypeError) as exc:print('[SUMMARY ERROR] '+str(exc));return 2
if __name__=='__main__':raise SystemExit(main())
