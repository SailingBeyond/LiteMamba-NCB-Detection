#!/usr/bin/env python3
"""Rebuild plot tables from the user's completed, frozen test results.

The compact bundled snapshot contains actual count summaries, not fabricated
examples. No training or full test inference is performed by this module.
"""
from __future__ import annotations
import argparse
import csv
import gzip
import hashlib
import json
from pathlib import Path
import numpy as np

HERE = Path(__file__).resolve().parent
DEFAULT_ROOT = HERE / 'results'
SEEDS = (42, 2024, 2025)
TASK_LABELS = {'6-class': ['A','T','C','G','X','Y'], '5X': ['A','T','C','G','X'], '5Y': ['A','T','C','G','Y']}
MODELS = ('GatedConv','Transformer','ConvMixer','BiGRU')


def sha256_file(path):
    h = hashlib.sha256()
    with Path(path).open('rb') as f:
        for block in iter(lambda: f.read(8*1024*1024), b''):
            h.update(block)
    return h.hexdigest()


def write_json(path, value):
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    Path(path).write_text(json.dumps(value, indent=2, ensure_ascii=False, sort_keys=True)+'\n', encoding='utf-8')


def metrics_from_confusion(cm):
    cm = np.asarray(cm, dtype=np.float64)
    if cm.ndim != 2 or cm.shape[0] != cm.shape[1] or not np.isfinite(cm).all() or np.any(cm<0) or np.any(cm!=np.floor(cm)):
        raise ValueError('Confusion counts must be a finite nonnegative integer square matrix')
    n = float(cm.sum())
    if n <= 0:
        raise ValueError('Empty confusion matrix')
    truth, pred, diag = cm.sum(axis=1), cm.sum(axis=0), np.diag(cm)
    recall = np.divide(diag, truth, out=np.zeros_like(diag), where=truth>0)
    f1 = np.divide(2*diag, truth+pred, out=np.zeros_like(diag), where=(truth+pred)>0)
    denominator = np.sqrt(max(0., n*n-np.dot(pred,pred))*max(0., n*n-np.dot(truth,truth)))
    return {'n_samples':int(n), 'accuracy':float(diag.sum()/n),
            'balanced_accuracy':float(recall[truth>0].mean()), 'f1_macro':float(f1.mean()),
            'mcc':float((diag.sum()*n-np.dot(pred,truth))/denominator) if denominator else 0.}


def write_csv(path, rows, fields):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open('w',encoding='utf-8',newline='') as f:
        w=csv.DictWriter(f,fieldnames=fields,extrasaction='raise')
        w.writeheader(); w.writerows(rows)


def read_frozen_snapshot(snapshot_path=None):
    path = Path(snapshot_path) if snapshot_path is not None else HERE/'frozen_results.json.gz'
    manifest_path = path.with_name('frozen_results_manifest.json')
    manifest = json.loads(manifest_path.read_text(encoding='utf-8'))
    if path.stat().st_size != manifest['size_bytes'] or sha256_file(path) != manifest['sha256']:
        raise ValueError('Frozen result snapshot SHA256/size mismatch; refuse to plot altered evidence')
    with gzip.open(path,'rt',encoding='utf-8') as f:
        snapshot = json.load(f)
    if snapshot.get('schema_version')!=1 or snapshot.get('seeds')!=list(SEEDS):
        raise ValueError('Unsupported frozen result schema/seed set')
    return snapshot, manifest


def collect_results(result_root=DEFAULT_ROOT, snapshot_path=None):
    root=Path(result_root).expanduser().resolve(); data=root/'data'; data.mkdir(parents=True,exist_ok=True)
    snapshot, manifest=read_frozen_snapshot(snapshot_path)
    expected={(task,model,seed) for task in TASK_LABELS for model in MODELS for seed in SEEDS}
    seen=set(); overall=[]; motifs=[]; confusion=[]; specs=[]
    for run in snapshot['runs']:
        spec=run['spec']; key=(spec['task'],spec['model'],int(spec['seed']))
        if key not in expected or key in seen:
            raise ValueError('Duplicate/unexpected task/model/seed: '+str(key))
        seen.add(key); labels=TASK_LABELS[key[0]]; raw=run['raw_summary']; cm=np.asarray(raw['confusion_matrix'])
        if cm.shape!=(len(labels),len(labels)):
            raise ValueError('Wrong biological class mapping: '+str(key))
        computed=metrics_from_confusion(cm)
        if int(raw['n'])!=computed['n_samples']:
            raise ValueError('Confusion/overall count conservation failed: '+str(key))
        for original,canonical in [('accuracy','accuracy'),('macro_f1','f1_macro'),('balanced_accuracy','balanced_accuracy')]:
            if abs(float(raw[original])-computed[canonical])>1e-10:
                raise ValueError('Reported metric differs from raw confusion counts: '+str((key,original)))
        base={'task':key[0],'model':key[1],'seed':key[2]}
        overall.append({**base,**computed})
        for i,true_name in enumerate(labels):
            for j,pred_name in enumerate(labels):
                confusion.append({**base,'true_label':true_name,'pred_label':pred_name,'count':int(cm[i,j])})
        subgroup=run.get('per_motif',[])
        if spec['model']=='GatedConv':
            if len(subgroup)!=1024 or len({m['motif'] for m in subgroup})!=1024:
                raise ValueError('Expected 1024 distinct GatedConv backgrounds: '+str(key))
            total=correct=0
            for m in subgroup:
                motif=m['motif']; n=int(m['n']); c=int(m['correct'])
                if len(motif)!=6 or motif[3]!='N' or any(motif[i]not in 'ACGT' for i in (0,1,2,4,5)) or n<=0 or not 0<=c<=n:
                    raise ValueError('Invalid motif count/identity: '+str(key)+' '+str(m))
                total+=n; correct+=c
                motifs.append({**base,'motif':motif,'accuracy':c/n,'n_samples':n,'n_correct':c})
            if total!=computed['n_samples'] or correct!=int(np.trace(cm)):
                raise ValueError('Per-motif counts do not conserve full test counts: '+str(key))
        # A portable frozen config is bundled; original server config remains untouched.
        if 'config' in spec:
            portable=dict(spec)
            cfg=dict(spec['config'])
            config_copy=data/'run_configs'/key[0]/key[1]/f'seed_{key[2]}.json'
            write_json(config_copy,cfg)
            portable['original_config_path']=portable.get('config_path')
            portable['original_config_sha256']=portable.get('expected_config_sha256')
            portable['config_path']=str(config_copy)
            portable['expected_config_sha256']=sha256_file(config_copy)
            portable['engine_dir']=str(HERE/spec['engine_dir'])
            portable['config']=cfg
            specs.append(portable)
        else:
            specs.append(dict(spec))
    if seen!=expected:
        raise ValueError('Missing task/model/seed combinations: '+str(sorted(expected-seen)))
    overall.sort(key=lambda r:(r['task'],r['model'],r['seed']))
    motifs.sort(key=lambda r:(r['task'],r['model'],r['seed'],r['motif']))
    write_csv(data/'overall_metrics.csv',overall,['task','model','seed','accuracy','balanced_accuracy','f1_macro','mcc','n_samples'])
    write_csv(data/'per_motif_metrics.csv',motifs,['task','model','seed','motif','accuracy','n_samples','n_correct'])
    write_csv(data/'confusion_counts.csv',confusion,['task','model','seed','true_label','pred_label','count'])
    write_json(data/'run_specs.json',specs)
    report={'version':'1.0.4','source':'actual already completed frozen test result exports supplied by user',
            'frozen_results_sha256':manifest['sha256'],'common_seeds':list(SEEDS),
            'overall_rows':len(overall),'gatedconv_motif_rows':len(motifs),'confusion_rows':len(confusion),
            'decision':'unthresholded full argmax; raw outputs; no test-dependent model/threshold selection',
            'training_started':False,'test_inference_repeated':False,
            'source_evidence':snapshot['source_evidence'],'limitations':snapshot['limitations'],
            'output_files':{name:{'sha256':sha256_file(data/name),'size_bytes':(data/name).stat().st_size}
                            for name in ['overall_metrics.csv','per_motif_metrics.csv','confusion_counts.csv','run_specs.json']}}
    write_json(data/'data_provenance.json',report)
    return report


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--result-root','--result_root',default=str(DEFAULT_ROOT))
    p.add_argument('--snapshot',default=None)
    args=p.parse_args(); report=collect_results(args.result_root,args.snapshot)
    print(json.dumps({k:v for k,v in report.items() if k not in ('source_evidence','output_files')},indent=2,ensure_ascii=False))


if __name__=='__main__':
    main()
