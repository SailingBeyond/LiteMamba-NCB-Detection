#!/usr/bin/env python3
"""Read-only verification and compact evidence for the additional GatedConv runs.

The original trainer's completion marker binds its checkpoint and temperature,
but does not contain output hashes. This module records hashes observed now;
it never represents them as hashes saved by the historical producer.
"""
from __future__ import annotations
import argparse
from datetime import datetime, timezone
import hashlib
import importlib.util
import json
import math
import os
from pathlib import Path
import tarfile

VERSION = '1.0.0'
HERE = Path(__file__).resolve().parent
MAX_FILE = 8 * 1024 ** 2
MAX_PAYLOAD = 128 * 1024 ** 2
MAX_ARCHIVE = 20 * 1024 ** 2
CORE_NAME = 'figure2_gatedconv_results_core_v1_0_0.tar.gz'
REQUIRED_EXPORT = ('logits.npy', 'labels.npy', 'predictions.csv.gz',
                   'prediction_metrics.json', 'analysis/metrics.json', 'export_complete.json')
SMALL_ANALYSIS = {'metrics.json', 'confusion_matrix.csv', 'confusion_matrix_normalized.csv',
                  'per_reference.csv', 'per_template_pair.csv', 'per_motif.csv',
                  'per_strand.csv', 'per_source.csv', 'per_run.csv', 'calibration_bins.csv',
                  'confidence_threshold_scan.csv', 'confidence_bins.csv',
                  'threshold_scan.csv', 'metadata_coverage.csv', 'bootstrap_replicates.csv'}


def _read(path):
    return json.loads(Path(path).read_text(encoding='utf-8-sig'))


def _write(path, value):
    path = Path(path); path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + '.writing')
    try:
        with temporary.open('w', encoding='utf-8') as stream:
            json.dump(value, stream, ensure_ascii=False, indent=2, allow_nan=False)
            stream.write('\n'); stream.flush(); os.fsync(stream.fileno())
        os.replace(temporary, path)
    finally:
        temporary.unlink(missing_ok=True)


def _fingerprint(path):
    path = Path(path); before = path.stat(); h = hashlib.sha256()
    with path.open('rb') as stream:
        for block in iter(lambda: stream.read(8 * 1024 ** 2), b''): h.update(block)
    after = path.stat()
    if (before.st_size, before.st_mtime_ns) != (after.st_size, after.st_mtime_ns):
        raise ValueError('File changed while summary was reading it: ' + str(path))
    return {'size_bytes': after.st_size, 'sha256': h.hexdigest()}


def _analyzer():
    spec = importlib.util.spec_from_file_location('_gatedconv_binary_original_analyzer', HERE/'analyze_results.py')
    module = importlib.util.module_from_spec(spec); spec.loader.exec_module(module)
    return module


def _display(model, variant='', readout=''):
    if model == 'litemamba' and variant == 'legacy_gatedconv':
        return 'GatedConv' if readout == 'target_attention' else 'GatedConv (' + str(readout) + ')'
    return {'litemamba':'LiteMamba (selective SSM)', 'transformer':'Transformer',
            'convmixer':'ConvMixer', 'bigru':'BiGRU'}.get(str(model), str(model))


def _display_id(value):
    fields = str(value).split('|'); tags = dict(x.split('=', 1) for x in fields[1:] if '=' in x)
    return _display(fields[0], tags.get('variant', ''), tags.get('readout', ''))


def _friendly_tables(directory):
    import pandas as pd
    for name in ('run_metrics.csv', 'summary_by_model.csv', 'paired_seed_deltas.csv'):
        path = directory/name
        table = pd.read_csv(path)
        if 'model_type' in table:
            table['model_display'] = [_display(r.get('model_type'), r.get('litemamba_variant'), r.get('readout'))
                                      for r in table.to_dict('records')]
        for key in ('model_a', 'model_b'):
            if key in table: table[key+'_display'] = table[key].map(_display_id)
        table.to_csv(path, index=False)


def _architecture(config, result, item):
    required = {'model_type':'litemamba', 'litemamba_variant':'legacy_gatedconv',
                'readout':'target_attention', 'feature_scheme':'full',
                'model_dim':96, 'num_blocks':4, 'raw_signal_len':30}
    for name, expected in required.items():
        if config.get(name) != expected:
            raise ValueError('Additional run is not the frozen GatedConv: ' + name)
    kwargs = config.get('model_kwargs', {})
    for name, expected in {'model_dim':96, 'num_blocks':4, 'target_index':3,
                           'litemamba_variant':'legacy_gatedconv', 'readout':'target_attention',
                           'feature_scheme':'full'}.items():
        if kwargs.get(name) != expected: raise ValueError('Model kwargs do not match final GatedConv: '+name)
    if result.get('model_type') != 'litemamba' or result.get('litemamba_variant') != 'legacy_gatedconv' or result.get('readout') != 'target_attention':
        raise ValueError('Result architecture does not match final GatedConv')
    if config.get('strand') not in ('pos', 'neg') or result.get('strand') != config.get('strand'):
        raise ValueError('Run strand identity conflicts')
    if config.get('seed') != result.get('seed') or config.get('seed') != item['requested_config'].get('seed'):
        raise ValueError('Run seed identity conflicts')
    if item['requested_config'].get('strand') != config['strand']:
        raise ValueError('Requested strand identity conflicts')
    if config.get('class_names') != {'0':'canonical', '1':'X' if config['strand']=='pos' else 'Y'}:
        raise ValueError('Binary class identity must remain canonical and strand-specific X/Y')


def _verify_run(item, output, frozen):
    run = Path(item['run_dir']).expanduser().resolve()
    if not run.is_relative_to(output) or run == output or run.relative_to(output).as_posix() != item['run_id']:
        raise ValueError('Queue run directory is outside its declared additive result directory')
    config = _read(run/'resolved_config.json'); result = _read(run/'results.json')
    _architecture(config, result, item)
    if _read(run/'queue_requested_config.json') != item['requested_config']:
        raise ValueError('Committed requested configuration differs from the queue')
    if result.get('status') not in ('train_complete', 'complete'):
        raise ValueError('Requested run has not finished training: '+item['run_id'])
    tested = result.get('status') == 'complete'
    if result.get('test_evaluated') is not tested:
        raise ValueError('Test status conflicts with completed result')
    checkpoint = _fingerprint(run/'best_checkpoint.pt')
    if checkpoint['size_bytes'] == 0 or result.get('provenance', {}).get('checkpoint_sha256') != checkpoint['sha256']:
        raise ValueError('Best checkpoint is not bound to the committed results')
    calibration = _read(run/'calibration.json'); temperature = calibration.get('temperature')
    if calibration.get('checkpoint_sha256') != checkpoint['sha256'] or calibration.get('fitted_on') != 'val' or type(temperature) not in (int, float) or not math.isfinite(temperature) or temperature <= 0:
        raise ValueError('Calibration is not fitted on validation for the frozen checkpoint')
    if tested or frozen:
        record = frozen.get(item['run_id'])
        if not record or record.get('checkpoint_sha256') != checkpoint['sha256'] or record.get('requested_config') != item['requested_config'] or Path(record.get('run_dir', '')).resolve() != run:
            raise ValueError('Test run is not bound to the complete frozen evaluation plan')
    versions = result.get('metrics_by_split', {})
    splits = ('train','val','test') if tested else ('train','val')
    if not tested and ('test' in versions or (run/'test/export_complete.json').exists()):
        raise ValueError('Training-only result contains uncommitted test metrics')
    fingerprints = {'best_checkpoint.pt': checkpoint}
    for name in ('last_checkpoint.pt', 'best_state_dict.pt', 'resolved_config.json', 'calibration.json'):
        fingerprints[name] = _fingerprint(run/name)
    for split in splits:
        marker = _read(run/split/'export_complete.json')
        if marker.get('split') != split or marker.get('checkpoint_sha256') != checkpoint['sha256'] or marker.get('temperature') != temperature:
            raise ValueError('Export marker is not bound to checkpoint/temperature: '+split)
        if marker.get('metrics') != versions.get(split) or _read(run/split/'prediction_metrics.json') != versions.get(split):
            raise ValueError('Split metrics are not bound to committed results: '+split)
        for name in REQUIRED_EXPORT:
            fingerprints[split+'/'+name] = _fingerprint(run/split/name)
    return run, result, {'run_id':item['run_id'], 'strand':config['strand'], 'seed':config['seed'],
                         'model_display':'GatedConv', 'parameter_count':301315, 'test_evaluated':tested,
                         'observed_file_fingerprints':fingerprints}


def _small_package(output, runs, package_dir, status):
    selected = {}; omissions = []
    def add(path, name, optional=False):
        if not path.is_file() or path.is_symlink(): return
        if path.stat().st_size > MAX_FILE:
            if not optional: raise ValueError('Required core file exceeds 8 MiB: '+name)
            omissions.append({'file':name, 'reason':'Optional detailed table exceeds 8 MiB; full file retained on server'})
            return
        selected[name] = path
    for path in output.glob('*.json'):
        if path.name != 'GATEDCONV_RESULT_PACKAGE_MANIFEST.json': add(path, path.name)
    for directory in ('summary', 'compare_existing'):
        for path in sorted((output/directory).rglob('*')):
            if path.is_file() and path.suffix in ('.json','.csv','.txt'):
                add(path,path.relative_to(output).as_posix())
    small = ('results.json','resolved_config.json','queue_requested_config.json','calibration.json',
             'model_config.json','environment.json','data_audit.json','data_manifest.json',
             'metrics.csv','epoch_metrics.jsonl','benchmark.json','architecture.txt')
    for run, result in runs:
        for name in small: add(run/name,(run/name).relative_to(output).as_posix())
        add(run/'source_snapshot/manifest.json',(run/'source_snapshot/manifest.json').relative_to(output).as_posix())
        for split in ('train','val','test'):
            if split not in result.get('metrics_by_split', {}): continue
            for name in ('export_complete.json','prediction_metrics.json'):
                add(run/split/name,(run/split/name).relative_to(output).as_posix())
            for name in SMALL_ANALYSIS:
                add(run/split/'analysis'/name,(run/split/'analysis'/name).relative_to(output).as_posix(),optional=True)
    if package_dir is not None:
        package_dir = Path(package_dir).resolve()
        for path in sorted(package_dir.rglob('*')):
            relative = path.relative_to(package_dir)
            if path.is_file() and not any(part in ('tests','validation','__pycache__') or part.startswith('.') for part in relative.parts) and path.suffix in ('.py','.sh','.md','.txt','.json'):
                add(path,'code/'+relative.as_posix(),optional=True)
    records = {name:_fingerprint(path) for name,path in selected.items()}
    if sum(r['size_bytes'] for r in records.values()) > MAX_PAYLOAD:
        raise ValueError('Core evidence exceeds 128 MiB; full results remain intact on server')
    manifest_path = output/'GATEDCONV_RESULT_PACKAGE_MANIFEST.json'
    _write(manifest_path, {'version':VERSION,'status':status,'files':records,'optional_omissions':omissions,
                         'server_retention':'Full per-window prediction CSVs, float32 logits, labels, curves and checkpoints stay on server. Embeddings, attention arrays and per-epoch sample predictions are deliberately omitted by the additive launcher.',
                         'hash_scope':'Hashes observed during summary; the original trainer did not pin export payload hashes.'})
    selected.pop(manifest_path.name,None)
    selected[manifest_path.name] = manifest_path
    target = output/CORE_NAME; temporary = target.with_name(target.name+'.writing')
    try:
        with tarfile.open(temporary,'w:gz',compresslevel=6) as archive:
            for name,path in sorted(selected.items()): archive.add(path,arcname=name,recursive=False)
        if temporary.stat().st_size > MAX_ARCHIVE: raise ValueError('Compressed core exceeds 20 MiB; full results remain intact')
        os.replace(temporary,target)
    finally: temporary.unlink(missing_ok=True)
    return target


def summarize(output_dir, baseline_dirs=None, package_dir=None):
    """Verify every explicitly requested additive run, then summarize all seeds."""
    output = Path(output_dir).expanduser().resolve(); manifest = _read(output/'queue_manifest.json')
    items = manifest.get('experiments')
    if not isinstance(items,list) or not items: raise ValueError('Missing explicit additional-run queue')
    ids = [item['run_id'] for item in items]
    if len(ids) != len(set(ids)): raise ValueError('Duplicate additional run IDs')
    frozen = {}; plan_path = output/'evaluation_plan.json'
    if plan_path.is_file():
        plan = _read(plan_path); records = plan.get('experiments', [])
        frozen = {r['run_id']:r for r in records}
        if len(frozen) != len(records) or set(frozen) != set(ids) or plan.get('policy') != 'validation_only_selection_then_fixed_queue_evaluation':
            raise ValueError('Evaluation plan does not freeze the entire requested GatedConv queue')
    verified = []; runs = []
    for item in items:
        run,result,record = _verify_run(item,output,frozen)
        runs.append((run,result)); verified.append(record)
    tested = sum(v['test_evaluated'] for v in verified)
    status = 'complete' if tested == len(items) else 'train_complete' if tested == 0 else 'partial'
    analyzer = _analyzer(); summary_dir = output/'summary'; summary_dir.mkdir(exist_ok=True)
    aggregate = {}
    for split in ('train','val','test'):
        ready = [str(run) for run,result in runs if split in result['metrics_by_split']]
        if ready:
            folder=summary_dir/split
            aggregate[split] = analyzer.aggregate_runs(ready, folder, split=split)
            _friendly_tables(folder)
    comparison = {}
    if baseline_dirs:
        if isinstance(baseline_dirs,(str,Path)): baseline_dirs=[baseline_dirs]
        roots = [str(Path(path).expanduser().resolve()) for path in baseline_dirs]
        for root in roots:
            if not Path(root).is_dir(): raise ValueError('Baseline result directory does not exist: '+root)
        for split in aggregate:
            folder=output/'compare_existing'/split
            comparison[split] = analyzer.aggregate_runs(roots+[str(run) for run,result in runs if split in result['metrics_by_split']],folder,split=split)
            _friendly_tables(folder)
    report = {'version':VERSION,'status':status,'model_display':'GatedConv','model_type_compatibility_key':'litemamba',
              'requested_runs':len(items),'trained_runs':len(runs),'tested_runs':tested,
              'strands':sorted({v['strand'] for v in verified}), 'seeds':sorted({v['seed'] for v in verified}),
              'aggregate':aggregate,'comparison':comparison,
              'seed_policy':'All requested seeds; sample SD within each strand and identical training/data signature. No best-test-seed selection.',
              'integrity_scope':'Original checkpoint/temperature/metric bindings verified. Export hashes are newly observed, not historical producer pins.',
              'limitations':['These new runs use the frozen final GatedConv architecture; previously completed models are not retrained.',
                             'Read-disjoint evaluation does not establish template or experimental-batch independence.',
                             'Optional baseline paired deltas require matching content-SHA signatures and seeds; they are descriptive, not significance tests.'],
              'core_path':str(output/CORE_NAME)}
    _write(summary_dir/'verified_artifacts.json',{'status':'PASS','runs':verified,
           'observed_at':datetime.now(timezone.utc).isoformat(),'hash_scope':'Current observed file hashes; original export markers have no payload hash pins.'})
    summary_path=summary_dir/'summary.json';_write(summary_path,report)
    core = _small_package(output,runs,package_dir,status)
    return {'status':status,'core_path':str(core),'summary_path':str(summary_path)}


def main(argv=None):
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output_dir',required=True)
    parser.add_argument('--baseline_dirs',nargs='*')
    args=parser.parse_args(argv)
    print(json.dumps(summarize(args.output_dir,args.baseline_dirs,HERE),ensure_ascii=False,indent=2))


if __name__=='__main__':main()
