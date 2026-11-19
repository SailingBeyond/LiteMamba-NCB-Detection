#!/usr/bin/env python3
"""Read-only validation of the already balanced Figure 3 five-class inputs.

Feature CSVs are never copied, rebalanced or repartitioned. Read membership is
indexed on disk, including the disclosed positive row-proxy identities.
"""
from __future__ import annotations
import csv
import hashlib
import importlib.util
import itertools
import json
import math
import os
from pathlib import Path
import re
import shutil
import sqlite3
import tempfile
import time
from collections import Counter

VERSION = '1.0.1'
HERE = Path(__file__).resolve().parent
SPLITS = ('train', 'validation', 'test')
FEATURE_COLUMNS = ('kmer', 'mean', 'std', 'median', 'dwell', 'quality',
                   'mismatch', 'insertion', 'deletion', 'signal', 'label')
GIB = 2 ** 30


def read_json(path):
    return json.loads(Path(path).read_text(encoding='utf-8-sig'))


def sha256(path):
    h = hashlib.sha256()
    with Path(path).open('rb') as f:
        for block in iter(lambda: f.read(8 * 1024 ** 2), b''): h.update(block)
    return h.hexdigest()


def write_json(path, value):
    path = Path(path); path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + '.writing.' + str(os.getpid()))
    try:
        with temporary.open('w', encoding='utf-8') as f:
            json.dump(value, f, ensure_ascii=False, indent=2, allow_nan=False)
            f.write('\n'); f.flush(); os.fsync(f.fileno())
        os.replace(temporary, path)
    finally: temporary.unlink(missing_ok=True)


def _upstream(reference_dir=None):
    ref = Path(reference_dir) if reference_dir else HERE / 'reference'
    source = ref / 'balance_and_split_xna_pc_5class_by_read.py'
    if not source.is_file(): raise ValueError('缺少随包冻结的数据制备代码：' + str(source))
    spec = importlib.util.spec_from_file_location('_fiveclass_balance_contract', source)
    module = importlib.util.module_from_spec(spec); spec.loader.exec_module(module)
    return module, sha256(source)


def _fingerprint(path, expected=None):
    path = Path(path); before = path.stat()
    if expected is not None:
        if before.st_size != int(expected['size_bytes']):
            raise ValueError('文件大小与制备指纹不一致：' + str(path))
    h = hashlib.sha256(); newlines = 0; last = b''
    with path.open('rb') as f:
        for block in iter(lambda: f.read(8 * 1024 ** 2), b''):
            h.update(block); newlines += block.count(b'\n'); last = block[-1:]
    value = {'path': str(path.resolve()), 'size_bytes': before.st_size,
             'size':before.st_size, 'mtime_ns':before.st_mtime_ns,
             'physical_lines':newlines+int(bool(last) and last!=b'\n'),
             'sha256':h.hexdigest()}
    after = path.stat()
    if (before.st_size, before.st_mtime_ns) != (after.st_size, after.st_mtime_ns):
        raise ValueError('审计时输入发生变化：' + str(path))
    if expected is not None and value['sha256'] != expected['sha256']:
        raise ValueError('完整SHA256与制备指纹不一致：' + str(path))
    return value


def _manifest_result(manifest, path):
    return {'data_dir': manifest['data_dir'], 'manifest_path': str(path),
            'rows': manifest['rows'], 'rows_by_split': manifest['rows_by_split'],
            'read_membership_database': manifest['read_membership_database'],
            'scenario': manifest['scenario'], 'class_names': manifest['class_names'],
            'positive_identity': manifest['positive_identity'],
            'split_contract': manifest['split_contract'],
            'input_fingerprints': manifest['input_fingerprints'],
            'producer_code_provenance': manifest['producer_code_provenance']}


def _verify_prior(dest, identity, resume):
    if not dest.exists(): return None
    if not resume: raise FileExistsError('准备结果已存在；请使用 --resume：' + str(dest))
    p = dest / 'prepare_manifest.json'
    if not p.is_file(): raise ValueError('已有准备目录不完整，不能覆盖：' + str(dest))
    manifest = read_json(p)
    if manifest.get('identity') != identity:
        raise ValueError('源数据、制备报告或代码身份已改变；请使用新的输出目录')
    for name, expected in manifest.get('prepared_payloads', {}).items():
        _fingerprint(dest / name, expected)
    return _manifest_result(manifest, p)


def _producer_provenance(root, task, reference_sha):
    """Separate historical producer identity from today's validation reference.

    A coherent historical hash is provenance, not evidence that the historical
    source algorithm has been inspected. Materialized outputs are still audited.
    """
    raw = {'root': root.get('code_sha256'), 'task': task.get('code_sha256')}
    digests = {}; helpers = {}; recovery_schema = {}
    def checked_sha(value, where):
        if not isinstance(value, str) or re.fullmatch(r'[0-9a-fA-F]{64}', value) is None:
            raise ValueError(where + ' 制备代码SHA格式无效；须为64位十六进制，不能伪造或改写历史记录')
        return value.lower()
    for name, value in raw.items():
        recovery_schema[name] = isinstance(value, dict) and bool({'original_script', 'recovery_helper'} & set(value))
        helpers[name] = None
        if isinstance(value, dict):
            if recovery_schema[name]:
                if not {'original_script', 'recovery_helper'}.issubset(value) or 'script' in value:
                    raise ValueError(name + ' 恢复制备代码SHA结构无效')
                helpers[name] = checked_sha(value['recovery_helper'], name + ' recovery_helper')
                value = value['original_script']
            elif value and 'script' not in value:
                raise ValueError(name + ' 制备代码SHA结构无效；非空对象必须包含script字段')
            else:
                value = value.get('script')
        if value is None or value == '':
            if recovery_schema[name]: raise ValueError(name + ' 恢复制备代码SHA格式无效')
            digests[name] = None
            continue
        digests[name] = checked_sha(value, name)
    known = {value for value in digests.values() if value is not None}
    if len(known) > 1:
        raise ValueError('根报告与任务报告的制备代码SHA相互冲突，不能混用不同制备结果')
    producer = next(iter(known), None)
    same = producer == reference_sha if producer is not None else None
    recovered = any(recovery_schema.values())
    recovery_reports = None
    helper = None
    if recovered:
        if not all(recovery_schema.values()) or len(set(helpers.values())) != 1:
            raise ValueError('根报告与任务报告的恢复脚本SHA或格式相互冲突')
        helper = helpers['root']; recovery_reports = {'root':root.get('recovery'), 'task':task.get('recovery')}
        total = root.get('input_windows')
        if type(total) is not int or total < 1 or task.get('input_windows_in_full_sixclass_file') != total:
            raise ValueError('恢复报告的完整输入行数不一致')
        for name, report in recovery_reports.items():
            if not isinstance(report, dict) or report.get('status') != 'complete' or report.get('frozen_selection_reused') is not True or report.get('no_resampling') is not True:
                raise ValueError(name + ' 恢复报告必须完成且复用冻结选择、没有重新抽样')
            if checked_sha(report.get('original_script_sha256'), name+' recovery original_script') != producer or checked_sha(report.get('helper_sha256'), name+' recovery helper') != helper:
                raise ValueError(name + ' 恢复报告与制备代码SHA相互冲突')
            if report.get('input_identity_rows_revalidated') != total:
                raise ValueError(name + ' 恢复报告未核对完整输入行身份')
    helper_reference = HERE/'reference'/'resume_fiveclass_export.py'
    helper_reference_sha = sha256(helper_reference) if helper_reference.is_file() else None
    return {'recorded_producer_code_fields': raw,
            'root_producer_sha256': digests['root'], 'task_producer_sha256': digests['task'],
            'producer_sha256': producer, 'validator_reference_sha256': reference_sha,
            'producer_matches_validator_reference': same,
            'recorded_identity_status': 'complete' if all(digests.values()) else 'partial' if known else 'unknown',
            'producer_source_bytes_verified_against_recorded_sha': bool(same),
            'recovery_export': recovered, 'recovery_helper_sha256': helper,
            'validator_recovery_reference_sha256': helper_reference_sha,
            'recovery_helper_source_bytes_verified_against_recorded_sha': helper is not None and helper == helper_reference_sha,
            'recorded_recovery_reports': recovery_reports,
            'initial_numeric_payload_identity_independently_verified': root.get('initial_numeric_payload_identity_independently_verified'),
            'compatibility_basis': 'materialized output contract: file hashes, aligned rows, class mapping, masking, read partition and balancing',
            'scope': 'Historical producer source is not available for source-code equivalence verification when hashes differ; original metadata is preserved.'}


def _verify_summary(root, task, scenario, reference_sha):
    if root.get('status') != 'complete' or task.get('status') != 'complete':
        raise ValueError('五分类制备 summary.json 未标记 complete')
    record = root.get('tasks', {}).get(scenario)
    if not record or record.get('directory') != 'fiveclass_' + scenario:
        raise ValueError('根报告缺少本任务或任务目录不正确：' + scenario)
    if task.get('task') != 'fiveclass_' + scenario or task.get('num_classes') != 5:
        raise ValueError('任务报告不是所选的五分类任务')
    mapping = task.get('label_mapping', {})
    expected = {str(k): k for k in range(4)}
    expected[str(4 if scenario == 'X' else 5)] = 4
    if mapping.get('source_to_output') != expected or mapping.get('sequence_target_index') != 3 or mapping.get('sequence_target_character') != 'N':
        raise ValueError('类别映射或第4位N的制备契约不匹配')
    if record.get('label_mapping') != mapping: raise ValueError('根报告与任务标签映射不同')
    names = {str(k): v for k, v in enumerate(['A','T','C','G',scenario])}
    if task.get('label_names') != names or mapping.get('output_label_names') != names:
        raise ValueError('五分类名称顺序不正确')
    if record.get('split_counts') != task.get('split_counts'):
        raise ValueError('根报告与任务划分计数不一致')
    if root.get('same_read_in_different_splits_across_tasks') != 0 or task.get('read_ids_in_multiple_splits') != 0:
        raise ValueError('原制备报告存在跨集合标识符')
    producer_provenance = _producer_provenance(root, task, reference_sha)
    contract = task.get('split_contract', {})
    if contract.get('method') != 'blake2b64-read-split-v1' or contract.get('window_ratio') != '7:2:1':
        raise ValueError('仅接受原已冻结read哈希分区及7:2:1窗口配额')
    if root.get('input_fingerprints') != task.get('input_fingerprints'):
        raise ValueError('根报告与任务原输入来源指纹不同')
    return producer_provenance


def _paired_rows(feature, metadata):
    with feature.open(encoding='utf-8-sig', newline='', buffering=8*1024**2) as ff, metadata.open(encoding='utf-8-sig', newline='', buffering=8*1024**2) as mf:
        fr = csv.reader(ff, strict=True); mr = csv.reader(mf, delimiter='\t', strict=True)
        fh = next(fr, None); mh = next(mr, None)
        if tuple(fh or ()) != FEATURE_COLUMNS: raise ValueError('特征表头须保持制备脚本的11列顺序：' + str(feature))
        if not mh or len(set(mh)) != len(mh) or not {'read_id','chrom','kmer','label','source_label','split','split_seed','split_ratios','split_method'}.issubset(mh):
            raise ValueError('metadata字段缺失或重复：' + str(metadata))
        missing = object()
        for index, (f, m) in enumerate(itertools.zip_longest(fr, mr, fillvalue=missing), 1):
            if f is missing or m is missing: raise ValueError('CSV与metadata行数不一致：' + str(feature))
            if len(f) != 11 or len(m) != len(mh): raise ValueError('CSV或metadata列数错误：%s:%d' % (feature,index))
            yield index, f, dict(zip(mh,m))


def prepare_inputs(source_dir, work_dir, scenario, resume=False,
                   disk_reserve_gib=8, reference_dir=None):
    """Validate one original task and return its original data directory."""
    if scenario not in ('X','Y'): raise ValueError('scenario必须为X或Y')
    source = Path(source_dir).expanduser().resolve(); work = Path(work_dir).expanduser().resolve()
    if work == source or work in source.parents: raise ValueError('结果目录不能覆盖数据源或其父目录')
    if not math.isfinite(disk_reserve_gib) or disk_reserve_gib < 0: raise ValueError('磁盘余量无效')
    upstream, reference_sha = _upstream(reference_dir)
    root_summary_path = source/'summary.json'; task_dir = source/('fiveclass_'+scenario)
    task_summary_path = task_dir/'summary.json'
    root_summary = read_json(root_summary_path); task = read_json(task_summary_path)
    producer_provenance = _verify_summary(root_summary, task, scenario, reference_sha)
    if producer_provenance['recovery_export']:
        print('[RECOVERY {}] 已识别完成的冻结选择恢复导出；分别核对原制备脚本与恢复脚本SHA'.format(scenario), flush=True)
    if producer_provenance['producer_matches_validator_reference'] is not True:
        print('[PROVENANCE {}] 保留原制备脚本SHA={}；按输出契约完整核验现有数据，不修改原summary'.format(
            scenario, producer_provenance['producer_sha256'] or 'unknown'), flush=True)
    root_outputs = root_summary.get('output_fingerprints', {}); task_outputs = task.get('output_fingerprints', {})
    def expected_file(name):
        a = task_outputs.get(name); b = root_outputs.get('fiveclass_'+scenario+'/'+name)
        if not a or not b or a != b: raise ValueError('根/任务制备指纹缺失或冲突：' + name)
        return a
    # Normal exports pin the task summary. The inspected historical recovery
    # helper excluded every summary.json from its parent fingerprints. Accept
    # that explicit completed recovery schema only, and freeze its current
    # summary bytes in our own identity without changing the historical report.
    summary_expected = root_outputs.get('fiveclass_'+scenario+'/summary.json')
    if not summary_expected and not producer_provenance['recovery_export']:
        raise ValueError('根报告未固定任务summary.json')
    summary_fingerprint = _fingerprint(task_summary_path, summary_expected)
    producer_provenance['task_summary_hash_binding'] = {
        'method':'root_report_fingerprint' if summary_expected else 'locally_frozen_recovery_report',
        'root_report_pins_task_summary':bool(summary_expected),
        'current_sha256':summary_fingerprint['sha256']}
    fingerprints = {}
    for split in SPLITS:
        for name in (split+'.csv', split+'_meta.tsv'):
            fingerprints[name] = _fingerprint(task_dir/name, expected_file(name))
    if (task_dir/'label_mapping.json').is_file():
        _fingerprint(task_dir/'label_mapping.json', expected_file('label_mapping.json'))
        if read_json(task_dir/'label_mapping.json') != task['label_mapping']: raise ValueError('label_mapping.json与summary不同')
    identity = {'scenario':scenario, 'data_dir':str(task_dir),
                'root_summary_sha256':sha256(root_summary_path),
                'task_summary_sha256':sha256(task_summary_path),
                'validator_reference_sha256':reference_sha,
                'producer_code_provenance':producer_provenance, 'prepare_source_sha256':sha256(Path(__file__)),
                'input_fingerprints':fingerprints}
    dest = work/'prepared_data'
    prior = _verify_prior(dest, identity, resume)
    if prior is not None: return prior
    work.mkdir(parents=True,exist_ok=True)
    estimate = int(task.get('selected_unique_reads', task['selected_windows']))*512 + 16*1024**2
    if shutil.disk_usage(work).free < estimate + disk_reserve_gib*GIB:
        raise OSError(28,'准备read索引预计需%.2fGiB并保留%.2fGiB；请清理输出盘后--resume' % (estimate/GIB,disk_reserve_gib))
    stage = Path(tempfile.mkdtemp(prefix='.prepared_data.partial-',dir=work))
    con = None
    try:
        db = stage/'read_membership.sqlite'; con = sqlite3.connect(db)
        con.execute('PRAGMA journal_mode=DELETE');con.execute('PRAGMA synchronous=NORMAL');con.execute('PRAGMA temp_store=FILE')
        con.execute('CREATE TABLE reads (read_id TEXT PRIMARY KEY, split TEXT NOT NULL, kind TEXT NOT NULL, role TEXT NOT NULL, windows INTEGER NOT NULL)')
        con.execute("CREATE TRIGGER reject_conflicting_read BEFORE UPDATE ON reads WHEN NEW.split!=OLD.split OR NEW.kind!=OLD.kind OR NEW.role!=OLD.role OR (NEW.role='positive' AND NEW.windows>1) BEGIN SELECT RAISE(ABORT,'read split/kind/role conflict or duplicate positive'); END")
        motifs = Counter(); references=Counter(); counts={}; rows_by_split={}; positive_proxy=0; positive_real=0
        source_counts=Counter(); contract=task['split_contract']; seed=int(contract['seed']); ratios=upstream.parse_split_ratios('7,2,1')
        for split in SPLITS:
            label_counts=Counter(); batch=[]; n=0
            for index,f,meta in _paired_rows(task_dir/(split+'.csv'), task_dir/(split+'_meta.tsv')):
                where = '%s/%s record %d' % (scenario,split,index)
                label=upstream.exact_label(f[-1],where)
                if label>4 or str(label)!=f[-1]: raise ValueError(where+': 输出label须为0..4整数字符串')
                motif=f[0]
                if motif!=motif.upper() or upstream.mask_kmer(motif,where)!=motif or motif[3]!='N': raise ValueError(where+': 第4位必须已经是N')
                if meta['label']!=str(label) or meta['kmer']!=motif: raise ValueError(where+': CSV与metadata序列/label不一致')
                source_label=label if label<4 else (4 if scenario=='X' else 5)
                for key in ('source_label','original_label'):
                    if meta.get(key,'') and upstream.exact_label(meta[key],where)!=source_label: raise ValueError(where+': '+key+'映射不一致')
                if meta.get('output_label',str(label))!=str(label): raise ValueError(where+': output_label冲突')
                if meta.get('task','fiveclass_'+scenario)!='fiveclass_'+scenario: raise ValueError(where+': task冲突')
                for key in ('true_base','label_base','source_class_name'):
                    if meta.get(key,'') and meta[key]!=['A','T','C','G',scenario][label]: raise ValueError(where+': '+key+'与类别冲突')
                for key in ('motif','output_kmer','masked_kmer','masked_6mer','6mer','original_kmer','raw_oriented_kmer','raw_oriented_6mer'):
                    if meta.get(key,'') and upstream.mask_kmer(meta[key],where)!=motif: raise ValueError(where+': '+key+'侧翼冲突')
                for key in ('original_kmer','raw_oriented_kmer','raw_oriented_6mer'):
                    v=meta.get(key,'').upper()
                    if label<4 and v and v[3] in 'ATCG' and v[3]!=['A','T','C','G'][label]: raise ValueError(where+': 天然中心与label冲突')
                read_id=meta['read_id']
                if read_id!=read_id.strip() or upstream.normalize_known(read_id)=='unknown': raise ValueError(where+': read_id无效')
                if meta['split']!=split or meta['split_method']!=contract['method'] or int(meta['split_seed'])!=seed or upstream.parse_split_ratios(meta['split_ratios'])!=ratios:
                    raise ValueError(where+': metadata分区契约冲突')
                if upstream.assign_read_split(read_id,seed,ratios)!=split: raise ValueError(where+': read哈希分区与文件冲突')
                proxy=upstream.is_positive_row_proxy(meta,label)
                if (meta.get('read_id_kind')=='positive_row_proxy' or read_id.startswith('positive-row:')) and not proxy:
                    raise ValueError(where+': 阳性行代理标志无效')
                if proxy and label!=4: raise ValueError(where+': 天然窗口不能使用阳性代理')
                if not proxy and upstream.normalize_known(meta['chrom'])=='unknown': raise ValueError(where+': 已提供read的参考身份缺失')
                if label==4: positive_proxy+=int(proxy);positive_real+=int(not proxy)
                for value in f[1:9]:
                    if len(value.split('|'))!=6: raise ValueError(where+': 数值特征应为6个位置')
                if len(f[9].split('|'))!=6: raise ValueError(where+': signal应为6段')
                # Float32 finiteness and every signal token are validated when the
                # shared engine cache is built; this stage does not parse twice.
                batch.append((read_id,split,'positive_row_proxy' if proxy else 'provided_real_read_id','positive' if label==4 else 'negative',1))
                label_counts[label]+=1;motifs[(motif,label,split)]+=1;references[(motif,label,meta['chrom'],split)]+=1
                source_counts[(split,label,meta.get('source_dataset','unknown'),meta.get('bam_strand','unknown'))]+=1
                n=index
                if len(batch)>=10000:
                    _flush(con,batch,where);batch.clear()
                if index%100000==0: print('[PREPARE {}/{}] audited {:,} rows'.format(scenario,split,index),flush=True)
                if index%100000==0 and shutil.disk_usage(work).free<disk_reserve_gib*GIB: raise OSError(28,'read索引写入期间空间低于保留值')
            if batch:_flush(con,batch,scenario+'/'+split)
            expected=task['split_counts'][split]
            if n!=expected['windows'] or {str(k):label_counts[k] for k in range(5)}!=expected['per_class']:
                raise ValueError('实际窗口/类别计数与summary不一致：'+split)
            if len(label_counts)!=5 or len(set(label_counts.values()))!=1: raise ValueError('分区五类不均衡：'+split)
            distinct=con.execute('SELECT COUNT(*) FROM reads WHERE split=?',(split,)).fetchone()[0]
            if distinct!=expected['unique_reads']: raise ValueError('实际唯一标识符数量与summary不同：'+split)
            counts[split]={str(k):label_counts[k] for k in range(5)};rows_by_split[split]=n
        retained=sorted(set(k[0] for k in motifs))
        if retained!=sorted(task['retained_motifs']) or len(retained)!=task['retained_motif_count']: raise ValueError('保留motif列表与summary不同')
        for motif in retained:
            for split in SPLITS:
                vals=[motifs[(motif,k,split)] for k in range(5)]
                if min(vals)<1 or len(set(vals))!=1: raise ValueError('motif内五类缺失或不等量：'+motif+'/'+split)
            for label in range(5):
                vals=[motifs[(motif,label,sp)] for sp in SPLITS]
                if vals[0]!=7*vals[2] or vals[1]!=2*vals[2]: raise ValueError('motif/class不满足精确7:2:1：'+motif)
        ref_balance=task.get('sampling',{}).get('reference_balance')
        if ref_balance=='equal':
            for motif in retained:
                for label in range(5):
                    refs={key[2] for key in references if key[0]==motif and key[1]==label}
                    for split in SPLITS:
                        if len({references[(motif,label,chrom,split)] for chrom in refs})!=1: raise ValueError('motif/class内参考比例不一致：'+motif)
        con.commit();con.close();con=None
        if sum(rows_by_split.values())!=task['selected_windows']: raise ValueError('总窗口数与summary不同')
        if positive_proxy!=task['positive']['selected_row_proxy_count'] or positive_real!=task['positive']['selected_real_read_id_count']:
            raise ValueError('实际阳性身份计数与summary不同')
        # Ensure no ordinary modification occurred during the streaming audit.
        for name,fp in fingerprints.items():
            stat=(task_dir/name).stat()
            if (stat.st_size,stat.st_mtime_ns)!=(fp['size_bytes'],fp['mtime_ns']):raise ValueError('审计期间输入大小/时间戳改变：'+name)
        positive_identity={'positive_row_proxy_windows':positive_proxy,'positive_provided_read_windows':positive_real,
            'positive_original_read_uuid_verified':False if positive_proxy else True,
            'basis':'user-declared one-positive-CSV-row-per-read experimental design' if positive_proxy else 'provided real sequencing read identifiers',
            'limitation':'Row proxies identify source CSV records, not independently recovered read UUIDs.' if positive_proxy else 'Read identity alone does not establish template/batch independence.'}
        audit={'status':'PASS','scenario':scenario,'rows_by_split':rows_by_split,'class_counts':counts,
            'producer_code_provenance':producer_provenance,
            'retained_motif_count':len(retained),'read_id_overlap_counts':{'train__validation':0,'train__test':0,'validation__test':0},
            'positive_identity':positive_identity,'reference_balance_requested':ref_balance,
            'scope':'Original broad-context Figure3 five-class task; canonical E.coli and X/Y CPLX observations remain source-associated. Not same-source strict77 control.',
            'feature_payload':'Original files used read-only; no rebalancing/repartitioning or recomputation. Engine preflight validates all numeric/signal payloads.'}
        write_json(stage/'prepare_audit.json',audit)
        payloads={name:{'size_bytes':(stage/name).stat().st_size,'sha256':sha256(stage/name)} for name in ['read_membership.sqlite','prepare_audit.json']}
        manifest={'status':'complete','version':VERSION,'identity':identity,'scenario':scenario,
            'producer_code_provenance':producer_provenance,
            'class_names':['A','T','C','G',scenario],'data_dir':str(task_dir),'rows':sum(rows_by_split.values()),'rows_by_split':rows_by_split,
            'input_fingerprints':fingerprints,'files':fingerprints,'source_summary_sha256':sha256(task_summary_path),
            'read_membership_database':str(dest/'read_membership.sqlite'),'positive_identity':positive_identity,
            'split_contract':contract,'prepared_payloads':payloads}
        write_json(stage/'prepare_manifest.json',manifest)
        if dest.exists():raise FileExistsError('另一个进程已创建准备结果；拒绝覆盖')
        os.rename(stage,dest)
        return _manifest_result(manifest,dest/'prepare_manifest.json')
    except BaseException as error:
        if con is not None:con.close()
        # This invocation owns this newly created staging directory. The read
        # index is reconstructible and cannot be resumed halfway through; remove
        # only its known temporary files before recording a small error report.
        # In particular, ENOSPC retries must not accumulate partial GiB indexes.
        for name in ('read_membership.sqlite','read_membership.sqlite-journal',
                     'read_membership.sqlite-wal','read_membership.sqlite-shm'):
            (stage/name).unlink(missing_ok=True)
        write_json(stage/'error.json',{'status':'failed','type':type(error).__name__,'error':str(error)})
        raise


def _flush(con,batch,where):
    try:
        con.executemany('INSERT INTO reads VALUES (?,?,?,?,?) ON CONFLICT(read_id) DO UPDATE SET windows=reads.windows+excluded.windows, split=excluded.split, kind=excluded.kind, role=excluded.role',batch)
        con.commit()
    except sqlite3.IntegrityError as exc:
        raise ValueError(where+': read跨分区/身份冲突，或阳性read重复窗口') from exc


def audit_cross_task_read_splits(prepared, output_dir):
    """Check cross-task read partition consistency using the two disk indexes."""
    if set(prepared)!= {'X','Y'}:raise ValueError('跨任务审计必须包含X和Y')
    producer_records={s:prepared[s]['producer_code_provenance'] for s in ('X','Y')}
    known={v['producer_sha256'] for v in producer_records.values() if v['producer_sha256'] is not None}
    if len(known)>1:raise ValueError('X/Y任务的制备代码SHA相互冲突，不能混用不同制备结果')
    helpers={v['recovery_helper_sha256'] for v in producer_records.values() if v['recovery_helper_sha256'] is not None}
    if len(helpers)>1 or len({v['recovery_export'] for v in producer_records.values()})>1:
        raise ValueError('X/Y任务的恢复脚本SHA或恢复格式相互冲突')
    left=Path(prepared['X']['read_membership_database']);right=Path(prepared['Y']['read_membership_database'])
    con=sqlite3.connect('file:'+str(left)+'?mode=ro',uri=True)
    try:
        con.execute('ATTACH DATABASE ? AS y',(str(right),))
        conflicts=con.execute("SELECT a.read_id,a.split,b.split FROM reads a JOIN y.reads b USING(read_id) WHERE a.split!=b.split OR a.kind!=b.kind OR a.role!=b.role OR a.role='positive' LIMIT 20").fetchall()
        if conflicts:raise ValueError('X/Y任务共享read的划分或身份不一致：'+repr(conflicts))
        overlap=con.execute('SELECT COUNT(*) FROM reads a JOIN y.reads b USING(read_id)').fetchone()[0]
        report={'status':'PASS','same_read_in_different_splits_across_tasks':0,'shared_identifiers':overlap,
            'producer_code_provenance':producer_records,
            'scope':'Provided real identifiers plus disclosed source-CSV positive row proxies; not original positive UUID recovery.',
            'prepared_manifest_sha256':{s:sha256(prepared[s]['manifest_path']) for s in ('X','Y')}}
        write_json(Path(output_dir)/'cross_task_read_audit.json',report)
        return report
    finally:con.close()
