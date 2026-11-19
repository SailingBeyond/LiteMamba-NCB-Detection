#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""CPLX同来源六类窗口：流式审计、read划分、定额导出和断点恢复。"""
import argparse
from collections import Counter
import contextlib
import csv
import fcntl
import hashlib
import io
import json
import math
import os
from pathlib import Path
import platform
import shutil
import signal
import sqlite3
import sys
import tarfile
import time

import numpy as np

from feature6 import ParsedRead, FEATURE_HEADER
from reference6 import ReferenceLibrary, LABELS

VERSION = '1.0.0'
HERE = Path(__file__).resolve().parent
SPLITS = ('train', 'validation', 'test')
META_HEADER = [
    'feature_row', 'batch_id', 'read_id', 'chrom', 'bam_strand', 'tombo_strand',
    'strand_match', 'split', 'source_line', 'row_start', 'row_end', 'target_pos',
    'kmer_ref_start', 'kmer_ref_end', 'raw_oriented_6mer', 'masked_6mer',
    'true_base', 'label', 'strict_match', 'row_segment_start_index_0based',
    'row_segment_end_index_exclusive_0based', 'segment_indices_oriented_0based',
    'segment_lengths_oriented', 'serialized_row_signal_start_index_0based',
    'serialized_row_signal_end_index_exclusive_0based', 'signal_coordinate_system',
]
STOP = False


def request_stop(signum, frame):
    global STOP
    STOP = True
    print('[STOP] 已收到停止请求；完成当前read后保存检查点。', flush=True)


def digest_text(*parts):
    return hashlib.sha256('\0'.join(str(x) for x in parts).encode('utf-8')).digest()


def assign_split(read_id, batch_id, seed, ratios=(7., 2., 1.)):
    u = int.from_bytes(digest_text(seed, batch_id, read_id)[:8], 'big') / 2**64
    total = sum(ratios)
    if u < ratios[0] / total:
        return 'train'
    if u < (ratios[0] + ratios[1]) / total:
        return 'validation'
    return 'test'


def atomic_json(path, value):
    path = Path(path)
    temp = path.with_name(path.name + '.writing')
    try:
        with temp.open('w', encoding='utf-8') as f:
            json.dump(value, f, ensure_ascii=False, indent=2, allow_nan=False)
            f.write('\n'); f.flush(); os.fsync(f.fileno())
        os.replace(temp, path)
    finally:
        if temp.exists():
            temp.unlink()


def write_csv(path, header, rows, delimiter=','):
    path = Path(path)
    temp = path.with_name(path.name + '.writing')
    try:
        with temp.open('w', newline='', encoding='utf-8') as f:
            writer = csv.writer(f, delimiter=delimiter)
            writer.writerow(header); writer.writerows(rows)
            f.flush(); os.fsync(f.fileno())
        os.replace(temp, path)
    finally:
        if temp.exists():
            temp.unlink()


def file_sha(path):
    h = hashlib.sha256()
    with Path(path).open('rb') as f:
        for block in iter(lambda: f.read(2**20), b''):
            h.update(block)
    return h.hexdigest()


def input_fingerprint(path):
    path = Path(path).resolve()
    st = path.stat()
    if not path.is_file():
        raise ValueError('输入必须是可seek的普通TSV文件')
    h = hashlib.sha256()
    with path.open('rb') as f:
        h.update(f.read(65536))
        if st.st_size > 65536:
            f.seek(max(0, st.st_size-65536)); h.update(f.read(65536))
    return {'path': str(path), 'size': st.st_size, 'mtime_ns': st.st_mtime_ns,
            'head_tail_sha256': h.hexdigest()}


def check_disk(args):
    checked = set()
    for directory in [args.output_dir, args.output.parent, args.meta_out.parent]:
        while not directory.exists():
            directory = directory.parent
        device = directory.stat().st_dev
        if device in checked:
            continue
        checked.add(device)
        free = shutil.disk_usage(directory).free
        if free < args.disk_reserve_gib * 2**30:
            raise OSError(28, '%s剩余空间低于预留 %.2f GiB；清理后加 --resume 继续' %
                          (directory, args.disk_reserve_gib))


def contract(args, lib):
    return {
        'version': VERSION, 'input': input_fingerprint(args.input),
        'runtime': {'python': platform.python_version(), 'numpy': np.__version__,
                    'platform': platform.system()},
        'reference_sha256': lib.sha256,
        'code_sha256': {p: file_sha(HERE/p) for p in
                        ['extract_xna_6mer_features_all.py', 'reference6.py', 'feature6.py']},
        'batch_id': args.batch_id, 'split_seed': args.split_seed,
        'selection_seed': args.selection_seed, 'split_ratios': list(args.split_ratios),
        'ncb_guard': args.ncb_guard, 'exclude_intervals': [list(x) for x in args.exclude_intervals],
        'context_scope': args.context_scope, 'context_count': len(lib.contexts),
        'reference_placeholder': args.reference_placeholder,
        'min_left_pos_ncb_only': args.min_left_pos,
        'require_strand_match': args.require_strand_match,
        'dwell_scale': args.dwell_scale, 'quality_scale': args.quality_scale,
        'mad_use_unique': args.mad_use_unique,
        'signal_round': args.signal_round, 'stat_round': args.stat_round,
        'feature_normalization': 'legacy float32 median/MAD on entire serialized overlap; unique default',
        'sampling_unit': 'at most one valid window per real read, context and class',
        'template_assignment': 'inherited BAM assignment; TSV has no MAPQ/barcode assignment scores',
        'raw_offsets': 'TSV row-relative only; absolute FAST5 raw sample offsets unavailable',
        'response_convention': 'legacy BAM-reference mismatch codes; reverse base order, do not recode values',
    }


def connect(path):
    db = sqlite3.connect(str(path))
    db.execute('PRAGMA journal_mode=WAL')
    db.execute('PRAGMA synchronous=FULL')
    db.execute('CREATE TABLE IF NOT EXISTS state (name TEXT PRIMARY KEY, payload TEXT NOT NULL)')
    db.execute('''CREATE TABLE IF NOT EXISTS reads
               (read_id TEXT PRIMARY KEY, line INTEGER NOT NULL, digest TEXT NOT NULL,
                split TEXT NOT NULL, chrom TEXT, strand TEXT, accepted INTEGER NOT NULL,
                reason TEXT NOT NULL)''')
    db.commit()
    return db


def get_state(db, name):
    row = db.execute('SELECT payload FROM state WHERE name=?', (name,)).fetchone()
    return json.loads(row[0]) if row else None


def set_state(db, name, value):
    db.execute('INSERT OR REPLACE INTO state VALUES (?,?)',
               (name, json.dumps(value, ensure_ascii=False, allow_nan=False)))


def key(*parts):
    return '|'.join(str(p) for p in parts)


def parse_read(raw, args, lib):
    parts = raw.decode('utf-8').rstrip('\r\n').split('\t')
    read = ParsedRead.from_fields(parts)
    if args.require_strand_match and not (
            read.strand_match == '1' and read.bam_strand == read.tombo_strand):
        raise ValueError('strand_mismatch')
    if read.chrom not in lib.sequences:
        raise RuntimeError('TSV参考名不存在于原始FASTA: %s' % read.chrom)
    sequence = lib.sequences[read.chrom]
    if read.row_start < 1 or read.row_end > len(sequence):
        raise RuntimeError('TSV参考坐标超出原始FASTA: %s' % read.read_id)
    expected = sequence[read.row_start-1:read.row_end].translate(
        str.maketrans({'X': args.reference_placeholder, 'Y': args.reference_placeholder}))
    if read.ref_seq.upper() != expected:
        raise RuntimeError('TSV参考片段与原始参考的X/Y替换版本不一致，停止以防错标。read=%s' % read.read_id)
    if not read.validate_full_signal():
        raise ValueError('full_signal:' + str(read.last_signal_error))
    return read


def read_candidates(read, lib, args):
    """每个read/context/class只保留一个确定性位置；天然read不要求覆盖NCB。"""
    result = {}
    scores = {}
    for site in lib.candidate_sites(read.chrom, read.bam_strand):
        if site['label'] >= 4 and read.row_start > args.min_left_pos:
            continue
        lo, hi = site['kmer_ref_start'], site['kmer_ref_end']
        if lo < read.row_start or hi > read.row_end:
            continue
        if not read.validate_window(lo, hi):
            continue
        cell = (site['masked_6mer'], site['label'])
        score = digest_text('position', args.selection_seed, args.batch_id, read.read_id,
                            cell[0], cell[1], site['target_pos'])
        if cell not in scores or score < scores[cell]:
            scores[cell] = score; result[cell] = site
    return result


def new_audit_state():
    return {'status': 'running', 'offset': 0, 'line': 0, 'chain_sha256': '00'*32,
            'stats': {}, 'counts': {}, 'by_strand': {}, 'by_reference_label': {},
            'strict_counts': {}, 'started_unix': time.time()}


def save_checkpoint(db, name, state, args):
    check_disk(args)
    set_state(db, name, state)
    db.commit()


def audit(args, lib, db):
    state = get_state(db, 'audit')
    if state is not None and not args.resume:
        raise RuntimeError('审计目录已有任务；继续请加 --resume，独立实验请使用新目录')
    if state and state['status'] == 'complete':
        print('[AUDIT RESUME] 审计已完成，复用原有计数。', flush=True)
        audit_reports(args, lib, db, state, only_missing=True)
        return state
    if state is None:
        state = new_audit_state()
        save_checkpoint(db, 'audit', state, args)
    stats = Counter(state['stats'])
    counts = Counter(state['counts'])
    strands = Counter(state['by_strand'])
    refs = Counter(state['by_reference_label'])
    strict = Counter(state['strict_counts'])
    progress = time.monotonic()
    dirty = 0

    def checkpoint():
        state.update(stats=dict(stats), counts=dict(counts), by_strand=dict(strands),
                     by_reference_label=dict(refs), strict_counts=dict(strict))
        save_checkpoint(db, 'audit', state, args)

    try:
        with Path(args.input).open('rb', buffering=2**22) as stream:
            stream.seek(state['offset'])
            for raw in stream:
                state['line'] += 1
                state['offset'] = stream.tell()
                row_digest = hashlib.sha256(raw).digest()
                state['chain_sha256'] = hashlib.sha256(
                    bytes.fromhex(state['chain_sha256']) + row_digest).hexdigest()
                if state['line'] == 1 and raw.lstrip(b'\xef\xbb\xbf').startswith(b'read_id\t'):
                    stats['headers'] += 1
                    continue
                stats['input_rows'] += 1
                dirty += 1
                rid = raw.split(b'\t', 1)[0].decode('utf-8', errors='strict')
                if not rid or '\n' in rid or '\r' in rid:
                    stats['invalid_read_id'] += 1
                else:
                    previous = db.execute('SELECT digest FROM reads WHERE read_id=?', (rid,)).fetchone()
                    if previous:
                        if previous[0] != row_digest.hex():
                            raise RuntimeError('同一真实read ID重复出现不同记录，不能按窗口拆开: %s' % rid)
                        stats['identical_duplicate_rows'] += 1
                    else:
                        split = assign_split(rid, args.batch_id, args.split_seed, args.split_ratios)
                        accepted, reason, chrom, strand = 0, '', '', ''
                        try:
                            read = parse_read(raw, args, lib)
                            chrom, strand = read.chrom, read.bam_strand
                            candidates = read_candidates(read, lib, args)
                            accepted = 1
                            stats['valid_unique_reads'] += 1
                            stats['reads_with_candidates'] += bool(candidates)
                            for (context, label), site in candidates.items():
                                counts[key(split, context, label)] += 1
                                strands[key(split, context, label, strand)] += 1
                                refs[key(split, chrom, strand, label)] += 1
                                if site['strict_match']:
                                    strict[key(split, chrom, strand, context, label)] += 1
                            stats['candidate_windows_after_read_cell_cap'] += len(candidates)
                        except (ValueError, UnicodeError) as exc:
                            reason = str(exc).split('\n')[0][:160]
                            stats['rejected_unique_reads'] += 1
                            stats['reject:' + reason] += 1
                        db.execute('INSERT INTO reads VALUES (?,?,?,?,?,?,?,?)',
                                   (rid, state['line'], row_digest.hex(), split,
                                    chrom, strand, accepted, reason))
                if dirty >= args.checkpoint_every or STOP:
                    checkpoint(); dirty = 0
                    if STOP:
                        raise InterruptedError('已保存审计检查点；用相同命令加 --resume 继续')
                if (args.log_every and stats['input_rows'] % args.log_every == 0) or (
                        time.monotonic()-progress >= args.log_seconds):
                    print('[AUDIT] rows=%s reads=%s candidates=%s rejected=%s offset=%.2f GiB' %
                          (f"{stats['input_rows']:,}", f"{stats['valid_unique_reads']:,}",
                           f"{stats['candidate_windows_after_read_cell_cap']:,}",
                           f"{stats['rejected_unique_reads']:,}", state['offset']/2**30), flush=True)
                    progress = time.monotonic()
        if input_fingerprint(args.input) != get_state(db, 'contract')['input']:
            raise RuntimeError('审计期间输入文件发生变化，不能继续使用此任务')
        state['status'] = 'complete'; state['completed_unix'] = time.time()
        checkpoint()
    except BaseException:
        db.rollback()
        raise
    audit_reports(args, lib, db, state)
    return state


def build_plan(args, state, contexts):
    counts = state['counts']
    quota = {}
    context_rows = []
    for context in sorted(contexts):
        per_split = {s: min(counts.get(key(s, context, label), 0) for label in range(6))
                     for s in SPLITS}
        if args.max_per_split_cell:
            per_split = {s: min(v, args.max_per_split_cell) for s, v in per_split.items()}
        total = sum(per_split.values())
        keep = (total >= args.min_total_per_class and
                per_split['train'] >= 1 and per_split['validation'] >= 1 and
                per_split['test'] >= args.min_test_per_class)
        context_rows.append({'context': context, 'keep': keep,
                             'per_class_by_split': per_split,
                             'per_class_total': total})
        if keep:
            for split in SPLITS:
                for label in range(6):
                    quota[key(split, context, label)] = per_split[split]
    return {'version': VERSION, 'status': 'ready' if quota else 'no_feasible_contexts',
            'min_total_per_class': args.min_total_per_class,
            'min_test_per_class': args.min_test_per_class,
            'max_per_split_cell': args.max_per_split_cell,
            'contexts_kept': sum(r['keep'] for r in context_rows),
            'total_rows': sum(quota.values()),
            'rows_by_split': {s: sum(v for k, v in quota.items() if k.split('|')[0] == s)
                              for s in SPLITS},
            'quota': quota, 'contexts': context_rows,
            'note': 'Exact class balance per context within each read-disjoint split; contexts may have different quotas.'}


def package_core(args, phase):
    names = ['input_contract.json', 'reference_context_scope.json', 'audit_summary.json',
             'candidate_counts.csv', 'candidate_counts_by_strand.csv',
             'candidate_counts_by_reference.csv', 'strict_candidate_counts.csv',
             'sampling_preview.json']
    if phase == 'extract':
        names += ['sampling_plan.json', 'extraction_summary.json', 'selected_counts.csv']
    out = args.output_dir / ('cplx_same_source_%s_core_v1_0_0.tar.gz' % phase)
    temp = out.with_name(out.name + '.writing')
    with tarfile.open(temp, 'w:gz') as tar:
        for name in names:
            path = args.output_dir/name
            if path.exists():
                tar.add(path, arcname=name, recursive=False)
    os.replace(temp, out)
    print('[CORE] %s' % out, flush=True)


def audit_reports(args, lib, db, state, only_missing=False):
    needed = ['audit_summary.json', 'sampling_preview.json', 'read_splits.tsv',
              'candidate_counts.csv', 'candidate_counts_by_strand.csv',
              'candidate_counts_by_reference.csv', 'strict_candidate_counts.csv',
              'cplx_same_source_audit_core_v1_0_0.tar.gz']
    if only_missing and all((args.output_dir/n).exists() for n in needed):
        return
    summary = {k: state[k] for k in ['status', 'line', 'offset', 'chain_sha256',
                                   'stats', 'started_unix', 'completed_unix']}
    summary['reference'] = lib.summary
    summary['count_unit'] = 'max one valid window per read/context/class; different cells may share a read'
    summary['normalization_domain'] = 'entire TSV overlap, not full unclipped FAST5'
    summary['limitations'] = get_state(db, 'contract')
    atomic_json(args.output_dir/'audit_summary.json', summary)
    preview = build_plan(args, state, lib.contexts)
    atomic_json(args.output_dir/'sampling_preview.json', preview)
    for field, filename, header in [
        ('counts', 'candidate_counts.csv', ['split', 'context', 'label']),
        ('by_strand', 'candidate_counts_by_strand.csv', ['split', 'context', 'label', 'strand']),
        ('by_reference_label', 'candidate_counts_by_reference.csv', ['split', 'chrom', 'strand', 'label']),
        ('strict_counts', 'strict_candidate_counts.csv', ['split', 'chrom', 'strand', 'context', 'label'])]:
        write_csv(args.output_dir/filename, header+['valid_read_cell_windows'],
                  (k.split('|')+[v] for k, v in sorted(state[field].items())))
    write_csv(args.output_dir/'read_splits.tsv',
              ['read_id', 'split', 'batch_id', 'chrom', 'bam_strand', 'accepted', 'reason'],
              ((rid, split, args.batch_id, chrom, strand, accepted, reason)
               for rid, split, chrom, strand, accepted, reason in db.execute(
                   'SELECT read_id,split,chrom,strand,accepted,reason FROM reads ORDER BY line')),
              delimiter='\t')
    package_core(args, 'audit')
    print('[AUDIT COMPLETE] 候选context=%d；预览保留%d；预计导出%d行。未写大特征文件。' %
          (len(lib.contexts), preview['contexts_kept'], preview['total_rows']), flush=True)


def row_bytes(values, delimiter=','):
    stream = io.StringIO(newline='')
    csv.writer(stream, delimiter=delimiter).writerow(values)
    return stream.getvalue().encode('utf-8')


def metadata(read, site, args, split, source_line, row_no):
    extra = read.window_metadata(site['kmer_ref_start'], site['kmer_ref_end'])
    values = {
        'feature_row': row_no, 'batch_id': args.batch_id, 'read_id': read.read_id,
        'chrom': read.chrom, 'bam_strand': read.bam_strand, 'tombo_strand': read.tombo_strand,
        'strand_match': read.strand_match, 'split': split, 'source_line': source_line,
        'row_start': read.row_start, 'row_end': read.row_end,
        **site, **extra,
    }
    return [int(values[h]) if isinstance(values[h], bool) else values[h] for h in META_HEADER]


def extraction(args, lib, db, audit_state):
    if audit_state is None or audit_state['status'] != 'complete':
        raise RuntimeError('请先完成 --phase audit，再运行 --phase extract')
    plan = build_plan(args, audit_state, lib.contexts)
    if not plan['quota']:
        raise RuntimeError('没有满足三集合六类共同支持和预设门槛的context；请查看sampling_preview.json')
    identity = {'output': str(args.output), 'meta_out': str(args.meta_out),
                'with_header': args.with_header, 'plan': plan}
    state = get_state(db, 'extract')
    prior = get_state(db, 'extract_identity')
    if prior and prior != identity:
        raise RuntimeError('导出路径、表头或配额规则改变，不能在同一目录混用不同导出方案')
    if state and not args.resume:
        raise RuntimeError('导出任务已存在；继续请加 --resume')
    if state and state['status'] == 'complete':
        for path, size in [(args.output, state['feature_offset']), (args.meta_out, state['meta_offset'])]:
            if not path.exists() or path.stat().st_size != size:
                raise RuntimeError('已完成的导出文件缺失或大小改变: %s' % path)
        summary_path = args.output_dir/'extraction_summary.json'
        if summary_path.exists():
            saved_summary = json.loads(summary_path.read_text(encoding='utf-8'))
            for field, path in [('features', args.output), ('metadata', args.meta_out)]:
                if file_sha(path) != saved_summary[field]['sha256']:
                    raise RuntimeError('已完成导出文件的SHA256改变，拒绝覆盖或伪装完成: %s' % path)
        required_reports = ['extraction_summary.json', 'selected_counts.csv',
                            'cplx_same_source_extract_core_v1_0_0.tar.gz']
        if not all((args.output_dir/name).exists() for name in required_reports):
            extraction_reports(args, db, state)
        print('[EXTRACT RESUME] 导出已完成，没有重复写入。', flush=True)
        return state
    if state is None:
        for path in [args.output, args.meta_out]:
            if path.exists():
                raise RuntimeError('拒绝覆盖已有文件，请使用新的输出路径: %s' % path)
            path.parent.mkdir(parents=True, exist_ok=True)
        state = {'status': 'running', 'offset': 0, 'line': 0, 'feature_offset': 0,
                 'meta_offset': 0, 'rows': 0, 'stats': {}, 'selected_counts': {},
                 'remaining': {k: audit_state['counts'][k] for k in plan['quota']},
                 'needed': dict(plan['quota']), 'started_unix': time.time()}
        set_state(db, 'extract_identity', identity)
        save_checkpoint(db, 'extract', state, args)
        atomic_json(args.output_dir/'sampling_plan.json', plan)
    progress = time.monotonic()
    dirty = 0
    # 已提交检查点之后的未提交字节在恢复时截断，两个输出保持逐行对应。
    try:
        with contextlib.ExitStack() as stack:
            outputs = []
            for path, offset_name in [(args.output, 'feature_offset'), (args.meta_out, 'meta_offset')]:
                offset = state[offset_name]
                if path.exists():
                    if path.stat().st_size < offset:
                        raise RuntimeError('输出比已提交检查点更短，不能恢复: %s' % path)
                    f = stack.enter_context(path.open('r+b', buffering=2**20))
                    f.seek(offset); f.truncate()
                elif offset:
                    raise RuntimeError('恢复所需输出文件缺失: %s' % path)
                else:
                    f = stack.enter_context(path.open('w+b', buffering=2**20))
                outputs.append(f)
            features, meta = outputs
            if state['feature_offset'] == 0 and args.with_header:
                features.write(row_bytes(FEATURE_HEADER))
            if state['meta_offset'] == 0:
                meta.write(row_bytes(META_HEADER, '\t'))

            def checkpoint():
                check_disk(args)
                features.flush(); meta.flush()
                os.fsync(features.fileno()); os.fsync(meta.fileno())
                state['feature_offset'], state['meta_offset'] = features.tell(), meta.tell()
                save_checkpoint(db, 'extract', state, args)

            checkpoint()
            stream = stack.enter_context(Path(args.input).open('rb', buffering=2**22))
            stream.seek(state['offset'])
            for raw in stream:
                state['line'] += 1; state['offset'] = stream.tell(); dirty += 1
                if state['line'] == 1 and raw.lstrip(b'\xef\xbb\xbf').startswith(b'read_id\t'):
                    continue
                rid = raw.split(b'\t', 1)[0].decode('utf-8')
                record = db.execute('SELECT line,digest,split,accepted FROM reads WHERE read_id=?',
                                    (rid,)).fetchone()
                if record is None:
                    if not rid or '\n' in rid or '\r' in rid:
                        continue
                    raise RuntimeError('输入出现审计未记录的read: %s' % rid)
                if hashlib.sha256(raw).hexdigest() != record[1]:
                    raise RuntimeError('输入行内容与审计不一致: %s' % rid)
                if record[0] == state['line'] and record[3]:
                    read = parse_read(raw, args, lib)
                    candidates = read_candidates(read, lib, args)
                    split = record[2]
                    for (context, label), site in candidates.items():
                        cell = key(split, context, label)
                        if cell not in state['remaining']:
                            continue
                        remaining, needed = state['remaining'][cell], state['needed'][cell]
                        if remaining <= 0:
                            raise RuntimeError('候选数量超过审计结果: ' + cell)
                        # 顺序无放回抽样：从remaining个候选中精确选择needed个。
                        u = int.from_bytes(digest_text('sample', args.selection_seed, args.batch_id,
                                                      rid, cell)[:8], 'big') / 2**64
                        choose = needed > 0 and (needed == remaining or u < needed/remaining)
                        state['remaining'][cell] -= 1
                        if choose:
                            values = read.features(site, dwell_scale=args.dwell_scale,
                                                   quality_scale=args.quality_scale,
                                                   mad_use_unique=bool(args.mad_use_unique),
                                                   signal_round=args.signal_round,
                                                   stat_round=args.stat_round)
                            row_no = state['rows'] + 1
                            features.write(row_bytes(values))
                            meta.write(row_bytes(metadata(read, site, args, split,
                                                          state['line'], row_no), '\t'))
                            state['rows'] = row_no
                            state['needed'][cell] -= 1
                            state['selected_counts'][cell] = state['selected_counts'].get(cell, 0) + 1
                if dirty >= args.checkpoint_every or STOP:
                    checkpoint(); dirty = 0
                    if STOP:
                        raise InterruptedError('已保存导出检查点；用相同命令加 --resume 继续')
                if (args.log_every and state['line'] % args.log_every == 0) or (
                        time.monotonic()-progress >= args.log_seconds):
                    print('[EXTRACT] input_rows=%s output_rows=%s remaining_to_select=%s' %
                          (f"{state['line']:,}", f"{state['rows']:,}",
                           f"{sum(state['needed'].values()):,}"), flush=True)
                    progress = time.monotonic()
            if state['line'] != audit_state['line'] or any(state['remaining'].values()):
                raise RuntimeError('输入行数或逐cell候选量与审计不一致')
            if any(state['needed'].values()) or state['rows'] != plan['total_rows']:
                raise RuntimeError('未达到冻结配额，不能将输出标记为完成')
            if input_fingerprint(args.input) != get_state(db, 'contract')['input']:
                raise RuntimeError('导出期间输入发生变化')
            state['status'] = 'complete'; state['completed_unix'] = time.time()
            checkpoint()
    except BaseException:
        db.rollback()
        raise
    extraction_reports(args, db, state)
    return state


def extraction_reports(args, db, state):
    identity = get_state(db, 'extract_identity')
    summary = {k: state[k] for k in ['status', 'rows', 'feature_offset', 'meta_offset',
                                   'started_unix', 'completed_unix']}
    summary.update(version=VERSION, batch_id=args.batch_id,
                   contexts_kept=identity['plan']['contexts_kept'],
                   rows_by_split=identity['plan']['rows_by_split'],
                   features={'path': str(args.output), 'sha256': file_sha(args.output)},
                   metadata={'path': str(args.meta_out), 'sha256': file_sha(args.meta_out)},
                   raw_sample_coordinate_system='serialized TSV overlap; not absolute FAST5',
                   strict_subset_note='Flags describe the selected six-class windows only; not a complete five-class training pool.')
    atomic_json(args.output_dir/'extraction_summary.json', summary)
    write_csv(args.output_dir/'selected_counts.csv', ['split', 'context', 'label', 'count'],
              (k.split('|')+[v] for k, v in sorted(state['selected_counts'].items())))
    package_core(args, 'extract')


def parse_args(argv=None):
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--version', action='version', version=VERSION)
    p.add_argument('--input', required=True)
    p.add_argument('--output_dir', type=Path)
    p.add_argument('--output', type=Path)
    p.add_argument('--meta_out', type=Path)
    p.add_argument('--phase', choices=['audit', 'extract', 'all'], default='audit')
    p.add_argument('--reference', type=Path, default=HERE/'reference/CPLX_XNA_all_original.fasta',
                   help='原始含X/Y的实验参考；不要传X_to_T比对参考')
    p.add_argument('--batch_id', default='CPLX-XNA-all')
    p.add_argument('--reference_placeholder', choices=list('ATCG'), default='T')
    p.add_argument('--context_scope', choices=['common', 'all'], default='common')
    p.add_argument('--ncb_guard', type=int, default=10)
    p.add_argument('--exclude_intervals', default='1238:1280')
    p.add_argument('--min_left_pos', type=int, default=1258,
                   help='只对NCB窗口保留旧覆盖条件，天然窗口不受此条件限制')
    p.add_argument('--split_ratios', default='7,2,1')
    p.add_argument('--split_seed', type=int, default=20260429)
    p.add_argument('--selection_seed', type=int, default=20261026)
    p.add_argument('--min_total_per_class', type=int, default=20,
                   help='每context最终三个split合计每类至少多少；不是统一配额')
    p.add_argument('--min_test_per_class', type=int, default=1)
    p.add_argument('--max_per_split_cell', type=int, default=0,
                   help='每split/context/class上限，0不额外限制')
    p.add_argument('--dwell_scale', type=float, default=1.)
    p.add_argument('--quality_scale', type=float, default=1.)
    p.add_argument('--mad_use_unique', type=int, choices=[0, 1], default=1)
    p.add_argument('--require_strand_match', type=int, choices=[0, 1], default=1)
    p.add_argument('--with_header', action='store_true')
    p.add_argument('--mask_base', choices=['N'], default='N')
    p.add_argument('--label_x', choices=[4], type=int, default=4)
    p.add_argument('--label_y', choices=[5], type=int, default=5)
    p.add_argument('--target_pos', type=int, default=1275)
    p.add_argument('--signal_round', type=int, default=6)
    p.add_argument('--stat_round', type=int, default=6)
    p.add_argument('--checkpoint_every', type=int, default=1000)
    p.add_argument('--disk_reserve_gib', type=float, default=2.)
    p.add_argument('--log_every', type=int, default=10000)
    p.add_argument('--log_seconds', type=float, default=30.)
    p.add_argument('--resume', action='store_true')
    args = p.parse_args(argv)
    if args.output_dir is None:
        if args.output is None:
            p.error('请提供 --output_dir，或用 --output 指定新的特征文件路径')
        args.output_dir = args.output.parent/(args.output.stem+'_same_source_v1_0_0')
    args.output_dir = args.output_dir.resolve()
    args.input = Path(args.input).resolve(); args.reference = args.reference.resolve()
    args.output = (args.output or args.output_dir/'features.csv').resolve()
    args.meta_out = (args.meta_out or args.output_dir/'features_meta.tsv').resolve()
    try:
        args.split_ratios = tuple(float(x) for x in args.split_ratios.split(','))
        if len(args.split_ratios) != 3 or any(not math.isfinite(x) or x <= 0 for x in args.split_ratios):
            raise ValueError()
        args.exclude_intervals = tuple(tuple(int(v) for v in term.split(':'))
                                       for term in args.exclude_intervals.split(',') if term)
        if any(len(v) != 2 for v in args.exclude_intervals):
            raise ValueError()
    except ValueError:
        p.error('split_ratios需三个正数；exclude_intervals格式为1238:1280,其他起点:终点')
    if args.checkpoint_every < 1 or args.min_total_per_class < 1 or args.min_test_per_class < 1:
        p.error('checkpoint_every及样本门槛必须为正整数')
    if (args.max_per_split_cell < 0 or args.ncb_guard < 0 or
            not 0 <= args.signal_round <= 15 or not 0 <= args.stat_round <= 15):
        p.error('上限和guard不能为负；小数位数需为0..15')
    for value in [args.dwell_scale, args.quality_scale, args.log_seconds]:
        if not math.isfinite(value) or value <= 0:
            p.error('缩放和log_seconds必须为有限正数')
    if not math.isfinite(args.disk_reserve_gib) or args.disk_reserve_gib < 0:
        p.error('disk_reserve_gib必须是有限非负数')
    if any(x == y for i, x in enumerate([args.input, args.reference, args.output, args.meta_out])
           for y in [args.input, args.reference, args.output, args.meta_out][i+1:]):
        p.error('输入、参考和两个输出必须是不同路径')
    if args.output_dir in (args.input, args.reference, HERE):
        p.error('请使用独立的输出目录')
    return args


def main(argv=None):
    args = parse_args(argv)
    args.output_dir.mkdir(parents=True, exist_ok=True)
    for sig in [signal.SIGINT, signal.SIGTERM]:
        signal.signal(sig, request_stop)
    with (args.output_dir/'.pipeline.lock').open('a') as lock:
        try:
            fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError:
            raise RuntimeError('该输出目录已有提取进程运行')
        check_disk(args)
        lib = ReferenceLibrary(args.reference, ncb_guard=args.ncb_guard,
                               exclude_intervals=args.exclude_intervals,
                               context_scope=args.context_scope)
        positions = {p for c in lib.sequences for p in lib.nc_positions(c)}
        if positions != {args.target_pos}:
            raise ValueError('本版CPLX流程要求参考NCB位点与--target_pos一致；当前参考位置=%r' % sorted(positions))
        current = contract(args, lib)
        with contextlib.closing(connect(args.output_dir/'audit.sqlite')) as db:
            previous = get_state(db, 'contract')
            if previous is not None and previous != current:
                raise RuntimeError('输入、源码、参考或冻结的提取／read划分参数改变；请使用新输出目录')
            if previous is None:
                set_state(db, 'contract', current); db.commit()
                atomic_json(args.output_dir/'input_contract.json', current)
                atomic_json(args.output_dir/'reference_context_scope.json',
                            {'summary': lib.summary, 'contexts': sorted(lib.contexts),
                             'definition': 'window context whitelist; never a read/reference whitelist'})
            state = get_state(db, 'audit')
            if args.phase in ('audit', 'all'):
                state = audit(args, lib, db)
            if args.phase in ('extract', 'all'):
                extraction(args, lib, db, state)
            print('[FULL PASS] phase=%s 输出目录=%s' % (args.phase, args.output_dir), flush=True)


if __name__ == '__main__':
    try:
        main()
    except (Exception, KeyboardInterrupt) as exc:
        print('[ERROR] %s: %s' % (type(exc).__name__, exc), file=sys.stderr, flush=True)
        sys.exit(2 if isinstance(exc, (KeyboardInterrupt, InterruptedError)) else 1)
