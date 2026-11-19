#!/usr/bin/env python3
"""Strictly audit corrected split files against their original paired streams.

This module reads tables only. It does not extract features, split/rewrite inputs,
generate read identities, or infer molecular independence from row identities.
"""
from __future__ import annotations

from collections import Counter
from contextlib import contextmanager
from decimal import Decimal, InvalidOperation
import csv
import hashlib
from itertools import chain, zip_longest
import json
from pathlib import Path
import re

from figure4_common import fingerprint, known, parse_reference


FEATURE_COLUMNS = ('kmer', 'mean', 'std', 'median', 'dwell', 'quality',
                   'mismatch', 'insertion', 'deletion', 'signal', 'label')
BASE_TO_LABEL = {'A': 0, 'T': 1, 'C': 2, 'G': 3}


def _integer_label(value, allowed, where):
    try:
        label = Decimal(str(value).strip())
    except InvalidOperation as error:
        raise ValueError(where + ': invalid label') from error
    if not label.is_finite() or label != label.to_integral_value() or int(label) not in allowed:
        raise ValueError(where + ': label must be an exact integer in ' + str(sorted(allowed)))
    return int(label)


def _kmer(value, where):
    sequence = str(value).strip()
    if re.fullmatch('[ACTGN]{6}', sequence) is None:
        raise ValueError(where + ': kmer must be uppercase A/C/T/G/N 6mer')
    return sequence


def _checked_rows(reader, width, path):
    for row in reader:
        if len(row) != width:
            raise ValueError(f'{path}:{reader.line_num}: expected {width} fields, got {len(row)}')
        if any('\n' in value or '\r' in value for value in row):
            raise ValueError(f'{path}:{reader.line_num}: multiline fields are unsupported')
        yield row


@contextmanager
def _feature_table(path):
    with Path(path).open(encoding='utf-8-sig', newline='') as stream:
        reader = csv.reader(stream)
        first = next(reader, None)
        if first is None:
            raise ValueError('Empty feature CSV: ' + str(path))
        if first and first[0].strip().lower() == 'kmer':
            if tuple(value.strip().lower() for value in first) != FEATURE_COLUMNS:
                raise ValueError('Unexpected feature header: ' + str(path))
            header = first
            rows = _checked_rows(reader, 11, path)
        else:
            header = None
            # A separate generator preserves reader.line_num without discarding row 1.
            def data_rows():
                for row in chain([first], reader):
                    if len(row) != 11 or any('\n' in value or '\r' in value for value in row):
                        raise ValueError(f'{path}:{reader.line_num}: expected 11 single-line fields')
                    yield row
            rows = data_rows()
        yield header, rows


@contextmanager
def _metadata_table(path):
    with Path(path).open(encoding='utf-8-sig', newline='') as stream:
        reader = csv.reader(stream, delimiter='\t')
        header = next(reader, None)
        if not header or any(not name for name in header) or len(set(header)) != len(header):
            raise ValueError('Missing/duplicate metadata header: ' + str(path))
        if 'chrom' not in header:
            raise ValueError('Original/split metadata requires chrom: ' + str(path))
        yield header, _checked_rows(reader, len(header), path)


class _Digest:
    def __init__(self):
        self.count = 0
        self.value = hashlib.sha256()

    def add(self, row):
        self.value.update((json.dumps(row, ensure_ascii=False, separators=(',', ':')) + '\n').encode('utf-8'))
        self.count += 1

    def report(self):
        return {'rows': self.count, 'canonical_row_sha256': self.value.hexdigest()}


def _key(template, role, strand):
    parsed_template, parsed_role = parse_reference(template)
    if parsed_role != 'target' or parsed_template not in {f'XNA{i}' for i in range(13, 17)}:
        raise ValueError('Correction is restricted to XNA13–16 template declarations')
    if role not in ('control', 'target') or strand not in ('pos', 'neg'):
        raise ValueError('Correction declarations require role=control/target and file_strand=pos/neg')
    return parsed_template, role, strand


def _reference_checks(metadata, original_label, strand, where):
    template, role = parse_reference(metadata['chrom'])
    if template not in {f'XNA{i}' for i in range(13, 17)}:
        raise ValueError(where + ': original reference is outside XNA/PC/DNA13–16')
    if original_label != (1 if role == 'target' else 0):
        raise ValueError(where + ': original binary label conflicts with chrom target/control role')
    if known(metadata.get('reference')):
        try:
            reference_identity = parse_reference(metadata['reference'])
        except ValueError:
            reference_identity = None  # Preserve an unparseable real reference name without guessing.
        if reference_identity is not None and reference_identity != (template, role):
            raise ValueError(where + ': metadata reference conflicts with chrom')
    bam_strand = str(metadata.get('bam_strand', '')).strip()
    if known(bam_strand) and bam_strand != ('+' if strand == 'pos' else '-'):
        raise ValueError(where + ': original bam_strand conflicts with original pos/neg extraction stream')
    return template, role


def audit_corrected_split(corrected_root, selected_items, original_pos_csv=None, original_neg_csv=None,
                          original_pos_meta=None, original_neg_meta=None):
    """Verify complete transformed row streams; fail instead of dropping/misaligning rows.

    ``selected_items`` contain feature_csv, metadata_tsv, template, role and
    file_strand. Normally the correction caller supplies exactly eight targets;
    requiring that global selection is the caller's responsibility. Original
    controls are fully audited but may be excluded from replacement.
    """
    root = Path(corrected_root).expanduser().resolve()
    supplied = {'pos_csv': original_pos_csv, 'neg_csv': original_neg_csv,
                'pos_meta': original_pos_meta, 'neg_meta': original_neg_meta}
    defaults = {strand + kind: root.parent / ('xna_pc_13_16_' + strand +
                ('.csv' if kind == '_csv' else '.meta.tsv'))
                for strand in ('pos', 'neg') for kind in ('_csv', '_meta')}
    originals = {key: Path(value).expanduser().resolve() if value is not None else defaults[key]
                 for key, value in supplied.items()}
    for key, path in originals.items():
        if not path.is_file():
            raise FileNotFoundError('Correction audit requires original unsplit paired inputs: ' + str(path) +
                '; provide --original_' + key + ' /actual/path (split metadata should be generated with --write_split_meta).')
    selected, all_paths = {}, set()
    for item in selected_items:
        key = _key(str(item['template']), str(item['role']), str(item['file_strand']))
        if key in selected:
            raise ValueError('Duplicate correction template/role/strand: ' + str(key))
        feature = Path(item['feature_csv']).expanduser().resolve()
        meta_value = str(item.get('metadata_tsv', '')).strip()
        if not meta_value:
            raise ValueError('Corrected split metadata is required; rerun split with --write_split_meta: ' + str(feature))
        metadata = Path(meta_value).expanduser().resolve()
        for path in (feature, metadata):
            if not path.is_file():
                raise FileNotFoundError('Corrected split input missing (use --write_split_meta): ' + str(path))
            if path in all_paths or path in originals.values():
                raise ValueError('Duplicate or unsplit-as-split input path: ' + str(path))
            all_paths.add(path)
        selected[key] = {'item': dict(item), 'feature': feature, 'metadata': metadata,
                         'expected_feature': _Digest(), 'expected_metadata': _Digest()}
    if not selected:
        raise ValueError('No corrected split inputs selected')
    original_fingerprints = {key: fingerprint(path) for key, path in originals.items()}
    split_fingerprints = {str(path): fingerprint(path) for path in sorted(all_paths)}
    original_stats, expected_headers = {}, {}
    missing = object()
    for strand in ('pos', 'neg'):
        source_feature, source_meta = originals[strand + '_csv'], originals[strand + '_meta']
        stats = Counter()
        with _feature_table(source_feature) as (feature_header, feature_rows), \
             _metadata_table(source_meta) as (meta_header, meta_rows):
            meta_index = {name: index for index, name in enumerate(meta_header)}
            expected_headers[strand] = (feature_header, meta_header)
            for number, (feature_row, meta_row) in enumerate(zip_longest(feature_rows, meta_rows,
                                                                       fillvalue=missing), 1):
                where = f'{source_feature} + {source_meta}:data_row={number}'
                if feature_row is missing or meta_row is missing:
                    raise ValueError(where + ': original feature/metadata row-count mismatch')
                sequence = _kmer(feature_row[0], where)
                label = _integer_label(feature_row[-1], {0, 1}, where)
                metadata = dict(zip(meta_header, meta_row))
                if 'kmer' in meta_index and _kmer(metadata['kmer'], where + ' metadata') != sequence:
                    raise ValueError(where + ': original feature/metadata full kmer mismatch')
                if 'label' in meta_index and _integer_label(metadata['label'], {0, 1}, where + ' metadata') != label:
                    raise ValueError(where + ': original feature/metadata binary label mismatch')
                template, role = _reference_checks(metadata, label, strand, where)
                if label == 0 and sequence[3] not in BASE_TO_LABEL:
                    raise ValueError(where + ': original natural fourth base is not A/T/C/G')
                new_label = (4 if strand == 'pos' else 5) if label == 1 else BASE_TO_LABEL[sequence[3]]
                new_sequence = sequence[:3] + 'N' + sequence[4:]
                converted_feature = list(feature_row)
                converted_feature[0], converted_feature[-1] = new_sequence, str(new_label)
                converted_meta = list(meta_row)
                if 'kmer' in meta_index:
                    converted_meta[meta_index['kmer']] = new_sequence
                if 'label' in meta_index:
                    converted_meta[meta_index['label']] = str(new_label)
                stats['rows'] += 1
                stats[template + '/' + role] += 1
                entry = selected.get((template, role, strand))
                if entry is None:
                    stats['excluded_rows'] += 1
                else:
                    stats['selected_rows'] += 1
                    entry['expected_feature'].add(converted_feature)
                    entry['expected_metadata'].add(converted_meta)
        original_stats[strand] = dict(stats)
    per_selected = []
    for (template, role, strand), entry in selected.items():
        expected_feature_header, expected_meta_header = expected_headers[strand]
        actual_feature, actual_meta = _Digest(), _Digest()
        with _feature_table(entry['feature']) as (feature_header, feature_rows), \
             _metadata_table(entry['metadata']) as (meta_header, meta_rows):
            if feature_header != expected_feature_header:
                raise ValueError('Split feature header differs from original: ' + str(entry['feature']))
            if meta_header != expected_meta_header:
                raise ValueError('Split metadata full header differs from original: ' + str(entry['metadata']))
            for number, (feature_row, meta_row) in enumerate(zip_longest(feature_rows, meta_rows,
                                                                       fillvalue=missing), 1):
                if feature_row is missing or meta_row is missing:
                    raise ValueError('Split feature/metadata row-count mismatch: ' + str(entry['feature']))
                _kmer(feature_row[0], str(entry['feature']) + ':' + str(number))
                _integer_label(feature_row[-1], set(range(6)), str(entry['feature']) + ':' + str(number))
                actual_feature.add(feature_row)
                actual_meta.add(meta_row)
        expected_feature, expected_meta = entry['expected_feature'].report(), entry['expected_metadata'].report()
        if not expected_feature['rows']:
            raise ValueError('Selected correction has no matching original rows: ' + str((template, role, strand)))
        if actual_feature.report() != expected_feature:
            raise ValueError('Split feature rows differ from complete expected transform (omission/reorder/value/label/reference): ' + str(entry['feature']))
        if actual_meta.report() != expected_meta:
            raise ValueError('Split full metadata rows differ from expected transform: ' + str(entry['metadata']))
        per_selected.append({'template': template, 'role': role, 'file_strand': strand,
            'feature_csv': str(entry['feature']), 'metadata_tsv': str(entry['metadata']),
            'expected_feature': expected_feature, 'actual_feature': actual_feature.report(),
            'expected_metadata': expected_meta, 'actual_metadata': actual_meta.report(), 'status': 'pass',
            'feature_fingerprint': split_fingerprints[str(entry['feature'])],
            'metadata_fingerprint': split_fingerprints[str(entry['metadata'])]})
    for key, path in originals.items():
        if fingerprint(path) != original_fingerprints[key]:
            raise ValueError('Original input changed during correction audit: ' + str(path))
    for path in sorted(all_paths):
        if fingerprint(path) != split_fingerprints[str(path)]:
            raise ValueError('Corrected split input changed during correction audit: ' + str(path))
    return {'status': 'pass', 'audit_policy': 'strict_original_pair_and_complete_transformed_row_stream_v1',
        'original_fingerprints': original_fingerprints, 'original_row_counts': original_stats,
        'selected_file_count': len(per_selected), 'selected_rows': sum(row['actual_feature']['rows'] for row in per_selected),
        'excluded_rows': sum(stats.get('excluded_rows', 0) for stats in original_stats.values()),
        'per_selected': per_selected,
        'notes': ['All original paired rows are checked; excluded references are not silently added to replacement.',
                  'Canonical row SHA256 covers all 11 CSV fields and full metadata rows in original per-reference order.',
                  'This audits splitting/transformation without rerunning feature extraction or inventing read identity.']}
