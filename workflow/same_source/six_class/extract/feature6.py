"""从整行信号TSV提取六位置特征，保留既有X/Y特征的数值定义。"""
from __future__ import annotations

import math
import re
from collections.abc import Mapping

import numpy as np


FEATURE_HEADER = [
    'kmer', 'mean', 'std', 'median', 'dwell', 'quality',
    'mismatch', 'insertion', 'deletion', 'signal', 'label',
]
LABELS = {'A': 0, 'T': 1, 'C': 2, 'G': 3, 'X': 4, 'Y': 5}
COMP_TABLE_XY = str.maketrans('ACGTNXYacgtnxy', 'TGCANYXtgcanyx')
_NUMBER = r'[+-]?(?:(?:\d+(?:\.\d*)?)|(?:\.\d+))(?:[eE][+-]?\d+)?'
_SCALARS_RE = re.compile(r'(?:' + _NUMBER + r')(?:\|(?:' + _NUMBER + r'))*\Z')
_SIGNAL_RE = re.compile(r'(?:' + _NUMBER + r')(?:(?:\*|\|)(?:' + _NUMBER + r'))*\Z')


def revcomp_xy(seq):
    return seq.translate(COMP_TABLE_XY)[::-1]


def fmt_num(x, ndigits=6):
    """保持旧代码的十进制舍入和尾零处理，包括负零。"""
    scalar = float(x)
    if not math.isfinite(scalar):
        raise ValueError('nonfinite_feature_output')
    value = f'{scalar:.{ndigits}f}'
    # 零位小数没有小数尾零；直接返回，避免10被截成1。
    if ndigits == 0:
        return value
    value = value.rstrip('0').rstrip('.')
    return value if value else '0'


def robust_mad_scale(arr, use_unique=True, eps=1e-8):
    """旧定义：整行float32信号的unique值中位数和1.4826×MAD。"""
    if arr.size == 0 or not np.all(np.isfinite(arr)):
        raise ValueError('bad_full_signal')
    base = np.unique(arr) if use_unique else arr
    center = float(np.median(base))
    mad = float(np.median(np.abs(base-center))*1.4826)
    if not np.isfinite(mad) or mad < eps:
        std = float(np.std(base))
        mad = std if np.isfinite(std) and std >= eps else 1.0
    if not math.isfinite(center) or not math.isfinite(mad):
        raise ValueError('bad_mmad')
    return center, mad


def _scalars(text, expected, name):
    if not _SCALARS_RE.fullmatch(text):
        raise ValueError('bad_' + name + '_numeric')
    values = np.fromstring(text, sep='|', dtype=np.float64)
    if values.size != expected:
        raise ValueError('bad_' + name + '_length')
    if not np.all(np.isfinite(values)):
        raise ValueError('nonfinite_' + name)
    return values


def _field(site, name):
    return site[name] if isinstance(site, Mapping) else getattr(site, name)


class ParsedRead:
    """一行参考正向坐标排列的信号；信号数值与normalizer均按read缓存。"""

    @classmethod
    def from_fields(cls, parts):
        if len(parts) < 12:
            raise ValueError('bad_tsv_field_count')
        obj = cls()
        obj.read_id, obj.chrom = parts[0], parts[1]
        if not obj.read_id or not obj.chrom:
            raise ValueError('empty_read_or_reference_id')
        if any(c in obj.read_id + obj.chrom for c in '\t\r\n'):
            raise ValueError('bad_read_or_reference_id')
        try:
            obj.row_start = int(parts[2])
        except (ValueError, TypeError):
            raise ValueError('bad_start')
        if obj.row_start < 1:
            raise ValueError('bad_start')
        obj.ref_seq = parts[3].upper()
        if not obj.ref_seq or any(b not in 'ACGTNXY' for b in obj.ref_seq):
            raise ValueError('bad_reference_sequence')
        obj.row_end = obj.row_start + len(obj.ref_seq)-1
        obj.bam_strand, obj.tombo_strand = parts[5], parts[6]
        obj.strand_match = parts[7]
        if obj.bam_strand not in ('+', '-') or obj.tombo_strand not in ('+', '-'):
            raise ValueError('bad_strand')
        expected_match = '1' if obj.bam_strand == obj.tombo_strand else '0'
        if obj.strand_match != expected_match:
            raise ValueError('inconsistent_strand_match')
        length = len(obj.ref_seq)
        obj.qualities = _scalars(parts[4], length, 'quality')
        obj.mismatch = _scalars(parts[9], length, 'mismatch')
        obj.insertion = _scalars(parts[10], length, 'insertion')
        obj.deletion = _scalars(parts[11], length, 'deletion')
        obj.signal_s = parts[8]
        obj.signal_segments = tuple(obj.signal_s.split('|'))
        if len(obj.signal_segments) != length:
            raise ValueError('bad_signal_length')
        if any(not s for s in obj.signal_segments):
            raise ValueError('empty_signal_segment')
        obj.extra_fields = tuple(parts[12:])
        obj._flat_signal = None
        obj._signal_error = None
        obj._normalizers = {}
        obj.last_signal_error = None
        obj.last_window_error = None
        return obj

    def _ensure_signal(self):
        if self._signal_error is not None:
            raise ValueError(self._signal_error)
        if self._flat_signal is not None:
            return
        # 先检查完整语法，再调用fromstring，防止非法token导致静默截断。
        if not _SIGNAL_RE.fullmatch(self.signal_s):
            self._signal_error = 'bad_signal_numeric'
            raise ValueError(self._signal_error)
        count = self.signal_s.count('*') + self.signal_s.count('|') + 1
        with np.errstate(over='ignore', invalid='ignore'):
            flat = np.fromstring(self.signal_s.replace('|', '*'), sep='*', dtype=np.float32)
        if flat.size != count or not flat.size:
            self._signal_error = 'bad_full_signal_length'
            raise ValueError(self._signal_error)
        if not np.all(np.isfinite(flat)):
            self._signal_error = 'nonfinite_full_signal'
            raise ValueError(self._signal_error)
        lengths = tuple(s.count('*')+1 for s in self.signal_segments)
        offsets = np.empty(len(lengths)+1, dtype=np.int64)
        offsets[0] = 0
        np.cumsum(lengths, out=offsets[1:])
        if int(offsets[-1]) != flat.size:
            self._signal_error = 'bad_full_signal_length'
            raise ValueError(self._signal_error)
        self._flat_signal = flat
        self.segment_lengths = lengths
        self._segment_offsets = offsets

    def validate_full_signal(self):
        """审计整行信号，不计算MAD；失败原因保存在last_signal_error。"""
        try:
            self._ensure_signal()
        except ValueError as error:
            self.last_signal_error = str(error)
            return False
        self.last_signal_error = None
        return True

    def _window_bounds(self, lo1, hi1):
        if isinstance(lo1, bool) or isinstance(hi1, bool):
            raise ValueError('bad_window_coordinates')
        if int(lo1) != lo1 or int(hi1) != hi1:
            raise ValueError('bad_window_coordinates')
        lo1, hi1 = int(lo1), int(hi1)
        if hi1-lo1 != 5:
            raise ValueError('bad_window_length')
        left, right = lo1-self.row_start, hi1-self.row_start+1
        if left < 0 or right > len(self.ref_seq):
            raise ValueError('window_not_fully_covered')
        return left, right

    def validate_window(self, lo1, hi1):
        """窗口含六个有效位置，且全行normalizer输入合法时返回True。"""
        try:
            self._window_bounds(lo1, hi1)
            self._ensure_signal()
        except (ValueError, TypeError, OverflowError) as error:
            self.last_window_error = str(error)
            return False
        self.last_window_error = None
        return True

    def normalizer(self, mad_use_unique=True):
        self._ensure_signal()
        key = bool(mad_use_unique)
        if key not in self._normalizers:
            self._normalizers[key] = robust_mad_scale(self._flat_signal, use_unique=key)
        return self._normalizers[key]

    def _indices(self, lo1, hi1):
        left, right = self._window_bounds(lo1, hi1)
        indices = list(range(left, right))
        return indices if self.bam_strand == '+' else indices[::-1]

    def window_metadata(self, lo1, hi1):
        """只报告TSV中可核验的segment坐标，不推断原始FAST5采样偏移。"""
        self._ensure_signal()
        left, right = self._window_bounds(lo1, hi1)
        indices = self._indices(lo1, hi1)
        return {
            'row_segment_start_index_0based': left,
            'row_segment_end_index_exclusive_0based': right,
            'segment_indices_oriented_0based': '|'.join(str(i) for i in indices),
            'segment_lengths_oriented': '|'.join(str(self.segment_lengths[i]) for i in indices),
            'serialized_row_signal_start_index_0based': int(self._segment_offsets[left]),
            'serialized_row_signal_end_index_exclusive_0based': int(self._segment_offsets[right]),
            'signal_coordinate_system': 'serialized_row_concatenation_not_absolute_fast5',
        }

    def features(self, site, dwell_scale=1.0, quality_scale=1.0,
                 mad_use_unique=True, signal_round=6, stat_round=6):
        for value, name in [(dwell_scale, 'dwell_scale'), (quality_scale, 'quality_scale')]:
            if not math.isfinite(value) or value <= 0:
                raise ValueError('bad_' + name)
        if (isinstance(signal_round, bool) or isinstance(stat_round, bool)
                or not isinstance(signal_round, int) or not isinstance(stat_round, int)
                or not 0 <= signal_round <= 15 or not 0 <= stat_round <= 15):
            raise ValueError('bad_rounding_digits')
        lo1, hi1 = _field(site, 'kmer_ref_start'), _field(site, 'kmer_ref_end')
        left, right = self._window_bounds(lo1, hi1)
        raw = _field(site, 'raw_oriented_6mer')
        masked = _field(site, 'masked_6mer')
        true_base, label = _field(site, 'true_base'), _field(site, 'label')
        if (len(raw) != 6 or true_base not in LABELS or raw[3] != true_base
                or masked != raw[:3]+'N'+raw[4:]
                or any(b not in 'ACGT' for i, b in enumerate(raw) if i != 3)
                or label != LABELS[true_base]):
            raise ValueError('inconsistent_site_truth')
        observed = self.ref_seq[left:right]
        if self.bam_strand == '-':
            observed = revcomp_xy(observed)
        # NCB中心可能在旧TSV中由X_to_T参考替换；真实标签来自原始FASTA。
        if observed[:3]+observed[4:] != raw[:3]+raw[4:]:
            raise ValueError('window_reference_flank_mismatch')
        if true_base in 'ACGT' and observed != raw:
            raise ValueError('window_reference_target_mismatch')
        indices = self._indices(lo1, hi1)
        center, scale = self.normalizer(mad_use_unique)
        means, stds, medians, dwells, raw_norm = [], [], [], [], []
        for i in indices:
            start, end = self._segment_offsets[i:i+2]
            # 逆链只倒置六个segment，保留每个segment内既有时间顺序。
            with np.errstate(over='ignore', invalid='ignore', divide='ignore'):
                seg = (self._flat_signal[start:end]-center)/scale
                mean_value = np.round(np.mean(seg), stat_round)
                std_value = np.round(np.std(seg), stat_round)
                median_value = np.round(np.median(seg), stat_round)
            means.append(fmt_num(mean_value, stat_round))
            stds.append(fmt_num(std_value, stat_round))
            medians.append(fmt_num(median_value, stat_round))
            dwells.append(fmt_num(np.round(seg.size/dwell_scale, stat_round), stat_round))
            raw_norm.append('*'.join(fmt_num(v, signal_round) for v in seg))
        def scalar_output(array, divisor=1.0):
            return '|'.join(fmt_num(np.round(float(array[i])/divisor, stat_round), stat_round)
                            for i in indices)
        return [masked, '|'.join(means), '|'.join(stds), '|'.join(medians), '|'.join(dwells),
                scalar_output(self.qualities, quality_scale), scalar_output(self.mismatch),
                scalar_output(self.insertion), scalar_output(self.deletion), '|'.join(raw_norm), label]
