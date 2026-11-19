"""参考序列上的六碱基候选位置与匹配范围。

坐标均为参考正向的 1-based、两端包含；strand 只决定输出窗口的
排列方向。目标在最终输出中始终为第 4 位。原始含 X/Y 的参考用于
确定真值，不能用将 X 替换成 T 的比对参考替代。

common 范围是天然四类跨参考、跨方向汇总后与 X/Y 取共同 context，
并不表示六类在同一参考、同一方向均有支持。strict_match 另行标记
同参考、同方向的 NCB 与四天然类匹配；它仍比较不同参考位置。
"""

from collections import Counter, OrderedDict
import hashlib
import itertools
from pathlib import Path


LABELS = {"A": 0, "T": 1, "C": 2, "G": 3, "X": 4, "Y": 5}
COMP_TABLE_XY = str.maketrans("ACGTNXYacgtnxy", "TGCANYXtgcanyx")
_CANONICAL = frozenset("ACGT")
_ALLOWED = frozenset("ACGTNXY")


def revcomp_xy(sequence):
    """反向互补并交换 X/Y 化学碱基。"""
    return sequence.translate(COMP_TABLE_XY)[::-1]


def interval_for_target(target1, strand):
    """返回使最终目标处于第 4 位的正向参考区间。"""
    if not isinstance(target1, int) or isinstance(target1, bool) or target1 < 1:
        raise ValueError("target1 must be a positive integer in 1-based coordinates")
    if strand == "+":
        return target1 - 3, target1 + 2
    if strand == "-":
        return target1 - 2, target1 + 3
    raise ValueError("strand must be '+' or '-'")


def mask_center_base(sequence):
    if len(sequence) != 6:
        raise ValueError("Expected exactly six bases")
    return sequence[:3] + "N" + sequence[4:]


def _load_fasta(path):
    sequences = {}
    current = None
    with Path(path).open("r", encoding="utf-8-sig") as stream:
        for line_no, line in enumerate(stream, 1):
            line = line.strip()
            if not line:
                continue
            if line.startswith(">"):
                name = line[1:].split()[0] if line[1:].strip() else ""
                if not name:
                    raise ValueError("Empty FASTA identifier at line %d" % line_no)
                if name in sequences:
                    raise ValueError("Duplicate FASTA identifier: %s" % name)
                sequences[name] = []
                current = name
                continue
            if current is None:
                raise ValueError("FASTA sequence before first identifier at line %d" % line_no)
            sequence = line.upper()
            bad = set(sequence) - _ALLOWED
            if bad:
                raise ValueError("Unsupported reference characters %r in %s at line %d" %
                                 (sorted(bad), current, line_no))
            sequences[current].append(sequence)
    if not sequences:
        raise ValueError("Reference FASTA is empty")
    result = {name: "".join(parts) for name, parts in sequences.items()}
    empty = [name for name, sequence in result.items() if not sequence]
    if empty:
        raise ValueError("Empty reference sequence: %s" % empty[0])
    return result


def _all_contexts():
    return frozenset("".join(parts[:3]) + "N" + "".join(parts[3:])
                     for parts in itertools.product("ACGT", repeat=5))


class ReferenceLibrary:
    """解析实验参考并给出可用于提取的窗口候选。

    exclude_intervals 中的坐标是参考正向 1-based、两端包含。天然
    窗口只要任何一个碱基与排除区相交就剔除；任一窗口碱基距任一
    X/Y 位点 <= ncb_guard 时也剔除。X/Y 窗口本身不受这些天然窗口
    排除规则影响，但要求五个侧翼均为 A/T/C/G。

    contexts 是窗口白名单，绝不是参考或 read 白名单。共享骨架只
    存一份天然候选模板，最终候选的按参考缓存最多保留 128 项。
    返回 candidate_sites 的字典应视为只读。
    """

    def __init__(self, fasta_path, ncb_guard=10,
                 exclude_intervals=((1238, 1280),), context_scope="common"):
        self.source_path = str(Path(fasta_path).resolve())
        if not isinstance(ncb_guard, int) or isinstance(ncb_guard, bool) or ncb_guard < 0:
            raise ValueError("ncb_guard must be a nonnegative integer")
        if context_scope not in ("common", "all"):
            raise ValueError("context_scope must be 'common' or 'all'")
        intervals = []
        for interval in exclude_intervals:
            if len(interval) != 2:
                raise ValueError("Each excluded interval needs start and end")
            lo, hi = interval
            if (not isinstance(lo, int) or not isinstance(hi, int)
                    or isinstance(lo, bool) or isinstance(hi, bool)
                    or lo < 1 or hi < lo):
                raise ValueError("Excluded intervals must be 1-based inclusive positive intervals")
            intervals.append((lo, hi))
        self.ncb_guard = ncb_guard
        self.exclude_intervals = tuple(sorted(intervals))
        self.context_scope = context_scope
        self.sha256 = hashlib.sha256(Path(fasta_path).read_bytes()).hexdigest()
        self.sequences = _load_fasta(fasta_path)
        self._positions = {
            chrom: tuple(i + 1 for i, base in enumerate(sequence) if base in "XY")
            for chrom, sequence in self.sequences.items()
        }
        if not any(self._positions.values()):
            raise ValueError("Reference contains no X/Y NCB markers. Use the original experimental "
                             "FASTA, not refdb_X_to_T.fasta.")

        # 标记全部不安全参考位置；相同骨架签名共享天然候选模板。
        self._signatures = {}
        self._natural_templates = {}
        self._nc_templates = {}
        self._strict_contexts = {}
        self._candidate_cache = OrderedDict()
        self.natural_contexts_by_strand = {
            strand: {base: set() for base in "ATCG"} for strand in ("+", "-")
        }
        self.ncb_contexts_by_label = {4: set(), 5: set()}
        self.strict_reference_counts = {"+": 0, "-": 0}

        for chrom, sequence in self.sequences.items():
            signature = self._natural_signature(chrom)
            # dict.setdefault 会仍保留逐参考的大字符串；intern 型查找让
            # 相同骨架共享同一个字符串对象。
            key = signature
            for existing in self._natural_templates:
                if existing == signature:
                    key = existing
                    break
            self._signatures[chrom] = key
            if key not in self._natural_templates:
                self._natural_templates[key] = {
                    strand: self._build_natural_template(key, strand)
                    for strand in ("+", "-")
                }
                for strand, sites in self._natural_templates[key].items():
                    for site in sites:
                        self.natural_contexts_by_strand[strand][site["true_base"]].add(
                            site["masked_6mer"])
            for strand in ("+", "-"):
                natural = self._natural_templates[key][strand]
                by_base = {base: set() for base in "ATCG"}
                for site in natural:
                    by_base[site["true_base"]].add(site["masked_6mer"])
                natural4 = set.intersection(*(by_base[base] for base in "ATCG"))
                nc_sites = []
                nc_contexts = set()
                for target1 in self._positions[chrom]:
                    lo, hi = interval_for_target(target1, strand)
                    if lo < 1 or hi > len(sequence):
                        continue
                    window = self.get_window(chrom, target1, strand)
                    if any(base not in _CANONICAL for i, base in enumerate(window) if i != 3):
                        continue
                    site = self._record(target1, strand, window)
                    site["strict_match"] = site["masked_6mer"] in natural4
                    nc_sites.append(site)
                    nc_contexts.add(site["masked_6mer"])
                    self.ncb_contexts_by_label[site["label"]].add(site["masked_6mer"])
                self._nc_templates[(chrom, strand)] = tuple(nc_sites)
                # 天然窗口作为 strict 对照时，应匹配确实四天然齐备的
                # 同参考 NCB context，而不只要求某一种天然类出现。
                self._strict_contexts[(chrom, strand)] = frozenset(nc_contexts & natural4)
                if nc_contexts & natural4:
                    self.strict_reference_counts[strand] += 1

        natural_all = {
            base: (self.natural_contexts_by_strand["+"][base]
                   | self.natural_contexts_by_strand["-"][base]) for base in "ATCG"
        }
        common = set.intersection(*(natural_all[base] for base in "ATCG"),
                                  self.ncb_contexts_by_label[4],
                                  self.ncb_contexts_by_label[5])
        self.common_contexts = frozenset(common)
        self.contexts = self.common_contexts if context_scope == "common" else _all_contexts()
        self.strict_reference_counts_selected = {
            strand: sum(any(site["strict_match"] and site["masked_6mer"] in self.contexts
                            for site in self._nc_templates[(chrom, strand)])
                        for chrom in self.sequences)
            for strand in ("+", "-")
        }
        # 对外提供不可变集合，避免调用方意外改变后续候选范围。
        self.natural_contexts_by_strand = {
            strand: {base: frozenset(values) for base, values in by_base.items()}
            for strand, by_base in self.natural_contexts_by_strand.items()
        }
        self.ncb_contexts_by_label = {
            label: frozenset(values) for label, values in self.ncb_contexts_by_label.items()
        }
        lengths = Counter(map(len, self.sequences.values()))
        positions = Counter(pos for values in self._positions.values() for pos in values)
        self.summary = {
            "reference_count": len(self.sequences),
            "length_counts": dict(sorted(lengths.items())),
            "ncb_count": sum(len(values) for values in self._positions.values()),
            "ncb_position_counts_1based": dict(sorted(positions.items())),
            "ncb_per_reference_counts": dict(sorted(Counter(map(len, self._positions.values())).items())),
            "common_context_count": len(self.common_contexts),
            "selected_context_count": len(self.contexts),
            "context_scope": self.context_scope,
            "strict_reference_counts": dict(self.strict_reference_counts),
            "strict_reference_counts_scope": "before context whitelist filtering",
            "strict_reference_counts_selected": dict(self.strict_reference_counts_selected),
            "natural_template_signatures": len(self._natural_templates),
            "ncb_guard": self.ncb_guard,
            "exclude_intervals_1based_inclusive": [list(interval) for interval in self.exclude_intervals],
            "sha256": self.sha256,
        }

    @staticmethod
    def interval_for_target(target1, strand):
        return interval_for_target(target1, strand)

    def nc_positions(self, chrom):
        return self._positions[chrom]

    def get_window(self, chrom, target1, strand):
        lo, hi = interval_for_target(target1, strand)
        sequence = self.sequences[chrom]
        if lo < 1 or hi > len(sequence):
            raise ValueError("6mer target %d on %s is not fully inside the reference" %
                             (target1, chrom))
        window = sequence[lo-1:hi]
        return window if strand == "+" else revcomp_xy(window)

    def _natural_signature(self, chrom):
        sequence = self.sequences[chrom]
        blocked = list(sequence)
        intervals = list(self.exclude_intervals)
        intervals.extend((max(1, pos-self.ncb_guard), pos+self.ncb_guard)
                         for pos in self._positions[chrom])
        for lo, hi in intervals:
            lo0, hi0 = max(0, lo-1), min(len(sequence), hi)
            if lo0 < hi0:
                blocked[lo0:hi0] = "#" * (hi0-lo0)
        return "".join(blocked)

    @staticmethod
    def _record(target1, strand, window):
        lo, hi = interval_for_target(target1, strand)
        base = window[3]
        return {
            "target_pos": target1,
            "kmer_ref_start": lo,
            "kmer_ref_end": hi,
            "raw_oriented_6mer": window,
            "masked_6mer": mask_center_base(window),
            "true_base": base,
            "label": LABELS[base],
            "strict_match": False,
        }

    def _build_natural_template(self, signature, strand):
        result = []
        for start0 in range(len(signature)-5):
            window = signature[start0:start0+6]
            if any(base not in _CANONICAL for base in window):
                continue
            if strand == "+":
                target1 = start0 + 4
            else:
                target1 = start0 + 3
                window = revcomp_xy(window)
            result.append(self._record(target1, strand, window))
        return tuple(result)

    def candidate_sites(self, chrom, strand):
        """返回此参考方向中属于 context 白名单的全部候选窗口。

        包含天然与 NCB 候选。调用方应逐窗口按 read 覆盖筛选，不能因
        某条 read 的 NCB context 不在白名单就丢弃其天然窗口。
        """
        if strand not in ("+", "-"):
            raise ValueError("strand must be '+' or '-'")
        if chrom not in self.sequences:
            raise KeyError(chrom)
        cache_key = (chrom, strand)
        if cache_key in self._candidate_cache:
            self._candidate_cache.move_to_end(cache_key)
            return self._candidate_cache[cache_key]
        strict_contexts = self._strict_contexts[cache_key]
        natural = self._natural_templates[self._signatures[chrom]][strand]
        result = []
        for original in natural:
            if original["masked_6mer"] not in self.contexts:
                continue
            site = dict(original)
            site["strict_match"] = site["masked_6mer"] in strict_contexts
            result.append(site)
        result.extend(dict(site) for site in self._nc_templates[cache_key]
                      if site["masked_6mer"] in self.contexts)
        result.sort(key=lambda site: (site["target_pos"], site["label"]))
        result = tuple(result)
        self._candidate_cache[cache_key] = result
        if len(self._candidate_cache) > 128:
            self._candidate_cache.popitem(last=False)
        return result
