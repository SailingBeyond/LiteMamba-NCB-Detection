#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Make optional whole-group six-class holdouts from the existing 7:2:1 splits.

This retains every row and never invents a reference, run or template identity.
The default positive CSV-row proxies cannot establish these identities. Unknown
required groups cause an explicit rejection, rather than silently dropping X/Y.
Group ratios approximate 7:2:1; exact window/class balance is not promised.
Numeric feature payloads are checked by the training input audit, not here.
"""
from __future__ import annotations

import argparse
from collections import Counter, defaultdict
import csv
from decimal import Decimal, InvalidOperation
import hashlib
from itertools import zip_longest
import json
from pathlib import Path
import random
import shutil
import sqlite3
import sys
import tempfile

VERSION = "sixclass-known-group-holdout-v1.0.0"
SPLITS = ("train", "validation", "test")
RATIOS = (0.7, 0.2, 0.1)
LABEL_NAMES = ("A", "T", "C", "G", "X", "Y")
FEATURE_COLUMNS = ("kmer", "mean", "std", "median", "dwell", "quality", "mismatch", "insertion", "deletion", "signal", "label")
UNKNOWN = {"", "unknown", "na", "n/a", "none", "null", "nan", "."}
PROVENANCE_COLUMNS = ("holdout_group", "holdout_group_key", "holdout_split", "holdout_previous_split", "holdout_source_feature", "holdout_source_row", "holdout_row_index")


class HoldoutValidationError(ValueError):
    pass


def known(value):
    value = str(value if value is not None else "").strip()
    return None if value.lower() in UNKNOWN else value


def sixclass_label(value, where="label"):
    try:
        decimal = Decimal(str(value).strip())
    except (InvalidOperation, ValueError, TypeError) as exc:
        raise HoldoutValidationError(f"{where}: label must be an integer in 0..5") from exc
    if not decimal.is_finite() or decimal != decimal.to_integral_value() or not 0 <= decimal <= 5:
        raise HoldoutValidationError(f"{where}: label must be an integer in 0..5; got {value!r}")
    return int(decimal)


def masked(sequence, where="kmer"):
    sequence = str(sequence).strip().upper()
    if len(sequence) != 6 or any(base not in "ATCGN" and not (i == 3 and base in "XY") for i, base in enumerate(sequence)):
        raise HoldoutValidationError(f"{where}: invalid 6mer {sequence!r}")
    return sequence[:3] + "N" + sequence[4:]


def row_proxy(meta, label):
    # Import the same provenance validator as the neural training loader. This
    # keeps acceptance of unknown positive references identical across helpers.
    from IterDataset import is_positive_row_proxy
    proxy = is_positive_row_proxy(meta, label)
    if (meta.get("read_id_kind") == "positive_row_proxy" or str(meta.get("read_id", "")).startswith("positive-row:")) and not proxy:
        raise HoldoutValidationError("Malformed positive row proxy or conflicting provenance flags")
    return proxy


class TrackedLines:
    def __init__(self, path):
        self.path = Path(path)
        self.before = self.path.stat()
        self.digest = hashlib.sha256()
        self.handle = self.path.open("rb")
        self.first = True

    def __iter__(self):
        return self

    def __next__(self):
        value = next(self.handle)
        self.digest.update(value)
        result = value.decode("utf-8-sig" if self.first else "utf-8")
        self.first = False
        return result

    def fingerprint(self):
        after = self.path.stat()
        if (after.st_size, after.st_mtime_ns) != (self.before.st_size, self.before.st_mtime_ns):
            raise HoldoutValidationError(f"Input changed during reading: {self.path}")
        return {"path": str(self.path), "size": after.st_size, "sha256": self.digest.hexdigest()}

    def close(self):
        self.handle.close()


def discover_sources(data_dir):
    sources, union = [], []
    for split in SPLITS:
        feature, metadata = data_dir / f"{split}.csv", data_dir / f"{split}_meta.tsv"
        if not feature.is_file() or not metadata.is_file():
            raise HoldoutValidationError(f"Required paired files: {feature} and {metadata}")
        with metadata.open(encoding="utf-8-sig", newline="") as handle:
            header = next(csv.reader(handle, delimiter="\t"), [])
        if len(set(header)) != len(header) or not {"read_id", "chrom", "bam_strand", "label"}.issubset(header):
            raise HoldoutValidationError(f"{metadata}: unique read_id/chrom/bam_strand/label columns required")
        if not any(key in header for key in ("kmer", "masked_kmer", "masked_6mer", "output_kmer", "6mer")):
            raise HoldoutValidationError(f"{metadata}: row-aligned kmer column required")
        union.extend(name for name in header if name not in union)
        sources.append({"previous_split": split, "feature": feature, "metadata": metadata})
    union.extend(name for name in PROVENANCE_COLUMNS if name not in union)
    return sources, union


def paired_rows(source):
    features, metadata = TrackedLines(source["feature"]), TrackedLines(source["metadata"])
    try:
        feature_reader = csv.reader(features)
        meta_reader = csv.DictReader(metadata, delimiter="\t")
        sentinel = object()
        def rows():
            for ordinal, row in enumerate(feature_reader):
                if ordinal == 0 and row and row[0].strip().lower() == "kmer":
                    if tuple(value.strip().lower() for value in row) != FEATURE_COLUMNS:
                        raise HoldoutValidationError(f"{source['feature']}: incompatible feature header")
                    continue
                yield feature_reader.line_num, row
        for item, meta in zip_longest(rows(), meta_reader, fillvalue=sentinel):
            if item is sentinel or meta is sentinel:
                raise HoldoutValidationError(f"Feature/metadata row counts differ: {source['feature']}")
            line, row = item
            where = f"{source['feature']}:{line}"
            if len(row) != 11 or None in meta or any(value is None for value in meta.values()):
                raise HoldoutValidationError(f"{where}: malformed feature/metadata row")
            label = sixclass_label(row[-1], where)
            motif = masked(row[0], where)
            sequences = [meta[key] for key in ("kmer", "masked_kmer", "masked_6mer", "output_kmer", "6mer") if known(meta.get(key))]
            if not sequences or any(masked(value, where + " metadata") != motif for value in sequences):
                raise HoldoutValidationError(f"{where}: feature/metadata kmer mismatch")
            if label != sixclass_label(meta["label"], where + " metadata"):
                raise HoldoutValidationError(f"{where}: feature/metadata label mismatch")
            read_id = known(meta["read_id"])
            if read_id is None or read_id != meta["read_id"]:
                raise HoldoutValidationError(f"{where}: known read_id without surrounding whitespace required")
            proxy = row_proxy(meta, label)
            if known(meta["chrom"]) is None and not proxy:
                raise HoldoutValidationError(f"{where}: unknown chrom allowed only for disclosed positive row proxies")
            for key in ("label_base", "true_base"):
                if known(meta.get(key)) and meta[key].upper() != LABEL_NAMES[label]:
                    raise HoldoutValidationError(f"{where}: {key} disagrees with six-class label")
            row[0], row[-1] = motif, str(label)
            meta["label"] = str(label)
            for key in ("kmer", "masked_kmer", "masked_6mer", "output_kmer", "6mer"):
                if key in meta:
                    meta[key] = motif
            yield line, row, meta, {"label": label, "read_id": read_id, "motif": motif, "proxy": proxy}
        source["last_fingerprints"] = {"feature": features.fingerprint(), "metadata": metadata.fingerprint()}
    finally:
        features.close()
        metadata.close()


def grouping_key(meta, info, group_key, where):
    value = known(meta.get(group_key))
    if info["proxy"] and group_key in {"chrom", "run_id", "template_group", "template_id", "pair_id", "template_pair", "bam_strand"}:
        raise HoldoutValidationError(
            f"{where}: positive CSV-row proxies have no verified {group_key}. This optional holdout is unavailable for the current positive CSV-only inputs; provide genuine row-aligned extraction identities for this experiment. Do not infer identities from labels, motifs or file paths."
        )
    if value is None:
        raise HoldoutValidationError(f"{where}: {group_key} is unknown; whole-group holdout cannot include all six classes without this identity. No rows were dropped.")
    if group_key in {"chrom", "template_group", "template_id", "pair_id", "template_pair"} and str(meta.get("reference_identity_known", "true")).lower() == "false":
        raise HoldoutValidationError(f"{where}: {group_key} is marked unverified")
    return value


def allocate_group_counts(n):
    if n < 3:
        raise HoldoutValidationError(f"At least three known groups are required; found {n}")
    counts = [max(1, int(n * ratio)) for ratio in RATIOS]
    while sum(counts) > n:
        candidates = [i for i, count in enumerate(counts) if count > 1]
        counts[max(candidates, key=lambda i: counts[i] - n * RATIOS[i])] -= 1
    while sum(counts) < n:
        counts[max(range(3), key=lambda i: n * RATIOS[i] - counts[i])] += 1
    return dict(zip(SPLITS, counts))


def plan_assignment(groups, seed, max_attempts):
    counts = allocate_group_counts(len(groups))
    per_class_groups = {label: sum(counter[label] > 0 for counter in groups.values()) for label in range(6)}
    if any(count < 3 for count in per_class_groups.values()):
        raise HoldoutValidationError(f"Every class needs at least three distinct known groups for train/validation/test coverage; found {per_class_groups}. A whole group will never be split to fill a missing class.")
    generator, keys = random.Random(seed), sorted(groups)
    for attempt in range(1, max_attempts + 1):
        order = keys.copy()
        generator.shuffle(order)
        assignment, cursor, coverage = {}, 0, {split: Counter() for split in SPLITS}
        for split in SPLITS:
            for group in order[cursor:cursor + counts[split]]:
                assignment[group] = split
                coverage[split].update(groups[group])
            cursor += counts[split]
        if all(counter[label] > 0 for counter in coverage.values() for label in range(6)):
            return assignment, counts, attempt
    raise HoldoutValidationError(f"No all-six-class assignment found in {max_attempts} metadata-only seeded attempts. Groups={counts}; add suitable known groups or change the declared grouping experiment. No model scores were consulted.")


def write_json(path, value):
    Path(path).write_text(json.dumps(value, ensure_ascii=False, indent=2, sort_keys=True, allow_nan=False) + "\n", encoding="utf-8")


def prepare_holdout(data_dir, output_dir, group_key="template_group", seed=42, max_attempts=10000, log_every=100000):
    data_dir, output_dir = Path(data_dir).expanduser().resolve(), Path(output_dir).expanduser().resolve()
    if group_key not in {"template_group", "template_id", "template_pair", "pair_id", "chrom", "run_id", "source_dataset"}:
        raise ValueError("Choose an explicit known metadata group key")
    if max_attempts < 1:
        raise ValueError("max_attempts must be positive")
    if output_dir == data_dir or output_dir in data_dir.parents or data_dir in output_dir.parents:
        raise ValueError("Use separate, non-nested input/output directories")
    if output_dir.exists():
        raise FileExistsError(f"Output already exists; choose a new path: {output_dir}")
    sources, columns = discover_sources(data_dir)
    output_dir.parent.mkdir(parents=True, exist_ok=True)
    temporary = Path(tempfile.mkdtemp(prefix=output_dir.name + ".building-", dir=output_dir.parent))
    connection = sqlite3.connect(str(temporary / "read_membership.sqlite"))
    connection.execute("CREATE TABLE source_reads(read_id TEXT PRIMARY KEY, group_id TEXT, previous_split TEXT, windows INTEGER, proxy INTEGER, conflict INTEGER)")
    groups, input_rows, proxy_rows = defaultdict(Counter), 0, 0
    try:
        for source in sources:
            source_rows, batch = 0, []
            for line, row, meta, info in paired_rows(source):
                group = grouping_key(meta, info, group_key, f"{source['feature']}:{line}")
                groups[group][info["label"]] += 1
                batch.append((info["read_id"], group, source["previous_split"], 1, int(info["proxy"]), 0))
                source_rows += 1
                proxy_rows += int(info["proxy"])
                if len(batch) >= 5000:
                    _store_reads(connection, batch)
                    batch.clear()
                if log_every > 0 and source_rows % log_every == 0:
                    print(f"[scan] {source['feature'].name}: {source_rows:,}", flush=True)
            _store_reads(connection, batch)
            if source_rows == 0:
                raise HoldoutValidationError(f"Empty source split: {source['feature']}")
            source["fingerprints"], source["sample_count"] = source.pop("last_fingerprints"), source_rows
            input_rows += source_rows
        connection.commit()
        conflicts = connection.execute("SELECT read_id,group_id,previous_split FROM source_reads WHERE conflict=1 OR (proxy=1 AND windows>1) LIMIT 20").fetchall()
        if conflicts:
            raise HoldoutValidationError(f"Read occurs across original splits/groups, or positive row proxy repeats: {conflicts}. Different labels on the same negative read are allowed within one split/group.")
        assignment, group_counts, attempts = plan_assignment(groups, int(seed), max_attempts)
        connection.execute("CREATE TABLE group_assignment(group_id TEXT PRIMARY KEY, split TEXT)")
        connection.executemany("INSERT INTO group_assignment VALUES (?,?)", sorted(assignment.items()))
        writers, handles = {}, []
        stats = {split: {"sample_count": 0, "class_counts": Counter(), "source_counts": Counter(), "read_id_kind_counts": Counter()} for split in SPLITS}
        try:
            for split in SPLITS:
                f = (temporary / f"{split}.csv").open("w", encoding="utf-8", newline="")
                m = (temporary / f"{split}_meta.tsv").open("w", encoding="utf-8", newline="")
                handles.extend((f, m))
                fw, mw = csv.writer(f, lineterminator="\n"), csv.DictWriter(m, fieldnames=columns, delimiter="\t", lineterminator="\n")
                fw.writerow(FEATURE_COLUMNS)
                mw.writeheader()
                writers[split] = fw, mw
            output_rows = 0
            for source in sources:
                for line, row, meta, info in paired_rows(source):
                    group = grouping_key(meta, info, group_key, f"{source['feature']}:{line}")
                    split = assignment[group]
                    record = dict(meta)
                    record.update(holdout_group=group, holdout_group_key=group_key, holdout_split=split, holdout_previous_split=source["previous_split"], holdout_source_feature=str(source["feature"]), holdout_source_row=line, holdout_row_index=stats[split]["sample_count"])
                    if "split" in record:
                        record["split"] = split
                    writers[split][0].writerow(row)
                    writers[split][1].writerow(record)
                    stats[split]["sample_count"] += 1
                    stats[split]["class_counts"][str(info["label"])] += 1
                    stats[split]["source_counts"][known(meta.get("source_dataset")) or "unknown"] += 1
                    stats[split]["read_id_kind_counts"]["positive_row_proxy" if info["proxy"] else "sequencing_read_id"] += 1
                    output_rows += 1
                if source.pop("last_fingerprints") != source["fingerprints"]:
                    raise HoldoutValidationError(f"Input changed between passes: {source['feature']}")
        finally:
            for handle in handles:
                handle.close()
        if output_rows != input_rows:
            raise HoldoutValidationError("Output sample count differs from input")
        for i, split in enumerate(SPLITS):
            stats[split]["sample_fraction"] = stats[split]["sample_count"] / input_rows
            stats[split]["sample_fraction_minus_requested"] = stats[split]["sample_fraction"] - RATIOS[i]
            stats[split]["groups"] = sorted(group for group, assigned in assignment.items() if assigned == split)
            stats[split]["unique_read_or_proxy_count"] = connection.execute("SELECT COUNT(*) FROM source_reads JOIN group_assignment USING(group_id) WHERE split=?", (split,)).fetchone()[0]
        with (temporary / "group_assignment.csv").open("w", encoding="utf-8", newline="") as handle:
            writer = csv.writer(handle, lineterminator="\n")
            writer.writerow(["group_id", "split", "sample_count"] + [f"label_{label}" for label in range(6)])
            for group in sorted(groups):
                writer.writerow([group, assignment[group], sum(groups[group].values())] + [groups[group][label] for label in range(6)])
        summary = {"status": "PASS", "version": VERSION, "group_key": group_key, "seed": int(seed), "label_mapping": dict(enumerate(LABEL_NAMES)), "requested_split_ratios": dict(zip(SPLITS, RATIOS)), "actual_group_counts": group_counts, "seeded_assignment_attempts": attempts, "input_sample_count": input_rows, "output_sample_count": output_rows, "samples_discarded": 0, "split_summary": stats, "input_fingerprints": [{"previous_split": source["previous_split"], "sample_count": source["sample_count"], **source["fingerprints"]} for source in sources], "read_cross_split_overlaps": 0, "group_cross_split_overlaps": 0, "positive_row_proxy_rows": proxy_rows, "read_identity_scope": "real negative/provided read IDs plus disclosed positive row proxies; original positive UUID independence is not independently verified", "independent_new_experiment": False, "assignment_policy": "first fixed-seed whole-group assignment with six-class coverage in every split; metadata/labels only, no model metrics", "scientific_limitations": ["Previously supplied samples are reused; this is not a newly collected experiment.", "Every row is retained once. Whole-group splitting need not retain exact per-class balance or 7:2:1 window fractions.", "Groups are explicit metadata values. A motif, path, class label or positive row proxy never establishes a run/template/reference identity.", "The E. coli A/T/C/G versus synthetic X/Y source confound cannot be removed by relabeling metadata or repartitioning these inputs."]}
        write_json(temporary / "summary.json", summary)
        write_json(temporary / "holdout_config.json", {"data_dir": str(data_dir), "output_dir": str(output_dir), "group_key": group_key, "seed": int(seed), "max_attempts": max_attempts})
        connection.commit()
        connection.close()
        temporary.rename(output_dir)
        print(f"[PASS] {group_key}: {len(groups)} whole groups, {input_rows:,} windows; {output_dir}", flush=True)
        return summary
    except BaseException:
        connection.close()
        shutil.rmtree(temporary, ignore_errors=True)
        raise


def _store_reads(connection, batch):
    if batch:
        connection.executemany("INSERT INTO source_reads VALUES (?,?,?,?,?,?) ON CONFLICT(read_id) DO UPDATE SET windows=windows+1, conflict=conflict OR group_id!=excluded.group_id OR previous_split!=excluded.previous_split OR proxy!=excluded.proxy", batch)


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument("--data_dir", required=True)
    parser.add_argument("--output_dir", required=True)
    parser.add_argument("--group_key", choices=("template_group", "template_id", "template_pair", "pair_id", "chrom", "run_id", "source_dataset"), default="template_group")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--max_attempts", type=int, default=10000)
    parser.add_argument("--log_every", type=int, default=100000)
    args = parser.parse_args(argv)
    try:
        prepare_holdout(**vars(args))
    except (ValueError, OSError, sqlite3.Error) as exc:
        print(f"[REJECTED] {exc}", file=sys.stderr, flush=True)
        raise SystemExit(2)


if __name__ == "__main__":
    main()
