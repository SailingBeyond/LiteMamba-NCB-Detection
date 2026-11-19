#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Regroup existing balanced samples into template- or run-held-out splits.

This operation retains every supplied sample exactly once. It neither extracts
new features nor creates an independently collected sequencing experiment.
Group assignments are shared between pos and neg, use metadata only, and never
use model performance. Missing grouping identities or read leakage cause failure.
The script uses only Python's standard library and streams CSV/TSV pairs twice.
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
import re
import shutil
import sqlite3
import sys
import tempfile


SPLITS = ("train", "val", "test")
RATIOS = (0.7, 0.2, 0.1)
FEATURE_COLUMNS = (
    "kmer", "mean", "std", "median", "dwell", "quality", "mismatch",
    "insertion", "deletion", "signal", "label",
)
REQUIRED_META = {"read_id", "chrom", "bam_strand", "kmer", "label"}
UNKNOWN_VALUES = {"", "unknown", "na", "n/a", "none", "null", "nan"}
PROVENANCE_COLUMNS = (
    "holdout_group", "holdout_group_by", "holdout_split", "holdout_previous_split",
    "holdout_source_feature", "holdout_source_row", "holdout_row_index",
    "pair_id", "holdout_original_pair_id",
)


class HoldoutValidationError(ValueError):
    pass


def known(value):
    value = str(value or "").strip()
    return value if value.lower() not in UNKNOWN_VALUES else None


def binary_label(value, where):
    try:
        result = Decimal(str(value).strip())
    except (InvalidOperation, TypeError, ValueError) as exc:
        raise HoldoutValidationError(f"{where}: invalid binary label {value!r}") from exc
    if not result.is_finite() or result not in (0, 1):
        raise HoldoutValidationError(f"{where}: binary labels must be 0 or 1; got {value!r}")
    return int(result)


def normalize_pair(value):
    value = known(value)
    if value is None:
        return None
    match = re.match(r"^(?:PC|DNA|XNA)[_-]?0*(\d+)(?:$|[^0-9])", value, re.I)
    numeric = re.fullmatch(r"(?:pair[_-]?)?0*(\d+)", value, re.I)
    if match or numeric:
        return f"pair_{int((match or numeric).group(1))}"
    return value


def template_identity(meta, where):
    """Accept an explicit paired identity, or documented PC/DNA/XNA numbering."""
    explicit = [
        normalize_pair(meta.get(name))
        for name in ("template_pair", "pair_id", "template_group")
        if known(meta.get(name)) is not None
    ]
    # A template_id may be an arbitrary oligo identifier; authoritative paired
    # fields take precedence instead of falsely comparing unlike identifiers.
    if not explicit and known(meta.get("template_id")) is not None:
        explicit = [normalize_pair(meta["template_id"])]
    if len(set(explicit)) > 1:
        raise HoldoutValidationError(f"{where}: conflicting explicit template identities: {explicit}")
    chrom = str(meta.get("chrom", "")).strip()
    if re.match(r"^(?:PC|DNA|XNA)[_-]?0*\d+(?:$|[^0-9])", chrom, re.I):
        reference_pair = normalize_pair(chrom)
        if explicit and explicit[0] != reference_pair:
            raise HoldoutValidationError(
                f"{where}: explicit template pair {explicit[0]!r} conflicts with known paired "
                f"PC/DNA/XNA reference {chrom!r} ({reference_pair}); paired controls cannot be separated"
            )
        return reference_pair, "explicit_metadata_verified_against_paired_reference" if explicit else "paired_PC_DNA_XNA_reference_number"
    if explicit:
        return explicit[0], "explicit_metadata"
    return None, "unknown"


class TrackedLines:
    def __init__(self, path):
        self.path = Path(path)
        self.before = self.path.stat()
        self.digest = hashlib.sha256()
        self.stream = self.path.open("rb")
        self.first = True

    def __iter__(self):
        return self

    def __next__(self):
        block = next(self.stream)
        self.digest.update(block)
        encoding = "utf-8-sig" if self.first else "utf-8"
        self.first = False
        return block.decode(encoding)

    def close(self):
        self.stream.close()

    def fingerprint(self):
        after = self.path.stat()
        if (self.before.st_size, self.before.st_mtime_ns) != (after.st_size, after.st_mtime_ns):
            raise HoldoutValidationError(f"Input changed while reading: {self.path}")
        return {
            "path": str(self.path), "size": after.st_size, "mtime_ns": after.st_mtime_ns,
            "sha256": self.digest.hexdigest(),
        }


def find_metadata(feature):
    candidates = (
        feature.with_name(feature.stem + "_meta.tsv"),
        Path(str(feature) + ".meta.tsv"), feature.with_suffix(".meta.tsv"),
    )
    matches = list(dict.fromkeys(path for path in candidates if path.is_file()))
    if len(matches) != 1:
        raise HoldoutValidationError(
            f"Exactly one row-aligned metadata file is required for {feature}; found {matches}"
        )
    return matches[0]


def discover_sources(data_dir, strands):
    sources, header_union = [], []
    for strand in strands:
        for split in SPLITS:
            feature = data_dir / f"xna_pc_6mer_features_{strand}_balanced_{split}.csv"
            if not feature.is_file():
                raise FileNotFoundError(feature)
            metadata = find_metadata(feature)
            with metadata.open("r", newline="", encoding="utf-8-sig") as stream:
                header = next(csv.reader(stream, delimiter="\t"), [])
            header = [name.strip() for name in header]
            if not header or len(set(header)) != len(header) or not REQUIRED_META.issubset(header):
                raise HoldoutValidationError(f"{metadata}: metadata needs unique columns including {sorted(REQUIRED_META)}")
            for name in header:
                if name not in header_union:
                    header_union.append(name)
            sources.append(dict(strand=strand, previous_split=split, feature=feature, metadata=metadata))
    for name in PROVENANCE_COLUMNS:
        if name not in header_union:
            header_union.append(name)
    return sources, header_union


def paired_rows(source):
    feature_lines, metadata_lines = TrackedLines(source["feature"]), TrackedLines(source["metadata"])
    try:
        feature_reader = csv.reader(feature_lines)
        meta_reader = csv.DictReader(metadata_lines, delimiter="\t")
        if not meta_reader.fieldnames:
            raise HoldoutValidationError(f"{source['metadata']}: empty metadata")
        meta_reader.fieldnames = [name.strip() for name in meta_reader.fieldnames]

        def features():
            for ordinal, row in enumerate(feature_reader):
                if ordinal == 0 and row and row[0].strip().lower() == "kmer":
                    if tuple(value.strip().lower() for value in row) != FEATURE_COLUMNS:
                        raise HoldoutValidationError(f"{source['feature']}: incompatible feature header")
                    continue
                yield feature_reader.line_num, row

        sentinel = object()
        for feature_item, meta in zip_longest(features(), meta_reader, fillvalue=sentinel):
            if feature_item is sentinel or meta is sentinel:
                raise HoldoutValidationError(f"CSV/metadata row counts differ: {source['feature']}")
            line, row = feature_item
            where = f"{source['feature']}:{line}"
            if len(row) != 11 or None in meta or any(value is None for value in meta.values()):
                raise HoldoutValidationError(f"{where}: malformed CSV/metadata row")
            if any("\n" in value or "\r" in value for value in row + list(meta.values())):
                raise HoldoutValidationError(f"{where}: multiline feature/metadata fields are unsupported")
            sequence = row[0].strip().upper()
            meta_sequence = meta["kmer"].strip().upper()
            if len(sequence) != 6 or any(
                base not in "ACTGN" and not (index == 3 and base in "XY")
                for index, base in enumerate(sequence)
            ):
                raise HoldoutValidationError(f"{where}: invalid six-base kmer {sequence!r}")
            label = binary_label(row[-1], where)
            if sequence != meta_sequence or label != binary_label(meta["label"], where + " metadata"):
                raise HoldoutValidationError(f"{where}: feature/metadata sequence or label mismatch")
            read_id, chrom = known(meta["read_id"]), known(meta["chrom"])
            expected_strand = "+" if source["strand"] == "pos" else "-"
            if read_id is None or chrom is None or meta["bam_strand"].strip() != expected_strand:
                raise HoldoutValidationError(f"{where}: known read_id/chrom and expected strand {expected_strand} are required")
            reference_family = re.match(r"^(PC|DNA|XNA)(?:[_-]?\d|$)", chrom, re.I)
            if reference_family:
                expected_label = int(reference_family.group(1).upper() == "XNA")
                if label != expected_label:
                    raise HoldoutValidationError(f"{where}: reference family disagrees with binary label")
            pair_id, pair_basis = template_identity(meta, where)
            info = dict(
                label=label, read_id=read_id, chrom=chrom,
                motif=sequence[:3] + "N" + sequence[4:],
                template_pair=pair_id or "unknown", template_identity_basis=pair_basis,
                run_id=known(meta.get("run_id")) or "unknown",
                source_dataset=known(meta.get("source_dataset")) or "unknown",
            )
            yield line, row, meta, info
        source["last_fingerprints"] = {
            "feature": feature_lines.fingerprint(), "metadata": metadata_lines.fingerprint(),
        }
    finally:
        feature_lines.close()
        metadata_lines.close()


def grouping_key(info, group_by, where):
    group = info["template_pair" if group_by == "template" else "run_id"]
    if known(group) is None:
        field = "an explicit paired template identity or recognized PC/DNA/XNA reference number" if group_by == "template" else "a real metadata run_id"
        raise HoldoutValidationError(f"{where}: {group_by} holdout requires {field}; identity is unknown")
    return group


def allocate_group_counts(n):
    if n < 3:
        raise HoldoutValidationError(f"At least three known groups are required, found {n}")
    counts = [max(1, int(n * ratio)) for ratio in RATIOS]
    while sum(counts) > n:
        eligible = [i for i, count in enumerate(counts) if count > 1]
        counts[max(eligible, key=lambda i: counts[i] - n * RATIOS[i])] -= 1
    while sum(counts) < n:
        counts[max(range(3), key=lambda i: n * RATIOS[i] - counts[i])] += 1
    return dict(zip(SPLITS, counts))


def plan_assignment(groups, strands, seed, max_attempts):
    counts = allocate_group_counts(len(groups))
    keys, generator = sorted(groups), random.Random(seed)
    for attempt in range(1, max_attempts + 1):
        order = keys.copy()
        generator.shuffle(order)
        assignment, cursor = {}, 0
        coverage = {(strand, split): Counter() for strand in strands for split in SPLITS}
        for split in SPLITS:
            for group in order[cursor:cursor + counts[split]]:
                assignment[group] = split
                for strand in strands:
                    coverage[strand, split].update(groups[group][strand])
            cursor += counts[split]
        if all(counter[0] > 0 and counter[1] > 0 for counter in coverage.values()):
            return assignment, counts, attempt
    raise HoldoutValidationError(
        f"No assignment with both binary classes in every requested strand/split was found in {max_attempts} "
        f"seeded attempts at group counts {counts}. Supply more groups or a suitable group annotation; "
        "the script will not split a group or reuse a read to fill missing classes."
    )


def empty_stats():
    return {
        "sample_count": 0, "class_counts": Counter(), "reference_counts": Counter(),
        "motif_counts": Counter(), "source_dataset_counts": Counter(), "run_counts": Counter(),
        "template_pair_counts": Counter(), "template_identity_basis_counts": Counter(),
        "reference_class_counts": defaultdict(Counter), "motif_class_counts": defaultdict(Counter),
        "source_class_counts": defaultdict(Counter), "run_class_counts": defaultdict(Counter),
    }


def update_stats(stats, info):
    stats["sample_count"] += 1
    label = str(info["label"])
    stats["class_counts"][label] += 1
    for info_key, count_key in (
        ("chrom", "reference_counts"), ("motif", "motif_counts"),
        ("source_dataset", "source_dataset_counts"), ("run_id", "run_counts"),
        ("template_pair", "template_pair_counts"), ("template_identity_basis", "template_identity_basis_counts"),
    ):
        stats[count_key][info[info_key]] += 1
    for info_key, count_key in (
        ("chrom", "reference_class_counts"), ("motif", "motif_class_counts"),
        ("source_dataset", "source_class_counts"), ("run_id", "run_class_counts"),
    ):
        stats[count_key][info[info_key]][label] += 1


def write_json(path, value):
    with path.open("w", encoding="utf-8") as stream:
        json.dump(value, stream, ensure_ascii=False, sort_keys=True, indent=2)
        stream.write("\n")


def prepare_holdout(data_dir, output_dir, group_by="template", strands=("pos", "neg"),
                    seed=42, max_attempts=10000, log_every=100000):
    data_dir = Path(data_dir).expanduser().resolve()
    output_dir = Path(output_dir).expanduser().resolve()
    strands = tuple(strands)
    if group_by not in ("template", "run") or not strands or len(set(strands)) != len(strands) or set(strands) - {"pos", "neg"}:
        raise ValueError("group_by must be template/run and strands must be distinct pos/neg values")
    if max_attempts < 1:
        raise ValueError("max_attempts must be positive")
    if output_dir == data_dir or output_dir in data_dir.parents or data_dir in output_dir.parents:
        raise ValueError("Use separate, non-nested input/output directories")
    if output_dir.exists():
        raise FileExistsError(f"Output directory already exists; choose a new path: {output_dir}")
    sources, meta_columns = discover_sources(data_dir, strands)
    output_dir.parent.mkdir(parents=True, exist_ok=True)
    temporary = Path(tempfile.mkdtemp(prefix=output_dir.name + ".building-", dir=output_dir.parent))
    database = temporary / "read_membership.sqlite"
    connection = sqlite3.connect(str(database))
    connection.execute("PRAGMA journal_mode=DELETE")
    connection.execute("CREATE TABLE source_reads(read_id TEXT PRIMARY KEY, group_id TEXT, previous_split TEXT, binary_label INTEGER, template_pair TEXT, chrom TEXT, windows INTEGER, conflict INTEGER)")
    connection.execute("CREATE TABLE output_read_membership(read_id TEXT, strand TEXT, split TEXT, windows INTEGER, PRIMARY KEY(read_id,strand,split))")
    groups = defaultdict(lambda: {strand: Counter() for strand in strands})
    input_rows, input_fingerprints, reference_pairs = 0, [], {}
    try:
        for source in sources:
            batch, source_rows = [], 0
            for line, row, meta, info in paired_rows(source):
                group = grouping_key(info, group_by, f"{source['feature']}:{line}")
                previous_pair = reference_pairs.setdefault(info["chrom"], info["template_pair"])
                if previous_pair != info["template_pair"]:
                    raise HoldoutValidationError(
                        f"{source['feature']}:{line}: the same reference/chrom {info['chrom']!r} "
                        f"has conflicting template pairs {previous_pair!r} and {info['template_pair']!r}; "
                        "a physical reference cannot be split into different template groups"
                    )
                groups[group][source["strand"]][info["label"]] += 1
                batch.append((
                    info["read_id"], group, source["previous_split"], info["label"],
                    info["template_pair"], info["chrom"], 1, 0,
                ))
                source_rows += 1
                if len(batch) >= 5000:
                    _store_source_reads(connection, batch)
                    batch.clear()
                if log_every > 0 and source_rows % log_every == 0:
                    print(f"[scan] {source['feature'].name}: {source_rows:,}", flush=True)
            _store_source_reads(connection, batch)
            if source_rows == 0:
                raise HoldoutValidationError(f"Empty source split: {source['feature']}")
            source["sample_count"] = source_rows
            source["fingerprints"] = source.pop("last_fingerprints")
            input_fingerprints.append(dict(
                strand=source["strand"], previous_split=source["previous_split"],
                sample_count=source_rows, **source["fingerprints"],
            ))
            input_rows += source_rows
        connection.commit()
        conflicts = connection.execute("SELECT read_id,group_id,previous_split FROM source_reads WHERE conflict=1 LIMIT 20").fetchall()
        if conflicts:
            raise HoldoutValidationError(
                "A read occurs in multiple chosen groups or multiple original splits, "
                "or has conflicting binary labels/template pairs/references; holdout rejected. "
                f"Examples (read_id, first_group, first_split): {conflicts}"
            )
        assignment, group_counts, attempts = plan_assignment(groups, strands, int(seed), max_attempts)
        connection.execute("CREATE TABLE group_assignment(group_id TEXT PRIMARY KEY, split TEXT)")
        connection.executemany("INSERT INTO group_assignment VALUES (?,?)", sorted(assignment.items()))
        stats = {strand: {split: empty_stats() for split in SPLITS} for strand in strands}
        writers, handles = {}, []
        try:
            for strand in strands:
                for split in SPLITS:
                    feature_name = f"xna_pc_6mer_features_{strand}_balanced_{split}.csv"
                    feature_handle = (temporary / feature_name).open("w", newline="", encoding="utf-8")
                    metadata_handle = (temporary / (Path(feature_name).stem + "_meta.tsv")).open("w", newline="", encoding="utf-8")
                    handles.extend((feature_handle, metadata_handle))
                    feature_writer = csv.writer(feature_handle, lineterminator="\n")
                    metadata_writer = csv.DictWriter(metadata_handle, fieldnames=meta_columns, delimiter="\t", lineterminator="\n")
                    feature_writer.writerow(FEATURE_COLUMNS)
                    metadata_writer.writeheader()
                    writers[strand, split] = feature_writer, metadata_writer
            output_rows, membership_batch = 0, []
            for source in sources:
                for line, row, meta, info in paired_rows(source):
                    group = grouping_key(info, group_by, f"{source['feature']}:{line}")
                    split, strand = assignment[group], source["strand"]
                    counter = stats[strand][split]
                    metadata = dict(meta)
                    metadata.update(
                        holdout_group=group, holdout_group_by=group_by, holdout_split=split,
                        holdout_previous_split=source["previous_split"],
                        holdout_source_feature=str(source["feature"]), holdout_source_row=line,
                        holdout_row_index=counter["sample_count"],
                        pair_id=info["template_pair"], holdout_original_pair_id=meta.get("pair_id", ""),
                    )
                    if "split" in metadata:
                        metadata["split"] = split
                    feature_writer, meta_writer = writers[strand, split]
                    feature_writer.writerow(row)
                    meta_writer.writerow(metadata)
                    update_stats(counter, info)
                    membership_batch.append((info["read_id"], strand, split, 1))
                    output_rows += 1
                    if len(membership_batch) >= 5000:
                        _store_output_reads(connection, membership_batch)
                        membership_batch.clear()
                    if log_every > 0 and output_rows % log_every == 0:
                        print(f"[write] {output_rows:,}/{input_rows:,} samples", flush=True)
                if source.pop("last_fingerprints") != source["fingerprints"]:
                    raise HoldoutValidationError(f"Input bytes changed between passes: {source['feature']}")
            _store_output_reads(connection, membership_batch)
        finally:
            for handle in handles:
                handle.close()
        connection.commit()
        overlaps = connection.execute(
            "SELECT read_id,COUNT(DISTINCT split) FROM output_read_membership GROUP BY read_id HAVING COUNT(DISTINCT split)>1 LIMIT 20"
        ).fetchall()
        if overlaps or output_rows != input_rows:
            raise HoldoutValidationError(f"Output read overlap or sample loss: overlaps={overlaps}, input={input_rows}, output={output_rows}")
        for strand in strands:
            total = sum(stats[strand][split]["sample_count"] for split in SPLITS)
            for index, split in enumerate(SPLITS):
                item = stats[strand][split]
                if not all(item["class_counts"][str(label)] > 0 for label in (0, 1)):
                    raise HoldoutValidationError(f"Output {strand}/{split} is missing a binary class")
                item["unique_read_count"] = connection.execute(
                    "SELECT COUNT(*) FROM output_read_membership WHERE strand=? AND split=?", (strand, split),
                ).fetchone()[0]
                item["sample_fraction_within_strand"] = item["sample_count"] / total
                item["sample_fraction_minus_requested"] = item["sample_count"] / total - RATIOS[index]
                item["group_count"] = sum(
                    assignment[group] == split and sum(groups[group][strand].values()) > 0 for group in groups
                )
                item["groups"] = sorted(group for group in groups if assignment[group] == split and sum(groups[group][strand].values()) > 0)
        with (temporary / "group_assignment.csv").open("w", newline="", encoding="utf-8") as stream:
            columns = ["group_id", "split", "sample_count"] + [f"{strand}_label_{label}" for strand in strands for label in (0, 1)]
            writer = csv.DictWriter(stream, fieldnames=columns, lineterminator="\n")
            writer.writeheader()
            for group in sorted(groups):
                record = dict(group_id=group, split=assignment[group], sample_count=sum(sum(counter.values()) for counter in groups[group].values()))
                record.update({f"{strand}_label_{label}": groups[group][strand][label] for strand in strands for label in (0, 1)})
                writer.writerow(record)
        summary = dict(
            status="PASS", group_by=group_by, seed=int(seed), strands=list(strands),
            requested_split_ratios=dict(zip(SPLITS, RATIOS)), group_count=len(groups),
            actual_group_counts=group_counts,
            actual_group_fractions={split: count / len(groups) for split, count in group_counts.items()},
            group_fraction_minus_requested={split: group_counts[split] / len(groups) - RATIOS[i] for i, split in enumerate(SPLITS)},
            seeded_assignment_attempts=attempts,
            assignment_policy="first fixed-seed random group assignment with both classes in every requested strand/split; group counts approximate 7:2:1",
            input_sample_count=input_rows, output_sample_count=output_rows, samples_discarded=0,
            input_fingerprints=input_fingerprints, split_summary=stats,
            reference_to_template_pair=reference_pairs,
            input_read_cross_split_or_group_conflicts=0, output_read_cross_split_overlaps=0,
            grouped_identity_cross_split_overlaps=0,
            evaluation_scope="paired-template-held-out from reused samples" if group_by == "template" else "known-run-held-out from reused samples",
            independent_new_experiment=False,
            scientific_limitations=[
                "All output samples come from previously supplied balanced files; these are not independently newly collected measurements.",
                "Whole-group splitting preserves every sample; per-class balance and 7:2:1 sample fractions are not guaranteed.",
                "Template identity is explicit metadata or paired PC/DNA/XNA numbering; known paired references must agree with explicit pair fields. Run identity is never inferred from filenames or source paths.",
                "Groups and class coverage are chosen using metadata only; no model metric is examined.",
                "Numeric signal/scalar validity is checked subsequently by the training input audit; this script checks identity, pairing and grouping.",
            ],
        )
        write_json(temporary / "summary.json", summary)
        write_json(temporary / "holdout_config.json", dict(
            data_dir=str(data_dir), output_dir=str(output_dir), group_by=group_by,
            strands=list(strands), seed=int(seed), max_attempts=max_attempts,
        ))
        connection.close()
        temporary.rename(output_dir)
        print(f"[PASS] {group_by} holdout: {len(groups)} groups, {input_rows:,} samples; {output_dir}", flush=True)
        return summary
    except BaseException:
        connection.close()
        shutil.rmtree(temporary, ignore_errors=True)
        raise


def _store_source_reads(connection, batch):
    if batch:
        connection.executemany(
            "INSERT INTO source_reads VALUES (?,?,?,?,?,?,?,?) ON CONFLICT(read_id) DO UPDATE SET "
            "windows=windows+1, conflict=conflict OR group_id!=excluded.group_id "
            "OR previous_split!=excluded.previous_split OR binary_label!=excluded.binary_label "
            "OR template_pair!=excluded.template_pair OR chrom!=excluded.chrom",
            batch,
        )


def _store_output_reads(connection, batch):
    if batch:
        connection.executemany(
            "INSERT INTO output_read_membership VALUES (?,?,?,?) ON CONFLICT(read_id,strand,split) DO UPDATE SET windows=windows+1",
            batch,
        )


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data_dir", required=True)
    parser.add_argument("--output_dir", required=True)
    parser.add_argument("--group_by", choices=("template", "run"), default="template")
    parser.add_argument("--strands", nargs="+", choices=("pos", "neg"), default=["pos", "neg"])
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--max_attempts", type=int, default=10000)
    parser.add_argument("--log_every", type=int, default=100000)
    args = parser.parse_args()
    try:
        summary = prepare_holdout(**vars(args))
    except (ValueError, OSError, sqlite3.Error) as exc:
        print(f"[REJECTED] {exc}", file=sys.stderr, flush=True)
        raise SystemExit(2)
    print(json.dumps({
        "group_counts": summary["actual_group_counts"],
        "sample_counts": {strand: {split: stats["sample_count"] for split, stats in values.items()} for strand, values in summary["split_summary"].items()},
        "independent_new_experiment": False,
    }, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
