"""Load and cross-check the original Figure4 threshold and baseline packages."""

from __future__ import annotations

import gzip
import hashlib
import io
import json
import tarfile
from pathlib import Path, PurePosixPath

import numpy as np
import pandas as pd


CLASSES = ("A", "T", "C", "G", "X", "Y")
COHORTS = ("XNA1-12", "XNA13-16", "XNA17-20")
PROBABILITY_TYPES = ("calibrated", "raw")
CALL_COLUMNS = [f"called_{name}_n" for name in CLASSES]
COUNT_COLUMNS = CALL_COLUMNS + ["no_call_n", "support_n"]
CONTEXT_KEYS = ["scope", "seed", "probability_type", "cohort", "role",
                "motif", "true_label", "true_class", "threshold"]
FINE_KEYS = CONTEXT_KEYS + ["dataset_id", "template"]
COHORT_KEYS = ["scope", "seed", "probability_type", "cohort", "threshold"]
MANIFEST_NAME = "RESULT_PACKAGE_MANIFEST.json"


def _read_package(path, requested):
    """Read manifest-bound files from a directory or tar.gz without extracting."""
    path = Path(path).expanduser().resolve()
    if not path.exists():
        raise FileNotFoundError(f"Missing statistics input: {path}")

    def read_manifest(read):
        manifest = json.loads(read(MANIFEST_NAME))
        files = manifest["files"]
        missing = set(requested).difference(files)
        if missing:
            raise ValueError(f"{path}: manifest does not list {sorted(missing)}")
        payload = {}
        for name, expected in files.items():
            member = PurePosixPath(name)
            if member.is_absolute() or ".." in member.parts:
                raise ValueError(f"{path}: invalid member: {name}")
            value = read(name)
            if (len(value) != expected["size_bytes"]
                    or hashlib.sha256(value).hexdigest() != expected["sha256"]):
                raise ValueError(f"{path}: size or SHA256 mismatch for {name}")
            if name in requested:
                payload[name] = value
        return payload

    if path.is_dir():
        return read_manifest(lambda name: (path / name).read_bytes())

    with tarfile.open(path, "r:gz") as archive:
        regular_files = {}
        for member in archive.getmembers():
            if member.isfile():
                name = member.name.removeprefix("./")
                if name in regular_files:
                    raise ValueError(f"{path}: duplicate member: {name}")
                regular_files[name] = member

        def read(name):
            if name not in regular_files:
                raise ValueError(f"{path}: missing archive member: {name}")
            with archive.extractfile(regular_files[name]) as reader:
                return reader.read()

        return read_manifest(read)


def _read_csv(payload, name):
    data = payload[name]
    if name.endswith(".gz"):
        data = gzip.decompress(data)
    return pd.read_csv(io.BytesIO(data))


def _validate_counts(counts, manifest):
    if counts[FINE_KEYS + COUNT_COLUMNS].isna().any().any():
        raise ValueError("Missing key or count in the threshold statistics.")
    if counts.duplicated(FINE_KEYS).any():
        raise ValueError("Duplicate threshold count row.")
    for name in COUNT_COLUMNS:
        if not pd.api.types.is_integer_dtype(counts[name]) or (counts[name] < 0).any():
            raise ValueError(f"{name} must contain nonnegative integer counts.")
    if not counts[CALL_COLUMNS + ["no_call_n"]].sum(axis=1).eq(counts.support_n).all():
        raise ValueError("Called counts plus no-call do not equal original support.")
    if (counts.support_n <= 0).any():
        raise ValueError("Each threshold count row needs positive support.")
    if not counts.true_label.isin(range(len(CLASSES))).all():
        raise ValueError("Unexpected true-class index.")
    if not counts.true_label.map(dict(enumerate(CLASSES))).eq(counts.true_class).all():
        raise ValueError("Class index and class name disagree.")
    if set(counts.probability_type) != set(PROBABILITY_TYPES):
        raise ValueError("Both raw and calibrated counts are required.")
    if not set(counts.scope).issubset({"validation", "external"}) or "external" not in set(counts.scope):
        raise ValueError("Unexpected scope in the threshold statistics.")
    if not counts.loc[counts.scope.eq("external"), "role"].isin({"target", "control"}).all():
        raise ValueError("Unexpected external window role.")
    if not counts.loc[counts.scope.eq("external"), "cohort"].isin(COHORTS).all():
        raise ValueError("Unexpected external cohort.")
    if not counts.loc[counts.scope.eq("external"), "motif"].str.fullmatch(r"[ACGT]{3}N[ACGT]{2}").all():
        raise ValueError("An external motif is inconsistent with 4th-position masking.")
    if not counts.loc[counts.scope.eq("external") & counts.role.eq("control"), "true_label"].lt(4).all():
        raise ValueError("A natural control has an X/Y truth label.")
    if set(counts.seed) != set(manifest["seeds"]):
        raise ValueError("Count records and manifest use different seeds.")
    grid = sorted(counts.threshold.unique())
    if grid != sorted(manifest["thresholds"]) or not grid or grid[0] != 0.0 or grid[-1] > 1.0:
        raise ValueError("Count records and manifest use different threshold grids.")
    recorded_rows = manifest.get("metric_qa", {}).get("fine_confusion_rows")
    if recorded_rows is not None and recorded_rows != len(counts):
        raise ValueError("Threshold row count does not agree with the manifest.")

    keys = [key for key in FINE_KEYS if key != "threshold"]
    ordered = counts.sort_values("threshold", kind="stable")
    groups = ordered.groupby(keys, sort=False)
    if not groups.size().eq(len(grid)).all():
        raise ValueError("Missing threshold within a dataset/motif/truth group.")
    if not groups.support_n.nunique().eq(1).all():
        raise ValueError("Support changes across thresholds.")
    if groups[CALL_COLUMNS].diff().gt(0).any().any() or groups.no_call_n.diff().lt(0).any():
        raise ValueError("Calls increase or no-call decreases as threshold rises.")
    at_zero = counts[counts.threshold.eq(0)]
    if at_zero.no_call_n.ne(0).any():
        raise ValueError("The argmax baseline contains rejected windows.")
    base_keys = [key for key in FINE_KEYS if key not in {"threshold", "probability_type"}]
    raw = at_zero[at_zero.probability_type.eq("raw")].set_index(base_keys)[COUNT_COLUMNS].sort_index()
    calibrated = at_zero[at_zero.probability_type.eq("calibrated")].set_index(base_keys)[COUNT_COLUMNS].sort_index()
    if not raw.equals(calibrated):
        raise ValueError("Raw and calibrated argmax baseline counts differ.")
    keys = [key for key in FINE_KEYS if key != "probability_type"]
    if not counts.groupby(keys).support_n.nunique().eq(1).all():
        raise ValueError("Support differs between raw and calibrated probabilities.")
    return grid


def _validate_baseline(counts, baseline, manifest, baseline_manifest):
    for key in ("model", "seeds", "plan_id", "plan_sha256"):
        if manifest[key] != baseline_manifest[key]:
            raise ValueError(f"Baseline and threshold sources disagree on {key}.")
    keys = ["seed", "dataset_id", "cohort", "template", "role",
            "motif", "true_label", "true_class"]
    if not pd.api.types.is_integer_dtype(baseline["count"]) or baseline["count"].lt(0).any():
        raise ValueError("Baseline prediction counts must be nonnegative integers.")
    if not baseline.predicted_label.map(dict(enumerate(CLASSES))).eq(baseline.predicted_class).all():
        raise ValueError("Baseline class index and class name disagree.")
    if set(baseline.model) != {manifest["model"]} or set(baseline.probability_type) != {"raw"}:
        raise ValueError("Baseline does not contain the expected original raw model predictions.")
    original = baseline.pivot_table(
        index=keys, columns="predicted_class", values="count", aggfunc="sum", fill_value=0,
    ).reindex(columns=CLASSES, fill_value=0).sort_index()
    original.columns = CALL_COLUMNS
    rows = counts[counts.scope.eq("external") & counts.probability_type.eq("raw")
                  & counts.threshold.eq(0)]
    recovered = rows.set_index(keys)[CALL_COLUMNS].sort_index()
    if not original.equals(recovered):
        raise ValueError("External argmax counts disagree with the original baseline.")
    n_scored = int(counts[counts.probability_type.eq("raw") & counts.threshold.eq(0)].support_n.sum())
    if n_scored != manifest["source_prediction_rows"]:
        raise ValueError("Original scored-window count does not agree with the manifest.")
    return int(len(baseline))


def _ratio(numerator, denominator):
    """Leave undefined percentages empty when the denominator is zero."""
    numerator = np.asarray(numerator, dtype=float)
    denominator = np.asarray(denominator, dtype=float)
    return np.divide(numerator, denominator, out=np.full_like(numerator, np.nan),
                     where=denominator > 0)


def _class_metrics(counts, keys):
    table = counts.groupby(keys, as_index=False)[COUNT_COLUMNS].sum()
    true = table.true_label.to_numpy()
    calls = table[CALL_COLUMNS].to_numpy()
    is_natural = true < 4
    is_control = is_natural & (table.role.eq("control") | table.scope.eq("validation"))
    is_target = ~is_natural & (table.role.eq("target") | table.scope.eq("validation"))
    table["accepted_n"] = calls.sum(axis=1)
    table["rejected_n"] = table.no_call_n
    table["correct_accepted_n"] = calls[np.arange(len(table)), true]
    table["wrong_accepted_n"] = table.accepted_n - table.correct_accepted_n
    table["natural_as_X_n"] = np.where(is_natural, table.called_X_n, 0)
    table["natural_as_Y_n"] = np.where(is_natural, table.called_Y_n, 0)
    table["canonical_wrong_n"] = calls[:, :4].sum(axis=1) - np.where(
        is_natural, table.correct_accepted_n, 0)
    table["xy_swap_n"] = np.where(true == 4, table.called_Y_n,
                                  np.where(true == 5, table.called_X_n, 0))
    table["coverage"] = _ratio(table.accepted_n, table.support_n)
    table["effective_correct_rate"] = _ratio(table.correct_accepted_n, table.support_n)
    table["accepted_accuracy"] = _ratio(table.correct_accepted_n, table.accepted_n)
    table["natural_as_X_rate"] = np.where(is_natural, _ratio(table.called_X_n, table.support_n), np.nan)
    table["natural_as_Y_rate"] = np.where(is_natural, _ratio(table.called_Y_n, table.support_n), np.nan)
    table["natural_as_XY_rate"] = table.natural_as_X_rate + table.natural_as_Y_rate
    table["control_false_X_rate"] = np.where(is_control, table.natural_as_X_rate, np.nan)
    table["control_false_Y_rate"] = np.where(is_control, table.natural_as_Y_rate, np.nan)
    table["control_false_XY_rate"] = table.control_false_X_rate + table.control_false_Y_rate
    table["identity_recall"] = np.where(is_target, table.effective_correct_rate, np.nan)
    table["canonical_wrong_rate"] = _ratio(table.canonical_wrong_n, table.support_n)
    table["xy_swap_rate"] = np.where(is_target, _ratio(table.xy_swap_n, table.support_n), np.nan)
    return table


def _cohort_metrics(classes):
    val = classes.scope.eq("validation")
    is_natural = classes.true_label.lt(4)
    control = is_natural & (val | classes.role.eq("control"))
    eligible_target = val | classes.role.eq("target")
    table = classes[COHORT_KEYS + ["support_n", "accepted_n", "rejected_n",
                                   "correct_accepted_n", "wrong_accepted_n",
                                   "called_X_n", "called_Y_n"]].copy()
    for destination, source in {
        "control_n": "support_n",
        "control_accepted_n": "accepted_n",
        "control_correct_accepted_n": "correct_accepted_n",
        "control_false_X_n": "called_X_n",
        "control_false_Y_n": "called_Y_n",
    }.items():
        table[destination] = np.where(control, classes[source], 0)
    table["other_natural_n"] = np.where(is_natural & ~control, classes.support_n, 0)
    for name, opposite in (("X", "Y"), ("Y", "X")):
        truth = classes.true_class.eq(name)
        target = truth & eligible_target
        prefix = f"target_{name}"
        for suffix, source in {
            "n": "support_n",
            "accepted_n": "accepted_n",
            "correct_n": "correct_accepted_n",
            f"as_{opposite}_n": f"called_{opposite}_n",
            "as_canonical_n": "canonical_wrong_n",
        }.items():
            table[f"{prefix}_{suffix}"] = np.where(target, classes[source], 0)
        table[f"correct_called_{name}_n"] = np.where(truth, classes[f"called_{name}_n"], 0)
    table = table.groupby(COHORT_KEYS, as_index=False).sum()
    ratios = {
        "coverage": ("accepted_n", "support_n"),
        "effective_correct_rate": ("correct_accepted_n", "support_n"),
        "accepted_accuracy": ("correct_accepted_n", "accepted_n"),
        "control_coverage": ("control_accepted_n", "control_n"),
        "control_effective_correct_rate": ("control_correct_accepted_n", "control_n"),
        "control_false_X_rate": ("control_false_X_n", "control_n"),
        "control_false_Y_rate": ("control_false_Y_n", "control_n"),
    }
    for name, opposite in (("X", "Y"), ("Y", "X")):
        prefix = f"target_{name}"
        ratios.update({
            f"{prefix}_coverage": (f"{prefix}_accepted_n", f"{prefix}_n"),
            f"{prefix}_identity_recall": (f"{prefix}_correct_n", f"{prefix}_n"),
            f"{prefix}_as_{opposite}_rate": (f"{prefix}_as_{opposite}_n", f"{prefix}_n"),
            f"{prefix}_as_canonical_rate": (f"{prefix}_as_canonical_n", f"{prefix}_n"),
            f"called_{name}_precision": (f"correct_called_{name}_n", f"called_{name}_n"),
        })
    for key, (num, den) in ratios.items():
        table[key] = _ratio(table[num], table[den])
    table["control_false_XY_rate"] = table.control_false_X_rate + table.control_false_Y_rate
    return table


def _check_table(computed, original, keys, name):
    if computed.duplicated(keys).any() or original.duplicated(keys).any():
        raise ValueError(f"{name}: duplicate metric keys.")
    a = computed.set_index(keys).sort_index()
    b = original.set_index(keys).sort_index()
    if not a.index.equals(b.index) or set(a.columns) != set(b.columns):
        raise ValueError(f"{name}: recomputed rows or columns differ.")
    for column in a.columns:
        numbers_match = np.allclose(a[column], b[column], rtol=0, atol=1e-12, equal_nan=True)
        if not numbers_match:
            raise ValueError(f"{name}: count-based metric mismatch in {column}.")


def _lowest_validation_threshold(cohort, target, seeds):
    validation = cohort[cohort.scope.eq("validation") & cohort.probability_type.eq("calibrated")]
    for threshold, group in validation.groupby("threshold", sort=True):
        if (len(group) == len(seeds) and set(group.seed) == set(seeds)
                and group.control_false_XY_rate.le(target + 1e-12).all()):
            return float(threshold)
    return None


def _resolve_selection(cohort, grid, candidates, seeds, probability_type,
                       threshold, validation_fpr_target):
    if probability_type not in PROBABILITY_TYPES:
        raise ValueError(f"Unknown probability type: {probability_type}")
    if probability_type == "raw" and threshold is None:
        raise ValueError("Raw probabilities require an explicit --threshold.")
    if not 0.0 <= validation_fpr_target <= 1.0:
        raise ValueError("--validation-fpr-target must be between 0 and 1.")
    for candidate in candidates.get("candidates", []):
        if candidate["status"] == "candidate" and _lowest_validation_threshold(
            cohort, candidate["natural_false_XY_target"], seeds
        ) != candidate["threshold"]:
            raise ValueError("A recorded threshold candidate is inconsistent with validation counts.")

    if threshold is None:
        selected = _lowest_validation_threshold(cohort, validation_fpr_target, seeds)
        if selected is None:
            raise ValueError("No tested threshold satisfies the target in every validation seed.")
        mode = "validation_only_candidate"
    else:
        selected = next((value for value in grid if abs(value - threshold) < 1e-12), None)
        if selected is None:
            raise ValueError(f"--threshold must be one of the actual tested values: {grid}")
        mode = "explicit_common_threshold"

    objective = validation_fpr_target if threshold is None else None
    validation_at_selected = cohort[
        cohort.scope.eq("validation") & cohort.probability_type.eq(probability_type)
        & cohort.threshold.eq(selected)
    ]
    selected_rows = validation_at_selected[
        ["seed", "control_n", "control_false_XY_rate", "target_X_identity_recall",
         "target_Y_identity_recall", "coverage"]
    ].to_dict("records")
    return {
        "mode": mode,
        "selected_using": "validation_only" if threshold is None else "explicit_user_argument",
        "threshold": selected,
        "probability_type": probability_type,
        "natural_false_XY_target": objective,
        "per_seed": selected_rows,
        "validation_fpr_target": objective,
        "external_data_used_for_selection": False if threshold is None else None,
        "formal_threshold_frozen": False,
        "description": (
            f"Lowest common tested threshold with validation false X/Y ≤ {objective:.2%} in every seed"
            if objective is not None else "Explicit common threshold on the tested grid"
        ),
    }


def load_data(input_path, baseline_input, probability_type="calibrated",
              threshold=None, validation_fpr_target=.005):
    """Read verified statistics and recompute plotting tables from integer counts."""
    threshold_files = [
        "threshold_confusion_counts.csv.gz",
        "context_metrics_per_seed.csv.gz",
        "cohort_class_metrics_per_seed.csv",
        "cohort_metrics_per_seed.csv",
        "collection_manifest.json",
        "metric_definitions.json",
        "validation_threshold_candidates.json",
    ]
    threshold_payload = _read_package(input_path, threshold_files)
    baseline_payload = _read_package(baseline_input, ["joint_counts.csv.gz", "collection_manifest.json"])
    manifest = json.loads(threshold_payload["collection_manifest.json"])
    baseline_manifest = json.loads(baseline_payload["collection_manifest.json"])
    if manifest["class_names"] != list(CLASSES) or manifest["model"] != "gatedconv_target":
        raise ValueError("Unexpected class order or model in threshold statistics.")
    counts = _read_csv(threshold_payload, "threshold_confusion_counts.csv.gz")
    baseline = _read_csv(baseline_payload, "joint_counts.csv.gz")
    grid = _validate_counts(counts, manifest)
    baseline_size = _validate_baseline(counts, baseline, manifest, baseline_manifest)

    context = _class_metrics(counts, CONTEXT_KEYS)
    class_keys = [key for key in CONTEXT_KEYS if key != "motif"]
    classes = _class_metrics(counts, class_keys)
    cohort = _cohort_metrics(classes)
    for computed, key, name in (
        (context, CONTEXT_KEYS, "context_metrics_per_seed.csv.gz"),
        (classes, class_keys, "cohort_class_metrics_per_seed.csv"),
        (cohort, COHORT_KEYS, "cohort_metrics_per_seed.csv"),
    ):
        _check_table(computed, _read_csv(threshold_payload, name), key, name)

    candidates = json.loads(threshold_payload["validation_threshold_candidates.json"])
    selection = _resolve_selection(cohort, grid, candidates, manifest["seeds"],
                                  probability_type, threshold, validation_fpr_target)
    context["no_call_rate"] = _ratio(context.no_call_n, context.support_n)
    for name in CLASSES:
        context[f"called_{name}_rate"] = _ratio(context[f"called_{name}_n"], context.support_n)
    cohort["control_accepted_accuracy"] = _ratio(cohort.control_correct_accepted_n,
                                                  cohort.control_accepted_n)

    def subset(table, probability):
        return table[table.probability_type.eq(probability)].reset_index(drop=True)

    return {
        "probability_type": probability_type,
        "threshold": selection["threshold"],
        "thresholds": grid,
        "seeds": list(manifest["seeds"]),
        "context": subset(context, probability_type),
        "cohort": subset(cohort, probability_type),
        "counts": subset(counts, probability_type),
        "cohort_class": subset(classes, probability_type),
        "raw_context": subset(context, "raw"),
        "raw_cohort": subset(cohort, "raw"),
        "calibrated_context": subset(context, "calibrated"),
        "calibrated_cohort": subset(cohort, "calibrated"),
        "baseline": baseline,
        "selection": selection,
        "validation_candidates": candidates,
        "manifest": manifest,
        "baseline_manifest": baseline_manifest,
        "metric_definitions": json.loads(threshold_payload["metric_definitions.json"]),
        "validation": {
            "status": "passed",
            "threshold_count_rows": len(counts),
            "context_metric_rows": len(context),
            "cohort_class_rows": len(classes),
            "cohort_metric_rows": len(cohort),
            "baseline_joint_count_rows": baseline_size,
            "baseline_argmax_counts_match": True,
            "shared_plan_id": manifest["plan_id"],
            "threshold_input": str(Path(input_path).expanduser().resolve()),
            "baseline_input": str(Path(baseline_input).expanduser().resolve()),
        },
    }
