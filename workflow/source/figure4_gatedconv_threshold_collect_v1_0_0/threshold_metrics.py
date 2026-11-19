#!/usr/bin/env python3
"""Derive rejection-aware six-class metrics from small confidence-bin counts.

This module never reads a model, features, or individual prediction scores.
All rates use explicitly named denominators; rejected calls remain in the
original population. Validation-only candidates are decision aids, not a
threshold chosen from Figure4 external-test results.
"""
from __future__ import annotations

from collections import Counter, defaultdict
import csv
import gzip
import json
import math
import os
from pathlib import Path
import statistics


CLASSES = ("A", "T", "C", "G", "X", "Y")
SEEDS = (42, 2024, 2025)
FINE_ID = ("scope", "seed", "probability_type", "dataset_id", "cohort", "template", "role", "motif", "true_label")
CONTEXT_ID = ("scope", "seed", "probability_type", "cohort", "role", "motif", "true_label")
CLASS_ID = ("scope", "seed", "probability_type", "cohort", "role", "true_label")
COHORT_ID = ("scope", "seed", "probability_type", "cohort")
CALL_FIELDS = tuple("called_" + name + "_n" for name in CLASSES) + ("no_call_n",)
CONTEXT_COUNT_FIELDS = ("support_n", "accepted_n", "rejected_n", "correct_accepted_n", "wrong_accepted_n",
                        "natural_as_X_n", "natural_as_Y_n", "canonical_wrong_n", "xy_swap_n")
CONTEXT_RATE_FIELDS = ("coverage", "effective_correct_rate", "accepted_accuracy", "natural_as_X_rate",
                      "natural_as_Y_rate", "natural_as_XY_rate", "control_false_X_rate", "control_false_Y_rate",
                      "control_false_XY_rate", "identity_recall", "canonical_wrong_rate", "xy_swap_rate")
COHORT_COUNT_FIELDS = ("support_n", "accepted_n", "rejected_n", "correct_accepted_n", "wrong_accepted_n",
                       "control_n", "control_accepted_n", "control_correct_accepted_n", "control_false_X_n",
                       "control_false_Y_n", "other_natural_n", "target_X_n", "target_Y_n", "target_X_accepted_n",
                       "target_Y_accepted_n", "target_X_correct_n", "target_Y_correct_n", "target_X_as_Y_n",
                       "target_Y_as_X_n", "target_X_as_canonical_n", "target_Y_as_canonical_n",
                       "called_X_n", "called_Y_n", "correct_called_X_n", "correct_called_Y_n")
COHORT_RATE_FIELDS = ("coverage", "effective_correct_rate", "accepted_accuracy", "control_coverage",
                      "control_effective_correct_rate", "control_false_X_rate", "control_false_Y_rate",
                      "control_false_XY_rate", "target_X_coverage", "target_Y_coverage", "target_X_identity_recall",
                      "target_Y_identity_recall", "target_X_as_Y_rate", "target_Y_as_X_rate",
                      "target_X_as_canonical_rate", "target_Y_as_canonical_rate", "called_X_precision", "called_Y_precision")


def require(condition, message):
    if not condition:
        raise ValueError(message)


def ratio(numerator, denominator):
    return numerator / denominator if denominator else None


def validate_thresholds(thresholds):
    values = tuple(thresholds)
    require(values and values[0] == 0, "Threshold grid must start at zero (the no-rejection argmax baseline)")
    require(all(isinstance(x, (int, float)) and not isinstance(x, bool) and math.isfinite(x) and 0 <= x <= 1 for x in values),
            "Thresholds must be finite values in [0, 1]")
    require(all(a < b for a, b in zip(values, values[1:])), "Thresholds must be unique and strictly increasing")
    return values


def _atomic_csv(path, fields, rows):
    path = Path(path)
    temporary = path.with_name(path.name + ".tmp-" + str(os.getpid()))
    try:
        opener = gzip.open if path.suffix == ".gz" else open
        with opener(temporary, "wt", encoding="utf-8", newline="") as stream:
            writer = csv.DictWriter(stream, fieldnames=fields)
            writer.writeheader()
            for row in rows:
                writer.writerow(row)
        temporary.replace(path)
    finally:
        temporary.unlink(missing_ok=True)


def _atomic_json(path, value):
    path = Path(path)
    temporary = path.with_name(path.name + ".tmp-" + str(os.getpid()))
    try:
        with temporary.open("w", encoding="utf-8") as stream:
            json.dump(value, stream, ensure_ascii=False, indent=2, allow_nan=False)
            stream.write("\n")
        temporary.replace(path)
    finally:
        temporary.unlink(missing_ok=True)


def _input_groups(records, threshold_count):
    groups = {}
    score_identity = defaultdict(Counter)
    input_rows, input_window_score_rows = 0, 0
    for row in records:
        require(all(name in row for name in FINE_ID + ("predicted_label", "max_passed_index", "count")), "Confidence-bin record lacks required fields")
        require(row["scope"] in ("external", "validation"), "Unexpected record scope")
        require(type(row["seed"]) is int and row["seed"] in SEEDS, "Unexpected model seed")
        require(row["probability_type"] in ("raw", "calibrated"), "Unexpected probability type")
        truth, pred, index, count = (row[name] for name in ("true_label", "predicted_label", "max_passed_index", "count"))
        require(type(truth) is int and type(pred) is int and 0 <= truth < 6 and 0 <= pred < 6, "Invalid truth/prediction label")
        require(type(index) is int and 0 <= index < threshold_count, "Invalid highest-passed threshold bin")
        require(type(count) is int and count > 0, "Confidence-bin counts must be positive integers")
        if row["scope"] == "external":
            require(row["role"] in ("control", "target"), "External role must remain control or target")
            require(row["role"] != "control" or truth < 4, "Natural control contains an X/Y truth")
        for field in ("dataset_id", "cohort", "template", "role", "motif"):
            require(isinstance(row[field], str) and row[field], "Empty or nontext identity field: " + field)
        key = tuple(row[name] for name in FINE_ID)
        if key not in groups:
            groups[key] = Counter()
        groups[key][pred, index] += count
        identity = tuple(row[name] for name in FINE_ID if name != "probability_type") + (pred,)
        score_identity[identity][row["probability_type"]] += count
        input_rows += 1
        input_window_score_rows += count
    require(groups, "No confidence-bin records supplied")
    for identity, counts in score_identity.items():
        require(set(counts) == {"raw", "calibrated"} and counts["raw"] == counts["calibrated"],
                "Raw/calibrated original argmax counts differ: " + str(identity))
    return groups, {"input_bin_records": input_rows, "window_probability_rows": input_window_score_rows,
                    "original_windows_across_seeds_and_scopes": input_window_score_rows // 2,
                    "raw_calibrated_denominators_equal": True}


def _expanded_calls(histogram, thresholds):
    total = sum(histogram.values())
    accepted = [0] * 6
    for (pred, _index), count in histogram.items():
        accepted[pred] += count
    removals = [[] for _ in thresholds]
    for (pred, index), count in histogram.items():
        removals[index].append((pred, count))
    for index, threshold in enumerate(thresholds):
        if index:
            for pred, count in removals[index - 1]:
                accepted[pred] -= count
        called = tuple(accepted) + (total - sum(accepted),)
        require(all(x >= 0 for x in called) and sum(called) == total, "Confusion count conservation failed")
        yield threshold, called


def _is_control(scope, role, truth):
    return truth < 4 and (scope == "validation" or role == "control")


def _is_target(scope, role, truth):
    return truth >= 4 and (scope == "validation" or role == "target")


def _context_metrics(identity, threshold, called):
    row = dict(identity, threshold=threshold, true_class=CLASSES[identity["true_label"]])
    row.update(zip(CALL_FIELDS, called))
    truth = identity["true_label"]
    support, accepted = sum(called), sum(called[:6])
    correct = called[truth]
    natural = truth < 4
    is_control = _is_control(identity["scope"], identity["role"], truth)
    is_target = _is_target(identity["scope"], identity["role"], truth)
    as_x, as_y = (called[4], called[5]) if natural else (0, 0)
    canonical_wrong = sum(called[:4]) - (correct if natural else 0)
    swap = called[9 - truth] if truth in (4, 5) else 0
    row.update(support_n=support, accepted_n=accepted, rejected_n=called[6], correct_accepted_n=correct,
               wrong_accepted_n=accepted - correct, natural_as_X_n=as_x, natural_as_Y_n=as_y,
               canonical_wrong_n=canonical_wrong, xy_swap_n=swap,
               coverage=ratio(accepted, support), effective_correct_rate=ratio(correct, support),
               accepted_accuracy=ratio(correct, accepted), natural_as_X_rate=ratio(as_x, support) if natural else None,
               natural_as_Y_rate=ratio(as_y, support) if natural else None,
               natural_as_XY_rate=ratio(as_x + as_y, support) if natural else None,
               control_false_X_rate=ratio(as_x, support) if is_control else None,
               control_false_Y_rate=ratio(as_y, support) if is_control else None,
               control_false_XY_rate=ratio(as_x + as_y, support) if is_control else None,
               identity_recall=ratio(correct, support) if is_target else None,
               canonical_wrong_rate=ratio(canonical_wrong, support), xy_swap_rate=ratio(swap, support) if is_target else None)
    return row


def _cohort_add(values, identity, called):
    truth = identity["true_label"]
    support, accepted, correct = sum(called), sum(called[:6]), called[truth]
    values.update(support_n=support, accepted_n=accepted, rejected_n=called[6], correct_accepted_n=correct,
                  wrong_accepted_n=accepted - correct, called_X_n=called[4], called_Y_n=called[5],
                  correct_called_X_n=called[4] if truth == 4 else 0,
                  correct_called_Y_n=called[5] if truth == 5 else 0)
    if _is_control(identity["scope"], identity["role"], truth):
        values.update(control_n=support, control_accepted_n=accepted, control_correct_accepted_n=correct,
                      control_false_X_n=called[4], control_false_Y_n=called[5])
    elif truth < 4:
        values.update(other_natural_n=support)
    if _is_target(identity["scope"], identity["role"], truth):
        base, other = ("X", "Y") if truth == 4 else ("Y", "X")
        values.update({"target_" + base + "_n": support, "target_" + base + "_accepted_n": accepted,
                       "target_" + base + "_correct_n": correct, "target_" + base + "_as_" + other + "_n": called[9 - truth],
                       "target_" + base + "_as_canonical_n": sum(called[:4])})


def _cohort_metrics(identity, threshold, counts):
    row = dict(identity, threshold=threshold)
    row.update({name: counts[name] for name in COHORT_COUNT_FIELDS})
    row.update(coverage=ratio(row["accepted_n"], row["support_n"]),
               effective_correct_rate=ratio(row["correct_accepted_n"], row["support_n"]),
               accepted_accuracy=ratio(row["correct_accepted_n"], row["accepted_n"]),
               control_coverage=ratio(row["control_accepted_n"], row["control_n"]),
               control_effective_correct_rate=ratio(row["control_correct_accepted_n"], row["control_n"]),
               control_false_X_rate=ratio(row["control_false_X_n"], row["control_n"]),
               control_false_Y_rate=ratio(row["control_false_Y_n"], row["control_n"]),
               control_false_XY_rate=ratio(row["control_false_X_n"] + row["control_false_Y_n"], row["control_n"]),
               called_X_precision=ratio(row["correct_called_X_n"], row["called_X_n"]),
               called_Y_precision=ratio(row["correct_called_Y_n"], row["called_Y_n"]))
    for name, other in (("X", "Y"), ("Y", "X")):
        row["target_" + name + "_coverage"] = ratio(row["target_" + name + "_accepted_n"], row["target_" + name + "_n"])
        row["target_" + name + "_identity_recall"] = ratio(row["target_" + name + "_correct_n"], row["target_" + name + "_n"])
        row["target_" + name + "_as_" + other + "_rate"] = ratio(row["target_" + name + "_as_" + other + "_n"], row["target_" + name + "_n"])
        row["target_" + name + "_as_canonical_rate"] = ratio(row["target_" + name + "_as_canonical_n"], row["target_" + name + "_n"])
    return row


def _check_monotonic(previous, row, names, denominator):
    require(row[denominator] == previous[denominator], "Original denominator changed across thresholds")
    for name in names:
        require(row[name] <= previous[name], "Count increased with stricter rejection threshold: " + name)
    require(row["rejected_n"] >= previous["rejected_n"], "Rejection count decreased with stricter threshold")


def _seed_aggregate(rows, identity_fields, metrics, level):
    grouped = defaultdict(list)
    for row in rows:
        key = tuple(row[name] for name in identity_fields if name != "seed") + (row["threshold"],)
        grouped[key].append(row)
    output = []
    fields = tuple(name for name in identity_fields if name != "seed") + ("threshold",)
    for key in sorted(grouped):
        values = grouped[key]
        seeds = sorted({row["seed"] for row in values})
        require(len(seeds) == len(values), "Duplicate seed in a seed-aggregate group")
        for metric in metrics:
            valid = [row[metric] for row in values if row.get(metric) is not None]
            result = dict(zip(fields, key))
            result.update(level=level, metric=metric, n_seeds=len(seeds), n_defined=len(valid),
                          seeds=";".join(str(seed) for seed in seeds),
                          mean=statistics.mean(valid) if valid else None,
                          sd=statistics.stdev(valid) if len(valid) > 1 else None,
                          minimum=min(valid) if valid else None, maximum=max(valid) if valid else None)
            output.append(result)
    return output


def validation_candidates(cohort_rows, thresholds, targets=(0.01, 0.005, 0.001)):
    """Use validation rows only; external labels can never affect a candidate."""
    selected = [row for row in cohort_rows if row["scope"] == "validation" and row["probability_type"] == "calibrated"]
    base = {"scope": "validation_only_decision_aid", "probability_type": "calibrated", "expected_seeds": list(SEEDS),
            "rule": "Use one shared grid threshold. Among thresholds satisfying the target in every validation seed, choose the lowest; accepted correct-identity counts cannot increase at a stricter threshold.",
            "external_data_used_for_selection": False,
            "formal_threshold_frozen": False,
            "limitations": ["These are validation-only candidates, not a declaration that an external false-positive target is attained.",
                            "Temperature calibration and validation threshold analysis reuse the existing validation split; no new independent calibration split was created.",
                            "Figure4 external threshold scans are post-hoc sensitivity analyses. Do not select a threshold separately for external motifs or cohorts.",
                            "A zero false-call rate obtained by rejecting all target calls is not an operationally useful solution.",
                            "Rates are window-level. Seeds are training repeats, not independent biological replicates."], "candidates": []}
    if not selected:
        base.update(status="no_validation", note="No validation predictions were collected; external-only data cannot choose a formal operating threshold.")
        return base
    lookup = defaultdict(list)
    for row in selected:
        lookup[row["seed"], row["threshold"]].append(row)
    for seed in SEEDS:
        for threshold in thresholds:
            rows = lookup[seed, threshold]
            if len(rows) != 1 or any(rows[0][name] <= 0 for name in ("control_n", "target_X_n", "target_Y_n")):
                base.update(status="incomplete_validation", note="Each seed/threshold needs exactly one validation cohort with natural, X and Y support.")
                return base
    base["status"] = "candidates_available"
    for target in targets:
        feasible = []
        for threshold in thresholds:
            rows = [lookup[seed, threshold][0] for seed in SEEDS]
            if all(row["control_false_XY_rate"] <= target for row in rows):
                feasible.append((threshold, rows))
        item = {"natural_false_XY_target": target, "comparison": "<=", "selected_using": "validation_only"}
        if not feasible:
            item.update(status="target_unattainable_on_grid", threshold=None)
        else:
            threshold, rows = feasible[0]
            recalls = [row["target_" + name + "_identity_recall"] for row in rows for name in ("X", "Y")]
            item.update(status="candidate" if min(recalls) > 0 else "candidate_zero_target_recall",
                        threshold=threshold, mean_XY_identity_recall=statistics.mean(recalls),
                        minimum_seed_class_identity_recall=min(recalls),
                        minimum_seed_coverage=min(row["coverage"] for row in rows),
                        all_target_calls_rejected=all(row["target_X_accepted_n"] + row["target_Y_accepted_n"] == 0 for row in rows),
                        operational_suitability_not_established=True,
                        per_seed=[{name: row[name] for name in ("seed", "control_n", "control_false_XY_rate", "target_X_n", "target_Y_n", "target_X_identity_recall", "target_Y_identity_recall", "target_X_coverage", "target_Y_coverage", "coverage")} for row in rows])
        base["candidates"].append(item)
    return base


def metric_definitions():
    return {"class_names": list(CLASSES), "no_call_label": 6, "rates_are_fractions_not_percentages": True,
            "decision_rule": "Keep the original six-class argmax if its score is >= threshold; otherwise label it no_call. Never reassign a rejected X/Y call to a natural class.",
            "baseline": "Threshold 0 accepts every original window, reproducing argmax counts. Six-class argmax is not a default 0.5 threshold.",
            "support_n": "All original windows in the indicated group, unchanged by threshold.",
            "coverage": "accepted_n / support_n",
            "effective_correct_rate": "correct_accepted_n / support_n; includes all rejected windows in its denominator.",
            "identity_recall": "For a true target X or Y, accepted correct identity / all original windows of that true class.",
            "accepted_accuracy": "correct_accepted_n / accepted_n; NA (empty CSV field) if accepted_n=0.",
            "external_control_false_XY_rate": "Accepted X or Y calls among role=control true natural windows / all original role=control natural windows. Canonical-truth windows in role=target are not added to this control denominator.",
            "validation_control_false_XY_rate": "Accepted X or Y calls among every true A/T/C/G validation window / every original A/T/C/G validation window; the role string does not limit the validation denominator.",
            "control_effective_correct_rate": "Accepted correct canonical identity / all original natural-control windows. A rejected natural window is not counted as correctly classified.",
            "called_X_precision": "Accepted correct X calls / all accepted X calls in the evaluated cohort; depends on its class mixture.",
            "called_Y_precision": "Accepted correct Y calls / all accepted Y calls in the evaluated cohort; depends on its class mixture.",
            "natural_as_XY_rate": "Within a particular true natural class, accepted X/Y calls / original support, whether role is control or target. See control_false_XY_rate for the designated control denominator.",
            "canonical_wrong_n": "Accepted predictions A/T/C/G with the wrong identity; for a true X/Y, all accepted natural predictions.",
            "xy_swap_n": "Accepted X->Y or Y->X calls.",
            "missing_support": "Undefined rates are NA, not zero. XNA17-20 has no natural controls and therefore no control FPR estimate.",
            "seed_aggregate": "Arithmetic mean and sample SD (ddof=1) across training seeds with defined rates; n_defined is explicit. These are not biological-repeat errors or confidence intervals.",
            "validation_context_aggregation": "Validation may be summarized as motif=ALL for operating-threshold choice; external Figure4 backgrounds remain separate.",
            "statistical_unit": "Window. No confidence intervals or read-independent bootstrap are calculated."}


def summarize(records, thresholds, output_dir):
    """Write a lightweight, denominator-preserving threshold summary.

    Return relative output filenames and QA. Input counts are read exactly once;
    they can be supplied as any iterable of dictionaries.
    """
    thresholds = validate_thresholds(thresholds)
    output = Path(output_dir)
    output.mkdir(parents=True, exist_ok=True)
    groups, qa = _input_groups(records, len(thresholds))
    contexts = defaultdict(lambda: [0] * 7)
    classes = defaultdict(lambda: [0] * 7)
    cohorts = defaultdict(Counter)
    fine_row_count = 0

    def fine_rows():
        nonlocal fine_row_count
        for key in sorted(groups):
            identity = dict(zip(FINE_ID, key))
            previous = None
            for threshold, called in _expanded_calls(groups[key], thresholds):
                row = _context_metrics(identity, threshold, called)
                if previous is not None:
                    _check_monotonic(previous, row, ("accepted_n", "correct_accepted_n", "natural_as_X_n", "natural_as_Y_n", "xy_swap_n"), "support_n")
                else:
                    require(row["rejected_n"] == 0, "Threshold zero rejected an original window")
                previous = row
                for fields, destination in ((CONTEXT_ID, contexts), (CLASS_ID, classes)):
                    group_key = tuple(identity[name] for name in fields) + (threshold,)
                    target = destination[group_key]
                    for index, count in enumerate(called):
                        target[index] += count
                cohort_key = tuple(identity[name] for name in COHORT_ID) + (threshold,)
                _cohort_add(cohorts[cohort_key], identity, called)
                fine_row_count += 1
                yield {name: row[name] for name in FINE_ID + ("true_class", "threshold") + CALL_FIELDS + ("support_n",)}

    files = ["threshold_confusion_counts.csv.gz", "context_metrics_per_seed.csv.gz", "cohort_class_metrics_per_seed.csv",
             "cohort_metrics_per_seed.csv", "seed_aggregate.csv", "validation_threshold_candidates.json", "metric_definitions.json"]
    _atomic_csv(output / files[0], FINE_ID + ("true_class", "threshold") + CALL_FIELDS + ("support_n",), fine_rows())
    context_rows = [_context_metrics(dict(zip(CONTEXT_ID, key[:-1])), key[-1], called) for key, called in sorted(contexts.items())]
    class_rows = [_context_metrics(dict(zip(CLASS_ID, key[:-1])), key[-1], called) for key, called in sorted(classes.items())]
    cohort_rows = [_cohort_metrics(dict(zip(COHORT_ID, key[:-1])), key[-1], counts) for key, counts in sorted(cohorts.items())]
    for level_rows, identity_fields, names in ((context_rows, CONTEXT_ID, ("accepted_n", "correct_accepted_n", "natural_as_X_n", "natural_as_Y_n", "xy_swap_n")),
                                              (class_rows, CLASS_ID, ("accepted_n", "correct_accepted_n", "natural_as_X_n", "natural_as_Y_n", "xy_swap_n")),
                                              (cohort_rows, COHORT_ID, ("accepted_n", "correct_accepted_n", "control_false_X_n", "control_false_Y_n", "target_X_correct_n", "target_Y_correct_n"))):
        previous = {}
        for row in level_rows:
            key = tuple(row[name] for name in identity_fields)
            if key in previous:
                _check_monotonic(previous[key], row, names, "support_n")
            else:
                require(row["threshold"] == 0 and row["rejected_n"] == 0, "A metric group lacks its complete argmax baseline")
            previous[key] = row
    context_fields = ("true_class", "threshold") + CALL_FIELDS + CONTEXT_COUNT_FIELDS + CONTEXT_RATE_FIELDS
    _atomic_csv(output / files[1], CONTEXT_ID + context_fields, context_rows)
    _atomic_csv(output / files[2], CLASS_ID + context_fields, class_rows)
    _atomic_csv(output / files[3], COHORT_ID + ("threshold",) + COHORT_COUNT_FIELDS + COHORT_RATE_FIELDS, cohort_rows)
    aggregate_rows = _seed_aggregate(context_rows, CONTEXT_ID, CONTEXT_RATE_FIELDS, "context")
    aggregate_rows += _seed_aggregate(cohort_rows, COHORT_ID, COHORT_RATE_FIELDS, "cohort")
    aggregate_fields = ("level", "scope", "probability_type", "cohort", "role", "motif", "true_label", "threshold", "metric", "n_seeds", "n_defined", "seeds", "mean", "sd", "minimum", "maximum")
    _atomic_csv(output / files[4], aggregate_fields, aggregate_rows)
    candidates = validation_candidates(cohort_rows, thresholds)
    _atomic_json(output / files[5], candidates)
    _atomic_json(output / files[6], metric_definitions())
    baseline_totals = defaultdict(dict)
    for row in cohort_rows:
        if row["threshold"] == 0:
            baseline_totals[(row["scope"], row["seed"], row["cohort"])][row["probability_type"]] = row["support_n"]
    require(all(set(values) == {"raw", "calibrated"} and values["raw"] == values["calibrated"] for values in baseline_totals.values()), "Cohort raw/calibrated baselines differ")
    qa.update(status="passed", threshold_count=len(thresholds), fine_confusion_rows=fine_row_count,
              context_metric_rows=len(context_rows), cohort_class_metric_rows=len(class_rows), cohort_metric_rows=len(cohort_rows),
              seed_aggregate_rows=len(aggregate_rows), denominators_preserved=True, rejection_column_present=True,
              monotonic_accepted_correct_false_calls=True, argmax_baseline_complete=True,
              external_data_used_for_threshold_selection=False, candidate_status=candidates["status"],
              undefined_values="NA in CSV (empty field), null in JSON", rate_units="fraction [0,1]")
    return {"outputs": files, "qa": qa, "validation_candidates": candidates}
