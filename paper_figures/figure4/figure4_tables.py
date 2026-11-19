"""Window-count source tables for unfiltered external Figure 4 panels."""

import re

import numpy as np
import pandas as pd

from figure4_data import CALL_COLUMNS, COHORTS


COUNTS = ["numerator_n", "support_n"]


def _summarize(per_seed, keys, seeds):
    """Average percentages across models; retain the single-seed window count."""
    if per_seed.duplicated(keys + ["seed"]).any():
        raise ValueError(f"Duplicate seed observation for {keys}.")
    if set(per_seed["seed"]) != seeds:
        raise ValueError("Plotting table has missing or unexpected seeds.")
    per_seed["rate_pct"] = 100 * per_seed["numerator_n"] / per_seed["support_n"]
    groups = per_seed.groupby(keys, sort=False)
    if not groups.size().eq(len(seeds)).all() or not groups["support_n"].nunique().eq(1).all():
        raise ValueError(f"Models did not evaluate the same support in each {keys} group.")
    summary = groups.agg(support_n=("support_n", "first"),
                         n_seeds=("seed", "size"),
                         mean_pct=("rate_pct", "mean"),
                         sd_pct=("rate_pct", "std")).reset_index()
    return summary


def _reference_strand_contexts(baseline, target, seeds):
    """Compute recall from the original counts that still identify the strand."""
    keys = ["cohort", "reference", "template", "file_strand", "motif", "true_class"]
    rows = baseline.loc[baseline.role.eq("target")].copy()
    if rows[keys].isna().any().any() or not rows.file_strand.isin(["pos", "neg"]).all():
        raise ValueError("Missing target identity or unexpected file_strand in the baseline counts.")
    rows["support_n"] = rows["count"]
    rows["numerator_n"] = rows["count"].where(rows.predicted_class.eq(rows.true_class), 0)
    per_seed = rows.groupby(keys + ["seed"], as_index=False)[COUNTS].sum()

    # Reconcile split windows against the independently verified argmax results.
    common_keys = ["cohort", "template", "motif", "true_class", "seed"]
    expected = target.groupby(common_keys)[COUNTS].sum().sort_index()
    actual = per_seed.groupby(common_keys)[COUNTS].sum().sort_index()
    if not actual.equals(expected):
        raise ValueError("Panel C split counts disagree with the external argmax counts.")
    summary = _summarize(per_seed, keys, seeds)

    # Natural sorting also handles reference names without zero-padded numbers.
    references = sorted(summary.reference.unique(), key=lambda name: tuple(
        int(part) if part.isdigit() else part.casefold() for part in re.split(r"(\d+)", name)))
    summary["_reference_order"] = summary.reference.map({name: i for i, name in enumerate(references)})
    summary["_strand_order"] = summary.file_strand.map({"pos": 0, "neg": 1})
    summary["_cohort_order"] = summary.cohort.map({name: i for i, name in enumerate(COHORTS)})
    summary = summary.sort_values(
        ["_cohort_order", "_reference_order", "template", "_strand_order", "motif", "true_class"],
        ignore_index=True,
    ).drop(columns=["_cohort_order", "_reference_order", "_strand_order"])
    summary["plot_order"] = summary.groupby("cohort", sort=False).cumcount() + 1
    per_seed = summary[keys + ["plot_order"]].merge(
        per_seed.sort_values("seed"), on=keys, validate="one_to_many", sort=False)
    return summary, per_seed


def build_tables(data):
    """One summary and one per-seed count table for each of the four panels.

    Percentages are 0–100. A support is the number of original windows for
    one model. D repeats that same denominator for every wrong destination.
    """
    if data["threshold"] != 0:
        raise ValueError("These panels describe unfiltered predictions. Load threshold=0.")
    seeds = set(data["seeds"])
    all_counts = data["counts"]
    rows = all_counts.loc[all_counts.scope.eq("external") & all_counts.threshold.eq(0)]
    if rows.no_call_n.ne(0).any() or set(rows.probability_type) != {data["probability_type"]}:
        raise ValueError("Expected a single unfiltered prediction set.")
    target = rows.loc[rows.role.eq("target")].copy()
    control = rows.loc[rows.role.eq("control")].copy()
    if set(target.true_class) != {"X", "Y"} or not control.true_class.isin(list("ATCG")).all():
        raise ValueError("Target and control roles are inconsistent with true classes.")

    target["numerator_n"] = np.where(target.true_class.eq("X"),
                                     target.called_X_n, target.called_Y_n)
    a_keys = ["cohort", "true_class"]
    a_cohorts = target.groupby(a_keys + ["seed"], as_index=False)[COUNTS].sum()
    a_pooled = a_cohorts.groupby(["true_class", "seed"], as_index=False)[COUNTS].sum()
    a_pooled["cohort"] = "XNA1-20"
    a_seed = pd.concat([a_cohorts, a_pooled], ignore_index=True)
    a = _summarize(a_seed, a_keys, seeds)

    # Source control templates use the XNA pair's identifier (e.g. XNA13).
    # The role, established before relabeling, identifies the canonical sample.
    control["template_number"] = control.template.str.extract(r"^XNA(\d+)$", expand=False).astype(int)
    control["template"] = "DNA" + control.template_number.astype(str)
    control["cohort"] = control.cohort.str.replace("XNA", "DNA", regex=False)
    control["numerator_n"] = control[["called_A_n", "called_T_n", "called_C_n", "called_G_n"]].sum(axis=1)
    b_keys = ["cohort", "template_number", "template"]
    b_seed = control.groupby(b_keys + ["seed"], as_index=False)[COUNTS].sum()
    b = _summarize(b_seed, b_keys, seeds)
    b = b.sort_values("template_number", ignore_index=True)
    if (b.template_number.tolist() != list(range(1, 17))
            or b.cohort.tolist() != ["DNA1-12"] * 12 + ["DNA13-16"] * 4):
        raise ValueError("Expected one control point per template in the two DNA cohorts.")
    # The lookup explains which masked contexts and true central bases were pooled.
    contexts = control[["template", "motif", "true_class"]].drop_duplicates().sort_values(
        ["template", "motif", "true_class"])
    contexts["masked_6mer_and_truth"] = contexts.motif + "|" + contexts.true_class
    template_lookup = contexts.groupby("template")["masked_6mer_and_truth"].agg(", ".join)
    b["included_backgrounds"] = b.template.map(template_lookup)

    c, c_seed = _reference_strand_contexts(data["baseline"], target, seeds)

    # Each wrong-call bar uses the full support of its true class, even if a
    # destination was rarely called or absent for a particular seed.
    d_wide = target.groupby(a_keys + ["seed"], as_index=False)[CALL_COLUMNS + ["support_n"]].sum()
    d_seed = d_wide.melt(id_vars=a_keys + ["seed", "support_n"], value_vars=CALL_COLUMNS,
                         var_name="called_class", value_name="numerator_n")
    d_seed["called_class"] = d_seed.called_class.str.slice(7, -2)
    d_seed = d_seed.loc[d_seed.true_class.ne(d_seed.called_class)].copy()
    d_keys = a_keys + ["called_class"]
    d = _summarize(d_seed, d_keys, seeds)

    return {"A": a, "B": b, "C": c, "D": d,
            "A_by_seed": a_seed, "B_by_seed": b_seed,
            "C_by_seed": c_seed, "D_by_seed": d_seed}
