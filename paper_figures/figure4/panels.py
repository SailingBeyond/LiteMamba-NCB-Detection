"""Individual Figure 4 panels and optional single-facet exports."""

import math

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.lines import Line2D
from matplotlib.patches import Patch


COHORTS = ("XNA1-12", "XNA13-16", "XNA17-20")
IDENTITY_COLORS = {"X": "#377DB8", "Y": "#D98484"}
COHORT_COLORS = ("#487294", "#CC8C55", "#749088")
DNA_COLOR = "#50847F"
INK = "#1B2935"
STYLE = {
    "font.family": "sans-serif",
    "font.sans-serif": ["Arial", "DejaVu Sans"],
    "font.size": 10.5,
    "axes.titlesize": 11.5,
    "axes.titleweight": "bold",
    "axes.edgecolor": "#283039",
    "axes.linewidth": 1.4,
    "axes.labelsize": 11,
    "axes.grid": False,
    "xtick.major.width": 1.15,
    "ytick.major.width": 1.15,
    "legend.frameon": False,
    "figure.facecolor": "white",
    "axes.facecolor": "white",
}


def _cohort_name(value):
    return value.replace("-", "–")


def _identity_handles():
    return [
        Line2D([], [], linestyle="", marker="o", color=color, label=identity)
        for identity, color in IDENTITY_COLORS.items()
    ]


def _identity_legend(axis):
    axis.legend(handles=_identity_handles(), loc="lower right", ncol=2,
                handlelength=0.9, handletextpad=0.28, columnspacing=0.9,
                prop={"size": 9, "weight": "bold"})


def _recall_panel(rows):
    fig, axis = plt.subplots(figsize=(5.15, 3.3))
    fig.subplots_adjust(left=0.16, right=0.97, top=0.92, bottom=0.18)
    indexed = rows.set_index(["cohort", "true_class"])
    for identity, x_offset in (("X", -0.14), ("Y", 0.14)):
        for pos, cohort in enumerate(COHORTS):
            row = indexed.loc[(cohort, identity)]
            axis.errorbar(pos + x_offset, row.mean_pct, yerr=row.sd_pct, fmt="o",
                          color=IDENTITY_COLORS[identity], ms=6.8, elinewidth=1.5,
                          capsize=3.4, capthick=1.25)
    cohort_rows = rows[rows.cohort.isin(COHORTS)]
    lower = min(83, math.floor((cohort_rows.mean_pct - cohort_rows.sd_pct).min()) - 1)
    upper = max(101, (cohort_rows.mean_pct + cohort_rows.sd_pct).max() + 0.7)
    axis.set(xticks=range(3), xticklabels=[_cohort_name(c) for c in COHORTS],
             xlim=(-0.49, 2.49), ylim=(lower, upper), ylabel="Recall (%)")
    axis.set_yticks(np.arange(math.ceil(lower / 5) * 5, 101, 5))
    _identity_legend(axis)
    return fig


def _specificity_panel(rows):
    fig, axis = plt.subplots(figsize=(5.15, 3.3))
    fig.subplots_adjust(left=0.22, right=0.97, top=0.92, bottom=0.22)
    lower = min(60, math.floor((rows.mean_pct.min() - 0.5) / 10) * 10)
    axis.set(xlim=(lower, 101), xticks=np.arange(lower, 101, 10), ylim=(-0.5, 1.5),
             yticks=[1, 0], yticklabels=["DNA1–12", "DNA13–16"], xlabel="Canonical call rate (%)")
    axis.axhline(0.5, color="#D8DEDE", linewidth=0.75, zorder=0)

    # Use physical distances so nearly identical rates remain individually visible.
    # Only the vertical offset changes. The horizontal coordinate is the exact rate.
    axis_box = axis.get_window_extent().transformed(fig.dpi_scale_trans.inverted())
    pt_per_x = axis_box.width * 72 / (101 - lower)
    pt_per_y = axis_box.height * 72 / 2
    min_distance = 7.0
    offsets = [0.0]
    for level in range(1, 7):
        distance = level * min_distance / pt_per_y
        offsets.extend((distance, -distance))
    for cohort, center in (("DNA1-12", 1), ("DNA13-16", 0)):
        placed = []
        group = rows[rows.cohort.eq(cohort)].sort_values(["mean_pct", "template_number"])
        for row in group.itertuples():
            rate = float(row.mean_pct)
            for dy in offsets:
                if all(math.hypot((rate - x) * pt_per_x, (dy - y) * pt_per_y)
                       >= min_distance - 1e-8 for x, y in placed):
                    break
            placed.append((rate, dy))
            dot = axis.scatter(rate, center + dy, s=38, facecolor=DNA_COLOR,
                               edgecolor="white", linewidth=0.5, zorder=3)
            dot.set_gid(f"template_{row.template}")
    return fig


def _contexts_panel(rows, facet):
    groups = COHORTS if facet is None else (facet,)
    if any(c not in COHORTS for c in groups):
        raise ValueError(f"Unknown XNA cohort {facet!r}")
    full_width = facet is None
    fig, axes = plt.subplots(
        1, len(groups), sharey=True, squeeze=False,
        figsize=(10.0 if full_width else 5.15, 3.65),
        gridspec_kw={"width_ratios": [rows.cohort.eq(c).sum() for c in groups]},
    )
    left = 0.075 if full_width else 0.15
    fig.subplots_adjust(left=left, right=0.985 if full_width else 0.97,
                        top=0.845, bottom=0.22, wspace=0.12)
    strand_markers = {"pos": "o", "neg": "^"}
    handles = [Patch(facecolor=color, label=f"True {identity}")
               for identity, color in IDENTITY_COLORS.items()]
    handles.extend(
        Line2D([], [], color=INK, linestyle="", marker=marker, markersize=5.3, label=label)
        for marker, label in zip(strand_markers.values(), ("+ strand", "− strand"))
    )
    fig.legend(handles=handles, ncol=4, loc="upper center", bbox_to_anchor=(0.5, 0.97),
               borderaxespad=0.0, handlelength=0.95, handletextpad=0.42,
               columnspacing=1.15, fontsize=9)
    lower = min(65, math.floor(rows.mean_pct.min() / 5) * 5 - 1)
    for index, (axis, cohort) in enumerate(zip(axes[0], groups)):
        group = rows[rows.cohort.eq(cohort)].sort_values("plot_order")
        # Reference spans follow the ordered table; each x is a separate
        # reference × file strand × masked 6mer × true identity.
        spans = group.groupby("reference", sort=False).agg(
            start=("plot_order", "min"), end=("plot_order", "max"), template=("template", "first"),
        )
        for boundary in spans.end.iloc[:-1] + 0.5:
            axis.axvline(boundary, color="#E1E5E8", lw=0.65, zorder=0)
        colors = group.true_class.map(IDENTITY_COLORS)
        axis.vlines(group.plot_order, lower, group.mean_pct,
                    colors=colors, alpha=0.27, lw=1.0, zorder=1)
        for strand, marker in strand_markers.items():
            selected = group.file_strand.eq(strand)
            axis.scatter(group.loc[selected, "plot_order"], group.loc[selected, "mean_pct"],
                         c=colors[selected], marker=marker, s=26, linewidths=0, zorder=2)
        tick_labels = spans.template if cohort == "XNA17-20" else spans.index
        axis.set(xlim=(0.35, len(group) + 0.65), ylim=(lower, 101),
                 xticks=(spans.start + spans.end) / 2, xticklabels=tick_labels,
                 yticks=np.arange(math.ceil(lower / 10) * 10, 101, 10))
        axis.tick_params(axis="x", labelsize=9, length=2.4, pad=3)
        for tick in axis.get_xticklabels():
            tick.set(rotation=58, ha="right", rotation_mode="anchor")
        if index == 0:
            axis.set_ylabel("Recall (%)")
    return fig


def _errors_panel(rows, facet):
    labels = ("X", "Y") if facet is None else (facet,)
    if any(label not in IDENTITY_COLORS for label in labels):
        raise ValueError(f"Unknown true class {facet!r}")
    full_width = facet is None
    fig, axes = plt.subplots(1, len(labels), sharey=True, squeeze=False,
                             figsize=(9.1 if full_width else 5.15, 3.45))
    left = 0.092 if full_width else 0.16
    fig.subplots_adjust(left=left, right=0.98, top=0.76, bottom=0.16, wspace=0.13)
    handles = [Patch(facecolor=color, label=_cohort_name(c)) for c, color in zip(COHORTS, COHORT_COLORS)]
    fig.legend(handles=handles, loc="upper center", bbox_to_anchor=(0.5, 0.97), ncol=3,
               borderaxespad=0.0, prop={"size": 9, "weight": "bold"}, handlelength=1.05,
               handletextpad=0.36, columnspacing=1.0)
    ymax = max(8, math.ceil(rows.mean_pct.max() / 2) * 2)
    for j, (axis, identity) in enumerate(zip(axes[0], labels)):
        destinations = ["A", "T", "C", "G", "Y" if identity == "X" else "X"]
        x = np.arange(len(destinations))
        for k, (cohort, color) in enumerate(zip(COHORTS, COHORT_COLORS)):
            group = rows[rows.true_class.eq(identity) & rows.cohort.eq(cohort)].set_index("called_class")
            values = group.loc[destinations, "mean_pct"].to_numpy()
            axis.bar(x + 0.24 * (k - 1), values, width=0.225, color=color)
        axis.set(xticks=x, xticklabels=destinations, ylim=(0, ymax + 0.15),
                 yticks=np.arange(0, ymax + 1, 2), xlim=(-0.55, 4.55))
        axis.set_title(f"True {identity}", pad=6)
        if j == 0:
            axis.set_ylabel("Miscall rate (%)")
    return fig


def draw_panel(letter, tables, facet=None):
    """Return an independent figure. Tables contain percent-scale means and SDs."""
    with plt.rc_context(STYLE):
        if letter == "C":
            return _contexts_panel(tables["C"], facet)
        if letter == "D":
            return _errors_panel(tables["D"], facet)
        if facet is not None:
            raise ValueError(f"Panel {letter} does not have separately selectable facets.")
        if letter == "A":
            return _recall_panel(tables["A"])
        if letter == "B":
            return _specificity_panel(tables["B"])
        raise ValueError(f"Unknown Figure 4 panel {letter!r}")
