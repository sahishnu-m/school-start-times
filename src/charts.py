"""Builds the four charts the study reports, and saves them to /outputs.

Design rules followed throughout, and the reason for each:

  One measure per axis. No chart here puts two different scales on left and
  right, because a reader cannot tell which line belongs to which axis and the
  apparent crossing points are an artifact of where the scales were set.

  Two colours at most, taken from a palette checked for colour vision
  deficiency. Blue and orange were validated as distinguishable under protanopia
  and tritanopia simulation, so a reader with red green colour blindness can
  still tell the two groups apart.

  Colour never carries meaning alone. Where there are two groups there is a
  legend and the groups are also separated by position, so the chart still works
  printed in black and white.

  Recessive chrome. Gridlines and axes are drawn lighter than the data, because
  the data is the thing being read.

  The charts are written as PNG files, which cannot respond to a dark theme.
  They are drawn on a light surface so they stay legible wherever they are
  embedded.
"""

from __future__ import annotations

import matplotlib

# Agg is a non-interactive backend. Setting it before pyplot is imported means
# the pipeline can render charts on a machine with no display attached, which
# is what happens when this runs on a server.
matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from .analyze import OUTCOME_LABELS, variable_label
from .clean import minutes_to_clock
from .config import outputs_dir, site_config

# Validated palette. Slot 1 and slot 2 of the categorical theme.
SERIES_1 = "#2a78d6"   # blue
SERIES_2 = "#eb6834"   # orange
SURFACE = "#fcfcfb"
INK_PRIMARY = "#0b0b0b"
INK_SECONDARY = "#52514e"
INK_MUTED = "#898781"
GRIDLINE = "#e1e0d9"
BASELINE = "#c3c2b7"

FONT_FAMILY = ["Segoe UI", "DejaVu Sans", "sans-serif"]


def _style_axes(ax, title: str, subtitle: str = "", xlabel: str = "", ylabel: str = "") -> None:
    """Apply the shared chart chrome.

    Pulled into one function so every chart in the study looks like it belongs
    to the same study, and so a change to the house style happens once.
    """
    ax.set_facecolor(SURFACE)
    ax.figure.set_facecolor(SURFACE)

    # Only the left and bottom spines are kept, drawn in the recessive baseline
    # colour. A box around a chart adds four lines and no information.
    for side in ("top", "right"):
        ax.spines[side].set_visible(False)
    for side in ("left", "bottom"):
        ax.spines[side].set_color(BASELINE)
        ax.spines[side].set_linewidth(1.0)

    ax.grid(axis="y", color=GRIDLINE, linewidth=0.8, zorder=0)
    ax.set_axisbelow(True)
    ax.tick_params(colors=INK_MUTED, labelsize=9, length=0)

    ax.set_xlabel(xlabel, color=INK_SECONDARY, fontsize=10, labelpad=8)
    ax.set_ylabel(ylabel, color=INK_SECONDARY, fontsize=10, labelpad=8)

    if subtitle:
        ax.set_title(subtitle, color=INK_SECONDARY, fontsize=10, loc="left", pad=8)
        ax.figure.suptitle(
            title, color=INK_PRIMARY, fontsize=13, fontweight="bold", x=0.02, ha="left"
        )
    else:
        ax.set_title(title, color=INK_PRIMARY, fontsize=13, fontweight="bold", loc="left", pad=10)


def _save(figure, filename: str, site: str | None = None) -> str:
    """Write a figure to this site's output folder and close it."""
    out_dir = outputs_dir(site)
    out_dir.mkdir(parents=True, exist_ok=True)
    path = out_dir / filename
    figure.savefig(path, dpi=160, bbox_inches="tight", facecolor=SURFACE)
    plt.close(figure)
    print(f"  wrote {filename}")
    return str(path)


def start_time_distribution(frame: pd.DataFrame, site: str | None = None) -> str:
    """Histogram of school start times.

    A histogram is the right form here because the question is about the shape
    of one variable: how much do Nevada high school start times actually vary?
    If nearly every school starts within fifteen minutes of every other school,
    there is not enough variation to detect an effect, and this chart is where
    that would show up. It is drawn first for exactly that reason.
    """
    times = frame["start_minutes"].dropna()
    figure, ax = plt.subplots(figsize=(8, 4.5))

    if len(times) == 0:
        ax.text(0.5, 0.5, "No start times collected yet", ha="center", va="center",
                color=INK_SECONDARY, fontsize=12, transform=ax.transAxes)
        _style_axes(ax, "Start time distribution")
        return _save(figure, "01_start_time_distribution.png", site)

    # Fifteen minute bins. Bell schedules are set on quarter hours, so finer
    # bins would produce empty gaps that look like structure but are not.
    low = int(np.floor(times.min() / 15) * 15)
    high = int(np.ceil(times.max() / 15) * 15) + 15
    bins = np.arange(low, high, 15)

    ax.hist(times, bins=bins, color=SERIES_1, edgecolor=SURFACE, linewidth=2, zorder=3)

    tick_positions = np.arange(low, high, 30)
    ax.set_xticks(tick_positions)
    ax.set_xticklabels([minutes_to_clock(t) for t in tick_positions], rotation=45, ha="right")

    median = times.median()
    ax.axvline(median, color=INK_SECONDARY, linewidth=2, linestyle="--", zorder=4)

    group_column = site_config(site)["group_column"]
    groups = sorted(frame.loc[frame["start_minutes"].notna(), group_column].dropna().unique())
    label = site_config(site)["group_label"].lower()
    coverage = ", ".join(groups) if len(groups) <= 3 else f"{len(groups)} {label}s"

    _style_axes(
        ax,
        "High school start times",
        f"{len(times)} schools with a known start time, in {coverage}. "
        f"Range {minutes_to_clock(times.min())} to {minutes_to_clock(times.max())}. "
        f"Dashed line is the median, {minutes_to_clock(median)}.",
        xlabel="Start time",
        ylabel="Number of schools",
    )
    return _save(figure, "01_start_time_distribution.png", site)


def unadjusted_scatter(frame: pd.DataFrame, outcome: str, correlation: dict, site: str | None = None) -> str:
    """Scatter of start time against an outcome, with a straight line fit.

    This chart is labelled unadjusted in its own subtitle rather than only in
    the surrounding text. A chart gets separated from its caption the moment
    someone screenshots it, and this one is the single most misreadable image
    in the study, so the warning travels with the picture.
    """
    subset = frame.dropna(subset=["start_hours", outcome])
    figure, ax = plt.subplots(figsize=(8, 5))

    if len(subset) < 3:
        ax.text(0.5, 0.5, "Not enough schools to plot", ha="center", va="center",
                color=INK_SECONDARY, fontsize=12, transform=ax.transAxes)
        _style_axes(ax, "Unadjusted relationship")
        return _save(figure, f"02_unadjusted_scatter_{outcome}.png", site)

    # A white ring around each point so overlapping schools stay countable.
    ax.scatter(
        subset["start_hours"], subset[outcome],
        s=64, color=SERIES_1, alpha=0.75,
        edgecolor=SURFACE, linewidth=1.5, zorder=3,
    )

    slope, intercept = np.polyfit(subset["start_hours"], subset[outcome], 1)
    line_x = np.linspace(subset["start_hours"].min(), subset["start_hours"].max(), 100)
    ax.plot(line_x, slope * line_x + intercept, color=SERIES_2, linewidth=2, zorder=4)

    tick_positions = np.arange(
        np.floor(subset["start_hours"].min() * 2) / 2,
        np.ceil(subset["start_hours"].max() * 2) / 2 + 0.25,
        0.5,
    )
    ax.set_xticks(tick_positions)
    ax.set_xticklabels([minutes_to_clock(t * 60) for t in tick_positions],
                       rotation=45, ha="right")

    subtitle = (
        f"Unadjusted. No controls. r = {correlation['r']:.2f} "
        f"(95% CI {correlation['ci_low']:.2f} to {correlation['ci_high']:.2f}), "
        f"n = {correlation['n']} schools."
    )
    _style_axes(
        ax,
        f"Start time and {OUTCOME_LABELS.get(outcome, outcome)}",
        subtitle,
        xlabel="School start time",
        ylabel=OUTCOME_LABELS.get(outcome, outcome),
    )
    return _save(figure, f"02_unadjusted_scatter_{outcome}.png", site)


def coefficient_plot(ladder: pd.DataFrame, outcome: str, site: str | None = None) -> str:
    """The start time coefficient and its interval across model specifications.

    A dot and interval plot is the right form because the quantity of interest
    is an estimate with uncertainty, and the comparison is across a small
    ordered set of models. Bars would be wrong here: a bar implies a magnitude
    measured from zero, and what matters is where the interval sits relative to
    zero, not the area of a rectangle.

    The vertical line at zero is the reference. An interval that crosses it is
    an estimate consistent with no relationship.
    """
    rows = ladder[ladder["outcome"] == outcome].sort_values("model_number")
    estimable = rows.dropna(subset=["coefficient"])

    figure, ax = plt.subplots(figsize=(8, 0.62 * max(len(rows), 3) + 1.8))

    if estimable.empty:
        ax.text(0.5, 0.5, "No model could be estimated", ha="center", va="center",
                color=INK_SECONDARY, fontsize=12, transform=ax.transAxes)
        _style_axes(ax, "Start time coefficient across models")
        return _save(figure, f"03_coefficient_plot_{outcome}.png", site)

    positions = np.arange(len(rows))[::-1]   # first model at the top

    ax.axvline(0, color=INK_SECONDARY, linewidth=1.5, linestyle="--", zorder=2)

    for position, (_, row) in zip(positions, rows.iterrows()):
        if pd.isna(row["coefficient"]):
            reason = str(row.get("note") or "not estimable")
            # Keep the on-chart reason short. The full sentence is in
            # outputs/model_ladder.csv and on the dashboard.
            if "not identified" in reason:
                reason = "not identified, only one district in the sample"
            ax.text(0, position, f"  {reason}", va="center", ha="left",
                    color=INK_MUTED, fontsize=8.5, style="italic")
            continue
        # Model 1 is the unadjusted estimate and is drawn in the second colour
        # so it is visually separated from the adjusted models beneath it.
        colour = SERIES_2 if row["model_number"] == 1 else SERIES_1
        ax.plot([row["ci_low"], row["ci_high"]], [position, position],
                color=colour, linewidth=2, solid_capstyle="round", zorder=3)
        ax.plot([row["coefficient"]], [position], marker="o", markersize=9,
                color=colour, markeredgecolor=SURFACE, markeredgewidth=1.5, zorder=4)
        ax.annotate(
            f"{row['coefficient']:+.2f}  (n = {int(row['n'])})",
            xy=(row["ci_high"], position), xytext=(8, 0), textcoords="offset points",
            va="center", color=INK_SECONDARY, fontsize=9,
        )

    ax.set_yticks(positions)
    ax.set_yticklabels(
        [f"{r['model_number']}. {r['specification']}" for _, r in rows.iterrows()],
        fontsize=9, color=INK_PRIMARY,
    )
    ax.grid(axis="y", visible=False)
    ax.grid(axis="x", color=GRIDLINE, linewidth=0.8)

    # Leave room on the right for the direct labels.
    left, right = ax.get_xlim()
    ax.set_xlim(left, right + (right - left) * 0.28)

    _style_axes(
        ax,
        "How the start time estimate moves as controls are added",
        f"Outcome: {OUTCOME_LABELS.get(outcome, outcome)}. "
        "Dot is the estimated change per hour later. Bar is the 95% confidence interval.",
        xlabel=f"Change in {OUTCOME_LABELS.get(outcome, outcome)} per hour later start",
    )
    return _save(figure, f"03_coefficient_plot_{outcome}.png", site)


def stratified_chart(strata: pd.DataFrame, outcome: str, site: str | None = None) -> str:
    """Early and late starting schools compared inside each poverty band.

    Grouped bars with the group counts written on them. The counts are on the
    chart rather than in a caption because several bands contain a handful of
    schools, and a bar drawn from four schools should not look as solid as a
    bar drawn from forty.
    """
    rows = strata[strata["outcome"] == outcome]
    figure, ax = plt.subplots(figsize=(8, 5))

    if rows.empty or rows[["mean_early", "mean_late"]].isna().all().all():
        ax.text(0.5, 0.5, "Not enough schools for a stratified comparison",
                ha="center", va="center", color=INK_SECONDARY, fontsize=12,
                transform=ax.transAxes)
        _style_axes(ax, "Stratified comparison")
        return _save(figure, f"04_stratified_{outcome}.png", site)

    cutoff_label = minutes_to_clock(site_config(site)["early_late_cutoff_minutes"])
    labels = rows["poverty_band"].tolist()
    positions = np.arange(len(labels))
    width = 0.38

    # A small gap between the paired bars, which the palette guidance asks for
    # so two adjacent fills never touch.
    early_bars = ax.bar(positions - width / 2 - 0.01, rows["mean_early"], width,
                        label=f"Starts before {cutoff_label}", color=SERIES_1,
                        edgecolor=SURFACE, linewidth=2, zorder=3)
    late_bars = ax.bar(positions + width / 2 + 0.01, rows["mean_late"], width,
                       label=f"Starts at or after {cutoff_label}", color=SERIES_2,
                       edgecolor=SURFACE, linewidth=2, zorder=3)

    for bars, count_column in ((early_bars, "n_early"), (late_bars, "n_late")):
        for bar, count in zip(bars, rows[count_column]):
            if pd.isna(bar.get_height()):
                continue
            ax.annotate(
                f"n = {int(count)}",
                xy=(bar.get_x() + bar.get_width() / 2, bar.get_height()),
                xytext=(0, 4), textcoords="offset points",
                ha="center", color=INK_SECONDARY, fontsize=8,
            )

    ax.set_xticks(positions)
    ax.set_xticklabels(labels, fontsize=9)

    legend = ax.legend(frameon=False, fontsize=9, loc="upper right")
    for text in legend.get_texts():
        text.set_color(INK_SECONDARY)

    _style_axes(
        ax,
        "Early and late starters within similar poverty bands",
        f"Schools grouped by {variable_label('poverty_pct', site).lower()}.",
        xlabel=f"Poverty band (lower {variable_label('poverty_pct', site).lower()} on the left)",
        ylabel=OUTCOME_LABELS.get(outcome, outcome),
    )
    return _save(figure, f"04_stratified_{outcome}.png", site)


def outcome_summary(frame: pd.DataFrame, ladder: pd.DataFrame, site: str | None = None) -> str:
    """Every outcome on one chart, in standard deviations of that outcome.

    The five outcomes are measured on different scales, so their raw
    coefficients cannot be compared to each other. A one point change in an ACT
    composite and a one point change in a graduation percentage are not the
    same size of thing.

    Dividing each estimate by the standard deviation of its own outcome fixes
    that. The result reads as "a school starting one hour later scores this many
    standard deviations higher or lower", which is comparable across outcomes
    and is how effect sizes are usually reported in education research.

    The fully adjusted model is used, which is the last rung of the ladder that
    could be estimated.
    """
    settings = site_config(site)
    rows = []

    for outcome in [settings["outcomes"]["primary"]] + settings["outcomes"]["secondary"]:
        available = ladder[(ladder["outcome"] == outcome) & ladder["coefficient"].notna()]
        if available.empty or outcome not in frame.columns:
            continue
        best = available.sort_values("model_number").iloc[-1]
        spread = pd.to_numeric(frame[outcome], errors="coerce").std()
        if not spread or pd.isna(spread) or spread == 0:
            continue
        rows.append(
            {
                "label": OUTCOME_LABELS.get(outcome, outcome),
                "estimate": best["coefficient"] / spread,
                "ci_low": best["ci_low"] / spread,
                "ci_high": best["ci_high"] / spread,
                "n": int(best["n"]),
                "specification": best["specification"],
            }
        )

    figure, ax = plt.subplots(figsize=(8, 0.72 * max(len(rows), 3) + 2))
    if not rows:
        ax.text(0.5, 0.5, "No estimable models", ha="center", va="center",
                color=INK_SECONDARY, fontsize=12, transform=ax.transAxes)
        _style_axes(ax, "Fully adjusted estimates")
        return _save(figure, "05_outcome_summary.png", site)

    summary = pd.DataFrame(rows)
    positions = np.arange(len(summary))[::-1]
    ax.axvline(0, color=INK_SECONDARY, linewidth=1.5, linestyle="--", zorder=2)

    for position, (_, row) in zip(positions, summary.iterrows()):
        # An interval that crosses zero is drawn in the second colour, so the
        # reader can see at a glance which estimates settle a direction and
        # which do not.
        crosses_zero = row["ci_low"] <= 0 <= row["ci_high"]
        colour = SERIES_2 if crosses_zero else SERIES_1
        ax.plot([row["ci_low"], row["ci_high"]], [position, position],
                color=colour, linewidth=2, solid_capstyle="round", zorder=3)
        ax.plot([row["estimate"]], [position], marker="o", markersize=9, color=colour,
                markeredgecolor=SURFACE, markeredgewidth=1.5, zorder=4)
        ax.annotate(f"{row['estimate']:+.2f} SD  (n = {row['n']})",
                    xy=(row["ci_high"], position), xytext=(8, 0),
                    textcoords="offset points", va="center",
                    color=INK_SECONDARY, fontsize=9)

    ax.set_yticks(positions)
    ax.set_yticklabels(summary["label"], fontsize=9, color=INK_PRIMARY)
    ax.grid(axis="y", visible=False)
    ax.grid(axis="x", color=GRIDLINE, linewidth=0.8)
    left, right = ax.get_xlim()
    ax.set_xlim(left, right + (right - left) * 0.30)

    _style_axes(
        ax,
        "Every outcome, fully adjusted, on one scale",
        "Estimated change per hour later start, in standard deviations of each "
        "outcome. Orange intervals cross zero.",
        xlabel="Standard deviations per hour later start",
    )
    return _save(figure, "05_outcome_summary.png", site)


def run(frame: pd.DataFrame, results: dict, site: str | None = None) -> list[str]:
    """Draw every chart for the primary outcome."""
    settings = site_config(site)
    site = settings["name"]
    print("Drawing charts")
    plt.rcParams["font.family"] = FONT_FAMILY

    primary = settings["outcomes"]["primary"]
    correlation_rows = results["correlations"]
    correlation = correlation_rows[correlation_rows["outcome"] == primary].iloc[0].to_dict()

    paths = [
        start_time_distribution(frame, site),
        unadjusted_scatter(frame, primary, correlation, site),
        stratified_chart(results["strata"], primary, site),
        outcome_summary(frame, results["ladders"], site),
    ]

    # A coefficient plot for every outcome, not only the primary one. The
    # secondary outcomes are where this study's two robust results are, and
    # burying them in a CSV would hide the most interesting part.
    for outcome in [primary] + settings["outcomes"]["secondary"]:
        paths.append(coefficient_plot(results["ladders"], outcome, site))

    return paths
