"""The statistics: unadjusted correlation, a ladder of regressions, and a
stratified comparison.

The order of the three is the argument of the study.

The unadjusted correlation is reported first and labelled as unadjusted,
because it is the number a casual look at the data produces and it is almost
certainly wrong. Start times in Nevada are not assigned at random. A district
staggers its bell schedules so that one set of buses can run three routes each
morning, and which schools get the early route tracks where the buses come
from, which tracks neighbourhood income. So a simple correlation between start
time and test scores is partly a correlation between poverty and test scores
wearing a disguise.

The regression ladder is the response to that. Controls are added one block at
a time and the start time coefficient is reported at every rung. If the
coefficient collapses toward zero once poverty enters, that is the finding: the
raw relationship was confounding. Showing the collapse is more informative than
showing only the final model, because it tells a reader which variable did the
work.

The stratified comparison is a check on the regression. Regression assumes the
relationship is a straight line and that the controls enter additively. The
stratified comparison assumes neither. It splits schools into poverty bands and
compares early and late starters inside each band. If the two methods disagree,
that disagreement is worth more than either answer alone.

Every number reported here comes with a sample size and an interval. A point
estimate on its own hides how little a sample of this size can settle.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from .config import outputs_dir, site_config
from .stats_core import design_matrix, fit_ols, pearson_correlation, welch_test

# The regressions, the correlation and the two sample test are all computed in
# src/stats_core.py, which uses numpy alone. statsmodels and scipy were used
# here originally and were removed after the deployed dashboard failed to
# import them: both are compiled against a specific numpy, and the hosted
# environment installed a combination that did not match. The replacements are
# checked against both libraries in tests/test_stats_core.py.

# Human readable names used in tables and on the dashboard. One dictionary
# covers both sites, since the column names do not collide.
VARIABLE_LABELS = {
    "start_hours": "Start time (per hour later)",
    "poverty_pct": "Student poverty",
    "enrollment_1000": "Enrollment (per 1,000 students)",
    "el_pct": "English learners (percent)",
    "swd_pct": "Students with disabilities (percent)",
    "applicants_per_seat": "Applicants per seat",
    "C(locale_group)": "Locale (city, suburb, town, rural)",
    "C(district)": "District fixed effects",
    "C(borough)": "Borough fixed effects",
    "C(admissions_method)": "Admissions method",
}

OUTCOME_LABELS = {
    # Nevada
    "act_composite": "ACT composite score",
    "ela_proficient_pct": "ELA proficiency (percent)",
    "math_proficient_pct": "Math proficiency (percent)",
    "chronic_absent_pct": "Chronic absenteeism (percent)",
    # New York City
    "advanced_regents_pct": "Advanced Regents diploma rate (percent)",
    "attendance_rate": "Attendance rate (percent)",
    "college_career_rate": "College and career readiness (percent)",
    "dropout_pct": "Dropout rate (percent)",
    # Both
    "grad_rate": "Four year graduation rate (percent)",
}


def variable_label(name: str, site: str | None = None) -> str:
    """A readable label, using the site's own name for the poverty measure."""
    if name == "poverty_pct":
        return site_config(site).get("poverty_label", "Student poverty")
    return VARIABLE_LABELS.get(name, name)


def analysis_sample(frame: pd.DataFrame, outcome: str, controls: list[str]) -> pd.DataFrame:
    """Drop rows that cannot enter a given model.

    A regression silently uses only complete rows. That is fine, but it means
    two models with different controls can be fitted on different schools,
    which would make their coefficients not comparable. This function is used
    to state the sample explicitly so every reported n is the real one.
    """
    needed = ["start_hours", outcome]
    for control in controls:
        # "C(district)" names the column "district" inside a formula wrapper.
        column = control.replace("C(", "").replace(")", "")
        needed.append(column)
    available = [column for column in dict.fromkeys(needed) if column in frame.columns]
    return frame.dropna(subset=available)


def unadjusted_correlation(frame: pd.DataFrame, outcome: str) -> dict:
    """Pearson correlation between start time and an outcome, with an interval.

    The confidence interval is built with the Fisher z transformation. A
    correlation is bounded at minus one and one, so its sampling distribution
    is skewed near the ends. Fisher z maps it to a scale where the distribution
    is close to normal, the interval is built there, and the ends are mapped
    back. Doing it the naive way would produce intervals that run past one.
    """
    subset = frame.dropna(subset=["start_hours", outcome])
    result = pearson_correlation(subset["start_hours"], subset[outcome])
    result["outcome"] = outcome
    result["outcome_label"] = OUTCOME_LABELS.get(outcome, outcome)
    return result


def fit_one_model(
    frame: pd.DataFrame, outcome: str, controls: list[str], site: str | None = None
) -> dict | None:
    """Fit one regression of an outcome on start time plus a set of controls."""
    subset = analysis_sample(frame, outcome, controls)

    # A model needs more observations than it has things to estimate. District
    # fixed effects can add twenty terms, which on a small sample leaves no
    # room. Rather than let statsmodels return meaningless numbers, the model
    # is skipped and the caller reports it as not estimable.
    approximate_terms = len(controls) + 2
    if "C(district)" in controls:
        approximate_terms += subset["district"].nunique()
    if "C(locale_group)" in controls:
        approximate_terms += subset["locale_group"].nunique()
    if len(subset) < approximate_terms + 5:
        return None

    # A categorical control with only one level in the data adds nothing. When
    # every school in the sample is in one district, district fixed effects
    # cannot be estimated, and the model is identical to the one without them.
    # Reporting it as a separate rung would suggest the district was accounted
    # for when in fact there was nothing to account for.
    for control in controls:
        if control.startswith("C("):
            column = control[2:-1]
            if column in subset.columns and subset[column].nunique() < 2:
                return {"unidentified": column}

    # Standard errors. The default is heteroskedasticity robust (HC3), which
    # handles the spread of an outcome widening with school size.
    #
    # Where a site declares a cluster column, cluster robust errors are used
    # instead. In New York City several schools share one building, and schools
    # in the same building share a neighbourhood, a facility, and often the
    # same pool of applicants. Treating them as independent observations would
    # make the standard errors too small and the intervals too narrow.
    cluster_column = site_config(site).get("cluster_column")
    groups = None
    if cluster_column and cluster_column in subset.columns:
        column = subset[cluster_column]
        if column.notna().all() and column.nunique() > 1:
            groups = column.to_numpy()

    try:
        X, names = design_matrix(subset, ["start_hours"] + controls)
        model = fit_ols(subset[outcome].to_numpy(dtype=float), X, names, clusters=groups)
    except Exception:
        return None

    return model.for_term("start_hours")


def model_ladder(
    frame: pd.DataFrame,
    outcome: str,
    blocks: list[dict] | None = None,
    site: str | None = None,
) -> pd.DataFrame:
    """Fit the sequence of models defined in config.yaml.

    The returned table is the centrepiece of the results: one row per model
    specification, showing what happens to the start time coefficient as
    controls are added.
    """
    blocks = blocks or site_config(site)["model_blocks"]
    rows = []

    for index, block in enumerate(blocks, start=1):
        controls = block["controls"]
        result = fit_one_model(frame, outcome, controls, site)
        unidentified = result.pop("unidentified", None) if result else None

        row = {
            "model_number": index,
            "specification": block["name"],
            "controls": ", ".join(variable_label(c, site) for c in controls) or "none",
            "outcome": outcome,
            "outcome_label": OUTCOME_LABELS.get(outcome, outcome),
        }
        if result is None or unidentified:
            if unidentified:
                distinct = frame[unidentified].dropna().unique()
                note = (
                    f"not identified, every school with a start time is in one "
                    f"{unidentified}"
                    + (f" ({distinct[0]})" if len(distinct) == 1 else "")
                )
            else:
                note = "not estimable, too few schools for this many terms"
            row.update(
                {
                    "coefficient": np.nan, "std_error": np.nan, "ci_low": np.nan,
                    "ci_high": np.nan, "p_value": np.nan, "n": np.nan,
                    "r_squared": np.nan, "adj_r_squared": np.nan,
                    "note": note,
                }
            )
        else:
            row.update(result)
            row["note"] = ""
        rows.append(row)

    return pd.DataFrame(rows)


def stratified_comparison(frame: pd.DataFrame, outcome: str, site: str | None = None) -> pd.DataFrame:
    """Compare early and late starting schools inside each poverty band.

    Within a band, schools serve broadly similar student populations, so a
    difference between early and late starters inside a band is harder to
    explain away as a poverty difference. The interval comes from a Welch two
    sample t test, which does not assume the two groups have equal variance.
    Equal variance is not a safe assumption when one group has eight schools
    and the other has thirty.
    """
    subset = frame.dropna(subset=["start_group", "poverty_band", outcome])
    rows = []

    for band in site_config(site)["stratification"]["poverty_band_labels"]:
        in_band = subset[subset["poverty_band"] == band]
        early = in_band[in_band["start_group"] == "Early"][outcome]
        late = in_band[in_band["start_group"] == "Late"][outcome]

        row = {
            "poverty_band": band,
            "outcome": outcome,
            "outcome_label": OUTCOME_LABELS.get(outcome, outcome),
            "n_early": len(early),
            "n_late": len(late),
            "mean_early": early.mean() if len(early) else np.nan,
            "mean_late": late.mean() if len(late) else np.nan,
        }

        # Two schools per side is the minimum for a variance to exist at all.
        # Below that, report the group means and say plainly that no comparison
        # can be made, rather than printing a difference with no interval.
        if len(early) >= 2 and len(late) >= 2:
            test = welch_test(late.to_numpy(), early.to_numpy())
            row.update(
                {
                    "difference_late_minus_early": test["difference"],
                    "ci_low": test["ci_low"],
                    "ci_high": test["ci_high"],
                    "p_value": test["p_value"],
                    "note": "",
                }
            )
        else:
            row.update(
                {
                    "difference_late_minus_early": np.nan, "ci_low": np.nan,
                    "ci_high": np.nan, "p_value": np.nan,
                    "note": "too few schools in one group to compare",
                }
            )
        rows.append(row)

    return pd.DataFrame(rows)


def descriptive_statistics(frame: pd.DataFrame) -> pd.DataFrame:
    """Summarise every analysis variable, including how much is missing.

    The missing count is not filler. A variable that is present for 40 of 90
    schools supports a much weaker claim than one present for all 90, and a
    reader cannot tell which is which from a mean alone.
    """
    # Every variable either site might have. Columns that are not in this
    # site's data are skipped, so one list serves both.
    columns = [
        "start_minutes", "start_hours",
        "act_composite", "grad_rate", "ela_proficient_pct", "math_proficient_pct",
        "chronic_absent_pct", "advanced_regents_pct", "attendance_rate",
        "college_career_rate", "dropout_pct",
        "poverty_pct", "frl_pct_ccd", "el_pct", "swd_pct", "enrollment",
        "applicants_per_seat",
    ]
    rows = []
    for column in columns:
        if column not in frame.columns:
            continue
        series = pd.to_numeric(frame[column], errors="coerce")
        rows.append(
            {
                "variable": column,
                "label": OUTCOME_LABELS.get(column, VARIABLE_LABELS.get(column, column)),
                "n_present": int(series.notna().sum()),
                "n_missing": int(series.isna().sum()),
                "mean": series.mean(),
                "std_dev": series.std(),
                "minimum": series.min(),
                "median": series.median(),
                "maximum": series.max(),
            }
        )
    return pd.DataFrame(rows)


def selective_school_sensitivity(frame: pd.DataFrame, site: str | None = None) -> pd.DataFrame:
    """Refit the models without the schools that admit by exam or audition.

    New York City has eight specialized high schools that admit on a citywide
    entrance exam, plus a handful that audition. They are the extreme of the
    distribution on both variables in this study: they start earlier than
    average, and 90 percent of their students earn an Advanced Regents diploma
    against about 14 percent citywide. Eight schools cannot be allowed to set
    the slope for four hundred, so the models are refitted without them and
    both versions are reported.
    """
    settings = site_config(site)
    excluded = settings.get("robustness_exclude_admissions")
    if not excluded or "admissions_method" not in frame.columns:
        return pd.DataFrame()

    kept = frame[~frame["admissions_method"].isin(excluded)]
    outcomes = [settings["outcomes"]["primary"]] + settings["outcomes"]["secondary"]
    ladders = pd.concat(
        [model_ladder(kept, outcome, site=site) for outcome in outcomes], ignore_index=True
    )
    ladders["specification"] = ladders["specification"] + " (exam and audition schools dropped)"
    return ladders


def censoring_sensitivity(frame: pd.DataFrame, site: str | None = None) -> pd.DataFrame:
    """Refit the graduation rate models after dropping censored values.

    Nevada publishes ">95" instead of a graduation rate for its highest
    performing schools. clean.py replaces that with 97.5. This check refits the
    same models on only the schools with an exact published rate. If the start
    time coefficient moves a lot, the midpoint substitution was doing real work
    and the graduation rate results should be treated with more caution.
    """
    if "outcome_censored" not in frame.columns:
        return pd.DataFrame()

    censored = frame["outcome_censored"].fillna(False).astype(bool)
    if not censored.any():
        # New York City publishes exact rates, so there is nothing to check.
        return pd.DataFrame()

    ladder = model_ladder(frame[~censored], "grad_rate", site=site)
    ladder["specification"] = ladder["specification"] + " (censored values dropped)"
    return ladder


def run(frame: pd.DataFrame, site: str | None = None) -> dict:
    """Run every piece of the analysis and write the tables to this site's outputs."""
    settings = site_config(site)
    site = settings["name"]
    print(f"Running analysis for {settings['label']}")

    out_dir = outputs_dir(site)
    out_dir.mkdir(parents=True, exist_ok=True)

    primary = settings["outcomes"]["primary"]
    outcomes = [primary] + settings["outcomes"]["secondary"]

    descriptives = descriptive_statistics(frame)
    descriptives.to_csv(out_dir / "descriptive_statistics.csv", index=False)

    correlations = pd.DataFrame(
        [unadjusted_correlation(frame, outcome) for outcome in outcomes]
    )
    correlations.to_csv(out_dir / "unadjusted_correlations.csv", index=False)

    ladders = pd.concat(
        [model_ladder(frame, outcome, site=site) for outcome in outcomes], ignore_index=True
    )
    ladders.to_csv(out_dir / "model_ladder.csv", index=False)

    strata = pd.concat(
        [stratified_comparison(frame, outcome, site=site) for outcome in outcomes],
        ignore_index=True,
    )
    strata.to_csv(out_dir / "stratified_comparison.csv", index=False)

    sensitivity = censoring_sensitivity(frame, site)
    if not sensitivity.empty:
        sensitivity.to_csv(out_dir / "censoring_sensitivity.csv", index=False)

    selective = selective_school_sensitivity(frame, site)
    if not selective.empty:
        selective.to_csv(out_dir / "selective_school_sensitivity.csv", index=False)

    primary_row = correlations[correlations["outcome"] == primary].iloc[0]
    print(f"  unadjusted correlation for {OUTCOME_LABELS.get(primary, primary)}: "
          f"r = {primary_row['r']:.3f} (n = {primary_row['n']})")

    return {
        "descriptives": descriptives,
        "correlations": correlations,
        "ladders": ladders,
        "strata": strata,
        "sensitivity": sensitivity,
        "selective": selective,
    }
