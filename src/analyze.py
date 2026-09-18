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
import statsmodels.formula.api as smf
from scipy import stats

from .config import CONFIG, OUTPUTS_DIR

# Robust standard errors. Schools vary enormously in size, so the spread of an
# outcome around the regression line is larger for a school of 200 students
# than for one of 3000. That is textbook heteroskedasticity, and it makes the
# ordinary standard errors too small. HC3 is the variant that behaves best in
# small samples, which is what this study has.
COVARIANCE_TYPE = "HC3"

# Human readable names used in tables and on the dashboard.
VARIABLE_LABELS = {
    "start_hours": "Start time (per hour later)",
    "direct_cert_pct": "Direct certification rate (percent)",
    "enrollment_1000": "Enrollment (per 1,000 students)",
    "el_pct": "English learners (percent)",
    "C(locale_group)": "Locale (city, suburb, town, rural)",
    "C(district)": "District fixed effects",
}

OUTCOME_LABELS = {
    "act_composite": "ACT composite score",
    "grad_rate": "Four year graduation rate (percent)",
    "ela_proficient_pct": "ELA proficiency (percent)",
    "math_proficient_pct": "Math proficiency (percent)",
    "chronic_absent_pct": "Chronic absenteeism (percent)",
}


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
    n = len(subset)
    if n < 4:
        return {"outcome": outcome, "n": n, "r": np.nan, "ci_low": np.nan,
                "ci_high": np.nan, "p_value": np.nan}

    r, p_value = stats.pearsonr(subset["start_hours"], subset[outcome])

    z = np.arctanh(r)
    standard_error = 1.0 / np.sqrt(n - 3)
    critical = stats.norm.ppf(0.975)
    ci_low = np.tanh(z - critical * standard_error)
    ci_high = np.tanh(z + critical * standard_error)

    return {
        "outcome": outcome,
        "outcome_label": OUTCOME_LABELS.get(outcome, outcome),
        "n": n,
        "r": r,
        "ci_low": ci_low,
        "ci_high": ci_high,
        "p_value": p_value,
    }


def fit_one_model(frame: pd.DataFrame, outcome: str, controls: list[str]) -> dict | None:
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

    formula = f"Q('{outcome}') ~ start_hours"
    for control in controls:
        formula += f" + {control}"

    try:
        model = smf.ols(formula, data=subset).fit(cov_type=COVARIANCE_TYPE)
    except Exception:
        return None

    if "start_hours" not in model.params.index:
        return None

    confidence = model.conf_int().loc["start_hours"]
    return {
        "coefficient": float(model.params["start_hours"]),
        "std_error": float(model.bse["start_hours"]),
        "ci_low": float(confidence.iloc[0]),
        "ci_high": float(confidence.iloc[1]),
        "p_value": float(model.pvalues["start_hours"]),
        "n": int(model.nobs),
        "r_squared": float(model.rsquared),
        "adj_r_squared": float(model.rsquared_adj),
        "model": model,
    }


def model_ladder(frame: pd.DataFrame, outcome: str, blocks: list[dict] | None = None) -> pd.DataFrame:
    """Fit the sequence of models defined in config.yaml.

    The returned table is the centrepiece of the results: one row per model
    specification, showing what happens to the start time coefficient as
    controls are added.
    """
    blocks = blocks or CONFIG["model_blocks"]
    rows = []

    for index, block in enumerate(blocks, start=1):
        controls = block["controls"]
        result = fit_one_model(frame, outcome, controls)
        unidentified = result.pop("unidentified", None) if result else None

        row = {
            "model_number": index,
            "specification": block["name"],
            "controls": ", ".join(VARIABLE_LABELS.get(c, c) for c in controls) or "none",
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
            result.pop("model")
            row.update(result)
            row["note"] = ""
        rows.append(row)

    return pd.DataFrame(rows)


def stratified_comparison(frame: pd.DataFrame, outcome: str) -> pd.DataFrame:
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

    for band in CONFIG["stratification"]["poverty_band_labels"]:
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
            difference = late.mean() - early.mean()
            test = stats.ttest_ind(late, early, equal_var=False)

            # Welch interval, built by hand so the degrees of freedom match the
            # test and the interval cannot disagree with the p value.
            se = np.sqrt(late.var(ddof=1) / len(late) + early.var(ddof=1) / len(early))
            critical = stats.t.ppf(0.975, test.df)
            row.update(
                {
                    "difference_late_minus_early": difference,
                    "ci_low": difference - critical * se,
                    "ci_high": difference + critical * se,
                    "p_value": float(test.pvalue),
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
    columns = [
        "start_minutes", "start_hours", "act_composite", "grad_rate",
        "ela_proficient_pct", "math_proficient_pct", "chronic_absent_pct",
        "direct_cert_pct", "frl_pct_ccd", "el_pct", "enrollment",
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


def censoring_sensitivity(frame: pd.DataFrame) -> pd.DataFrame:
    """Refit the graduation rate models after dropping censored values.

    Nevada publishes ">95" instead of a graduation rate for its highest
    performing schools. clean.py replaces that with 97.5. This check refits the
    same models on only the schools with an exact published rate. If the start
    time coefficient moves a lot, the midpoint substitution was doing real work
    and the graduation rate results should be treated with more caution.
    """
    if "grad_rate_censored" not in frame.columns:
        return pd.DataFrame()

    uncensored = frame[frame["grad_rate_censored"] != True]  # noqa: E712
    ladder = model_ladder(uncensored, "grad_rate")
    ladder["specification"] = ladder["specification"] + " (censored values dropped)"
    return ladder


def run(frame: pd.DataFrame) -> dict:
    """Run every piece of the analysis and write the tables to /outputs."""
    print("Running analysis")
    OUTPUTS_DIR.mkdir(parents=True, exist_ok=True)

    primary = CONFIG["outcomes"]["primary"]
    outcomes = [primary] + CONFIG["outcomes"]["secondary"]

    descriptives = descriptive_statistics(frame)
    descriptives.to_csv(OUTPUTS_DIR / "descriptive_statistics.csv", index=False)

    correlations = pd.DataFrame(
        [unadjusted_correlation(frame, outcome) for outcome in outcomes]
    )
    correlations.to_csv(OUTPUTS_DIR / "unadjusted_correlations.csv", index=False)

    ladders = pd.concat(
        [model_ladder(frame, outcome) for outcome in outcomes], ignore_index=True
    )
    ladders.to_csv(OUTPUTS_DIR / "model_ladder.csv", index=False)

    strata = pd.concat(
        [stratified_comparison(frame, outcome) for outcome in outcomes], ignore_index=True
    )
    strata.to_csv(OUTPUTS_DIR / "stratified_comparison.csv", index=False)

    sensitivity = censoring_sensitivity(frame)
    if not sensitivity.empty:
        sensitivity.to_csv(OUTPUTS_DIR / "censoring_sensitivity.csv", index=False)

    primary_row = correlations[correlations["outcome"] == primary].iloc[0]
    print(f"  unadjusted correlation for {OUTCOME_LABELS.get(primary, primary)}: "
          f"r = {primary_row['r']:.3f} (n = {primary_row['n']})")

    return {
        "descriptives": descriptives,
        "correlations": correlations,
        "ladders": ladders,
        "strata": strata,
        "sensitivity": sensitivity,
    }
