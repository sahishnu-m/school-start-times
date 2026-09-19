"""Checks the hand written statistics against statsmodels and scipy.

The analysis used to call statsmodels and scipy. Both are compiled extensions
built against a particular numpy, and on the deployed dashboard a slightly
different set of versions got installed and importing them failed, which took
the whole page down. src/stats_core.py replaces them with plain numpy.

Replacing a trusted library with your own arithmetic is only acceptable if you
check it, so this file does the checking. It refits every model the study
actually runs, both ways, and compares.

Run it with statsmodels and scipy installed:

    python -m pytest tests/test_stats_core.py -v

or directly, which prints a summary and needs no test runner:

    python tests/test_stats_core.py

The comparison libraries are only needed here. Nothing the pipeline or the
dashboard imports depends on them.

One deliberate difference. statsmodels builds its confidence intervals from
normal critical values by default, even for cluster robust errors. This project
uses Student t instead, with n minus rank degrees of freedom for HC3 and
clusters minus one for cluster robust errors. That is the Stata convention and
it is slightly more conservative, so the intervals here are a little wider than
the ones statsmodels reports. Coefficients and standard errors agree to about
one part in a billion, which is what this file asserts.
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.config import analysis_table_path, site_config  # noqa: E402
from src.stats_core import (  # noqa: E402
    design_matrix,
    fit_ols,
    pearson_correlation,
    t_critical,
    t_two_sided_p,
    welch_test,
)

statsmodels = pytest.importorskip("statsmodels.formula.api", reason="comparison library")
scipy_stats = pytest.importorskip("scipy.stats", reason="comparison library")


def test_t_distribution_matches_scipy():
    """The t tail probability and its inverse agree with scipy."""
    for df in [1, 2, 5, 10, 30, 100, 404, 1.0e7]:
        for t_value in [0.0, 0.5, 1.0, 1.96, 3.0, 7.5]:
            assert t_two_sided_p(t_value, df) == pytest.approx(
                2 * scipy_stats.t.sf(abs(t_value), df), abs=1e-8
            )
        assert t_critical(df) == pytest.approx(scipy_stats.t.ppf(0.975, df), abs=1e-6)


def _model_cases():
    """Every outcome and control block the study fits, for both sites."""
    for site in ["nyc", "nevada"]:
        settings = site_config(site)
        path = analysis_table_path(site)
        if not path.exists():
            continue
        frame = pd.read_csv(path)
        cluster = settings.get("cluster_column")
        outcomes = [settings["outcomes"]["primary"]] + settings["outcomes"]["secondary"]
        for block in settings["model_blocks"]:
            for outcome in outcomes:
                yield site, frame, cluster, block["controls"], outcome


def test_ols_matches_statsmodels():
    """Coefficients, standard errors and R squared agree with statsmodels."""
    compared = 0
    for site, frame, cluster, controls, outcome in _model_cases():
        needed = ["start_hours", outcome]
        needed += [c.replace("C(", "").replace(")", "") for c in controls]
        if cluster:
            needed.append(cluster)
        needed = [c for c in dict.fromkeys(needed) if c in frame.columns]
        subset = frame.dropna(subset=needed)

        if len(subset) < len(controls) + 10:
            continue
        if any(c.startswith("C(") and subset[c[2:-1]].nunique() < 2 for c in controls):
            continue

        formula = f"Q('{outcome}') ~ start_hours" + "".join(f" + {c}" for c in controls)
        use_cluster = bool(cluster) and cluster in subset.columns and subset[cluster].nunique() > 1
        if use_cluster:
            reference = statsmodels.ols(formula, data=subset).fit(
                cov_type="cluster", cov_kwds={"groups": subset[cluster]}
            )
            groups = subset[cluster].to_numpy()
        else:
            reference = statsmodels.ols(formula, data=subset).fit(cov_type="HC3")
            groups = None

        X, names = design_matrix(subset, ["start_hours"] + controls)
        mine = fit_ols(subset[outcome].to_numpy(), X, names, clusters=groups)
        term = mine.for_term("start_hours")

        assert term["coefficient"] == pytest.approx(reference.params["start_hours"], abs=1e-8)
        assert term["std_error"] == pytest.approx(reference.bse["start_hours"], abs=1e-8)
        assert term["r_squared"] == pytest.approx(reference.rsquared, abs=1e-10)
        compared += 1

    assert compared >= 30, f"only {compared} models compared, expected the full set"


def test_confidence_intervals_use_t_not_normal():
    """The intervals are the t version, which is wider than the statsmodels default.

    This is the one place the two implementations are meant to disagree, so it
    is asserted rather than left as a surprise.
    """
    frame = pd.read_csv(analysis_table_path("nyc"))
    subset = frame.dropna(subset=["start_hours", "grad_rate", "poverty_pct", "building_code"])

    reference = statsmodels.ols("grad_rate ~ start_hours + poverty_pct", data=subset).fit(
        cov_type="cluster", cov_kwds={"groups": subset["building_code"]}
    )
    X, names = design_matrix(subset, ["start_hours", "poverty_pct"])
    mine = fit_ols(
        subset["grad_rate"].to_numpy(), X, names, clusters=subset["building_code"].to_numpy()
    )
    term = mine.for_term("start_hours")

    reference_ci = reference.conf_int().loc["start_hours"]
    mine_width = term["ci_high"] - term["ci_low"]
    reference_width = reference_ci.iloc[1] - reference_ci.iloc[0]

    assert mine_width > reference_width
    # Wider, but only slightly. A large gap would mean the degrees of freedom
    # were wrong rather than merely more careful.
    assert mine_width / reference_width < 1.05

    groups = subset["building_code"].nunique()
    expected = t_critical(groups - 1) / scipy_stats.norm.ppf(0.975)
    assert mine_width / reference_width == pytest.approx(expected, rel=1e-6)


def test_correlation_matches_scipy():
    frame = pd.read_csv(analysis_table_path("nyc")).dropna(subset=["start_hours", "grad_rate"])
    mine = pearson_correlation(frame["start_hours"], frame["grad_rate"])
    r, p = scipy_stats.pearsonr(frame["start_hours"], frame["grad_rate"])
    assert mine["r"] == pytest.approx(r, abs=1e-12)
    assert mine["p_value"] == pytest.approx(p, abs=1e-10)


def test_welch_matches_scipy():
    frame = pd.read_csv(analysis_table_path("nyc")).dropna(subset=["start_hours", "grad_rate"])
    early = frame.loc[frame["start_hours"] < 8.25, "grad_rate"].to_numpy()
    late = frame.loc[frame["start_hours"] >= 8.25, "grad_rate"].to_numpy()

    mine = welch_test(early, late)
    reference = scipy_stats.ttest_ind(early, late, equal_var=False)

    assert mine["p_value"] == pytest.approx(reference.pvalue, abs=1e-12)
    assert mine["df"] == pytest.approx(reference.df, rel=1e-12)


if __name__ == "__main__":
    # Runnable without pytest, so the check is easy to repeat.
    test_t_distribution_matches_scipy()
    print("t distribution matches scipy")
    test_ols_matches_statsmodels()
    print("OLS coefficients and standard errors match statsmodels")
    test_confidence_intervals_use_t_not_normal()
    print("confidence intervals use t, slightly wider than statsmodels by design")
    test_correlation_matches_scipy()
    print("correlation matches scipy")
    test_welch_matches_scipy()
    print("Welch test matches scipy")
    print("\nall checks passed")
