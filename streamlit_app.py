"""Streamlit dashboard for the Nevada school start time study.

The point of the dashboard is the controls panel in the results tab. A reader
can switch each control variable on and off and watch the start time
coefficient move. That is a better way to understand confounding than reading a
table of models somebody else chose, because the reader does the switching
themselves and sees which variable is doing the work.

This file sits at the repository root because Streamlit Community Cloud expects
the main script there.
"""

from __future__ import annotations

from pathlib import Path

import pandas as pd
import streamlit as st

from src.analyze import OUTCOME_LABELS, fit_one_model, stratified_comparison, unadjusted_correlation
from src.clean import minutes_to_clock
from src.config import ANALYSIS_TABLE, CONFIG, OUTPUTS_DIR

st.set_page_config(
    page_title="Nevada school start times",
    page_icon="clock",
    # Centered rather than wide. A wide layout puts content in columns that a
    # phone has to scroll sideways to read.
    layout="centered",
    initial_sidebar_state="collapsed",
)

# Mobile adjustments. Streamlit stacks columns on a narrow screen on its own,
# so the work left here is trimming the default page padding, which wastes a
# lot of a phone screen, and stopping wide tables from forcing a sideways
# scroll of the whole page.
st.markdown(
    """
    <style>
      .block-container { padding-top: 2rem; padding-bottom: 3rem; }
      @media (max-width: 640px) {
        .block-container { padding-left: 1rem; padding-right: 1rem; }
        h1 { font-size: 1.6rem !important; }
        h2 { font-size: 1.25rem !important; }
      }
      div[data-testid="stDataFrame"] { overflow-x: auto; }
    </style>
    """,
    unsafe_allow_html=True,
)


@st.cache_data
def load_data() -> pd.DataFrame | None:
    """Read the merged analysis table, if the pipeline has produced one."""
    if not ANALYSIS_TABLE.exists():
        return None
    return pd.read_csv(ANALYSIS_TABLE, dtype={"nrc_school_code": str})


def show_image_if_present(filename: str, caption: str) -> None:
    """Display a chart from /outputs, quietly skipping one that is not there."""
    path = OUTPUTS_DIR / filename
    if path.exists():
        st.image(str(path), caption=caption, use_container_width=True)


frame = load_data()

st.title("Do later school start times go with better academic performance?")
st.caption("Nevada public high schools, 2022-2023 school year")

if frame is None:
    st.error(
        "No analysis table found. Run `python run_pipeline.py` to build "
        "data/processed/analysis_table.csv, then reload this page."
    )
    st.stop()

with_start_time = frame["start_minutes"].notna().sum()

tab_question, tab_data, tab_results, tab_limits = st.tabs(
    ["Question", "Data and method", "Results", "Limitations"]
)

# ---------------------------------------------------------------------------
# Research question
# ---------------------------------------------------------------------------
with tab_question:
    st.header("The research question")
    st.markdown(
        """
        Do Nevada public high schools with later start times show higher
        academic performance than schools with earlier start times, after
        controlling for student demographics and school characteristics?

        The reason this question needs the second half of that sentence is that
        start times are not handed out at random. A district staggers its bell
        schedules so one set of buses can run several routes each morning.
        Which schools get the early route depends on where the buses are based
        and how far they travel, and that tends to line up with neighbourhood
        income. So a plain comparison of early and late schools is partly a
        comparison of richer and poorer neighbourhoods.

        This study reports the plain comparison first, labels it as unadjusted,
        and then adds controls one at a time so a reader can see how much of
        the plain comparison survives.
        """
    )

    st.subheader("Background from the sleep research")
    st.markdown(
        """
        Existing research on adolescent sleep is the reason anyone asks this
        question. Adolescent circadian rhythms shift later during puberty, so a
        teenager asked to be alert at 7:30 in the morning is being asked
        something biologically harder than the same request to an adult. The
        American Academy of Pediatrics recommended in 2014 that middle and high
        schools start no earlier than 8:30 AM. Studies that followed districts
        through an actual bell schedule change, such as the Seattle study
        published in Science Advances in 2018, reported longer sleep and
        improved grades after the change.

        That literature is background for why the question matters. It is not
        evidence for what this study finds. This study looks at Nevada schools
        at one point in time, which is a much weaker design than following the
        same schools through a change, and it cannot inherit the credibility of
        those studies.
        """
    )

# ---------------------------------------------------------------------------
# Data and method
# ---------------------------------------------------------------------------
with tab_data:
    st.header("Where the data comes from")
    st.markdown(
        """
        **Nevada Report Card** (nevadareportcard.nv.gov) is the Nevada
        Department of Education accountability site. It supplies graduation
        rates, ACT scores, proficiency rates, chronic absenteeism, and English
        learner percentage. The numbers are pulled from the same API the public
        site uses, so they match what a person would read off the website.

        **NCES Common Core of Data** is the federal census of public schools,
        reached through the Urban Institute Education Data API. It supplies
        enrollment, the city, suburb, town, or rural classification, and the
        direct certification count.

        **Start times** are collected by hand and by a scraper that reads
        district and school websites. No agency publishes them as a dataset.
        The scraper checks robots.txt before every site, waits between requests,
        and records every page it could not read along with the reason. Where a
        hand entered time and a scraped time disagree, the hand entered time is
        used.
        """
    )

    st.subheader("How the sample was built")
    sample_path = OUTPUTS_DIR / "sample_construction.csv"
    if sample_path.exists():
        st.dataframe(pd.read_csv(sample_path), hide_index=True, use_container_width=True)

    st.subheader("Why direct certification instead of free and reduced lunch")
    st.markdown(
        f"""
        The obvious poverty measure is the percentage of students on free or
        reduced price lunch, and this study does not use it as the main control.

        Clark County School District feeds every student free of charge under
        the federal Community Eligibility Provision, so its reported free lunch
        percentage is 100 for every school regardless of how wealthy the
        neighbourhood is. In this sample the median reported free lunch rate is
        {frame['frl_pct_ccd'].median():.0f} percent, which is not a description
        of Nevada poverty, it is a description of a billing rule.

        Direct certification counts students confirmed low income through SNAP,
        TANF, or foster care records. It is unaffected by the feeding rule. In
        this sample it ranges from
        {frame['direct_cert_pct'].min():.0f} to
        {frame['direct_cert_pct'].max():.0f} percent with a median of
        {frame['direct_cert_pct'].median():.0f}, which is the kind of spread a
        control variable needs in order to control for anything.

        Reported free lunch percentage is still in the data file so anyone can
        check this reasoning.
        """
    )

    st.subheader("The method")
    st.markdown(
        """
        1. Report the correlation between start time and each outcome with no
           controls, clearly labelled as unadjusted.
        2. Fit a sequence of regressions, adding one block of controls at a
           time, and report the start time coefficient at every step.
        3. Split schools into poverty bands and compare early and late starters
           inside each band, which does not assume the relationship is a
           straight line.
        4. Report a sample size and a confidence interval with every estimate.

        Standard errors are heteroskedasticity robust, because the spread of an
        outcome around the fitted line is wider for a school of 200 students
        than for one of 3,000.
        """
    )

# ---------------------------------------------------------------------------
# Results
# ---------------------------------------------------------------------------
with tab_results:
    st.header("Results")

    if with_start_time == 0:
        st.warning(
            "No start times have been collected yet, so no result can be "
            "computed. Fill in the start_time column in "
            "data/manual/start_times.csv and run the pipeline again. "
            "The study deliberately does not invent start times to fill the gap."
        )
        st.stop()

    outcome_choice = st.selectbox(
        "Outcome",
        options=[CONFIG["outcomes"]["primary"]] + CONFIG["outcomes"]["secondary"],
        format_func=lambda key: OUTCOME_LABELS.get(key, key),
    )

    st.subheader("Step 1. The unadjusted correlation")
    correlation = unadjusted_correlation(frame, outcome_choice)
    left, right = st.columns(2)
    left.metric("Correlation (r)", f"{correlation['r']:.3f}")
    right.metric("Schools", f"{correlation['n']}")
    st.caption(
        f"95% confidence interval {correlation['ci_low']:.3f} to "
        f"{correlation['ci_high']:.3f}. This number has no controls in it and "
        "should not be read as an effect of start times."
    )

    st.subheader("Step 2. Add controls and watch the coefficient")
    st.markdown(
        "Each switch adds one control to the regression. The number below is "
        "the estimated change in the outcome for a school starting one hour "
        "later."
    )

    # The interactive core of the dashboard.
    available_controls = {
        "direct_cert_pct": "Student poverty (direct certification rate)",
        "enrollment_1000": "School size (enrollment)",
        "el_pct": "English learner percentage",
        "C(locale_group)": "Locale (city, suburb, town, rural)",
        "C(district)": "District fixed effects",
    }
    chosen = [key for key, label in available_controls.items() if st.checkbox(label, value=False)]

    result = fit_one_model(frame, outcome_choice, chosen)
    unidentified = result.pop("unidentified", None) if result else None

    if unidentified:
        st.warning(
            f"This control cannot be estimated. Every school in the sample with "
            f"a known start time has the same {unidentified}, so there is "
            "nothing for this control to compare. Switch it off."
        )
    elif result is None:
        st.warning(
            "This combination of controls cannot be estimated with the number "
            "of schools available. Switch some controls off."
        )
    else:
        column_one, column_two = st.columns(2)
        column_one.metric(
            "Change per hour later start", f"{result['coefficient']:+.3f}"
        )
        column_two.metric("Schools in this model", f"{result['n']}")
        st.caption(
            f"95% confidence interval {result['ci_low']:+.3f} to "
            f"{result['ci_high']:+.3f}. "
            f"R squared {result['r_squared']:.3f}."
        )

        # State plainly whether the interval includes zero. A reader should not
        # have to compare two signed numbers in their head to get the headline.
        if result["ci_low"] <= 0 <= result["ci_high"]:
            st.info(
                "The confidence interval includes zero. With this sample, the "
                "data is consistent with there being no relationship between "
                "start time and this outcome once these controls are in place."
            )
        else:
            direction = "higher" if result["coefficient"] > 0 else "lower"
            st.success(
                f"The confidence interval does not include zero. Later starting "
                f"schools score {direction} on this outcome in this sample, "
                "with these controls. This is still not evidence of cause."
            )

    st.subheader("Step 3. The full ladder of models")
    ladder_path = OUTPUTS_DIR / "model_ladder.csv"
    if ladder_path.exists():
        ladder = pd.read_csv(ladder_path)
        ladder = ladder[ladder["outcome"] == outcome_choice]
        display = ladder[
            ["model_number", "specification", "coefficient", "ci_low", "ci_high", "n", "r_squared"]
        ].rename(
            columns={
                "model_number": "Model",
                "specification": "Specification",
                "coefficient": "Per hour later",
                "ci_low": "CI low",
                "ci_high": "CI high",
                "n": "Schools",
                "r_squared": "R squared",
            }
        )
        st.dataframe(display.round(3), hide_index=True, use_container_width=True)

    show_image_if_present(
        f"03_coefficient_plot_{outcome_choice}.png",
        "The start time estimate and its interval across model specifications",
    )

    st.subheader("Step 4. Comparison within similar poverty bands")
    strata = stratified_comparison(frame, outcome_choice)
    display_strata = strata[
        ["poverty_band", "n_early", "n_late", "mean_early", "mean_late",
         "difference_late_minus_early", "ci_low", "ci_high"]
    ].rename(
        columns={
            "poverty_band": "Poverty band",
            "n_early": "Early schools",
            "n_late": "Late schools",
            "mean_early": "Mean, early",
            "mean_late": "Mean, late",
            "difference_late_minus_early": "Late minus early",
            "ci_low": "CI low",
            "ci_high": "CI high",
        }
    )
    st.dataframe(display_strata.round(2), hide_index=True, use_container_width=True)
    show_image_if_present(
        f"04_stratified_{outcome_choice}.png", "Early and late starters within poverty bands"
    )

    st.subheader("The data behind all of this")
    show_image_if_present("01_start_time_distribution.png", "Start time distribution")
    show_image_if_present(
        f"02_unadjusted_scatter_{outcome_choice}.png",
        "Unadjusted relationship, shown with no controls",
    )

# ---------------------------------------------------------------------------
# Limitations
# ---------------------------------------------------------------------------
with tab_limits:
    st.header("Limitations")
    st.markdown(
        f"""
        ### This study cannot show that start times cause anything

        Schools were not assigned their start times at random. A district sets
        bell schedules around bus routes, and bus routes follow geography and
        budget. Any school characteristic that travels with geography and budget
        also travels with start time. Controlling for the characteristics
        measured here removes the part of that overlap that was measured. It
        does nothing about the part that was not.

        To show cause you would need either random assignment, which nobody is
        going to do to a school district, or a study that follows the same
        schools through an actual bell schedule change and compares them to
        similar schools that did not change. This study does neither.

        ### The data is cross sectional

        Every school is observed once, in the 2022-2023 school year. There is
        no before and after. A school that starts late and scores well might
        have scored just as well when it started early, and this data cannot
        tell the difference.

        ### The sample is small

        {len(frame)} Nevada high schools meet the sample criteria, and
        {with_start_time} of them have a known start time. That is a small
        sample for a regression with several controls. It is why every estimate
        here is reported with a confidence interval. A wide interval is the
        honest description of what a sample this size can settle, and several of
        the intervals here are wide.

        The sample is also concentrated. Clark County alone accounts for
        {(frame['district'] == 'Clark').sum()} of the schools. A result driven
        by one district is a result about that district.

        ### Start times are imperfectly measured

        Start times were collected from school websites, which are not always
        current and do not always distinguish a first bell from a warning bell.
        Some schools run different schedules on different days of the week, and
        a single number cannot represent that. Errors in a predictor variable
        generally push an estimated relationship toward zero, so the estimates
        here may understate a real relationship if one exists.

        ### Some published values are censored

        Nevada publishes ">95" rather than an exact graduation rate for its
        highest performing schools. Those are replaced with the midpoint of the
        implied range. The analysis includes a sensitivity check that drops
        those schools, and any conclusion about graduation rates should be read
        alongside it.

        ### A null result is still a result

        If the confidence intervals include zero once controls are added, the
        correct conclusion is that this data does not show a relationship. That
        is not a failed study. It is a finding, and it is reported as one rather
        than being buried.
        """
    )

st.divider()
st.caption(
    "Sources: Nevada Report Card, Nevada Department of Education. "
    "Common Core of Data, National Center for Education Statistics, via the "
    "Urban Institute Education Data API. Start times collected from district "
    "and school websites."
)
