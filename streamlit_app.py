"""Streamlit dashboard for the school start time study.

The point of the dashboard is the controls panel in the results tab. A reader
can switch each control variable on and off and watch the start time
coefficient move. That is a better way to understand confounding than reading a
table of models somebody else chose, because the reader does the switching
themselves and sees which variable is doing the work.

This file sits at the repository root because Streamlit Community Cloud expects
the main script there.
"""

from __future__ import annotations

import pandas as pd
import streamlit as st

from src.analyze import (
    OUTCOME_LABELS,
    fit_one_model,
    stratified_comparison,
    unadjusted_correlation,
    variable_label,
)
from src.clean import minutes_to_clock
from src.config import CONFIG, analysis_table_path, outputs_dir, site_config

st.set_page_config(
    page_title="School start times and academic performance",
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
        h1 { font-size: 1.5rem !important; }
        h2 { font-size: 1.2rem !important; }
      }
      div[data-testid="stDataFrame"] { overflow-x: auto; }
    </style>
    """,
    unsafe_allow_html=True,
)


@st.cache_data
def load_data(site: str) -> pd.DataFrame | None:
    """Read one site's merged analysis table, if the pipeline has produced one."""
    path = analysis_table_path(site)
    if not path.exists():
        return None
    return pd.read_csv(path)


@st.cache_data
def load_table(site: str, name: str) -> pd.DataFrame | None:
    path = outputs_dir(site) / name
    return pd.read_csv(path) if path.exists() else None


def show_image(site: str, filename: str, caption: str) -> None:
    """Display a chart, quietly skipping one that has not been generated."""
    path = outputs_dir(site) / filename
    if path.exists():
        st.image(str(path), caption=caption, use_container_width=True)


st.title("Do later school start times go with better academic performance?")

# The site picker. New York City is the analysis that can answer the question.
# Nevada is kept because it shows why the question is hard to answer at all.
site_names = list(CONFIG["sites"].keys())
site = st.radio(
    "Sample",
    options=site_names,
    format_func=lambda key: site_config(key)["label"],
    horizontal=True,
)
settings = site_config(site)
st.caption(f"{settings['label']}, {settings['year_label']}")

frame = load_data(site)
if frame is None:
    st.error(
        f"No analysis table found for {site}. Run "
        f"`python run_pipeline.py --site {site}` and reload this page."
    )
    st.stop()

with_start_time = int(frame["start_minutes"].notna().sum())

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
        Do public high schools with later start times show higher academic
        performance than schools with earlier start times, after controlling for
        student demographics and school characteristics?

        The reason the question needs the second half of that sentence is that
        start times are not handed out at random. A district staggers its bell
        schedules so one set of buses can run several routes each morning. Which
        schools get the early route depends on where the buses are based and how
        far they travel, and that tends to line up with neighbourhood income. So
        a plain comparison of early and late schools is partly a comparison of
        richer and poorer neighbourhoods.

        This study reports the plain comparison first, labels it as unadjusted,
        and then adds controls one at a time so a reader can see how much of the
        plain comparison survives.
        """
    )

    st.subheader("Why two samples")
    st.markdown(
        """
        The study began in Nevada and ran into a wall. Clark County sets one
        bell time for nearly all of its high schools, so 29 of the 38 Nevada
        schools with a known start time begin at exactly 7:00 AM. A study of
        whether start time predicts performance needs schools that start at
        different times, and Nevada does not supply them.

        New York City does. It publishes a start time for every high school, and
        those times run from 7:15 AM to past 9:00 AM across more than 400
        schools. Because every one of those schools sits inside a single school
        system, the things that differ between districts, such as busing
        budgets, union contracts, and state funding, are held constant instead
        of being tangled up with start time.

        Nevada is kept in the dashboard because it is a useful demonstration of
        the problem, and because a study that hides the sample that did not work
        is not being honest about how it got its answer.
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
        evidence for what this study finds. This study looks at schools at one
        point in time, which is a much weaker design than following the same
        schools through a change, and it cannot inherit the credibility of those
        studies.
        """
    )

# ---------------------------------------------------------------------------
# Data and method
# ---------------------------------------------------------------------------
with tab_data:
    st.header("Where the data comes from")

    if site == "nyc":
        st.markdown(
            """
            Everything comes from NYC Open Data, joined on the DBN, which is the
            code that identifies a New York City school.

            **2019 DOE High School Directory** supplies the start time, the
            attendance rate, the college and career readiness rate, the
            admissions method, and the number of applicants per seat.

            **2018-19 School Demographic Snapshot** supplies the Economic Need
            Index, the English learner percentage, the share of students with
            disabilities, and enrollment.

            **Graduation results for cohorts 2012 to 2019** supplies the four
            year graduation rate, the Advanced Regents diploma rate, and the
            dropout rate for the cohort that entered in 2015 and was due to
            finish in June 2019.
            """
        )
    else:
        st.markdown(
            """
            **Nevada Report Card** supplies graduation rates, ACT scores,
            proficiency rates, chronic absenteeism, and English learner
            percentage, pulled from the same API the public site uses.

            **NCES Common Core of Data**, through the Urban Institute Education
            Data API, supplies enrollment, the city, suburb, town, or rural
            classification, and the direct certification count.

            **Start times** come from a Clark County district document, a
            website scraper, and a hand entered file that overrides both. No
            agency publishes Nevada start times as a dataset.
            """
        )

    st.subheader("How the sample was built")
    sample_steps = load_table(site, "sample_construction.csv")
    if sample_steps is not None:
        st.dataframe(sample_steps, hide_index=True, use_container_width=True)

    st.subheader("The poverty measure")
    if site == "nyc":
        st.markdown(
            f"""
            Poverty is measured with the Economic Need Index, which estimates
            the share of students facing economic hardship using public
            assistance records, neighbourhood poverty, and family circumstances.

            Free lunch eligibility is not used. New York City offers free meals
            to every student, so its reported free lunch rate says nothing about
            how poor a school is. Clark County does the same thing, which is why
            the Nevada sample had to find a different measure too.

            In this sample the index runs from
            {frame['poverty_pct'].min():.0f} to {frame['poverty_pct'].max():.0f}
            percent, with a median of {frame['poverty_pct'].median():.0f}.
            """
        )
    else:
        st.markdown(
            f"""
            Poverty is measured with the direct certification rate: students
            confirmed low income through SNAP, TANF, or foster care records.

            Free and reduced lunch percentage is unusable in Nevada. Clark
            County feeds every student under the federal Community Eligibility
            Provision, so its reported rate is 100 percent at every school
            regardless of neighbourhood income. Across this sample the median
            reported free lunch rate is
            {frame['frl_pct_ccd'].median():.0f} percent, which describes a
            billing rule rather than Nevada poverty.

            Direct certification runs from {frame['poverty_pct'].min():.0f} to
            {frame['poverty_pct'].max():.0f} percent here, which is the kind of
            spread a control variable needs in order to control for anything.
            """
        )

    if site == "nyc":
        st.subheader("Why admissions method is a control")
        st.markdown(
            """
            New York City high schools are chosen, not zoned. A screened school,
            an audition school, or one of the eight specialized schools that
            admit on a citywide exam takes students who were already doing well.
            Its results say as much about who got in as about what the school
            did.

            The last model adds the admissions method and the number of
            applicants per seat. Without them, a school being selective could
            show up as an effect of its start time.
            """
        )

    st.subheader("The method")
    cluster = settings.get("cluster_column")
    st.markdown(
        f"""
        1. Report the correlation between start time and each outcome with no
           controls, clearly labelled as unadjusted.
        2. Fit a sequence of regressions, adding one block of controls at a
           time, and report the start time coefficient at every step.
        3. Split schools into poverty bands and compare early and late starters
           inside each band, which does not assume the relationship is a
           straight line.
        4. Report a sample size and a confidence interval with every estimate.

        {"Standard errors are clustered on the school building. Several New York City schools share one building, and schools in the same building share a facility, a neighbourhood, and often the same pool of applicants, so treating them as independent observations would make the intervals too narrow." if cluster else "Standard errors are heteroskedasticity robust, because the spread of an outcome around the fitted line is wider for a small school than for a large one."}
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
            "data/manual/start_times.csv and run the pipeline again. The study "
            "deliberately does not invent start times to fill the gap."
        )
        st.stop()

    times = frame["start_minutes"].dropna()
    left, middle, right = st.columns(3)
    left.metric("Schools", f"{with_start_time}")
    middle.metric("Earliest", minutes_to_clock(times.min()))
    right.metric("Latest", minutes_to_clock(times.max()))

    if site == "nevada":
        st.warning(
            f"{int((times == times.mode().iloc[0]).sum())} of {len(times)} Nevada "
            f"schools start at exactly {minutes_to_clock(times.mode().iloc[0])}. "
            "There is very little start time variation here, so the estimates "
            "below rest on a handful of schools and should not be read as an "
            "answer to the research question."
        )

    st.subheader("All outcomes at a glance")
    show_image(site, "05_outcome_summary.png",
               "Fully adjusted estimate per hour later, in standard deviations")

    outcome_choice = st.selectbox(
        "Outcome to examine",
        options=[settings["outcomes"]["primary"]] + settings["outcomes"]["secondary"],
        format_func=lambda key: OUTCOME_LABELS.get(key, key),
    )

    st.subheader("Step 1. The unadjusted correlation")
    correlation = unadjusted_correlation(frame, outcome_choice)
    one, two = st.columns(2)
    one.metric("Correlation (r)", f"{correlation['r']:.3f}")
    two.metric("Schools", f"{correlation['n']}")
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

    # Offer exactly the controls this site has, taken from the richest model
    # block so the list stays in step with config.yaml.
    available_controls = []
    for block in settings["model_blocks"]:
        for control in block["controls"]:
            if control not in available_controls:
                available_controls.append(control)

    chosen = [
        control for control in available_controls
        if st.checkbox(variable_label(control, site), value=False, key=f"{site}_{control}")
    ]

    result = fit_one_model(frame, outcome_choice, chosen, site)
    unidentified = result.pop("unidentified", None) if result else None

    if unidentified:
        st.warning(
            f"This control cannot be estimated. Every school in the sample with "
            f"a known start time has the same {unidentified}, so there is "
            "nothing for it to compare. Switch it off."
        )
    elif result is None:
        st.warning(
            "This combination of controls cannot be estimated with the number "
            "of schools available. Switch some controls off."
        )
    else:
        one, two = st.columns(2)
        one.metric("Change per hour later start", f"{result['coefficient']:+.3f}")
        two.metric("Schools in this model", f"{result['n']}")
        st.caption(
            f"95% confidence interval {result['ci_low']:+.3f} to "
            f"{result['ci_high']:+.3f}. R squared {result['r_squared']:.3f}."
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
                f"schools score {direction} on this outcome in this sample, with "
                "these controls. This is still not evidence of cause."
            )

    st.subheader("Step 3. The full ladder of models")
    ladder = load_table(site, "model_ladder.csv")
    if ladder is not None:
        rows = ladder[ladder["outcome"] == outcome_choice]
        display = rows[
            ["model_number", "specification", "coefficient", "ci_low", "ci_high", "n", "r_squared"]
        ].rename(
            columns={
                "model_number": "Model", "specification": "Specification",
                "coefficient": "Per hour later", "ci_low": "CI low",
                "ci_high": "CI high", "n": "Schools", "r_squared": "R squared",
            }
        )
        st.dataframe(display.round(3), hide_index=True, use_container_width=True)
        notes = rows["note"].dropna().astype(str)
        for note in notes[notes.str.strip() != ""].unique():
            st.caption(note)

    show_image(site, f"03_coefficient_plot_{outcome_choice}.png",
               "The start time estimate and its interval across model specifications")

    st.subheader("Step 4. Comparison within similar poverty bands")
    strata = stratified_comparison(frame, outcome_choice, site)
    display_strata = strata[
        ["poverty_band", "n_early", "n_late", "mean_early", "mean_late",
         "difference_late_minus_early", "ci_low", "ci_high"]
    ].rename(
        columns={
            "poverty_band": "Poverty band", "n_early": "Early schools",
            "n_late": "Late schools", "mean_early": "Mean, early",
            "mean_late": "Mean, late",
            "difference_late_minus_early": "Late minus early",
            "ci_low": "CI low", "ci_high": "CI high",
        }
    )
    st.dataframe(display_strata.round(2), hide_index=True, use_container_width=True)
    show_image(site, f"04_stratified_{outcome_choice}.png",
               "Early and late starters within poverty bands")

    selective = load_table(site, "selective_school_sensitivity.csv")
    if selective is not None:
        st.subheader("Robustness: dropping the exam and audition schools")
        st.markdown(
            "The eight specialized high schools admit on a citywide exam. They "
            "start earlier than average and about 90 percent of their students "
            "earn an Advanced Regents diploma, against roughly 14 percent "
            "citywide. Eight schools should not be allowed to set the slope for "
            "four hundred, so here are the same models without them."
        )
        rows = selective[selective["outcome"] == outcome_choice]
        st.dataframe(
            rows[["model_number", "coefficient", "ci_low", "ci_high", "n"]]
            .rename(columns={
                "model_number": "Model", "coefficient": "Per hour later",
                "ci_low": "CI low", "ci_high": "CI high", "n": "Schools",
            })
            .round(3),
            hide_index=True, use_container_width=True,
        )

    st.subheader("The data behind all of this")
    show_image(site, "01_start_time_distribution.png", "Start time distribution")
    show_image(site, f"02_unadjusted_scatter_{outcome_choice}.png",
               "Unadjusted relationship, shown with no controls")

# ---------------------------------------------------------------------------
# Limitations
# ---------------------------------------------------------------------------
with tab_limits:
    st.header("Limitations")
    st.markdown(
        f"""
        ### This study cannot show that start times cause anything

        Schools were not assigned their start times at random. Controlling for
        the characteristics measured here removes the part of the overlap that
        was measured. It does nothing about the part that was not. To show cause
        you would need either random assignment, which nobody is going to do to
        a school district, or a study that follows the same schools through an
        actual bell schedule change and compares them to similar schools that
        did not change. This study does neither.

        ### The data is cross sectional

        Every school is observed once, in {settings['year_label']}. There is no
        before and after. A school that starts late and scores well might have
        scored just as well when it started early, and this data cannot tell the
        difference.

        ### Sample size and coverage

        {with_start_time} schools have a known start time, out of {len(frame)}
        in the sample.
        """
    )

    if site == "nyc":
        st.markdown(
            """
            ### New York City is one city

            Every school here is in one school system. That is the strength of
            this sample and also its limit. Holding the district constant
            removes a pile of confounders, and it also means the result
            describes New York City rather than American high schools. A city
            where most students take public transport to a school they applied
            to is not a typical American district.

            ### Who starts late is not random

            The whole difficulty of this study shows up in one place: schools
            that start later are different kinds of schools. In this sample the
            specialized exam schools start earliest, and small newer schools
            start latest. The models control for poverty, size, English
            learners, disability, borough, and admissions method, but a school
            choosing a later bell time is making a decision that probably
            correlates with other things nobody measured.

            ### The results point in two directions

            Two estimates survive every control and every robustness check, and
            they disagree. Later starting schools have lower Advanced Regents
            diploma rates and higher college and career readiness rates. If a
            later start simply helped students, both would move the same way. A
            split like that is what residual confounding looks like, and it is
            reported rather than tidied away.

            ### Start times are self reported free text

            The directory field is typed by school staff. It contains entries
            such as "8am or 8:45am" and one school listed at "8:20pm". One
            implausible value was set to missing. A single number also cannot
            represent a school that runs different schedules on different days.
            """
        )
    else:
        st.markdown(
            """
            ### There is almost no start time variation to work with

            Clark County sets the same 7:00 AM bell time at nearly all of its
            comprehensive high schools. When most of the sample shares one start
            time there is little for a regression to detect, and district fixed
            effects cannot be estimated at all because every school with a start
            time is in one district. This is a limitation of the data, not a
            finding about sleep.

            ### Start times are imperfectly measured

            They come from school websites and district documents, which are not
            always current and do not always distinguish a first bell from a
            warning bell. Measurement error in a predictor generally pushes an
            estimated relationship toward zero.

            ### Some published values are censored

            Nevada publishes ">95" rather than an exact graduation rate for its
            highest performing schools. Those are replaced with the midpoint of
            the implied range, and a sensitivity check that drops them is
            reported alongside. That check matters here: the graduation rate
            result does not survive it.
            """
        )

    st.markdown(
        """
        ### The sleep literature is background, not support

        The studies cited under the research question used stronger designs than
        this one. This study cannot borrow their credibility, and nothing here
        should be read as confirming them.

        ### A null result is a result

        Where the confidence intervals include zero once controls are added, the
        correct conclusion is that this data does not show a relationship. That
        is reported plainly rather than buried.
        """
    )

st.divider()
st.caption(
    "Sources: NYC Open Data (high school directory, demographic snapshot, "
    "graduation results). Nevada Report Card, Nevada Department of Education. "
    "Common Core of Data, NCES, via the Urban Institute Education Data API."
)
