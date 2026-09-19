"""Streamlit dashboard for the school start time study.

The page is laid out as five numbered parts a reader works through in order:
pick a comparison, see how much start times vary, build a model themselves,
compare every outcome, then read the limitations.

The part that matters most is Part III. A reader switches each control variable
on and off and watches the start time estimate move. That teaches confounding
better than a table of models somebody else chose, because the reader does the
switching and sees which variable did the work.

This file sits at the repository root because Streamlit Community Cloud expects
the main script there. It imports only numpy, pandas, and Streamlit by way of
src, so there is no compiled statistics library to fail on a hosted runtime.
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
    layout="centered",
    initial_sidebar_state="collapsed",
)

# The two series colours, taken from a palette checked for colour vision
# deficiency. Blue is the adjusted estimate, orange the unadjusted one.
BLUE = "#2a78d6"
ORANGE = "#eb6834"
INK = "#0b0b0b"
MUTED = "#6b6a66"
RULE = "#e1e0d9"

st.markdown(
    f"""
    <style>
      /* A narrow measure. Long lines of text are hard to read, and the whole
         page is meant to be read top to bottom rather than scanned. */
      .block-container {{
          max-width: 780px;
          padding-top: 3rem;
          padding-bottom: 4rem;
      }}

      /* Streamlit sets its own heading font with high specificity, so the
         title needs to be targeted through the app container to win. */
      .stApp h1, [data-testid="stHeading"] h1 {{
          font-family: Georgia, "Times New Roman", serif !important;
          font-weight: 700;
          letter-spacing: -0.01em;
          line-height: 1.15;
      }}

      /* Section headers. A rule underneath separates the parts without needing
         extra whitespace. */
      .part {{
          font-family: Georgia, "Times New Roman", serif;
          font-size: 1.45rem;
          font-weight: 700;
          color: {INK};
          margin-top: 2.6rem;
          margin-bottom: 1.1rem;
          padding-bottom: 0.45rem;
          border-bottom: 1px solid {RULE};
      }}

      .sub {{
          font-family: Georgia, "Times New Roman", serif;
          font-size: 1.12rem;
          font-weight: 700;
          margin-top: 1.8rem;
          margin-bottom: 0.6rem;
      }}

      .statlabel {{
          font-size: 0.82rem;
          color: {MUTED};
          margin-bottom: 0.1rem;
      }}
      .statvalue {{
          font-size: 2.3rem;
          font-weight: 700;
          line-height: 1.05;
      }}
      .statnote {{
          font-size: 0.82rem;
          color: {MUTED};
          margin-top: 0.15rem;
      }}

      .note {{
          font-size: 0.86rem;
          color: {MUTED};
          line-height: 1.5;
      }}

      /* The interval bar in Part I and Part III. A track with a zero line, a
         coloured span for the confidence interval, and a dot at the estimate. */
      .track {{
          position: relative;
          height: 26px;
          background: #f4f4f2;
          border-radius: 4px;
          margin: 0.5rem 0 0.3rem 0;
      }}
      .zero {{
          position: absolute;
          top: -3px;
          bottom: -3px;
          width: 2px;
          background: {MUTED};
      }}
      .span {{
          position: absolute;
          top: 9px;
          height: 8px;
          border-radius: 4px;
      }}
      .dot {{
          position: absolute;
          top: 5px;
          width: 16px;
          height: 16px;
          border-radius: 50%;
          border: 2px solid #ffffff;
          margin-left: -8px;
      }}
      .ends {{
          display: flex;
          justify-content: space-between;
          font-size: 0.75rem;
          color: {MUTED};
      }}

      @media (max-width: 640px) {{
        .block-container {{ padding-left: 1rem; padding-right: 1rem; }}
        h1 {{ font-size: 1.6rem !important; }}
        .part {{ font-size: 1.25rem; }}
        .statvalue {{ font-size: 1.8rem; }}
      }}

      div[data-testid="stDataFrame"] {{ overflow-x: auto; }}
    </style>
    """,
    unsafe_allow_html=True,
)


# ---------------------------------------------------------------------------
# Small rendering helpers
# ---------------------------------------------------------------------------

def part(title: str) -> None:
    st.markdown(f'<div class="part">{title}</div>', unsafe_allow_html=True)


def sub(title: str) -> None:
    st.markdown(f'<div class="sub">{title}</div>', unsafe_allow_html=True)


def note(text: str) -> None:
    st.markdown(f'<div class="note">{text}</div>', unsafe_allow_html=True)


def stat(label: str, value: str, colour: str = INK, footnote: str = "") -> None:
    """One large number with a label above and an optional line beneath."""
    st.markdown(
        f'<div class="statlabel">{label}</div>'
        f'<div class="statvalue" style="color:{colour}">{value}</div>'
        f'<div class="statnote">{footnote}</div>',
        unsafe_allow_html=True,
    )


def interval_bar(estimate: float, low: float, high: float, unit: str = "") -> None:
    """Draw a confidence interval against a zero line.

    The zero line is the whole point of the picture. An interval that covers it
    is an estimate consistent with no relationship, and that is much easier to
    see as a position than to work out from two signed numbers.
    """
    if any(pd.isna(v) for v in (estimate, low, high)):
        return

    # Pad the drawn range so the interval never runs to the very edge, and
    # always include zero so the reference line is on screen.
    lowest = min(low, 0.0)
    highest = max(high, 0.0)
    pad = max((highest - lowest) * 0.15, 1e-9)
    left_edge, right_edge = lowest - pad, highest + pad
    width = right_edge - left_edge

    def position(value: float) -> float:
        return 100.0 * (value - left_edge) / width

    crosses_zero = low <= 0 <= high
    colour = ORANGE if crosses_zero else BLUE

    st.markdown(
        f"""
        <div class="track">
          <div class="zero" style="left:{position(0.0):.2f}%"></div>
          <div class="span" style="left:{position(low):.2f}%;
               width:{position(high) - position(low):.2f}%; background:{colour}"></div>
          <div class="dot" style="left:{position(estimate):.2f}%; background:{colour}"></div>
        </div>
        <div class="ends"><span>{low:+.2f}{unit}</span>
        <span>95% confidence interval</span><span>{high:+.2f}{unit}</span></div>
        """,
        unsafe_allow_html=True,
    )


def show_image(site: str, filename: str, caption: str = "") -> None:
    """Display a chart, skipping one that has not been generated.

    The keyword that makes an image fill the column was renamed across
    Streamlit versions, so both spellings are tried. The deployed version is
    whatever the host installs.
    """
    path = outputs_dir(site) / filename
    if not path.exists():
        return
    try:
        st.image(str(path), caption=caption, use_container_width=True)
    except TypeError:
        st.image(str(path), caption=caption, use_column_width=True)


def table(frame: pd.DataFrame) -> None:
    try:
        st.dataframe(frame, hide_index=True, use_container_width=True)
    except TypeError:
        st.dataframe(frame, use_container_width=True)


@st.cache_data
def load_data(site: str) -> pd.DataFrame | None:
    path = analysis_table_path(site)
    return pd.read_csv(path) if path.exists() else None


@st.cache_data
def load_table(site: str, name: str) -> pd.DataFrame | None:
    path = outputs_dir(site) / name
    return pd.read_csv(path) if path.exists() else None


# ---------------------------------------------------------------------------
# Header
# ---------------------------------------------------------------------------

st.title("School Start Times and Academic Performance")

st.markdown(
    """
    This is a school research project that asks whether public high schools
    with **later start times** show better academic results than schools that
    start earlier, once student poverty, school size, English learners and
    school selectivity are accounted for.

    The main sample is **425 New York City high schools**, where start times run
    from 7:15 AM to past 9:00 AM and every school sits inside one school system.
    A second sample of 38 Nevada high schools is kept because it shows why this
    question is hard to study at all.
    """
)

note(
    "Data: "
    '<a href="https://data.cityofnewyork.us/d/uq7m-95z8">NYC high school directory</a>, '
    '<a href="https://data.cityofnewyork.us/d/45j8-f6um">demographic snapshot</a> and '
    '<a href="https://data.cityofnewyork.us/d/mjm3-8dw8">graduation results</a>, all 2018-19. '
    'Nevada figures come from <a href="https://nevadareportcard.nv.gov/DI/">Nevada Report Card</a> '
    "and the federal Common Core of Data, 2022-23."
)

# ---------------------------------------------------------------------------
# Part I
# ---------------------------------------------------------------------------

part("Part I: Pick a comparison")

def requested_site(available: list[str]) -> int:
    """Which sample to show first, from the ?site= part of the address.

    This makes a link to one sample shareable, so a reader can be pointed
    straight at the Nevada comparison rather than being told to change a
    dropdown. Streamlit renamed the query parameter API, so both spellings are
    tried and an unknown value falls back to the first sample.
    """
    value = None
    try:
        value = st.query_params.get("site")
    except Exception:
        try:
            value = st.experimental_get_query_params().get("site", [None])[0]
        except Exception:
            value = None
    return available.index(value) if value in available else 0


site_options = list(CONFIG["sites"].keys())

left, right = st.columns(2)
with left:
    site = st.selectbox(
        "Sample",
        options=site_options,
        index=requested_site(site_options),
        format_func=lambda key: site_config(key)["label"],
    )
settings = site_config(site)

frame = load_data(site)
if frame is None:
    st.error(
        f"No analysis table found for {site}. Run "
        f"`python run_pipeline.py --site {site}` and reload this page."
    )
    st.stop()

outcomes = [settings["outcomes"]["primary"]] + settings["outcomes"]["secondary"]
with right:
    outcome = st.selectbox(
        "Outcome", options=outcomes, format_func=lambda key: OUTCOME_LABELS.get(key, key)
    )

with_start_time = int(frame["start_minutes"].notna().sum())
if with_start_time == 0:
    st.warning(
        "No start times have been collected for this sample yet, so nothing can "
        "be computed. The study deliberately does not invent start times."
    )
    st.stop()

# The two headline numbers: the estimate with no controls, and the estimate
# from the fullest model that could be fitted.
correlation = unadjusted_correlation(frame, outcome)
ladder = load_table(site, "model_ladder.csv")
full_row = None
if ladder is not None:
    rows = ladder[(ladder["outcome"] == outcome) & ladder["coefficient"].notna()]
    if not rows.empty:
        full_row = rows.sort_values("model_number").iloc[-1]

unadjusted_row = None
if ladder is not None:
    first = ladder[(ladder["outcome"] == outcome) & (ladder["model_number"] == 1)]
    if not first.empty and pd.notna(first.iloc[0]["coefficient"]):
        unadjusted_row = first.iloc[0]

label = OUTCOME_LABELS.get(outcome, outcome)
left, right = st.columns(2)
with left:
    stat(
        "No controls",
        f"{unadjusted_row['coefficient']:+.2f}" if unadjusted_row is not None else "n/a",
        ORANGE,
        "change per hour later start",
    )
with right:
    stat(
        "All controls",
        f"{full_row['coefficient']:+.2f}" if full_row is not None else "n/a",
        BLUE,
        "change per hour later start",
    )

if full_row is not None:
    interval_bar(full_row["coefficient"], full_row["ci_low"], full_row["ci_high"])
    covers_zero = full_row["ci_low"] <= 0 <= full_row["ci_high"]
    note(
        f"Both numbers are the change in {label.lower()} for a school that starts "
        f"one hour later. The fully adjusted model uses {int(full_row['n'])} schools. "
        + (
            "Its interval covers zero, so this data is consistent with no relationship."
            if covers_zero
            else "Its interval does not cover zero, which still is not evidence of cause."
        )
    )

sub("What happens as controls are added")
show_image(site, f"03_coefficient_plot_{outcome}.png")
note(
    "Each rung adds one block of controls. Watch whether the estimate moves "
    "toward zero, which would mean the raw number was mostly confounding."
)

with st.expander("See the model ladder as a table"):
    if ladder is not None:
        rows = ladder[ladder["outcome"] == outcome]
        table(
            rows[["model_number", "specification", "coefficient", "ci_low", "ci_high", "n", "r_squared"]]
            .rename(
                columns={
                    "model_number": "Model", "specification": "Controls added",
                    "coefficient": "Per hour later", "ci_low": "CI low",
                    "ci_high": "CI high", "n": "Schools", "r_squared": "R squared",
                }
            )
            .round(3)
        )
        for message in rows["note"].dropna().astype(str).unique():
            if message.strip():
                st.caption(message)

# ---------------------------------------------------------------------------
# Part II
# ---------------------------------------------------------------------------

part("Part II: How much do start times actually vary?")

st.markdown(
    "A study of whether start time predicts performance needs schools that "
    "start at different times. This is the first thing worth checking, and it "
    "is where the Nevada sample falls down."
)

times = frame["start_minutes"].dropna()
columns = st.columns(4)
with columns[0]:
    stat("Schools", f"{with_start_time}")
with columns[1]:
    stat("Earliest", minutes_to_clock(times.min()))
with columns[2]:
    stat("Median", minutes_to_clock(times.median()))
with columns[3]:
    stat("Latest", minutes_to_clock(times.max()))

show_image(site, "01_start_time_distribution.png")

most_common = times.mode().iloc[0]
share = int((times == most_common).sum())
if share / len(times) > 0.5:
    st.warning(
        f"{share} of {len(times)} schools start at exactly "
        f"{minutes_to_clock(most_common)}. There is very little variation here, "
        "so the estimates rest on a handful of schools and should not be read "
        "as an answer to the research question."
    )
else:
    note(
        f"The middle half of schools start between "
        f"{minutes_to_clock(times.quantile(0.25))} and "
        f"{minutes_to_clock(times.quantile(0.75))}, and the standard deviation "
        f"is {times.std():.0f} minutes. That is enough spread to measure against."
    )

sub("The raw picture, with no controls")
show_image(site, f"02_unadjusted_scatter_{outcome}.png")
note(
    f"Unadjusted correlation r = {correlation['r']:.3f} "
    f"(95% CI {correlation['ci_low']:.3f} to {correlation['ci_high']:.3f}), "
    f"n = {correlation['n']} schools. This has no controls in it, and start "
    "times are not handed out at random, so it should not be read as an effect."
)

# ---------------------------------------------------------------------------
# Part III
# ---------------------------------------------------------------------------

part("Part III: Build the model yourself")

st.markdown(
    "Switch a control on and the regression is refitted in front of you. This "
    "is the part of the study worth playing with: it shows which variable is "
    "doing the work behind the raw number."
)

available_controls: list[str] = []
for block in settings["model_blocks"]:
    for control in block["controls"]:
        if control not in available_controls:
            available_controls.append(control)

chosen = [
    control
    for control in available_controls
    if st.checkbox(variable_label(control, site), value=False, key=f"{site}_{control}")
]

result = fit_one_model(frame, outcome, chosen, site)
unidentified = result.pop("unidentified", None) if result else None

if unidentified:
    st.warning(
        f"This control cannot be estimated. Every school in the sample with a "
        f"known start time has the same {unidentified}, so there is nothing for "
        "it to compare. Switch it off."
    )
elif result is None:
    st.warning(
        "This combination cannot be estimated with the number of schools "
        "available. Switch some controls off."
    )
else:
    left, right = st.columns(2)
    with left:
        stat(
            "Estimate",
            f"{result['coefficient']:+.2f}",
            ORANGE if result["ci_low"] <= 0 <= result["ci_high"] else BLUE,
            f"change in {label.lower()} per hour later",
        )
    with right:
        stat("Schools", f"{result['n']}", INK, f"R squared {result['r_squared']:.3f}")

    interval_bar(result["coefficient"], result["ci_low"], result["ci_high"])

    if result["ci_low"] <= 0 <= result["ci_high"]:
        st.info(
            "The confidence interval includes zero. With this sample, the data "
            "is consistent with there being no relationship between start time "
            "and this outcome once these controls are in place."
        )
    else:
        direction = "higher" if result["coefficient"] > 0 else "lower"
        st.success(
            f"The confidence interval excludes zero. Later starting schools "
            f"score {direction} on this outcome, with these controls. This is "
            "still not evidence of cause."
        )

sub("A cruder check that assumes less")
st.markdown(
    "Regression assumes the relationship is a straight line. Splitting schools "
    "into poverty bands and comparing early against late inside each band does "
    "not. If the two methods disagree, that matters more than either answer."
)
show_image(site, f"04_stratified_{outcome}.png")

with st.expander("See the band by band numbers"):
    strata = stratified_comparison(frame, outcome, site)
    table(
        strata[
            ["poverty_band", "n_early", "n_late", "mean_early", "mean_late",
             "difference_late_minus_early", "ci_low", "ci_high"]
        ]
        .rename(
            columns={
                "poverty_band": "Poverty band", "n_early": "Early", "n_late": "Late",
                "mean_early": "Mean, early", "mean_late": "Mean, late",
                "difference_late_minus_early": "Late minus early",
                "ci_low": "CI low", "ci_high": "CI high",
            }
        )
        .round(2)
    )

# ---------------------------------------------------------------------------
# Part IV
# ---------------------------------------------------------------------------

part("Part IV: Every outcome on one scale")

st.markdown(
    "The outcomes are measured on different scales, so their raw coefficients "
    "cannot be compared. Dividing each estimate by the standard deviation of "
    "its own outcome fixes that, and the result reads as how many standard "
    "deviations a school moves per hour later start."
)

show_image(site, "05_outcome_summary.png")

if ladder is not None:
    finals = (
        ladder[ladder["coefficient"].notna()]
        .sort_values("model_number")
        .groupby("outcome", as_index=False)
        .tail(1)
    )
    table(
        finals[["outcome_label", "coefficient", "ci_low", "ci_high", "n"]]
        .rename(
            columns={
                "outcome_label": "Outcome", "coefficient": "Per hour later",
                "ci_low": "CI low", "ci_high": "CI high", "n": "Schools",
            }
        )
        .round(2)
    )

if site == "nyc":
    note(
        "Two estimates survive every control and they disagree. Later starting "
        "schools have lower Advanced Regents diploma rates and higher college "
        "and career readiness. If a later bell simply helped students, both "
        "would move the same way. A split like that is what residual "
        "confounding looks like."
    )

selective = load_table(site, "selective_school_sensitivity.csv")
if selective is not None:
    with st.expander("Robustness: drop the exam and audition schools"):
        st.markdown(
            "The eight specialized high schools admit on a citywide exam. They "
            "start earlier than average and about 90 percent of their students "
            "earn an Advanced Regents diploma, against roughly 14 percent "
            "citywide. Eight schools should not set the slope for four hundred, "
            "so here are the same models without them."
        )
        rows = selective[selective["outcome"] == outcome]
        table(
            rows[["model_number", "coefficient", "ci_low", "ci_high", "n"]]
            .rename(
                columns={
                    "model_number": "Model", "coefficient": "Per hour later",
                    "ci_low": "CI low", "ci_high": "CI high", "n": "Schools",
                }
            )
            .round(3)
        )

censoring = load_table(site, "censoring_sensitivity.csv")
if censoring is not None:
    with st.expander("Robustness: drop the censored graduation rates"):
        st.markdown(
            "Nevada publishes \">95\" rather than an exact graduation rate for "
            "its highest performing schools, and the analysis replaces that "
            "with 97.5. These are the same models fitted only on schools with "
            "an exact published rate. The graduation rate result does not "
            "survive the check."
        )
        table(
            censoring[["model_number", "coefficient", "ci_low", "ci_high", "n"]]
            .rename(
                columns={
                    "model_number": "Model", "coefficient": "Per hour later",
                    "ci_low": "CI low", "ci_high": "CI high", "n": "Schools",
                }
            )
            .round(3)
        )

with st.expander("How the sample was built"):
    sample_steps = load_table(site, "sample_construction.csv")
    if sample_steps is not None:
        table(sample_steps)
    if site == "nyc":
        st.markdown(
            "Poverty is the Economic Need Index rather than free lunch "
            "eligibility, because New York City feeds every student and its "
            "reported free lunch rate says nothing about how poor a school is. "
            "Clark County does the same thing, which is why the Nevada sample "
            "needed a different measure too."
        )
    else:
        st.markdown(
            "Poverty is the direct certification rate: students confirmed low "
            "income through SNAP, TANF, or foster care records. Free and "
            "reduced lunch is unusable in Nevada because Clark County feeds "
            "every student, putting its reported rate at 100 percent "
            "everywhere."
        )

# ---------------------------------------------------------------------------
# Part V
# ---------------------------------------------------------------------------

part("Part V: What this study cannot tell you")

st.markdown(
    f"""
    **It cannot show that start times cause anything.** Schools were not
    assigned their start times at random. Controlling for the characteristics
    measured here removes the part of the overlap that was measured and does
    nothing about the part that was not. Showing cause would need random
    assignment, which nobody will do to a school district, or a study following
    the same schools through an actual bell schedule change.

    **The data is cross sectional.** Every school is observed once, in
    {settings['year_label']}. There is no before and after. A school that starts
    late and scores well might have scored just as well when it started early.

    **{with_start_time} schools have a known start time**, out of {len(frame)} in
    the sample.
    """
)

if site == "nyc":
    st.markdown(
        """
        **New York City is one city.** Holding the district constant removes a
        pile of confounders and also means the result describes New York City
        rather than American high schools. A city where most students take
        public transport to a school they applied to is not typical.

        **Who starts late is not random.** Specialized exam schools start
        earliest and small newer schools start latest. A school choosing a later
        bell time is making a decision that probably correlates with things
        nobody measured.

        **Start times are self reported free text.** The directory field is
        typed by school staff and contains entries such as "8am or 8:45am". One
        school was listed at "8:20pm" and was treated as a typing error. A
        single number also cannot represent a school running different schedules
        on different days.
        """
    )
else:
    st.markdown(
        """
        **There is almost no start time variation to work with.** Clark County
        sets the same 7:00 AM bell time at nearly all of its comprehensive high
        schools, and district fixed effects cannot be estimated at all because
        every school with a start time is in one district. This is a limitation
        of the data, not a finding about sleep.

        **Start times are imperfectly measured.** They come from school websites
        and district documents, which are not always current and do not always
        distinguish a first bell from a warning bell. Measurement error in a
        predictor generally pushes an estimated relationship toward zero.
        """
    )

st.markdown(
    """
    **The sleep literature is background, not support.** Adolescent circadian
    rhythms shift later during puberty, the American Academy of Pediatrics
    recommended in 2014 that high schools start no earlier than 8:30 AM, and
    districts that actually moved their bell schedules later reported longer
    sleep and better grades. Those studies used stronger designs than this one.
    This study cannot borrow their credibility.

    **A null result is a result.** Where the intervals include zero, the correct
    conclusion is that this data does not show a relationship, and that is what
    is reported rather than something more flattering.
    """
)

st.divider()
note(
    "Built as a school data analysis project. Estimates describe associations "
    "between schools at one point in time and cannot establish cause. Source "
    "code and full method: "
    '<a href="https://github.com/sahishnu-m/school-start-times">github.com/sahishnu-m/school-start-times</a>.'
)
