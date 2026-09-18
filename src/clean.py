"""Turns the raw downloads into tidy, correctly typed analysis variables.

Nothing in this module goes to the internet. It takes what the fetch modules
saved and produces analysis variables with clear names and real number types.
Keeping this separate from fetching means the cleaning rules can be changed and
rerun in seconds without touching a server.

The other job this module does is make the two sites look the same. New York
City and Nevada publish completely different files, with different column
names, different suppression markers, and different poverty measures. Both are
converted here into one shared set of columns:

    start_minutes    when the school day begins, minutes after midnight
    poverty_pct      the site's best available measure of student poverty
    enrollment       students enrolled
    el_pct           percentage of students who are English learners
    <outcome>        whatever outcomes that site publishes

Everything downstream, the merging, the models, the charts, and the dashboard,
is written once against those names and runs on either site.
"""

from __future__ import annotations

import re

import numpy as np
import pandas as pd

from .config import INTERIM_DIR, site_config

# A school day starts in the morning. Anything outside this window is a typing
# mistake in the source rather than a real bell time. The New York City
# directory is free text typed by school staff, and it contains one school
# listed at "8:20pm". Rather than let a 12 hour error sit in the data and drag
# a regression line with it, values outside the window are set to missing and
# counted, so the count can be reported.
EARLIEST_PLAUSIBLE_START = 6 * 60          # 6:00 AM
LATEST_PLAUSIBLE_START = 10 * 60 + 30      # 10:30 AM

# Values a state or city publishes in place of a number when the underlying
# count is too small to report without risking the identification of a student.
MISSING_MARKERS = {
    "N/A", "-", "", "*", "**", "n/a", "NA", "None", "nan", "s", "S", "na",
}


def parse_number(value) -> float:
    """Convert a published cell to a number.

    Education agencies publish three kinds of non-number, and they mean
    different things:

      "N/A", "-", "s"   the value was not reported, which is missing data
      ">95"             the true value is somewhere above 95
      "<5"              the true value is somewhere below 5

    The last two are censored, not missing. Throwing them away would bias the
    sample, because ">95" appears almost only at high performing schools and
    "<5" almost only at struggling ones. Dropping both ends would cut the top
    and the bottom off the outcome and flatten any real relationship.

    They are converted to the midpoint of the range they imply, so ">95"
    becomes 97.5 and "<5" becomes 2.5. A companion flag records that the value
    was censored, and the analysis reports a sensitivity check that drops those
    rows so a reader can see whether the midpoint choice changed anything.
    """
    if value is None:
        return np.nan
    text = str(value).strip().replace("%", "").replace(",", "")
    if text in MISSING_MARKERS:
        return np.nan

    greater = re.match(r"^>\s*(\d+(?:\.\d+)?)$", text)
    if greater:
        return (float(greater.group(1)) + 100.0) / 2.0

    less = re.match(r"^<\s*(\d+(?:\.\d+)?)$", text)
    if less:
        return float(less.group(1)) / 2.0

    try:
        return float(text)
    except ValueError:
        return np.nan


def is_censored(value) -> bool:
    """True when a published value was a bound such as >95 rather than a number."""
    text = str(value).strip()
    return text.startswith(">") or text.startswith("<")


def parse_clock_to_minutes(value) -> float:
    """Convert a written time to minutes after midnight.

    This has to cope with everything the two sources write. Nevada and the
    manual file use "7:45 AM". The New York City directory is free text typed
    by school staff and contains "8am", "8:15am", "8;00am", "8:35m", and
    "8am or 8:45am". Minutes after midnight is used throughout because it is a
    plain number that can go straight into a regression, where a coefficient on
    it reads as "per minute".
    """
    if value is None:
        return np.nan
    text = str(value).strip().lower()
    if text in MISSING_MARKERS:
        return np.nan

    # Normalise the typing mistakes seen in the NYC directory: a semicolon for
    # a colon, and a stray "m" where "am" was meant.
    text = text.replace(";", ":").replace(".", "")
    # "8am or 8:45am" gives two times. Take the first, which is the regular
    # start, and let the later one go. Two start times in one cell cannot be
    # represented by one number.
    text = re.split(r"\bor\b|/", text)[0].strip()

    match = re.match(r"^(\d{1,2})(?::(\d{2}))?\s*([ap])?\.?\s*m?\.?$", text)
    if not match:
        return np.nan

    hour = int(match.group(1))
    minute = int(match.group(2) or 0)
    meridiem = match.group(3)

    if meridiem == "p" and hour != 12:
        hour += 12
    elif meridiem == "a" and hour == 12:
        hour = 0
    elif meridiem is None and 1 <= hour <= 5:
        # A bare "1:30" on a school schedule is the afternoon.
        hour += 12

    if not (0 <= hour <= 23 and 0 <= minute <= 59):
        return np.nan
    return float(hour * 60 + minute)


def minutes_to_clock(minutes) -> str:
    """Format minutes after midnight as a readable clock time.

    Used for axis labels and anything a person reads. The analysis keeps
    working in plain minutes and only converts at the moment of display.
    """
    if minutes is None or pd.isna(minutes):
        return ""
    total = int(round(float(minutes)))
    hour, minute = divmod(total, 60)
    suffix = "AM" if hour < 12 else "PM"
    display_hour = hour % 12 or 12
    return f"{display_hour}:{minute:02d} {suffix}"


# ---------------------------------------------------------------------------
# New York City
# ---------------------------------------------------------------------------

# The admissions method a school used, collapsed into a few groups. The raw
# field has many long labels, and several of them describe the same kind of
# selection. What matters for this study is only how strongly a school screens
# its applicants, because that determines how much of its results reflect who
# was admitted rather than what the school did.
ADMISSIONS_GROUPS = [
    (r"specialized|test", "Specialized or test"),
    (r"audition|screened.*talent", "Audition"),
    (r"screened", "Screened"),
    (r"ed\.?\s*opt|educational option", "Educational option"),
    (r"limited unscreened", "Limited unscreened"),
    (r"unscreened|open", "Unscreened"),
    (r"zoned", "Zoned"),
]


def group_admissions_method(value) -> str:
    """Reduce a long admissions method label to one of a few groups."""
    text = str(value).strip().lower()
    if text in MISSING_MARKERS:
        return "Unknown"
    for pattern, label in ADMISSIONS_GROUPS:
        if re.search(pattern, text):
            return label
    return "Other"


# Borough letter in the DBN. The middle character of a DBN such as "16K498"
# names the borough, which is the geographic grouping used for fixed effects.
BOROUGH_LETTERS = {
    "M": "Manhattan",
    "X": "Bronx",
    "K": "Brooklyn",
    "Q": "Queens",
    "R": "Staten Island",
}


def clean_nyc(frame: pd.DataFrame) -> pd.DataFrame:
    """Rename and type the New York City fields the study uses."""
    out = pd.DataFrame()
    out["school_id"] = frame["dbn"].astype(str).str.strip()
    out["school_name"] = frame["school_name"]

    # The borough letter sits in the third character of the DBN.
    out["borough"] = out["school_id"].str[2].map(BOROUGH_LETTERS).fillna("Unknown")
    # Every school is in one school system, so the district column that the
    # shared code expects is filled with a constant. The models use borough.
    out["district"] = "New York City"

    out["start_minutes"] = frame["start_time"].map(parse_clock_to_minutes)
    out["start_time_raw"] = frame["start_time"]

    # NYC co-locates several schools in one building, and schools in the same
    # building share a neighbourhood, a facility, and often a catchment. Their
    # results are therefore not independent of each other, so the building code
    # is carried through and used to cluster the standard errors.
    out["building_code"] = frame["building_code"].astype(str).str.strip()

    # Outcomes. The directory reports attendance and college readiness as
    # proportions, and the graduation file reports percentages, so both are put
    # on a 0 to 100 scale here.
    out["grad_rate"] = frame["grads_1"].map(parse_number)
    out["advanced_regents_pct"] = frame["advanced_regents_of_cohort"].map(parse_number)
    out["dropout_pct"] = frame["dropout_1"].map(parse_number)
    out["attendance_rate"] = frame["attendance_rate"].map(parse_number) * 100.0
    out["college_career_rate"] = frame["college_career_rate"].map(parse_number) * 100.0

    out["cohort_size"] = frame["total_cohort"].map(parse_number)

    # Student characteristics. The Economic Need Index arrives as a proportion.
    out["poverty_pct"] = frame["economic_need_index"].map(parse_number) * 100.0
    out["el_pct"] = frame["english_language_learners_1"].map(parse_number) * 100.0
    out["swd_pct"] = frame["students_with_disabilities_1"].map(parse_number) * 100.0
    out["enrollment"] = frame["total_enrollment"].map(parse_number)

    # Selection controls, which matter in a school system where students apply
    # to high schools rather than being assigned one.
    out["admissions_method"] = frame["method1"].map(group_admissions_method)
    out["applicants_per_seat"] = frame["grade9geapplicantsperseat1"].map(parse_number)

    # No outcome here is published as a bound, so nothing is censored. The
    # column is still created so that the shared code can rely on it existing.
    out["outcome_censored"] = False

    return out


# ---------------------------------------------------------------------------
# Nevada
# ---------------------------------------------------------------------------

def clean_nevada_report_card(frame: pd.DataFrame) -> pd.DataFrame:
    """Rename and type the Nevada Report Card fields the study uses."""
    out = pd.DataFrame()
    out["nrc_school_code"] = frame["nrc_school_code"].astype(str).str.strip()
    out["school_name"] = frame["longSchoolName"].fillna(frame["nrc_school_name"])
    out["district"] = frame["district"]
    out["city"] = frame.get("city")
    out["school_type"] = frame.get("schTyp")
    out["grade_levels"] = frame.get("gradeLevels")

    out["act_composite"] = frame["act"].map(parse_number)
    out["grad_rate"] = frame["gradrate"].map(parse_number)
    out["outcome_censored"] = frame["gradrate"].map(is_censored)
    out["ela_proficient_pct"] = frame["reading_high"].map(parse_number)
    out["math_proficient_pct"] = frame["math_high"].map(parse_number)
    out["chronic_absent_pct"] = frame["chronicAbst"].map(parse_number)

    out["enrollment_nrc"] = frame["enrollment"].map(parse_number)
    out["el_pct"] = frame["sP_PctEnrLEP"].map(parse_number)
    out["iep_pct"] = frame["sP_PctEnrIEP"].map(parse_number)
    out["frl_pct_reported"] = frame["sP_PctEnrFRL"].map(parse_number)

    return out


def clean_ccd(frame: pd.DataFrame) -> pd.DataFrame:
    """Rename and type the federal directory fields the Nevada site uses."""
    out = pd.DataFrame()
    out["nrc_school_code"] = frame["nrc_school_code"].astype(str).str.strip()
    out["ncessch"] = frame["ncessch"].astype(str)
    out["lea_name"] = frame["lea_name"]
    out["locale_group"] = frame["locale_group"]
    out["charter"] = pd.to_numeric(frame["charter"], errors="coerce")
    out["enrollment_ccd"] = pd.to_numeric(frame["enrollment"], errors="coerce")
    out["latitude"] = pd.to_numeric(frame["latitude"], errors="coerce")
    out["longitude"] = pd.to_numeric(frame["longitude"], errors="coerce")

    frl_count = pd.to_numeric(frame["free_or_reduced_price_lunch"], errors="coerce")
    direct_cert_count = pd.to_numeric(frame["direct_certification"], errors="coerce")

    # Counts below zero are the NCES missing data codes such as -1 and -2.
    frl_count = frl_count.where(frl_count >= 0)
    direct_cert_count = direct_cert_count.where(direct_cert_count >= 0)

    out["frl_pct_ccd"] = 100.0 * frl_count / out["enrollment_ccd"]
    out["direct_cert_pct"] = 100.0 * direct_cert_count / out["enrollment_ccd"]

    return out


# ---------------------------------------------------------------------------
# Shared
# ---------------------------------------------------------------------------

def build_analysis_variables(frame: pd.DataFrame, site: str | None = None) -> pd.DataFrame:
    """Create the derived variables the models use, for either site."""
    settings = site_config(site)
    out = frame.copy()

    # Enrollment in thousands. The raw number runs from about 100 to 4,500, so
    # a coefficient per student would be a number with four leading zeros. Per
    # thousand students reads as a sentence a person can say out loud.
    out["enrollment_1000"] = out["enrollment"] / 1000.0

    # Start time expressed two ways. Minutes is the underlying measure. Hours
    # makes the reported coefficient mean "per hour later", which is the unit
    # the sleep research literature uses and the one a reader can picture.
    # Drop start times that cannot be real before anything is derived from them.
    implausible = out["start_minutes"].notna() & (
        (out["start_minutes"] < EARLIEST_PLAUSIBLE_START)
        | (out["start_minutes"] > LATEST_PLAUSIBLE_START)
    )
    if implausible.any():
        print(f"  set {int(implausible.sum())} implausible start time(s) to missing "
              f"(outside {minutes_to_clock(EARLIEST_PLAUSIBLE_START)} to "
              f"{minutes_to_clock(LATEST_PLAUSIBLE_START)})")
        out.loc[implausible, "start_minutes"] = np.nan

    out["start_hours"] = out["start_minutes"] / 60.0

    cutoff = settings["early_late_cutoff_minutes"]
    out["start_group"] = pd.Series(pd.NA, index=out.index, dtype="object")
    known = out["start_minutes"].notna()
    out.loc[known & (out["start_minutes"] < cutoff), "start_group"] = "Early"
    out.loc[known & (out["start_minutes"] >= cutoff), "start_group"] = "Late"

    strat = settings["stratification"]
    out["poverty_band"] = pd.cut(
        out["poverty_pct"],
        bins=strat["poverty_band_edges"],
        labels=strat["poverty_band_labels"],
        include_lowest=True,
    )

    return out


def run_nevada() -> pd.DataFrame:
    """Clean both Nevada sources and write one tidy table per source."""
    print("Cleaning Nevada data")

    nrc_raw = pd.read_csv(
        INTERIM_DIR / "nevada_report_card.csv", dtype={"nrc_school_code": str}
    )
    ccd_raw = pd.read_csv(
        INTERIM_DIR / "ccd_directory.csv", dtype={"nrc_school_code": str}
    )

    nrc = clean_nevada_report_card(nrc_raw)
    ccd = clean_ccd(ccd_raw)

    nrc.to_csv(INTERIM_DIR / "clean_nevada_report_card.csv", index=False)
    ccd.to_csv(INTERIM_DIR / "clean_ccd.csv", index=False)
    print(f"  cleaned {len(nrc)} Nevada Report Card rows and {len(ccd)} CCD rows")
    return nrc


def run_nyc() -> pd.DataFrame:
    """Clean the New York City data and write one tidy table."""
    print("Cleaning New York City data")
    raw = pd.read_csv(INTERIM_DIR / "nyc_raw.csv", dtype={"dbn": str}, low_memory=False)
    cleaned = clean_nyc(raw)
    cleaned.to_csv(INTERIM_DIR / "clean_nyc.csv", index=False)
    print(f"  cleaned {len(cleaned)} New York City schools")
    return cleaned
