"""Turns the raw downloads into tidy, correctly typed variables.

Nothing in this module goes to the internet. It takes what the fetch modules
saved and produces analysis variables with clear names and real number types.
Keeping this separate from fetching means the cleaning rules can be changed and
rerun in seconds without touching a server.

Two decisions in here carry more weight than the rest, so they are explained
where they happen rather than buried in the README: how suppressed values are
handled, and which variable is used to measure student poverty.
"""

from __future__ import annotations

import re

import numpy as np
import pandas as pd

from .config import CONFIG, INTERIM_DIR

# Values Nevada publishes in place of a number when the underlying count is too
# small to report without risking the identification of a student.
MISSING_MARKERS = {"N/A", "-", "", "*", "**", "n/a", "NA", "None", "nan"}


def parse_number(value) -> float:
    """Convert a Nevada Report Card cell to a number.

    The site publishes three kinds of non-numbers, and they mean different
    things:

      "N/A" or "-"   the value was not reported at all, which is missing data
      ">95"          the true value is somewhere above 95
      "<5"           the true value is somewhere below 5

    The last two are censored, not missing. Throwing them away would bias the
    sample, because ">95" appears almost only at high performing schools and
    "<5" almost only at struggling ones. Dropping both ends would cut the top
    and bottom off the outcome and flatten any real relationship.

    They are converted to the midpoint of the range they imply, so ">95" becomes
    97.5 and "<5" becomes 2.5. A companion flag records that the value was
    censored, and the analysis reports a sensitivity check that drops those
    rows so a reader can see whether the midpoint choice changed anything.
    """
    if value is None:
        return np.nan
    text = str(value).strip().replace("%", "").replace(",", "")
    if text in MISSING_MARKERS:
        return np.nan

    greater = re.match(r"^>\s*(\d+(?:\.\d+)?)$", text)
    if greater:
        bound = float(greater.group(1))
        return (bound + 100.0) / 2.0

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
    """Convert a written time such as "7:45 AM" to minutes after midnight.

    Minutes after midnight is used throughout instead of a time object because
    it is a plain number, which means it can go straight into a regression and
    a coefficient on it reads as "per minute". Converting to hours would make
    the coefficient larger and easier to read, and the analysis module does
    exactly that when it reports results.
    """
    if value is None:
        return np.nan
    text = str(value).strip().lower()
    if text in MISSING_MARKERS:
        return np.nan

    match = re.match(r"^(\d{1,2})\s*:\s*(\d{2})\s*([ap])?\.?\s*m?\.?$", text)
    if not match:
        return np.nan

    hour, minute, meridiem = int(match.group(1)), int(match.group(2)), match.group(3)
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

    Used for axis labels and for anything a person reads. The analysis keeps
    working in plain minutes, and only converts at the moment of display.
    """
    if minutes is None or pd.isna(minutes):
        return ""
    total = int(round(float(minutes)))
    hour, minute = divmod(total, 60)
    suffix = "AM" if hour < 12 else "PM"
    display_hour = hour % 12 or 12
    return f"{display_hour}:{minute:02d} {suffix}"


def clean_nevada_report_card(frame: pd.DataFrame) -> pd.DataFrame:
    """Rename and type the Nevada Report Card fields the study uses."""
    out = pd.DataFrame()
    out["nrc_school_code"] = frame["nrc_school_code"].astype(str).str.strip()
    out["school_name"] = frame["longSchoolName"].fillna(frame["nrc_school_name"])
    out["district"] = frame["district"]
    out["city"] = frame.get("city")
    out["school_type"] = frame.get("schTyp")
    out["grade_levels"] = frame.get("gradeLevels")

    # Outcomes.
    out["act_composite"] = frame["act"].map(parse_number)
    out["grad_rate"] = frame["gradrate"].map(parse_number)
    out["grad_rate_censored"] = frame["gradrate"].map(is_censored)
    out["ela_proficient_pct"] = frame["reading_high"].map(parse_number)
    out["math_proficient_pct"] = frame["math_high"].map(parse_number)
    out["chronic_absent_pct"] = frame["chronicAbst"].map(parse_number)

    # Student characteristics as Nevada publishes them.
    out["enrollment_nrc"] = frame["enrollment"].map(parse_number)
    out["el_pct"] = frame["sP_PctEnrLEP"].map(parse_number)
    out["iep_pct"] = frame["sP_PctEnrIEP"].map(parse_number)
    out["frl_pct_reported"] = frame["sP_PctEnrFRL"].map(parse_number)

    return out


def clean_ccd(frame: pd.DataFrame) -> pd.DataFrame:
    """Rename and type the federal directory fields the study uses."""
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


def build_analysis_variables(frame: pd.DataFrame) -> pd.DataFrame:
    """Create the derived variables the models use."""
    out = frame.copy()

    # Enrollment in thousands. The raw number runs from roughly 100 to 3500, so
    # a coefficient per student would be a number with four leading zeros. Per
    # thousand students reads as a sentence a person can say out loud.
    out["enrollment"] = out["enrollment_ccd"].fillna(out["enrollment_nrc"])
    out["enrollment_1000"] = out["enrollment"] / 1000.0

    # Start time expressed two ways. Minutes is what the regression uses.
    # Hours makes the reported coefficient mean "per hour later", which is the
    # unit the sleep research literature uses and the one a reader can picture.
    out["start_hours"] = out["start_minutes"] / 60.0

    # Early or late, with missing start times left missing. The grouping is
    # built with pandas rather than np.where because mixing a float NaN and
    # text labels in one numpy array has no common type.
    cutoff = CONFIG["stratification"]["early_late_cutoff_minutes"]
    out["start_group"] = pd.Series(pd.NA, index=out.index, dtype="object")
    known = out["start_minutes"].notna()
    out.loc[known & (out["start_minutes"] < cutoff), "start_group"] = "Early"
    out.loc[known & (out["start_minutes"] >= cutoff), "start_group"] = "Late"

    # Poverty bands for the stratified comparison.
    edges = CONFIG["stratification"]["poverty_band_edges"]
    labels = CONFIG["stratification"]["poverty_band_labels"]
    out["poverty_band"] = pd.cut(
        out["direct_cert_pct"], bins=edges, labels=labels, include_lowest=True
    )

    return out


def apply_sample_filters(frame: pd.DataFrame) -> tuple[pd.DataFrame, list[dict]]:
    """Keep the schools the study is about, and record every exclusion.

    The counts are returned alongside the filtered table so that the README and
    the dashboard can both show how the sample was built. A sample that shrinks
    without explanation is the easiest way for a study to mislead, including by
    accident.
    """
    sample_config = CONFIG["sample"]
    steps: list[dict] = []
    out = frame.copy()
    steps.append({"step": "High schools in the federal directory", "schools": len(out)})

    excluded_types = set(sample_config["excluded_school_types"])
    out = out[~out["school_type"].isin(excluded_types)]
    steps.append({"step": "After removing alternative and special education schools", "schools": len(out)})

    out = out[out["enrollment"] >= sample_config["min_enrollment"]]
    steps.append(
        {"step": f"After removing schools under {sample_config['min_enrollment']} students",
         "schools": len(out)}
    )

    return out.reset_index(drop=True), steps


def run() -> pd.DataFrame:
    """Clean both sources and write one tidy table per source."""
    print("Cleaning downloaded data")

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


if __name__ == "__main__":
    run()
