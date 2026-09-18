"""Downloads New York City high school data, including start times.

This is the module that makes the study answerable, and it is worth explaining
why it exists at all.

The Nevada analysis ran into a wall that no amount of statistics could get
around. Clark County sets one bell time for almost all of its high schools, so
29 of the 38 Nevada schools with a known start time begin at exactly 7:00 AM.
A study of whether start time predicts performance needs schools that start at
different times, and Nevada does not supply them.

New York City does. NYC publishes a start time for every high school in its
public high school directory, and those times run from about 7:15 AM to past
9:00 AM. That gives roughly 400 schools with real variation, and all of them
sit inside one school system.

Being inside one school system is the important part. In the Nevada design,
anything that differs between districts, such as busing budgets, union
contracts, state funding, or the local labour market, was tangled up with start
time. Comparing NYC high schools to each other holds all of that constant by
construction. It is the closest thing to the comparison the original study
wanted and could not make.

Four datasets are used, all from NYC Open Data, all joined on the DBN, which is
the district-borough-number code that identifies a NYC school:

  uq7m-95z8   2019 DOE High School Directory
              start_time, attendance_rate, college_career_rate, admissions
              method, and applicants per seat

  45j8-f6um   2018-19 School Demographic Snapshot
              economic need index, poverty, English learners, students with
              disabilities, enrollment

  mjm3-8dw8   Graduation results for cohorts 2012 to 2019
              four year graduation rate and advanced Regents diploma rate for
              the cohort that entered in 2015 and was due to graduate in 2019

The year is 2018-2019 throughout. The directory published for 2019 admissions
describes schools as they operated in 2018-19, the demographic snapshot is the
2018-19 file, and the graduation cohort is the one that finished in June 2019.
"""

from __future__ import annotations

import json

import pandas as pd
import requests

from .config import INTERIM_DIR, RAW_DIR
from .polite import USER_AGENT, polite_json_get

SOCRATA_ROOT = "https://data.cityofnewyork.us/resource/"

HIGH_SCHOOL_DIRECTORY = "uq7m-95z8"
DEMOGRAPHIC_SNAPSHOT = "45j8-f6um"
GRADUATION_RESULTS = "mjm3-8dw8"

# The school year this study covers, written the way each dataset writes it.
DEMOGRAPHIC_YEAR = "2018-19"

# The graduating class of 2019 is the cohort that started grade 9 in 2015.
# "4 year June" counts students who finished in four years by June, which is
# the standard on-time graduation measure and the one NYC reports publicly.
COHORT_YEAR = "2015"
COHORT_TYPE = "4 year June"

# NYC suppresses a value when the group is small enough that a student could be
# identified, and writes "s" in the cell.
SUPPRESSED = {"s", "S", "", "na", "NA", "N/A"}


def _session() -> requests.Session:
    session = requests.Session()
    session.headers.update({"User-Agent": USER_AGENT, "Accept": "application/json"})
    return session


def _fetch_all(session, dataset: str, params: dict, clock: list[float]) -> list[dict]:
    """Page through a Socrata dataset until it stops returning rows.

    Socrata caps a single response, so anything larger than the cap has to be
    read in pages. The loop stops when a page comes back smaller than the page
    size, which means the end was reached.
    """
    page_size = 1000
    offset = 0
    rows: list[dict] = []

    while True:
        query = dict(params, **{"$limit": page_size, "$offset": offset})
        payload = polite_json_get(session, SOCRATA_ROOT + f"{dataset}.json", query, clock)
        if not payload:
            break
        rows.extend(payload)
        if len(payload) < page_size:
            break
        offset += page_size

    return rows


def _cached_fetch(session, dataset: str, params: dict, clock: list[float], name: str) -> pd.DataFrame:
    """Fetch a dataset once and keep the raw response on disk."""
    path = RAW_DIR / f"nyc_{name}.json"
    if path.exists():
        rows = json.loads(path.read_text(encoding="utf-8"))
        print(f"  reusing cached {name} ({len(rows)} rows)")
    else:
        rows = _fetch_all(session, dataset, params, clock)
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(rows), encoding="utf-8")
        print(f"  downloaded {name} ({len(rows)} rows)")
    return pd.DataFrame(rows)


def run() -> pd.DataFrame:
    """Fetch the three NYC datasets, join them, and write data/interim."""
    print("Fetching New York City high school data for 2018-2019")
    session = _session()
    clock = [0.0]

    directory = _cached_fetch(session, HIGH_SCHOOL_DIRECTORY, {}, clock, "hs_directory_2019")

    demographics = _cached_fetch(
        session,
        DEMOGRAPHIC_SNAPSHOT,
        {"$where": f"year='{DEMOGRAPHIC_YEAR}'"},
        clock,
        "demographics_2018_19",
    )

    graduation = _cached_fetch(
        session,
        GRADUATION_RESULTS,
        {
            "$where": (
                "report_category='School' AND category='All Students' "
                f"AND cohort_year='{COHORT_YEAR}' AND cohort='{COHORT_TYPE}'"
            )
        },
        clock,
        "graduation_cohort_2015",
    )

    # The graduation file names its school identifier geographic_subdivision,
    # which holds the DBN for rows at the School level.
    if not graduation.empty:
        graduation = graduation.rename(columns={"geographic_subdivision": "dbn"})

    merged = directory.merge(
        demographics.drop(columns=["school_name"], errors="ignore"),
        on="dbn",
        how="left",
        suffixes=("", "_demo"),
    )
    merged = merged.merge(
        graduation.drop(columns=["school_name"], errors="ignore"),
        on="dbn",
        how="left",
        suffixes=("", "_grad"),
    )

    print(f"  {len(directory)} schools in the directory")
    print(f"  {merged['economic_need_index'].notna().sum()} matched to a demographic record")
    print(f"  {merged['total_cohort'].notna().sum()} matched to a graduation cohort")

    out_path = INTERIM_DIR / "nyc_raw.csv"
    merged.to_csv(out_path, index=False)
    print(f"  wrote {out_path.name}")
    return merged


if __name__ == "__main__":
    run()
