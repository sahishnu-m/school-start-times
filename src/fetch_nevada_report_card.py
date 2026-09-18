"""Downloads school level data from Nevada Report Card.

Nevada Report Card (nevadareportcard.nv.gov) is the Nevada Department of
Education accountability site. The public site is a JavaScript application
that reads from a JSON API at /DIWAPI-NVReportCard/api/. This module uses two
endpoints from that API:

  OrganizationHierarchyTreeYear
      The list of districts and the schools inside each one, for a given year.
      This is how the study learns which schools exist and what their internal
      identifiers are.

  Dashboard
      One record per school per year. This is the record behind the summary
      page a parent sees when they look up a school. It carries enrollment,
      graduation rate, ACT scores, proficiency rates, chronic absenteeism, and
      the demographic percentages.

Using the same API the public site uses means the numbers here are the same
numbers a person would read off the website. Nothing is recomputed.

A note on year labels: Nevada Report Card labels a school year by the calendar
year it ends in. Passing year=2023 returns the 2022-2023 school year.
"""

from __future__ import annotations

import json

import pandas as pd
import requests

from .config import INTERIM_DIR, RAW_DIR, site_config
from .polite import USER_AGENT, polite_json_get

API_ROOT = "https://nevadareportcard.nv.gov/DIWAPI-NVReportCard/api/"

# Nevada suppresses small counts rather than publishing them, and writes the
# suppressed value as a string like ">95" or "<5". Those are real information
# about the direction of the value, but they are not numbers, so they are
# handled explicitly in clean.py rather than being coerced to NaN here.
SUPPRESSION_MARKERS = ("N/A", "-", "", "*", "**")


def _make_session() -> requests.Session:
    session = requests.Session()
    session.headers.update(
        {
            "User-Agent": USER_AGENT,
            # The API is the backend for the public site, so it expects
            # requests that look like they came from that site.
            "Referer": "https://nevadareportcard.nv.gov/DI/",
            "Accept": "application/json",
        }
    )
    return session


def fetch_school_list(year: int, session: requests.Session, clock: list[float]) -> pd.DataFrame:
    """Get every district and school Nevada Report Card knows about for a year.

    The API returns a nested tree: state, then districts, then schools. It is
    flattened here into one row per school, because a flat table is what every
    later step wants.
    """
    tree = polite_json_get(
        session, API_ROOT + "OrganizationHierarchyTreeYear",
        params={"year": str(year)}, last_call=clock,
    )
    if not tree:
        raise RuntimeError("Nevada Report Card returned no organization tree")

    (RAW_DIR / f"nrc_organizations_{year}.json").write_text(
        json.dumps(tree, indent=2), encoding="utf-8"
    )

    rows = []
    for state in tree:
        for district in state.get("districts", []):
            for school in district.get("schools", []):
                rows.append(
                    {
                        "nrc_org_id": school["id"],
                        # The state school code is the join key that connects
                        # Nevada Report Card to the federal CCD file. The first
                        # two digits are the district, the rest is the school.
                        "nrc_school_code": str(school["code"]),
                        "nrc_school_name": school["name"],
                        "district": district["name"],
                        "nrc_district_code": str(district["code"]),
                        "zip_code": str(school.get("ZipCode", "")).strip(),
                    }
                )
    return pd.DataFrame(rows)


def fetch_dashboard_records(
    org_ids: list[int], year: int, session: requests.Session, clock: list[float]
) -> pd.DataFrame:
    """Pull the Dashboard record for each school.

    This is one request per school, which is why the rate limiter matters. For
    a few hundred Nevada high schools at roughly two to three seconds each the
    run takes several minutes. Results are written to data/raw so that a rerun
    of a later pipeline step does not repeat the download.
    """
    raw_path = RAW_DIR / f"nrc_dashboard_{year}.json"

    # If the raw pull already exists, reuse it. Refetching identical data from
    # a state agency server on every run would be rude and slow.
    if raw_path.exists():
        records = json.loads(raw_path.read_text(encoding="utf-8"))
        print(f"  reusing cached Nevada Report Card pull ({len(records)} schools)")
        return pd.DataFrame(records)

    records: list[dict] = []
    total = len(org_ids)
    for index, org_id in enumerate(org_ids, start=1):
        payload = polite_json_get(
            session, API_ROOT + "Dashboard",
            params={"year": str(year), "orgId": str(org_id)}, last_call=clock,
        )
        if payload:
            # The endpoint returns a list with a single record per school.
            record = payload[0]
            record["nrc_org_id"] = org_id
            records.append(record)
        if index % 25 == 0 or index == total:
            print(f"  Nevada Report Card: {index}/{total} schools")

    raw_path.write_text(json.dumps(records, indent=2), encoding="utf-8")
    return pd.DataFrame(records)


def run(year: int | None = None, restrict_to_codes: set[str] | None = None) -> pd.DataFrame:
    """Fetch Nevada Report Card data and write it to data/interim.

    restrict_to_codes limits the download to a set of Nevada school codes. The
    pipeline passes in the high school codes taken from the federal CCD file.
    This matters because the Dashboard endpoint serves one school per request,
    and Nevada has around 770 public schools but only around 170 high schools.
    Restricting the pull cuts roughly 25 minutes of requests off the run and
    avoids asking a state agency server for several hundred records the study
    will never use.

    The output of this step is deliberately close to what the API returned.
    Renaming and type conversion happen in clean.py, so that if a number ever
    looks wrong it is possible to trace it back to the source without guessing
    what this module did to it.
    """
    year = year or site_config("nevada")["nrc_year"]
    session = _make_session()
    clock = [0.0]   # shared timestamp for rate limiting across all calls

    print(f"Fetching Nevada Report Card school list for {year - 1}-{year}")
    schools = fetch_school_list(year, session, clock)
    print(f"  found {len(schools)} schools in {schools['district'].nunique()} districts")

    if restrict_to_codes is not None:
        before = len(schools)
        schools = schools[schools["nrc_school_code"].isin(restrict_to_codes)]
        print(f"  narrowed to {len(schools)} high schools (from {before}) using the CCD frame")

    dashboard = fetch_dashboard_records(
        schools["nrc_org_id"].tolist(), year, session, clock
    )

    merged = schools.merge(dashboard, on="nrc_org_id", how="inner")
    merged["nrc_year"] = year

    out_path = INTERIM_DIR / "nevada_report_card.csv"
    merged.to_csv(out_path, index=False)
    print(f"  wrote {len(merged)} rows to {out_path.name}")
    return merged


if __name__ == "__main__":
    run()
