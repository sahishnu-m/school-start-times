"""Downloads the federal school directory for Nevada.

The source is the National Center for Education Statistics Common Core of Data
(CCD), which is the federal census of every public school in the country. It is
reached here through the Urban Institute Education Data API, which republishes
the NCES files as JSON. The Urban Institute mirror is used instead of the raw
NCES download page for two reasons: the NCES download server was not reachable
from this machine, and the API returns the whole state in a single request
instead of a zip archive that has to be unpacked and column-matched by hand.
The underlying numbers are the NCES numbers either way.

What this step provides that Nevada Report Card does not:

  urban_centric_locale
      The federal city, suburb, town, or rural classification. This is the
      locale control in the regression.

  direct_certification
      The count of students verified as low income through SNAP, TANF, or
      foster care records. This matters a great deal for this study, and the
      reason is explained under the poverty measure note in clean.py.

  highest_grade_offered, school_level, virtual, charter
      Used to define which schools are in the sample.

  seasch
      The Nevada state school code, formatted as district-school. Stripping the
      district prefix gives exactly the school code Nevada Report Card uses, so
      this is the join key between the federal file and the state file.

A note on year labels: CCD labels a school year by the calendar year it starts
in. Passing year=2022 returns the 2022-2023 school year, which is the same year
Nevada Report Card calls 2023.
"""

from __future__ import annotations

import json

import pandas as pd
import requests

from .config import CONFIG, INTERIM_DIR, RAW_DIR
from .polite import USER_AGENT, polite_json_get

API_ROOT = "https://educationdata.urban.org/api/v1/"

# FIPS 32 is Nevada. The API filters server side, so this keeps the response to
# one state instead of downloading the whole country and throwing most of it
# away.
NEVADA_FIPS = 32

# The CCD urban-centric locale code is a two digit number. The first digit is
# the broad category and the second is a size or distance subdivision. The
# subdivisions split Nevada schools into groups too small to estimate, so the
# study collapses them to the four broad categories.
LOCALE_GROUPS = {1: "City", 2: "Suburb", 3: "Town", 4: "Rural"}


def _make_session() -> requests.Session:
    session = requests.Session()
    session.headers.update({"User-Agent": USER_AGENT, "Accept": "application/json"})
    return session


def run(year: int | None = None) -> pd.DataFrame:
    """Fetch the Nevada CCD directory and write it to data/interim."""
    year = year or CONFIG["ccd_year"]
    raw_path = RAW_DIR / f"ccd_directory_{year}.json"

    if raw_path.exists():
        results = json.loads(raw_path.read_text(encoding="utf-8"))
        print(f"  reusing cached CCD pull ({len(results)} schools)")
    else:
        print(f"Fetching NCES Common Core of Data directory for {year}-{year + 1}")
        session = _make_session()
        payload = polite_json_get(
            session,
            API_ROOT + f"schools/ccd/directory/{year}/",
            params={"fips": NEVADA_FIPS},
            last_call=[0.0],
        )
        if not payload or "results" not in payload:
            raise RuntimeError("CCD directory request returned nothing usable")
        results = payload["results"]
        raw_path.write_text(json.dumps(results, indent=2), encoding="utf-8")
        print(f"  downloaded {len(results)} Nevada schools")

    frame = pd.DataFrame(results)

    # Build the join key to Nevada Report Card. seasch looks like "02-02612",
    # where "02" is the district and "02612" is the school code that Nevada
    # Report Card publishes. Splitting on the dash and keeping the second piece
    # gives an exact match, which is far more reliable than matching on school
    # names that are abbreviated differently in the two sources.
    frame["nrc_school_code"] = (
        frame["seasch"].astype(str).str.split("-").str[-1].str.strip()
    )

    # Collapse the two digit locale code to its broad category.
    frame["locale_group"] = (
        pd.to_numeric(frame["urban_centric_locale"], errors="coerce")
        .floordiv(10)
        .map(LOCALE_GROUPS)
    )

    out_path = INTERIM_DIR / "ccd_directory.csv"
    frame.to_csv(out_path, index=False)
    print(f"  wrote {len(frame)} rows to {out_path.name}")
    return frame


def high_school_frame(ccd: pd.DataFrame) -> pd.DataFrame:
    """Narrow the CCD directory to schools that are high schools.

    The sampling frame is defined from the federal file rather than from school
    names. A name based filter would miss schools called "Academy" or "Career
    and Technical" and would wrongly catch middle schools with "High" in an
    unusual name. The federal file records the highest grade a school offers,
    which is the thing the study actually cares about.
    """
    sample_config = CONFIG["sample"]
    frame = ccd.copy()

    if sample_config["require_grade_12"]:
        frame = frame[pd.to_numeric(frame["highest_grade_offered"], errors="coerce") == 12]

    if sample_config["exclude_virtual"]:
        frame = frame[pd.to_numeric(frame["virtual"], errors="coerce") != 1]

    # school_status 1 means open. Closed and merged schools still appear in the
    # directory and would otherwise contribute rows with no current data.
    frame = frame[pd.to_numeric(frame["school_status"], errors="coerce") == 1]

    return frame.reset_index(drop=True)


if __name__ == "__main__":
    directory = run()
    print(f"high school frame: {len(high_school_frame(directory))} schools")
