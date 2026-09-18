"""Builds the single analysis table each site's models read.

For New York City this is mostly a filter, because the fetch step already
joined the directory, the demographics, and the graduation cohort on the DBN.

For Nevada it is a real join. The key is the Nevada school code, which comes
from Nevada Report Card directly and from the federal directory by stripping
the district prefix off the state school code. Matching on a code rather than a
school name avoids a whole category of quiet errors, since the two sources
abbreviate names differently.

This module also settles the rule the study set for Nevada start times: when
the scraper, a district document, and the hand entered file disagree, the hand
entered value wins.
"""

from __future__ import annotations

import pandas as pd

from .clean import build_analysis_variables, parse_clock_to_minutes
from .config import (
    INTERIM_DIR,
    MANUAL_START_TIMES,
    analysis_table_path,
    outputs_dir,
    site_config,
)


# ---------------------------------------------------------------------------
# Nevada start times
# ---------------------------------------------------------------------------

def load_start_times() -> pd.DataFrame:
    """Combine every Nevada start time source into one column, in order of trust.

    Three sources feed this, and they are applied worst first so that better
    ones overwrite them:

      1. the page scraper, which guesses from free text on a web page
      2. a district published bell schedule document, which is the district
         stating its own times in one table
      3. the hand entered file, where a person read the schedule themselves
    """
    scraped = pd.DataFrame(columns=["nrc_school_code", "start_minutes", "url"])
    scraped_path = INTERIM_DIR / "start_times_scraped.csv"
    if scraped_path.exists():
        scraped = pd.read_csv(scraped_path, dtype={"nrc_school_code": str})

    documents = pd.DataFrame(columns=["nrc_school_code", "start_minutes"])
    documents_path = INTERIM_DIR / "start_times_documents.csv"
    if documents_path.exists():
        documents = pd.read_csv(documents_path, dtype={"nrc_school_code": str})

    manual = pd.DataFrame(columns=["nrc_school_code", "start_time", "source_url"])
    if MANUAL_START_TIMES.exists():
        manual = pd.read_csv(MANUAL_START_TIMES, dtype={"nrc_school_code": str})

    rows: dict[str, dict] = {}

    # Order is the whole mechanism here. Each block overwrites the one above it.
    for row in scraped.itertuples():
        code = str(row.nrc_school_code)
        rows[code] = {
            "nrc_school_code": code,
            "start_minutes": float(row.start_minutes),
            "start_time_source": "scraped",
            "start_time_url": getattr(row, "url", ""),
        }

    for row in documents.itertuples():
        code = str(row.nrc_school_code)
        rows[code] = {
            "nrc_school_code": code,
            "start_minutes": float(row.start_minutes),
            "start_time_source": "district document",
            "start_time_url": getattr(row, "source_url", ""),
        }

    for row in manual.itertuples():
        minutes = parse_clock_to_minutes(getattr(row, "start_time", ""))
        if pd.isna(minutes):
            continue
        code = str(row.nrc_school_code)
        rows[code] = {
            "nrc_school_code": code,
            "start_minutes": float(minutes),
            "start_time_source": "manual",
            "start_time_url": getattr(row, "source_url", ""),
        }

    columns = ["nrc_school_code", "start_minutes", "start_time_source", "start_time_url"]
    if not rows:
        return pd.DataFrame(columns=columns)
    return pd.DataFrame(list(rows.values()))


# ---------------------------------------------------------------------------
# Sample filters
# ---------------------------------------------------------------------------

def apply_sample_filters(frame: pd.DataFrame, site: str) -> tuple[pd.DataFrame, list[dict]]:
    """Keep the schools the study is about, and record every exclusion.

    The counts are returned alongside the filtered table so the README and the
    dashboard can both show how the sample was built. A sample that shrinks
    without explanation is the easiest way for a study to mislead, including by
    accident.
    """
    settings = site_config(site)
    sample = settings["sample"]
    steps: list[dict] = []
    out = frame.copy()
    steps.append({"step": "Schools in the source data", "schools": len(out)})

    if "excluded_school_types" in sample:
        out = out[~out["school_type"].isin(set(sample["excluded_school_types"]))]
        steps.append(
            {"step": "After removing alternative and special education schools",
             "schools": len(out)}
        )

    if "exclude_name_patterns" in sample:
        # Transfer schools and Young Adult Borough Centers enroll students who
        # already left or fell behind at another school. Their graduation rates
        # are measured on a different population and are not comparable.
        pattern = "|".join(sample["exclude_name_patterns"])
        before = len(out)
        out = out[~out["school_name"].str.contains(pattern, case=False, na=False)]
        steps.append(
            {"step": f"After removing transfer schools and borough centers ({before - len(out)} removed)",
             "schools": len(out)}
        )

    out = out[out["enrollment"] >= sample["min_enrollment"]]
    steps.append(
        {"step": f"After removing schools under {sample['min_enrollment']} students",
         "schools": len(out)}
    )

    return out.reset_index(drop=True), steps


# ---------------------------------------------------------------------------
# Per site assembly
# ---------------------------------------------------------------------------

def merge_nyc() -> pd.DataFrame:
    """Assemble the New York City analysis table."""
    return pd.read_csv(INTERIM_DIR / "clean_nyc.csv", dtype={"school_id": str})


def merge_nevada() -> pd.DataFrame:
    """Join the Nevada sources and attach start times."""
    nrc = pd.read_csv(
        INTERIM_DIR / "clean_nevada_report_card.csv", dtype={"nrc_school_code": str}
    )
    ccd = pd.read_csv(INTERIM_DIR / "clean_ccd.csv", dtype={"nrc_school_code": str})

    # An inner join on purpose. A school that appears in only one source cannot
    # contribute both an outcome and a full set of controls, so it could not
    # enter a regression anyway.
    merged = nrc.merge(ccd, on="nrc_school_code", how="inner", validate="one_to_one")
    print(f"  {len(merged)} schools matched between Nevada Report Card and the federal directory")

    merged = merged.merge(load_start_times(), on="nrc_school_code", how="left")

    # Put Nevada onto the shared column names.
    merged["school_id"] = merged["nrc_school_code"]
    merged["enrollment"] = merged["enrollment_ccd"].fillna(merged["enrollment_nrc"])
    merged["poverty_pct"] = merged["direct_cert_pct"]
    return merged


def run(site: str | None = None) -> tuple[pd.DataFrame, list[dict]]:
    """Build and write the analysis table for one site."""
    settings = site_config(site)
    site = settings["name"]
    print(f"Merging sources for {settings['label']}")

    merged = merge_nyc() if site == "nyc" else merge_nevada()

    merged = build_analysis_variables(merged, site)
    merged, sample_steps = apply_sample_filters(merged, site)

    with_time = int(merged["start_minutes"].notna().sum())
    sample_steps.append(
        {"step": "With a known start time (the analysis sample)", "schools": with_time}
    )

    for step in sample_steps:
        print(f"    {step['schools']:>4}  {step['step']}")

    path = analysis_table_path(site)
    path.parent.mkdir(parents=True, exist_ok=True)
    merged.to_csv(path, index=False)

    out_dir = outputs_dir(site)
    out_dir.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(sample_steps).to_csv(out_dir / "sample_construction.csv", index=False)

    print(f"  wrote {path.name} with {len(merged)} rows")
    return merged, sample_steps
