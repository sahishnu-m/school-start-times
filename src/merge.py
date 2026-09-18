"""Joins the cleaned sources into the single table the analysis reads.

The join key is the Nevada school code. It comes from Nevada Report Card
directly, and from the federal directory by stripping the district prefix off
the state school code. Matching on a code rather than on a school name avoids a
whole category of quiet errors, since the two sources abbreviate names
differently and Nevada has several schools whose names differ by one word.

This module also settles the rule the study set for start times: when the
scraper and the hand entered file both have a value for a school, the hand
entered value wins.
"""

from __future__ import annotations

import pandas as pd

from .clean import apply_sample_filters, build_analysis_variables, parse_clock_to_minutes
from .config import ANALYSIS_TABLE, INTERIM_DIR, MANUAL_START_TIMES, OUTPUTS_DIR


def load_start_times() -> pd.DataFrame:
    """Combine every start time source into one column, in order of trust.

    Three sources feed this, and they are applied worst first so that better
    ones overwrite them:

      1. the page scraper, which guesses from free text on a web page
      2. a district published bell schedule document, which is the district
         stating its own times in one table
      3. the hand entered file, where a person read the schedule themselves

    The returned table carries a start_time_source column, so the analysis can
    report how many times came from each and a reader can judge the data
    collection rather than take it on faith.
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

    rows = {}

    # Order is the whole mechanism here. Each block overwrites the one above it.
    if not scraped.empty:
        for row in scraped.itertuples():
            code = str(row.nrc_school_code)
            rows[code] = {
                "nrc_school_code": code,
                "start_minutes": float(row.start_minutes),
                "start_time_source": "scraped",
                "start_time_url": getattr(row, "url", ""),
                "start_time_evidence": getattr(row, "evidence", ""),
            }

    if not documents.empty:
        for row in documents.itertuples():
            code = str(row.nrc_school_code)
            rows[code] = {
                "nrc_school_code": code,
                "start_minutes": float(row.start_minutes),
                "start_time_source": "district document",
                "start_time_url": getattr(row, "source_url", ""),
                "start_time_evidence": (
                    f"matched to {getattr(row, 'document_school_name', '')} "
                    f"in {getattr(row, 'source_document', '')}"
                ),
            }

    if not manual.empty:
        for row in manual.itertuples():
            raw_value = getattr(row, "start_time", "")
            minutes = parse_clock_to_minutes(raw_value)
            if pd.isna(minutes):
                continue
            code = str(row.nrc_school_code)
            rows[code] = {
                "nrc_school_code": code,
                "start_minutes": float(minutes),
                "start_time_source": "manual",
                "start_time_url": getattr(row, "source_url", ""),
                "start_time_evidence": getattr(row, "notes", ""),
            }

    if not rows:
        return pd.DataFrame(
            columns=[
                "nrc_school_code",
                "start_minutes",
                "start_time_source",
                "start_time_url",
                "start_time_evidence",
            ]
        )
    return pd.DataFrame(list(rows.values()))


def run() -> tuple[pd.DataFrame, list[dict]]:
    """Merge everything and write data/processed/analysis_table.csv."""
    print("Merging sources")

    nrc = pd.read_csv(
        INTERIM_DIR / "clean_nevada_report_card.csv", dtype={"nrc_school_code": str}
    )
    ccd = pd.read_csv(INTERIM_DIR / "clean_ccd.csv", dtype={"nrc_school_code": str})

    # An inner join on purpose. A school that appears in only one source cannot
    # contribute both an outcome and a full set of controls, so it could not
    # enter a regression anyway.
    merged = nrc.merge(ccd, on="nrc_school_code", how="inner", validate="one_to_one")
    print(f"  {len(merged)} schools matched between Nevada Report Card and the federal directory")

    start_times = load_start_times()
    merged = merged.merge(start_times, on="nrc_school_code", how="left")

    merged = build_analysis_variables(merged)
    merged, sample_steps = apply_sample_filters(merged)

    with_time = merged["start_minutes"].notna().sum()
    sample_steps.append(
        {"step": "With a known start time (the analysis sample)", "schools": int(with_time)}
    )

    for step in sample_steps:
        print(f"    {step['schools']:>4}  {step['step']}")

    ANALYSIS_TABLE.parent.mkdir(parents=True, exist_ok=True)
    merged.to_csv(ANALYSIS_TABLE, index=False)

    OUTPUTS_DIR.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(sample_steps).to_csv(OUTPUTS_DIR / "sample_construction.csv", index=False)

    print(f"  wrote {ANALYSIS_TABLE.name} with {len(merged)} rows")
    return merged, sample_steps


if __name__ == "__main__":
    run()
