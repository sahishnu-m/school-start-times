"""Runs the whole study end to end.

    python run_pipeline.py

Each stage is a separate module and each writes its output to disk before the
next one starts. That is deliberate. It means a stage can be rerun on its own
while working on it, and it means that when a number in the final table looks
wrong you can open the intermediate files and find the stage where it went
wrong, instead of reading the whole pipeline top to bottom.

Stages:

  1. fetch_ccd                  federal school directory, one request
  2. fetch_nevada_report_card   state accountability data, one request per school
  3. clean                      typing, renaming, suppressed value handling
  4. merge                      join sources, attach start times, filter sample
  5a. fetch_bell_documents      district published bell schedule documents
  5b. scrape_start_times        collect start times from school websites
  6. merge again                bring the newly scraped times into the table
  7. analyze                    correlations, model ladder, stratified comparison
  8. charts                     write the four charts to /outputs

Merge runs twice because the scraper needs to know which schools are in the
sample before it goes looking for them, and the sample is not known until the
two data sources have been joined. The second merge folds the scraped times
back in.

Command line options:

  --skip-scrape   leave the start time collection alone and use whatever is
                  already in data/manual/start_times.csv. Useful while working
                  on the analysis, since the scrape is the slow stage.
"""

from __future__ import annotations

import argparse
import sys

import pandas as pd

from src import analyze, charts, clean, fetch_bell_documents, fetch_ccd
from src import fetch_nevada_report_card, merge, scrape_start_times
from src.config import ensure_directories


def main() -> int:
    parser = argparse.ArgumentParser(description="Run the school start time study")
    parser.add_argument(
        "--skip-scrape",
        action="store_true",
        help="do not visit school websites, use the existing manual start times",
    )
    arguments = parser.parse_args()

    ensure_directories()

    print("=" * 70)
    print("Stage 1 and 2: fetch")
    print("=" * 70)
    ccd = fetch_ccd.run()
    high_schools = fetch_ccd.high_school_frame(ccd)
    fetch_nevada_report_card.run(restrict_to_codes=set(high_schools["nrc_school_code"]))

    print()
    print("=" * 70)
    print("Stage 3 and 4: clean and merge")
    print("=" * 70)
    clean.run()
    frame, _ = merge.run()

    if not arguments.skip_scrape:
        print()
        print("=" * 70)
        print("Stage 5: collect start times")
        print("=" * 70)
        schools = frame[["nrc_school_code", "school_name", "district"]]

        # District published documents first. They cover whole districts at
        # once, so anything they answer is a page the scraper does not need to
        # go looking for.
        fetch_bell_documents.run(schools)
        scrape_start_times.run(
            schools.rename(columns={"school_name": "nrc_school_name"})
        )

        print()
        print("=" * 70)
        print("Stage 6: merge again, now with start times")
        print("=" * 70)
        frame, _ = merge.run()

    known_times = int(frame["start_minutes"].notna().sum())
    if known_times == 0:
        print()
        print("No start times are available, so there is nothing to analyse.")
        print("Open data/manual/start_times.csv, fill in the start_time column,")
        print("and run this script again with --skip-scrape.")
        print()
        print("The pipeline does not guess or simulate start times. A study")
        print("built on invented data would produce a result that looks real")
        print("and means nothing.")
        return 1

    print()
    print("=" * 70)
    print("Stage 7 and 8: analyse and draw")
    print("=" * 70)
    results = analyze.run(frame)
    charts.run(frame, results)

    print()
    print("=" * 70)
    print(f"Done. {known_times} schools with a start time, of {len(frame)} in the sample.")
    print("Tables and charts are in outputs/.")
    print("Run the dashboard with: streamlit run streamlit_app.py")
    print("=" * 70)
    return 0


if __name__ == "__main__":
    sys.exit(main())
