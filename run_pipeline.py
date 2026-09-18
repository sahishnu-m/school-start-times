"""Runs the study end to end for one site.

    python run_pipeline.py                 runs the site named in config.yaml
    python run_pipeline.py --site nyc      New York City, the main analysis
    python run_pipeline.py --site nevada   Nevada, the comparison case
    python run_pipeline.py --site all      both, one after the other

Each stage is a separate module and each writes its output to disk before the
next one starts. That is deliberate. It means a stage can be rerun on its own
while working on it, and when a number in the final table looks wrong you can
open the intermediate files and find the stage where it went wrong, instead of
reading the whole pipeline top to bottom.

New York City stages:

  1. fetch_nyc      the high school directory, the demographic snapshot, and
                    the graduation cohort, joined on the DBN
  2. clean          typing, renaming, start time parsing
  3. merge          sample filters
  4. analyze        correlations, model ladder, stratified comparison
  5. charts         the four charts

Nevada stages add the start time collection, which New York City does not need
because its start times are published as data:

  1. fetch_ccd                  federal school directory
  2. fetch_nevada_report_card   state accountability data
  3. clean and merge
  4. fetch_bell_documents       district published bell schedule documents
  5. scrape_start_times         crawl school websites
  6. merge again                fold the collected start times in
  7. analyze and charts

Options:

  --skip-scrape   Nevada only. Leave the start time collection alone and use
                  what is already in data/manual/start_times.csv. Useful while
                  working on the analysis, since the scrape is the slow stage.
"""

from __future__ import annotations

import argparse
import os
import sys

from src import analyze, charts, clean, merge
from src.config import ensure_directories, site_config


def run_nyc() -> int:
    """New York City: fetch, clean, merge, analyse, draw."""
    from src import fetch_nyc

    print("=" * 70)
    print("Stage 1: fetch")
    print("=" * 70)
    fetch_nyc.run()

    print()
    print("=" * 70)
    print("Stage 2 and 3: clean and merge")
    print("=" * 70)
    clean.run_nyc()
    frame, _ = merge.run("nyc")

    print()
    print("=" * 70)
    print("Stage 4 and 5: analyse and draw")
    print("=" * 70)
    results = analyze.run(frame, "nyc")
    charts.run(frame, results, "nyc")

    known = int(frame["start_minutes"].notna().sum())
    print()
    print(f"Done. {known} schools with a start time, of {len(frame)} in the sample.")
    return 0


def run_nevada(skip_scrape: bool) -> int:
    """Nevada: fetch, clean, merge, collect start times, merge again, analyse."""
    from src import fetch_bell_documents, fetch_ccd, fetch_nevada_report_card, scrape_start_times

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
    clean.run_nevada()
    frame, _ = merge.run("nevada")

    if not skip_scrape:
        print()
        print("=" * 70)
        print("Stage 5: collect start times")
        print("=" * 70)
        schools = frame[["nrc_school_code", "school_name", "district"]]

        # District published documents first. They cover whole districts at
        # once, so anything they answer is a page the scraper does not need to
        # go looking for.
        fetch_bell_documents.run(schools)
        scrape_start_times.run(schools.rename(columns={"school_name": "nrc_school_name"}))

        print()
        print("=" * 70)
        print("Stage 6: merge again, now with start times")
        print("=" * 70)
        frame, _ = merge.run("nevada")

    known = int(frame["start_minutes"].notna().sum())
    if known == 0:
        print()
        print("No start times are available, so there is nothing to analyse.")
        print("Open data/manual/start_times.csv, fill in the start_time column,")
        print("and run this again with --skip-scrape.")
        print()
        print("The pipeline does not guess or simulate start times. A study built")
        print("on invented data would produce a result that looks real and means")
        print("nothing.")
        return 1

    print()
    print("=" * 70)
    print("Stage 7 and 8: analyse and draw")
    print("=" * 70)
    results = analyze.run(frame, "nevada")
    charts.run(frame, results, "nevada")

    print()
    print(f"Done. {known} schools with a start time, of {len(frame)} in the sample.")
    return 0


def main() -> int:
    parser = argparse.ArgumentParser(description="Run the school start time study")
    parser.add_argument(
        "--site",
        default=None,
        choices=["nyc", "nevada", "all"],
        help="which site to run (default: the active_site in config.yaml)",
    )
    parser.add_argument(
        "--skip-scrape",
        action="store_true",
        help="Nevada only: do not visit school websites, use existing start times",
    )
    arguments = parser.parse_args()

    requested = arguments.site or os.environ.get("STUDY_SITE") or None
    sites = ["nyc", "nevada"] if requested == "all" else [requested] if requested else [None]

    status = 0
    for site in sites:
        name = site_config(site)["name"]
        # Downstream modules read the active site from the environment, so set
        # it here rather than passing it through every single call.
        os.environ["STUDY_SITE"] = name
        ensure_directories(name)

        print()
        print("#" * 70)
        print(f"# {site_config(name)['label']}, {site_config(name)['year_label']}")
        print("#" * 70)

        status |= run_nyc() if name == "nyc" else run_nevada(arguments.skip_scrape)

    print()
    print("=" * 70)
    print("Tables and charts are in outputs/<site>/.")
    print("Run the dashboard with: streamlit run streamlit_app.py")
    print("=" * 70)
    return status


if __name__ == "__main__":
    sys.exit(main())
