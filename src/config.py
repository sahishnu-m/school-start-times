"""Loads config.yaml and defines the project's directory layout.

Every other module imports paths from here instead of building its own, so
there is exactly one definition of where data lives.
"""

from pathlib import Path

import yaml

# PROJECT_ROOT is the folder containing config.yaml. Resolving it from this
# file's location means the pipeline works no matter which directory you run
# it from.
PROJECT_ROOT = Path(__file__).resolve().parent.parent

# raw       untouched responses from the data sources, exactly as downloaded
# cache     saved web pages so that re-running the scraper does not refetch
# interim   one tidy table per source, cleaned but not yet joined
# processed the single merged analysis table
# manual    hand-entered data, the only data folder tracked in git
DATA_DIR = PROJECT_ROOT / "data"
RAW_DIR = DATA_DIR / "raw"
CACHE_DIR = DATA_DIR / "cache"
INTERIM_DIR = DATA_DIR / "interim"
PROCESSED_DIR = DATA_DIR / "processed"
MANUAL_DIR = DATA_DIR / "manual"
OUTPUTS_DIR = PROJECT_ROOT / "outputs"

# The two files a person edits by hand.
MANUAL_START_TIMES = MANUAL_DIR / "start_times.csv"
SCRAPER_SOURCES = MANUAL_DIR / "scraper_sources.csv"

# The merged table that the analysis and the dashboard both read.
ANALYSIS_TABLE = PROCESSED_DIR / "analysis_table.csv"

# Logs that record what the scraper did and why it skipped things.
SCRAPE_LOG = OUTPUTS_DIR / "scrape_log.csv"
ROBOTS_LOG = OUTPUTS_DIR / "robots_decisions.csv"


def load_config(path: Path | None = None) -> dict:
    """Read config.yaml into a plain dictionary."""
    path = path or (PROJECT_ROOT / "config.yaml")
    with open(path, "r", encoding="utf-8") as handle:
        return yaml.safe_load(handle)


def ensure_directories() -> None:
    """Create every data and output directory if it does not already exist.

    Called at the start of the pipeline so that no later step has to worry
    about whether its destination folder is there.
    """
    for directory in (
        RAW_DIR,
        CACHE_DIR,
        INTERIM_DIR,
        PROCESSED_DIR,
        MANUAL_DIR,
        OUTPUTS_DIR,
    ):
        directory.mkdir(parents=True, exist_ok=True)


CONFIG = load_config()
