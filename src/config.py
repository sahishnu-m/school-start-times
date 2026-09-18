"""Loads config.yaml and defines the project's directory layout.

Every other module imports paths and settings from here, so there is exactly
one definition of where data lives and one definition of what the current site
is.

The study covers two sites, New York City and Nevada. Most of the pipeline is
written once and runs against either, because the cleaning step converts both
into the same set of column names. What differs between them lives in
config.yaml under the sites key: which outcomes exist, what the poverty measure
is called, and which controls enter each model.
"""

from __future__ import annotations

import os
from pathlib import Path

import yaml

# PROJECT_ROOT is the folder containing config.yaml. Resolving it from this
# file's location means the pipeline works no matter which directory you run
# it from.
PROJECT_ROOT = Path(__file__).resolve().parent.parent

# raw       untouched responses from the data sources, exactly as downloaded
# cache     saved web pages so that re-running the scraper does not refetch
# interim   one tidy table per source, cleaned but not yet joined
# processed one merged analysis table per site
# manual    hand-entered data, the only data folder tracked in git
DATA_DIR = PROJECT_ROOT / "data"
RAW_DIR = DATA_DIR / "raw"
CACHE_DIR = DATA_DIR / "cache"
INTERIM_DIR = DATA_DIR / "interim"
PROCESSED_DIR = DATA_DIR / "processed"
MANUAL_DIR = DATA_DIR / "manual"
OUTPUTS_DIR = PROJECT_ROOT / "outputs"

# The two files a person edits by hand, both specific to the Nevada site.
MANUAL_START_TIMES = MANUAL_DIR / "start_times.csv"
SCRAPER_SOURCES = MANUAL_DIR / "scraper_sources.csv"


def load_config(path: Path | None = None) -> dict:
    """Read config.yaml into a plain dictionary."""
    path = path or (PROJECT_ROOT / "config.yaml")
    with open(path, "r", encoding="utf-8") as handle:
        return yaml.safe_load(handle)


CONFIG = load_config()


def active_site() -> str:
    """Which site the pipeline is currently working on.

    The environment variable wins over config.yaml so that the command line
    flag in run_pipeline.py, and the site picker in the dashboard, can switch
    sites without rewriting the config file on disk.
    """
    return os.environ.get("STUDY_SITE", CONFIG.get("active_site", "nyc"))


def site_config(site: str | None = None) -> dict:
    """The settings block for one site, with shared defaults filled in.

    A site may override a shared setting by defining it in its own block. That
    is how Nevada gets its own poverty band edges, since its poverty measure
    runs on a different scale from the New York City one.
    """
    site = site or active_site()
    if site not in CONFIG["sites"]:
        raise KeyError(f"unknown site {site!r}, expected one of {sorted(CONFIG['sites'])}")

    settings = dict(CONFIG["sites"][site])
    settings["name"] = site
    settings.setdefault("stratification", CONFIG["stratification"])
    settings.setdefault("early_late_cutoff_minutes", CONFIG["early_late_cutoff_minutes"])
    return settings


def analysis_table_path(site: str | None = None) -> Path:
    """Where the merged analysis table for a site is written."""
    return PROCESSED_DIR / f"analysis_table_{site or active_site()}.csv"


def outputs_dir(site: str | None = None) -> Path:
    """Where charts and result tables for a site are written.

    Each site gets its own folder so that running one does not overwrite the
    other's results.
    """
    return OUTPUTS_DIR / (site or active_site())


def ensure_directories(site: str | None = None) -> None:
    """Create every data and output directory if it does not already exist."""
    for directory in (
        RAW_DIR,
        CACHE_DIR,
        INTERIM_DIR,
        PROCESSED_DIR,
        MANUAL_DIR,
        OUTPUTS_DIR,
        outputs_dir(site),
    ):
        directory.mkdir(parents=True, exist_ok=True)
