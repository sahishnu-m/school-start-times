"""Collects school start times from district and school websites.

Start times are the one variable in this study that does not exist as a
dataset. No state or federal agency collects them. They live on individual
school websites, usually on a page called something like "Bell Schedule" or
"School Hours", written as free text in whatever format that school chose.

That shapes the design in two ways.

First, this scraper is expected to miss schools. It is a best effort collector,
not a complete one. Every page it visits and every decision it makes is written
to outputs/scrape_log.csv, so a person can see what it found, what it could not
reach, and what it was not permitted to read.

Second, the scraper is not the authority. A separate file,
data/manual/start_times.csv, holds times entered by hand. When both the scraper
and the manual file have a time for the same school, the manual value wins. A
person who reads a bell schedule page is more reliable than a regular
expression, so the human entry is treated as the correction and the scraped
value is treated as the draft.

Anything the scraper extracts is recorded with the sentence it came from, so a
suspicious number can be checked against its source without refetching.
"""

from __future__ import annotations

import csv
import re
from urllib.parse import urljoin, urlparse

import pandas as pd
from bs4 import BeautifulSoup

from .clean import minutes_to_clock
from .config import CONFIG, INTERIM_DIR, OUTPUTS_DIR, ROBOTS_LOG, SCRAPE_LOG, SCRAPER_SOURCES
from .polite import PoliteFetcher

SCRAPE_CONFIG = CONFIG["scraping"]

# A clock time such as "7:45 AM", "7:45am", or "8:00". The meridiem is optional
# because plenty of bell schedule tables leave it off once the column header
# has established it.
TIME_PATTERN = re.compile(
    r"\b(\d{1,2})\s*:\s*(\d{2})\s*([ap])\.?\s*m\.?\b|\b(\d{1,2})\s*:\s*(\d{2})\b",
    re.IGNORECASE,
)

# Phrases that suggest a nearby time is when the instructional day begins.
# Weighted, because "first bell" is much stronger evidence than "schedule".
START_CUES = {
    "first bell": 5,
    "1st bell": 5,
    "school begins": 5,
    "school starts": 5,
    "classes begin": 5,
    "class begins": 5,
    "instruction begins": 5,
    "start time": 4,
    "school hours": 3,
    "tardy bell": 4,
    "period 1": 3,
    "1st period": 3,
    "first period": 3,
    "warning bell": 2,
    "bell schedule": 2,
    "daily schedule": 2,
    "start of day": 4,
}

# Phrases that mean a nearby time is something other than the start of the
# instructional day. These subtract, so a time sitting next to "dismissal"
# is not mistaken for a start time.
NOT_START_CUES = {
    "dismissal": -6,
    "dismissed": -6,
    "end of day": -6,
    "school ends": -6,
    "release": -4,
    "office hours": -5,
    "doors open": -3,
    "breakfast": -3,
    "lunch": -4,
    "buses depart": -4,
    "early release": -3,
    "late start": -1,
    "after school": -4,
    "practice": -3,
    # Zero period is an optional class before the school day starts. Douglas
    # High School publishes a bell schedule whose first row is a period 0 at
    # 6:35 AM, and an earlier version of this scraper reported that as the
    # school start time. The real first bell there is 7:30 AM.
    "period 0": -6,
    "0 period": -6,
    "zero period": -6,
    "0 hour": -6,
    "zero hour": -6,
    "optional": -2,
}

# The minimum score a candidate needs before it is accepted at all.
#
# Set to 3 so that the mere presence of the words "bell schedule" somewhere on
# the page, which scores 2, is not enough on its own. A page that says "bell
# schedule" and nothing more specific is a page listing a whole day of times,
# and picking one of them is guessing. The study would rather have a gap, which
# is visible in the data, than a wrong number, which is not.
MINIMUM_CONFIDENCE = 3

# A plausible window for the start of a high school day, in minutes after
# midnight. Anything outside 6:30 AM to 10:00 AM is almost certainly a lunch
# period, a dismissal, or an office phone number that happened to look like a
# time. This filter removes most false positives on its own.
EARLIEST_PLAUSIBLE_MINUTES = 6 * 60 + 30
LATEST_PLAUSIBLE_MINUTES = 10 * 60

# Links are followed in priority order rather than in the order they appear on
# the page. The crawl budget per district is small, and a district home page
# offers far more links than the budget allows. Spending it on a link that says
# "Bell Schedule" before one that says "About Us" is the difference between
# finding a start time and running out of budget in a news archive.
STRONG_LINK_CUES = re.compile(
    r"bell|school\s*hours|start\s*time|daily\s*schedule|schedule", re.IGNORECASE
)
WEAK_LINK_CUES = re.compile(
    r"our\s*schools|schools?/|high\s*school|\bhs\b|academy|students|about", re.IGNORECASE
)

# A link that plainly says "bell schedule" but points off the district domain,
# at a PDF, or at a Google Sheet, cannot be parsed here. Several Nevada
# districts publish their bell schedules exactly that way. Rather than drop
# those, they are written to outputs/start_time_leads.csv so that a person
# filling in the manual file has the address to open.
LEAD_FILE_HINTS = re.compile(r"\.pdf$|docs\.google|drive\.google|\.xlsx?$|\.docx?$", re.IGNORECASE)

# A lead is only worth recording if the link is about bell times. The broader
# word "schedule" is useless here: it caught salary schedules, fee schedules,
# and athletics game schedules, which buried the one Humboldt link that
# actually said "All Schools Bell Schedules".
LEAD_LINK_CUES = re.compile(
    r"bell|school\s*hours|start\s*time|start\s*and\s*end|daily\s*schedule", re.IGNORECASE
)


def to_minutes(hour: int, minute: int, meridiem: str | None) -> int | None:
    """Convert a clock time to minutes after midnight.

    When a page writes "7:45" with no AM or PM, the hour itself decides. An
    hour of 6 through 11 on a school schedule page means morning. An hour of 1
    through 5 means afternoon, which is outside the plausible start window and
    will be filtered out anyway.
    """
    if meridiem:
        meridiem = meridiem.lower()
        if meridiem == "p" and hour != 12:
            hour += 12
        elif meridiem == "a" and hour == 12:
            hour = 0
    else:
        # No meridiem given. Hours 1 through 5 on a school page are afternoon.
        if 1 <= hour <= 5:
            hour += 12

    if not (0 <= hour <= 23 and 0 <= minute <= 59):
        return None
    return hour * 60 + minute


def score_candidate(context: str) -> int:
    """Score how much a snippet of text looks like a start time statement."""
    lowered = context.lower()
    score = 0
    for phrase, weight in START_CUES.items():
        if phrase in lowered:
            score += weight
    for phrase, weight in NOT_START_CUES.items():
        if phrase in lowered:
            score += weight
    return score


def extract_start_time_candidates(html: str, url: str) -> list[dict]:
    """Pull every plausible start time out of one page, with its evidence.

    The function returns candidates rather than a single answer. Deciding which
    candidate to believe happens later, once all pages for a school are in
    hand, because a district page and a school page may disagree.
    """
    soup = BeautifulSoup(html, "lxml")

    # Strip the parts of a page that never contain a bell schedule but often
    # contain times, such as a news ticker in the footer or a script block.
    for tag in soup(["script", "style", "noscript"]):
        tag.decompose()

    text = soup.get_text(separator=" ", strip=True)
    text = re.sub(r"\s+", " ", text)

    page_title = soup.title.get_text(strip=True) if soup.title else ""
    heading = soup.find(["h1", "h2"])
    page_heading = heading.get_text(strip=True) if heading else ""

    candidates = []
    for match in TIME_PATTERN.finditer(text):
        if match.group(1):
            hour, minute, meridiem = int(match.group(1)), int(match.group(2)), match.group(3)
        else:
            hour, minute, meridiem = int(match.group(4)), int(match.group(5)), None

        minutes = to_minutes(hour, minute, meridiem)
        if minutes is None:
            continue
        if not (EARLIEST_PLAUSIBLE_MINUTES <= minutes <= LATEST_PLAUSIBLE_MINUTES):
            continue

        # Take a window of text either side of the match. 140 characters is
        # wide enough to catch a table header sitting to the left of a time
        # and narrow enough not to drag in an unrelated paragraph.
        start = max(0, match.start() - 140)
        end = min(len(text), match.end() + 140)
        context = text[start:end]

        score = score_candidate(context)
        if score < MINIMUM_CONFIDENCE:
            continue

        candidates.append(
            {
                "url": url,
                "page_title": page_title,
                "page_heading": page_heading,
                "start_minutes": minutes,
                "start_clock": minutes_to_clock(minutes),
                "confidence_score": score,
                "evidence": context.strip(),
            }
        )

    return candidates


def normalize_school_name(name: str) -> str:
    """Reduce a school name to a form that can be compared across sources.

    Nevada Report Card writes "Coronado HS" where a school website writes
    "Coronado High School". Lowercasing, dropping punctuation, and expanding
    the common abbreviations makes those two strings equal.
    """
    text = str(name).lower()
    text = re.sub(r"[^a-z0-9 ]", " ", text)
    text = re.sub(r"\bhs\b", "high school", text)
    text = re.sub(r"\bjr\b", "junior", text)
    text = re.sub(r"\bsr\b", "senior", text)
    text = re.sub(r"\bacad\b", "academy", text)
    text = re.sub(r"\bsch\b", "school", text)
    # These words appear in nearly every school name and carry no signal for
    # telling two schools apart.
    text = re.sub(r"\b(high school|school|academy|the|of|at)\b", " ", text)
    return re.sub(r"\s+", " ", text).strip()


def match_school(page_text: str, school_lookup: dict[str, str]) -> str | None:
    """Decide which school a page belongs to.

    school_lookup maps a normalized school name to its Nevada school code. A
    page is assigned to a school when that school name appears in the page
    title, heading, or URL. Names shorter than four characters are skipped,
    since a two letter fragment would match almost anything.
    """
    haystack = normalize_school_name(page_text)
    best_code, best_length = None, 0
    for normalized_name, code in school_lookup.items():
        if len(normalized_name) < 4:
            continue
        if normalized_name in haystack and len(normalized_name) > best_length:
            best_code, best_length = code, len(normalized_name)
    return best_code


def crawl_district(
    fetcher: PoliteFetcher,
    district: str,
    seed_url: str,
    school_lookup: dict[str, str],
    log_rows: list[dict],
    lead_rows: list[dict],
) -> list[dict]:
    """Walk outward from one district seed page looking for bell schedules.

    The crawl is bounded by a depth limit and a page limit, both set in
    config.yaml, and it visits pages in priority order. A link whose text says
    "Bell Schedule" is visited before a link that says "About Us", and shallow
    pages are visited before deep ones. Without that ordering the crawl budget
    gets spent on staff directories and news posts, which was exactly what
    happened in the first version of this scraper.
    """
    seed_host = urlparse(seed_url).netloc
    # The registrable domain, so that a school subdomain such as
    # foothillhs.example.net is recognised as part of example.net and followed,
    # while an outbound link to a vendor site is not.
    base_domain = ".".join(seed_host.split(".")[-2:])

    # Entries are (priority, depth, url). Lower priority number is visited
    # first. A list kept sorted is used rather than heapq because the queue
    # never holds more than a few hundred items and staying readable is worth
    # more here than the difference in speed.
    queue: list[tuple[int, int, str]] = [(0, 0, seed_url)]
    seen = {seed_url}
    found: list[dict] = []
    pages_fetched = 0

    while queue and pages_fetched < SCRAPE_CONFIG["max_pages_per_district"]:
        queue.sort(key=lambda item: (item[0], item[1]))
        priority, depth, url = queue.pop(0)
        result = fetcher.get(url)
        pages_fetched += 1

        log_rows.append(
            {
                "district": district,
                "url": url,
                "depth": depth,
                "status": result.status,
                "reason": result.reason,
                "candidates_found": 0,
            }
        )

        if result.status not in ("ok", "cached"):
            # A disallowed or failed page ends this branch. The reason is
            # already in the log, which is what the study promised to record.
            continue

        candidates = extract_start_time_candidates(result.html, url)
        for candidate in candidates:
            identifier = " ".join(
                [candidate["page_title"], candidate["page_heading"], url]
            )
            candidate["district"] = district
            candidate["nrc_school_code"] = match_school(identifier, school_lookup)
            found.append(candidate)
        log_rows[-1]["candidates_found"] = len(candidates)

        if depth >= SCRAPE_CONFIG["max_depth"]:
            continue

        soup = BeautifulSoup(result.html, "lxml")
        for anchor in soup.find_all("a", href=True):
            link = urljoin(url, anchor["href"]).split("#")[0]
            parsed = urlparse(link)
            if parsed.scheme not in ("http", "https"):
                continue

            anchor_text = anchor.get_text(" ", strip=True)
            target = f"{anchor_text} {parsed.path}"
            is_strong = bool(STRONG_LINK_CUES.search(target))

            off_domain = not parsed.netloc.endswith(base_domain)
            unparseable = bool(LEAD_FILE_HINTS.search(link))

            # A link that clearly points at a bell schedule but cannot be read
            # here is recorded as a lead instead of being discarded. Carson
            # City, for one, publishes its bell schedules in a Google Sheet.
            if LEAD_LINK_CUES.search(target) and (off_domain or unparseable):
                lead_rows.append(
                    {
                        "district": district,
                        "link_text": anchor_text,
                        "url": link,
                        "found_on": url,
                        "reason": "google sheet, document, or off-site page that this scraper cannot parse",
                    }
                )
                continue

            if off_domain or link in seen:
                continue
            if not is_strong and not WEAK_LINK_CUES.search(target):
                continue

            seen.add(link)
            queue.append((0 if is_strong else 1, depth + 1, link))

    return found


def choose_best_per_school(candidates: pd.DataFrame) -> pd.DataFrame:
    """Reduce many candidate times per school to one.

    The rule is to take the highest confidence score, and to break a tie by
    taking the earliest time. The tie breaker leans earlier because a bell
    schedule page lists the whole day, and among several equally well
    supported morning times the first one is the one the day starts with.
    """
    if candidates.empty:
        return candidates

    ranked = candidates.sort_values(
        ["nrc_school_code", "confidence_score", "start_minutes"],
        ascending=[True, False, True],
    )
    return ranked.groupby("nrc_school_code", as_index=False).first()


def write_manual_template(schools: pd.DataFrame, scraped: pd.DataFrame) -> None:
    """Create or update data/manual/start_times.csv for hand entry.

    Every school in the sample gets a row whether or not the scraper found it,
    so the file doubles as a checklist. If the file already exists, values a
    person has typed are preserved. Only new schools are appended. Overwriting
    hand entered work on every pipeline run would make the file useless.
    """
    columns = [
        "nrc_school_code",
        "school_name",
        "district",
        "start_time",
        "source_url",
        "notes",
    ]

    template = schools[["nrc_school_code", "nrc_school_name", "district"]].copy()
    template = template.rename(columns={"nrc_school_name": "school_name"})
    template["start_time"] = ""
    template["source_url"] = ""
    template["notes"] = ""
    template = template[columns]

    if SCRAPER_SOURCES.parent.exists() and (SCRAPER_SOURCES.parent / "start_times.csv").exists():
        existing = pd.read_csv(
            SCRAPER_SOURCES.parent / "start_times.csv", dtype={"nrc_school_code": str}
        )
        known = set(existing["nrc_school_code"])
        new_rows = template[~template["nrc_school_code"].isin(known)]
        combined = pd.concat([existing, new_rows], ignore_index=True)
        kept = len(existing[existing["start_time"].astype(str).str.strip() != ""])
        print(f"  manual file already exists, kept {kept} hand entered times, "
              f"added {len(new_rows)} new rows")
    else:
        combined = template
        print(f"  created manual start time template with {len(combined)} rows")

    # Show what the pipeline already has for each school, so the file works as
    # a to-do list. Without this a person cannot tell which of 117 rows still
    # need looking up. These columns are never read back by the pipeline. Only
    # the start_time column is, and a value typed there overrides everything.
    combined["already_collected"] = ""
    combined["already_collected_source"] = ""

    documents_path = INTERIM_DIR / "start_times_documents.csv"
    if documents_path.exists():
        documents = pd.read_csv(documents_path, dtype={"nrc_school_code": str})
        known = documents.set_index("nrc_school_code")["start_clock"].to_dict()
        combined["already_collected"] = combined["nrc_school_code"].map(known).fillna("")
        combined.loc[combined["already_collected"] != "", "already_collected_source"] = (
            "district document"
        )

    if not scraped.empty:
        suggestions = scraped.set_index("nrc_school_code")["start_clock"].to_dict()
        source = scraped.set_index("nrc_school_code")["url"].to_dict()
        evidence = scraped.set_index("nrc_school_code")["evidence"].to_dict()
        # A scraped value only fills a gap the document did not already cover,
        # matching the order of trust the pipeline uses.
        gap = combined["already_collected"] == ""
        combined.loc[gap, "already_collected"] = (
            combined.loc[gap, "nrc_school_code"].map(suggestions).fillna("")
        )
        combined.loc[gap & (combined["already_collected"] != ""), "already_collected_source"] = (
            "scraper, check before trusting"
        )
        combined["scraper_source_url"] = combined["nrc_school_code"].map(source)
        combined["scraper_evidence"] = combined["nrc_school_code"].map(evidence)

    still_needed = int((combined["already_collected"] == "").sum())
    print(f"  {still_needed} of {len(combined)} schools still need a start time entered by hand")

    combined.to_csv(SCRAPER_SOURCES.parent / "start_times.csv", index=False)


def write_default_sources() -> None:
    """Write the district seed list if it is not already there.

    The list is a CSV rather than a hard coded Python dictionary so that adding
    a district, or fixing a URL that changed, does not require editing code.
    Three Nevada districts are listed with a blank URL because their websites
    did not resolve when the list was built. Their high schools still appear in
    the manual start time file and can be filled in by hand.
    """
    if SCRAPER_SOURCES.exists():
        return

    rows = [
        ("Carson City", "https://www.carsoncityschools.com/", ""),
        ("Churchill", "https://www.churchillcsd.com/", ""),
        ("Clark", "https://www.ccsd.net/schools/", ""),
        ("Douglas", "https://www.dcsd.net/", ""),
        ("Elko", "https://www.ecsdnv.net/", ""),
        ("Esmeralda", "https://esmeralda.k12.nv.us/en-US", ""),
        ("Eureka", "https://www.ecsdnv.org/", ""),
        ("Humboldt", "https://www.hcsdnv.com/", ""),
        ("Lander", "https://www.lander.k12.nv.us/", ""),
        ("Lincoln", "https://www.lcsdnv.com/", ""),
        ("Lyon", "https://www.lyoncsd.org/", ""),
        ("Mineral", "", "district website did not resolve, enter times by hand"),
        ("Nye", "https://www.nye.k12.nv.us/", ""),
        ("Pershing", "", "district website did not resolve, enter times by hand"),
        ("Storey", "", "site certificate error, enter times by hand"),
        ("Washoe", "https://www.washoeschools.net/", ""),
        ("White Pine", "https://www.whitepine.k12.nv.us/", ""),
        ("State Public Charter Schools", "https://charterschools.nv.gov/", ""),
    ]

    SCRAPER_SOURCES.parent.mkdir(parents=True, exist_ok=True)
    with open(SCRAPER_SOURCES, "w", newline="", encoding="utf-8") as handle:
        writer = csv.writer(handle)
        writer.writerow(["district", "seed_url", "notes"])
        writer.writerows(rows)


def run(schools: pd.DataFrame) -> pd.DataFrame:
    """Scrape start times for the schools in the sample.

    schools must carry nrc_school_code, nrc_school_name, and district.
    """
    write_default_sources()
    sources = pd.read_csv(SCRAPER_SOURCES).fillna("")

    fetcher = PoliteFetcher()
    log_rows: list[dict] = []
    lead_rows: list[dict] = []
    all_candidates: list[dict] = []

    for _, source in sources.iterrows():
        district = source["district"]
        seed_url = str(source["seed_url"]).strip()

        if not seed_url:
            log_rows.append(
                {
                    "district": district,
                    "url": "",
                    "depth": 0,
                    "status": "skipped",
                    "reason": source["notes"] or "no seed URL configured",
                    "candidates_found": 0,
                }
            )
            print(f"  {district}: skipped, {source['notes'] or 'no seed URL'}")
            continue

        # Restrict name matching to schools in this district. Two districts can
        # both have a "Career and Technical Academy", and a district scoped
        # lookup keeps a Clark page from being credited to a Washoe school.
        in_district = schools[schools["district"] == district]
        lookup = {
            normalize_school_name(row.nrc_school_name): row.nrc_school_code
            for row in in_district.itertuples()
        }

        print(f"  {district}: crawling from {seed_url}")
        found = crawl_district(fetcher, district, seed_url, lookup, log_rows, lead_rows)
        matched = sum(1 for f in found if f["nrc_school_code"])
        print(f"    {len(found)} time candidates, {matched} matched to a school")
        all_candidates.extend(found)

    OUTPUTS_DIR.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(log_rows).to_csv(SCRAPE_LOG, index=False)
    pd.DataFrame(fetcher.robots_decisions).to_csv(ROBOTS_LOG, index=False)

    # Leads are addresses a person should open by hand. They are deduplicated
    # because a district will link the same bell schedule from several pages.
    if lead_rows:
        leads = pd.DataFrame(lead_rows).drop_duplicates(subset=["url"])
        leads.to_csv(OUTPUTS_DIR / "start_time_leads.csv", index=False)
        print(f"  recorded {len(leads)} bell schedule links that need to be opened by hand")

    candidates = pd.DataFrame(all_candidates)
    if not candidates.empty:
        candidates.to_csv(INTERIM_DIR / "start_time_candidates.csv", index=False)
        candidates = candidates[candidates["nrc_school_code"].notna()]

    best = choose_best_per_school(candidates)
    if not best.empty:
        best.to_csv(INTERIM_DIR / "start_times_scraped.csv", index=False)

    print(f"  scraper produced times for {len(best)} of {len(schools)} schools")
    write_manual_template(schools, best)
    return best
