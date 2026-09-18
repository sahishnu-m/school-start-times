"""Reads start times out of district published bell schedule documents.

Some districts do publish all of their start times in one place, as a PDF
rather than as a dataset. Clark County School District is the important case
for this study, because Clark alone is close to half the sample and its school
websites are built in a way the page scraper cannot follow.

A document like that is better evidence than anything the page scraper
produces. It is the district stating its own bell times in one table, so it is
consistent across schools and it is dated. That is why start times from here
outrank scraped times, and are outranked only by a time a person entered by
hand.

Matching a document row to a school is the hard part and it is where this
module is deliberately cautious. The school code printed in the Clark document
is the district internal school number, and it is not the state school code
that the rest of this study joins on. Joining on it would look like it worked
and would attach the wrong start time to most schools. So the match is made on
school names, and every match is written out with the document name beside the
state name so a person can check the pairing.
"""

from __future__ import annotations

import difflib
import re

import pandas as pd
import requests
from pypdf import PdfReader

from .clean import minutes_to_clock, parse_clock_to_minutes
from .config import CONFIG, INTERIM_DIR, RAW_DIR
from .polite import USER_AGENT

# Each entry is one district published document.
#
# The Clark document is the district start and end time table for 2022-2023,
# the study year. It is retrieved from a copy hosted by a Las Vegas television
# station, because that copy is reachable and stable. The document itself is a
# Clark County School District publication and is dated on its first page.
BELL_DOCUMENTS = [
    {
        "district": "Clark",
        "url": (
            "https://ewscripps.brightspotcdn.com/8d/09/49c15ff94026813f63113026bd4a/"
            "2022-2023-ccsd-school-start-end-times.pdf"
        ),
        "description": "Clark County School District 2022-2023 school start and end times",
        "filename": "ccsd_start_end_times_2022_2023.pdf",
    },
]

# One row of the Clark table: region, internal school number, school name, the
# morning in bell, and the afternoon out bell.
BELL_ROW_PATTERN = re.compile(
    r"(\d)\s+(\d{3})\s+(.+?)\s+(\d{1,2}:\d{2}\s*[AP]M)\s+(\d{1,2}:\d{2}\s*[AP]M)"
)

# Only two kinds of match are accepted: the normalized names are identical, or
# the document name is the opening words of the school name. An approximate
# string similarity match was tried and removed. It paired Southeast Career and
# Technical Academy and West Career and Technical Academy with East Career and
# Technical Academy, which are three separate schools with separate campuses.
# A wrong match is worse than a gap, because a gap is visible in the data and a
# wrong value is not.


# The level marker the Clark document puts after a school name: "- HS" for a
# high school, "- ES" for elementary, "- MS" for middle, and combined forms
# such as "- H-E" for a school that covers several levels.
#
# This matters more than it looks. Nevada reuses place names across levels, so
# the district has a Virgin Valley elementary school and a Virgin Valley high
# school. Stripping the level marker before comparing names made those two
# identical, and the high school picked up the elementary school 9:05 AM start
# time. Filtering the document to rows that can be a high school fixes it.
#
# The marker appears with a dash ("VIRGIN VALLEY - HS"), without one
# ("SANDY VALLEY ES"), and as a combination ("SANDY VALLEY HS/MS",
# "WEST PREP - H-E"). All three forms have to be recognised, because a marker
# that slips through unrecognised is a marker that does not get filtered.
LEVEL_PATTERN = re.compile(
    r"(?:^|\s|-)\s*((?:[EMHJS]{1,3}S?)(?:\s*[/-]\s*(?:[EMHJS]{1,3}S?))*)\s*$"
)

# Level codes that mean this row is definitely not a high school.
NON_HIGH_LEVELS = {"ES", "MS", "E", "M", "JHS", "JH"}


def level_marker(document_name: str) -> str | None:
    """Return the level code at the end of a document name, if there is one."""
    match = LEVEL_PATTERN.search(document_name.strip().upper())
    if not match:
        return None
    return re.sub(r"\s+", "", match.group(1))


def could_be_a_high_school(document_name: str) -> bool:
    """True unless the document marks this row as some other level.

    A row with no level marker at all, such as "ADV TECH ACADEMY", is kept.
    Those are the magnet and career academies, which are high schools. A row
    marked with a combination such as "HS/MS" or "H-E" is also kept, since it
    covers high school grades among others.
    """
    marker = level_marker(document_name)
    if marker is None:
        return True
    parts = re.split(r"[/-]", marker)
    return not all(part in NON_HIGH_LEVELS for part in parts)


def normalize(name: str) -> str:
    """Reduce a school name so the two sources can be compared.

    The document writes "ADV TECH ACADEMY" and "CANYON SPRINGS HS W/MAGNET"
    where the state file writes "Advanced Technologies Academy" and "Canyon
    Springs High School and the Leadership and Law Preparatory Academy". The
    aim here is to strip the parts that differ for formatting reasons and keep
    the parts that identify the school.
    """
    text = str(name).strip()
    # Take the level marker off the end first, while the original punctuation
    # is still there to anchor it.
    marker = level_marker(text)
    if marker:
        text = LEVEL_PATTERN.sub("", text.strip())

    text = text.lower()
    text = re.sub(r"\bw\s*/\s*(magnet|gate)\b", " ", text)
    text = re.sub(r"\bmagnet\b", " ", text)
    text = re.sub(r"[^a-z0-9 ]", " ", text)
    text = re.sub(r"\b(hs|ms|es|ss)\b", " ", text)
    text = re.sub(r"\bjr\b", "junior", text)
    text = re.sub(r"\bsr\b", "senior", text)
    text = re.sub(r"\badv\b", "advanced", text)
    # One spelling for the career and technical academies, which the two
    # sources write as "TECH", "technical", and "Technologies".
    text = re.sub(r"\btech(nical|nologies)?\b", "tech", text)
    text = re.sub(r"\bprep(aratory)?\b", "prep", text)
    text = re.sub(r"\bint(ernational)?\b", "international", text)
    # Words that appear in most school names and so cannot tell schools apart.
    #
    # "academy" is deliberately not in this list. Dropping it made "Las Vegas
    # Academy of the Arts" and "Las Vegas High School" the same string, and the
    # arts magnet picked up the comprehensive high school start time.
    text = re.sub(r"\b(high|school|the|of|at|and|for|charter|studies)\b", " ", text)
    return re.sub(r"\s+", " ", text).strip()


def prefix_match(school_key: str, document_keys) -> str | None:
    """Match a document name that is the opening words of the school name.

    The state file writes out a school's full official name while the district
    document uses the short form everyone actually says. "Canyon Springs High
    School and the Leadership and Law Preparatory Academy" appears in the
    document as "CANYON SPRINGS HS W/MAGNET", and "Cimarron-Memorial High
    School" appears as "CIMARRON - HS". In both cases the document name is the
    first word or two of the state name.

    Two guards keep this from matching things it should not. The document name
    must be at least five characters, so a stray one word entry cannot swallow
    a school. And if two different document names both fit, nothing is
    returned, because a guess between two candidates is how a wrong start time
    gets attached to a real school.
    """
    school_tokens = school_key.split()
    candidates = []
    for key in document_keys:
        if len(key) < 5:
            continue
        key_tokens = key.split()
        if len(key_tokens) >= len(school_tokens):
            continue
        if school_tokens[: len(key_tokens)] == key_tokens:
            candidates.append(key)

    if not candidates:
        return None
    # When several document names fit, take the longest. It is the most
    # specific, and specificity is what tells "LAS VEGAS ACADEMY" apart from
    # "LAS VEGAS - HS" when the school is Las Vegas Academy of the Arts.
    return max(candidates, key=lambda key: len(key.split()))


def download(document: dict) -> bytes | None:
    """Fetch one document, reusing the saved copy if it is already on disk."""
    path = RAW_DIR / document["filename"]
    if path.exists():
        return path.read_bytes()

    try:
        response = requests.get(
            document["url"],
            headers={"User-Agent": USER_AGENT},
            timeout=CONFIG["scraping"]["request_timeout_seconds"],
        )
    except requests.RequestException as error:
        print(f"  could not fetch {document['district']} document: {error}")
        return None

    if response.status_code != 200:
        print(f"  could not fetch {document['district']} document: HTTP {response.status_code}")
        return None

    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(response.content)
    return response.content


def parse_rows(pdf_path) -> list[dict]:
    """Pull every school row out of the bell schedule PDF."""
    reader = PdfReader(pdf_path)
    text = "\n".join(page.extract_text() or "" for page in reader.pages)

    rows = []
    for _region, _code, name, start, _end in BELL_ROW_PATTERN.findall(text):
        minutes = parse_clock_to_minutes(start)
        if pd.isna(minutes):
            continue
        rows.append(
            {
                "document_school_name": name.strip(),
                "start_minutes": minutes,
                "start_clock": minutes_to_clock(minutes),
            }
        )
    return rows


def match_to_schools(rows: list[dict], schools: pd.DataFrame, district: str) -> pd.DataFrame:
    """Pair document rows with schools in the sample, by name.

    Only schools in the named district are considered, and a row is used at
    most once. Every pairing is returned with both names and a similarity
    score so the result can be read and checked rather than trusted.
    """
    in_district = schools[schools["district"] == district]
    if in_district.empty or not rows:
        return pd.DataFrame()

    # Build the lookup, and throw out any name the document uses twice with two
    # different start times. The Clark document has two rows called "CLARK - HS"
    # starting at 7:00 and 8:04, and there is no way to tell from the document
    # which one the state file means. Guessing between them would put a wrong
    # number on a real school, so neither is used.
    lookup: dict[str, dict] = {}
    ambiguous: set[str] = set()
    for row in rows:
        if not could_be_a_high_school(row["document_school_name"]):
            continue
        key = normalize(row["document_school_name"])
        if not key:
            continue
        if key in lookup:
            if lookup[key]["start_minutes"] != row["start_minutes"]:
                ambiguous.add(key)
            continue
        lookup[key] = row

    for key in ambiguous:
        lookup.pop(key, None)
    if ambiguous:
        print(f"    dropped {len(ambiguous)} document names that appear twice with different times")

    matches = []
    for school in in_district.itertuples():
        target = normalize(school.school_name)
        if not target:
            continue

        if target in lookup:
            best, score, how = target, 1.0, "exact name"
        else:
            best, how = prefix_match(target, lookup.keys()), "leading words"
            if best is None:
                continue
            score = difflib.SequenceMatcher(None, target, best).ratio()

        row = lookup[best]
        matches.append(
            {
                "nrc_school_code": school.nrc_school_code,
                "school_name": school.school_name,
                "document_school_name": row["document_school_name"],
                "match_type": how,
                "match_score": round(score, 3),
                "start_minutes": row["start_minutes"],
                "start_clock": row["start_clock"],
                "district": district,
            }
        )

    return pd.DataFrame(matches)


def run(schools: pd.DataFrame) -> pd.DataFrame:
    """Read every configured bell schedule document and match it to the sample.

    schools must carry nrc_school_code, school_name, and district.
    """
    print("Reading district published bell schedule documents")
    collected = []

    for document in BELL_DOCUMENTS:
        content = download(document)
        if content is None:
            continue

        path = RAW_DIR / document["filename"]
        rows = parse_rows(path)
        print(f"  {document['district']}: {len(rows)} rows read from the document")

        matched = match_to_schools(rows, schools, document["district"])
        if not matched.empty:
            matched["source_document"] = document["description"]
            matched["source_url"] = document["url"]
            exact = int((matched["match_score"] == 1.0).sum())
            print(
                f"    matched {len(matched)} schools by name "
                f"({exact} on an exact name, {len(matched) - exact} on leading words)"
            )
            collected.append(matched)

    if not collected:
        return pd.DataFrame()

    result = pd.concat(collected, ignore_index=True)
    out_path = INTERIM_DIR / "start_times_documents.csv"
    result.to_csv(out_path, index=False)
    print(f"  wrote {out_path.name}, check the match_score column before trusting a low score")
    return result
