# School Start Times and Academic Performance

Tests whether public high schools that start later show better academic results
than schools that start earlier, once student poverty, school size, English
learners and school selectivity are accounted for.

The main sample is 425 New York City high schools, where start times run from
7:15 AM to past 9:00 AM and every school sits inside one school system. A
second sample of 38 Nevada high schools is kept because it shows why this
question is hard to study at all.

**[Live app](https://school-start-times.streamlit.app)**

---

## Motivation

Adolescent sleep research has been consistent for years. Circadian rhythms
shift later during puberty, so a teenager asked to be alert at 7:00 in the
morning is being asked something biologically harder than the same request made
of an adult. The American Academy of Pediatrics recommended in 2014 that high
schools start no earlier than 8:30 AM, and districts that actually moved their
bell schedules later, including Seattle in a 2018 Science Advances study,
reported longer sleep and better grades.

This project asks whether any of that shows up in the numbers schools already
publish, and takes one problem seriously from the start. Start times are not
assigned at random. A district staggers bell schedules so one set of buses can
run several routes each morning, and which schools get the early route depends
on geography and budget. Both track neighbourhood income, which is among the
strongest predictors of school results there is. A raw correlation between
start time and performance is at serious risk of being a correlation between
poverty and performance wearing a disguise.

---

## Quick start

```bash
git clone https://github.com/sahishnu-m/school-start-times.git
cd school-start-times

python -m venv .venv
.venv\Scripts\activate        # Windows
# source .venv/bin/activate   # macOS / Linux

pip install -r requirements.txt

python run_pipeline.py --site nyc
streamlit run app.py
```

It opens at `http://localhost:8501`. Add `?site=nevada` to the address to open
the Nevada comparison instead, which makes a link to either sample shareable.

The Nevada half needs no scraping if you keep the collected start times:

```bash
python run_pipeline.py --site nevada --skip-scrape
python run_pipeline.py --site all --skip-scrape     # both
```

Everything is cached under `data/`, so later runs are fast.

---

## Why two samples

The project began in Nevada and hit a wall worth describing, because it
explains the whole design.

Clark County sets one bell time for almost all of its comprehensive high
schools. Of the 38 Nevada high schools whose start time could be verified, 29
begin at exactly 7:00 AM, a standard deviation of 17 minutes. A study of
whether start time predicts performance needs schools that start at different
times, and Nevada does not supply enough of them. District fixed effects could
not even be estimated, because every school with a known start time sat in the
same district.

New York City solves both problems. It publishes a start time for every public
high school, those times span nearly three hours, and all 425 schools sit
inside one school system. Busing budgets, union contracts, state funding and
the local labour market are therefore held constant by construction instead of
being tangled up with start time. That is the comparison the Nevada version
wanted to make and could not.

Nevada is kept rather than deleted. A study that quietly drops the sample that
did not work is not being honest about how it reached its answer.

---

## Method

| Step | File | What happens |
|---|---|---|
| 1. Fetch, NYC | [`src/fetch_nyc.py`](src/fetch_nyc.py) | Pulls the high school directory, the demographic snapshot and the graduation cohort from NYC Open Data and joins them on the DBN. |
| 1. Fetch, Nevada | [`src/fetch_ccd.py`](src/fetch_ccd.py), [`src/fetch_nevada_report_card.py`](src/fetch_nevada_report_card.py) | Federal school directory, then state accountability data for the high schools the federal file identifies. |
| 2. Start times, Nevada | [`src/fetch_bell_documents.py`](src/fetch_bell_documents.py), [`src/scrape_start_times.py`](src/scrape_start_times.py) | Reads the Clark County bell schedule PDF, then crawls district websites for the rest. New York City needs neither, since it publishes start times as data. |
| 3. Clean | [`src/clean.py`](src/clean.py) | Parses start times from free text, handles suppressed values, and converts both samples into one shared set of column names so everything downstream is written once. |
| 4. Merge | [`src/merge.py`](src/merge.py) | Joins the sources, applies the sample filters, and records every exclusion. |
| 5. Analyse | [`src/analyze.py`](src/analyze.py), [`src/stats_core.py`](src/stats_core.py) | Unadjusted correlation, a ladder of regressions adding one control block at a time, a stratified comparison, and two robustness checks. |
| 6. Charts | [`src/charts.py`](src/charts.py) | Five charts per sample, written to `outputs/<site>/`. |
| 7. App | [`app.py`](app.py) | Five numbered parts, including a panel where the reader toggles each control and watches the estimate move. |

**The poverty control is not free lunch.** New York City feeds every student,
and Clark County does the same under the federal Community Eligibility
Provision, so in both places the reported free lunch rate describes a billing
rule rather than how poor a school is. The median reported rate across the
Nevada sample is 99.8 percent. New York City uses the Economic Need Index
instead, and Nevada uses the direct certification rate, which counts students
confirmed low income through SNAP, TANF or foster care records. Both vary
properly, which is what a control variable needs in order to control anything.

**Admissions method is a control, because NYC high schools are chosen rather
than zoned.** A screened school, an audition school, or one of the eight
specialized schools that admit on a citywide exam takes students who were
already doing well, and the specialized schools also start earlier than
average. Leaving selectivity out would let it masquerade as an effect of start
time.

**Standard errors are clustered on the school building.** Several NYC schools
share one building, and schools in a building share a facility, a neighbourhood
and often the same applicants. There are 254 buildings behind the 427 schools.
Nevada uses heteroskedasticity robust (HC3) errors.

**The statistics run on numpy alone.** [`src/stats_core.py`](src/stats_core.py)
implements least squares, HC3 and cluster robust errors, the Student t
distribution, the Pearson correlation and the Welch test.
[`tests/test_stats_core.py`](tests/test_stats_core.py) refits every model in the
study both ways and checks it against statsmodels and scipy, which agree to
about one part in a billion. The reason for writing them out is under the
second bug below.

---

## Results

Mostly nothing, and the two clearest results point in opposite directions.

Unadjusted correlations, New York City:

| Outcome | n | r | 95% CI |
|---|---|---|---|
| Four year graduation rate | 421 | -0.04 | -0.14 to 0.05 |
| Advanced Regents diploma rate | 421 | -0.13 | -0.23 to -0.04 |
| Attendance rate | 423 | -0.08 | -0.18 to 0.01 |
| College and career readiness | 383 | 0.05 | -0.05 to 0.15 |
| Dropout rate | 421 | -0.00 | -0.10 to 0.10 |

Fully adjusted, per hour later start:

| Outcome | Estimate | 95% CI | n | In SDs |
|---|---|---|---|---|
| Four year graduation rate | +1.14 | -1.36 to 3.63 | 405 | +0.07 |
| Advanced Regents diploma rate | -5.05 | -9.16 to -0.94 | 405 | -0.23 |
| Attendance rate | -0.45 | -1.34 to 0.44 | 407 | -0.07 |
| College and career readiness | +4.16 | 0.91 to 7.41 | 367 | +0.22 |
| Dropout rate | -1.23 | -2.41 to -0.05 | 405 | -0.17 |

The primary outcome, graduation rate, shows nothing at any rung of the ladder.
It starts at -1.68 with no controls and ends at +1.14 with all of them, and
every interval along the way contains zero. Attendance shows nothing either.

Three estimates do exclude zero. Later starting schools have a **lower**
Advanced Regents diploma rate, a **higher** college and career readiness rate,
and a slightly **lower** dropout rate. All three survive dropping the exam and
audition schools, so they are not an artifact of the eight specialized high
schools.

If starting later simply helped students, those outcomes would move together.
They do not. The strictest academic measure moves against later start times by
about a quarter of a standard deviation while college readiness moves with them
by about the same amount. That pattern is what residual confounding looks like.
It is more consistent with later starting schools being a different kind of
school than with a later bell changing what students learn: the specialized
exam schools start earliest and the small, newer, career focused schools start
latest, and the available controls cannot fully separate that from the clock.

The stratified comparison supports the cautious reading. Split into poverty
bands, the Advanced Regents difference is negative in three of four bands and
every interval contains zero.

**Nevada cannot answer the question**, because of the sample rather than the
statistics. Its adjusted models produce estimates of +2.9 to +3.5 ACT points
per hour, about a full standard deviation, resting on the nine schools that
start anywhere other than 7:00. Its graduation rate result also fails its
robustness check: +6.35 points per hour (95% CI 1.17 to 11.52) becomes +0.25
(95% CI -34.82 to 35.32) once the eight schools whose rate was published as
">95" are dropped.

---

## Data sources and scraping ethics

New York City, all 2018-19 and joined on the DBN:

- [2019 DOE High School Directory](https://data.cityofnewyork.us/d/uq7m-95z8):
  start time, attendance, college and career readiness, admissions method,
  applicants per seat.
- [2018-19 School Demographic Snapshot](https://data.cityofnewyork.us/d/45j8-f6um):
  Economic Need Index, English learners, students with disabilities, enrollment.
- [Graduation results, cohorts 2012 to 2019](https://data.cityofnewyork.us/d/mjm3-8dw8):
  graduation rate, Advanced Regents rate and dropout rate for the cohort that
  entered in 2015 and was due to finish in June 2019.

Nevada, 2022-23:

- [Nevada Report Card](https://nevadareportcard.nv.gov/DI/), read through the
  same JSON API the public site uses, so the values match what a person would
  read off the website.
- NCES Common Core of Data through the
  [Urban Institute Education Data API](https://educationdata.urban.org/documentation/schools.html).
- The Clark County School District
  [2022-2023 start and end times PDF](https://ewscripps.brightspotcdn.com/8d/09/49c15ff94026813f63113026bd4a/2022-2023-ccsd-school-start-end-times.pdf),
  a district publication dated on its first page, from a copy hosted by a Las
  Vegas television station.

Nevada Report Card labels a school year by the year it ends in and the Common
Core of Data labels the same year by the year it starts in, so `nrc_year: 2023`
and `ccd_year: 2022` in `config.yaml` are the same school year. Getting that
wrong is the easiest way to silently merge two different years.

The scraper treats robots.txt as binding. It checks before requesting any page
on a host, skips anything disallowed, and records the reason in
`outputs/nevada/robots_decisions.csv`. It waits two to three seconds between
requests to the same host and honours a longer `Crawl-delay` where a site
publishes one; several Nevada districts ask for five seconds. The user agent
names the project and gives a contact address, and every page is cached so a
rerun costs the sites nothing.

It read 253 pages across 15 district sites, was refused 12, and produced zero
start times it was confident enough to keep. Most Nevada districts build their
school lists in the browser, so the links are not in the HTML the scraper
receives. Seven bell schedule pages it found but could not parse are listed in
`outputs/nevada/start_time_leads.csv`.

---

## Limitations

**This design cannot establish causation.** Schools were not assigned start
times at random. Controlling for the characteristics measured here removes the
part of the confounding that was measured and does nothing about the rest.

**The data is cross sectional.** Every school is observed once. There is no
before and after, so a school that starts late and scores well might have
scored just as well when it started early.

**New York City is one city.** Holding the district constant removes a pile of
confounders and also means the result describes New York City rather than
American high schools.

**Who starts late is not random.** Specialized exam schools start earliest and
small newer schools start latest. A school choosing a later bell time is making
a decision that probably correlates with things nobody measured.

**Start times are self reported free text.** The NYC directory field is typed
by school staff and contains entries such as "8am or 8:45am". One school was
listed at "8:20pm" and was treated as a typing error. Measurement error in a
predictor generally pushes an estimated relationship toward zero.

**Nevada has almost no start time variation**, which is why its estimates are
implausibly large and why its graduation rate result does not survive the
censoring check.

**The sleep literature is background, not support.** Those studies used
stronger designs. This one cannot borrow their credibility.

**A null result is a result.** Where the intervals include zero, the correct
conclusion is that this data does not show a relationship, and that is what is
reported.

---

## Project layout

```
config.yaml              every tunable choice, split by site
run_pipeline.py          runs all stages for one site or both
app.py                   the Streamlit dashboard

src/
  config.py                    paths and per site settings
  polite.py                    robots.txt, rate limiting, page caching
  fetch_nyc.py                 the three NYC Open Data datasets
  fetch_ccd.py                 federal school directory
  fetch_nevada_report_card.py  state accountability data
  fetch_bell_documents.py      district published bell schedule PDFs
  scrape_start_times.py        crawls Nevada school websites
  clean.py                     typing, renaming, shared column names
  merge.py                     joins sources, applies the sample filter
  stats_core.py                least squares, robust and cluster standard
                               errors, the t distribution, in numpy
  analyze.py                   correlations, model ladder, robustness checks
  charts.py                    the five charts

tests/test_stats_core.py       checks stats_core against statsmodels and scipy
.streamlit/config.toml         pins the dashboard to the light theme
docs/DEPLOY.md                 Streamlit Community Cloud steps

data/
  raw/ cache/ interim/   downloads and working files, not tracked
  processed/             one merged analysis table per site
  manual/                hand entered Nevada start times, tracked

outputs/nyc/             charts and result tables
outputs/nevada/          charts, result tables, and the scraping logs
```

---

## Two bugs worth knowing about

Both attached a real looking number to the wrong school, which is worse than a
gap, because a gap is visible in the data and a wrong value is not.

**A school code that was not the school code it looked like.** The Clark County
bell schedule PDF prints a three digit number beside every school, and the
state school code is also a number ending in three digits. Joining on it ran
without an error and matched 36 of 54 schools. Almost all of those matches were
wrong: the PDF number is the district internal school number and has nothing to
do with the state code, so Arbor View High School was handed the start time of
whichever unrelated school shared its digits. The fix was to match on school
names instead and write both names into the output so every pairing can be
read. A join that succeeds is not the same as a join that is right.

**A level marker that was stripped before it was used.** Matching on names
meant normalising them, and normalising stripped the "- HS" and "- ES" suffixes
that the district uses to tell its schools apart. Nevada reuses place names
across levels, so Virgin Valley High School and Virgin Valley Elementary both
became "virgin valley", and the high school picked up the elementary school's
9:05 AM start instead of its own 7:15. The fix reads the level marker first and
filters the document to rows that can be a high school, before the name is
normalised. The same pass caught an approximate string matcher pairing
Southeast and West Career and Technical Academy with East Career and Technical
Academy, which are three separate schools, so that matcher was removed in
favour of exact and leading word matches only.

A third one is in the scraper rather than the data: the first version reported
Douglas High School as starting at 6:35 AM, which is an optional zero period.
It now demands stronger evidence and returns nothing rather than a plausible
looking wrong number.

---

## License

MIT. NYC Open Data is public domain under the
[NYC Open Data Terms of Use](https://www.nyc.gov/html/data/terms.html). Nevada
Report Card and NCES Common Core of Data are public records. School names and
district publications belong to their owners.
