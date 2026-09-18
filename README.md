# Do later school start times go with better academic performance?

A study of public high schools, using New York City as the main sample and
Nevada as a comparison case.

Repository: <https://github.com/sahishnu-m/school-start-times>

Live dashboard: deploy it with the steps in `docs/DEPLOY.md`.

## Motivation

Adolescent sleep research has been consistent for years. Circadian rhythms shift
later during puberty, so a teenager asked to be alert at 7:00 in the morning is
being asked something biologically harder than the same request made of an
adult. The American Academy of Pediatrics recommended in 2014 that middle and
high schools start no earlier than 8:30 AM. Districts that actually moved their
bell schedules later, including Seattle in a study published in Science Advances
in 2018, reported that students slept longer and grades improved.

This project asks whether that shows up in the numbers schools already publish.

Answering it honestly requires taking one problem seriously from the beginning.
Start times are not assigned at random. A district staggers bell schedules so
one set of buses can run several routes each morning, and which schools get the
early route depends on geography and budget. Geography and budget also track
neighbourhood income, and neighbourhood income is one of the strongest
predictors of school test scores there is. A raw correlation between start time
and performance is at serious risk of being a correlation between poverty and
performance in disguise.

## Why the study uses two samples

The project began in Nevada and hit a wall worth describing, because it explains
the whole design.

Clark County School District sets one bell time for almost all of its
comprehensive high schools. Of the 38 Nevada high schools for which a start time
could be verified, 29 begin at exactly 7:00 AM. The full range is 7:00 to 8:00,
a standard deviation of 17 minutes. A study of whether start time predicts
performance needs schools that start at different times, and Nevada does not
supply enough of them. District fixed effects could not even be estimated,
because every school with a known start time was in the same district.

New York City solves both problems at once. It publishes a start time for every
public high school, and those times run from 7:15 AM to past 9:00 AM across more
than 400 schools. Because all of those schools sit inside one school system, the
things that differ between districts, such as busing budgets, union contracts,
state funding, and the local labour market, are held constant by construction
rather than being tangled up with start time. That is the comparison the Nevada
version wanted to make and could not.

Nevada is kept in the repository and the dashboard. A study that quietly deletes
the sample that did not work is not being honest about how it reached its
answer, and the Nevada case is a clean demonstration of why this question is
hard to study at all.

## Data sources

### New York City, the main sample

All three datasets come from NYC Open Data and are joined on the DBN, the code
that identifies a New York City school.

- **2019 DOE High School Directory** ([uq7m-95z8](https://data.cityofnewyork.us/d/uq7m-95z8)).
  Start time, attendance rate, college and career readiness rate, admissions
  method, and applicants per seat.
- **2018-19 School Demographic Snapshot** ([45j8-f6um](https://data.cityofnewyork.us/d/45j8-f6um)).
  Economic Need Index, English learner percentage, students with disabilities,
  enrollment.
- **Graduation results for cohorts 2012 to 2019** ([mjm3-8dw8](https://data.cityofnewyork.us/d/mjm3-8dw8)).
  Four year graduation rate, Advanced Regents diploma rate, and dropout rate for
  the cohort that entered in 2015 and was due to finish in June 2019.

The year is 2018-19 throughout. The directory published for 2019 admissions
describes schools as they operated in 2018-19, the demographic snapshot is the
2018-19 file, and the graduation cohort is the one that finished in June 2019.

### Nevada, the comparison case

- **Nevada Report Card**, Nevada Department of Education.
  <https://nevadareportcard.nv.gov/DI/>
  Graduation rates, ACT composite scores, ELA and math proficiency, chronic
  absenteeism, enrollment, and English learner percentage. Retrieved through the
  JSON API the public site itself uses, so the values match what a person would
  read off the website.
- **Common Core of Data**, NCES, through the
  [Urban Institute Education Data API](https://educationdata.urban.org/documentation/schools.html).
  Enrollment, locale, charter status, free and reduced lunch counts, and direct
  certification counts.
- **Clark County School District 2022-2023 school start and end times**, a
  district publication listing the morning bell time for every Clark County
  school.
  [PDF](https://ewscripps.brightspotcdn.com/8d/09/49c15ff94026813f63113026bd4a/2022-2023-ccsd-school-start-end-times.pdf)
  The copy linked here is hosted by a Las Vegas television station. The document
  itself is a district publication, dated on its first page.
- **Everything else**, collected from district websites by the scraper in
  `src/scrape_start_times.py` and by hand into `data/manual/start_times.csv`.

### A note on years

Nevada Report Card labels a school year by the year it ends in. The Common Core
of Data labels the same year by the year it starts in. So `nrc_year: 2023` and
`ccd_year: 2022` in `config.yaml` refer to the same school year. Getting this
wrong is the easiest way to silently merge two different years.

### Scraping conduct

The Nevada scraper checks robots.txt before requesting any page on a host and
skips anything disallowed, recording the reason in
`outputs/nevada/robots_decisions.csv`. It waits two to three seconds between
requests to the same host, and where a site publishes a longer `Crawl-delay` it
waits for that instead. Several Nevada district sites ask for five seconds, and
the scraper honours it. The user agent names the project and gives a contact
address. Every page fetched is cached, so re-running the pipeline does not
refetch anything.

New York City needs no scraping. Its start times are published as data.

## Method

### Outcomes

New York City, in order: four year graduation rate (primary), Advanced Regents
diploma rate, attendance rate, college and career readiness rate, and dropout
rate.

Nevada, in order: ACT composite score (primary), graduation rate, ELA
proficiency, math proficiency, and chronic absenteeism. Nevada gives the ACT to
every grade 11 student in a public school, so a school average is not distorted
by which students chose to sit the test.

### The poverty control

The obvious control is the percentage of students on free or reduced price
lunch, and neither sample uses it.

New York City offers free meals to every student, and Clark County does the same
under the federal Community Eligibility Provision. In both places the reported
free lunch rate describes a billing rule rather than how poor a school is. The
median reported free lunch rate across the Nevada sample is 99.8 percent.

New York City uses the Economic Need Index instead, which estimates the share of
students facing economic hardship from public assistance records, neighbourhood
poverty, and family circumstances. Nevada uses the direct certification rate:
students confirmed low income through SNAP, TANF, or foster care records. Both
vary properly across their samples, which is what a control variable needs in
order to control for anything.

### The models

1. The unadjusted correlation between start time and each outcome, reported
   first and clearly labelled as having no controls.
2. A ladder of regressions, adding one block of controls at a time, reporting
   the start time coefficient at every rung. For New York City:
   - Model 1: no controls
   - Model 2: plus Economic Need Index
   - Model 3: plus enrollment, English learners, students with disabilities
   - Model 4: plus borough fixed effects
   - Model 5: plus admissions method and applicants per seat
3. A stratified comparison: schools split into poverty bands, then early and
   late starters compared within each band. This does not assume the
   relationship is a straight line, so it checks the regression rather than
   repeating it.
4. Robustness checks. For New York City, refitting without the exam and audition
   schools. For Nevada, refitting without graduation rates that were published
   as a bound such as ">95".

Standard errors for New York City are clustered on the school building. Several
NYC schools share one building, and schools in the same building share a
facility, a neighbourhood, and often the same pool of applicants, so treating
them as independent observations would make the intervals too narrow. There are
254 buildings behind the 427 schools. Nevada uses heteroskedasticity robust
(HC3) errors.

### Why admissions method is a control

New York City high schools are chosen, not zoned. A screened school, an audition
school, or one of the eight specialized schools that admit on a citywide exam
takes students who were already doing well, so its results say as much about who
got in as about what the school did. The specialized schools also start earlier
than average. Leaving selectivity out would let it masquerade as an effect of
start time.

## Results

**Headline: mostly nothing, and the two clearest results point in opposite
directions.**

### What the sample looks like

425 New York City high schools have a usable start time. The median is 8:15 AM,
the range is 7:15 AM to 9:50 AM, and the standard deviation is 23 minutes. One
school listed at "8:20pm" was treated as a typing error and set to missing.

### Unadjusted correlations

| Outcome | n | r | 95% CI |
|---|---|---|---|
| Four year graduation rate | 421 | -0.04 | -0.14 to 0.05 |
| Advanced Regents diploma rate | 421 | -0.13 | -0.23 to -0.04 |
| Attendance rate | 423 | -0.08 | -0.18 to 0.01 |
| College and career readiness | 383 | 0.05 | -0.05 to 0.15 |
| Dropout rate | 421 | -0.00 | -0.10 to 0.10 |

### The fully adjusted estimates

Model 5, per hour later start:

| Outcome | Estimate | 95% CI | n | In SDs |
|---|---|---|---|---|
| Four year graduation rate | +1.14 | -1.35 to 3.62 | 405 | +0.07 |
| Advanced Regents diploma rate | -5.05 | -9.14 to -0.96 | 405 | -0.23 |
| Attendance rate | -0.45 | -1.34 to 0.44 | 407 | -0.07 |
| College and career readiness | +4.16 | 0.93 to 7.40 | 367 | +0.22 |
| Dropout rate | -1.23 | -2.40 to -0.05 | 405 | -0.17 |

The primary outcome, graduation rate, shows nothing at any rung of the ladder.
The unadjusted estimate is -1.68 and the fully adjusted one is +1.14, and every
interval along the way contains zero. Attendance shows nothing either.

Three estimates do exclude zero in the full model. Later starting schools have a
**lower** Advanced Regents diploma rate, a **higher** college and career
readiness rate, and a slightly **lower** dropout rate. All three survive
dropping the exam and audition schools, so they are not an artifact of the eight
specialized high schools.

### Why the split matters more than any single estimate

If starting later simply helped students, the outcomes would move together. They
do not. The Advanced Regents rate, which is the strictest academic measure here,
moves against later start times by about a quarter of a standard deviation,
while college and career readiness moves with them by about the same amount.

That pattern is what residual confounding looks like. It is more consistent with
later starting schools being a different kind of school than with a later bell
changing what students learn. In this sample the specialized exam schools start
earliest and the small, newer, career focused schools start latest, and the
controls available here cannot fully separate that from the clock.

The stratified comparison supports the cautious reading. Split into poverty
bands, the Advanced Regents difference between early and late starters is
negative in three of four bands, and every interval contains zero.

### The Nevada comparison

Nevada cannot answer the question, and the reason is the sample rather than the
statistics. 29 of the 38 schools with a known start time begin at exactly
7:00 AM. All unadjusted correlations contain zero. The adjusted models produce
estimates of +2.9 to +3.5 ACT points per hour, which is about a full standard
deviation and is not believable, resting on the nine schools that start anywhere
other than 7:00. District fixed effects are reported as not identified rather
than silently repeating the previous model.

The Nevada graduation rate result also fails its robustness check. The
unadjusted estimate of +6.35 points per hour (95% CI 1.17 to 11.52) becomes
+0.25 (95% CI -34.82 to 35.32) once the eight schools whose rate was published
as ">95" are dropped. It was an artifact of substituting the midpoint.

### The Nevada scraper

The page scraper read 253 pages across 15 district websites, was refused 12 by
robots.txt, hit 21 errors, and skipped 3 districts whose sites did not resolve.
It produced zero start times it was confident enough to keep, for two reasons
recorded in `outputs/nevada/scrape_log.csv`.

Most Nevada districts run their sites on a platform that builds the school list
in the browser, so links to individual school sites are not in the HTML the
scraper receives. And where the scraper did reach a bell schedule, it could not
reliably tell which time was the start of the day: on the Douglas High School
schedule it initially reported 6:35 AM, which is an optional zero period rather
than the 7:30 first bell. It now requires stronger evidence and returns nothing
rather than a plausible looking wrong number.

Seven bell schedule links it could not parse are listed in
`outputs/nevada/start_time_leads.csv`.

### A near miss worth recording

During data collection a news report said Washoe County approved an 8:30 AM high
school start for 2022-23. Following it up showed the board reversed the decision
a month later and the change was never implemented. Using the first report would
have put a wrong start time on 16 schools, and since those would have been the
only non-Clark schools in the sample, they would have driven the entire Nevada
result.

### Conclusion

This study does not find that later start times go with better academic
performance, and it does not rule it out. The primary outcome shows nothing. The
secondary results that survive every control disagree with each other in a way
that points at confounding rather than at an effect.

The honest summary is that comparing different schools at one point in time is a
weak way to answer this question, even with 400 schools inside one district and
a reasonable set of controls. What would actually settle it is following the
same schools through a bell schedule change.

## Limitations

**This design cannot establish causation.** Schools were not assigned start
times at random. Controlling for the characteristics measured here removes the
part of the confounding that was measured and does nothing about the part that
was not.

**The data is cross sectional.** Every school is observed once. There is no
before and after.

**New York City is one city.** Holding the district constant removes a pile of
confounders and also means the result describes New York City rather than
American high schools. A city where most students take public transport to a
school they applied to is not a typical American district.

**Who starts late is not random.** Specialized exam schools start earliest and
small newer schools start latest. A school choosing a later bell time is making
a decision that probably correlates with things nobody measured.

**Start times are self reported free text.** The NYC directory field is typed by
school staff and contains entries such as "8am or 8:45am". A single number
cannot represent a school running different schedules on different days.
Measurement error in a predictor generally pushes an estimated relationship
toward zero.

**Nevada has almost no start time variation**, which is why it cannot answer the
question, and why several of its estimates are implausibly large.

**Some Nevada values are censored.** ">95" is replaced with the midpoint of the
implied range, and the sensitivity check that drops those schools matters: the
graduation rate result does not survive it.

**The sleep literature is background, not support.** The studies cited in the
motivation used stronger designs. This study cannot borrow their credibility.

**A null result is a result.** Where the intervals include zero, the correct
conclusion is that this data does not show a relationship, and that is what is
reported.

## How to run it locally

You need Python 3.11 or newer.

```bash
git clone https://github.com/sahishnu-m/school-start-times.git
cd school-start-times

python -m venv .venv
source .venv/bin/activate        # Windows: .venv\Scripts\activate

pip install -r requirements.txt
```

Run the main analysis:

```bash
python run_pipeline.py --site nyc
```

Run the Nevada comparison, skipping the slow website crawl:

```bash
python run_pipeline.py --site nevada --skip-scrape
```

Run both:

```bash
python run_pipeline.py --site all --skip-scrape
```

Start the dashboard:

```bash
streamlit run streamlit_app.py
```

Everything is cached under `data/`, so later runs are fast.

### Filling in missing Nevada start times

`data/manual/start_times.csv` has one row per Nevada school in the sample, with
an `already_collected` column showing what the pipeline already has. Where
`start_time` is blank, no start time was found. Fill it in as `7:45 AM` and
re-run with `--skip-scrape`. Hand entered values always win over anything
collected automatically.

## Repository layout

```
config.yaml              every tunable choice, split by site
run_pipeline.py          runs all stages for one site or both
streamlit_app.py         the dashboard

src/
  config.py                    paths, site settings
  polite.py                    robots.txt, rate limiting, page caching
  fetch_nyc.py                 the three NYC Open Data datasets
  fetch_ccd.py                 federal school directory (Nevada)
  fetch_nevada_report_card.py  state accountability data (Nevada)
  fetch_bell_documents.py      district published bell schedule PDFs
  scrape_start_times.py        crawls Nevada school websites
  clean.py                     typing, renaming, shared column names
  merge.py                     joins sources, applies the sample filter
  analyze.py                   correlations, model ladder, robustness checks
  charts.py                    the charts

data/
  raw/        untouched downloads
  cache/      cached web pages
  interim/    one tidy table per source
  processed/  one merged analysis table per site
  manual/     hand entered Nevada start times, tracked in git

outputs/nyc/      charts and result tables for New York City
outputs/nevada/   charts, result tables, and the scraping logs for Nevada
```

## Deploying the dashboard

See `docs/DEPLOY.md`.
