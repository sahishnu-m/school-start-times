# Do later school start times go with better academic performance?

A study of Nevada public high schools in the 2022-2023 school year.

Live dashboard: see the deployment section below.

## Motivation

Adolescent sleep research has been consistent for years. Circadian rhythms shift
later during puberty, so a teenager asked to be alert at 7:00 in the morning is
being asked something biologically harder than the same request made of an
adult. The American Academy of Pediatrics recommended in 2014 that middle and
high schools start no earlier than 8:30 AM. Districts that actually moved their
bell schedules later, including Seattle in a study published in Science Advances
in 2018, reported that students slept longer and grades improved.

Nevada high schools vary in when they start. This project asks whether that
variation shows up in the schools' published academic results.

The honest answer to that question requires taking one problem seriously from
the beginning. Start times are not assigned at random. A district staggers bell
schedules so that one set of buses can run several routes each morning, and
which schools get the early route depends on geography and budget. Geography and
budget also track neighbourhood income, and neighbourhood income is one of the
strongest predictors of school test scores there is. So a raw correlation
between start time and performance is at serious risk of being a correlation
between poverty and performance in disguise.

This project reports the raw correlation, labels it as unadjusted, and then
shows what happens to it as controls are added.

## Data sources

**Nevada Report Card**, Nevada Department of Education.
<https://nevadareportcard.nv.gov/DI/>
Graduation rates, ACT composite scores, ELA and math proficiency rates, chronic
absenteeism, enrollment, and English learner percentage. Retrieved through the
JSON API that the public site itself uses, so the values match what a person
would read off the website.

**Common Core of Data**, National Center for Education Statistics, retrieved
through the Urban Institute Education Data API.
<https://educationdata.urban.org/documentation/schools.html>
<https://nces.ed.gov/ccd/>
Enrollment, the city, suburb, town, and rural classification, charter status,
free and reduced price lunch counts, and direct certification counts. The Urban
Institute mirror is used rather than the raw NCES download because it returns
the whole state in one request, and because the NCES download server was not
reachable from the machine this was built on. The underlying numbers are the
NCES numbers.

**Clark County School District 2022-2023 school start and end times**, a
district publication listing the morning bell time for every Clark County
school.
<https://ewscripps.brightspotcdn.com/8d/09/49c15ff94026813f63113026bd4a/2022-2023-ccsd-school-start-end-times.pdf>
This matters because Clark County is close to half of the sample and its school
websites cannot be crawled. The copy linked above is hosted by a Las Vegas
television station. The document itself is a district publication, dated on its
first page.

**Start times for every other district**, collected from district and school
websites by the scraper in `src/scrape_start_times.py`, and by hand into
`data/manual/start_times.csv`.

### A note on years

Nevada Report Card labels a school year by the year it ends in. The Common Core
of Data labels the same year by the year it starts in. So `nrc_year: 2023` and
`ccd_year: 2022` in `config.yaml` refer to the same school year, 2022-2023.
Getting this wrong is the easiest way to silently merge two different years.

### Scraping conduct

The scraper checks robots.txt before requesting any page on a host and skips
anything disallowed, recording the reason in `outputs/robots_decisions.csv`. It
waits between two and three seconds between requests to the same host, and where
a site publishes a longer `Crawl-delay` in robots.txt it waits for that instead.
Several Nevada district sites ask for five seconds, and the scraper honours it.
The user agent names the project and gives a contact address. Every page fetched
is cached to `data/cache/`, so re-running the pipeline does not refetch anything.

Every page visited is logged to `outputs/scrape_log.csv` with its status and the
reason. Bell schedule links that point at a Google Sheet, a PDF, or an off-site
page that the scraper cannot parse are written to `outputs/start_time_leads.csv`
so that a person can open them and fill in the value by hand.

## Method

### Choosing the outcome

The primary outcome is the school average ACT composite score. Nevada
administers the ACT to every grade 11 student in a public school, so a school
average is not distorted by which students chose to sit the test. That makes it
more comparable across schools than a national ACT average would be. Graduation
rate, ELA proficiency, math proficiency, and chronic absenteeism are reported as
secondary outcomes.

### Choosing the poverty control

The obvious control for student poverty is the percentage of students on free or
reduced price lunch, and this study does not use it as the main control.

Clark County School District feeds every student free of charge under the
federal Community Eligibility Provision. Its reported free lunch rate is
therefore 100 percent at every school, wealthy or not. Across this sample the
median reported free lunch rate is about 99.8 percent, which describes a billing
rule rather than Nevada poverty.

The study uses the direct certification rate instead: the share of students
confirmed low income through SNAP, TANF, or foster care records. It is
unaffected by the feeding rule and it varies properly across the sample. The
reported free lunch percentage is kept in the data file so anyone can check this
reasoning for themselves.

### The models

1. The unadjusted correlation between start time and each outcome, reported
   first and clearly labelled as having no controls.
2. A ladder of regressions, adding one block of controls at a time, reporting
   the start time coefficient at every rung:
   - Model 1: no controls
   - Model 2: plus direct certification rate
   - Model 3: plus enrollment and English learner percentage
   - Model 4: plus locale (city, suburb, town, rural)
   - Model 5: plus district fixed effects
3. A stratified comparison: schools split into poverty bands, then early and
   late starters compared within each band. This does not assume the
   relationship is a straight line, so it is a check on the regression rather
   than a repeat of it.
4. A sensitivity check that drops schools whose graduation rate Nevada published
   as a bound such as ">95" rather than an exact number.

Standard errors are heteroskedasticity robust (HC3), because the spread of an
outcome around the fitted line is wider for a school of 200 students than for a
school of 3,000. Every estimate is reported with a sample size and a 95 percent
confidence interval.

## Results

**The short version: this data cannot answer the research question, and the
reason is that Nevada high schools barely differ in when they start.**

### What was collected

117 Nevada high schools meet the sample criteria. 38 of them have a verified
start time, all from the Clark County start and end time document. The page
scraper found no start time it was confident enough to keep, which is covered
below.

Of those 38 schools, 29 start at exactly 7:00 AM. The whole range is 7:00 AM to
8:00 AM, and the standard deviation is 17 minutes. Clark County sets one bell
time for almost all of its comprehensive high schools, and the schools that
differ are mostly magnet and career academies.

That is the finding that governs everything else. A study of whether start time
predicts performance needs schools that start at different times, and this
sample has nine of them.

### The unadjusted correlations

None of the five outcomes shows a raw relationship with start time. Every
confidence interval includes zero.

| Outcome | n | r | 95% CI |
|---|---|---|---|
| ACT composite | 38 | 0.10 | -0.23 to 0.41 |
| Four year graduation rate | 38 | 0.28 | -0.04 to 0.55 |
| ELA proficiency | 38 | 0.03 | -0.29 to 0.35 |
| Math proficiency | 38 | 0.09 | -0.24 to 0.40 |
| Chronic absenteeism | 38 | -0.28 | -0.55 to 0.04 |

### The model ladder

For the ACT composite, the estimated change per hour later start:

| Model | Controls added | Estimate | 95% CI | n |
|---|---|---|---|---|
| 1 | none | +0.80 | -1.54 to 3.13 | 38 |
| 2 | poverty | +2.19 | -0.72 to 5.10 | 38 |
| 3 | plus size and English learners | +3.49 | 1.15 to 5.84 | 36 |
| 4 | plus locale | +2.93 | 0.58 to 5.28 | 36 |
| 5 | plus district fixed effects | not identified | | |

Model 5 cannot be estimated at all. Every school with a known start time is in
Clark County, so there is no between-district variation for district fixed
effects to use. That is reported as not identified rather than quietly repeating
model 4.

The coefficient grows as controls are added rather than collapsing. That is the
opposite of the pattern the study was set up to look for, and it should not be
read as evidence of an effect. Three things argue against taking it seriously:

The size is not believable. The ACT composite has a standard deviation of about
3 points across Nevada high schools, so an estimate of +2.9 to +3.5 points per
hour says that starting an hour later moves a school a full standard deviation.
No study in the sleep literature reports an effect anywhere near that size.

The estimate rests on nine schools. Only nine schools in the sample start
anywhere other than 7:00 AM, and only two start at 8:00 AM. A handful of
observations are carrying the entire slope.

The stratified check cannot be run. No poverty band contains at least two early
and two late schools, so there is no band in which the comparison can be made.
The regression and the stratified comparison were meant to check each other, and
one of the two is unavailable.

### The other outcomes

Math proficiency produces an estimate of +25 percentage points per hour later
start in model 3, on a scale whose sample mean is 23. An estimate that implies
more than doubling a school's proficiency rate is a sign of an unstable model,
not a discovery.

### The censoring sensitivity check

Graduation rate was the one outcome with an unadjusted interval that nearly
excluded zero, at +6.35 points per hour (95% CI 1.17 to 11.52). Eight of the 38
schools have a graduation rate Nevada published as ">95" rather than an exact
number, replaced in the main analysis with 97.5.

Dropping those eight schools moves the unadjusted estimate to +0.25 (95% CI
-34.82 to 35.32, n = 30). The graduation rate result was an artifact of the
midpoint substitution, and it does not survive.

### What the scraper found

The page scraper read 253 pages across 15 district websites, was refused 12
pages by robots.txt, hit 21 errors, and skipped 3 districts whose sites did not
resolve. It produced zero start times it was confident enough to keep. Two reasons, both recorded in `outputs/scrape_log.csv`:

Most Nevada districts run their sites on a platform that builds the school list
in the browser, so the links to individual school sites are not in the HTML the
scraper receives. The crawl reaches district pages and stops there.

Where the scraper did reach a bell schedule, it could not tell which time was
the start of the day. On the Douglas High School schedule it initially reported
6:35 AM, which is an optional zero period. The real first bell is 7:30 AM. The
scraper now requires stronger evidence and returns nothing rather than a
plausible looking wrong number.

Seven links to bell schedules the scraper could not parse are listed in
`outputs/start_time_leads.csv`, including a Carson City Google Sheet and a
Humboldt page titled "All Schools Bell Schedules". Those are the fastest places
to start filling in `data/manual/start_times.csv`.

### Conclusion

This study does not find evidence that later start times go with better academic
performance in Nevada high schools, and it is not able to rule it out either.
The unadjusted correlations are all consistent with zero. The adjusted estimates
point upward but are implausibly large, rest on nine schools, and cannot be
checked against a stratified comparison.

The binding constraint is not the statistics. It is that 29 of the 38 schools
with a known start time start at the same minute. Collecting start times for the
remaining 79 schools, particularly the 16 Washoe County schools and the 22
state charter schools, would be worth more than any further modelling of what is
here.

A note on a near miss during data collection: a news report said Washoe County
approved an 8:30 AM high school start for 2022-2023. Following that up showed
the board reversed the decision a month later and the change was never
implemented. Using the first report would have put a wrong start time on 16
schools, and because those would have been the only non-Clark schools in the
sample, they would have driven the entire result.

## Limitations

**This design cannot establish causation.** Schools were not assigned start
times at random. Controlling for the characteristics measured here removes the
part of the confounding that was measured, and does nothing about the part that
was not. Establishing cause would require either random assignment, which nobody
will do to a school district, or a study following the same schools through an
actual bell schedule change against a comparison group that did not change. This
study does neither.

**The data is cross sectional.** Every school is observed once, in 2022-2023.
There is no before and after. A school that starts late and performs well may
have performed just as well when it started early, and this data cannot tell the
difference.

**The sample is small and concentrated.** Around 117 Nevada high schools meet
the sample criteria, and fewer than that have a known start time. That is a
small sample for a regression carrying several controls, which is why every
estimate here comes with a confidence interval and why several of those
intervals are wide. Clark County alone is roughly half the sample.

**There is very little start time variation to work with.** Clark County sets
the same 7:00 AM bell time at almost all of its comprehensive high schools. When
most of the sample shares one start time, there is little for a regression to
detect, and district fixed effects absorb nearly all of the variation that
remains. This is a limitation of the data, not a finding about sleep.

**Start times are imperfectly measured.** They come from school websites and
district documents, which are not always current and do not always distinguish a
first bell from a warning bell. Some schools run different schedules on
different days. Measurement error in a predictor generally pushes an estimated
relationship toward zero, so these estimates may understate a real relationship
if one exists.

**Some published values are censored.** Nevada publishes ">95" rather than an
exact graduation rate for its highest performing schools. Those are replaced
with the midpoint of the implied range, and a sensitivity check that drops them
is reported alongside.

**The sleep literature is background, not evidence for these findings.** The
studies cited in the motivation used stronger designs than this one. This study
cannot borrow their credibility, and nothing here should be read as confirming
them.

**A null result is a result.** If the confidence intervals include zero once
controls are added, the correct conclusion is that this data does not show a
relationship. That is reported plainly rather than buried.

## How to run it locally

You need Python 3.11 or newer.

```bash
git clone https://github.com/YOUR_USERNAME/school-start-times.git
cd school-start-times

python -m venv .venv
source .venv/bin/activate        # Windows: .venv\Scripts\activate

pip install -r requirements.txt
```

Run the whole pipeline:

```bash
python run_pipeline.py
```

The first run downloads from Nevada Report Card one school at a time with a rate
limit, so it takes several minutes. Everything is cached to `data/`, so later
runs are fast. To skip the website crawl and use only the start times already
collected:

```bash
python run_pipeline.py --skip-scrape
```

Start the dashboard:

```bash
streamlit run streamlit_app.py
```

### Filling in missing start times

`data/manual/start_times.csv` has one row per school in the sample. Where the
`start_time` column is blank, no start time was found. Fill it in as `7:45 AM`
and re-run `python run_pipeline.py --skip-scrape`. Hand entered values always
win over anything collected automatically.

`outputs/start_time_leads.csv` lists bell schedule pages the scraper found but
could not read, which is the fastest place to start.

## Repository layout

```
config.yaml              every tunable choice: year, sample rules, models
run_pipeline.py          runs all stages in order
streamlit_app.py         the dashboard

src/
  config.py                    paths and config loading
  polite.py                    robots.txt, rate limiting, page caching
  fetch_ccd.py                 federal school directory
  fetch_nevada_report_card.py  state accountability data
  fetch_bell_documents.py      district published bell schedule PDFs
  scrape_start_times.py        crawls school websites for bell schedules
  clean.py                     typing, renaming, suppressed value handling
  merge.py                     joins the sources, applies the sample filter
  analyze.py                   correlations, model ladder, stratification
  charts.py                    the four charts

data/
  raw/        untouched downloads
  cache/      cached web pages
  interim/    one tidy table per source
  processed/  the merged analysis table
  manual/     hand entered start times, tracked in git

outputs/      charts, result tables, and the scraping logs
```

## Deploying the dashboard

See `docs/DEPLOY.md`.
