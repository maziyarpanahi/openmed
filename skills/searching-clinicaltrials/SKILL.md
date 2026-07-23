---
name: searching-clinicaltrials
description: "Searches ClinicalTrials.gov for studies by condition, intervention, and recruitment status using the modern v2 REST API with cursor (pageToken) pagination. Use when the user wants to find trials for a diagnosis or drug, screen patients against open studies, build a trial-matching feature, or pull a trial corpus for analysis. Trigger keywords: clinical trial, ClinicalTrials.gov, NCT number, trial search, recruiting studies, eligibility, query.cond, query.intr, pageToken, v2 API. Pairs adjacent to OpenMed: take Disease/Pharmaceutical entities from openmed.analyze_text and turn them into query.cond / query.intr filters; the returned eligibility text feeds parsing-trial-eligibility. ClinicalTrials.gov API v2 is fully public — no API key, no license."
license: Apache-2.0
metadata:
  project: OpenMed
  category: research-genomics
  pairs: adjacent
  version: "1.0"
---

# Searching ClinicalTrials.gov (v2 REST API)

Query **ClinicalTrials.gov** — the U.S. registry of clinical studies — for trials
matching a condition, intervention, and recruitment status. This skill uses the
**modern v2 REST API** (`/api/v2/studies`), which returns structured JSON and
paginates with an opaque cursor (`pageToken`), not page numbers.

The v2 API is **fully public**: no API key, no registration, no license barrier.
The legacy v1/classic API and the older `query_term`-style endpoints are
deprecated — do not build on them.

## When to use

- OpenMed extracted a diagnosis ("metastatic colorectal cancer") or a drug
  ("pembrolizumab") and you want open trials for it.
- You are building a patient-to-trial matching feature and need candidate studies
  before applying eligibility logic (`parsing-trial-eligibility`).
- You need a corpus of trial records (eligibility text, outcomes) to feed back
  into `openmed.analyze_text` for biomedical NER.

If you already have an NCT number, fetch the single study directly
(`/api/v2/studies/NCT01234567`) instead of searching.

## Quick start (real v2 API call)

Base URL: `https://clinicaltrials.gov/api/v2`. No auth. JSON by default.

```python
import requests

BASE = "https://clinicaltrials.gov/api/v2"

def search_trials(condition: str, intervention: str | None = None,
                  status: str = "RECRUITING", page_size: int = 50) -> dict:
    """One page of studies for a condition (+ optional intervention)."""
    params = {
        "query.cond": condition,            # condition / disease search
        "filter.overallStatus": status,     # comma-separated enum values
        "pageSize": min(page_size, 1000),   # max 1000; default 10
        "countTotal": "true",               # include totalCount on first page
        "format": "json",
    }
    if intervention:
        params["query.intr"] = intervention  # drug / intervention search
    r = requests.get(f"{BASE}/studies", params=params, timeout=30)
    r.raise_for_status()
    return r.json()

data = search_trials("breast cancer", intervention="trastuzumab")
print(data["totalCount"])                    # total matches (first page only)
for study in data["studies"]:
    ps = study["protocolSection"]
    nct = ps["identificationModule"]["nctId"]
    title = ps["identificationModule"]["briefTitle"]
    print(nct, "-", title)
```

Equivalent cURL:

```bash
curl "https://clinicaltrials.gov/api/v2/studies?query.cond=breast+cancer\
&query.intr=trastuzumab&filter.overallStatus=RECRUITING&pageSize=50&format=json"
```

## Response shape

Top level: `studies` (array), `nextPageToken` (present only if more results),
and `totalCount` (only when `countTotal=true`, on the first page). Each study is
a `protocolSection` of typed modules:

| Field path | Meaning |
| --- | --- |
| `identificationModule.nctId` | `NCT........` study id |
| `identificationModule.briefTitle` | short title |
| `statusModule.overallStatus` | `RECRUITING`, `COMPLETED`, … |
| `conditionsModule.conditions` | list of condition strings |
| `armsInterventionsModule.interventions` | drugs / procedures |
| `eligibilityModule.eligibilityCriteria` | free-text inclusion/exclusion |
| `eligibilityModule.sex` / `minimumAge` / `maximumAge` | demographic gates |
| `contactsLocationsModule.locations` | recruiting sites |

## Cursor pagination

There are **no page numbers**. Loop until `nextPageToken` is absent. The token is
opaque — pass it back verbatim. Do not re-send `countTotal` after page 1.

```python
def iter_all(condition: str, status: str = "RECRUITING"):
    params = {"query.cond": condition, "filter.overallStatus": status,
              "pageSize": 1000, "format": "json"}
    while True:
        r = requests.get(f"{BASE}/studies", params=params, timeout=30)
        r.raise_for_status()
        page = r.json()
        yield from page.get("studies", [])
        token = page.get("nextPageToken")
        if not token:
            break
        params["pageToken"] = token        # cursor for the next page
```

### Trimming payloads

Default responses are large. Restrict to the fields you need with `fields` (dotted
paths or module names) to cut bandwidth:

```python
params["fields"] = ("NCTId,BriefTitle,OverallStatus,"
                    "Condition,EligibilityCriteria")
```

## Workflow

1. **Build the query from OpenMed facts.** Map extracted Disease spans →
   `query.cond`; Pharmaceutical spans → `query.intr`. Free-text keywords go in
   `query.term`. Combine status filters as `filter.overallStatus=RECRUITING,NOT_YET_RECRUITING`.
2. **Page through** with the cursor until `nextPageToken` is gone; cap total pulls.
3. **Persist** `nctId`, status, conditions, interventions, and the raw eligibility
   text. Eligibility goes to `parsing-trial-eligibility`.
4. **Optionally re-NER** the eligibility / outcomes text with
   `openmed.analyze_text` to structure inclusion criteria.

## Hand-off to / from OpenMed

- **From OpenMed → trial search.** `openmed.analyze_text(note, model_name="disease_detection_superclinical")`
  yields Disease and Pharmaceutical entities. Use the surface forms (or a grounded
  term from `coding-icd10` / `normalizing-rxnorm`) as `query.cond` / `query.intr`.
- **Trial text → OpenMed.** Feed `eligibilityModule.eligibilityCriteria` and brief
  summaries back through `openmed.analyze_text` to extract conditions, meds, and
  labs mentioned in the criteria. Then hand to `parsing-trial-eligibility` for
  inclusion/exclusion matching against patient facts.
- Keep patient data local. The API call carries only the **query terms**
  (condition/drug names), never the patient note or any PHI.

## Edge cases & gotchas

- **Synonyms & spelling.** The condition matcher is fuzzy but not infinite —
  "MI" will not match "myocardial infarction". Normalize OpenMed output first
  (ICD-10 / RxNorm) and consider issuing a few synonym variants.
- **Status enums are exact.** Valid values include `RECRUITING`,
  `NOT_YET_RECRUITING`, `ENROLLING_BY_INVITATION`, `ACTIVE_NOT_RECRUITING`,
  `COMPLETED`, `SUSPENDED`, `TERMINATED`, `WITHDRAWN`, `UNKNOWN`. Comma-separate;
  do not lowercase.
- **`totalCount` is first-page only.** Request `countTotal=true` once; it is not
  repeated on subsequent pages.
- **Page size cap is 1000.** Larger values are silently clamped.
- **Rate limits.** No key required, but throttle politely (a short sleep between
  pages); aggressive scraping can be blocked. For bulk/offline work, consider the
  full registry data dump rather than thousands of paged calls.
- **`pageToken` expires** if the underlying index shifts; restart the query if a
  token is rejected.
- **Not medical advice.** A trial appearing in results does not mean the patient
  qualifies — eligibility is decided downstream and reviewed by a clinician.

## Standards & references

- ClinicalTrials.gov API v2 — https://clinicaltrials.gov/data-api/api
- Study data structure (modules / field paths) —
  https://clinicaltrials.gov/data-api/about-api/study-data-structure
- Search areas & query syntax —
  https://clinicaltrials.gov/data-api/about-api/search-areas
- OpenAPI / interactive reference — https://clinicaltrials.gov/api/v2/
