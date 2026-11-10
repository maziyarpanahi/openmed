---
name: detecting-pv-signals
description: "Computes disproportionality signals — PRR, ROR, EBGM, and IC (BCPNN) — over FAERS / OpenFDA drug-event data to flag potential safety signals. Use when the user wants to mine spontaneous-report data for drug-reaction associations, build a 2x2 contingency table, compute a Proportional Reporting Ratio or Reporting Odds Ratio, run Empirical Bayes (EBGM/EB05) or Information Component shrinkage, or screen a drug for over-reported reactions. Trigger keywords: disproportionality, signal detection, PRR, ROR, EBGM, EB05, IC, BCPNN, MGPS, 2x2 table, signal of disproportionate reporting, SDR, OpenFDA, FAERS. Pairs adjacent to OpenMed: aggregate de-identified, coded cases (from reporting-adverse-events) then query the public OpenFDA /drug/event count API to build the contingency table. Reaction terms are MedDRA PTs (licensed, user-supplied)."
license: Apache-2.0
metadata:
  project: OpenMed
  category: safety-pharmacovigilance
  pairs: adjacent
  version: "1.0"
---

# Detecting pharmacovigilance signals (disproportionality)

Spontaneous-report databases like the FDA's **FAERS** are mined for
**signals of disproportionate reporting (SDR)**: drug-reaction pairs that occur
together *more than expected* given the background of all reports. The core
device is a **2x2 contingency table** and a disproportionality metric computed
from it — **PRR**, **ROR**, **EBGM**, or **IC (BCPNN)**.

You can build the 2x2 table directly from the **public, free OpenFDA**
`/drug/event` endpoint (no PHI, no MedDRA license to *query*; the reaction terms
returned are already MedDRA PTs). This skill is **statistical screening**: a high
PRR is a *hypothesis*, not a confirmed adverse drug reaction.

## When to use

- You have a drug of interest and want to see which reactions are over-reported.
- You need a PRR / ROR with confidence interval, or an Empirical Bayes EBGM/EB05
  / IC025 to control for the small-count noise PRR/ROR suffer from.
- You are building a routine signal-screening run over OpenFDA or your own
  aggregated case counts.

## The 2x2 table

For one drug D and one reaction R, classify every report:

|            | Reaction R | Not R |
| ---------- | ---------- | ----- |
| Drug D     | **a**      | **b** |
| Not D      | **c**      | **d** |

- **PRR** = [a/(a+b)] / [c/(c+d)]
- **ROR** = (a·d)/(b·c)
- **IC** (BCPNN, log2 information component) ≈ log2( a·(a+b+c+d) / ((a+b)·(a+c)) )
- **EBGM** = Empirical Bayes Geometric Mean — a gamma-Poisson *shrinkage* of the
  observed/expected ratio (the MGPS method) that pulls small-count estimates
  toward 1; report **EB05** (the 5th percentile) as the conservative signal.

Common signal thresholds (screening only): PRR ≥ 2 with χ² ≥ 4 and a ≥ 3; ROR
lower 95% CI > 1; **IC025 > 0**; **EB05 ≥ 2**.

## Quick start (real OpenFDA count queries)

Base endpoint: `https://api.fda.gov/drug/event.json`. No key needed to try it
(240 req/min, 1,000/day per IP; with a free `api_key=` key: 240/min,
120,000/day). The `count=<field>.exact` parameter returns a terms histogram, and
`search=` with `+AND+` filters the population — that is all you need for a 2x2.

```python
import requests

BASE = "https://api.fda.gov/drug/event.json"

def fda_count(search: str | None, count_field: str) -> int:
    """Total reports matching `search` (sum of the .exact histogram)."""
    params = {"count": count_field}
    if search:
        params["search"] = search
    r = requests.get(BASE, params=params, timeout=30)
    if r.status_code == 404:        # OpenFDA returns 404 for an empty result set
        return 0
    r.raise_for_status()
    return sum(row["count"] for row in r.json()["results"])

def cell_count(search: str | None) -> int:
    """Number of reports matching `search` (use meta.results.total via limit=1)."""
    params = {"limit": 1}
    if search:
        params["search"] = search
    r = requests.get(BASE, params=params, timeout=30)
    if r.status_code == 404:
        return 0
    r.raise_for_status()
    return r.json()["meta"]["results"]["total"]

# Build the 2x2 for warfarin x "gastrointestinal haemorrhage".
DRUG = 'patient.drug.openfda.generic_name:"warfarin"'
RXN  = 'patient.reaction.reactionmeddrapt.exact:"gastrointestinal haemorrhage"'

a = cell_count(f"{DRUG}+AND+{RXN}")          # drug & reaction
b = cell_count(DRUG) - a                      # drug, not reaction
c = cell_count(RXN) - a                       # reaction, not drug
N = cell_count(None)                          # total reports in FAERS
d = N - a - b - c
```

Compute the metrics from `(a, b, c, d)`:

```python
import math

def prr(a, b, c, d):
    return (a / (a + b)) / (c / (c + d))

def ror(a, b, c, d):
    return (a * d) / (b * c)

def ror_ci(a, b, c, d):
    lnror = math.log((a * d) / (b * c))
    se = math.sqrt(1/a + 1/b + 1/c + 1/d)     # Woolf's method
    lo, hi = math.exp(lnror - 1.96 * se), math.exp(lnror + 1.96 * se)
    return lo, hi

def ic(a, b, c, d):
    n = a + b + c + d
    expected = (a + b) * (a + c) / n
    return math.log2(a / expected) if a and expected else float("nan")

print("PRR", round(prr(a, b, c, d), 2))
print("ROR", round(ror(a, b, c, d), 2), "95% CI", ror_ci(a, b, c, d))
print("IC",  round(ic(a, b, c, d), 2))
```

For **EBGM / EB05** use a maintained Empirical Bayes implementation (e.g. the
`openEBGM` R package or `PhViD` in R) on the same `(a, b, c, d)` rather than
hand-rolling the gamma-Poisson MGPS shrinkage — the shrinkage prior is the whole
point and easy to get wrong.

## Workflow

1. **Pick the population.** Decide your denominator: all of FAERS, or a
   restricted background (e.g. one drug class, one year via
   `receivedate:[20230101+TO+20231231]`). The choice of `c`/`d` defines the
   "expected".
2. **Resolve the drug field.** Prefer `patient.drug.openfda.generic_name` (RxNorm
   ingredient-normalized) over the free-text `medicinalproduct` to avoid brand
   fragmentation. Restrict to suspect drugs with
   `patient.drug.drugcharacterization:1` if you want suspect-only signals.
2. **Use `.exact`** for the reaction field so "injection site reaction" counts as
   one phrase, not three words: `patient.reaction.reactionmeddrapt.exact`.
3. **Build the 2x2** with the cell counts above. Verify `a + b + c + d == N`.
4. **Compute PRR and ROR with CIs**; add **IC025** / **EB05** for small counts.
5. **Apply thresholds** (e.g. PRR ≥ 2, χ² ≥ 4, a ≥ 3) — but treat them as a
   *triage filter*, not a verdict.
6. **Hand flagged pairs to a safety scientist** for medical review, confounder
   assessment, and labeling/expectedness checks.

## Hand-off to / from OpenMed

- **From** `reporting-adverse-events`: your own coded, de-identified ICSRs give
  internal counts you can use *instead of* or *alongside* OpenFDA — the same
  2x2 math applies. Aggregate only counts; never put narrative PHI in the table.
- **From** `normalizing-rxnorm`: normalize the drug name to an RxNorm ingredient
  before querying so brand/generic synonyms collapse to one cell.
- **To** `querying-openfda-labels`: for every signal, check whether the reaction
  is already on the label (expected) via `/drug/label`. **To**
  `reporting-adverse-events`: a confirmed signal may require expedited reporting.
- OpenMed runs NER/de-id **on-device**; only de-identified drug/reaction *codes*
  (no PHI) are sent to OpenFDA.

## Edge cases & gotchas

- **Disproportionality ≠ causality.** A high PRR reflects reporting patterns,
  notoriety bias, and indication confounding — not a proven causal link.
- **Small counts break PRR/ROR.** With `a < 3` the ratios are unstable and CIs
  explode. This is exactly why **EBGM/EB05** and **IC025** (shrinkage) exist —
  prefer them for rare events.
- **OpenFDA is a sample, not all of FAERS, and is not deduplicated** the way the
  curated FAERS quarterly files are. Use it for screening; reproduce confirmed
  signals against the official FAERS extracts.
- **`.exact` is mandatory for counting phrases.** Without it, OpenFDA tokenizes
  the reaction and your counts are wrong.
- **OpenFDA returns HTTP 404 for an empty result set** (not an empty list) — the
  helpers above treat 404 as zero. Respect the rate limits; register a free key
  for routine runs.
- **MedDRA versioning.** OpenFDA reaction terms are MedDRA PTs at FDA's coding
  version; if you join to your own MedDRA-coded cases, align the version. MedDRA
  itself is licensed — you query OpenFDA's already-coded terms, you do not need a
  MedDRA license to read them, but you do to code your own cases.

## Standards & references

- OpenFDA drug adverse event API: https://open.fda.gov/apis/drug/event/
- OpenFDA query syntax (`count`, `.exact`, `search` AND/OR): https://open.fda.gov/apis/query-syntax/
- OpenFDA authentication & rate limits: https://open.fda.gov/apis/authentication/
- Evans et al., PRR for signal generation (Pharmacoepidemiol Drug Saf, 2001): https://pubmed.ncbi.nlm.nih.gov/11828828/
- Bate et al., BCPNN / Information Component (Eur J Clin Pharmacol, 1998): https://pubmed.ncbi.nlm.nih.gov/9696956/
- DuMouchel, Empirical Bayes / MGPS (EBGM): https://www.tandfonline.com/doi/abs/10.1080/00031305.1999.10474456
- CIOMS VIII — Practical Aspects of Signal Detection: https://cioms.ch/publications/product/practical-aspects-of-signal-detection-in-pharmacovigilance/
