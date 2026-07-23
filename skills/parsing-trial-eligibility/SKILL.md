---
name: parsing-trial-eligibility
description: "Parses free-text clinical-trial eligibility criteria into structured inclusion and exclusion logic, then matches them against patient facts that OpenMed extracted. Use when the user wants to turn a ClinicalTrials.gov eligibility block into machine-readable rules, screen a synthetic patient for trial fit, or explain why a patient does or does not meet criteria. Trigger keywords: eligibility criteria, inclusion, exclusion, trial matching, patient screening, criteria parsing, eligibilityModule, age/sex gates. Pairs after OpenMed and after searching-clinicaltrials: consume the eligibilityModule text from a study, structure it, and match against conditions, medications, labs, and demographics from openmed.analyze_text. Decision-support only — never autonomous enrollment."
license: Apache-2.0
metadata:
  project: OpenMed
  category: research-genomics
  pairs: after
  version: "1.0"
---

# Parsing trial eligibility & matching patients

A ClinicalTrials.gov study exposes its eligibility as a single free-text block
(`protocolSection.eligibilityModule.eligibilityCriteria`) plus a few typed fields
(`sex`, `minimumAge`, `maximumAge`, `healthyVolunteers`). This skill turns that
prose into **structured inclusion / exclusion criteria** and matches each rule
against patient facts that **OpenMed** extracted — producing an explainable
`eligible | ineligible | unknown` verdict per criterion.

This is **decision support, not enrollment**. The output is a candidate list and
a rationale for a clinician to review, never an automated eligibility decision.

## When to use

- You pulled a study with `searching-clinicaltrials` and need its eligibility as
  machine-readable rules.
- You have a (synthetic) patient profile and want to screen it against one or many
  trials, with a per-criterion reason.
- You want to highlight which patient facts are missing to decide a criterion.

## Quick start

The typed gates are deterministic — apply them first. The free-text criteria need
parsing into bullet-level inclusion/exclusion items.

```python
# Study from ClinicalTrials.gov v2 (see searching-clinicaltrials)
elig = study["protocolSection"]["eligibilityModule"]

raw = elig["eligibilityCriteria"]            # free text, often markdown bullets
sex = elig.get("sex", "ALL")                 # ALL | FEMALE | MALE
min_age = elig.get("minimumAge")             # e.g. "18 Years"
max_age = elig.get("maximumAge")             # e.g. "75 Years"
healthy_ok = elig.get("healthyVolunteers")   # bool

def split_criteria(text: str) -> dict[str, list[str]]:
    """Split the prose into inclusion / exclusion bullet lists."""
    sections, current = {"inclusion": [], "exclusion": []}, None
    for line in text.splitlines():
        low = line.strip().lower()
        if "inclusion criteria" in low:
            current = "inclusion"; continue
        if "exclusion criteria" in low:
            current = "exclusion"; continue
        bullet = line.strip(" -*•\t")
        if bullet and current:
            sections[current].append(bullet)
    return sections

criteria = split_criteria(raw)
```

Each bullet is a candidate rule. Structure it into a comparable predicate:
condition present/absent, lab threshold, age/sex, prior-therapy, performance
status (e.g. ECOG ≤ 2), pregnancy status, etc.

```python
from dataclasses import dataclass

@dataclass
class Criterion:
    kind: str            # "condition" | "lab" | "age" | "sex" | "medication" | "other"
    polarity: str        # "include" | "exclude"
    text: str            # original bullet
    target: str | None   # e.g. "ECOG", "diabetes", "metformin"
    op: str | None = None  # "<=", ">=", "==", "present", "absent"
    value: float | str | None = None
```

## Matching against OpenMed-extracted patient facts

Build the patient profile from `openmed.analyze_text` outputs plus structured
demographics, then evaluate each criterion to a three-valued result.

```python
patient = {
    "age": 61, "sex": "FEMALE",
    "conditions": {"type 2 diabetes", "hypertension"},   # OpenMed Disease spans
    "medications": {"metformin", "lisinopril"},          # OpenMed Pharmaceutical
    "labs": {"hba1c": 8.1, "ecog": 1},                   # from a labs extractor
}

def evaluate(c: Criterion, p: dict) -> str:
    if c.kind == "sex" and c.target:
        return "pass" if p["sex"] == c.target or c.target == "ALL" else "fail"
    if c.kind == "condition" and c.target:
        has = c.target.lower() in {x.lower() for x in p["conditions"]}
        ok = has if c.polarity == "include" else not has
        return "pass" if ok else "fail"
    if c.kind == "lab" and c.target and c.target.lower() in p["labs"]:
        v = p["labs"][c.target.lower()]
        cmp = {"<=": v <= c.value, ">=": v >= c.value, "==": v == c.value}
        return "pass" if cmp.get(c.op, False) else "fail"
    return "unknown"   # fact not present → needs human review, never assume pass
```

Aggregate: a patient is a **candidate** only if every inclusion criterion is
`pass` (or `unknown`, flagged) and every exclusion criterion is not `fail`.
Surface the `unknown` items prominently — missing data is the most common reason a
real screen needs a human.

## Workflow

1. Apply the **typed gates** (`sex`, `minimumAge`, `maximumAge`) — cheap, exact.
2. **Split** the free text into inclusion / exclusion bullets.
3. **Structure** each bullet into a `Criterion` (kind, polarity, target, op,
   value). NER on the bullet via `openmed.analyze_text` finds the condition / drug
   / lab targets; numeric thresholds come from a regex/units pass.
4. **Evaluate** each criterion against the OpenMed-derived patient profile to
   `pass | fail | unknown`.
5. **Report** a verdict with a per-criterion rationale and an explicit list of
   `unknown` facts that block a confident decision.

## Hand-off to / from OpenMed

- **From OpenMed (patient side).** Run `openmed.analyze_text` over the patient note
  to populate `conditions` (Disease), `medications` (Pharmaceutical), and oncology
  context; normalize via `coding-icd10` / `normalizing-rxnorm` so comparisons are
  code-based, not string-based.
- **From OpenMed (trial side).** Run `openmed.analyze_text` over each eligibility
  bullet to identify the condition / drug / lab the rule references, improving
  `target` extraction beyond keyword spotting.
- **From searching-clinicaltrials.** Studies arrive with their `eligibilityModule`
  already populated — this skill is the next stage.
- Keep everything **local**: matching runs on-device against the patient profile;
  no PHI leaves the process. Examples here use a **synthetic** patient.

## Edge cases & gotchas

- **Three-valued logic is mandatory.** Treating `unknown` as `pass` enrolls
  ineligible patients; treating it as `fail` drops eligible ones. Surface it.
- **Negation & temporality.** "No prior chemotherapy" vs "prior chemotherapy"
  flips polarity; "active infection" vs "history of infection" differs in time.
  Use `openmed.clinical` (see `resolving-clinical-context`) so negated/historical
  mentions are not counted as present.
- **Units & ranges.** "Creatinine clearance ≥ 60 mL/min", "platelets > 100,000/µL"
  — normalize units before comparing; LOINC grounding (`mapping-loinc`) helps.
- **Compound bullets.** One sentence may carry several predicates ("age 18-75 and
  ECOG 0-1"). Split into atomic criteria.
- **Inconsistent headings.** Some studies omit explicit "Inclusion/Exclusion"
  labels or use "Key Inclusion Criteria". Default unlabeled bullets to inclusion
  and flag for review.
- **Not a medical device.** Output is a ranked candidate list with rationale for a
  clinician — never an autonomous enrollment or exclusion decision.

## Standards & references

- ClinicalTrials.gov study structure (eligibilityModule) —
  https://clinicaltrials.gov/data-api/about-api/study-data-structure
- Protocol Registration eligibility data definitions —
  https://clinicaltrials.gov/policy/protocol-definitions
- Common Data Element: eligibility criteria —
  https://clinicaltrials.gov/data-api/about-api/study-data-structure#eligibilityModule
- OpenMed clinical context (negation/temporality) — `resolving-clinical-context`
