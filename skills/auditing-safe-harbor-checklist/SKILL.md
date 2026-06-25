---
name: auditing-safe-harbor-checklist
description: "Verify OpenMed de-identified output against all 18 HIPAA Safe Harbor identifier categories and report residual re-identification risk. Use when the user must confirm a note meets HIPAA Safe Harbor (45 CFR 164.514(b)(2)), needs a coverage checklist mapping detected entities to the 18 categories, wants to flag gaps like ages over 89, rare geography, fax vs phone, or biometrics, or asks whether masking was complete. Maps OpenMed CANONICAL_LABELS to the 18 HIPAA classes and uses extract_pii / deidentify to check coverage. Pairs with OpenMed deidentifying-clinical-text and auditing-deidentification-runs."
license: Apache-2.0
metadata:
  project: OpenMed
  category: de-identification
  pairs: after
  version: "1.0"
---

# Auditing against the HIPAA Safe Harbor checklist

The Safe Harbor method (45 CFR 164.514(b)(2)) de-identifies PHI by removing
**18 specific identifier categories** for the individual and their relatives,
employers, and household members — and requires the covered entity to have **no
actual knowledge** that the remainder could re-identify anyone. This skill turns
that legal checklist into a concrete coverage check over OpenMed output: which
of the 18 categories were detected and handled, and where the gaps are.

The full mapping table lives in
[references/safe-harbor-identifiers.md](references/safe-harbor-identifiers.md) —
all 18 categories, their OpenMed HIPAA class, the matching `CANONICAL_LABELS`,
and per-category cautions. Read it when you need the authoritative cross-walk.

## When to use this skill

Use it after a de-identification run to *prove* coverage, or before release to
decide whether Safe Harbor is even achievable for this text. If the user needs a
signed, retained record of the run, hand off to `auditing-deidentification-runs`.

## Quick start: coverage check

```python
import openmed
from openmed.core.labels import LABEL_TO_HIPAA, HIPAA_SAFE_HARBOR_CLASSES

note = (
    "Patient John Doe (MRN 1234567), age 92, of Smalltown, seen 2024-03-02. "
    "SSN 123-45-6789, phone 617-555-0142."
)

# 1) Detect identifiers (spans only; no rewrite).
detected = openmed.extract_pii(note)

# 2) Roll each detected span up to its HIPAA Safe Harbor class.
covered = set()
for ent in detected.entities:
    canonical = openmed.normalize_label(ent.label)        # -> CANONICAL_LABELS form
    hipaa_class = LABEL_TO_HIPAA.get(canonical)            # -> one of 18 classes
    if hipaa_class:
        covered.add(hipaa_class)

# 3) Report which of the 18 classes were touched and which weren't observed.
missing = sorted(HIPAA_SAFE_HARBOR_CLASSES - covered)
print("covered:", sorted(covered))
print("not observed in this note:", missing)
```

"Not observed" is **not** the same as "absent" — a category may simply not occur
in this note, or may have been missed. That is exactly what the human review
step (below) is for.

## Workflow

1. **De-identify** with a Safe Harbor profile:
   `openmed.deidentify(note, policy="hipaa_safe_harbor")`. This masks every
   identifier class by default and runs the mandatory structured-ID safety sweep.
2. **Map detected spans to the 18 classes** via `LABEL_TO_HIPAA` (as above).
   Build a table of category → detected? → action taken.
3. **Walk the checklist** in
   [references/safe-harbor-identifiers.md](references/safe-harbor-identifiers.md)
   and flag the known gaps explicitly:
   - **Ages > 89** (`AGE`) must be aggregated to "90+"; OpenMed flags but does
     not auto-cap — see `shifting-clinical-dates`.
   - **Dates** keep only the year; everything else (admit/discharge/DOB) goes.
   - **ZIP** beyond the first 3 digits, and small-population areas → mask whole.
   - **Rare geography** (small towns) and **rare characteristics** (unusual
     occupation) can re-identify even when masked field-by-field.
   - **Fax** shares the `PHONE` label; **biometrics** and **full-face photos**
     are out of scope for text — handle in the imaging/intake pipeline.
4. **Assess residual risk.** Run `audit=True` and read `residual_risk`
   (`auditing-deidentification-runs`). Non-zero projected leakage → review.
5. **Record the "no actual knowledge" judgment.** A human must sign off that the
   remaining text cannot re-identify the individual. Automated coverage is
   necessary, not sufficient.

## Hand-off to / from OpenMed

- **Detect / de-id:** `openmed.extract_pii` (spans) and `openmed.deidentify`
  (rewrite) — see `deidentifying-clinical-text`.
- **Label mapping:** `openmed.CANONICAL_LABELS`, `openmed.normalize_label`, and
  `LABEL_TO_HIPAA` / `HIPAA_SAFE_HARBOR_CLASSES` in `openmed/core/labels.py`.
- **Signed record + residual risk:** `auditing-deidentification-runs`
  (`audit=True` → `AuditReport.residual_risk`).
- **Profile choice:** `configuring-privacy-policies` — if you must keep dates or
  geography, Safe Harbor fails; use Expert Determination
  (`hipaa_expert_review_assist`) or a Limited Data Set
  (`research_limited_dataset`).

## Edge cases & gotchas

- **Coverage ≠ compliance.** Detecting all 18 categories does not satisfy Safe
  Harbor on its own — the "no actual knowledge" residual-risk judgment is
  required and is a human decision.
- **Ages over 89 are a transformation, not a detection.** Masking the digits is
  fine; if you keep age, aggregate to "90+". OpenMed will not cap automatically.
- **ZIP / date rules are transformations.** Safe Harbor permits keeping
  3-digit ZIP (population-gated) and the year — implement the truncation; do not
  assume detection handles it.
- **Some categories have no text label** (biometrics, full-face photos). Mark
  them N/A for text and ensure another pipeline stage covers them.
- **Combination re-identification.** Several non-identifying quasi-identifiers
  together (rare diagnosis + small town + outlier age) can identify someone;
  this is precisely why `strict_no_leak` exists for high-stakes data.
- **No raw PHI in the checklist output** — report categories, counts, offsets,
  and hashes, never the underlying identifiers.

## Standards & references

- 45 CFR 164.514(b)(2) (Safe Harbor): https://www.ecfr.gov/current/title-45/subtitle-A/subchapter-C/part-164/subpart-E/section-164.514
- HHS de-identification guidance (incl. the 18 identifiers and "actual
  knowledge"): https://www.hhs.gov/hipaa/for-professionals/privacy/special-topics/de-identification/index.html
- Local checklist: [references/safe-harbor-identifiers.md](references/safe-harbor-identifiers.md)
- OpenMed source: `openmed/core/labels.py` (`LABEL_TO_HIPAA`,
  `HIPAA_SAFE_HARBOR_CLASSES`, `CANONICAL_LABELS`, `normalize_label`).
