---
name: shifting-clinical-dates
description: "Apply consistent per-patient date shifting in OpenMed that preserves intervals between events while satisfying HIPAA Safe Harbor's date rule. Use when the user needs to de-identify dates but keep temporal structure for research, shift all dates by the same offset per patient, preserve days-between-events for survival or longitudinal analysis, cap ages over 89, or strip everything but the year. Covers deidentify(method=\"shift_dates\", date_shift_days=..., keep_year=...) and per-patient reproducible offsets via consistent=True, seed=.... Pairs with OpenMed deidentifying-clinical-text and auditing-safe-harbor-checklist."
license: Apache-2.0
metadata:
  project: OpenMed
  category: de-identification
  pairs: after
  version: "1.0"
---

# Shifting clinical dates

HIPAA Safe Harbor forbids keeping dates more specific than the year. But naively
deleting dates destroys the temporal structure research depends on — time to
event, length of stay, intervals between visits. **Date shifting** is the
compromise: move every date by a single random offset so the *absolute* dates
become meaningless while the *intervals* between them are preserved exactly.
OpenMed does this on-device with `deidentify(method="shift_dates", ...)`.

## When to use this skill

Use it when downstream analysis needs temporal relationships (survival curves,
sepsis-to-antibiotic time, readmission gaps) but the calendar dates must be
de-identified. If you can throw dates away entirely, plain `method="mask"` is
simpler — reach for shifting only when intervals matter.

## Quick start

```python
import openmed

note = (
    "Admitted 2024-03-02, started antibiotics 2024-03-04, discharged 2024-03-09. "
    "Follow-up scheduled 2024-04-02."
)

# Shift every date by the SAME offset -> intervals preserved, dates obscured.
result = openmed.deidentify(
    note,
    method="shift_dates",
    consistent=True,        # one stable offset for this run
    seed=20240519,          # reproducible per-patient offset (use a per-patient key)
    keep_year=False,        # do NOT retain the year (Safe Harbor: year-only is the max)
)
print(result.deidentified_text)
# Admit -> antibiotics is still 2 days; admit -> discharge still 7 days; etc.
```

## How interval preservation works

All dates in the document are moved by **one** offset (auto-selected as a random
non-zero value in roughly ±1 year, or fixed with `date_shift_days=`). Because
the offset is identical for every date, the difference between any two dates is
unchanged:

```
real:    Mar 2  ──2d──▶ Mar 4  ──5d──▶ Mar 9
shifted: Jul 18 ──2d──▶ Jul 20 ──5d──▶ Jul 25     (offset = +138 days, intervals intact)
```

That is why survival time, length of stay, and visit gaps survive
de-identification while the actual calendar is destroyed.

## Per-patient consistent offsets

Each patient should get their **own** offset, and that offset should be
**stable** across documents and reproducible across runs. Derive a per-patient
seed (e.g. from a secret keyed hash of the patient ID — never the raw MRN) and
pass it as `seed=`:

```python
def patient_offset_seed(patient_key: str) -> int:
    import hashlib, hmac
    # keyed so the mapping from patient -> offset is itself a secret
    digest = hmac.new(b"<vault-secret>", patient_key.encode(), hashlib.sha256).digest()
    return int.from_bytes(digest[:8], "big")

for doc in patient_documents:
    openmed.deidentify(
        doc, method="shift_dates",
        consistent=True, seed=patient_offset_seed(patient_id),
        keep_year=False,
    )
```

Same patient → same offset everywhere (their notes stay internally consistent);
different patients → different offsets (cross-patient dates cannot be aligned).

## Workflow

1. **Decide the offset policy.** Per-patient is standard for clinical research.
   Use `consistent=True` + a per-patient `seed`. Use a fixed `date_shift_days=`
   only when a deterministic, externally-managed offset is required.
2. **Set `keep_year=False`** for Safe Harbor. (`keep_year=True` retains the year,
   which is permissible only if dates aren't tied to an individual's care.)
3. **Cap ages over 89 separately.** Date shifting moves dates; it does **not**
   aggregate ages. Safe Harbor requires ages >89 and any date implying age >89
   to collapse to a single "90+" — handle `AGE` spans explicitly
   (`auditing-safe-harbor-checklist`).
4. **Keep names/IDs handled too.** `shift_dates` only touches dates. Run a normal
   redaction pass (or a policy) for `PERSON`, `ID_NUM`, etc.
5. **Verify** intervals are preserved and no calendar leaked via `audit=True`.

## Hand-off to / from OpenMed

- **Core de-id:** `deidentifying-clinical-text` — combine date shifting with a
  `policy=` so names/IDs are redacted in the same pipeline.
- **Safe Harbor checklist:** `auditing-safe-harbor-checklist` — the date rule and
  the age-90 cap are categories C in the 18.
- **Audit:** `auditing-deidentification-runs` records the method and per-span
  actions (offsets/hashes, not raw dates).
- **Other surfaces:** MCP `openmed_deidentify` / REST `POST /pii/deidentify`
  accept the same method and parameters.

## Edge cases & gotchas

- **Offset 0 is forbidden.** An auto-selected offset is always non-zero — a
  zero shift would silently leave dates unchanged and defeat de-identification.
- **Reuse the seed per patient, not globally.** A single global offset lets an
  attacker re-align all patients to a real anchor date; per-patient offsets break
  that. Derive seeds from a *secret*, not the plaintext MRN.
- **`keep_year=True` is not Safe Harbor by itself** when the year reveals the
  age of someone >89 or ties to care episodes — pair with age capping.
- **Shifting does not cap ages.** Age >89 is a separate transformation; date
  shifting won't fix an explicit "age 94" in the text.
- **Day-first locales.** For non-English notes set `lang=` so `11/04/2024` is
  parsed in the right order before shifting (`deidentifying-multilingual-text`).
- **Store the per-patient seed/offset like PHI** — it re-identifies the calendar
  if leaked. Keep it in a vault, separate from the output.

## Standards & references

- HIPAA Safe Harbor date rule, 45 CFR 164.514(b)(2)(i)(C): https://www.ecfr.gov/current/title-45/subtitle-A/subchapter-C/part-164/subpart-E/section-164.514
- HHS de-identification guidance (dates & ages >89): https://www.hhs.gov/hipaa/for-professionals/privacy/special-topics/de-identification/index.html
- OpenMed source: `openmed/core/pii.py` (`deidentify(method="shift_dates")`,
  `_shift_date`, `_random_nonzero_shift`).
