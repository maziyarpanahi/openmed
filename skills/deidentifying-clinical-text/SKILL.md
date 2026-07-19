---
name: deidentifying-clinical-text
description: "Remove, mask, or replace PHI/PII in clinical free text on-device with OpenMed's deidentify(). Use when the user needs to de-identify medical notes, strip patient identifiers, redact PHI before sharing or analysis, anonymize discharge summaries, or pick a de-id method (mask vs remove vs replace vs hash vs shift_dates). Covers confidence_threshold for safety, consistent+seed for stable surrogates, keep_mapping for reversible de-id, policy= profiles, and the DeidentificationResult fields. Pairs with OpenMed extract_pii (detect spans), reidentify (restore), configuring-privacy-policies, and auditing-deidentification-runs."
license: Apache-2.0
metadata:
  project: OpenMed
  category: openmed-core
  pairs: adjacent
  version: "1.0"
---

# De-identifying clinical text

`openmed.deidentify` detects PHI/PII and rewrites the text so it can be shared,
stored, or analyzed without exposing patients. It runs **fully on-device** after
a one-time model download — no network calls, no telemetry, no raw PHI leaving
the process. This is the single most important OpenMed entry point for privacy
work; everything else (policies, audit, multilingual, date-shifting) layers on
top of it.

## When to use this skill

Reach for `deidentify` when you need to transform text — replace, mask, remove,
hash, or date-shift the identifiers. If you only need to **locate** PHI spans
without changing the text, use `extract_pii` (see `extracting-pii-entities`). To
**restore** masked text later, use `reidentify` (see `reidentifying-text`).

## Quick start

```python
import openmed

note = (
    "Patient John Doe (MRN 1234567) was seen on 2024-03-02 by Dr. Alice Reed. "
    "Contact: john.doe@example.com, 617-555-0142."
)

result = openmed.deidentify(
    note,
    method="mask",                 # mask | remove | replace | hash | shift_dates
    confidence_threshold=0.7,      # safety default; raise to reduce false negatives' impact
    policy="hipaa_safe_harbor",    # optional bundled profile (see below)
)

print(result.deidentified_text)
# Patient [NAME] (MRN [ID_NUM]) was seen on [DATE] by Dr. [NAME]. ...

for e in result.pii_entities:
    # NEVER log e.text / e.original_text — those are raw PHI. Use offsets + label.
    print(e.canonical_label, e.start, e.end, round(e.confidence, 3))
```

`deidentify` returns a `DeidentificationResult` with these fields (note the
exact names):

| Field | What it holds |
| --- | --- |
| `.deidentified_text` | the rewritten, PHI-safe string (your output) |
| `.pii_entities` | `list[PIIEntity]` — each has `start`, `end`, `canonical_label`, `confidence`, `action`, `surrogate`; `original_text`/`text` hold raw PHI |
| `.mapping` | redacted→original dict, only when `keep_mapping=True` (secret) |
| `.method` | the method actually applied |
| `.metadata` | run metadata (model, policy, counts) |

## The five methods

| `method=` | Effect | Reversible? | Use when |
| --- | --- | --- | --- |
| `"mask"` | `John Doe` → `[NAME]` | with `keep_mapping=True` | default; clear that redaction happened |
| `"remove"` | deletes the span entirely | no | minimal-footprint output |
| `"replace"` | type-matched fake value (`John Doe`→`Mark Lee`) | with `keep_mapping=True` | keep notes readable/parseable (see `generating-synthetic-surrogates`) |
| `"hash"` | stable hash per value, links repeats | no (one-way) | cohort linkage without revealing identity |
| `"shift_dates"` | moves dates, preserves intervals | n/a | research needing temporal structure (see `shifting-clinical-dates`) |

## Workflow

1. **Pick a method and a policy.** Start from a bundled `policy=` profile
   (`hipaa_safe_harbor`, `gdpr_pseudonymization`, `research_limited_dataset`, …)
   so per-label actions are set for you. See `configuring-privacy-policies`.
2. **Set `confidence_threshold` deliberately.** Default is `0.7`. For de-id,
   prefer *over*-redaction: a missed identifier is a breach, an over-redacted
   token is just noise. The bundled safety sweep catches structured IDs
   (SSN, MRN-like, emails) even below threshold.
3. **Run `deidentify`.** Inspect `result.pii_entities` by **offset and label**,
   not raw text, to confirm coverage.
4. **For stable surrogates**, pass `consistent=True, seed=<int>` so the same
   input maps to the same fake value every run (reproducible pipelines).
5. **For reversibility**, pass `keep_mapping=True` and store `result.mapping`
   in a secured vault — never alongside the de-identified output.
6. **Verify, don't assume.** Check residual risk with `audit=True`
   (`auditing-deidentification-runs`) and the 18-identifier checklist
   (`auditing-safe-harbor-checklist`).

## Consistent surrogates and reversibility

```python
# Same fake identity for every mention of the same person, reproducibly:
r = openmed.deidentify(note, method="replace", consistent=True, seed=42)

# Reversible de-id (keep the mapping secret and separate from output):
r = openmed.deidentify(note, method="mask", keep_mapping=True)
restored = openmed.reidentify(r.deidentified_text, r.mapping)
assert restored == note
```

## Hand-off to / from OpenMed

- **Detect only:** `openmed.extract_pii(text)` → `PredictionResult` with
  `.entities` (spans, no rewrite). Use it to preview coverage first.
- **Restore:** `openmed.reidentify(deidentified_text, mapping)` — requires
  `keep_mapping=True` at de-id time and proper authorization.
- **Policies:** `configuring-privacy-policies` to choose/customize a `policy=`.
- **Audit:** `deidentify(..., audit=True)` → `AuditReport` with offsets, hashes,
  detector provenance, and residual-risk — never plaintext.
- **Other surfaces (same engine):** MCP tool `openmed_deidentify`; REST
  `POST /pii/deidentify`. There is **no** CLI de-id command.

## Edge cases & gotchas

- **Attribute names.** It is `result.deidentified_text` and
  `result.pii_entities` — not `.text`/`.entities`. (`extract_pii` returns a
  `PredictionResult` whose spans are at `.entities`.)
- **Raw PHI never leaves the span objects.** `PIIEntity.text` and
  `.original_text` contain real identifiers. Do not print, log, or cache them.
  Audit and logs use offsets, `canonical_label`, and hashes only.
- **Threshold is a safety dial, not an accuracy dial.** Lowering it redacts
  *more*; in de-id, false positives are cheap and false negatives are breaches.
- **`shift_dates` is for dates only**; combine with `keep_year`/`date_shift_days`
  (see `shifting-clinical-dates`). It does not touch names or IDs.
- **`keep_mapping` output is sensitive as PHI.** The mapping re-identifies
  everyone — store it encrypted, access-controlled, and apart from the output.
- **Multilingual:** pass `lang=` (and `locale=` for surrogates) for non-English
  notes; see `deidentifying-multilingual-text`. Do not run English models on
  other languages.
- **De-id is verified, not assumed.** Gate releases on leakage/residual-risk,
  not F1 alone.

## Standards & references

- HIPAA De-identification, 45 CFR 164.514(b) — Safe Harbor & Expert
  Determination: https://www.hhs.gov/hipaa/for-professionals/privacy/special-topics/de-identification/index.html
- HHS Safe Harbor 18-identifier guidance: see `auditing-safe-harbor-checklist`.
- OpenMed source: `openmed/core/pii.py` (`deidentify`, `extract_pii`,
  `reidentify`, `DeidentificationResult`, `PIIEntity`).
