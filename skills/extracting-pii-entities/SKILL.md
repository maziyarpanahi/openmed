---
name: extracting-pii-entities
description: "Detect PHI/PII spans in clinical text with OpenMed's extract_pii without altering the text. Use when the user wants to find names, dates, MRNs, phone numbers, addresses, SSNs, or other identifiers and get their offsets and labels (not redact them), inspect what would be removed before de-identifying, route spans to a custom redactor, normalize labels to a canonical taxonomy, or filter by confidence and language. Covers extract_pii, the PIIEntity fields, CANONICAL_LABELS / normalize_label, and how it differs from deidentify. Pairs before reidentifying-text and deidentifying-clinical-text."
license: Apache-2.0
metadata:
  project: OpenMed
  category: openmed-core
  pairs: adjacent
  version: "1.0"
---

# Extracting PII Entities

`openmed.extract_pii` finds PHI/PII spans and **returns them without changing the
text**. Use it when you need to *see* the identifiers — to audit, route to a
custom redactor, or decide a policy — rather than produce redacted output. It runs
on-device.

## When to use

- You want the **spans and labels** of identifiers, with the original text intact.
- You need a **preview** of what `deidentify` would act on before committing.
- You are feeding detected spans into a **downstream redactor** (your own,
  Presidio, or `deidentify`).
- You want to **normalize** model labels to a stable canonical taxonomy.

If you instead want redacted/masked output directly, use
`deidentifying-clinical-text` (`openmed.deidentify`). If you need reversible
masking, see `reidentifying-text`.

## extract_pii vs deidentify

| | `extract_pii` | `deidentify` |
| --- | --- | --- |
| Changes the text? | **No** | Yes (mask/remove/replace/hash/shift) |
| Returns | `PredictionResult` (spans) | `DeidentificationResult` (redacted text) |
| Default threshold | `0.5` | `0.7` (safety-biased) |
| Use for | detection, audit, routing | producing safe output |

## Install

```bash
pip install "openmed[hf]"
```

## Quick start

```python
import openmed

note = "Patient John Doe (MRN 00481726), DOB 1970-01-15, phone 617-555-0142."

result = openmed.extract_pii(note, confidence_threshold=0.5)

for ent in result.entities:
    print(f"{ent.label:10} {ent.text!r:18} {ent.confidence:.2f} [{ent.start}:{ent.end}]")
```

`extract_pii(...)` returns a `PredictionResult`. Its `.entities` are `PIIEntity`
objects (synthetic example fields shown):

```text
ent.text            # the identifier surface string, e.g. "617-555-0142"
ent.label           # detected label, e.g. "PHONE"
ent.confidence      # model score in [0, 1]   (NOTE: .confidence, not .score)
ent.start / ent.end # character offsets into the original note
ent.canonical_label # label mapped to OpenMed's canonical taxonomy (if set)
ent.entity_type     # same as label
```

The text is unchanged — `result.text` is your original input.

## Signature & key parameters

```python
openmed.extract_pii(
    text,
    model_name="OpenMed/OpenMed-PII-SuperClinical-Small-44M-v1",  # default EN model
    confidence_threshold=0.5,     # raise for precision, lower for recall
    use_smart_merging=True,       # merge fragmented spans into whole units
    lang="en",                    # en es pt fr de it nl hi te ar tr ja
    loader=None,                  # reuse a ModelLoader across calls
)
```

- **`use_smart_merging=True`** (default) reassembles fragmented predictions into
  complete units (a full phone number, a full date) — keep it on.
- **`lang`** selects the language-appropriate default model and regex patterns.
  Pass the right language; do not run the English model on non-English text. Use
  `openmed.get_default_pii_model(lang)` to confirm coverage.

## Normalize labels to the canonical taxonomy

Different models may emit slightly different label spellings. Normalize them to
OpenMed's canonical set so downstream logic is stable:

```python
import openmed
from openmed import CANONICAL_LABELS, normalize_label

result = openmed.extract_pii("Email jane.roe@example.org; SSN 123-45-6789.")

for ent in result.entities:
    canon = ent.canonical_label or normalize_label(ent.label)
    assert canon in CANONICAL_LABELS or canon == "OTHER"
    print(ent.text, "->", canon)
```

`CANONICAL_LABELS` is a frozenset of UPPER_SNAKE_CASE labels (e.g. `PERSON`,
`DATE`, `PHONE`, `EMAIL`, `SSN`, `ID_NUM`, `LOCATION`). `normalize_label(label)`
accepts messy inputs (`"FIRSTNAME"`, `"first_name"`, `"B-EMAIL"`) and maps unknown
labels to `OTHER` rather than raising.

## Feed spans to a downstream redactor

`extract_pii` gives you offsets; you decide the action. A simple offset-based
redactor (replace highest-offset first so positions stay valid):

```python
import openmed

note = "Patient John Doe, MRN 00481726, seen 2024-03-02."
result = openmed.extract_pii(note, confidence_threshold=0.6)

redacted = note
for ent in sorted(result.entities, key=lambda e: e.start, reverse=True):
    redacted = redacted[:ent.start] + f"[{ent.label}]" + redacted[ent.end:]

print(redacted)   # Patient [PERSON], MRN [ID_NUM], seen [DATE].
```

For production redaction, masking strategies, and policy profiles, hand the work
to `openmed.deidentify` instead of hand-rolling — it adds a safety sweep and
date-shifting (see `deidentifying-clinical-text`).

## Hand-off to / from OpenMed

- **To `deidentifying-clinical-text`:** once you have reviewed the spans, call
  `openmed.deidentify(note, method="mask", policy="hipaa_safe_harbor")` to produce
  safe output — it re-detects with a higher default threshold for safety.
- **To `reidentifying-text`:** if you need reversibility, use
  `openmed.deidentify(..., keep_mapping=True)` and store the mapping securely.
- **To Presidio / custom anonymizers:** `extract_pii` spans (`label`, `start`,
  `end`) translate cleanly into other recognizers' result formats; OpenMed also
  ships an `Anonymizer` (`openmed.Anonymizer`) for richer surrogate generation.

## Edge cases & gotchas

- **Attribute is `.confidence`, not `.score`.** `PIIEntity` extends
  `EntityPrediction`.
- **Detection is not redaction.** `extract_pii` never changes text — if a caller
  expected redacted output, they want `deidentify`.
- **Threshold trade-off:** `0.5` favors recall (good for *finding* PHI to review).
  For *removing* PHI, prefer `deidentify`'s safety-biased `0.7` default.
- **Language matters:** wrong `lang` silently lowers recall. Verify with
  `get_default_pii_model(lang)`.
- **No raw PHI in logs/audit.** Record offsets, labels, and hashes — never the
  identifier text. Use synthetic data in examples and tests.
- **Local-first.** No cloud calls in PHI workflows; models run on-device after a
  one-time download.

## Standards & references

- HIPAA Safe Harbor 18 identifiers: 45 CFR §164.514(b)(2) —
  https://www.hhs.gov/hipaa/for-professionals/privacy/special-topics/de-identification/
- OpenMed canonical labels: `openmed.CANONICAL_LABELS` / `openmed.normalize_label`.
- OpenMed PII models: https://huggingface.co/OpenMed
