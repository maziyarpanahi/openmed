---
name: generating-synthetic-surrogates
description: "Replace detected PHI with realistic, type-matched fake values in OpenMed so clinical notes stay readable and parseable instead of full of [REDACTED] markers. Use when the user wants surrogate names, MRNs, addresses, or dates rather than opaque masks, needs consistent fake identities across a document, must keep notes natural for downstream NLP, or wants to register a custom surrogate generator or provider. Covers deidentify(method=\"replace\", consistent=True, seed=..., locale=...), register_label_generator, register_clinical_provider, and Anonymizer/AnonymizerConfig. Pairs with OpenMed deidentifying-clinical-text and configuring-privacy-policies."
license: Apache-2.0
metadata:
  project: OpenMed
  category: de-identification
  pairs: after
  version: "1.0"
---

# Generating synthetic surrogates

`method="replace"` swaps each detected identifier for a **realistic, type-matched
fake** — `John Doe` becomes `Mark Lee`, a phone becomes a plausible phone, a date
becomes a plausible date. Unlike opaque `[REDACTED]`/`[NAME]` masks, surrogate
text reads naturally and stays parseable by downstream NLP, while still
containing no real PHI. OpenMed generates surrogates on-device via Faker-backed
providers keyed to each canonical label.

## When to use this skill

Use surrogates when the de-identified text must remain **readable or machine-
parseable**: training data for clinical NLP, demos, QA, or notes a human still
needs to skim. If you only need the identifiers gone and don't care about
readability, plain `method="mask"` is simpler and more obviously redacted.

## Quick start

```python
import openmed

note = (
    "Patient John Doe (MRN 1234567) saw Dr. John Doe's colleague on 2024-03-02. "
    "Reach John Doe at 617-555-0142."
)

result = openmed.deidentify(
    note,
    method="replace",
    consistent=True,     # every "John Doe" -> the SAME surrogate within this call
    seed=42,             # reproducible across runs
    locale="en_US",      # shapes the fakes; defaults from lang via LANG_TO_LOCALE
)
print(result.deidentified_text)
# Patient Mark Lee (MRN 8830127) saw Dr. Mark Lee's colleague on 2024-07-18. ...
```

`consistent=True` is what makes the output coherent: the three mentions of
"John Doe" collapse to one fake identity instead of three different ones, so the
note still makes sense. `seed=` makes that mapping reproducible run to run.

## Surrogates vs opaque redaction

| | `method="mask"` (`[NAME]`) | `method="replace"` (surrogate) |
| --- | --- | --- |
| Readability | low — placeholders | high — reads like a real note |
| Downstream NLP | tokenizers see `[NAME]` everywhere | natural distribution preserved |
| Co-reference | lost (all `[NAME]`) | preserved with `consistent=True` |
| Obvious it's de-identified | yes | no (must be tracked out-of-band) |
| Reversible | with `keep_mapping=True` | with `keep_mapping=True` |

## Custom providers and label generators

When a built-in surrogate doesn't match your house format (e.g. your MRNs are
`H` + 7 digits), register a generator or a Faker provider.

```python
from openmed import (
    register_label_generator, register_clinical_provider,
    Anonymizer, AnonymizerConfig,
)

# Override the surrogate for one canonical label. Signature: (faker, original, *, locale)
def hospital_mrn(faker, original, *, locale):
    return f"H{faker.numerify('#######')}"

register_label_generator("ID_NUM", hospital_mrn)   # global, all new Anonymizers

# Add a whole custom Faker provider (e.g. proprietary identifier formats):
register_clinical_provider(MyClinicalProvider)     # a faker BaseProvider subclass

# Per-instance control (preferred for isolation): pass providers via config,
# and pull a single surrogate directly when you need one.
anon = Anonymizer(AnonymizerConfig(
    lang="en", consistent=True, seed=7, custom_providers=[MyClinicalProvider],
))
fake = anon.surrogate("1234567", "ID_NUM")
```

Use `register_label_generator(canonical_label, fn)` to swap one label's
surrogate; `register_clinical_provider(provider)` to add providers globally; or
`AnonymizerConfig.custom_providers` for per-run scoping. Validate any custom
label against `openmed.CANONICAL_LABELS`.

## Workflow

1. **Choose `method="replace"`** (or a profile like `gdpr_pseudonymization` /
   `canada_pipeda` that replaces by default — see `configuring-privacy-policies`).
2. **Enable consistency** with `consistent=True` and a `seed=` so repeated
   mentions resolve to one identity and the result is reproducible.
3. **Set `locale=`** so surrogates look native (`pt_BR`, `de_DE`, …); it
   defaults from `lang` via `LANG_TO_LOCALE`
   (`deidentifying-multilingual-text`).
4. **Register custom generators** for any house-specific formats (MRN, account,
   address) before the run.
5. **If reversibility is needed**, add `keep_mapping=True` and store
   `result.mapping` as a secret, separate from the output.
6. **Verify** no surrogate collides with a real value and residual risk is low
   (`auditing-deidentification-runs`).

## Hand-off to / from OpenMed

- **Core de-id:** `deidentifying-clinical-text` — `method`, thresholds,
  `keep_mapping`, policies.
- **Policies that replace:** `configuring-privacy-policies`
  (`gdpr_pseudonymization`, `canada_pipeda`).
- **Multilingual surrogates:** `deidentifying-multilingual-text` (`lang`/`locale`).
- **Restore:** `openmed.reidentify(text, mapping)` when `keep_mapping=True`.
- **Other surfaces:** MCP `openmed_deidentify` / REST `POST /pii/deidentify`.

## Edge cases & gotchas

- **Surrogates must not collide with real values.** A fake MRN that happens to be
  a real patient's MRN re-identifies them. Keep generated identifiers out of the
  real ID space (dedicated prefix/range) and check against your live keys.
- **Surrogates look real but are not labeled.** Anyone reading the output cannot
  tell it's de-identified. Track provenance out-of-band (e.g. an `AuditReport`)
  so surrogate notes are never mistaken for source records.
- **Keep the mapping secret.** With `keep_mapping=True`, `result.mapping`
  re-identifies everyone — encrypt it and store it apart from the output.
- **`register_label_generator` is global and process-wide.** It mutates a shared
  registry; for isolation use `AnonymizerConfig.custom_providers` instead.
- **Consistency is per-document by default.** `consistent=True` makes mentions
  agree within a call; cross-document stability requires the same `seed`.
- **Permissive licensing only.** Don't build providers from
  UMLS/SNOMED/CPT/MIMIC/i2b2/n2c2; call restricted resources out-of-process.

## Standards & references

- GDPR pseudonymization, Regulation (EU) 2016/679 Art. 4(5): https://eur-lex.europa.eu/eli/reg/2016/679/oj
- HIPAA de-identification, 45 CFR 164.514(b): https://www.hhs.gov/hipaa/for-professionals/privacy/special-topics/de-identification/index.html
- OpenMed source: `openmed/core/pii.py` (`deidentify(method="replace")`),
  `openmed/core/anonymizer/` (`Anonymizer`, `AnonymizerConfig`,
  `register_label_generator`, `register_clinical_provider`, `LANG_TO_LOCALE`).
