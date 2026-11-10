---
name: pseudonymizing-for-gdpr
description: "Apply GDPR-grade pseudonymization to clinical or personal text with OpenMed, keeping a separately-held re-linkage key so the data can be controlled-re-linked later. Use when the user must process EU personal/health data under GDPR, asks for pseudonymization vs anonymization, needs Art. 4(5) / Art. 9 / Recital 26 alignment, wants a reversible mapping/key vault held apart from the data, or needs controlled re-linkage. Covers openmed.deidentify(policy=\"gdpr_pseudonymization\", keep_mapping=True), storing the mapping in a separate key vault, reidentify() for authorized re-linkage, and retention. Pairs after extracting-pii-entities and configuring-privacy-policies."
license: Apache-2.0
metadata:
  project: OpenMed
  category: de-identification
  pairs: after
  version: "1.0"
---

# Pseudonymizing for GDPR

Pseudonymization under the GDPR (Art. 4(5)) means processing personal data so it
"can no longer be attributed to a specific data subject without the use of
**additional information**" — provided that additional information (the
re-linkage key) is "kept separately and is subject to technical and
organisational measures." Crucially, **pseudonymized data is still personal
data** (Recital 26): re-linkage is possible, so GDPR still applies. This is the
opposite of anonymization, where re-identification is irreversibly prevented and
the data falls outside the GDPR.

OpenMed implements this with a single reversible de-identification pass plus a
mapping you store **away from the data**. This skill covers producing that
mapping, vaulting the key separately, and re-linking under authorization.

## When to use

- You process EU residents' personal or special-category health data (Art. 9)
  and need a lawful, reversible safeguard rather than full anonymization.
- You need to keep a record-linkage capability (e.g. to recontact a patient,
  reconcile longitudinal records, or honor a Subject Access Request) but must
  separate the linkage key from the working dataset.
- A reviewer asks for the pseudonymization-vs-anonymization distinction in
  writing, or for the ENISA-style "additional information kept separately"
  control to be demonstrable.

Do **not** use this when the goal is irreversible anonymization for open release
— there, drop the mapping entirely and gate residual risk with
`reviewing-reidentification-risk`. Pseudonymization keeps a key; anonymization
must not.

## Quick start

```python
import openmed

# Synthetic record — never run this skill's examples on real PHI.
note = "Patient Maria Schmidt (ID 4471) seen 2024-03-02; contact maria@example.de."

result = openmed.deidentify(
    note,
    method="replace",                 # realistic surrogates, not [LABEL] holes
    policy="gdpr_pseudonymization",   # bundled GDPR profile
    keep_mapping=True,                # produce the reversible re-linkage map
    consistent=True,                  # same input -> same surrogate in the doc
    seed=20240302,                    # cross-run reproducibility of surrogates
)

pseudonymized_text = result.deidentified_text   # safe to process / analyze
relink_key = result.mapping                      # surrogate -> original; SECRET
```

`result.deidentified_text` is the pseudonymized payload. `result.mapping` is the
"additional information" GDPR Art. 4(5) requires be kept separately — it is the
key that makes re-linkage possible, and therefore the most sensitive artifact in
the whole flow.

## Workflow

1. **Choose reversible pseudonymization, not masking.** Use `method="replace"`
   with `policy="gdpr_pseudonymization"` and `keep_mapping=True`. Replacement
   surrogates keep the text usable for downstream NLP while remaining
   non-identifying. `consistent=True` (optionally with `seed=`) makes repeated
   mentions resolve to one stable surrogate so intra-document linkage survives.
2. **Split the data from the key immediately.** The moment `deidentify` returns,
   route `result.deidentified_text` to your working store and `result.mapping`
   to a **separate, access-controlled key vault** — different system, different
   credentials, different backups. Never persist them in the same row, file,
   bucket, or log line. This separation is the technical-and-organisational
   measure that makes the data pseudonymized rather than just "personal data
   with PII in it."
3. **Process the pseudonymized text freely.** Run `analyze_text`, analytics,
   model training, or transfer on `deidentified_text`. The key never leaves the
   vault during ordinary processing.
4. **Re-link only under authorization.** When a lawful basis exists (e.g. an
   authorized SAR or recontact), fetch the mapping from the vault and call
   `openmed.reidentify(deidentified_text, mapping)`. Log *that* a re-linkage
   happened (who, when, why, record id) — but never log the restored plaintext.
5. **Apply retention to the key.** The mapping has its own retention clock. When
   the lawful basis for re-linkage ends, **destroy the mapping**. Once the key
   is irreversibly gone and no other re-identification path remains, the
   remaining text approaches anonymization and GDPR obligations shrink
   accordingly. Verify that claim with `reviewing-reidentification-risk` before
   relying on it.

## Hand-off to / from OpenMed

- **From** `extracting-pii-entities` / `configuring-privacy-policies`: confirm
  the detector recall and the active policy profile before pseudonymizing, since
  any identifier the detector misses leaks into `deidentified_text`.
- **OpenMed call:** Python `from openmed import deidentify, reidentify`; the same
  capability is exposed as MCP tool `openmed_deidentify` and REST `/deidentify`.
  Pass `policy="gdpr_pseudonymization"`, `keep_mapping=True`.
- **To** `auditing-deid-leakage`: scan `result.deidentified_text` for residual
  identifiers before it leaves the boundary — pseudonymization is only as strong
  as detection.
- **To** `reviewing-reidentification-risk`: quasi-identifier (age, ZIP, dates)
  re-identification still applies to pseudonymized data; score k-anonymity on the
  output and document residual risk.

## Edge cases & gotchas

- **Pseudonymized ≠ anonymized.** As long as `mapping` exists anywhere, the data
  is personal data under Recital 26. Do not market a `keep_mapping=True` output
  as "anonymous."
- **The mapping is the crown jewel.** A leaked mapping re-identifies everything
  at once. Treat it as the highest-sensitivity secret: encrypt at rest, restrict
  access, audit reads.
- **Surrogates can still carry quasi-identifiers.** `method="replace"` swaps the
  identifier text, but free-text age, rare diagnosis, ZIP, or admission dates
  remain. Pseudonymization does not address singling-out; pair with QI risk
  scoring.
- **Reproducibility cuts both ways.** A fixed `seed` makes surrogates stable
  across runs (good for linkage) but means an attacker who learns the seed and
  algorithm can reproduce surrogates — keep the seed with the key, not the data.
- **Special-category data (Art. 9).** Health data needs a lawful basis *before*
  processing; pseudonymization is a safeguard, not a lawful basis on its own.
- **Local-first.** Run entirely on-device. Do not send EU personal data to a
  cloud de-identification service to satisfy GDPR — that may itself be a transfer.

## Standards & references

- GDPR Art. 4(5) — definition of pseudonymization:
  https://gdpr-info.eu/art-4-gdpr/
- GDPR Art. 9 — processing of special categories (health) data:
  https://gdpr-info.eu/art-9-gdpr/
- GDPR Recital 26 — pseudonymous data is personal data; anonymization test:
  https://gdpr-info.eu/recitals/no-26/
- ENISA, *Pseudonymisation techniques and best practices* (2019):
  https://www.enisa.europa.eu/publications/pseudonymisation-techniques-and-best-practices
- EDPB Guidelines on pseudonymisation (01/2025):
  https://www.edpb.europa.eu/
