---
name: reidentifying-text
description: "Reversibly de-identify clinical text with OpenMed and later restore the original PHI from a saved mapping. Use when the user needs pseudonymization rather than permanent anonymization, wants to mask PHI now and re-link it later under authorization (e.g. recontact, adjudication, GDPR pseudonymization), asks about deidentify keep_mapping, reidentify, or how to store and protect the re-identification mapping. Covers when reversibility is and is not appropriate (pseudonymization vs HIPAA Safe Harbor anonymization). Pairs after extracting-pii-entities and deidentifying-clinical-text."
license: Apache-2.0
metadata:
  project: OpenMed
  category: openmed-core
  pairs: adjacent
  version: "1.0"
---

# Reidentifying Text

Some workflows need to remove PHI **for processing** but keep the ability to
restore it later under authorization — adjudication, patient recontact, linking
results back to a record. That is **pseudonymization** (reversible), not
**anonymization** (irreversible). OpenMed supports it with
`deidentify(..., keep_mapping=True)` to capture a mapping, and `reidentify` to
restore. Everything runs on-device.

## When to use

- You need to **re-link** redacted output to the original record later.
- You are doing **GDPR pseudonymization** (Art. 4(5)): identifiers held separately,
  reversible under controlled conditions.
- A reviewer must **spot-check** redactions against originals.

**Do NOT use reversibility when:**

- The goal is **HIPAA Safe Harbor anonymization** or a true **anonymous** release —
  a re-identification mapping defeats anonymization. Use `method="remove"` and keep
  **no** mapping.
- The redacted text leaves your trust boundary and the mapping might travel with
  it. The mapping is the secret; never co-locate it with the de-identified output.

## Install

```bash
pip install "openmed[hf]"
```

## Quick start: reversible round-trip

```python
import openmed

note = "Patient John Doe (MRN 00481726) seen on 2024-03-02 by Dr. Alice Smith."

# 1) De-identify AND capture the reversal mapping
deid = openmed.deidentify(
    note,
    method="mask",          # or "replace" for realistic surrogates
    keep_mapping=True,       # <-- required to enable reidentify()
    policy="gdpr_pseudonymization",
)

safe_text = deid.deidentified_text       # ship/process this
mapping   = deid.mapping                  # SECRET: store separately, encrypted

# 2) Later, under authorization, restore the original
restored = openmed.reidentify(safe_text, mapping)
assert restored == note
```

`reidentify(deidentified_text, mapping)` performs the inverse substitution. The
`mapping` is a `dict[str, str]` of **redacted → original** text, produced only when
`keep_mapping=True`.

## Use `consistent` surrogates for stable pseudonyms

For replacement that maps the *same* identifier to the *same* surrogate across a
document (and reproducibly across runs with a seed):

```python
import openmed

deid = openmed.deidentify(
    "Mr. John Doe called. John Doe's MRN is 00481726.",
    method="replace",
    consistent=True,    # same input value -> same surrogate within the run
    seed=42,            # reproducible across runs (implies consistent=True)
    keep_mapping=True,
)
print(deid.deidentified_text)
restored = openmed.reidentify(deid.deidentified_text, deid.mapping)
```

`consistent=True` keeps surrogates stable so analytics on the pseudonymized text
stay coherent; `seed` makes them reproducible. Either way, reversal still requires
the saved `mapping`.

## Store the mapping securely — separate from the text

The mapping is the re-identification key. Treat it like a secret:

- **Never** write it to the same store/file/log as the de-identified text.
- Encrypt at rest; restrict access; audit every reversal.
- Key the store by an opaque document id, not by any patient identifier.

```python
import json, os
import openmed

note = "Patient John Doe (MRN 00481726), DOB 1970-01-15."
deid = openmed.deidentify(note, method="mask", keep_mapping=True, seed=7)

doc_id = "doc-7f3a"   # opaque id, no PHI

# De-identified text -> general processing store (safe to share downstream)
with open(f"deid/{doc_id}.txt", "w", encoding="utf-8") as fh:
    fh.write(deid.deidentified_text)

# Mapping -> SEPARATE, access-controlled, encrypted vault (illustrative path)
os.makedirs("vault", exist_ok=True)
with open(f"vault/{doc_id}.map.json", "w", encoding="utf-8") as fh:
    json.dump(deid.mapping, fh)   # encrypt this store in production
```

To re-identify later, load **only** the mapping for the authorized `doc_id`:

```python
import json, openmed
with open("vault/doc-7f3a.map.json", encoding="utf-8") as fh:
    mapping = json.load(fh)
with open("deid/doc-7f3a.txt", encoding="utf-8") as fh:
    safe_text = fh.read()
original = openmed.reidentify(safe_text, mapping)
```

## Reversible vs irreversible: pick deliberately

| Goal | Call | Mapping |
| --- | --- | --- |
| GDPR pseudonymization (reversible) | `deidentify(..., keep_mapping=True, policy="gdpr_pseudonymization")` | keep, encrypted, separate |
| HIPAA Safe Harbor anonymization | `deidentify(..., method="remove", policy="hipaa_safe_harbor")` | **none** |
| Irreversible token linking | `deidentify(..., method="hash")` | none (one-way) |

`method="hash"` yields consistent, **one-way** tokens — good for joining records
without ever restoring the original. That is not reversible and needs no mapping.

## Hand-off to / from OpenMed

- **From `extracting-pii-entities`:** preview the spans first if you want to confirm
  what will be masked before committing to a reversible run.
- **From `deidentifying-clinical-text`:** that skill covers methods, policies, and
  the safety sweep; this one adds the `keep_mapping` + `reidentify` round-trip.
- **To downstream NER:** run `openmed.analyze_text` on `deid.deidentified_text`;
  re-identify only the final, authorized output — never intermediate logs.

## Edge cases & gotchas

- **`keep_mapping=True` is mandatory** for `reidentify` to work; without it
  `deid.mapping` is `None`.
- **Result field is `.deidentified_text`** (and `.pii_entities`, `.mapping`), not
  `.text`/`.entities`.
- **Mapping direction is redacted → original.** `reidentify` substitutes those keys
  back into the text.
- **Mask collisions:** with `method="mask"`, identical placeholders (e.g. two
  `[NAME]`) cannot be distinguished on reversal. For lossless round-trips use
  `method="replace"` with `consistent=True`/`seed`, which produces distinct,
  reversible surrogates.
- **Never anonymize-and-keep-mapping.** If the release must be anonymous, keep no
  mapping — a stored mapping makes it pseudonymous, not anonymous.
- **Authorization & audit.** Re-identification is privileged; log who/when/why and
  keep the mapping out of general PHI logs.

## Standards & references

- GDPR pseudonymization: Regulation (EU) 2016/679, Art. 4(5) & Recital 26 —
  https://gdpr-info.eu/art-4-gdpr/
- HIPAA de-identification (Safe Harbor / Expert Determination):
  https://www.hhs.gov/hipaa/for-professionals/privacy/special-topics/de-identification/
- See also OpenMed policy profiles `gdpr_pseudonymization` and
  `hipaa_safe_harbor` (`configuring-privacy-policies`).
