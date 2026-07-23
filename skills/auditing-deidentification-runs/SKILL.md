---
name: auditing-deidentification-runs
description: "Produce a signed, reproducible, no-PHI audit trail for an OpenMed de-identification run via deidentify(audit=True). Use when the user needs compliance evidence, a tamper-evident record of what was redacted and why, to verify nothing was changed, to retain proof for HIPAA/GDPR audits, or to review de-id decisions without exposing plaintext PHI. Covers the AuditReport / AuditSignature / AuditSpan / DetectorInfo fields, why audits store offsets+hashes+provenance+residual-risk and never plaintext, signing with .sign(key), and verifying with .verify(key). Pairs with OpenMed deidentifying-clinical-text and auditing-safe-harbor-checklist."
license: Apache-2.0
metadata:
  project: OpenMed
  category: de-identification
  pairs: after
  version: "1.0"
---

# Auditing de-identification runs

`deidentify(..., audit=True)` returns an `AuditReport` instead of the rewritten
text: a deterministic, **PHI-free** record of every redaction decision —
offsets, label, detector confidence and threshold, the action taken, content
hashes, model provenance, and projected residual risk. Sign it to make it
tamper-evident, retain it as compliance evidence, and verify it later without
ever touching the original text. Runs **on-device**.

## When to use this skill

Use it whenever a de-identification needs to be *defensible*: regulatory
retention, internal review, reproducibility checks, or proving to an auditor
that identifiers were detected and handled — all without storing or exposing the
PHI itself.

## Quick start

```python
import openmed

note = "Patient John Doe (MRN 1234567) seen 2024-03-02. SSN 123-45-6789."

# Returns an AuditReport, NOT a DeidentificationResult, when audit=True.
report = openmed.deidentify(note, policy="hipaa_safe_harbor", audit=True)

# Make it tamper-evident with a release HMAC key (keep the key in a vault).
report.sign(b"my-release-hmac-key", key_id="release-2026")

# Persist the no-PHI report (no plaintext identifiers inside).
import json
with open("deid_audit.json", "w") as fh:
    json.dump(report.to_dict(), fh, indent=2)

# Later: verify integrity. Optionally bind to the exact texts via their hashes.
ok = report.verify(b"my-release-hmac-key", original_text=note)
assert ok
```

## What an AuditReport contains

`AuditReport` (from `openmed.core.audit`) carries no plaintext PHI. Key fields:

| Field | Meaning |
| --- | --- |
| `policy` | policy profile name in effect |
| `resolved_profile` | the concrete settings applied (method, model, thresholds, language, sweep flags) |
| `detectors` | `list[DetectorInfo]` — provenance of each detector |
| `safety_sweep` | structured-ID sweep metadata (patterns version, source) |
| `spans` | `list[AuditSpan]` — one entry per detected identifier |
| `thresholds` | per-label confidence thresholds used |
| `residual_risk` | projected leakage / re-identification risk summary |
| `openmed_version` | library version that produced the report |
| `manifest_hash` | hash of the model manifest used |
| `document_length` | character length of the input |
| `input_hash` | `sha256:` hash of the original text |
| `deidentified_text_hash` | `sha256:` hash of the de-identified output |
| `repro_hash` | deterministic hash over the canonical payload |
| `signature` | `AuditSignature` once `.sign()` is called (else `None`) |

`AuditSpan` (per identifier, no plaintext): `start`, `end`, `label`,
`canonical_label`, `sources`, `confidence`, `threshold`, `action`, `surrogate`,
`text_hash` (hash of the span text), plus `evidence` and a redacted `context`.

`DetectorInfo`: `source`, `model_id`, `model_format`, `commit`, `metadata`.

`AuditSignature`: `key_id`, `algorithm` (HMAC-SHA256), `value` (hex digest).

## Why audits store offsets + hashes + provenance, never plaintext

The whole point of de-identification is to *not* retain identifiers. An audit
log full of raw names and MRNs would itself be a PHI breach. So OpenMed records:

- **offsets** (`start`/`end`) — *where* an identifier was, not what it was;
- **hashes** (`input_hash`, `deidentified_text_hash`, per-span `text_hash`) —
  prove what was processed without revealing it;
- **provenance** (`detectors`, `manifest_hash`, `openmed_version`) — *which*
  model/version made each call, for reproducibility;
- **residual risk** (`residual_risk`) — projected leakage so a reviewer can
  judge whether the output is safe to release.

This lets an auditor confirm coverage and integrity while the report stays
shareable.

## Workflow

1. **Run with `audit=True`** and a `policy=` to get an `AuditReport`.
2. **Inspect `residual_risk`** before releasing the output — non-zero projected
   leakage means review (cross-check with `auditing-safe-harbor-checklist`).
3. **Sign** with `report.sign(key, key_id=...)` using a secret release HMAC key
   stored in a vault (never in the repo).
4. **Persist** `report.to_dict()` (JSON). For a reviewer view without full
   context, use `report.export_review_bundle()`.
5. **Verify on retrieval** with `report.verify(key)`. Pass `original_text=` /
   `deidentified_text=` to also confirm the texts match the recorded hashes.
6. **Retain** per your retention schedule alongside the de-identified output
   (but keep any reversible mapping separate and encrypted).

## Hand-off to / from OpenMed

- **Produce the de-id:** `deidentifying-clinical-text` (`deidentify`); add
  `audit=True` to get the report from the same call.
- **Coverage check:** `auditing-safe-harbor-checklist` maps span labels to the
  18 HIPAA identifier categories and flags gaps.
- **Policies:** `configuring-privacy-policies` — `resolved_profile` reflects the
  chosen `policy=`.
- **Other surfaces:** MCP `openmed_deidentify` and REST `POST /pii/deidentify`
  drive the same engine; audit output is available through them.

## Edge cases & gotchas

- **`audit=True` changes the return type** to `AuditReport`. You get the report,
  not `.deidentified_text`. Run a separate normal call if you also need the text.
- **An unsigned report is not tamper-evident.** `.sign()` is a deliberate second
  step; `signature` is `None` until you call it. Empty/None keys are rejected.
- **`verify` needs the same key.** It checks the HMAC and the `repro_hash`;
  optionally it re-hashes texts you pass to confirm they are the audited ones.
- **Never put plaintext PHI back into the report.** Do not stuff raw identifiers
  into `metadata`/`evidence`; the design is hash-and-offset only.
- **Store the signing key in a secret manager**, not in source or the report.
- **Residual risk is advisory, not a pass/verdict.** Combine it with the Safe
  Harbor checklist and human review for release decisions.

## Standards & references

- HIPAA de-identification & documentation, 45 CFR 164.514(b): https://www.hhs.gov/hipaa/for-professionals/privacy/special-topics/de-identification/index.html
- HMAC-SHA256: RFC 2104 / FIPS 198-1.
- OpenMed source: `openmed/core/audit.py` (`AuditReport`, `AuditSpan`,
  `AuditSignature`, `DetectorInfo`, `sign`, `verify`), `openmed/core/pii.py`
  (`deidentify(audit=True)`).
