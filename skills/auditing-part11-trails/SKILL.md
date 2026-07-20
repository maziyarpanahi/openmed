---
name: auditing-part11-trails
description: "Generates and verifies 21 CFR Part 11-style audit trails — who/what/when, electronic signatures, and tamper-evidence — for OpenMed pipelines in GxP and clinical-trial (GCP) settings. Use when the user runs OpenMed in a regulated/validated environment and needs an attributable, time-stamped, tamper-evident record of each processing action, electronic-signature manifestations, or computer-system-validation (CSV) evidence. Trigger keywords: 21 CFR Part 11, Part 11, audit trail, electronic signature, e-signature, GxP, GCP, GLP, GMP, CSV, computer system validation, data integrity, ALCOA, tamper-evident, contemporaneous. Pairs adjacent to OpenMed: maps directly onto OpenMed deidentify(audit=True) -> signed AuditReport with .sign(key)/.verify(key), whose repro_hash + HMAC give the tamper-evidence and attribution Part 11 expects. This is a compliance-enablement aid, not a validation certification."
license: Apache-2.0
metadata:
  project: OpenMed
  category: compliance-regulatory
  pairs: adjacent
  version: "1.0"
---

# Auditing 21 CFR Part 11 trails for OpenMed pipelines

In FDA-regulated **GxP** work (GCP clinical trials, GLP, GMP) any electronic
record used to support a regulatory decision must meet **21 CFR Part 11**: it has
to be **attributable** (who), **contemporaneous and time-stamped** (when),
describe **what** changed, be **tamper-evident**, and — where a signing event
occurs — carry a controlled **electronic signature**. These map onto the
**ALCOA+** data-integrity expectations (Attributable, Legible, Contemporaneous,
Original, Accurate, +Complete/Consistent/Enduring/Available).

OpenMed's `deidentify(..., audit=True)` already emits a deterministic,
**PHI-free `AuditReport`** that you can `.sign()` (HMAC-SHA256) and later
`.verify()`. That gives you the *tamper-evidence* and *attribution* primitives;
this skill wraps them in the who/when/what/e-signature envelope Part 11 wants.

This is a **compliance-enablement aid**. Part 11 compliance also requires
validated systems (CSV), SOPs, and access controls that live outside any single
library — a QA/validation lead signs off.

## When to use

- OpenMed runs inside a validated/GxP environment and each run must leave an
  attributable, tamper-evident record.
- You need to wrap an OpenMed `AuditReport` with **who/when/what** + an
  **e-signature manifestation** (meaning, signer, timestamp).
- You must **verify** a stored trail hasn't been altered, or produce CSV evidence
  for an inspection.

## How OpenMed's AuditReport maps to Part 11

| Part 11 expectation | 21 CFR cite | OpenMed mechanism |
| --- | --- | --- |
| Tamper-evident, accurate copies | 11.10(b),(c) | `AuditReport.to_json()` + `repro_hash` over the canonical payload |
| Audit trail: what changed, when | 11.10(e) | `AuditReport.spans` (action per identifier), `input_hash`/`deidentified_text_hash`, `openmed_version`, `manifest_hash` |
| Operational/authority checks; attribution | 11.10(d),(g) | `AuditSignature.key_id` (signer/key identity) + your envelope's user id |
| Signature manifestation (name, date, meaning) | 11.50 | Your envelope fields `signer`, `signed_at`, `meaning` |
| Signature/record linking, non-repudiation | 11.70, 11.200 | HMAC-SHA256 over the canonical payload via `.sign()` / `.verify()` |

The HMAC binds the signature to *that exact* report content: any later edit to a
span, hash, or field changes `repro_hash`, so `.verify()` fails — that is the
tamper-evidence.

## Quick start

```python
import openmed, json, datetime as dt

note = "Subject S-014 (DOB 1962-08-09) reported headache on 2024-05-01."

# 1) Produce the deterministic, PHI-free audit record for this processing step.
report = openmed.deidentify(note, policy="hipaa_safe_harbor", audit=True)

# 2) Sign it with a controlled release key (stored in a vault / HSM, never in code).
report.sign(b"<release-hmac-key>", key_id="omv-signer-2026")

# 3) Wrap in a Part 11 envelope: who / when / what / signature meaning.
trail = {
    "record": report.to_dict(),              # tamper-evident, no PHI
    "who": "j.smith@sponsor.example",        # authenticated user (your IdP)
    "when": dt.datetime.now(dt.timezone.utc).isoformat(),
    "what": "PHI de-identification of source narrative (study X, subject S-014)",
    "signature_manifestation": {             # 21 CFR 11.50
        "signer_printed_name": "Jane Smith",
        "meaning": "reviewed and approved",
        "signed_at": dt.datetime.now(dt.timezone.utc).isoformat(),
    },
    "system": {"openmed_version": report.openmed_version,
               "manifest_hash": report.manifest_hash},
}
with open("part11_trail.json", "w") as fh:
    json.dump(trail, fh, indent=2, sort_keys=True)

# 4) Later — verify integrity (optionally bind to the exact source/output text).
ok = report.verify(b"<release-hmac-key>", original_text=note)
assert ok, "AUDIT TRAIL TAMPERED OR KEY MISMATCH"
```

## Workflow

1. **Authenticate the actor** in your own IdP/access system (Part 11 needs unique
   IDs and operational checks — outside the library). Capture the user id.
2. **Run the processing step with `audit=True`** to get the deterministic record.
3. **Sign** with a controlled release key from a vault/HSM; record `key_id`.
   Never embed the key in source or the trail.
4. **Build the Part 11 envelope** — who, when (UTC, contemporaneous), what, and
   the **signature manifestation** (printed name, meaning, timestamp) per 11.50.
5. **Append, never overwrite.** Store trails write-once (WORM / append-only
   store). The audit trail itself must be protected and retained.
6. **Verify on retrieval** with `.verify(key, original_text=..., deidentified_text=...)`
   to confirm neither the record nor the bound texts changed.
7. **Retain** per the study/retention schedule; keep keys and any reversible
   mapping in a separate, access-controlled store.

## Hand-off to / from OpenMed

- **Produce the record:** `auditing-deidentification-runs`
  (`deidentify(audit=True)` → `AuditReport`) is the source of the signed,
  PHI-free trail this skill envelopes.
- **Coverage evidence:** `auditing-safe-harbor-checklist` documents that the
  18 identifier categories were handled — useful as a CSV artifact.
- **No-PHI logging:** `enforcing-nophi-logging` ensures the surrounding
  application logs don't leak identifiers into the trail.
- **HIPAA overlap:** `checking-hipaa-compliance` — Part 11 audit controls and the
  HIPAA Security Rule audit-controls standard (164.312(b)) reinforce each other.
- OpenMed runs **on-device**, so the record-generating step stays inside your
  validated boundary.

## Edge cases & gotchas

- **Part 11 ≠ one library.** The signed `AuditReport` gives tamper-evidence and
  attribution, but Part 11 also requires validated systems (CSV), SOPs, training,
  and access controls you implement around it. Don't claim "Part 11 compliant"
  from the audit object alone.
- **Unsigned = not tamper-evident.** `.sign()` is a deliberate step; `signature`
  is `None` until called. Empty/None keys are rejected.
- **Key management is the crux.** The HMAC is only as trustworthy as the key.
  Use a vault/HSM, rotate via `key_id`, and never store the key with the trail.
- **Contemporaneous timestamps.** Use a synchronized, trusted clock (UTC) at the
  moment of the action — back-dating breaks ALCOA "Contemporaneous".
- **Append-only retention.** A trail you can silently overwrite isn't an audit
  trail. Use WORM/append-only storage and protect it from the operators it audits.
- **No PHI in the envelope.** The `AuditReport` is hash-and-offset only; don't
  reintroduce identifiers in the `what`/`who` free-text fields.
- **HMAC is symmetric.** It proves integrity to holders of the key, not public
  non-repudiation. If you need third-party non-repudiation, layer an asymmetric
  signature over `report.to_json()`.

## Standards & references

- 21 CFR Part 11 (Electronic Records; Electronic Signatures): https://www.ecfr.gov/current/title-21/chapter-I/subchapter-A/part-11
- FDA guidance — Part 11 Scope and Application: https://www.fda.gov/regulatory-information/search-fda-guidance-documents/part-11-electronic-records-electronic-signatures-scope-and-application
- FDA — Electronic Systems, Records, and Signatures in Clinical Investigations (2023): https://www.fda.gov/regulatory-information/search-fda-guidance-documents/electronic-systems-electronic-records-and-electronic-signatures-clinical-investigations
- FDA — Data Integrity and Compliance With Drug CGMP (ALCOA+): https://www.fda.gov/regulatory-information/search-fda-guidance-documents/data-integrity-and-compliance-drug-cgmp-questions-and-answers-guidance-industry
- GAMP 5 (ISPE) — risk-based CSV: https://ispe.org/publications/guidance-documents/gamp-5-guide-2nd-edition
- HMAC-SHA256: RFC 2104 / FIPS 198-1. OpenMed source: `openmed/core/audit.py`.
