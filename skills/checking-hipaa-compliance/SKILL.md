---
name: checking-hipaa-compliance
description: "Runs a HIPAA Privacy and Security Rule checklist over a data pipeline and produces a gap report before deploying OpenMed on PHI. Use when the user is about to process protected health information, needs a pre-deployment compliance review, wants to know which administrative, physical, and technical safeguards apply, is scoping a Business Associate Agreement, or must document minimum-necessary and de-identification controls. Trigger keywords: HIPAA, Privacy Rule, Security Rule, 45 CFR 164, PHI, BAA, business associate, minimum necessary, safeguards, Safe Harbor, Expert Determination, gap analysis, compliance review. Pairs adjacent to OpenMed: the checklist shows where openmed.deidentify and signed audit reports satisfy the de-identification and audit-control requirements. The control list lives in references/hipaa-checklist.md. This is a structured self-assessment aid, not legal advice."
license: Apache-2.0
metadata:
  project: OpenMed
  category: compliance-regulatory
  pairs: adjacent
  version: "1.0"
---

# Checking HIPAA compliance before deploying on PHI

Before any pipeline touches **protected health information (PHI)**, the
operating entity (a covered entity or its business associate) must have the
HIPAA **Privacy Rule** and **Security Rule** safeguards in place. This skill
walks a concrete pipeline against those requirements and emits a **gap report**:
which controls are met, which are missing, and where OpenMed's on-device
de-identification and signed audit trail satisfy a requirement.

The full control list — administrative, physical, and technical safeguards with
their 45 CFR citations — is in
[references/hipaa-checklist.md](references/hipaa-checklist.md). This skill is a
**self-assessment aid**, not legal advice; a Privacy Officer signs off on
compliance.

## When to use

- You are about to deploy OpenMed (or any pipeline) on real PHI and need a
  go/no-go compliance review.
- You must document **safeguards** and **minimum-necessary** controls for an
  audit, a BAA, or a security questionnaire.
- You want to decide between **Safe Harbor** and **Expert Determination**
  de-identification and record the rationale.
- You need a reproducible gap report you can hand to a Privacy/Security Officer.

## The two paths to "no longer PHI"

HIPAA recognizes two de-identification methods (45 CFR 164.514):

1. **Safe Harbor** — remove all **18 identifier categories** and have no actual
   knowledge the result can re-identify. Deterministic, the common path.
2. **Expert Determination** — a qualified statistician certifies "very small"
   re-identification risk. Used when you must keep some quasi-identifiers.

OpenMed's `deidentify(..., policy="hipaa_safe_harbor")` targets the Safe Harbor
identifier set on-device, and `deidentify(..., audit=True)` produces a signed,
PHI-free `AuditReport` that documents *what* was removed — the evidence a Safe
Harbor attestation and a Security Rule audit control both want.

## Quick start

```python
import openmed

# A representative record from the pipeline (synthetic — never log real PHI).
sample = "John Doe (MRN 1234567), DOB 1970-01-15, seen 2024-03-02 in Boston."

# 1) De-identify on-device under the Safe Harbor policy.
result = openmed.deidentify(sample, method="replace", policy="hipaa_safe_harbor")
print(result.deidentified_text)        # identifiers removed/surrogated

# 2) Produce the signed, no-PHI audit record for the compliance file.
report = openmed.deidentify(sample, policy="hipaa_safe_harbor", audit=True)
report.sign(b"<release-hmac-key-from-vault>", key_id="hipaa-2026")

# 3) Walk the checklist (see references/hipaa-checklist.md) and record gaps.
controls = {
    "encryption_at_rest": True,
    "encryption_in_transit": True,
    "access_controls_rbac": True,
    "audit_logging": True,          # satisfied in part by the signed AuditReport
    "minimum_necessary": False,     # <-- gap: pipeline pulls full notes
    "baa_in_place": True,
    "deidentification_method": "safe_harbor",
}
gaps = [name for name, ok in controls.items() if not ok]
print("GAPS:", gaps)
```

## Workflow

1. **Map the data flow.** Diagram every place PHI is created, received,
   maintained, or transmitted — including model caches, temp files, and logs.
2. **Confirm the legal basis.** Is the operator a covered entity or business
   associate? Is a **BAA** in place with every downstream vendor that touches
   PHI? OpenMed running on-device means *no* third-party processor for the NLP
   step — note that as a control in your favor.
3. **Run the three safeguard groups** from
   [references/hipaa-checklist.md](references/hipaa-checklist.md):
   administrative (risk analysis, workforce training, sanctions),
   physical (facility/device controls), and technical (access control, audit
   controls, integrity, transmission security).
4. **Enforce minimum necessary.** Pull only the fields the task needs; mask the
   rest. De-identify as early in the flow as the use case allows.
5. **Record the de-identification method** (Safe Harbor vs Expert Determination)
   and attach the signed `AuditReport` as evidence.
6. **Emit the gap report** — met / not-met / N/A per control, with the 45 CFR
   citation and the remediation owner. Hand it to the Privacy Officer.

## Hand-off to / from OpenMed

- **De-identification:** `deidentifying-clinical-text` (`openmed.deidentify`,
  `policy="hipaa_safe_harbor"`) is the technical control that converts PHI to
  non-PHI on-device — the heart of a HIPAA pipeline.
- **Identifier coverage:** `auditing-safe-harbor-checklist` maps detected spans
  to the 18 Safe Harbor categories so you can prove each is handled.
- **Audit control:** `auditing-deidentification-runs` (`audit=True` →
  `AuditReport.sign()/.verify()`) gives the tamper-evident, PHI-free record the
  Security Rule audit-controls standard (164.312(b)) expects.
- **No-PHI logging:** `enforcing-nophi-logging` keeps identifiers out of logs and
  traces (a recurring audit finding).
- OpenMed is **local-first** — the NLP step adds no new business associate.

## Edge cases & gotchas

- **De-identified data is out of scope — but only if done right.** Safe Harbor
  requires *all 18* categories removed *and* no actual knowledge of
  re-identifiability. A residual rare ZIP3 or a free-text name the model missed
  re-introduces PHI. Verify coverage; don't assume.
- **Limited Data Sets are still PHI.** Dates and ZIPs retained under a Data Use
  Agreement (164.514(e)) are not de-identified — different rules apply.
- **Logs and caches are PHI too.** Model caches, exception messages, and temp
  files holding raw notes are in scope. This is the most common gap.
- **A BAA is required for every vendor** that creates/receives/maintains/
  transmits PHI on your behalf — including cloud storage and any LLM API. Running
  OpenMed on-device avoids adding one for the NLP step.
- **Minimum necessary is a duty, not a nicety** (164.502(b)). Don't pull full
  charts when a problem list suffices.
- **Breach notification clock.** Unsecured PHI exposure triggers 164.400-414
  duties; encryption to NIST standards renders data "secured" and can avoid the
  notification trigger.
- **Not legal advice.** This checklist supports, but does not replace, a Privacy
  Officer's determination and (for Expert Determination) a qualified statistician.

## Standards & references

- HHS HIPAA for Professionals (hub): https://www.hhs.gov/hipaa/for-professionals/index.html
- Privacy Rule, 45 CFR Part 164 Subpart E: https://www.ecfr.gov/current/title-45/subtitle-A/subchapter-C/part-164/subpart-E
- Security Rule, 45 CFR Part 164 Subpart C: https://www.ecfr.gov/current/title-45/subtitle-A/subchapter-C/part-164/subpart-C
- De-identification guidance (Safe Harbor & Expert Determination), 164.514: https://www.hhs.gov/hipaa/for-professionals/privacy/special-topics/de-identification/index.html
- HHS Security Risk Assessment Tool: https://www.healthit.gov/topic/privacy-security-and-hipaa/security-risk-assessment-tool
- NIST SP 800-66r2 (implementing the Security Rule): https://csrc.nist.gov/pubs/sp/800/66/r2/final
