# HIPAA safeguards checklist (45 CFR Part 164)

A control-by-control checklist for a pipeline that processes PHI. Each row cites
its standard. `R` = Required, `A` = Addressable (you must implement it, adopt an
equivalent alternative, or document why it is not reasonable — *not* "optional").
"OpenMed fit" notes where OpenMed satisfies or supports the control.

This is a self-assessment aid. A Privacy/Security Officer makes the final
determination; this is not legal advice.

---

## 1. Administrative safeguards — 45 CFR 164.308

| # | Control | Cite | R/A | Met? | OpenMed fit |
|---|---------|------|-----|------|-------------|
| A1 | Security Management Process: conduct an accurate, thorough **risk analysis** | 164.308(a)(1)(ii)(A) | R | ☐ | De-id step adds no external processor (on-device) |
| A2 | **Risk management** — reduce risks to a reasonable level | 164.308(a)(1)(ii)(B) | R | ☐ | Early de-identification shrinks PHI surface |
| A3 | **Sanction policy** for workforce violations | 164.308(a)(1)(ii)(C) | R | ☐ | — |
| A4 | **Information system activity review** (audit log review) | 164.308(a)(1)(ii)(D) | R | ☐ | Signed `AuditReport` feeds review |
| A5 | Assigned **security responsibility** (named Security Official) | 164.308(a)(2) | R | ☐ | — |
| A6 | **Workforce authorization / supervision / clearance** | 164.308(a)(3) | A | ☐ | — |
| A7 | **Information access management** — isolate, authorize, modify access | 164.308(a)(4) | R/A | ☐ | RBAC on de-id outputs |
| A8 | **Security awareness & training** (reminders, malware, login, passwords) | 164.308(a)(5) | A | ☐ | — |
| A9 | **Security incident procedures** — identify, respond, report | 164.308(a)(6) | R | ☐ | — |
| A10 | **Contingency plan** — backup, DR, emergency mode, testing | 164.308(a)(7) | R/A | ☐ | — |
| A11 | **Evaluation** — periodic technical & non-technical review | 164.308(a)(8) | R | ☐ | Re-run this checklist on changes |
| A12 | **Business Associate Agreement** with every vendor touching PHI | 164.308(b)(1), 164.314(a) | R | ☐ | On-device NLP needs no new BAA |

## 2. Physical safeguards — 45 CFR 164.310

| # | Control | Cite | R/A | Met? | OpenMed fit |
|---|---------|------|-----|------|-------------|
| P1 | **Facility access controls** — contingency, security plan, validation | 164.310(a) | A | ☐ | — |
| P2 | **Workstation use** policy (appropriate functions/manner) | 164.310(b) | R | ☐ | — |
| P3 | **Workstation security** (physical safeguards for access) | 164.310(c) | R | ☐ | — |
| P4 | **Device & media controls** — disposal, re-use, accountability, backup | 164.310(d) | R/A | ☐ | Wipe model caches that held PHI |

## 3. Technical safeguards — 45 CFR 164.312

| # | Control | Cite | R/A | Met? | OpenMed fit |
|---|---------|------|-----|------|-------------|
| T1 | **Access control** — unique user ID, emergency access, auto-logoff | 164.312(a)(1) | R/A | ☐ | — |
| T2 | **Encryption/decryption at rest** | 164.312(a)(2)(iv) | A | ☐ | Encrypt de-id outputs & any mapping |
| T3 | **Audit controls** — record & examine system activity | 164.312(b) | R | ☐ | **Signed `AuditReport` (audit=True)** documents every redaction |
| T4 | **Integrity** — protect PHI from improper alteration/destruction | 164.312(c) | R/A | ☐ | `AuditReport.verify()` = tamper-evidence (HMAC) |
| T5 | **Person/entity authentication** | 164.312(d) | R | ☐ | — |
| T6 | **Transmission security** — integrity controls + **encryption in transit** | 164.312(e) | R/A | ☐ | TLS on any export; HTTPS-only to OpenFDA etc. |

## 4. Privacy Rule controls — 45 CFR Part 164 Subpart E

| # | Control | Cite | Met? | OpenMed fit |
|---|---------|------|------|-------------|
| V1 | **Minimum necessary** — limit PHI to the purpose | 164.502(b), 164.514(d) | ☐ | Mask fields the task doesn't need |
| V2 | **De-identification** — Safe Harbor (18 IDs) or Expert Determination | 164.514(a)–(c) | ☐ | `deidentify(policy="hipaa_safe_harbor")` |
| V3 | **Limited Data Set** + Data Use Agreement (if dates/ZIP retained) | 164.514(e) | ☐ | Still PHI — different controls |
| V4 | **Uses & disclosures** authorized or permitted | 164.502, 164.506, 164.508 | ☐ | — |
| V5 | **Accounting of disclosures** | 164.528 | ☐ | Audit trail supports records |
| V6 | **Breach notification** assessment & duties | 164.400–414 | ☐ | NIST-grade encryption may avoid trigger |

## 5. The 18 HIPAA Safe Harbor identifiers — 164.514(b)(2)

All must be removed for Safe Harbor de-identification:

1. Names
2. Geographic subdivisions smaller than a state (street, city, county, precinct,
   ZIP and equivalent — ZIP3 only retainable when the area population > 20,000)
3. All date elements (except year) directly related to an individual; ages > 89
   and all elements of dates indicating such an age aggregated to "90+"
4. Telephone numbers
5. Fax numbers
6. Email addresses
7. Social Security numbers
8. Medical record numbers
9. Health plan beneficiary numbers
10. Account numbers
11. Certificate/license numbers
12. Vehicle identifiers and serial numbers (incl. license plates)
13. Device identifiers and serial numbers
14. Web URLs
15. IP addresses
16. Biometric identifiers (finger/voice prints)
17. Full-face photographs and comparable images
18. Any other unique identifying number, characteristic, or code

Plus the residual condition: **no actual knowledge** the remaining data could
identify the individual.

> Cross-check span coverage with the `auditing-safe-harbor-checklist` skill,
> which maps OpenMed detection labels to these 18 categories.
