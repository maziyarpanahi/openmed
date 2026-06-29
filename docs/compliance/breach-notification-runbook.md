# Breach Notification Runbook

## Purpose

This runbook provides guidance for responding to potential PHI or PII breaches involving OpenMed deployments and associated workflows.

It is intended to support operators in meeting obligations under:

* HIPAA Breach Notification Rule
* GDPR Articles 33 and 34

This document does not constitute legal advice and should be used alongside internal compliance procedures and legal review.

---

## 1. Detection

Potential breaches may be identified through:

* Audit reports and audit-chain evidence
* Security tooling and monitoring alerts
* Red-team exercises and findings
* User reports or third-party notifications
* Access-log anomalies and unexpected data movement

Record the detection timestamp immediately after confirmation.

---

## 2. Containment

Initial containment actions may include:

* Isolating affected systems
* Revoking compromised credentials
* Disabling affected integrations
* Preserving forensic evidence
* Preventing further unauthorized access

---

## 3. Assessment

Determine:

* Categories of affected information
* Number of impacted records
* Systems and services involved
* Whether PHI or PII was exposed
* Likelihood of misuse or unauthorized disclosure

---

## 4. Roles and Responsibilities

| Role                | Responsibility                      |
| ------------------- | ----------------------------------- |
| Incident Lead       | Coordinates incident response       |
| Security Team       | Investigation and containment       |
| Compliance Team     | Regulatory assessment and reporting |
| Legal Team          | Jurisdictional review               |
| Communications Team | Stakeholder notifications           |

---

## 5. HIPAA Notification Requirements

Under the HIPAA Breach Notification Rule:

* Affected individuals must generally be notified without unreasonable delay and no later than 60 calendar days after discovery.
* Breaches involving more than 500 individuals may require notification to HHS and media outlets as applicable.

---

## 6. GDPR Notification Requirements

Under GDPR Article 33:

* Supervisory authorities should generally be notified within 72 hours of becoming aware of the breach unless the breach is unlikely to result in risk to individuals.

Under GDPR Article 34:

* Affected individuals should be informed when the breach is likely to result in a high risk to their rights and freedoms.

---

## 7. Evidence Sources

Evidence may include:

* OpenMed audit reports
* Signed audit artifacts
* Audit-chain records and provenance data
* Security logs
* Access records
* Red-team findings
* Incident timelines
* System snapshots

---

## 8. Synthetic Example

### Incident Summary

A synthetic dataset containing 125 non-production patient records was unintentionally exposed through an incorrectly configured storage policy.

### Timeline

* Detection: 09:15 UTC
* Containment: 10:05 UTC
* Investigation completed: 13:30 UTC

### Actions Taken

* Public access removed
* Access credentials rotated
* Audit reports reviewed
* Security controls updated

No real patient information was involved.
