# Breach Notification Runbook

This runbook helps OpenMed operators coordinate a suspected PHI or PII breach
response. It is an operational checklist, not legal advice. Use it with your
organization's incident response plan, counsel, privacy officer, and local
regulatory requirements.

Use the companion
[breach report template](templates/breach-report-template.md) to record the
incident facts, notification decisions, evidence sources, and approvals.

## First Hour

1. Open an incident record and assign an incident lead, privacy/compliance lead,
   security lead, legal reviewer, and communications owner.
2. Record the discovery or awareness timestamp, time zone, reporting clock
   owner, and how the event was detected.
3. Preserve evidence before modifying affected systems. Capture log hashes,
   audit report hashes, model and policy versions, configuration snapshots, and
   access-control state.
4. Contain the exposure. Disable affected integrations, rotate credentials,
   isolate compromised systems, and stop any data export path.
5. Avoid copying raw PHI or PII into tickets, chat, public issues, committed
   files, screenshots, or report examples. Use offsets, hashes, counts, and
   synthetic examples whenever possible.

## Triage Questions

- Does the event involve unsecured PHI, PII, personal data, audit artifacts, or
  data that can be re-identified when combined with other context?
- Is OpenMed acting in a HIPAA covered entity, business associate, GDPR
  controller, or GDPR processor workflow?
- Was the data actually acquired or viewed, or only exposed?
- Which individuals, jurisdictions, systems, integrations, and records may be
  affected?
- Are protections such as encryption, key separation, pseudonymization, access
  controls, or short-lived credentials enough to lower notification risk?
- Are notifications legally required, voluntarily appropriate, or not required?

## Notification Timelines

Validate final deadlines with counsel before sending notices.

| Framework | Trigger | Clock |
|---|---|---|
| HIPAA individual notice | Breach of unsecured PHI affecting individuals | Without unreasonable delay and no later than 60 calendar days after discovery. |
| HIPAA media notice | Breach affecting more than 500 residents of a state or jurisdiction | Without unreasonable delay and no later than 60 calendar days after discovery. |
| HIPAA notice to HHS, 500 or more individuals | Breach of unsecured PHI affecting 500 or more individuals | Without unreasonable delay and no later than 60 calendar days after discovery. |
| HIPAA notice to HHS, fewer than 500 individuals | Breach of unsecured PHI affecting fewer than 500 individuals | May be logged and submitted annually, no later than 60 days after the end of the calendar year in which it was discovered. |
| HIPAA business associate notice | Breach at or by a business associate | Notify the covered entity without unreasonable delay and no later than 60 days after discovery. |
| GDPR supervisory authority notice | Personal data breach likely to result in risk to rights and freedoms | Without undue delay and, where feasible, within 72 hours after becoming aware. Include reasons if later than 72 hours. |
| GDPR data subject communication | Personal data breach likely to result in high risk to rights and freedoms | Without undue delay, unless an Article 34 exception applies. |

Authoritative references:

- [HHS HIPAA Breach Notification Rule](https://www.hhs.gov/hipaa/for-professionals/breach-notification/index.html)
- [GDPR Articles 33 and 34](https://eur-lex.europa.eu/legal-content/EN/TXT/HTML/?uri=CELEX%3A02016R0679-20160504)

## Required Notice Content

For HIPAA individual notices, prepare:

- A brief description of what happened, including breach and discovery dates if
  known.
- The types of unsecured PHI involved.
- Steps affected individuals should take to protect themselves.
- What the organization is doing to investigate, mitigate harm, and prevent
  recurrence.
- Contact procedures for questions, including the required contact channel.

For GDPR supervisory authority notices, prepare:

- The nature of the personal data breach.
- Categories and approximate numbers of affected data subjects.
- Categories and approximate numbers of personal data records.
- Data protection officer or other contact point details.
- Likely consequences.
- Measures taken or proposed, including mitigation.
- Reasons for delay if notice is not made within 72 hours.

For GDPR data subject communications, use clear plain language and include the
nature of the breach, contact point, likely consequences, and measures taken or
proposed.

## OpenMed Evidence Sources

Collect privacy-preserving evidence that supports the incident timeline and
notification assessment:

- [Compliance posture](../compliance.md) evidence: policy profile, audit trail,
  leakage metrics, residual-risk scores, and benchmark outputs.
- [Audit reports](https://github.com/maziyarpanahi/openmed/blob/master/openmed/core/audit.py):
  span provenance, manifest hashes, residual-risk snapshots, and optional
  signatures.
- [Audit diffs](https://github.com/maziyarpanahi/openmed/blob/master/openmed/risk/audit_diff.py):
  before/after comparisons that avoid storing plaintext identifiers.
- [Eval harness metrics](../eval-harness.md#metric-bundle): leakage and
  reproducibility outputs from synthetic or approved local fixtures.
- [Security disclosure workflow](../security/disclosure-policy.md): private
  handling path for redaction bypasses or PHI/PII leaks.
- [Supply chain controls](../security/supply-chain.md) and
  [SBOM evidence](../security/sbom.md): dependency, build, and release context.
- Red-team, adversarial, or security findings, stored without real PHI/PII.

## Containment And Investigation

1. Freeze relevant logs and audit artifacts using immutable storage if
   available.
2. Disable exposed routes, keys, tokens, connectors, or batch jobs.
3. Rotate secrets and revoke affected sessions.
4. Identify all downstream consumers and replicated stores.
5. Reconstruct the event timeline from alerts, audit reports, access logs,
   deployment records, and operator actions.
6. Estimate affected individuals and records. Keep both best-current estimates
   and confidence levels.
7. Reassess after each new fact. If the notification decision changes, record
   who approved the change and when.

## Synthetic Worked Example

### Scenario

An internal staging deployment exported 125 synthetic patient-like records to a
misconfigured object-storage path. The records were generated fixtures and did
not contain real PHI or PII.

### Timeline

| Time (UTC) | Event |
|---|---|
| 09:15 | Monitoring alert showed anonymous reads from the staging path. |
| 09:25 | Incident lead opened `OM-IR-2026-001` and assigned privacy, legal, and security owners. |
| 10:05 | Security disabled public access and rotated the storage credential. |
| 11:30 | Audit reports confirmed fixture provenance and no production export job. |
| 13:30 | Privacy lead documented that statutory breach notices were not required because only synthetic records were involved. |

### Evidence Preserved

- Storage access-log hash and request counts.
- OpenMed audit report hash for the fixture export.
- Dataset-generation script commit hash.
- Containment action timestamps.
- Legal and privacy approval notes.

### Follow-Up

- Add a release gate that rejects public object-storage policies for staging
  exports.
- Add the incident to tabletop training.
- Keep the incident report synthetic and do not add real records to the
  repository.
