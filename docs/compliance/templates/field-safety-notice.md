# Field-Safety Notice Template

Use this template to tell deployers about a safety-relevant defect in a
released OpenMed version or artifact. Replace every bracketed placeholder
with reviewed information before distribution.

OpenMed is not a medical device. This template makes no diagnostic,
certification, or regulatory compliance claim. Every notice requires human
sign-off; completing the template does not authorize publication.

Include only version identifiers, artifact digests, aggregate counts, and
synthetic or redacted examples. Do not include patient data, credentials,
private paths, or unreviewed reproduction details.

## Summary

- Notice ID: [opaque notice identifier]
- Notice version and issue date: [revision and date with time zone]
- Status: [draft, approved, superseded, or withdrawn]
- Subject: [short description of the confirmed defect]
- Intended recipients: [affected deployment roles]
- Previous notice: [public reference or not applicable]
- Human approval: [approving role, approval reference, and date]

## Affected Versions And Artifacts

- Library or SDK versions: [inclusive affected version range]
- Model or other artifact revisions: [repository identifiers and pinned revisions]
- Artifact digests: [reviewed digests or not applicable]
- Relevant settings or operating conditions: [affected configurations]
- How to determine whether a deployment is affected: [local check]
- Known unaffected versions or configurations: [verified scope or not established]

## What Can Go Wrong

- Observed failure: [plain-language description]
- Triggering conditions: [reviewed conditions using synthetic examples only]
- Potential impact: [consequences for affected workflows]
- Evidence: [aggregate results and opaque evidence references]
- Scope and uncertainty: [what is confirmed and what remains under review]

Do not interpret absence of a reported incident as proof that no exposure
occurred. Route suspected identifier leakage privately under the security
policy linked below.

## What To Do Now

- Immediate action: [reviewed containment or mitigation steps]
- Responsible deployment role: [role]
- Action deadline: [date and time zone, or reviewed urgency]
- Temporary workaround and limitations: [approved steps or none available]
- Previously produced outputs: [review, quarantine, or reprocessing instructions]
- Acknowledgement: [approved channel and required confirmation]

## Fixed Version

- Fixed library or SDK version: [version or fix pending]
- Fixed artifact revision and digest: [revision and digest or not applicable]
- Release reference: [reviewed release notes]
- Upgrade or rollback procedure: [approved instructions]
- Remaining limitations: [known residual limitations]

## How To Verify

- Synthetic regression check: [offline command or reproducible procedure]
- Expected outcome: [controlled result or aggregate acceptance criterion]
- Evidence to retain: [versions, digests, and aggregate results only]
- If verification fails: [stop condition and escalation path]
- Human confirmation before resuming affected workflows: [reviewing role and record]

## Contact Path

- Notice questions and acknowledgement: [approved support channel]
- Security defects or suspected identifier leakage: use the private reporting
  route in the [Security Policy](https://github.com/maziyarpanahi/openmed/blob/master/SECURITY.md).
  Do not attach real patient data.
- Suspected personal-data incident: follow the
  [breach-notification runbook](../breach-notification-runbook.md) and record
  reviewed facts in the [breach-report template](breach-report-template.md).
- Next update: [date, approved channel, and accountable role]
