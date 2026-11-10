# Breach Report Template

Use this template for suspected PHI, PII, or personal-data incidents involving
OpenMed workflows. Do not include raw PHI or PII in committed copies, public
tickets, or shared examples. Replace identifying details with hashes, offsets,
counts, and synthetic examples unless counsel approves a controlled system of
record.

## Incident Information

- Incident ID:
- Report version:
- Prepared by:
- Reviewed by:
- Approved by:
- Incident lead:
- Security lead:
- Privacy/compliance lead:
- Legal reviewer:
- Communications owner:
- Data protection officer or privacy contact:
- Time zone used for this report:

## Discovery And Reporting Clock

- Detection timestamp:
- Discovery or awareness timestamp:
- Containment timestamp:
- Initial assessment timestamp:
- Report completion timestamp:
- Notification deadline owner:
- HIPAA 60-day deadline, if applicable:
- GDPR 72-hour deadline, if applicable:
- Reason for any delayed GDPR supervisory authority notice:

## Incident Summary

- What happened:
- How it was detected:
- Systems, services, or integrations involved:
- Whether OpenMed acted as a covered entity, business associate, controller,
  processor, or local tool in another organization's workflow:
- Current status:

## Affected Data

- Data categories involved:
- Types of identifiers involved:
- Approximate number of affected individuals or data subjects:
- Approximate number of affected records:
- Jurisdictions or states involved:
- Whether the data was encrypted, pseudonymized, redacted, or otherwise
  protected:
- Whether the data was actually acquired or viewed:
- Likelihood of misuse or re-identification:

## Timeline

| Timestamp | Source | Event | Evidence reference |
|---|---|---|---|
| | | | |
| | | | |
| | | | |

## Risk Assessment

### HIPAA

- Does the event involve unsecured PHI?
- Does an exception to the definition of breach apply?
- Risk assessment factors considered:
  - Nature and extent of PHI involved:
  - Unauthorized person who used or received the PHI:
  - Whether PHI was actually acquired or viewed:
  - Extent to which the risk was mitigated:
- Individual notice required:
- HHS notice required:
- Media notice required:
- Business associate or covered entity notice required:
- Rationale:

### GDPR

- Does the event involve personal data?
- Is notice to the supervisory authority required?
- Is communication to data subjects required because of likely high risk?
- Controller or processor role:
- Supervisory authority:
- Rationale:

## Required Notice Content

### HIPAA Individual Notice

- Brief description of what happened:
- Date of breach, if known:
- Date of discovery:
- Types of unsecured PHI involved:
- Steps individuals should take:
- Investigation, mitigation, and prevention steps:
- Contact procedures:

### HIPAA HHS Or Media Notice

- Affected count:
- State or jurisdiction count, if media notice may apply:
- HHS submission owner:
- HHS submission date:
- Media notice owner:
- Media notice date:
- Public statement location:

### GDPR Supervisory Authority Notice

- Nature of the personal data breach:
- Categories and approximate number of affected data subjects:
- Categories and approximate number of affected personal data records:
- DPO or other contact point:
- Likely consequences:
- Measures taken or proposed:
- Mitigation of possible adverse effects:
- Phased-notice plan, if facts are incomplete:
- Reasons for delay, if submitted after 72 hours:

### GDPR Data Subject Communication

- Plain-language nature of the breach:
- Contact point:
- Likely consequences:
- Measures taken or proposed:
- Whether an Article 34 exception applies:
- Communication channel:
- Communication date:

## OpenMed Evidence Sources

- Audit report path or controlled-system reference:
- Audit report reproducibility hash:
- Audit report signature status:
- Model, manifest, and policy profile versions:
- Redaction or transformation policy:
- Leakage metrics:
- Residual-risk score:
- Audit diff reference:
- Eval or red-team finding reference:
- Access-log hash:
- Security alert reference:
- Supply-chain or SBOM reference:
- Deployment, release, or configuration reference:

## Containment Actions

- Systems isolated:
- Credentials rotated:
- Integrations disabled:
- Data exports stopped:
- Downstream recipients notified:
- Evidence preserved:
- Verification that exposure stopped:

## Remediation And Prevention

- Root cause:
- Corrective actions completed:
- Preventive controls planned:
- Owner:
- Due date:
- Validation evidence:

## Notifications And Approvals

| Recipient | Required | Owner | Due date | Sent date | Evidence |
|---|---|---|---|---|---|
| Affected individuals | | | | | |
| HHS | | | | | |
| Media | | | | | |
| GDPR supervisory authority | | | | | |
| GDPR data subjects | | | | | |
| Covered entity or business associate | | | | | |
| Internal leadership | | | | | |

## Synthetic Example Attachment

Use this section only for synthetic or fully redacted illustrations.

- Synthetic scenario:
- Synthetic affected record count:
- Synthetic timeline:
- Synthetic evidence references:
- Notes confirming no real PHI or PII is included:
