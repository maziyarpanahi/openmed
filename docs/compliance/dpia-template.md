# Data Protection Impact Assessment Template

> **Template — requires legal review. Not legal advice.** Adapt this document to
> the actual processing, jurisdiction, supervisory-authority guidance, and
> deployment controls before relying on it.

This template helps a controller document a Data Protection Impact Assessment
(DPIA) for a deployment that uses OpenMed. It is structured around GDPR Article
35 and the OpenMed de-identification evidence that can inform, but cannot replace,
the controller's assessment.

Replace every `[bracketed placeholder]`. Record only references, aggregate
metrics, hashes, and approved classifications in this document. Do not paste raw
health data, identifiers, de-identification mappings, or cleartext audit spans.

## 1. Document Control

| Field | Entry |
|---|---|
| Controller and business unit | [Legal entity and owner] |
| Joint controllers | [None, or legal entities and allocation reference] |
| Processors and sub-processors | [Legal entities, services, locations, and DPA references] |
| Processing activity | [Name and short description] |
| Planned launch and end dates | [Dates, or ongoing with review trigger] |
| DPIA version | [Version and change summary] |
| DPIA owner | [Name or role] |
| Data protection officer | [Name or role; or not appointed, with reason] |
| Security reviewer | [Name or role] |
| OpenMed version and model | [Version, model id, model revision, artifact hash] |
| Deployment environment | [Device, workstation, private service, or other] |
| Status | [Draft / approved / rejected / prior consultation required] |
| Approval date | [YYYY-MM-DD] |
| Next review | [Date or triggering event] |
| Assessment scope and exclusions | [Systems, processing stages, and explicit exclusions] |
| Publication and sharing | [Public summary, internal circulation, or reason not published] |
| Related records | [ROPA, retention policy, incident plan, DPA, model card] |

## 2. Screening and Consultation Decision

### Why a DPIA is or is not required

- Nature, scope, context, and purposes of the processing: [Assessment]
- Special-category or health data involved: [Yes/no and categories]
- Scale, frequency, duration, and number of data subjects: [Assessment]
- Vulnerable data subjects, systematic monitoring, matching, profiling, or
  automated decisions: [Assessment]
- New technology or material change to an existing system: [Assessment]
- Applicable supervisory-authority DPIA list or sector rule: [Citation and
  conclusion]
- Final screening decision and rationale: [Decision]

### Consultation record

| Consultation | Record |
|---|---|
| DPO advice | [Advice, date, and how it was addressed] |
| Security and clinical safety advice | [Advice and disposition] |
| Data-subject or representative views | [Views, or why consultation was not appropriate] |
| Prior supervisory-authority consultation | [Not required / required under Article 36 / reference] |

If high residual risk remains after reasonable measures, stop the processing
until counsel and the DPO determine whether prior consultation is required.

## 3. Processing Description

### Purpose and outcome

- Intended purpose: [Specific, explicit purpose]
- Expected benefit: [Benefit to patients, staff, research, or operations]
- Output and downstream use: [De-identified text, structured entities, review
  bundle, or other]
- Prohibited uses: [Uses outside the approved purpose]
- Human decisions influenced by the output: [None, or describe safeguards]

### Data inventory

| Item | Description |
|---|---|
| Data subjects | [Patients, clinicians, staff, research participants, or other] |
| Personal-data categories | [Identifiers, demographics, contact data, or other] |
| Special-category data | [Health, genetic, biometric, or other Article 9 data] |
| Sources | [Systems and collection channels] |
| Recipients | [Roles, systems, legal entities, and locations] |
| Volume and frequency | [Records, size, cadence, peak load] |
| Retention | [Input, output, audit evidence, mapping, backups] |
| International transfers | [None, or transfer mechanism and destination] |

### OpenMed data flow

1. `[input source]` supplies clinical content inside `[controlled environment]`.
2. OpenMed loads `[local model path or approved model id/revision]` and applies
   `[policy literal]` with `[method and threshold configuration]`.
3. The deployment stores or forwards only `[approved output]` to `[recipient]`.
4. The deployment records a privacy-safe audit report containing offsets,
   hashes, provenance, configuration, and residual-risk evidence, not raw span
   text.
5. If reversible pseudonymization is enabled, the mapping and its key are stored
   separately under the key-custody controls in section 7.
6. Input, output, caches, logs, temporary files, and backups follow `[retention
   and deletion controls]`.

Document every network boundary. OpenMed supports local processing after model
download and collects no telemetry by default, but the controller must separately
assess model acquisition, service wrappers, storage, monitoring, backups, and any
third-party infrastructure.

## 4. Policy Profile and Configuration

Use the exact runtime literal and attach the resolved profile from the audit
report. A profile is a technical control, not a legal conclusion.

| Exact policy literal | OpenMed posture | Deployment decision |
|---|---|---|
| `hipaa_safe_harbor` | Masks configured identifier and clinical labels; `keep_mapping=False` and `reversible_id=False`. It does not establish GDPR anonymization. | [Used / not used; why] |
| `gdpr_pseudonymization` | Replaces identifiers with reversible pseudonyms; `keep_mapping=True` and `reversible_id=True`. Pseudonymized data remains personal data. | [Used / not used; why] |

- Selected literal: `[hipaa_safe_harbor or gdpr_pseudonymization]`
- OpenMed and model version: [Values]
- Model artifact and manifest hashes: [Values]
- Languages and document types: [Values]
- Detection thresholds and safety-sweep state: [Values]
- Replacement, date-shifting, and consistency settings: [Values]
- Unsupported inputs and fallback behavior: [Values]
- Configuration owner and change-control reference: [Values]

## 5. Lawfulness, Necessity, and Proportionality

| Question | Assessment |
|---|---|
| GDPR Article 6 lawful basis | [Basis and legal analysis] |
| GDPR Article 9 condition | [Condition for special-category data] |
| Purpose limitation | [Why each operation is compatible with the stated purpose] |
| Data minimization | [Fields removed, excluded, or transformed before processing] |
| Accuracy | [Validation, correction, and version-control process] |
| Storage limitation | [Retention periods and deletion enforcement] |
| Transparency | [Notice supplied to data subjects] |
| Rights handling | [Access, rectification, erasure, restriction, objection, portability] |
| Approved codes of conduct | [Applicable code and adherence evidence, or none] |
| Less intrusive alternatives | [Alternatives assessed and why rejected] |
| Proportionality | [Why benefits justify the remaining interference] |

State whether de-identification is an objective in itself or a safeguard for a
separate processing purpose. Pseudonymization does not create a new lawful basis
and does not remove data-subject rights.

## 6. OpenMed Evidence and Residual Risk

Record evidence for the actual model, language, document type, policy, thresholds,
and deployment device. Do not substitute aggregate benchmark performance for
local validation.

| Evidence | Source | Recorded value or reference |
|---|---|---|
| Policy and resolved configuration | `AuditReport.policy` and `AuditReport.resolved_profile` | [Value/reference] |
| Audit residual-risk score | `AuditReport.residual_risk.risk_report_record_score` | [0.0–1.0 value, run range, and report hash] |
| Confidence-derived leakage indicator | `AuditReport.residual_risk.projected_leakage` | [Value and interpretation] |
| Re-identification details | `AuditReport.residual_risk.risk_report` | [Leakage/re-id rates and privacy-safe summary] |
| Direct-identifier recall | `BenchmarkReport.metrics.recall_slices` | [Value, confidence interval, suite hash] |
| Character leakage | `BenchmarkReport.metrics.leakage.overall` | [Overall and critical-label results] |
| Span accuracy | `BenchmarkReport.metrics.exact_span_f1.f1` and `relaxed_span_f1.f1` | [Values] |
| Robustness | [Perturbation/adversarial `BenchmarkReport` references] | [Values and failures] |

`risk_report_record_score` is a technical record-level indicator derived from
measured leakage, re-identification rate, and singleton risk. It is not the DPIA's
overall risk rating. Interpret it together with severity, scale, downstream
linkability, adversary access, key compromise, and consequences for data
subjects.

### Risk-assessment method and baseline

- Method and source: [Method, guidance, or organizational standard]
- Likelihood and severity scales: [Definitions, evidence, and time horizon]
- Risk-prioritization and acceptance rules: [Matrix, thresholds, and approver]
- Baseline for comparison: [Risk before planned controls / with current controls]
- Normal-operation risks: [Harms possible even when processing works as designed]
- Failure, accidental, and adversarial risks: [Errors, outages, misuse, or attacks]

Do not reduce likelihood because a safeguard is merely planned. Record its
implementation status, evidence, and tested effectiveness before taking credit
for it in the residual-risk rating.

### Risk register

Use one row per credible risk and assess it from the data subject's perspective.

| Risk scenario | Affected rights or harms | Existing controls | Likelihood | Severity | Residual risk | Owner and due date |
|---|---|---|---|---|---|---|
| Identifier missed by the model, rule layer, or safety sweep | Confidentiality loss, discrimination, distress, identity harm | Local benchmark, direct-identifier recall gate, human sampling | [L/M/H] | [L/M/H] | [L/M/H] | [Owner/date] |
| Reversible mapping or pseudonymization key is exposed | Re-identification and unauthorized disclosure | Separated key store, least privilege, rotation, access logging | [L/M/H] | [L/M/H] | [L/M/H] | [Owner/date] |
| Output is linked with auxiliary data | Singling out or inference of identity | Recipient limits, minimization, linkage testing, contractual controls | [L/M/H] | [L/M/H] | [L/M/H] | [Owner/date] |
| Model underperforms for a language, format, or population | Unequal protection and hidden leakage | Representative fixtures, per-language metrics, abstention or block | [L/M/H] | [L/M/H] | [L/M/H] | [Owner/date] |
| Raw content enters logs, traces, caches, or support channels | Unauthorized disclosure and loss of control | No-raw-PHI policy, static tests, encrypted storage, deletion | [L/M/H] | [L/M/H] | [L/M/H] | [Owner/date] |
| Incorrect reliance on de-identified output | Unlawful disclosure or incompatible reuse | Legal review, recipient validation, purpose and use restrictions | [L/M/H] | [L/M/H] | [L/M/H] | [Owner/date] |

Overall residual-risk decision: [Accept / remediate / reject / consult authority].

Approved quantitative and qualitative thresholds: [Thresholds, approver, and
rationale].

## 7. Measures and Safeguards

For every measure below, record `[planned / partially implemented / implemented;
owner; evidence; tested effectiveness]` rather than only naming the control.

### Technical measures

- Run locally or on approved infrastructure with network egress restricted after
  model acquisition: [Control]
- Encrypt inputs, outputs, audit reports, mappings, keys, and backups: [Control]
- Prevent raw PHI or personal data in logs, traces, metrics, support bundles, and
  exception messages: [Control]
- Pin and verify OpenMed, model, manifest, and configuration versions: [Control]
- Validate direct-identifier recall, critical leakage, span integrity, and
  robustness on representative synthetic or lawfully obtained fixtures: [Control]
- Block unsupported languages, formats, or low-confidence outputs: [Control]
- Apply access control, authentication, least privilege, and tamper-evident audit
  retention: [Control]
- Test deletion, backup expiry, incident response, and recovery: [Control]

### Reversible pseudonymization and key custody

Complete this subsection when using `gdpr_pseudonymization`.

- Mapping owner and approved re-identification purpose: [Values]
- Key-management system and physical/legal location: [Values]
- Separation between pseudonymized data, mapping, and key: [Control]
- Roles allowed to create, read, rotate, recover, or destroy keys: [Roles]
- Rotation, escrow, recovery, revocation, and deletion schedule: [Process]
- Authorization and audit process for re-identification: [Process]
- Response to suspected key or mapping compromise: [Process]

### Organizational measures

- Training, confidentiality, and acceptable-use obligations: [Controls]
- Human review, escalation, and override authority: [Controls]
- Processor and sub-processor contracts: [References]
- Data-subject request and complaint handling: [Process]
- Incident detection, notification, and evidence preservation: [Process]

## 8. Approval and Review

| Role | Decision, name, date |
|---|---|
| Controller/business owner | [Decision] |
| Data protection officer | [Advice] |
| Security owner | [Decision] |
| Clinical safety or domain owner | [Decision] |
| Legal reviewer | [Decision] |

Reopen this DPIA after a purpose, dataset, model, language, policy, threshold,
deployment boundary, recipient, key-custody, retention, legal, or material-risk
change, and after relevant incidents or significant validation regressions.

## Primary References

- [GDPR, including Articles 28, 32, 35, and 36](https://eur-lex.europa.eu/eli/reg/2016/679/oj)
- [EDPB data protection impact assessment guidance](https://www.edpb.europa.eu/topics/accountability-and-compliance-tools/data-protection-impact-assessment_en)
- [EDPB DPIA template consultation page (2026 draft, not final guidance)](https://www.edpb.europa.eu/public-consultations/template-for-data-protection-impact-assessment_en)
- [OpenMed compliance posture](../compliance.md)
- [OpenMed evaluation harness and metric bundle](../eval-harness.md)
- [OpenMed no-raw-PHI logging policy](../security/no-raw-phi-logging.md)
