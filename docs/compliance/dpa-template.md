# Data Processing Agreement Template

> **Template — requires legal review. Not legal advice.** This is not a signed
> agreement and does not establish that OpenMed's maintainers are a processor.
> Adapt it to the parties, processing, jurisdiction, and deployment before use.

OpenMed is a local-first software library. Installing or running it does not by
itself send personal data to the OpenMed project or create a
controller-processor relationship. Use this template only when the actual facts
create a processor relationship, such as between a deployer and an organization
operating OpenMed on the deployer's documented instructions.

Replace every `[bracketed placeholder]`. If the deployment adds hosted model
downloads, remote inference, storage, observability, support, or other services,
describe those services and their sub-processors instead of retaining the local
runtime defaults below.

## Agreement Details

This Data Processing Agreement (DPA) forms part of `[master agreement]` between:

- **Controller:** `[legal name, address, registration number]`
- **Processor:** `[legal name, address, registration number]`
- **Effective date and term:** `[date and duration]`
- **Governing law and venue:** `[jurisdiction]`
- **Privacy contacts:** `[controller contact]`; `[processor contact]`

The parties intend this DPA to address GDPR Article 28 and applicable national
data-protection law. Defined terms have the meanings assigned by the GDPR unless
the master agreement states otherwise.

## 1. Scope and Documented Instructions

1. The Processor shall process personal data only on the Controller's documented
   instructions, including instructions concerning transfers, unless Union or
   Member State law requires otherwise. The Processor shall inform the Controller
   of that legal requirement before processing unless prohibited by law.
2. The current instructions are the processing details in Annex A, the approved
   OpenMed configuration, and later written instructions agreed by authorized
   contacts.
3. The Processor shall immediately inform the Controller if, in its opinion, an
   instruction infringes applicable data-protection law and shall pause the
   affected processing where legally and contractually appropriate.
4. Processing for the Processor's own analytics, advertising, model training,
   product telemetry, or unrelated purposes is prohibited unless separately
   documented with a valid legal basis and role allocation.

## 2. Confidentiality and Personnel

The Processor shall limit access to authorized personnel who need it to perform
the services, are bound by confidentiality, receive appropriate privacy and
security training, and are subject to documented access review and revocation.

## 3. Security of Processing

The Processor shall maintain measures appropriate to the risk, including the
controls in Annex B and any measures required under GDPR Article 32. The parties
shall record who is responsible for each control rather than assuming that a
library default covers the surrounding deployment.

## 4. On-Device and Local Processing Baseline

For the agreed baseline deployment:

- processing occurs **on-device** or inside `[controller-approved local
  environment]` after approved model artifacts have been acquired;
- OpenMed has **no telemetry by default** and personal data shall not be added to
  analytics, product telemetry, error reporting, or support bundles;
- runtime network egress is `[blocked / restricted as described]`;
- raw input, de-identified output, audit evidence, caches, and temporary files
  remain within `[location and legal entity]` unless Annex A states otherwise;
- logs use counts, durations, labels, offsets, hashes, and safe identifiers, not
  raw clinical text, identifiers, or reversible mappings; and
- changes to these facts require documented Controller approval and a DPA/DPIA
  review before deployment.

Model acquisition, operating-system services, backups, remote administration,
hosted APIs, and monitoring are separate processing surfaces and must be listed
in Annex A even when OpenMed inference itself is local.

## 5. Sub-Processors

**Sub-processors: none for the OpenMed local runtime baseline.**

The Processor shall not engage a sub-processor for the covered processing
without the Controller's prior specific or general written authorization. For a
general authorization, the Processor shall give `[notice period]` advance notice
of additions or replacements so the Controller can object on reasonable
data-protection grounds.

The Processor shall impose obligations providing at least the same protection as
this DPA and remains responsible for the sub-processor's performance. Any actual
hosting, storage, model registry, support, monitoring, or remote administration
provider must be entered in Annex D. Do not leave "none" in place if such a
provider can access personal data.

## 6. Assistance to the Controller

Taking into account the nature of processing and available information, the
Processor shall reasonably assist the Controller with:

- data-subject requests and verification of access, rectification, erasure,
  restriction, objection, and portability workflows;
- security, breach assessment, notification, and communication obligations;
- DPIAs, risk reassessment, and prior consultation with a supervisory authority;
- records demonstrating compliance, including configuration, aggregate metrics,
  audit hashes, and approved residual-risk evidence; and
- inquiries from supervisory authorities.

The assistance process, contacts, response times, and fees are: `[terms]`.

## 7. Personal Data Breaches

The Processor shall notify the Controller without undue delay and within
`[contractual target]` after becoming aware of a personal data breach. The notice
shall include available information about the nature and scope, affected data and
data subjects, likely consequences, containment, remediation, and a contact for
follow-up. Information may be supplied in phases without delaying the initial
notice.

Notification does not transfer the Controller's legal assessment or
supervisory-authority notification duties to the Processor.

## 8. Audit and Demonstration of Compliance

The Processor shall make information reasonably necessary to demonstrate
compliance available to the Controller and allow proportionate audits or
inspections by the Controller or its mandated auditor. The parties shall agree
scope, confidentiality, security, scheduling, and cost controls without preventing
an audit required by law or a competent authority.

Privacy-safe evidence should use hashes, offsets, counts, aggregate metrics,
configuration, and provenance. It shall not include raw patient content or
re-identification mappings unless strictly necessary, authorized, and protected.

## 9. International Transfers and Processing Locations

- Approved processing and storage locations: `[locations]`
- Approved remote-access locations: `[locations or none]`
- Transfer mechanism and supplementary measures: `[not applicable / adequacy /
  SCCs / other]`
- Transfer impact assessment reference: `[reference]`

The Processor shall not change a processing location or enable a restricted
transfer without documented authorization and required safeguards.

## 10. Return, Deletion, and End of Services

At the Controller's choice, the Processor shall return or securely delete the
personal data and copies at the end of services, unless applicable law requires
retention. The Processor shall identify legally retained data, isolate it from
ordinary use, and delete it when the legal period expires.

Deletion must cover or separately schedule inputs, outputs, local caches,
temporary files, logs, mappings, keys, audit artifacts, replicas, and backups.
Deletion method, timing, evidence, and exceptions: `[terms]`.

## 11. Reversible Pseudonymization and Key Custody

This section applies when the deployment uses the exact policy literal
`gdpr_pseudonymization`, which sets `keep_mapping=True` and
`reversible_id=True`. Pseudonymized data remains personal data.

1. The Controller retains `[exclusive / allocated]` custody and control of the
   re-identification key and mapping.
2. The key, mapping, and pseudonymized dataset shall be separated using
   `[technical and organizational separation]`.
3. The Processor shall not re-identify, link, disclose, export, or use the
   mapping except on a documented instruction from `[authorized role]` for
   `[approved purpose]`.
4. Key creation, storage, access, rotation, recovery, revocation, destruction,
   and emergency use shall follow Annex C.
5. Key and mapping access shall be least-privilege, authenticated, logged, and
   periodically reviewed without recording cleartext identifiers in operational
   logs.
6. Suspected key or mapping compromise shall be handled as a security incident
   and assessed as a potential personal data breach.

If the policy literal `hipaa_safe_harbor` is used instead, the parties shall
document that its masking posture does not itself establish anonymization under
the GDPR and shall still assess residual identifiability.

## 12. Liability, Order of Precedence, and Changes

Liability, indemnity, governing law, and dispute terms are `[master agreement
references or negotiated clauses]`. If this DPA conflicts with the master
agreement on personal-data protection, `[order of precedence]` applies. Material
changes to purpose, data, model, policy, hosting, recipients, locations,
sub-processors, retention, or key custody require written change control and a
privacy-risk review.

## Annex A — Processing Details

| Field | Agreed detail |
|---|---|
| Subject matter | [OpenMed-assisted de-identification or other service] |
| Duration | [Term plus deletion period] |
| Nature and purpose | [Operations and explicit purposes] |
| Data subjects | [Categories] |
| Personal-data types | [Categories, including any health or genetic data] |
| Processing operations | [Collect, load, detect, replace/mask, review, store, delete] |
| OpenMed policy literal | `[hipaa_safe_harbor or gdpr_pseudonymization]` |
| Model and software | [Ids, versions, revisions, hashes] |
| Inputs and sources | [Systems and formats] |
| Outputs and recipients | [Systems, roles, legal entities] |
| Locations and network boundaries | [Locations and data flow] |
| Retention | [Per artifact] |
| Controller instructions | [References] |

## Annex B — Technical and Organizational Measures

| Control area | Measure, owner, and evidence |
|---|---|
| Local/offline execution and egress | [Measure] |
| Encryption in transit and at rest | [Measure] |
| Identity, access, and privilege review | [Measure] |
| No-raw-PHI logs and telemetry controls | [Measure] |
| Model, manifest, dependency, and configuration integrity | [Measure] |
| Leakage, recall, residual-risk, and robustness validation | [Measure] |
| Vulnerability and patch management | [Measure] |
| Backup, restore, retention, and deletion | [Measure] |
| Incident detection and response | [Measure] |
| Business continuity and availability | [Measure] |
| Personnel confidentiality and training | [Measure] |
| Control testing and review cadence | [Measure] |

## Annex C — Pseudonymization Key-Custody Record

| Field | Agreed detail |
|---|---|
| Key owner and system | [Values] |
| Mapping owner and system | [Values] |
| Legal and physical locations | [Values] |
| Authorized roles | [Values] |
| Separation of duties | [Values] |
| Rotation and expiry | [Values] |
| Backup and recovery | [Values] |
| Re-identification approval | [Values] |
| Access-log review | [Values] |
| Revocation and destruction | [Values] |
| Compromise response | [Values] |

## Annex D — Authorized Sub-Processors

| Sub-processor | Service | Data and access | Location/transfer | Authorization date |
|---|---|---|---|---|
| None for the OpenMed local runtime baseline | Not applicable | None | Not applicable | Not applicable |

## Signatures

| Party | Name, title, signature, date |
|---|---|
| Controller | [Values] |
| Processor | [Values] |

## Primary References

- [GDPR, including Articles 28, 32, 35, and 36](https://eur-lex.europa.eu/eli/reg/2016/679/oj)
- [OpenMed compliance posture](../compliance.md)
- [OpenMed no-telemetry policy](../security/no-telemetry.md)
- [OpenMed no-raw-PHI logging policy](../security/no-raw-phi-logging.md)
- [OpenMed DPIA template](dpia-template.md)
