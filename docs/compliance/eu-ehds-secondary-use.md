# EHDS Secondary Use: Mapping OpenMed Evidence

> **Template — requires legal review. Not legal advice.** This page maps
> OpenMed artifacts to concepts in the European Health Data Space Regulation.
> It does not establish that any processing complies with that Regulation, the
> GDPR or national law, and it does not replace the requirements of a health
> data access body.

The European Health Data Space Regulation, Regulation (EU) 2025/327 of
11 February 2025 (the EHDS Regulation), sets rules for the secondary use of
electronic health data in its Chapter IV (Articles 50 to 81). This page is for
EU health data
holders, and for the teams that support them, who prepare pseudonymised or
anonymised electronic health data with OpenMed. It lists which OpenMed
artifacts can serve as technical evidence for that work, and which duties
OpenMed does not cover.

Article references are to the EHDS Regulation as published in the Official
Journal of the European Union (OJ L, 2025/327, 5 March 2025) unless another act
is named. Check the current text on EUR-Lex before relying on any reference
here.

## What OpenMed is not

- OpenMed is not a health data access body. Member States designate those
  bodies (Article 55(1)), and their tasks include deciding on data access
  applications and issuing data permits (Article 57(1)(a)).
- OpenMed issues no data permits (Article 68) and answers no health data
  requests (Article 69).
- OpenMed is not a secure processing environment. Article 73 sets the
  requirements for those environments, and health data access bodies provide
  access under a data permit only through them (Article 73(1)).
- OpenMed does not decide whether data are anonymised or pseudonymised in law.
  Using OpenMed does not make a dataset anonymous and does not make anyone
  compliant with the EHDS Regulation.

OpenMed is a local software library. A data holder, a health data access body,
or a health data user working inside a secure processing environment may run
it, depending on national arrangements and the terms of a data permit. Which of
these roles applies to a given team, and what each role must do, is a legal
question for that team.

## Application dates

These dates are quoted from Article 105 of the EHDS Regulation. They were
checked against the Official Journal text on 2026-11-16.

| Provision | Applies from |
| --- | --- |
| The Regulation in general | 26 March 2027 ("This Regulation shall apply from 26 March 2027.") |
| Chapter IV, secondary use | 26 March 2029 ("Chapter IV shall apply from 26 March 2029"), with the exceptions below |
| Article 55(6), Article 70, Article 73(5), Article 75(1) and (12), Article 77(4) and Article 78(6) | 26 March 2027 |
| Article 51(1), points (b), (f), (g), (m) and (p) | 26 March 2031 |
| Article 75(5) | 26 March 2035 |

Article 105 also sets later, staged dates for the primary-use and EHR system
provisions. They are outside the scope of this page.

## Terms used on this page

| Term | Where it is defined |
| --- | --- |
| Secondary use | Article 2(2)(e): processing of electronic health data for the purposes in Chapter IV |
| Health data holder | Article 2(2)(t) |
| Health data user | Article 2(2)(u) |
| Data permit | Article 2(2)(v): an administrative decision issued to a health data user by a health data access body |
| Secure processing environment | Article 2(1)(c), which applies the definition in Regulation (EU) 2022/868 |
| Pseudonymisation | Article 2(1)(a), which applies the definition in Regulation (EU) 2016/679 (the GDPR) |

Article 2 does not define "anonymised" on its own. Its definition of
non-personal electronic health data (Article 2(2)(b)) includes data that have
been anonymised so that they no longer relate to an identified or identifiable
natural person. Whether that is true of a dataset is a legal and factual
assessment that also involves the GDPR, which treats pseudonymised data as
information on an identifiable person and leaves anonymous information outside
its scope (Recital 26 of Regulation (EU) 2016/679). OpenMed's de-identification
methods, listed in [PII Anonymization](../anonymization.md), produce
pseudonymised data unless a separate assessment supports an anonymisation
claim.

## Mapping table

Each row names an EHDS concept, the article that sets it, the OpenMed artifact
that can serve as evidence, and what remains with the data holder, the health
data access body or the health data user.

| EHDS concept | Article | OpenMed evidence | Not covered by OpenMed |
| --- | --- | --- | --- |
| Duty of health data holders to make data available on request, within three months, extendable by up to three months | Article 60(1) and (2) | None. OpenMed does not receive requests or track deadlines. | Responding to the health data access body and meeting the deadline. |
| Dataset description for the national catalogue, checked at least annually | Article 60(3), Article 77 | [Privacy-artifact lineage manifests](artifact-lineage.md) record digests, transformations, policy fingerprints and schema versions of derived artifacts. | Writing and maintaining the dataset description. |
| Documentation for a data quality and utility label | Article 60(4), Article 78 | None specific. | Producing the documentation the label requires. |
| Data minimisation: only data that are adequate, relevant and limited to what is necessary for the stated purpose | Article 66(1) | [Minimum-necessary structured field selection](../security/minimum-necessary.md) and [access-scope minimisation evidence](access-scope.md) show which fields and scopes a workflow selected and used. | Deciding what is necessary for the permit's purpose. |
| Data provided in anonymised format where the purpose can be achieved with such data | Article 66(2) | [Structured release risk and expert-review evidence](../reidentification-risk.md) measures k-anonymity, l-diversity, t-closeness and reference-population risk for structured data. [Structured privacy risk lab](../privacy/structured-risk-lab.md) and the [privacy budget ledger](../security/privacy-budget.md) cover differentially private aggregates. | Concluding that a dataset is anonymous. OpenMed reports measurements; it does not certify anonymity. |
| Pseudonymised format only where anonymised data cannot achieve the purpose, with the information needed to reverse the pseudonymisation available only to the health data access body or an entity acting as a trusted third party under national law | Article 66(3), Article 67(2)(e) | The `gdpr_pseudonymization` policy and keyed surrogates and date shifts ([PII Anonymization](../anonymization.md), [policy literals](../compliance.md)). Reversal material can be kept apart from the output with [encrypted surrogate mappings](../security/encrypted-surrogate-vault.md), [key management](../security/key-management.md) and [key custody metadata](../security/key-custody-metadata.md). [Surrogate-map audits](../security/surrogate-audit.md) check referential integrity. | Choosing who holds keys and mappings, and the justification for pseudonymised access in the application. |
| Health data access body tasks, including processing data and their pseudonymisation or anonymisation | Article 57(1)(b) | The same de-identification and risk artifacts, if a health data access body chooses to run OpenMed. | The body's own procedures and decisions. |
| Prohibition on health data users re-identifying or attempting to re-identify natural persons | Article 61(3) | None. OpenMed cannot prevent re-identification attempts by someone who holds the data. | Contractual, technical and legal controls on users. |
| Results of secondary use published within 18 months and containing only anonymous data | Article 61(4) | Release-time checks: the [privacy release-gate aggregator](privacy-release-gate.md), the [evidence freshness gate](evidence-freshness.md) and the aggregate privacy measurements above. | Deciding that published results are anonymous. |
| Responses to health data requests only in anonymised statistical format | Article 69(1) | Differentially private aggregates and their budget ledger ([structured privacy risk lab](../privacy/structured-risk-lab.md), [privacy budget ledger](../security/privacy-budget.md)). | The response itself, which the health data access body provides. |
| Secure processing environment: access limited to persons named in the permit, protection against unauthorised reading, copying, modification or removal, individual user identities, access and activity logs kept for at least one year, and monitoring | Article 73(1)(a) to (f) | OpenMed can run inside such an environment without network egress or telemetry ([no-telemetry guarantee](../security/no-telemetry.md)) and without raw identifiers in logs ([no-raw-PHI logging policy](../security/no-raw-phi-logging.md)). De-identification runs can be recorded in a [tamper-evident audit log](../security/tamper-evident-audit-log.md), with [signed audit envelopes](audit-envelopes.md) and a [retention planner](../security/audit-retention.md) for counts-only audit artifacts. | Operating the environment, access control, user identities, log retention and audits. |
| Only non-personal data, including anonymised statistical outputs, downloaded from the secure processing environment, after review by the health data access body | Article 73(2) | The release-time checks and aggregate privacy measurements above can support that review. | The review and the download decision. |
| Natural persons' right to opt out of secondary use | Article 71 | [Consent and data-use tags](consent-data-use-tags.md) can carry a caller-supplied `no-secondary-use` tag that denies the `secondary-use` action in an OpenMed pipeline. | Running the opt-out mechanism and deciding which records it covers. |

## Records to keep with an application or a release

The following OpenMed outputs contain codes, counts, offsets, digests and
policy fingerprints rather than raw text. They can be attached to internal
records when preparing data for a health data access body:

- the policy document and its digest from [privacy policy as data](../security/policy-as-data.md);
- de-identification audit reports and their reproducibility hashes, as listed
  in the [DPIA template](dpia-template.md) evidence table;
- structured release-risk evidence from
  [structured release risk and expert-review evidence](../reidentification-risk.md);
- lineage manifests from [privacy-artifact lineage manifests](artifact-lineage.md);
  and
- release-gate records from the
  [privacy release-gate aggregator](privacy-release-gate.md).

None of these outputs is a data permit, a dataset description under Article 77
or a data quality and utility label under Article 78.

## Related pages

- [EU evidence index](eu-evidence.md) lists the EU-facing pages in one place.
- [Compliance posture](../compliance.md) lists OpenMed's evidence and the
  framework split.
- [Data Protection Impact Assessment template](dpia-template.md) for the GDPR
  assessment that secondary-use processing may also need.
- [GDPR DSAR subject-access export](gdpr-dsar-export.md) for data-subject
  requests.
- [EU AI Act and GDPR model-card field specification](model-card-eu-ai-act-fields.md)
  for model documentation.

OpenMed has no EHDS-specific export pack or check today. This page maps only
existing artifacts. Dedicated EHDS evidence tooling is tracked for a later
release.
