# EU Evidence Index

> **Index only. Not legal advice.** This page lists existing OpenMed documents
> that are relevant to GDPR and EU AI Act work. It does not classify a system,
> establish a lawful basis, complete a DPIA, or demonstrate conformity. Validate
> every document against your own processing, jurisdiction, and counsel.

Use this page as the entry point when you assemble EU-facing evidence for a
deployment that uses OpenMed. Each entry says what the linked page is for and
what it is not. New EU-relevant pages link themselves here when they land. For
the overall split between what OpenMed supplies and what the deployer owns, see
[Compliance Posture](../compliance.md).

## GDPR

### [Data Protection Impact Assessment Template](dpia-template.md)

A fill-in template, structured around GDPR Article 35, for a controller that
documents a DPIA for an OpenMed deployment. It shows where OpenMed
de-identification evidence can inform the assessment. It is not a completed
DPIA, and the controller's own risk assessment and approval stay required.

### [Data Processing Agreement Template](dpa-template.md)

A GDPR Article 28 agreement template for deployments where the facts create a
controller-processor relationship, with a local-first processing baseline and
key-custody annex. It is not a signed agreement, and installing OpenMed does not
make the OpenMed maintainers a processor.

### [GDPR DSAR Subject-Access Export](gdpr-dsar-export.md)

How `openmed.compliance.dsar` assembles a local Article 15 access package from
surrogate-vault entries, logs the export to a tamper-evident audit chain, and
previews an Article 17 erasure without deleting anything. It does not verify the
requester's identity or discover data held outside OpenMed.

## EU AI Act

### [EU AI Act and GDPR Model-Card Field Specification](model-card-eu-ai-act-fields.md)

The model-card block for EU AI Act and GDPR fields: intended purpose, known
limitations, accuracy and leakage metrics, human oversight, robustness, and
de-identification evidence, with rendering rules and field owners. It does not
decide whether a system is high-risk, perform a conformity assessment, or
replace the technical documentation a deployment requires.

## Supporting Controls

### [ISO 27701/27001 Evidence Pack](iso-27701-control-evidence.md)

How to generate a deterministic, offline evidence pack that maps OpenMed privacy
and release-safety capabilities to ISO/IEC 27701 and 27001 controls, which EU
deployers often reuse for GDPR accountability records. It is an implementation
crosswalk, not a certification audit, Statement of Applicability, or
organization-specific policy.
