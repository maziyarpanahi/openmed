# Africa Data-Residency Deployment Guide

This guide is decision-support material, not legal advice, a regulator filing,
or a certification of compliance. Data-residency outcomes depend on the entire
system: intake, backups, logs, model delivery, support tooling, exports, and
operator access—not only the OpenMed process. Confirm the current rules and the
deployment design with local counsel and the responsible privacy authority.

OpenMed de-identification runs in the application process. The attestation API
records PHI-free evidence about that run: policy name and posture, local model
artifact paths and content checksums, captured offline flags, host-local
execution wording, counts, timestamps, and hashes. It never copies source text,
de-identified text, detected identifiers, surrogates, or reversible mappings.

## Choose a deployment pattern

### Air-gapped host

Use a host with no physical or logical route to an external network. Transfer
the OpenMed wheel, its locked dependencies, model artifacts, policy files, and
checksums through an approved removable-media process. Store the original
checksums outside the air gap and compare them after transfer. Keep input,
output, audit, and attestation artifacts on storage governed by the same local
retention and access controls.

This is the strongest no-egress pattern, but the OpenMed assertion is only one
piece of evidence. Network diagrams, firewall configuration, removable-media
logs, backup destinations, and access records remain necessary.

### Network-isolated clinical subnet

Place processing hosts and local model storage in a subnet with deny-by-default
egress. Permit only explicitly reviewed internal routes, such as a local package
mirror or records system, and document whether those routes cross a national
boundary. Preload models before the production window, then remove temporary
download access. Test DNS, proxy, IPv4, IPv6, and direct-IP paths at the network
control layer.

### Managed on-device application

Bundle or provision model artifacts to a managed workstation or mobile device,
enable OpenMed local-only mode, and use device-management controls to block
unapproved data sharing, cloud backup, diagnostics, and support collection.
Record the model checksum and app release with each attestation. Treat OS crash
reports, keyboard services, clipboard synchronization, and third-party SDKs as
separate potential transfer paths.

## Verify no-network operation

1. Pre-download every model and dependency into an approved local path. Record
   each model artifact checksum before the production run.
2. Set `OPENMED_OFFLINE=1`. OpenMed also enables `HF_HUB_OFFLINE`,
   `TRANSFORMERS_OFFLINE`, and `HF_DATASETS_OFFLINE` for cache-only dependency
   behavior.
3. Construct `OpenMedConfig(local_only=True)` for an explicit application-level
   assertion. OpenMed requests its outbound socket guard during guarded model
   operations when local-only mode is active.
4. Enforce no egress outside the Python process as well: firewall or sandbox the
   host, disable unapproved proxies and DNS, and monitor attempted connections.
5. Run a synthetic canary with the network controls enabled. Confirm the model
   loads from the approved local path and a forced remote model reference fails
   closed.
6. Generate the attestation immediately after the audited run. Require
   `execution.offline_assertion.asserted` to be `true`; if it is `false`, keep
   the artifact as host-local evidence but do not describe it as verified
   no-egress evidence.
7. Validate the attestation against OpenMed's bundled `attestation` JSON Schema
   and retain both the JSON and Markdown forms with the firewall, image, and
   deployment evidence.

The offline assertion proves only that OpenMed local-only mode and the bundled
dependency flags were captured together. It does not prove firewall state,
physical location, absence of other processes, or the behavior of the wider
application.

## Generate an attestation through Python

The feature extends the audit-report API; it does not add a CLI subcommand.

```python
from datetime import datetime, timezone
from pathlib import Path

from openmed import OpenMedConfig, deidentify

config = OpenMedConfig(local_only=True)
audit = deidentify(
    "Synthetic patient Example Person, record TEST-0001.",
    policy="strict_no_leak",
    config=config,
    audit=True,
)
attestation = audit.attest(
    "rwanda-law-058-2021",
    model_artifacts={"pii-model": "/opt/openmed/models/pii"},
    generated_at=datetime.now(timezone.utc),
)

Path("attestation.json").write_text(attestation.to_json(indent=2), encoding="utf-8")
Path("attestation.md").write_text(attestation.to_markdown(), encoding="utf-8")
assert attestation.repro_hash_matches()
assert attestation.integrity_hash_matches()
```

The jurisdiction profile requires the audit's `strict_no_leak` policy. The
generator loads that bundled policy and records its exact canonical name,
posture, schema version, and default action. A missing policy, an invalid policy,
or a policy that does not match the jurisdiction profile fails with a clear
error. Model paths must exist; files and directory contents are hashed locally.

The `repro_hash` is stable for the same audit, template, offline evidence, and
model contents, even when `generated_at` changes. The `integrity_hash` covers the
full JSON payload, including the timestamp. These are SHA-256 integrity hashes,
not digital signatures or regulator-issued seals.

## Rwanda

- Attestation profile: `rwanda-law-058-2021`
- Policy profile: `strict_no_leak`
- Template field: `residency_statement`

[Rwanda Law No. 058/2021](https://dpo.gov.rw/fileadmin/DPO/Law_relating_to_the_protection_of_personal_data_and_privacy.pdf)
addresses transfers outside Rwanda in Articles 48-49. Article 50 provides for
storage in Rwanda and a registration-certificate route for storage outside the
country. A host-local, offline deployment can reduce transfer and offshore
storage paths, but the attestation does not establish where backups, upstream
systems, or downstream exports reside.

Decision questions:

- Are the processing host, input store, output store, logs, and backups all in
  Rwanda?
- Does any operator, support tool, model fetch, or synchronization service make
  the data available outside Rwanda?
- If any offshore storage remains, has the organization obtained and retained
  the required authorization evidence?

## Ethiopia

- Attestation profile: `ethiopia-proclamation-1321-2024`
- Policy profile: `strict_no_leak`
- Template field: `residency_statement`

[Ethiopia's Personal Data Protection Proclamation No. 1321/2024](https://pdp.eca.et/files/pdfs/pdp-proclamation/pdp_personal_data_protection_proclamation_No_1321_2024.pdf)
sets cross-border transfer conditions in Articles 18-21. Article 22 requires
personal data collected or obtained locally to be stored on a server or in a
data center located in Ethiopia, permits the Authority to designate critical
personal-data categories for local-only processing, and requires prior Authority
approval for cross-border transfer of sensitive personal data; the definition
of sensitive data includes physical or mental health information. The
attestation can support a finding that a particular de-identification run stayed
host-local, but it is not Authority approval.

Decision questions:

- Does the workflow handle health, biometric, genetic, or other sensitive
  personal data before de-identification?
- Has the Authority designated this processing category for an Ethiopia-located
  server or data center?
- Is any cross-border access or transfer proposed, and if so, is prior approval
  required and documented?

## Kenya

- Attestation profile: `kenya-dpa-2019`
- Policy profile: `strict_no_leak`
- Template field: `residency_statement`

[Kenya's Data Protection Act No. 24 of 2019](https://new.kenyalaw.org/akn/ke/act/2019/24/eng@2022-12-31)
sets transfer conditions and safeguard duties in sections 48-49. Section 49
addresses consent and appropriate safeguards for sensitive personal data, while
section 50 allows prescribed processing categories to be limited to a server or
data center in Kenya. Regulation 26 of the
[Data Protection (General) Regulations, 2021](https://new.kenyalaw.org/akn/ke/act/ln/2021/263/eng@2022-01-14)
applies that local-processing or serving-copy rule to specified state interests,
including primary or secondary health care. Host-local operation is useful
evidence, but the organization must still determine whether a transfer occurs
and which category or condition applies.

Decision questions:

- Can the organization prove appropriate safeguards for every transfer path?
- Is valid data-subject consent and confirmation of safeguards required for the
  sensitive-data flow?
- Has a section 50 prescription or sector-specific rule made local processing
  mandatory for this workload?

## Egypt

- Attestation profile: `egypt-pdpl-151-2020`
- Policy profile: `strict_no_leak`
- Template field: `residency_statement`

Egypt enacted [Personal Data Protection Law No. 151 of 2020][egypt-pdpl].
An [English reference translation](https://eg.andersen.com/wp-content/uploads/2025/06/Law-No.-151-OF-2020.pdf)
describes licensing or authorization conditions for cross-border transfer,
storage, sharing, or processing in Articles 14-16. A local deployment can avoid
those paths for the recorded run, but the attestation is not a licence, permit,
adequacy decision, or substitute for checking current implementing regulations.

[egypt-pdpl]: https://sis.gov.eg/ar/%D8%A7%D9%84%D8%B1%D8%A6%D8%A7%D8%B3%D8%A9/%D8%B4%D8%A6%D9%88%D9%86-%D8%AF%D8%A7%D8%AE%D9%84%D9%8A%D8%A9/%D8%A7%D9%84%D9%82%D8%B1%D8%A7%D8%B1%D8%A7%D8%AA-%D8%A7%D9%84%D8%B1%D8%A6%D8%A7%D8%B3%D9%8A%D8%A9/%D8%A7%D9%84%D8%B1%D8%A6%D9%8A%D8%B3-%D8%A7%D9%84%D8%B3%D9%8A%D8%B3%D9%89%D9%8A-%D9%8A-%D8%B5%D8%AF-%D9%82-%D8%B9%D9%84%D9%89-%D9%82%D8%A7%D9%86%D9%88%D9%86-%D8%AD%D9%85%D8%A7%D9%8A%D8%A9-%D8%A7%D9%84%D8%A8%D9%8A%D8%A7%D9%86%D8%A7%D8%AA-%D8%A7%D9%84%D8%B4%D8%AE%D8%B5%D9%8A%D8%A9/

Decision questions:

- Do model delivery, backups, support, analytics, or remote administration make
  personal data available outside Egypt?
- Does the workflow require a processing, sensitive-data, or cross-border
  licence or authorization even if the core inference is local?
- Have current executive regulations and Personal Data Protection Center
  procedures been checked for this deployment?

## Review and retention checklist

- Validate the JSON against the bundled attestation schema.
- Verify `repro_hash_matches()` and `integrity_hash_matches()` after copying or
  archiving the artifact.
- Compare model checksums with the approved deployment manifest.
- Confirm the exact policy name and posture match the approved privacy design.
- Attach firewall, network, device-management, physical-location, backup, and
  access-control evidence; the attestation does not replace them.
- Keep raw documents and reversible mappings out of the attestation package.
- Record counsel or privacy-officer review separately using the
  [attestation review template](templates/africa-data-residency-attestation.md).
- Reassess after model, policy, template, network, storage, or legal changes.
