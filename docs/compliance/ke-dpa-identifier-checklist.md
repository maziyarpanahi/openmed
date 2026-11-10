# Kenya DPA Identifier Checklist

This checklist maps the sensitive-personal-data classes in Kenya's Data
Protection Act No. 24 of 2019 to the canonical labels and actions in OpenMed's
`ke_dpa` policy profile. The profile is deliberately conservative: it masks
direct identifiers, county-level and finer location data, sensitive attributes,
and health-related concepts; retains no reversible mapping; uses high-recall
arbitration; and always runs the deterministic safety sweep.

This is decision-support guidance, not legal advice, compliance certification,
or a substitute for a controller's lawful-basis, consent, transfer, security,
and professional-confidentiality review. De-identification lowers disclosure
risk but cannot by itself establish compliance or guarantee that a person is no
longer identifiable.

## Statutory posture

The source of record is the
[Data Protection Act, 2019](https://new.kenyalaw.org/akn/ke/act/2019/24/eng@2022-12-31)
published by Kenya Law.

- [Section 2](https://new.kenyalaw.org/akn/ke/act/2019/24/eng@2022-12-31#sec_2)
  defines personal identifiers, health data, biometric data, sensitive personal
  data, anonymisation, and pseudonymisation. Its sensitive classes drive the
  table below.
- [Section 44](https://new.kenyalaw.org/akn/ke/act/2019/24/eng@2022-12-31#sec_44)
  restricts processing of sensitive personal data, while
  [section 45](https://new.kenyalaw.org/akn/ke/act/2019/24/eng@2022-12-31#sec_45)
  lists permitted grounds. Selecting `ke_dpa` does not determine whether a
  ground applies.
- [Section 46](https://new.kenyalaw.org/akn/ke/act/2019/24/eng@2022-12-31#sec_46)
  limits health-data processing to healthcare-provider responsibility,
  professional secrecy, public-health necessity, or another legal duty of
  confidentiality. The profile therefore masks clinical concepts as well as
  identifiers.
- [Section 48](https://new.kenyalaw.org/akn/ke/act/2019/24/eng@2022-12-31#sec_48)
  sets conditions for transfers outside Kenya, and
  [section 49](https://new.kenyalaw.org/akn/ke/act/2019/24/eng@2022-12-31#sec_49)
  requires consent and appropriate safeguards before sensitive personal data is
  processed outside Kenya. Section 49 also permits the Data Commissioner to
  prohibit, suspend, or condition a transfer.
- [Section 50](https://new.kenyalaw.org/akn/ke/act/2019/24/eng@2022-12-31#sec_50)
  permits specified processing to be required on a server or in a data centre
  located in Kenya.

## Section 2 sensitive-data action map

`OTHER` is the existing conservative catch-all for sensitive attributes that do
not yet have a dedicated canonical detector label. Under `ke_dpa`, it is masked,
so those attributes cannot fall through to `keep`. Biometric template or
reference identifiers, Huduma Namba, national identity numbers, and
health-linked member identifiers use `ID_NUM`; raw biometric media requires a
modality-specific workflow outside this text profile.

| Section 2 sensitive class | Canonical labels | `ke_dpa` action |
|---|---|---|
| Race | `OTHER` | `mask` |
| Health status | `CONDITION`, `MEDICATION`, `LAB_TEST`, `PROCEDURE`, `BODY_SITE` | `mask` |
| Ethnic social origin | `OTHER` | `mask` |
| Conscience | `OTHER` | `mask` |
| Belief | `OTHER` | `mask` |
| Genetic data | `GENE_SYMBOL`, `VARIANT_DESCRIPTOR`, `PROTEIN_CHANGE`, `ZYGOSITY`, `CLINICAL_SIGNIFICANCE` | `mask` |
| Biometric data | `ID_NUM`, `EYE_COLOR`, `HEIGHT` | `mask` |
| Property details | `STREET_ADDRESS`, `LOCATION`, `ACCOUNT_NUMBER` | `mask` |
| Marital status | `OTHER` | `mask` |
| Family details | `PERSON`, `FIRST_NAME`, `LAST_NAME` | `mask` |
| Sex | `GENDER` | `mask` |
| Sexual orientation | `OTHER` | `mask` |

## Clinical identifier checklist

Before disclosing or transferring a clinical text, confirm that the local audit
evidence shows each applicable item was handled:

- Patient, relative, clinician, and caregiver names are mapped to `PERSON` or a
  name component and masked.
- Huduma Namba, national ID, passport, health-plan, facility, and medical-record
  identifiers are mapped to `ID_NUM` and masked.
- Kenyan phone numbers, including `+254` forms, are mapped to `PHONE` and
  masked.
- Street, building, postal, GPS, and county-level location details are mapped to
  their location labels and masked. County values remain quasi-identifying in
  sparse clinical populations and are not retained by this profile.
- Health status, diagnoses, procedures, medicines, tests, body sites, and other
  clinical concepts are masked to keep the text outside the section 46
  confidentiality perimeter only after the caller validates the output.
- Ethnicity, race, beliefs, marital or family status, sex, sexual orientation,
  genetic attributes, and biometric references are mapped through the table
  above and never use a `keep` action.
- The audit report identifies `ke_dpa`, records that the safety sweep was
  enabled, and reports zero direct-identifier leakage for the validated fixture
  set. Production data needs its own leakage and re-identification assessment.

## Local-first transfer and residency rationale

Run model inference, deterministic sweeping, redaction, and audit generation on
the device or within the controller's Kenyan environment before any proposed
external transfer. This reduces the chance that raw clinical text leaves the
section 46 confidentiality perimeter and supports assessment of the safeguards
and consent required by sections 48 and 49. It also preserves the option to
keep covered processing in Kenya if a section 50 requirement applies.

Local-first processing is a risk-control posture, not a legal conclusion. The
controller still decides whether processing is permitted under sections 44 and
45, whether the section 46 actor and confidentiality conditions are met, and
whether any transfer or residency requirement applies.
