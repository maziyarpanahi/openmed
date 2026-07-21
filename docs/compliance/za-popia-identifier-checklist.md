# South Africa POPIA identifier checklist

> **Decision support only — not legal advice.** This checklist describes the
> technical defaults in OpenMed's `za_popia` de-identification profile. It does
> not determine whether processing is lawful, satisfy an authorisation, certify
> POPIA compliance, or replace advice from South African counsel or an
> information officer.

The profile is an irreversible anonymisation posture: it requires the safety
sweep, does not retain a replacement mapping, and does not keep any canonical
identifier or special-personal-information class by default. The class mapping
starts from the definition of personal information in section 1 of the
[Protection of Personal Information Act 4 of 2013 (POPIA)][popia].

Sections 26 and 27 establish the prohibition and general authorisations for
special personal information. Section 32 provides health- and sex-life-specific
authorisations and requires confidentiality for the responsible parties it
covers. Section 72 governs transfers to recipients outside South Africa. The
[2026 health-information Regulations][health-regulations] further address
section 32(6), confidentiality safeguards, and section 72 transfers for the
responsible parties within their scope.

## Per-class action reference

Every label below is a member of `CANONICAL_LABELS`. `replace` means an
irreversible synthetic replacement under this profile; `mask` means a typed
placeholder. Both are non-`keep` actions. Section 72 applies to every row when
personal information is transferred to a third party in another country.

<!-- popia-identifier-table:start -->
| POPIA class key | Statutory reference and identifier class | Canonical labels | Profile action |
|---|---|---|---|
| `NAME` | Section 1(h): a name when linked to other personal information or revealing information by itself | `PERSON`, `FIRST_NAME`, `LAST_NAME`, `MIDDLE_NAME`, `PREFIX` | `replace` |
| `CONTACT_DETAILS` | Section 1(c): e-mail address and telephone number | `EMAIL`, `PHONE` | `replace` |
| `LOCATION_INFORMATION` | Section 1(c): physical address and location information | `LOCATION`, `STREET_ADDRESS`, `BUILDING_NUMBER`, `ZIPCODE`, `GPS_COORDINATES`, `ORDINAL_DIRECTION` | `replace` |
| `IDENTIFYING_NUMBER` | Section 1(c): identifying number, symbol, or other particular assignment | `ID_NUM`, `SSN`, `ACCOUNT_NUMBER`, `PIN`, `CREDIT_CARD`, `CVV`, `IBAN`, `BIC`, `BITCOIN_ADDRESS`, `ETHEREUM_ADDRESS`, `LITECOIN_ADDRESS`, `MASKED_NUMBER`, `VIN`, `VEHICLE_REGISTRATION` | `mask` |
| `BIOMETRIC_INFORMATION` | Sections 1(d), 26, and 33: biometric information used for personal identification | `EYE_COLOR`, `HEIGHT` | `mask` |
| `ONLINE_IDENTIFIER` | Section 1(c): online identifiers and related account or device assignments | `USERNAME`, `URL`, `PASSWORD`, `API_KEY`, `IP_ADDRESS`, `MAC_ADDRESS`, `USER_AGENT`, `IMEI` | `replace` for `USERNAME` and `URL`; otherwise `mask` |
| `DEMOGRAPHIC_AND_HISTORY_ATTRIBUTES` | Sections 1(a)-(b) and 26: demographic attributes and medical, financial, criminal, or employment history; section 26 expressly includes race or ethnic origin | `DATE`, `DATE_OF_BIRTH`, `TIME`, `AGE`, `CREDIT_CARD_ISSUER`, `AMOUNT`, `CURRENCY`, `GENDER`, `ORGANIZATION`, `JOB_TITLE`, `JOB_DEPARTMENT`, `OCCUPATION` | `replace` for `ORGANIZATION`; otherwise `mask` |
| `HEALTH_INFORMATION` | Sections 1(a)-(b), 26-27, and 32: physical or mental health, medical history, treatment, care, and related clinical information | `MICROORGANISM`, `ANTIBIOTIC`, `SUSCEPTIBILITY`, `CONDITION`, `MEDICATION`, `LAB_TEST`, `PROCEDURE`, `BODY_SITE`, `ANESTHESIA_TYPE`, `ANESTHETIC_AGENT`, `AIRWAY_MANAGEMENT`, `ASA_CLASS`, `DIET_TYPE`, `NUTRITION_TARGET`, `FEEDING_ROUTE`, `NUTRITIONAL_STATUS`, `VACCINE_NAME`, `DOSE_NUMBER`, `ADMINISTRATION_ROUTE`, `VACCINE_LOT`, `VACCINE_SERIES`, `GENE_SYMBOL`, `VARIANT_DESCRIPTOR`, `PROTEIN_CHANGE`, `ZYGOSITY`, `CLINICAL_SIGNIFICANCE`, `GLYCEMIC_MEASURE`, `THYROID_MEASURE`, `HORMONE_LEVEL`, `INSULIN_REGIMEN`, `ENDOSCOPIC_FINDING`, `GI_SYMPTOM`, `GI_SCORE`, `POLYP_DESCRIPTOR`, `CKD_STAGE`, `DIALYSIS_MODALITY`, `RENAL_FUNCTION_MEASURE`, `URINE_FINDING`, `SPIROMETRY_MEASURE`, `OXYGEN_SUPPORT`, `RESPIRATORY_FINDING`, `DYSPNEA_GRADE`, `GROWTH_PARAMETER`, `GROWTH_PERCENTILE`, `DEVELOPMENTAL_MILESTONE` | `mask` |
| `OTHER_SPECIAL_INFORMATION` | Sections 1(e)-(g) and 26-27: opinions, private correspondence, and special categories without a dedicated canonical label, including beliefs, union membership, political persuasion, sex life, and criminal behaviour | `OTHER` | `mask` |
<!-- popia-identifier-table:end -->

## Implementation notes

- The mapping is implemented in `LABEL_TO_POPIA`; every canonical label has
  exactly one primary POPIA checklist class.
- OpenMed does not currently expose dedicated canonical labels for every
  section 26 category or for fingerprint, DNA, retinal, or voice templates.
  Detector/plugin outputs without a dedicated canonical label fall back to
  `OTHER`, which this profile masks. Sites should validate their detectors
  against their actual data and add approved custom recognizers when needed.
- `za_popia` is a technical de-identification control. It does not evaluate
  consent, public-interest grounds, section 27 or 32 authorisations, duties of
  confidentiality, cross-border adequacy, contracts, or transfer consent.
- The fixtures and tests for this profile are synthetic and must not be
  replaced with real patient data.

[popia]: https://www.justice.gov.za/legislation/acts/2013-004.pdf
[health-regulations]: https://www.gov.za/sites/default/files/gcis_document/202603/54268gon7198.pdf
