# Nigeria NDPA 2023 identifier checklist

> **Decision support only - not legal advice.** This checklist describes the
> technical defaults in OpenMed's `ng_ndpa` de-identification profile. It does
> not determine whether processing is lawful, establish a section 30 ground,
> certify compliance with the Nigeria Data Protection Act 2023 (NDPA), or
> replace advice from Nigerian counsel or a data protection officer.

The profile is a strict, irreversible de-identification posture: it requires
the safety sweep, does not retain a replacement mapping, and assigns a
non-`keep` action to every canonical label. The NDPA's interpretation provision
in section 65 defines sensitive personal data to include genetic and biometric
data used for unique identification, race or ethnic origin, religious or
similar beliefs, health status, sex life, political opinions or affiliations,
trade union memberships, and further information prescribed by the Nigeria
Data Protection Commission under section 30(2). Section 30(1) restricts
processing of sensitive personal data unless one of its stated grounds applies.
The full statutory text is available from the [Nigeria Data Protection
Commission][ndpa].

## Section 30 sensitive-data category reference

Every label below is a member of `CANONICAL_LABELS`. `replace` means an
irreversible synthetic replacement under this profile; `mask` means a typed
placeholder. Both are non-`keep` actions. These mappings are conservative
technical anchors, not findings that every occurrence of a label is sensitive
personal data under the NDPA.

<!-- ndpa-sensitive-data-table:start -->
| NDPA class key | Statutory category | Canonical label anchors | Profile action |
|---|---|---|---|
| `GENETIC_AND_BIOMETRIC_DATA` | Sections 65 and 30: genetic and biometric data used to uniquely identify a natural person | `ID_NUM`, `EYE_COLOR`, `HEIGHT`, `GENE_SYMBOL`, `VARIANT_DESCRIPTOR`, `PROTEIN_CHANGE`, `ZYGOSITY`, `CLINICAL_SIGNIFICANCE` | `mask` |
| `RACE_OR_ETHNIC_ORIGIN` | Sections 65 and 30: race or ethnic origin | `OTHER` | `mask` |
| `RELIGIOUS_OR_SIMILAR_BELIEFS` | Sections 65 and 30: religious or similar beliefs, including conscience or philosophy | `OTHER` | `mask` |
| `HEALTH_STATUS` | Sections 65 and 30: health status, including health-record identifiers and clinical content under this profile | `ID_NUM`, `MICROORGANISM`, `ANTIBIOTIC`, `SUSCEPTIBILITY`, `CONDITION`, `MEDICATION`, `LAB_TEST`, `PROCEDURE`, `BODY_SITE`, `ANESTHESIA_TYPE`, `ANESTHETIC_AGENT`, `AIRWAY_MANAGEMENT`, `ASA_CLASS`, `DIET_TYPE`, `NUTRITION_TARGET`, `FEEDING_ROUTE`, `NUTRITIONAL_STATUS`, `VACCINE_NAME`, `DOSE_NUMBER`, `ADMINISTRATION_ROUTE`, `VACCINE_LOT`, `VACCINE_SERIES`, `GENE_SYMBOL`, `VARIANT_DESCRIPTOR`, `PROTEIN_CHANGE`, `ZYGOSITY`, `CLINICAL_SIGNIFICANCE`, `GLYCEMIC_MEASURE`, `THYROID_MEASURE`, `HORMONE_LEVEL`, `INSULIN_REGIMEN`, `ENDOSCOPIC_FINDING`, `GI_SYMPTOM`, `GI_SCORE`, `POLYP_DESCRIPTOR`, `CKD_STAGE`, `DIALYSIS_MODALITY`, `RENAL_FUNCTION_MEASURE`, `URINE_FINDING`, `SPIROMETRY_MEASURE`, `OXYGEN_SUPPORT`, `RESPIRATORY_FINDING`, `DYSPNEA_GRADE`, `GROWTH_PARAMETER`, `GROWTH_PERCENTILE`, `DEVELOPMENTAL_MILESTONE` | `mask` |
| `SEX_LIFE` | Sections 65 and 30: sex life | `GENDER`, `OTHER` | `mask` |
| `POLITICAL_OPINIONS_OR_AFFILIATIONS` | Sections 65 and 30: political opinions or affiliations | `ORGANIZATION`, `OTHER` | `mask` |
| `TRADE_UNION_MEMBERSHIPS` | Sections 65 and 30: trade union memberships | `ORGANIZATION`, `JOB_DEPARTMENT`, `OTHER` | `mask` |
| `OTHER_COMMISSION_PRESCRIBED_DATA` | Sections 30(2) and 65: other information prescribed as sensitive personal data by the Commission | `OTHER` | `mask` |
<!-- ndpa-sensitive-data-table:end -->

`OTHER` is the conservative fallback for sensitive attributes that do not yet
have a dedicated canonical label. The profile action protects a correctly
normalised detection; it does not itself discover ethnicity, belief, sex-life,
political, or trade-union terms. Sites should validate detector coverage on
synthetic or otherwise approved local data and add approved custom recognizers
where necessary. `EYE_COLOR` and `HEIGHT` are treated only as biometric hints;
the table does not claim that they are biometric data in every context.

## Linked identifier controls

The strict profile also removes identifiers that can link sensitive clinical
content to a person. NIN-style and BVN-style values are technical examples in
this table; their inclusion is a conservative de-identification default, not a
claim that the NDPA categorically defines every NIN or BVN as sensitive personal
data.

| Technical identifier class | Canonical labels | Profile action |
|---|---|---|
| Names and user handles | `PERSON`, `FIRST_NAME`, `LAST_NAME`, `MIDDLE_NAME`, `PREFIX`, `USERNAME` | `replace` |
| Contact details | `EMAIL`, `PHONE`, `URL` | `replace` |
| Lagos, Abuja, and other location details | `LOCATION`, `STREET_ADDRESS`, `BUILDING_NUMBER`, `ZIPCODE`, `GPS_COORDINATES`, `ORDINAL_DIRECTION` | `mask` |
| Dates and demographic quasi-identifiers | `DATE`, `DATE_OF_BIRTH`, `TIME`, `AGE`, `GENDER`, `EYE_COLOR`, `HEIGHT` | `mask` |
| NIN-style national IDs, health-record IDs, and biometric-template IDs | `ID_NUM`, `SSN` | `mask` |
| BVN-style and other account or financial identifiers | `ACCOUNT_NUMBER`, `CREDIT_CARD`, `CVV`, `IBAN`, `BIC`, `BITCOIN_ADDRESS`, `ETHEREUM_ADDRESS`, `LITECOIN_ADDRESS`, `MASKED_NUMBER` | `mask` |
| Authentication, network, device, and vehicle identifiers | `PASSWORD`, `PIN`, `API_KEY`, `IP_ADDRESS`, `MAC_ADDRESS`, `USER_AGENT`, `VIN`, `VEHICLE_REGISTRATION`, `IMEI` | `mask` |

## Sections 41-42 and local-first processing

Section 41 restricts transfers of personal data from Nigeria to another country
unless the recipient is covered by an adequate protection mechanism or a
section 43 condition applies. It also requires the transfer basis and section 42
adequacy to be recorded. Section 42 describes adequacy and the factors used to
assess it.

When OpenMed runs entirely on the user's device and sends neither input text nor
de-identified output to a recipient in another country, the de-identification
operation itself is not a cross-border transfer. A fully local workflow
therefore avoids the sections 41-42 transfer-basis and adequacy question for
that operation. Later cloud storage, telemetry, support access, model calls,
exports, or other disclosures can create a separate transfer and must be
assessed independently. The profile does not evaluate adequacy, section 43
conditions, consent, contracts, or any other lawful basis.

## Implementation notes

- `NDPA_SENSITIVE_CLASS_LABELS` is a many-to-many statutory cross-map because
  broad fallback labels such as `OTHER` can anchor more than one category.
- `ng_ndpa` contains no jurisdiction-specific engine branch. It uses the common
  policy loader, compiler, action resolver, mandatory safety sweep, and risk
  budget machinery.
- All committed fixtures for this profile are synthetic and must not be
  replaced with real patient, NIN, BVN, address, phone, or biometric data.

[ndpa]: https://ndpc.gov.ng/wp-content/uploads/2024/03/Nigeria_Data_Protection_Act_2023.pdf
