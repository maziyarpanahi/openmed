# HIPAA Safe Harbor — the 18 identifier categories

Authority: **45 CFR 164.514(b)(2)** — the Safe Harbor method of de-identification
under the HIPAA Privacy Rule. To qualify, **all 18** categories below must be
removed for the individual *and* their relatives, employers, and household
members, and the covered entity must have no actual knowledge that the residual
information could re-identify the individual.

Each row maps the Safe Harbor category to the OpenMed HIPAA class
(`openmed/core/labels.py` → `HIPAA_SAFE_HARBOR_CLASSES`) and to the
`CANONICAL_LABELS` that OpenMed detects for it (`LABEL_TO_HIPAA`). The
**Gaps / cautions** column flags where automated detection needs human review.

| # | Safe Harbor identifier (45 CFR 164.514(b)(2)(i)) | OpenMed HIPAA class | OpenMed `CANONICAL_LABELS` | Gaps / cautions |
| --- | --- | --- | --- | --- |
| A | Names | `NAME` | `PERSON`, `FIRST_NAME`, `LAST_NAME`, `MIDDLE_NAME`, `PREFIX` | Nicknames, initials, and provider names in free text may be missed; review. |
| B | Geographic subdivisions smaller than a state (street, city, county, precinct, **ZIP** — keep only first 3 ZIP digits if the area has >20,000 people, else `000`) | `GEOGRAPHIC_SUBDIVISION` | `LOCATION`, `STREET_ADDRESS`, `BUILDING_NUMBER`, `ZIPCODE`, `GPS_COORDINATES`, `ORDINAL_DIRECTION` | **The 3-digit-ZIP rule is a transformation, not just detection** — masking the whole ZIP is safe; truncation logic is the user's. Rare-geography small towns can re-identify. |
| C | All **dates** (except year) directly related to an individual — birth date, admission, discharge, death; **and all ages over 89** and any date/age aggregating to >89 | `DATE_ELEMENT` | `DATE`, `DATE_OF_BIRTH`, `TIME`, `AGE` | **Ages > 89 must be aggregated to a single "90+" category** — OpenMed flags `AGE` but does not auto-cap; see `shifting-clinical-dates`. Year alone may stay. |
| D | Telephone numbers | `TELEPHONE_NUMBER` | `PHONE` | — |
| E | Fax numbers | `FAX_NUMBER` | `PHONE` (fax shares the phone label) | OpenMed does not distinguish fax from phone; both are caught as `PHONE`. |
| F | Email addresses | `EMAIL_ADDRESS` | `EMAIL` | — |
| G | Social Security numbers | `SOCIAL_SECURITY_NUMBER` | `SSN` | Structured-ID safety sweep catches these even below model threshold. |
| H | Medical record numbers | `MEDICAL_RECORD_NUMBER` | `ID_NUM` | MRNs surface as the generic `ID_NUM`; verify hospital-specific formats. |
| I | Health plan beneficiary numbers | `HEALTH_PLAN_BENEFICIARY_NUMBER` | `ID_NUM` | Detected as `ID_NUM`; no dedicated label. |
| J | Account numbers | `ACCOUNT_NUMBER` | `ACCOUNT_NUMBER`, `CREDIT_CARD`, `CVV`, `IBAN`, `BIC`, `MASKED_NUMBER`, crypto addresses | — |
| K | Certificate / license numbers | `CERTIFICATE_LICENSE_NUMBER` | `ID_NUM` | Detected as `ID_NUM`; review professional-license formats. |
| L | Vehicle identifiers and serial numbers, including license plates | `VEHICLE_IDENTIFIER` | `VIN`, `VEHICLE_REGISTRATION` | — |
| M | Device identifiers and serial numbers | `DEVICE_IDENTIFIER` | `IMEI`, `MAC_ADDRESS` | Implant/device serials embedded in narrative text may be missed; review. |
| N | Web URLs | `URL` | `URL` | — |
| O | IP addresses | `IP_ADDRESS` | `IP_ADDRESS` | — |
| P | Biometric identifiers (finger/voice prints) | `BIOMETRIC_IDENTIFIER` | — (no dedicated text label) | **Gap:** biometrics are rarely present as text; handle out-of-band. |
| Q | Full-face photographs and comparable images | `FULL_FACE_PHOTO` | — (out of scope for text) | **Gap:** image content is not text PII; strip in the imaging pipeline. |
| R | Any other unique identifying number, characteristic, or code | `UNIQUE_IDENTIFIER` | `USERNAME`, `PASSWORD`, `PIN`, `API_KEY`, `ORGANIZATION`, `JOB_TITLE`, `OCCUPATION`, `GENDER`, `USER_AGENT`, `OTHER`, etc. | Catch-all; rare characteristics (e.g. an unusual occupation + small town) can re-identify even when each field alone is fine. |

## The "actual knowledge" requirement

Safe Harbor also requires the covered entity to have **no actual knowledge** that
the remaining information could be used alone or in combination to identify the
individual. Detecting and masking all 18 categories is necessary but **not
sufficient** — a reviewer must still judge residual re-identification risk
(rare diagnoses, outlier ages, unusual geography). Cross-check with the
`residual_risk` field of an `AuditReport` (see `auditing-deidentification-runs`).

## When Safe Harbor is the wrong tool

If you must retain dates, ages, or geography for analysis, Safe Harbor cannot be
met; use **Expert Determination** (45 CFR 164.514(b)(1)) or a **Limited Data
Set** under a Data Use Agreement (45 CFR 164.514(e)). See
`configuring-privacy-policies` for the matching OpenMed profiles
(`hipaa_expert_review_assist`, `research_limited_dataset`).

## Source

- 45 CFR 164.514(b)(2): https://www.ecfr.gov/current/title-45/subtitle-A/subchapter-C/part-164/subpart-E/section-164.514
- HHS de-identification guidance: https://www.hhs.gov/hipaa/for-professionals/privacy/special-topics/de-identification/index.html
- OpenMed mapping: `openmed/core/labels.py` (`LABEL_TO_HIPAA`,
  `HIPAA_SAFE_HARBOR_CLASSES`, `CANONICAL_LABELS`).
