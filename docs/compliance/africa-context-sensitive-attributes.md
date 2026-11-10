# African context-sensitive attributes

OpenMed's deterministic safety sweep includes a small, extensible seed set for
African healthcare text. It covers named healthcare facilities, mobile-money
references, and explicitly contextualized ethnic or tribal affiliation mentions.
The seed is intentionally not a pan-African gazetteer and contains only synthetic
test data.

This feature is technical decision support, not legal advice. Organisations remain
responsible for choosing a lawful basis, applying local requirements, and reviewing
whether a broader or narrower redaction posture is appropriate.

## Runtime mapping

| Context shape | Canonical label | Policy class | Default posture |
|---|---|---|---|
| Named clinic, mission hospital, dispensary, health centre, or referral facility | `ORGANIZATION` | `QUASI_IDENTIFIER` | Non-keep in the six African profiles |
| M-Pesa, MTN MoMo, Airtel Money, and other configured wallet transaction or account references | `ACCOUNT_NUMBER` | `DIRECT_IDENTIFIER` | Non-keep in the six African profiles |
| Race, ethnicity, or tribal-affiliation mention with explicit nearby context | `ETHNICITY` | `SENSITIVE_ATTRIBUTE` | `mask` in the six African profiles and `strict_no_leak` |

Terms, context words, regex templates, and profile defaults live in
`openmed/core/data/africa_context_terms.json`. The merger only renders those data
entries into its existing `PIIPattern` configuration. Ethnic-affiliation seed terms
require nearby context such as `ethnicity`, `tribe`, or `identifies as`; a standalone
term is not sufficient. Facility matching likewise requires a capitalized facility
name plus a configured healthcare-facility cue. These constraints reduce unrelated
matches while preserving the no-leak posture for planted fixtures.

## Statutory basis

The mappings are conservative engineering controls informed by official legal
texts and, where noted, a reference translation:

- South Africa's [Protection of Personal Information Act 4 of 2013](https://www.justice.gov.za/legislation/acts/2013-004.pdf), section 26, treats race or ethnic origin and health information as special personal information, subject to the Act's authorisations.
- Nigeria's [Data Protection Act 2023](https://www.ndpc.gov.ng/ndp-act-2023/), including section 30 and the Act's definition of sensitive personal data, covers race or ethnic origin and health status.
- The African Union [Convention on Cyber Security and Personal Data Protection](https://au.int/sites/default/files/treaties/29560-treaty-0048_-_african_union_convention_on_cyber_security_and_personal_data_protection_e.pdf), article 14(1), addresses processing that reveals racial, ethnic, or regional origin and other sensitive attributes.
- Kenya's [Data Protection Act 2019](https://new.kenyalaw.org/akn/ke/act/2019/24/eng%402022-12-31), section 2, treats economic identity as potentially identifying and includes property details among sensitive personal data. A mobile-wallet transaction or account reference is therefore handled conservatively as an account identifier rather than as ordinary prose.
- Egypt enacted [Personal Data Protection Law No. 151 of 2020](https://sis.gov.eg/en/media-center/news/sisi-endorses-law-on-personal-data-protection/). An [English reference translation](https://eg.andersen.com/wp-content/uploads/2025/06/Law-No.-151-OF-2020.pdf) describes financial data as sensitive personal data. Mobile-money account and transaction references use the existing `ACCOUNT_NUMBER` canonical label.

## Extending the seed set

Add operators, facility cues, affiliation terms, or regional variants only in
`africa_context_terms.json`. Keep additions synthetic, explain the regional scope,
and add a planted fixture that proves both detection and redaction. Do not add real
patient, facility-account, wallet, or transaction data. Avoid broad affiliation
terms without contextual gating because many group names are also languages,
places, surnames, or ordinary clinical-note content.
