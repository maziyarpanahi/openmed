# Egypt PDPL and Morocco Law 09-08 decision checklist

!!! warning "Decision support, not legal advice"
    This checklist helps engineering and privacy teams identify controls and
    questions for qualified counsel. It does not determine whether a processing
    activity is lawful, obtain a licence or authorization, assess foreign-country
    adequacy, or replace instructions from Egypt's Personal Data Protection Centre
    or Morocco's CNDP.

## Legal anchors

### Egypt — Personal Data Protection Law No. 151 of 2020

The [official State Information Service notice](https://sis.gov.eg/ar/%D8%A7%D9%84%D8%B1%D8%A6%D8%A7%D8%B3%D8%A9/%D8%B4%D8%A6%D9%88%D9%86-%D8%AF%D8%A7%D8%AE%D9%84%D9%8A%D8%A9/%D8%A7%D9%84%D9%82%D8%B1%D8%A7%D8%B1%D8%A7%D8%AA-%D8%A7%D9%84%D8%B1%D8%A6%D8%A7%D8%B3%D9%8A%D8%A9/%D8%A7%D9%84%D8%B1%D8%A6%D9%8A%D8%B3-%D8%A7%D9%84%D8%B3%D9%8A%D8%B3%D9%89%D9%8A-%D9%8A-%D8%B5%D8%AF-%D9%82-%D8%B9%D9%84%D9%89-%D9%82%D8%A7%D9%86%D9%88%D9%86-%D8%AD%D9%85%D8%A7%D9%8A%D8%A9-%D8%A7%D9%84%D8%A8%D9%8A%D8%A7%D9%86%D8%A7%D8%AA-%D8%A7%D9%84%D8%B4%D8%AE%D8%B5%D9%8A%D8%A9/)
records the enactment of Law No. 151 of 2020. An
[Arabic/English reference copy](https://www.privacylaws.com/media/3263/egypt-data-protection-law-151-of-2020.pdf)
provides the article text used for this engineering map.

- Article 12 prohibits collection, transfer, storage, retention, processing, or
  access to sensitive personal data without a Centre licence. Outside cases
  authorized by law, it also requires explicit written consent; child data has
  additional guardian-consent requirements.
- Article 13 requires security policies and procedures intended to prevent a
  breach or violation of sensitive personal data.
- Article 14 conditions foreign transfer, sharing, or storage on protection at
  least equivalent to the Law and a Centre licence or permit.
- Articles 15–16 define limited transfer paths and conditions. Do not treat a
  technical de-identification profile as evidence that an exception applies.

### Morocco — Law No. 09-08

The Moroccan Ministry of Justice publishes the
[official consolidated French text of Law No. 09-08](https://adala.justice.gov.ma/api/uploads/2024/04/30/Protection%20des%20personnes%20physiques-1714464099884.pdf).

- Article 1 defines personal data broadly and defines sensitive data to include
  health and genetic data.
- Article 21 makes processing sensitive data subject to authorization by law or,
  absent that, authorization from the CNDP. Article 22 contains specific health
  processing declaration cases that require legal review before use.
- Article 43 permits transfer to a foreign state only where the state provides a
  sufficient level of protection, assessed against the listed factors.
- Article 44 lists consent and necessity derogations and permits CNDP-authorized
  transfers backed by sufficient guarantees. A profile cannot decide that a
  derogation, adequacy finding, or authorization exists.

## Canonical control map

Both bundled profiles deliberately use `mask` for every canonical label, disable
reversible mappings, require the safety sweep, and select the high-recall cascade.
This is a conservative engineering posture for data that has not yet passed the
applicable licence, authorization, consent, purpose, and transfer review.

| Data class | Canonical labels | `eg_pdpl` | `ma_law_09_08` |
|---|---|---:|---:|
| Person and identity | `PERSON`, `FIRST_NAME`, `LAST_NAME`, `ID_NUM` | `mask` | `mask` |
| Contact and address | `PHONE`, `EMAIL`, `STREET_ADDRESS`, `LOCATION` | `mask` | `mask` |
| Dates and demographics | `DATE`, `DATE_OF_BIRTH`, `AGE`, `GENDER` | `mask` | `mask` |
| Health and care | `CONDITION`, `MEDICATION`, `LAB_TEST`, `PROCEDURE` | `mask` | `mask` |
| Genetic information | `GENE_SYMBOL`, `VARIANT_DESCRIPTOR`, `ZYGOSITY` | `mask` | `mask` |
| Financial information | `ACCOUNT_NUMBER`, `CREDIT_CARD`, `IBAN`, `AMOUNT` | `mask` | `mask` |
| Unclassified sensitive data | `OTHER` | `mask` | `mask` |

## Engineering checklist

- [ ] Identify the controller, processor, purpose, data subjects, data classes,
  retention period, recipients, and every location where data is processed.
- [ ] Obtain legal confirmation of the required PDPC licence or CNDP
  authorization/declaration and the applicable consent path before processing.
- [ ] Keep raw clinical data on device by default. Record and review any network,
  cloud, telemetry, support, backup, or model-serving path before enabling it.
- [ ] Treat remote inference, hosted logging, cross-region backup, and foreign
  support access as potential cross-border access or transfer requiring review.
- [ ] Use `policy="eg_pdpl", lang="ar", locale="ar_EG"` for Egyptian data or
  `policy="ma_law_09_08", lang="ar", locale="ar_MA"` for Moroccan data.
- [ ] Test Arabic and Latin-script names, local addresses, `+20`/`+212` phones,
  Egyptian national IDs or Moroccan CINs, and Gregorian/Hijri date renderings.
- [ ] Confirm the output contains no residual identifiers and inspect only
  raw-text-free audit evidence. Never place raw identifiers in logs or fixtures.
- [ ] Keep `keep_mapping=false`; store no re-identification map unless counsel and
  the approving authority explicitly require and govern a separate workflow.
- [ ] Re-run the legal review whenever purpose, recipient, hosting country,
  processor, model endpoint, retention, or applicable regulations change.
