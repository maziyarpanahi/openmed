# Pan-African Malabo Convention baseline

!!! warning "Decision support, not legal advice"
    The `africa_malabo_baseline` profile is a conservative technical baseline.
    It does not determine whether processing is lawful, establish that an
    Article 14 exception applies, certify compliance, or replace review by
    qualified counsel and the relevant national authority.

The profile gives deployments a common local-first starting point when a more
specific national profile is unavailable. It is aligned to the sensitive-data
categories in Article 14(1) of the
[African Union Convention on Cyber Security and Personal Data Protection][au-text].
It masks every canonical label, requires the deterministic safety sweep, uses
high-recall arbitration, retains no replacement mapping, and creates no
reversible identifier.

This is a floor for technical de-identification, not a harmonized legal rule.
Applicable national law prevails wherever it is stricter or more specific.
Use `za_popia`, `ng_ndpa`, `ke_dpa`, `eg_pdpl`, or `ma_law_09_08` when the
corresponding national law applies, and validate any additional local
requirements independently.

## Article 14(1) category map

The category text in the first column is reproduced verbatim from Article
14(1). The canonical labels are conservative technical anchors. `OTHER` is the
non-`keep` fallback for sensitive concepts that do not yet have a dedicated
canonical label.

| Article 14(1) category | Canonical label anchors | Profile action |
|---|---|---|
| `racial, ethnic and regional origin` | `ETHNICITY`, `LOCATION`, `OTHER` | `mask` |
| `parental filiation` | `PERSON`, `FIRST_NAME`, `LAST_NAME`, `OTHER` | `mask` |
| `political opinions` | `ORGANIZATION`, `OTHER` | `mask` |
| `religious or philosophical beliefs` | `OTHER` | `mask` |
| `trade union membership` | `ORGANIZATION`, `JOB_DEPARTMENT`, `OTHER` | `mask` |
| `sex life` | `GENDER`, `OTHER` | `mask` |
| `genetic information` | `GENE_SYMBOL`, `VARIANT_DESCRIPTOR`, `PROTEIN_CHANGE`, `ZYGOSITY`, `CLINICAL_SIGNIFICANCE` | `mask` |
| `data on the state of health of the data subject` | `CONDITION`, `MEDICATION`, `LAB_TEST`, `PROCEDURE`, `BODY_SITE`, `GENE_SYMBOL`, `CLINICAL_SIGNIFICANCE` | `mask` |

The complete machine-readable crosswalk lives in
`tests/unit/core/fixtures/africa_statute_coverage.json`. Its data-driven test
compiles every listed African profile and verifies that each declared class
resolves to canonical labels with non-`keep` actions.

## Operating limits

- A policy action protects a correctly detected span; it does not guarantee
  discovery of every belief, affiliation, family relationship, genetic
  attribute, or health reference. Validate detector recall against approved
  synthetic or otherwise lawfully controlled local data.
- `OTHER` is intentionally masked, but sites should add approved custom
  recognizers when a material sensitive class lacks a dedicated detector.
- Article 14 contains exceptions that require a separate legal assessment.
  Selecting this profile does not assert consent, necessity, public interest,
  or any other exception.
- Local processing reduces exposure but does not decide transfer, residency,
  retention, security, or disclosure obligations. Review every later network,
  cloud, telemetry, support, backup, and export path.
- A zero-leakage synthetic test is release evidence, not a guarantee for
  production data. Re-run leakage and re-identification assessment for the
  actual languages, formats, models, and deployment boundary.

The [African Union treaty page][au-page] provides the official treaty record
and language editions.

[au-page]: https://au.int/en/treaties/african-union-convention-cyber-security-and-personal-data-protection
[au-text]: https://au.int/sites/default/files/treaties/29560-treaty-0048_-_african_union_convention_on_cyber_security_and_personal_data_protection_e.pdf
