# EU Pseudonymisation and Anonymisation Terms for OpenMed Outputs

> **Template — requires legal review. Not legal advice.** This page explains
> how EU data protection law and guidance use the terms "pseudonymisation" and
> "anonymisation", and how they apply by default to OpenMed outputs. It does
> not determine the legal status of any dataset.

OpenMed's documentation uses "de-identification" for the whole process of
detecting identifiers and transforming them. Some pages, including
[PII Anonymization](../anonymization.md), use "anonymization" loosely for the
same process. EU law draws a sharper line (GDPR Article 4(5) and Recital 26),
and the defaults below follow it: masking, hashing, surrogates and date
shifting produce pseudonymised data, which remain personal data, unless a
separate risk assessment supports a claim that the data are anonymous.

Sources were checked on 2026-11-16. "GDPR" means Regulation (EU) 2016/679.

## Terms

| Term | Meaning | Source |
| --- | --- | --- |
| Personal data | Information relating to an identified or identifiable natural person | GDPR Article 4(1) |
| Pseudonymisation | Processing so that personal data can no longer be attributed to a specific person without additional information, kept separately and protected by technical and organisational measures | GDPR Article 4(5); EDPB Guidelines 01/2025, paragraph 16 |
| Pseudonymised data | Personal data that could be attributed to a person by the use of additional information. They remain information on an identifiable person. | GDPR Recital 26; EDPB Guidelines 01/2025, paragraph 22 |
| Anonymous information | Information that does not relate to an identified or identifiable person, or data rendered anonymous so that the person is no longer identifiable. The GDPR does not apply to it. | GDPR Recital 26 |
| Additional information | Information, such as lookup tables or cryptographic keys, whose use allows pseudonymised data to be attributed to a person | EDPB Guidelines 01/2025, paragraphs 19 and 20 |
| Pseudonymisation domain | The context, chosen by the controller, in which pseudonymisation is meant to prevent attribution | EDPB Guidelines 01/2025, paragraph 35 |
| De-identification | OpenMed's umbrella term for detecting and transforming identifiers. It is not a GDPR term and implies no legal category. | This documentation |

## What the sources say

- Pseudonymised data that could be attributed to a person by the use of
  additional information are information on an identifiable person (GDPR
  Recital 26). The EDPB adds that this holds even when the pseudonymised data
  and the additional information are held by different persons, and that
  erasing the controller's additional information makes the data anonymous
  only if the conditions for anonymity are met (EDPB Guidelines 01/2025,
  paragraph 22).
- Identifiability is assessed against all the means reasonably likely to be
  used, by the controller or by another person, taking into account costs,
  time and available technology (GDPR Recital 26). In Case C-582/14, Breyer
  (judgment of 19 October 2016), the Court of Justice held that a dynamic IP
  address is personal data for a website operator that has the legal means to
  identify the user with additional data held by the internet service provider
  (operative part, point 1). Means are not reasonably likely to be used if
  identification is prohibited by law or practically impossible
  (paragraph 46). That case was decided under Directive 95/46/EC.
- In Case C-413/23 P, EDPS v SRB (judgment of 4 September 2025), the Court of
  Justice held that pseudonymised data must not be regarded as personal data
  in all cases and for every person (paragraph 86). For a recipient that
  cannot lift the pseudonymisation measures or re-identify the persons by other
  means, the data may not be personal data (paragraph 77). Whether a person is
  identifiable for the controller's duty to inform is assessed at the time of
  collection and from the controller's point of view (paragraph 111). The case
  concerned Regulation (EU) 2018/1725, which applies to EU institutions and
  defines personal data and pseudonymisation in the same terms as the GDPR.
  EDPB Guidelines 01/2025 were adopted before this judgment.
- The GDPR names pseudonymisation as a measure for data protection by design
  (Article 25(1)), for security of processing (Article 32(1)(a)), for
  assessing whether further processing is compatible (Article 6(4)(e)) and as
  a safeguard for research and statistics (Article 89(1)). Recital 28 notes
  that pseudonymisation can reduce risks but does not preclude other measures.
- The Article 29 Working Party's Opinion 05/2014 on Anonymisation Techniques
  (WP216, adopted on 10 April 2014) states that pseudonymisation is not a
  method of anonymisation. It assesses anonymisation techniques against three
  questions: whether a person can still be singled out, whether records about
  a person can be linked, and whether information about a person can be
  inferred.

The EDPB published Guidelines 01/2025 on Pseudonymisation, adopted on
16 January 2025, as a version for public consultation. Check the EDPB website
for a later final version before relying on paragraph numbers.

## OpenMed actions and their default category

The policy actions come from [privacy policy as data](../security/policy-as-data.md)
(`keep`, `redact`, `replace`, `mask`, `hash`, `format_preserve`). The
`deidentify()` methods come from [PII Anonymization](../anonymization.md)
(`mask`, `remove`, `replace`, `hash`, `shift_dates`) and the `deidentify()`
docstring, which also lists `aadhaar_mask` and `format_preserve`. The
structured-data and aggregate outputs come from
[structured release risk](../reidentification-risk.md), the
[structured privacy risk lab](../privacy/structured-risk-lab.md) and the
[privacy budget ledger](../security/privacy-budget.md).

"Default category" is the starting assumption for an output while the
controller or anyone else holds the source records, keys, mappings or other
additional information. It is not a legal conclusion about a dataset.

| OpenMed action | What the output keeps | Default category | Evidence an anonymisation claim would need |
| --- | --- | --- | --- |
| `keep` | The original value. | Personal data (GDPR Article 4(1)). | None can support it: the identifier is still present. |
| `mask` and `redact` | A typed placeholder such as `[NAME]`. With `keep_mapping=True`, a mapping that restores the original. | Personal data; pseudonymised data where the mapping or source records are kept separately and protected (GDPR Article 4(5), Recital 26). | An assessment of the remaining text against singling out, linkability and inference (WP216), covering quasi-identifiers that the placeholders leave in place, and the deletion or isolation of any mapping (EDPB Guidelines 01/2025, paragraph 22). |
| `aadhaar_mask` | For a valid Aadhaar number, the masked form `XXXX XXXX NNNN` with only the last four digits; other entities get placeholders as with `mask`. | Personal data; pseudonymised data where the source records are kept separately (GDPR Article 4(5), Recital 26). | The assessment for `mask`, including whether the last four digits can be linked with other data (WP216). |
| `remove` | Nothing in place of the span. | Personal data; pseudonymised data where the source records are kept separately (GDPR Article 4(5), Recital 26). | The same assessment as for `mask`. |
| `hash` | An entity-typed token built from an unkeyed SHA-256 digest of the span, so the same value always gives the same token. | Pseudonymised data at best (GDPR Recital 26). Anyone who can guess candidate values can recompute the digest and match it (EDPB Guidelines 01/2025, paragraph 88 and its footnote 23). | Evidence that the token cannot be recomputed or linked by anyone with means reasonably likely to be used (GDPR Recital 26), plus the assessment for `mask`. Keyed transformations are the EDPB's suggested approach (EDPB Guidelines 01/2025, paragraphs 88 and 89). |
| `replace` (surrogates) | A realistic surrogate. With `consistent=True`, a `seed` or a `SurrogateVault`, the same value maps to the same surrogate. | Pseudonymised data (GDPR Article 4(5)). Seeds, vaults and mappings are additional information (EDPB Guidelines 01/2025, paragraph 20). | Evidence that seeds, vaults and mappings are destroyed or held outside the pseudonymisation domain, and the assessment for `mask`, including linkability through consistent surrogates (WP216). |
| `format_preserve` | A synthetic value that keeps the original's shape and separators; unsupported labels are masked. | Pseudonymised data (GDPR Article 4(5)). | The evidence for `replace`, plus a check that the kept format does not itself identify a person (GDPR Recital 26). |
| Reversible pseudonyms (`keep_mapping=True` with `reidentify()`, or the `gdpr_pseudonymization` policy) | Output plus a mapping that restores the original. | Pseudonymised data (GDPR Article 4(5)); the mapping is additional information (EDPB Guidelines 01/2025, paragraph 20). | Not available while the mapping exists, for whoever can use it (EDPB Guidelines 01/2025, paragraph 22; Case C-413/23 P, paragraphs 85 and 111). |
| `shift_dates` | Dates moved by a fixed offset or by a per-patient offset derived with HMAC from a caller-supplied secret, so intervals are preserved. | Pseudonymised data at best (GDPR Article 4(5)). The offset or secret is additional information (EDPB Guidelines 01/2025, paragraph 20). | Evidence that preserved intervals and the remaining content do not single out or link a person (WP216), and that the secret is protected or destroyed. |
| k-anonymity outputs (generalisation and suppression of structured records) | Generalised quasi-identifiers and whole privacy units suppressed, with k-anonymity, l-diversity, t-closeness and reference-population measurements. | Personal data until an assessment supports anonymity (GDPR Recital 26). | The measurements, reviewed column roles, the reference population, recipients, auxiliary data and composition with other releases, and a documented conclusion by a qualified reviewer, assessed against singling out, linkability and inference (WP216). |
| Differential-privacy outputs (noisy aggregates) | Noisy aggregate values and accounting metadata. Outputs are marked `aggregate_only` and `row_level_anonymization: false`. | The source records stay personal data. Whether released aggregates relate to an identifiable person depends on the parameters and the release context (GDPR Recital 26). | The epsilon and delta spent, composition across releases, confirmation that no test seed was used, and an assessment of inference risk for the released statistics (WP216). |

OpenMed reports measurements and records. It does not decide that any output is
anonymous, and it does not make any processing compliant with the GDPR.

## Using this page

- Record the default category, not "anonymised", in a
  [Data Protection Impact Assessment](dpia-template.md) or data-sharing
  record, unless a documented assessment supports anonymity.
- Keep mappings, vaults, seeds and keys apart from pseudonymised output, for
  example with [encrypted surrogate mappings](../security/encrypted-surrogate-vault.md)
  and [key management](../security/key-management.md).
- Use [structured release risk and expert-review evidence](../reidentification-risk.md)
  for structured releases. Its output supports a qualified reviewer's
  conclusion and is not that conclusion.

See the [EU evidence index](eu-evidence.md) and the
[compliance posture](../compliance.md) for OpenMed's other evidence pages.
