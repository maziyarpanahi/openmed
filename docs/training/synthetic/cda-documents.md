# Synthetic C-CDA documents with element-level PHI truth

`openmed.training.synthetic.cda_documents` renders seeded continuity of care
documents (CCD), discharge summaries and progress notes, records every PHI
value it places, and runs a leakage harness over
[`redact_cda()`](../../getting-started.md). Every value comes from
repository-authored synthetic pools. The module bundles no real PHI, no
restricted vocabulary and no model weights.

```python
from openmed.training.synthetic.cda_documents import (
    evaluate_cda_leakage,
    generate_cda_documents,
    resolve_cda_locator,
)

documents = generate_cda_documents(9, seed=3885)
truth = documents[0].truth[0]
node_value = resolve_cda_locator(documents[0].xml, truth)
assert node_value[truth.start : truth.end] == truth.value

report = evaluate_cda_leakage(documents, date_shift_days=17)
for position in report.residual_positions:
    print(position.path, position.label, position.residual_count)
```

## Documents

Each document is a `ClinicalDocument` in the `urn:hl7-org:v3` namespace that
passes `is_cda_document()`. Headers carry a `recordTarget` with patient
identifiers, address, telecom, name and birth time, a guardian and a
birthplace, one or more authors, related-person informants, a custodian
organization, a legal authenticator and next-of-kin participants. CCDs add a
`documentationOf/serviceEvent` interval; discharge summaries and progress
notes add a `componentOf/encompassingEncounter` interval.

Section narratives use paragraphs, tables, lists and `content` elements with
`ID` attributes. Allergy, problem and procedure entries point at their
narrative rows with `<text><reference value="#..."/></text>`, and procedure
entries carry a `performer`. `CdaSynthesisConfig` bounds the number of
authors, informants, next-of-kin participants and entries, and can omit the
guardian or birthplace.

Terminology codes are placeholders: every `code` attribute is `SYN-*` in the
`CDA_PLACEHOLDER_CODE_SYSTEM` (`2.16.840.1.113883.19.999`, under the HL7
example arc). Document and section `templateId` values are public C-CDA
template identifiers, not vocabulary codes.

## Truth model

Each `CdaTruthValue` records:

- `path`: an indexed path such as
  `/ClinicalDocument/recordTarget[1]/patientRole[1]/telecom[1]`.
- `node`: `attribute` (with `attribute`), `text`, or `tail` (the text after
  the element's end tag, as in `xml.etree.ElementTree`).
- `start`/`end`: half-open offsets into that node value. Telecom offsets skip
  the `tel:` or `mailto:` scheme.
- `document_start`/`document_end`: half-open offsets into the serialized
  document text.
- `label` (canonical OpenMed label), `role` (for example `guardian` or
  `custodian`) and `context` (`header`, `entry` or `narrative`).

`CdaMentionLink` connects each narrative mention to the header or entry values
it repeats, with the enclosing `content/@ID` when there is one.
`CdaContentReference` records each entry `reference` and the narrative
`content` it resolves to. `manifest()` returns a JSON-compatible record with
the document digest, and `manifest_sha256` hashes it.

The same seed, settings and package version produce byte-identical documents
and manifests; different seeds produce different documents.

## Leakage harness

`evaluate_cda_leakage()` runs `redact_cda()` on each document, parses the
output after refusing DTD and entity declarations, and requires the root to
pass `is_cda_document()`. It never fetches schemas. A truth value is residual
when its value is still present, as a whole token, in the node at the same
indexed path. A multi-token value that is not residual counts as partial when
one of its tokens of three or more characters remains. Malformed output
counts every truth value of that document as residual.

`CdaLeakageReport` contains only index-free paths such as
`/ClinicalDocument/author/time/@value`, canonical labels and counts. It never
stores or renders values or XML, and `CdaLeakageError` carries only a
controlled code and document index. The harness accepts the same
`element_map`, `text_redactor`, `date_shift_days`, `keep_year` and
`hash_salt` options as `redact_cda()`, so candidate rule sets can be measured
before they change the defaults.

## Known residuals with the default rules

With `generate_cda_documents(300, seed=3885)` and `date_shift_days=17`, the
default rules leave these values in place (rate is residual / truth count):

| Path | Labels | Rate |
| --- | --- | --- |
| `/ClinicalDocument/author/time/@value` | DATE | 1.00 |
| `/ClinicalDocument/author/assignedAuthor/id/@extension` | ID_NUM | 1.00 |
| `/ClinicalDocument/author/assignedAuthor/addr/*` | STREET_ADDRESS, LOCATION, ZIPCODE | 1.00 |
| `/ClinicalDocument/author/assignedAuthor/telecom/@value` | PHONE | 1.00 |
| `/ClinicalDocument/custodian/.../representedCustodianOrganization/*` | ORGANIZATION, STREET_ADDRESS, LOCATION, ZIPCODE, PHONE | 1.00 |
| `/ClinicalDocument/legalAuthenticator/time/@value` | DATE | 1.00 |
| `/ClinicalDocument/legalAuthenticator/assignedEntity/id/@extension` | ID_NUM | 1.00 |
| `.../entry/procedure/performer/assignedEntity/assignedPerson/name/*` | FIRST_NAME, LAST_NAME | 1.00 |
| Narrative mentions of the custodian organization | ORGANIZATION | 0.72 to 0.76 |
| Narrative table cells naming procedure performers | PERSON | 0.28 (shared with redacted contact names) |

Patient, guardian, birthplace, next-of-kin and informant values, and
narrative dates and phone numbers, have no full residuals. The harness also
reports partial narrative dates such as `Aug[ADDRESS]t 31, 2019`: two-letter
address surfaces (for example a country or state code) are matched without
regard to case inside words, and adjacent table cells are joined without a
boundary, so the date sweep no longer matches the value. Fixing the default
rules is out of scope for the generator; treat these rows as a baseline.

## Unsafe options

`schema_location` and `stylesheet` accept only local `.xsd` and `.xsl` file
names. URLs, `file:` and protocol-relative references, absolute or parent
paths, and values carrying markup, `DOCTYPE`, `ENTITY` or entity references
raise `CdaSynthesisError` with the code `unsafe_schema_location` or
`unsafe_stylesheet`; the message never echoes the rejected value. Generated
documents never contain a `DOCTYPE`, entity declarations or entity references.

This module creates evaluation data only. It does not validate C-CDA
conformance (Schematron), certify de-identification, or make clinical
decisions.
