# OpenMed workflow router

Use [ask-openmed](https://github.com/maziyarpanahi/openmed/blob/master/skills/ask-openmed/SKILL.md) when a request is broad or
spans several OpenMed skills. The router returns one existing skill identifier
and a next handoff; it does not inspect clinical payloads, make a clinical
decision, download a model, or call a network service.

## Fixed routing order

Normalize the stated goal by lowercasing it, trimming it, and collapsing
whitespace. First select a matching intake route, even when the goal also names
a later stage. If sensitivity is unclear, its next handoff is the privacy gate,
which must precede extraction, exchange, or verification. For a goal with no
intake cue, unclear clinical or personal content starts at the privacy gate.
Otherwise, inspect the remaining tables and their rows in this fixed order:
privacy → extraction → exchange → verification. Match each comma-separated cue
as a case-insensitive substring; the first match wins and no new synonyms are
inferred.

| Goal | First route |
| --- | --- |
| Scan, fax, image, PDF, CSV, or document intake | [ingesting-clinical-documents](https://github.com/maziyarpanahi/openmed/blob/master/skills/ingesting-clinical-documents/SKILL.md) |
| Raw or unredacted clinical text, identifiers, or sharing | [deidentifying-clinical-text](https://github.com/maziyarpanahi/openmed/blob/master/skills/deidentifying-clinical-text/SKILL.md) |
| Clinical or biomedical entity extraction | [extracting-clinical-entities](https://github.com/maziyarpanahi/openmed/blob/master/skills/extracting-clinical-entities/SKILL.md) |
| FHIR resource or Bundle creation | [exporting-to-fhir](https://github.com/maziyarpanahi/openmed/blob/master/skills/exporting-to-fhir/SKILL.md) |
| Leakage, audit, risk, or conformance verification | [auditing-deid-leakage](https://github.com/maziyarpanahi/openmed/blob/master/skills/auditing-deid-leakage/SKILL.md) |
| No recognizable goal | [building-with-openmed](https://github.com/maziyarpanahi/openmed/blob/master/skills/building-with-openmed/SKILL.md) |

## Ambiguity examples

- “Turn this note into FHIR” has no sensitivity declaration: choose the
  privacy gate first, then [extracting-clinical-entities](https://github.com/maziyarpanahi/openmed/blob/master/skills/extracting-clinical-entities/SKILL.md),
  then [exporting-to-fhir](https://github.com/maziyarpanahi/openmed/blob/master/skills/exporting-to-fhir/SKILL.md).
- “Extract diagnoses from a synthetic note” is explicit about safety: choose
  [extracting-clinical-entities](https://github.com/maziyarpanahi/openmed/blob/master/skills/extracting-clinical-entities/SKILL.md)
  directly.
- “Is this de-identified output safe to share?” chooses
  [auditing-deid-leakage](https://github.com/maziyarpanahi/openmed/blob/master/skills/auditing-deid-leakage/SKILL.md); if
  de-identification is not stated, apply the privacy gate first.

Route diagnostics must contain only the category, selected identifier, matched
rule index, and next handoff. Do not copy source values into logs, exceptions,
reports, or fixtures. The focused skill may define a user-supplied service or
network step, but the router has no mandatory network dependency.
