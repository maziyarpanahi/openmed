# OpenMed workflow router

Use [ask-openmed](../../skills/ask-openmed/SKILL.md) when a request is broad or
spans several OpenMed skills. The router returns one existing skill identifier
and a next handoff; it does not inspect clinical payloads, make a clinical
decision, download a model, or call a network service.

## Fixed routing order

Normalize the stated goal by lowercasing it, trimming it, and collapsing
whitespace. If a downstream operation touches content whose sensitivity is
unclear, start at the privacy gate. An intake-only request may start with its
format parser, but the gate must precede extraction, exchange, or verification.
For an explicit multi-stage request, continue in this order: intake → privacy →
extraction → exchange → verification.

| Goal | First route |
| --- | --- |
| Scan, fax, image, PDF, CSV, or document intake | [ingesting-clinical-documents](../../skills/ingesting-clinical-documents/SKILL.md) |
| Raw or unredacted clinical text, identifiers, or sharing | [deidentifying-clinical-text](../../skills/deidentifying-clinical-text/SKILL.md) |
| Clinical or biomedical entity extraction | [extracting-clinical-entities](../../skills/extracting-clinical-entities/SKILL.md) |
| FHIR resource or Bundle creation | [exporting-to-fhir](../../skills/exporting-to-fhir/SKILL.md) |
| Leakage, audit, risk, or conformance verification | [auditing-deid-leakage](../../skills/auditing-deid-leakage/SKILL.md) |
| No recognizable goal | [building-with-openmed](../../skills/building-with-openmed/SKILL.md) |

## Ambiguity examples

- “Turn this note into FHIR” has no sensitivity declaration: choose the
  privacy gate first, then [extracting-clinical-entities](../../skills/extracting-clinical-entities/SKILL.md),
  then [exporting-to-fhir](../../skills/exporting-to-fhir/SKILL.md).
- “Extract diagnoses from a synthetic note” is explicit about safety: choose
  [extracting-clinical-entities](../../skills/extracting-clinical-entities/SKILL.md)
  directly.
- “Is this de-identified output safe to share?” chooses
  [auditing-deid-leakage](../../skills/auditing-deid-leakage/SKILL.md); if
  de-identification is not stated, apply the privacy gate first.

Route diagnostics must contain only the category, selected identifier, rule
index, and next handoff. Do not copy source values into logs, exceptions,
reports, or fixtures. The focused skill may define a user-supplied service or
network step, but the router has no mandatory network dependency.
