---
name: querying-terminology-service
description: "Call a user-supplied FHIR terminology server ($validate-code, $expand, $lookup, $translate) to validate and expand clinical codes without bundling restricted vocabulary (SNOMED CT, RxNorm, LOINC, ICD-10) into OpenMed. Covers a thin local client, ValueSet $expand with filters/ECL, CodeSystem $lookup, ConceptMap $translate, and pointing at Ontoserver / HAPI / tx.fhir.org. Use as the grounding step for OpenMed coding skills — turn an OpenMed entity span into a validated coded CodeableConcept — when the user mentions terminology server, $validate-code, $expand, ValueSet, ECL, SNOMED/RxNorm/LOINC lookups, or code validation. Pairs adjacent."
license: Apache-2.0
metadata:
  project: OpenMed
  category: fhir-interop
  pairs: adjacent
  version: "1.0"
---

# Querying a Terminology Service

OpenMed deliberately **bundles no restricted vocabulary** — no SNOMED CT,
RxNorm, LOINC, ICD-10, UMLS. So when an OpenMed entity span needs a *validated*
code (the grounding step `exporting-to-fhir` references), you call a **FHIR
terminology server** the user already operates, with their own license. This
skill is the thin client the coding skills lean on.

## When to use

Use it whenever a span must become a *coded* `CodeableConcept`, when you need to
confirm a code is valid in a system, expand a ValueSet for a picklist, look up a
display, or map between vocabularies. Triggers: "terminology server",
"$validate-code", "$expand", "ValueSet", "ECL", "is this a valid SNOMED/LOINC/
RxNorm code", "translate ICD-10 to SNOMED". It sits between OpenMed NER and
`exporting-to-fhir`.

## Bring your own server

The four operations are standard FHIR; point the client at whichever server the
user is licensed for:

- **Ontoserver** (CSIRO) — production SNOMED CT/LOINC, full ECL.
- **HAPI FHIR** terminology module — self-hosted.
- **tx.fhir.org** — HL7 public server (open content only; **not** for licensed
  SNOMED/full LOINC, and not for PHI).

OpenMed never ships or proxies these — the credentials and content are the
user's.

## The four operations

```
POST [tx]/CodeSystem/$validate-code   -> is this code valid in this system?
POST [tx]/ValueSet/$expand            -> enumerate the codes in a value set
POST [tx]/CodeSystem/$lookup          -> display + properties for a code
POST [tx]/ConceptMap/$translate       -> map a code from one system to another
```

### `$validate-code` — confirm before you emit

```bash
curl -s -X POST 'https://tx.example/fhir/CodeSystem/$validate-code' \
  -H 'Content-Type: application/fhir+json' -d '{
    "resourceType": "Parameters",
    "parameter": [
      {"name": "url",  "valueUri":  "http://snomed.info/sct"},
      {"name": "code", "valueCode": "44054006"},
      {"name": "display", "valueString": "Diabetes mellitus type 2"}
    ]}'
# -> Parameters: { result: true, display: "Diabetes mellitus type 2" }
```

### `$expand` — enumerate a ValueSet (with ECL for SNOMED)

```bash
# Expand "disorders of the lung" via an implicit SNOMED ECL value set
curl -s -X POST 'https://tx.example/fhir/ValueSet/$expand' \
  -H 'Content-Type: application/fhir+json' -d '{
    "resourceType": "Parameters",
    "parameter": [
      {"name": "url", "valueUri":
        "http://snomed.info/sct?fhir_vs=ecl/<<19829001"},
      {"name": "filter", "valueString": "pneumonia"},
      {"name": "count", "valueInteger": 20}
    ]}'
```

`<<19829001` is ECL for "19829001 (Disorder of lung) or any subtype". Use
`$expand` + `filter` to power autocomplete and to constrain which codes a span
may map to.

### `$lookup` and `$translate`

```bash
# Display + properties for a LOINC code
POST [tx]/CodeSystem/$lookup  { url=http://loinc.org, code=4548-4 }

# Map an ICD-10-CM code to SNOMED via a ConceptMap
POST [tx]/ConceptMap/$translate {
  url=<conceptmap-url>, system=http://hl7.org/fhir/sid/icd-10-cm,
  code=E11.9, targetsystem=http://snomed.info/sct }
```

## A thin client used by the coding skills

```python
import requests

class TxClient:
    def __init__(self, base, token=None):
        self.base = base.rstrip("/")
        self.h = {"Content-Type": "application/fhir+json"}
        if token:
            self.h["Authorization"] = f"Bearer {token}"

    def _params(self, **kv):
        return {"resourceType": "Parameters",
                "parameter": [{"name": k, **v} for k, v in kv.items()]}

    def validate_code(self, system, code, display=None):
        body = self._params(url={"valueUri": system}, code={"valueCode": code},
                            **({"display": {"valueString": display}} if display else {}))
        out = requests.post(f"{self.base}/CodeSystem/$validate-code",
                            json=body, headers=self.h, timeout=15).json()
        params = {p["name"]: p for p in out.get("parameter", [])}
        return bool(params.get("result", {}).get("valueBoolean"))

# Ground an OpenMed span only if the code validates:
tx = TxClient("https://tx.example/fhir", token="...")
if tx.validate_code("http://snomed.info/sct", "44054006", "Diabetes mellitus type 2"):
    from openmed.clinical.exporters.codeable_concept_simple import coding, codeable_concept
    cc = codeable_concept([coding("snomed", "44054006",
                                  "Diabetes mellitus type 2")], text=span.text)
```

The system URIs here line up with OpenMed's `system_uri`
(`snomed`/`loinc`/`rxnorm`/`icd-10-cm`/`hpo`/`mesh`), so a validated code drops
straight into `coding(...)`.

## Hand-off to / from OpenMed

- **From OpenMed:** an `EntityPrediction.text` (the span surface form) plus your
  candidate code(s) are the input to `$validate-code`/`$translate`.
- **To OpenMed:** a *validated* (system, code, display) tuple →
  `coding(...)` → `codeable_concept(...)` (`exporting-to-fhir`). If a span fails
  validation, emit `CodeableConcept` with only `text` and flag it via
  `OperationOutcomeIssue(severity="warning", code="code-invalid", ...)`.
- **No PHI to the server.** You send *codes and concept text*, not patient
  notes. Never POST a clinical note or identifier to a terminology server.

## Edge cases & gotchas

- **Out-of-process by design.** OpenMed does not call the server for you; this
  thin client runs alongside, with the user's credentials. Keep it that way.
- **Licensing is the user's.** SNOMED CT / full LOINC / RxNorm require the right
  affiliate/license; `tx.fhir.org` only serves open content. Do not route
  licensed lookups through a public server.
- **`$expand` can be enormous.** Always pass `count` (paginate with `offset`)
  and `filter`; an unfiltered expand of a large hierarchy can time out.
- **ECL is SNOMED-specific.** Use it via the implicit value set
  `http://snomed.info/sct?fhir_vs=ecl/<expression>`; other systems use
  `$expand` with `filter`/`property`.
- **Cache validated codes.** The mapping from a normalised span to a validated
  code is stable; cache it to cut latency and server load — cache the *code*,
  never the source note.
- **`version` matters.** SNOMED/LOINC editions change; pin the `version`
  parameter for reproducible validation in CI.
- **No PHI to `tx.fhir.org`.** It is a public service — only synthetic/coded
  data.

## Standards & references

- FHIR terminology service: https://hl7.org/fhir/R4/terminology-service.html
- `$validate-code`: https://hl7.org/fhir/R4/valueset-operation-validate-code.html
- ValueSet `$expand`: https://hl7.org/fhir/R4/valueset-operation-expand.html
- CodeSystem `$lookup`: https://hl7.org/fhir/R4/codesystem-operation-lookup.html
- ConceptMap `$translate`: https://hl7.org/fhir/R4/conceptmap-operation-translate.html
- SNOMED ECL: https://confluence.ihtsdotools.org/display/DOCECL
- Ontoserver: https://ontoserver.csiro.au/
