---
name: mapping-loinc
description: "Maps laboratory and clinical observation names extracted by OpenMed to LOINC codes using the public Regenstrief LOINC and FHIR terminology APIs. Use when the user wants to code lab tests, vital signs, or observations to LOINC, resolve a test name plus specimen and method to the correct LOINC part-model code, attach UCUM units, or build a US Core Laboratory Result Observation. Trigger keywords: LOINC, lab coding, observation code, UCUM units, specimen, method, US Core lab, FHIR Observation, lab result mapping, panel vs analyte. Pairs after OpenMed NER: consume Disease/Chemical/lab-name entities from openmed.analyze_text and map each measurement to a LOINC code. LOINC is free to use under the Regenstrief license (registration/terms-of-use, no fee); UMLS/SNOMED stay user-supplied and out-of-process."
license: Apache-2.0
metadata:
  project: OpenMed
  category: terminology-coding
  pairs: after
  version: "1.0"
---

# Mapping lab/observation names to LOINC

Ground free-text lab and observation names that OpenMed surfaces to **LOINC**
(Logical Observation Identifiers Names and Codes), the universal standard for
identifying *what was measured*. A LOINC code is a fully specified observation —
not just an analyte but the full six-axis model: **Component, Property, Time,
System (specimen), Scale, Method**.

LOINC is **free** to use. It is published by the Regenstrief Institute under the
LOINC license: you accept terms-of-use (and register to download the table), but
there is no fee and no per-use restriction. The standard, public path for
license-clean mapping is a **FHIR terminology server** exposing LOINC via
`$lookup` / `$validate-code`, or Regenstrief's hosted **fhir.loinc.org**.

## When to use

- A note or report contains lab/observation names ("serum potassium",
  "hemoglobin A1c", "blood pressure") and you need a stable LOINC code each.
- You must disambiguate by **specimen/system** ("glucose in serum" vs
  "glucose in urine") or **method** ("HbA1c by HPLC").
- You need **UCUM** units to pair with the result value, or a US Core
  `Observation` (Laboratory Result) coded with LOINC.
- You are mapping a **panel** (e.g. CBC, BMP) vs its individual **analytes**.

For diagnoses/procedures use `coding-icd10`; for drugs use `normalizing-rxnorm`;
LOINC is for *observations and measurements*.

## Quick start (real LOINC / FHIR terminology calls)

Regenstrief hosts a public FHIR terminology endpoint at `https://fhir.loinc.org`
(HTTP Basic auth with your free LOINC account). Many sites instead point at their
own server (HAPI, Ontoserver, Snowstorm-with-LOINC). The operations are the same.

```python
import requests
from requests.auth import HTTPBasicAuth

FHIR = "https://fhir.loinc.org"
AUTH = HTTPBasicAuth("YOUR_LOINC_USER", "YOUR_LOINC_PASSWORD")  # free account
LOINC_SYSTEM = "http://loinc.org"

def lookup(code: str) -> dict:
    """$lookup: return the fully specified name + axes for a LOINC code."""
    r = requests.get(
        f"{FHIR}/CodeSystem/$lookup",
        params={"system": LOINC_SYSTEM, "code": code},
        auth=AUTH, headers={"Accept": "application/fhir+json"}, timeout=15,
    )
    r.raise_for_status()
    return r.json()

def validate(code: str, display: str) -> bool:
    r = requests.get(
        f"{FHIR}/CodeSystem/$validate-code",
        params={"url": LOINC_SYSTEM, "code": code, "display": display},
        auth=AUTH, headers={"Accept": "application/fhir+json"}, timeout=15,
    )
    r.raise_for_status()
    params = {p["name"]: p.get("valueBoolean") for p in r.json().get("parameter", [])}
    return bool(params.get("result"))

print(lookup("2823-3"))     # Potassium [Moles/volume] in Serum or Plasma
```

Search candidate LOINC codes from a text name with the Regenstrief search API
(`https://loinc.org/search/`) or a `ValueSet/$expand` filter on your server:

```python
def expand_filter(text: str, count: int = 10) -> list[dict]:
    """Text-filter the LOINC code system to candidate concepts."""
    r = requests.get(
        f"{FHIR}/ValueSet/$expand",
        params={"url": "http://loinc.org/vs", "filter": text, "count": count},
        auth=AUTH, headers={"Accept": "application/fhir+json"}, timeout=20,
    )
    r.raise_for_status()
    return r.json().get("expansion", {}).get("contains", [])
```

## Workflow

1. **Extract** observation/analyte mentions with OpenMed.
2. **Assemble the axes you have** from surrounding text: component (what),
   specimen/system (serum, urine, blood), method (HPLC, immunoassay), and scale
   (quantitative vs ordinal). More axes → a more specific, correct LOINC.
3. **Search** candidates via `$expand?filter=` (or Regenstrief search).
4. **Disambiguate** by matching specimen and property. "Glucose" alone is
   ambiguous; "glucose, serum, mass/volume" resolves to one code.
5. **Validate** the chosen code with `$validate-code`, then `$lookup` to pull the
   long common name and the canonical **UCUM** example unit.
6. **Emit** `{system: "http://loinc.org", code, display}` plus the UCUM unit for
   the result value, into a US Core Observation.

## Hand-off from OpenMed

`openmed.analyze_text(..., output_format="dict")` returns `entities`, each a dict
with `text`, `label`, `confidence`, `start`, `end`. Lab analytes often surface
under Chemical/Disease models; run the relevant model and feed the spans in:

```python
import openmed

note = "Labs: serum potassium 5.1 mmol/L, hemoglobin A1c 7.8 %."
result = openmed.analyze_text(
    note,
    model_name="chemical_detection_pubmed",   # Chemical category (analytes)
    output_format="dict",
)

for ent in result["entities"]:
    name = ent["text"]                         # e.g. "potassium"
    candidates = expand_filter(name, count=5)  # LOINC candidates
    # carry OpenMed offsets so the code is traceable to the source span
    print(name, ent["start"], ent["end"], "->",
          [(c["code"], c["display"]) for c in candidates[:3]])
```

Pair the matched LOINC with the *value and unit* you parse from the same line —
LOINC names the test, UCUM names the unit, the value stays in the Observation.
Keep only offsets and codes in your mapping table; never persist raw report text.

## Edge cases & gotchas

- **Specimen ambiguity is the #1 error.** Always resolve System/specimen before
  choosing a code. Defaulting to "Serum or Plasma" when the note says urine
  produces a wrong but plausible LOINC.
- **Panel vs analyte.** "CBC" is an order/panel LOINC; the individual results
  (WBC, Hgb, Plt) are separate analyte LOINCs. Map at the granularity your data
  is recorded at.
- **Method matters for some assays** (e.g. HbA1c, troponin generations). If the
  method is documented, pick the method-specific code; otherwise use the
  method-less "any method" code rather than guessing.
- **UCUM, not free text, for units.** Convert "mg/dL" to the UCUM string
  `mg/dL`; reject units LOINC's example unit cannot reconcile with.
- **Licensing (free, with terms).** LOINC is free but Regenstrief-licensed:
  accept the LOINC terms-of-use and register for a (free) account to call
  `fhir.loinc.org` or download the table. Do **not** obtain LOINC by bundling
  UMLS or SNOMED — those carry separate restricted licenses and must stay
  user-supplied and out-of-process (see `mapping-to-snomed`, `linking-umls-concepts`).
- **Local-first.** OpenMed NER runs on-device; only the de-identified analyte
  string should reach the terminology server. No PHI over the wire.

## Standards & references

- LOINC home & license: https://loinc.org/ and https://loinc.org/license/
- LOINC FHIR terminology service: https://loinc.org/fhir/
- FHIR `$lookup` / `$validate-code`: https://hl7.org/fhir/codesystem-operation-lookup.html
- UCUM units of measure: https://ucum.org/
- US Core Laboratory Result Observation:
  https://hl7.org/fhir/us/core/StructureDefinition-us-core-observation-lab.html
