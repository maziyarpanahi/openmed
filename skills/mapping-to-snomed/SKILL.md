---
name: mapping-to-snomed
description: "Maps clinical concept spans extracted by OpenMed to SNOMED CT concepts through a USER-SUPPLIED terminology server (the user's own Ontoserver, Snowstorm, or UMLS/UTS), never a bundled vocabulary. Use when the user wants to code findings, disorders, procedures, body structures, or substances to SNOMED CT, run an ECL query, translate via a ConceptMap, or resolve a span to a concept id with FHIR $lookup/$translate/$validate-code. Trigger keywords: SNOMED CT, SNOMED concept id, ECL, ConceptMap, $translate, $lookup, Ontoserver, Snowstorm, SCTID, post-coordination, terminology server. Pairs after OpenMed NER: consume Disease/Anatomy/Pharmaceutical entities from openmed.analyze_text and map each span out-of-process. SNOMED CT is license-restricted — it is NEVER bundled; the user calls their own affiliate-licensed server."
license: Apache-2.0
metadata:
  project: OpenMed
  category: terminology-coding
  pairs: after
  version: "1.0"
---

# Mapping OpenMed spans to SNOMED CT

Ground clinical concept spans that OpenMed extracts — disorders, findings,
procedures, body structures, substances — to **SNOMED CT**, the comprehensive
clinical reference terminology. The atom is the **SCTID** (a SNOMED CT concept
identifier), organized into a description-logic hierarchy you can query with
**ECL** (Expression Constraint Language).

> **Hard licensing boundary — read first.** SNOMED CT is **license-restricted**.
> OpenMed and this skill **never bundle, ship, cache, or redistribute** any
> SNOMED CT content. All mapping happens **out-of-process against a terminology
> server the user supplies and is licensed for** — their own **Ontoserver**,
> **Snowstorm**, the NLM's **UTS/UMLS** FHIR endpoint, or a national release
> server. SNOMED International requires an Affiliate License (free in member
> territories like the US via the NLM; check your country). Your code receives a
> **base URL + credentials from the user**; it must work with *any* compliant
> FHIR terminology server and store nothing but the returned codes.

## When to use

- You need rich, hierarchy-aware clinical codes (more granular than ICD-10) for
  problems, procedures, or body sites.
- You want to **translate** an existing code (ICD-10-CM, local code) to SNOMED CT
  via a `ConceptMap`/`$translate`.
- You need subsumption/ECL queries ("is this a descendant of *Diabetes
  mellitus*?") for cohorting or decision support.

For billing codes use `coding-icd10`; for drugs `normalizing-rxnorm`; for labs
`mapping-loinc`. SNOMED CT is the clinical-meaning layer.

## Quick start (user-supplied FHIR terminology server)

Configuration is injected, never hardcoded. The operations are standard FHIR R4.

```python
import os, requests

# Provided by the USER — their licensed server. Nothing bundled.
TX = os.environ["FHIR_TX_URL"]              # e.g. https://snowstorm.example.org/fhir
TOKEN = os.environ.get("FHIR_TX_TOKEN")     # if the server requires auth
SNOMED = "http://snomed.info/sct"
HDRS = {"Accept": "application/fhir+json"}
if TOKEN:
    HDRS["Authorization"] = f"Bearer {TOKEN}"

def lookup(code: str) -> dict:
    """$lookup: fully specified name + properties for an SCTID."""
    r = requests.get(f"{TX}/CodeSystem/$lookup",
                     params={"system": SNOMED, "code": code},
                     headers=HDRS, timeout=15)
    r.raise_for_status()
    return r.json()

def find_concepts(text: str, ecl: str = "<<404684003", count: int = 10):
    """Text search constrained by ECL (default: descendants of Clinical finding)."""
    vs = f"{SNOMED}?fhir_vs=ecl/{ecl}"
    r = requests.get(f"{TX}/ValueSet/$expand",
                     params={"url": vs, "filter": text, "count": count},
                     headers=HDRS, timeout=20)
    r.raise_for_status()
    return r.json().get("expansion", {}).get("contains", [])

def translate(code: str, source_system: str, conceptmap_url: str):
    """$translate an existing code to SNOMED CT via a ConceptMap."""
    r = requests.get(f"{TX}/ConceptMap/$translate",
                     params={"url": conceptmap_url, "system": source_system,
                             "code": code, "targetsystem": SNOMED},
                     headers=HDRS, timeout=20)
    r.raise_for_status()
    return r.json()

# ECL examples: 64572001=disease, 71388002=procedure, 123037004=body structure
print(find_concepts("type 2 diabetes", ecl="<<64572001"))
```

## Workflow

1. **Extract** spans with OpenMed (Disease, Anatomy, Pharmaceutical models).
2. **Pick a semantic constraint (ECL)** from the OpenMed label so you search the
   right hierarchy: disorder span → `<<64572001`; anatomy span → `<<123037004`;
   substance/drug → `<<105590001`; procedure → `<<71388002`.
3. **Search** with `ValueSet/$expand?filter=<span>` under that ECL.
4. **Rank & disambiguate** by display match and confidence; prefer the most
   specific concept whose meaning is fully entailed by the text (do not over-code).
5. **Validate** with `$validate-code`; `$lookup` to capture the FSN and any
   needed properties.
6. **Translate** instead of searching when you already hold an ICD-10/local code
   and the user's server has the relevant `ConceptMap`.
7. **Emit** `{system: "http://snomed.info/sct", code, display}` — the SCTID plus
   the OpenMed source offsets for traceability.

## Hand-off from OpenMed

`openmed.analyze_text(..., output_format="dict")` returns `entities`, each a dict
with `text`, `label`, `confidence`, `start`, `end`. Route each label to an ECL
hierarchy and map out-of-process:

```python
import openmed

note = "Assessment: type 2 diabetes mellitus with diabetic nephropathy."
result = openmed.analyze_text(
    note,
    model_name="disease_detection_superclinical",   # Disease category
    output_format="dict",
)

ECL_FOR_LABEL = {
    "DISEASE":   "<<64572001",     # | Disease |
    "CONDITION": "<<64572001",
    "PATHOLOGY": "<<64572001",
    "ANATOMY":   "<<123037004",    # | Body structure |
    "ORGAN":     "<<123037004",
}

for ent in result["entities"]:
    ecl = ECL_FOR_LABEL.get(ent["label"], "<<404684003")  # fallback: Clinical finding
    candidates = find_concepts(ent["text"], ecl=ecl, count=5)
    print(ent["text"], ent["start"], ent["end"], "->",
          [(c["code"], c["display"]) for c in candidates[:3]])
```

Carry OpenMed's `start`/`end` offsets next to each SCTID so every code is
auditable back to its span. Persist codes and offsets only — never the raw note,
and never a local copy of SNOMED content.

## Edge cases & gotchas

- **Never bundle SNOMED CT.** Do not vendor a release, embed an export, or cache
  descriptions to disk for reuse. If you find yourself shipping SNOMED data, stop
  — the design must call the user's licensed server live, out-of-process.
- **Affiliate licensing.** Confirm the user holds (or their territory grants) a
  SNOMED International Affiliate License. In the US it is free via the NLM/UMLS;
  elsewhere it varies. Surface this requirement; do not assume entitlement.
- **Pre- vs post-coordination.** Some clinical meanings need a post-coordinated
  expression (e.g. finding + body site + severity). Prefer a single
  pre-coordinated concept when one exists; only post-coordinate when your server
  and downstream systems support SNOMED CT expressions.
- **Edition/version drift.** SCTIDs are stable but content differs across
  editions (International vs US vs UK) and monthly releases. Record the edition
  the server reports; do not mix codes across editions silently.
- **Negation/uncertainty stays in OpenMed.** A span "no evidence of pneumonia"
  must not be coded as present pneumonia. Resolve assertion/negation with
  OpenMed's clinical-context layer *before* mapping.
- **Don't over-specify.** Map to the concept actually supported by the text;
  inventing severity or laterality the note never stated is a coding error.
- **Local-first.** OpenMed NER runs on-device; only de-identified concept
  strings reach the terminology server. No PHI over the wire.

## Standards & references

- SNOMED CT (SNOMED International): https://www.snomed.org/
- SNOMED CT licensing & Affiliate program: https://www.snomed.org/get-snomed
- NLM SNOMED CT (US, via UMLS/UTS): https://www.nlm.nih.gov/healthit/snomedct/index.html
- Expression Constraint Language (ECL): https://confluence.ihtsdotools.org/display/DOCECL
- FHIR `$translate` / `$lookup` / `$validate-code`:
  https://hl7.org/fhir/terminology-service.html
- Snowstorm (reference terminology server): https://github.com/IHTSDO/snowstorm
- Ontoserver: https://ontoserver.csiro.au/
