---
name: normalizing-rxnorm
description: "Normalizes drug mentions extracted by OpenMed to RxNorm RxCUIs using the free public RxNav/RxNorm REST API. Use when the user wants to code, standardize, or de-duplicate medication names, resolve a brand/generic/ingredient to a stable RxCUI, link strength+dose-form to an SCD/SBD, attach NDCs, or build a US Core Medication resource. Trigger keywords: RxNorm, RxCUI, RxNav, drug normalization, medication coding, NDC, ingredient, SCD, SBD, brand vs generic, getApproximateMatch. Pairs after OpenMed NER: consume Pharmaceutical/Chemical entities from openmed.analyze_text and map each drug span to an RxCUI. RxNorm and RxNav are fully public and free — no API key, no license barrier, the lowest-friction terminology in this set."
license: Apache-2.0
metadata:
  project: OpenMed
  category: terminology-coding
  pairs: after
  version: "1.0"
---

# Normalizing drug mentions to RxNorm

Map free-text medication mentions that OpenMed extracts to **RxNorm** — the U.S.
National Library of Medicine's normalized drug nomenclature. The unit of meaning
is the **RxCUI** (RxNorm Concept Unique Identifier): a stable integer that ties
together brand, generic, ingredient, strength, and dose form.

RxNorm and the **RxNav REST API** are **fully public and free**: no API key, no
license agreement, no rate-limit registration for normal use. Of every skill in
this terminology batch, this one has the highest value-to-friction ratio — start
here when grounding medications.

## When to use

- A clinical note names drugs ("metformin 500 mg", "Lipitor", "amox/clav") and
  you need one stable code per drug for storage, analytics, or interoperability.
- You must distinguish **ingredient** ("metformin", `IN`) from a prescribable
  product — **SCD** (Semantic Clinical Drug, generic) or **SBD** (Semantic Brand
  Drug) — e.g. "metformin 500 MG Oral Tablet".
- You need to de-duplicate brand/generic synonyms onto one concept.
- You need **NDC** codes (package-level) for a product, or a US Core
  `Medication`/`MedicationRequest` coded with RxNorm.

If the source text is non-English or you need ATC/SNOMED links instead, see
`mapping-to-snomed`; RxNorm itself is U.S.-centric.

## Quick start (real RxNav API calls)

Base URL: `https://rxnav.nlm.nih.gov/REST`. No auth. JSON via `?...&...` paths
ending in nothing or `.json` depending on endpoint; the REST root returns XML by
default, so request JSON explicitly.

```python
import requests

BASE = "https://rxnav.nlm.nih.gov/REST"

def rxcui_for(name: str) -> str | None:
    """Exact-match RxCUI lookup for a normalized drug name."""
    r = requests.get(f"{BASE}/rxcui.json", params={"name": name}, timeout=10)
    r.raise_for_status()
    ids = r.json().get("idGroup", {}).get("rxnormId", [])
    return ids[0] if ids else None

def approximate(name: str, max_entries: int = 3) -> list[dict]:
    """Fuzzy match for misspelled or abbreviated drug text."""
    r = requests.get(
        f"{BASE}/approximateTerm.json",
        params={"term": name, "maxEntries": max_entries},
        timeout=10,
    )
    r.raise_for_status()
    return r.json().get("approximateGroup", {}).get("candidate", [])

print(rxcui_for("metformin"))                       # -> '6809' (ingredient)
print(approximate("metformin 500"))                 # fuzzy -> candidate RxCUIs
```

Resolve a full prescribable product (ingredient + strength + form) to an SCD:

```python
# getApproximateMatch / getRxConceptProperties give term type (TTY)
def properties(rxcui: str) -> dict:
    r = requests.get(f"{BASE}/rxcui/{rxcui}/properties.json", timeout=10)
    r.raise_for_status()
    return r.json().get("properties", {})

# Find the SCD ("metformin 500 MG Oral Tablet") from the ingredient:
def related_by_tty(rxcui: str, tty: str) -> list[dict]:
    r = requests.get(
        f"{BASE}/rxcui/{rxcui}/related.json", params={"tty": tty}, timeout=10
    )
    r.raise_for_status()
    groups = r.json().get("relatedGroup", {}).get("conceptGroup", [])
    out = []
    for g in groups:
        out.extend(g.get("conceptProperties", []) or [])
    return out
```

Attach NDCs and check interactions (both public):

```python
ndcs = requests.get(f"{BASE}/rxcui/{rxcui}/ndcs.json").json()   # package codes
```

## Workflow

1. **Extract** drug spans with OpenMed (`pharma_detection_superclinical`).
2. **Parse** each span into name + strength + dose form when present
   ("metformin 500 mg tablet" → ingredient `metformin`, strength `500 MG`,
   form `Oral Tablet`).
3. **Exact match** the cleaned name with `/rxcui.json?name=`. If empty, fall back
   to `/approximateTerm.json`.
4. **Pick the right term type (TTY)** for your use case:
   - `IN` ingredient — analytics, allergy lists, class rollups.
   - `SCD` generic product / `SBD` brand product — orders, US Core Medication.
   - `BN` brand name, `PIN` precise ingredient — display/lineage.
5. **Validate** by reading `/rxcui/{rxcui}/properties.json` and confirming the
   `tty` and `name` match expectations; record the `score` from approximate
   matches as a confidence signal.
6. **Emit** `{system: "http://www.nlm.nih.gov/research/umls/rxnorm", code, display}`.

## Hand-off from OpenMed

OpenMed's `analyze_text` returns a `dict` whose `entities` list contains, per
span, the keys `text`, `label`, `confidence`, `start`, `end`. Consume the
Pharmaceutical/Chemical entities directly:

```python
import openmed, requests

note = "Patient on metformin 500 mg BID and atorvastatin 20 mg nightly."
result = openmed.analyze_text(
    note,
    model_name="pharma_detection_superclinical",   # Pharmaceutical category
    output_format="dict",
)

DRUG_LABELS = {"DRUG", "MEDICATION", "CHEM"}        # OpenMed Pharmaceutical labels
for ent in result["entities"]:
    if ent["label"] in DRUG_LABELS:
        span = ent["text"]                          # e.g. "metformin"
        rxcui = rxcui_for(span) or (
            (approximate(span) or [{}])[0].get("rxcui")
        )
        print(span, "->", rxcui, f"(conf {ent['confidence']:.2f})")
```

Keep OpenMed's character offsets (`start`/`end`) alongside the RxCUI so every
code is traceable back to the exact source span — never store the raw note text
in your mapping table.

## Edge cases & gotchas

- **Strength/form live in separate spans.** OpenMed labels the drug name; the
  "500 mg" and "tablet" may be adjacent tokens. Reassemble using offsets before
  querying for an SCD, or you will only get the ingredient.
- **Combination products** ("amoxicillin/clavulanate") normalize to a single
  multi-ingredient SCD; do not split them into two RxCUIs.
- **Brand vs generic.** `Lipitor` (SBD/BN) and `atorvastatin` (IN/SCD) are
  different RxCUIs of the same drug. Decide up front which TTY your pipeline
  stores and map the other via `/related.json`.
- **Approximate-match noise.** `approximateTerm` will happily return a candidate
  for garbage input. Gate on the returned `score` and re-validate with
  `/properties.json` before trusting it.
- **Obsolete RxCUIs.** Use `/rxcui/{rxcui}/historystatus.json` to detect
  retired/remapped concepts; follow the remap rather than storing a dead code.
- **Licensing: none for RxNorm/RxNav.** RxNorm is public domain. *But* RxNorm
  includes source vocabularies (e.g. some proprietary drug data) whose own
  terms-of-use apply if you redistribute the full dataset — calling the live API
  for normalization is unrestricted. Do not bundle UMLS to get RxNorm; RxNav is
  the clean path.
- **Local-first stays intact.** Run OpenMed NER on-device; only the
  *de-identified* drug string leaves the process to hit RxNav. Never send a raw
  note containing PHI to the API.

## Standards & references

- RxNav REST API: https://rxnav.nlm.nih.gov/RxNormAPIs.html
- RxNorm overview & files: https://www.nlm.nih.gov/research/umls/rxnorm/index.html
- RxNorm term types (TTY): https://www.nlm.nih.gov/research/umls/rxnorm/docs/appendix5.html
- RxNav interaction/NDC APIs: https://rxnav.nlm.nih.gov/
- US Core Medication: https://hl7.org/fhir/us/core/StructureDefinition-us-core-medication.html
