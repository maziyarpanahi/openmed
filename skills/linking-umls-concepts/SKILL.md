---
name: linking-umls-concepts
description: "Links entities extracted by OpenMed to UMLS Metathesaurus CUIs using the USER'S OWN UTS API key, with nothing from the Metathesaurus bundled or cached. Use when the user wants to normalize concepts across vocabularies to a single CUI, resolve synonyms via the UMLS, filter by semantic type, or cross-walk between SNOMED CT, ICD-10, RxNorm and MeSH through their shared CUI. Trigger keywords: UMLS, CUI, Metathesaurus, UTS API key, semantic type, TUI, MetaMap, QuickUMLS, concept normalization, cross-vocabulary. Pairs after OpenMed NER: consume Disease/Pharmaceutical/Chemical/Anatomy entities from openmed.analyze_text and resolve each span to a CUI out-of-process. UMLS is license-restricted — the Metathesaurus is NEVER bundled; every call uses the user's UTS account."
license: Apache-2.0
metadata:
  project: OpenMed
  category: terminology-coding
  pairs: after
  version: "1.0"
---

# Linking OpenMed entities to UMLS CUIs

Resolve concept spans that OpenMed extracts to **UMLS Metathesaurus** concepts.
The atom is the **CUI** (Concept Unique Identifier, e.g. `C0011860`): one CUI
unifies synonyms from many source vocabularies (SNOMED CT, ICD-10-CM, RxNorm,
MeSH, LOINC), making the CUI the natural hub for cross-vocabulary normalization.
Every concept also carries one or more **semantic types** (TUIs, e.g. *Disease or
Syndrome* `T047`) for type-based filtering.

> **Hard licensing boundary — read first.** The UMLS Metathesaurus is
> **license-restricted**. OpenMed and this skill **never bundle, ship, or cache**
> Metathesaurus content. Concept linking runs **out-of-process against the NLM
> UTS (UMLS Terminology Services) REST API using the user's own UTS API key**.
> A free UTS account + API key is required (request at uts.nlm.nih.gov and accept
> the UMLS license). The Metathesaurus stays user-supplied: your code holds only
> the key (from the environment) and stores only returned CUIs/strings.

## When to use

- You need **one canonical id across vocabularies** — e.g. to unify a SNOMED CT
  disorder, an ICD-10 code, and a free-text mention onto a single CUI.
- You want **synonym normalization** ("MI", "myocardial infarction", "heart
  attack" → `C0027051`).
- You need **semantic-type filtering** to keep only, say, *Pharmacologic
  Substance* or *Disease or Syndrome* entities.
- You are cross-walking codes and need the CUI as the join key before pivoting to
  RxNorm (`normalizing-rxnorm`) or SNOMED (`mapping-to-snomed`).

## Quick start (user-supplied UTS API key)

The UTS REST API base is `https://uts-ws.nlm.nih.gov/rest`. Authentication uses
your API key as the `apiKey` query parameter (the modern, simplest method).

```python
import os, requests

UTS = "https://uts-ws.nlm.nih.gov/rest"
API_KEY = os.environ["UTS_API_KEY"]          # USER's own key — never hardcoded
VERSION = "current"                           # or a fixed release like 2024AB

def search(term: str, sabs: str | None = None, count: int = 10) -> list[dict]:
    """Search the Metathesaurus for a term; optionally restrict source vocabs."""
    params = {"string": term, "apiKey": API_KEY, "pageSize": count}
    if sabs:                                   # e.g. "SNOMEDCT_US,RXNORM,ICD10CM"
        params["sabs"] = sabs
    r = requests.get(f"{UTS}/search/{VERSION}", params=params, timeout=15)
    r.raise_for_status()
    return r.json().get("result", {}).get("results", [])

def concept(cui: str) -> dict:
    """Pull a concept's preferred name and semantic types."""
    r = requests.get(f"{UTS}/content/{VERSION}/CUI/{cui}",
                     params={"apiKey": API_KEY}, timeout=15)
    r.raise_for_status()
    return r.json().get("result", {})

def crosswalk(cui: str, target_sab: str) -> list[dict]:
    """Atoms of a CUI in a target vocabulary (the cross-walk)."""
    r = requests.get(f"{UTS}/content/{VERSION}/CUI/{cui}/atoms",
                     params={"apiKey": API_KEY, "sabs": target_sab,
                             "pageSize": 50}, timeout=20)
    r.raise_for_status()
    return r.json().get("result", [])

hits = search("type 2 diabetes")              # -> [{ui: 'C0011860', name: ...}, ...]
sct = crosswalk("C0011860", "SNOMEDCT_US")    # CUI -> SNOMED CT codes
```

## Workflow

1. **Extract** spans with OpenMed (Disease, Pharmaceutical, Chemical, Anatomy).
2. **Search** each span via `/search/{version}` for candidate CUIs.
3. **Filter by semantic type (TUI)** so a drug span resolves to a substance
   concept, not a same-named disease. Pull semantic types from
   `/content/.../CUI/{cui}` and keep only the expected group.
4. **Rank** candidates (exact preferred-name match > synonym match) and combine
   with OpenMed's `confidence` to choose one CUI.
5. **Cross-walk** the chosen CUI to whatever target you actually store —
   SNOMEDCT_US, ICD10CM, RXNORM, MSH — via `/CUI/{cui}/atoms?sabs=`.
6. **Emit** the CUI plus the target code(s) and OpenMed source offsets.

## Hand-off from OpenMed

`openmed.analyze_text(..., output_format="dict")` returns `entities`, each a dict
with `text`, `label`, `confidence`, `start`, `end`. Use the label to pick the
semantic-type group you keep:

```python
import openmed

note = "History of myocardial infarction; started on lisinopril."
result = openmed.analyze_text(
    note,
    model_name="disease_detection_superclinical",   # Disease category
    output_format="dict",
)

# OpenMed label -> acceptable UMLS semantic-type groups (TUI prefixes)
KEEP_STY = {
    "DISEASE":  {"Disease or Syndrome", "Sign or Symptom", "Neoplastic Process"},
    "DRUG":     {"Pharmacologic Substance", "Clinical Drug"},
    "CHEM":     {"Pharmacologic Substance", "Organic Chemical"},
}

for ent in result["entities"]:
    for hit in search(ent["text"], count=5):
        cui = hit["ui"]
        stys = {s["name"] for s in concept(cui).get("semanticTypes", [])}
        if not KEEP_STY.get(ent["label"]) or stys & KEEP_STY[ent["label"]]:
            print(ent["text"], ent["start"], ent["end"], "->", cui, hit["name"])
            break
```

Keep OpenMed's `start`/`end` offsets beside each CUI for traceability. Store only
CUIs and codes — never the raw note, never a local copy of the Metathesaurus.

## Edge cases & gotchas

- **Never bundle or cache the Metathesaurus.** No vendored MRCONSO, no local
  concept dump baked into the package. If you precompute, do it inside the
  *user's* licensed environment, not in distributed OpenMed assets.
- **The UTS key is the user's.** Read it from the environment/secret store; never
  embed it, log it, or commit it. One key, the user's license, their rate limits.
- **Semantic-type filtering is essential.** Many strings are polysemous across
  types ("cold" = symptom vs temperature). Without TUI filtering you will link to
  the wrong concept family.
- **Version pin for reproducibility.** `current` drifts at each UMLS release. Pin
  a release (e.g. `2024AB`) for stable, auditable mappings; record it.
- **Source-vocab restriction.** Restrict `sabs` to the vocabularies you are
  licensed for and actually need; this both narrows results and respects per-source
  license terms inside UMLS.
- **CUI as hub, not endpoint.** Downstream systems usually want a target code
  (SNOMED/ICD/RxNorm), so resolve to CUI then cross-walk — don't store only the
  CUI if your consumers expect billable/clinical codes.
- **Offline alternatives are still user-licensed.** Tools like MetaMap or
  QuickUMLS run locally but require a UMLS download under the user's license;
  OpenMed neither ships nor requires those datasets.
- **Local-first.** OpenMed NER runs on-device; only de-identified concept strings
  reach UTS. No PHI over the wire.

## Standards & references

- UMLS Metathesaurus: https://www.nlm.nih.gov/research/umls/index.html
- UMLS license & UTS account: https://uts.nlm.nih.gov/uts/
- UTS REST API docs: https://documentation.uts.nlm.nih.gov/rest/home.html
- Semantic Network (types/TUIs): https://www.nlm.nih.gov/research/umls/META3_current_semantic_types.html
- MetaMap: https://lhncbc.nlm.nih.gov/ii/tools/MetaMap.html
- QuickUMLS: https://github.com/Georgetown-IR-Lab/QuickUMLS
