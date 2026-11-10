---
name: coding-icd10
description: "Suggests candidate ICD-10-CM diagnosis codes (and ICD-10-PCS procedure codes) for diagnoses and procedures extracted by OpenMed, with rationale and a human-coder caveat. Use when the user wants to code a problem list, map a diagnosis span to a billable ICD-10-CM code, route a finding to the right chapter, cross-walk ICD-9 via GEMs, or pre-fill an encounter for coder review. Trigger keywords: ICD-10-CM, ICD-10-PCS, diagnosis coding, billable code, GEMs, problem list coding, encounter diagnosis, chapter range, CMS code lookup. references/icd10-chapters.md holds the chapter/section ranges. Pairs after OpenMed NER: consume Disease/Pathology entities from openmed.analyze_text and propose codes a certified coder validates. ICD-10-CM/PCS files are public domain from CMS — no license barrier (unlike CPT, which is restricted and out of scope)."
license: Apache-2.0
metadata:
  project: OpenMed
  category: terminology-coding
  pairs: after
  version: "1.0"
---

# Coding OpenMed diagnoses to ICD-10-CM / PCS

Suggest **ICD-10-CM** diagnosis codes (and **ICD-10-PCS** for inpatient
procedures) for the diagnosis and procedure spans OpenMed extracts. This is
**decision support for a certified coder**, not autonomous billing: OpenMed +
this skill narrow the candidate set and explain *why*; a human validates the
final, billable code.

ICD-10-CM and ICD-10-PCS are **public domain**. CMS publishes the complete
annual code files, addenda, and indexes for free. (CPT/HCPCS procedure codes are
**AMA-licensed and restricted** — out of scope here; obtain those separately
under the user's own AMA license.)

## When to use

- A note yields diagnoses ("type 2 diabetes with diabetic CKD", "community-
  acquired pneumonia") and you want candidate ICD-10-CM codes plus rationale.
- You need to **route** a span to the right chapter quickly
  (see `references/icd10-chapters.md` for code ranges).
- You hold legacy ICD-9 codes and need an approximate **GEM** cross-walk.
- You are pre-filling encounter diagnoses for a coder's review queue.

For clinical-meaning codes use `mapping-to-snomed`; for HCC/risk capture use
`coding-hcc-risk-adjustment`; this skill is for the ICD-10 classification.

## Quick start (public data + public FHIR lookup)

Two complementary paths, both license-clean:

**A) CMS files, loaded locally** (public domain; you download once):

```python
# CMS publishes the order/addenda file; load the code->description table.
# Columns: code (no dot), description; you insert the dot for display.
icd10cm = {}                       # "E1122" -> "Type 2 diabetes mellitus with diabetic chronic kidney disease"
with open("icd10cm_order_2025.txt", encoding="latin-1") as fh:
    for line in fh:
        code = line[6:13].strip()
        billable = line[14] == "1"     # '1' = valid billable code
        long_desc = line[77:].strip()
        if billable:
            icd10cm[code] = long_desc

def search_local(term: str, limit: int = 5):
    t = term.lower()
    hits = [(c, d) for c, d in icd10cm.items() if t in d.lower()]
    return sorted(hits, key=lambda cd: len(cd[1]))[:limit]
```

**B) A FHIR terminology server that hosts ICD-10-CM** (public servers exist;
e.g. an NLM Clinical Tables endpoint or your own HAPI/Ontoserver):

```python
import requests

# NLM Clinical Tables (public, no key) — ICD-10-CM autocomplete/search:
def search_icd10cm(term: str, count: int = 7):
    r = requests.get(
        "https://clinicaltables.nlm.nih.gov/api/icd10cm/v3/search",
        params={"sf": "code,name", "terms": term, "maxList": count}, timeout=10,
    )
    r.raise_for_status()
    _total, codes, _extra, display = r.json()
    return list(zip(codes, [d[1] for d in display]))   # [(code, name), ...]

print(search_icd10cm("type 2 diabetes nephropathy"))
```

## Workflow

1. **Extract** diagnosis/procedure spans with OpenMed (Disease/Pathology models).
2. **Route to a chapter** using the span's clinical theme and
   `references/icd10-chapters.md` (e.g. endocrine → E00–E89, circulatory →
   I00–I99). This shrinks the search space and catches obvious mis-hits.
3. **Search** the code text (local CMS table or the NLM API) for candidates.
4. **Apply ICD-10-CM specificity rules** in your rationale: laterality,
   acute/chronic, episode of care, "with"/"due to" combination codes, and
   "code first / use additional code" notes. Flag where the note lacks the detail
   a billable code requires.
5. **Rank** candidates; present the top few **with rationale and the missing-
   detail caveat**, not a single auto-selected code.
6. **Emit** `{system: "http://hl7.org/fhir/sid/icd-10-cm", code, display}`
   marked `status: needs-coder-review`, with OpenMed source offsets.

## Hand-off from OpenMed

`openmed.analyze_text(..., output_format="dict")` returns `entities`, each a dict
with `text`, `label`, `confidence`, `start`, `end`. Consume Disease/Pathology
spans:

```python
import openmed

note = "Assessment: type 2 diabetes with diabetic nephropathy; CAP."
result = openmed.analyze_text(
    note,
    model_name="disease_detection_superclinical",   # Disease category
    output_format="dict",
)

DX_LABELS = {"DISEASE", "CONDITION", "PATHOLOGY"}
for ent in result["entities"]:
    if ent["label"] in DX_LABELS:
        candidates = search_icd10cm(ent["text"], count=5)
        print(ent["text"], ent["start"], ent["end"],
              f"(conf {ent['confidence']:.2f}) ->", candidates)
        # surface as SUGGESTIONS for a coder — never auto-bill
```

Keep OpenMed's `start`/`end` offsets next to each suggested code so the coder can
jump to the exact supporting text. Store offsets and codes only — never the raw
note in your suggestion log.

## Edge cases & gotchas

- **Human-in-the-loop is mandatory.** ICD-10-CM coding has legal/financial
  weight. Output candidates with rationale; a certified coder assigns the final
  billable code. Never present a suggestion as an authorized claim.
- **Specificity & unspecified codes.** Many billable codes demand laterality,
  episode, or "with" detail the note may not state. Prefer flagging "documentation
  insufficient for a specific code" over forcing an `.9`/unspecified code.
- **Combination codes.** ICD-10-CM bundles related conditions (e.g. E11.22 =
  diabetes *with* diabetic CKD). Don't emit two separate codes where one
  combination code is required; let the search surface combinations.
- **"Code first" / "use additional code" / Excludes1/Excludes2** sequencing notes
  change which codes coexist. Carry these as rationale for the coder.
- **GEMs are approximate.** ICD-9↔ICD-10 General Equivalence Mappings are
  many-to-many and lossy; treat a GEM result as a starting hint, not a billable
  mapping.
- **Annual updates.** Codes change every fiscal year (Oct 1). Pin the file year
  you loaded and refresh annually; record which version produced a suggestion.
- **Licensing.** ICD-10-CM/PCS are public domain (CMS). Do **not** pull in CPT or
  proprietary code maps that require an AMA/other license — those stay
  user-supplied and out-of-process.
- **Local-first.** OpenMed NER runs on-device; if you query the NLM API, send only
  the de-identified diagnosis string. No PHI over the wire.

## Standards & references

- ICD-10-CM files (CMS, public domain):
  https://www.cms.gov/medicare/coding-billing/icd-10-codes/icd-10-cm-cms-hcc
  and https://www.cms.gov/medicare/coding-billing/icd-10-codes
- ICD-10-PCS files (CMS):
  https://www.cms.gov/medicare/coding-billing/icd-10-codes/icd-10-pcs
- NLM Clinical Tables ICD-10-CM API (public, no key):
  https://clinicaltables.nlm.nih.gov/apidoc/icd10cm/v3/doc.html
- GEMs (General Equivalence Mappings):
  https://www.cms.gov/medicare/coding-billing/icd-10-codes/icd-10-cm-pcs-gems-archive
- Chapter/section ranges: `references/icd10-chapters.md`
