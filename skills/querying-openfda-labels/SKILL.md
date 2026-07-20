---
name: querying-openfda-labels
description: "Looks up FDA drug labels, NDC directory entries, indications, boxed warnings, and recalls/enforcement actions via the free public OpenFDA API to enrich drugs that OpenMed extracts. Use when the user wants the prescribing information for a drug, its boxed warning, approved indications, dosage forms and routes, package NDC codes, RxCUI, or whether a product has an open recall. Trigger keywords: OpenFDA, drug label, SPL, prescribing information, boxed warning, black box warning, indications, NDC, package code, recall, enforcement, Class I recall, drug enrichment. Pairs adjacent to OpenMed NER: take a drug name (or RxNorm RxCUI) from openmed.analyze_text and resolve its label, NDC, and recall status. OpenFDA is public and free — no license barrier; send only de-identified drug names, never raw clinical notes."
license: Apache-2.0
metadata:
  project: OpenMed
  category: safety-pharmacovigilance
  pairs: adjacent
  version: "1.0"
---

# Querying OpenFDA drug labels, NDC, and recalls

Once OpenMed has pulled a drug name out of a note, you often need authoritative
product facts: the **boxed warning**, approved **indications**, **dosage form /
route**, package **NDC** codes, and whether the product is under **recall**. The
FDA's **OpenFDA** API exposes the Structured Product Labeling (SPL), the NDC
directory, and enforcement (recall) reports — all **public and free**.

This skill is enrichment: it attaches regulatory facts to an extracted drug. It
is **not** clinical decision support — a label lookup informs a human, it does
not prescribe.

## When to use

- You extracted a drug and need its **boxed warning** or **indications** for
  display, alerting, or expectedness checks.
- You need **NDC** package codes, dosage form, or route for a product.
- You want to know if a drug/lot is under an **open recall** (enforcement).
- You want to map a brand name to its generic ingredient and **RxCUI** via the
  label's `openfda` block.

## The three endpoints

| Endpoint | Use | Key fields |
| --- | --- | --- |
| `https://api.fda.gov/drug/label.json` | SPL prescribing info | `boxed_warning`, `indications_and_usage`, `warnings`, `dosage_and_administration`, `openfda.brand_name`, `openfda.generic_name`, `openfda.rxcui`, `openfda.product_ndc` |
| `https://api.fda.gov/drug/ndc.json` | NDC directory | `product_ndc`, `generic_name`, `brand_name`, `dosage_form`, `route`, `active_ingredients` |
| `https://api.fda.gov/drug/enforcement.json` | Recalls | `product_description`, `reason_for_recall`, `classification` (Class I/II/III), `recalling_firm`, `status`, `recall_initiation_date` |

No key needed to try it (240 req/min, 1,000/day per IP). A free `api_key=` raises
the daily cap to 120,000.

## Quick start (real OpenFDA queries)

```python
import requests

def openfda(endpoint: str, search: str, limit: int = 1) -> list[dict]:
    url = f"https://api.fda.gov/drug/{endpoint}.json"
    r = requests.get(url, params={"search": search, "limit": limit}, timeout=30)
    if r.status_code == 404:        # OpenFDA returns 404 for zero matches
        return []
    r.raise_for_status()
    return r.json().get("results", [])

# 1) Label: boxed warning + indications for a generic drug.
label = openfda("label", 'openfda.generic_name:"warfarin"')
if label:
    rec = label[0]
    print("Boxed warning:", rec.get("boxed_warning", ["(none)"])[0][:200])
    print("Indication:", rec.get("indications_and_usage", ["(none)"])[0][:200])
    print("RxCUI:", rec.get("openfda", {}).get("rxcui"))

# 2) NDC: package codes, form, route.
ndc = openfda("ndc", 'generic_name:"warfarin"', limit=5)
for rec in ndc:
    print(rec["product_ndc"], rec.get("dosage_form"), rec.get("route"))

# 3) Enforcement: open recalls for a product.
recalls = openfda("enforcement",
                  'product_description:"warfarin"+AND+status:"Ongoing"', limit=5)
for rec in recalls:
    print(rec["classification"], "-", rec["reason_for_recall"][:120])
```

## Workflow

1. **Normalize the drug name first.** Use `openmed.analyze_text` to get the span,
   then prefer the **RxNorm ingredient** (see `normalizing-rxnorm`) as your query
   term — `openfda.generic_name` and the NDC `generic_name` index on the
   ingredient, so a normalized name hits far more records than raw note text.
2. **Query `/drug/label`** with `openfda.generic_name:"<ingredient>"` (or
   `openfda.rxcui:"<rxcui>"` for an exact product). Read `boxed_warning`,
   `indications_and_usage`, `warnings_and_cautions`.
3. **Query `/drug/ndc`** for package-level codes, dosage form, and route.
4. **Query `/drug/enforcement`** filtered to `status:"Ongoing"` to surface open
   recalls; gate alerts on `classification` (Class I = most serious).
5. **Cache** results — labels change rarely; you do not need to re-query per note.
6. **Attach the facts to the extracted drug** keyed by RxCUI/NDC for traceability.

## Hand-off to / from OpenMed

OpenMed's `analyze_text` returns a `dict`; `result["entities"]` items carry
`text`, `label`, `confidence`, `start`, `end`.

- **From** `extracting-clinical-entities`: Pharmaceutical/Chemical entities are
  the query seeds. **From** `normalizing-rxnorm`: pass the RxCUI to
  `openfda.rxcui:"..."` for an exact label match.
- **To** `reporting-adverse-events`: the boxed warning / indications support an
  **expectedness** judgment (is this reaction labeled?). **To**
  `detecting-pv-signals`: confirm whether a disproportionality signal is already
  on-label before escalating.
- OpenMed runs NER **on-device**; only a **de-identified drug name or RxCUI**
  leaves the process to hit OpenFDA. **Never** send a raw note containing PHI to
  the API — de-identify with `openmed.deidentify` first if you must derive the
  query from patient text.

## Edge cases & gotchas

- **OpenFDA returns 404 for an empty result set**, not an empty `results` list —
  handle it as "no match" (the helper above does).
- **Multi-value fields are lists.** `boxed_warning`, `indications_and_usage`, and
  most SPL sections are arrays of strings (`rec["boxed_warning"][0]`). Many
  products have *no* boxed warning — the key is simply absent.
- **Brand vs generic.** `openfda.brand_name` and `openfda.generic_name` differ;
  query the generic (ingredient) for coverage, the brand for a specific product.
- **Labels are SPL snapshots, not real-time.** OpenFDA mirrors DailyMed SPL; a
  brand-new labeling change may lag. For the definitive current label, cross-check
  DailyMed.
- **NDC formats vary** (`product_ndc` is the 2-segment labeler-product code;
  package NDCs add a third segment). Normalize before joining to claims data.
- **Recall `status`** is one of `Ongoing`, `Completed`, `Terminated` — filter to
  `Ongoing` for active risk; `classification` Class I > II > III by severity.
- **Public and free, but rate-limited.** Register a free key and cache; do not
  hammer the API per-note in a batch pipeline.

## Standards & references

- OpenFDA drug label API: https://open.fda.gov/apis/drug/label/
- OpenFDA NDC directory API: https://open.fda.gov/apis/drug/ndc/
- OpenFDA drug enforcement (recalls) API: https://open.fda.gov/apis/drug/enforcement/
- OpenFDA query syntax & rate limits: https://open.fda.gov/apis/query-syntax/ , https://open.fda.gov/apis/authentication/
- FDA Structured Product Labeling (SPL): https://www.fda.gov/industry/structured-product-labeling-resources
- DailyMed (authoritative labels): https://dailymed.nlm.nih.gov/dailymed/
