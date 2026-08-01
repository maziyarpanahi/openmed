---
name: mining-pubmed-literature
description: "Searches and fetches PubMed and PMC via NCBI E-utilities (ESearch then EFetch/ESummary) to gather biomedical evidence and build text corpora. Use when the user wants citations for a condition or drug, abstracts to summarize, MeSH-based searches, or a corpus of literature to run NER over. Trigger keywords: PubMed, PMC, NCBI, E-utilities, ESearch, EFetch, ESummary, MeSH, PMID, literature search, abstracts, evidence. Pairs adjacent to OpenMed: fetched abstracts feed openmed.analyze_text for biomedical NER, and OpenMed-extracted diagnoses/drugs/genes become the search terms. E-utilities are public; an optional free API key raises rate limits from 3 to 10 requests/second."
license: Apache-2.0
metadata:
  project: OpenMed
  category: research-genomics
  pairs: adjacent
  version: "1.0"
---

# Mining PubMed & PMC literature (NCBI E-utilities)

Search **PubMed** (citations/abstracts) and **PMC** (full text) programmatically
with **NCBI E-utilities** — the stable HTTP interface to Entrez. The core pattern
is two steps: **ESearch** returns matching record IDs (PMIDs), then **EFetch** (or
**ESummary**) downloads the records. The **Entrez History server** (`usehistory=y`)
lets you chain the two without re-sending thousands of IDs.

E-utilities are public. **No key is required**, but a free API key raises your
limit from **3 to 10 requests/second** and is strongly recommended for batch work.

## When to use

- OpenMed extracted a diagnosis, drug, or gene and you want supporting literature.
- You need abstracts to summarize or to assemble a corpus for biomedical NER.
- You want MeSH-anchored, reproducible searches (date ranges, article types).

For ClinicalTrials.gov use `searching-clinicaltrials`; this skill is for the
published literature.

## Quick start (real E-utilities calls)

Base URL: `https://eutils.ncbi.nlm.nih.gov/entrez/eutils/`. JSON for ESearch/
ESummary via `retmode=json`; EFetch returns text or XML (no JSON for PubMed).

```python
import requests, time

BASE = "https://eutils.ncbi.nlm.nih.gov/entrez/eutils"
API_KEY = None   # set to your free NCBI key to get 10 req/s instead of 3

def _params(**kw):
    if API_KEY:
        kw["api_key"] = API_KEY
    return kw

def esearch(term: str, retmax: int = 50) -> dict:
    """Find PMIDs; usehistory=y stores them on the Entrez History server."""
    r = requests.get(f"{BASE}/esearch.fcgi", params=_params(
        db="pubmed", term=term, retmax=retmax,
        usehistory="y", retmode="json"), timeout=30)
    r.raise_for_status()
    res = r.json()["esearchresult"]
    return {"count": int(res["count"]), "ids": res["idlist"],
            "webenv": res["webenv"], "query_key": res["querykey"]}

def efetch_abstracts(webenv: str, query_key: str, retmax: int = 50) -> str:
    """Pull abstracts by reference to the stored result set (no ID list needed)."""
    r = requests.get(f"{BASE}/efetch.fcgi", params=_params(
        db="pubmed", WebEnv=webenv, query_key=query_key,
        retmax=retmax, rettype="abstract", retmode="text"), timeout=60)
    r.raise_for_status()
    return r.text

hits = esearch('("type 2 diabetes"[MeSH]) AND metformin AND 2023:2025[pdat]')
print(hits["count"], "papers")
abstracts = efetch_abstracts(hits["webenv"], hits["query_key"])
```

Equivalent cURL (search then fetch one PMID's abstract):

```bash
curl "https://eutils.ncbi.nlm.nih.gov/entrez/eutils/esearch.fcgi?db=pubmed&term=metformin&retmode=json"
curl "https://eutils.ncbi.nlm.nih.gov/entrez/eutils/efetch.fcgi?db=pubmed&id=38000000&rettype=abstract&retmode=text"
```

## ESummary for structured metadata

When you need titles/authors/journal/date as JSON (not the full abstract), use
ESummary — it returns one record per ID:

```python
def esummary(ids: list[str]) -> dict:
    r = requests.get(f"{BASE}/esummary.fcgi", params=_params(
        db="pubmed", id=",".join(ids), retmode="json"), timeout=30)
    r.raise_for_status()
    return r.json()["result"]   # keyed by PMID: title, pubdate, source, authors…
```

For PMC full text, repeat with `db=pmc` and EFetch `rettype=""`/`retmode=xml`
(JATS XML). Respect each article's license before redistributing full text.

## Workflow

1. **Build the query.** Combine OpenMed-extracted terms with MeSH tags and field
   filters: `"<disease>"[MeSH] AND <drug>[tiab] AND 2020:2025[pdat]`. Use
   `[tiab]` (title/abstract), `[au]` (author), `[pdat]` (publication date).
2. **ESearch with `usehistory=y`** to capture `WebEnv` + `query_key` and the count.
3. **Batch-fetch** with EFetch/ESummary in pages of ≤ ~200 IDs (or by history),
   sleeping to stay under your rate limit.
4. **Parse** abstracts/metadata; store PMID, title, journal, date, abstract text.
5. **NER the abstracts** with `openmed.analyze_text` to extract diseases, drugs,
   genes, and oncology entities for downstream synthesis.

## Hand-off to / from OpenMed

- **OpenMed facts → query.** `openmed.analyze_text(note)` yields Disease,
  Pharmaceutical, Genomics, and Oncology entities. Turn the top spans into the
  ESearch `term` (optionally grounded: ICD-10 label, RxNorm ingredient, gene
  symbol) to retrieve targeted evidence.
- **Abstracts → OpenMed.** Feed fetched abstracts straight into
  `openmed.analyze_text(abstract, model_name="disease_detection_superclinical")`
  (or a Genomics/Oncology model) to structure the literature into entities for
  evidence tables or knowledge-graph edges.
- Queries and abstracts are **public** literature, not PHI. Still run locally and
  never embed patient text in a search term.

## Edge cases & gotchas

- **Rate limits.** 3 req/s without a key, 10 with one — exceed it and NCBI returns
  HTTP 429. Add `api_key`, throttle, and retry with backoff. NCBI also requests a
  `tool=` and `email=` parameter identifying your application.
- **EFetch has no JSON for PubMed.** Use `retmode=text` (human-readable) or
  `retmode=xml` (PubMedArticle XML) and parse XML for structured fields.
- **History expires.** `WebEnv`/`query_key` are session-scoped — fetch promptly
  after searching, or re-run ESearch.
- **Large result sets.** Page with `retstart`/`retmax` (or history) rather than
  pulling everything at once; cap total fetches.
- **MeSH lag.** Very recent articles may not yet be MeSH-indexed — include `[tiab]`
  term variants so you do not miss them.
- **Full-text licensing.** PMC full text carries per-article licenses; many are
  not redistributable. Store PMIDs/abstracts freely; check the license before
  republishing full text.

## Standards & references

- E-utilities In-Depth (parameters & syntax) — https://www.ncbi.nlm.nih.gov/books/NBK25499/
- E-utilities Quick Start — https://www.ncbi.nlm.nih.gov/books/NBK25497/
- General introduction & policies — https://www.ncbi.nlm.nih.gov/books/NBK25501/
- API keys & rate limits — https://support.nlm.nih.gov/kbArticle/?pn=KA-05317
- PubMed search field tags — https://pubmed.ncbi.nlm.nih.gov/help/
- MeSH browser — https://meshb.nlm.nih.gov/
