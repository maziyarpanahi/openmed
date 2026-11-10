---
name: annotating-variants
description: "Annotates VCF variants and normalizes HGVS nomenclature with public, license-free annotators (Ensembl VEP REST, VEP/SnpEff/ANNOVAR offline) and links variants to gnomAD population frequencies and the clinical context OpenMed extracts. Use when the user wants to predict variant consequences, map HGVS to genomic coordinates, annotate a VCF, attach allele frequencies, or pair variants with phenotype/oncology context. Trigger keywords: VCF, HGVS, variant annotation, VEP, SnpEff, ANNOVAR, consequence, missense, gnomAD, allele frequency, GRCh38, rsID, transcript. Pairs adjacent to OpenMed: combine annotated variants with Genomics/Oncology entities and phenotype from openmed.analyze_text. Tools used are free; restricted clinical databases are user-supplied."
license: Apache-2.0
metadata:
  project: OpenMed
  category: research-genomics
  pairs: adjacent
  version: "1.0"
---

# Annotating variants & normalizing HGVS

Turn raw genomic variants — VCF rows, rsIDs, or **HGVS** strings — into annotated,
consequence-predicted records, and link them to the **clinical context** OpenMed
extracts from text (genes, variants, oncology findings, phenotype). The workhorse
for a quick, no-install annotation is the **Ensembl VEP REST API**; for scale,
run **VEP**, **SnpEff**, or **ANNOVAR** offline.

These annotators are **free and license-permissive**. Restricted clinical
interpretation databases (e.g. licensed HGMD) are **user-supplied** — this skill
sticks to open resources (Ensembl, gnomAD, ClinVar).

## When to use

- You have a VCF / HGVS / rsID and need consequence predictions (missense,
  stop-gain, splice), affected transcripts, and protein change.
- You need to **normalize HGVS** to genomic coordinates (and back) on a known
  build (GRCh38 by default; GRCh37 via the dedicated endpoint).
- You want **gnomAD** population allele frequencies to flag common vs rare.
- You are pairing molecular findings with the phenotype/oncology context that
  OpenMed pulls from notes or literature.

## Quick start (real Ensembl VEP REST call)

Base URL: `https://rest.ensembl.org` (GRCh38). For GRCh37 use
`https://grch37.rest.ensembl.org`. Default species is `human`/`homo_sapiens`.

```python
import requests

REST = "https://rest.ensembl.org"
HEADERS = {"Content-Type": "application/json", "Accept": "application/json"}

def vep_hgvs(hgvs: str) -> list[dict]:
    """Annotate a single HGVS variant (GET)."""
    r = requests.get(f"{REST}/vep/human/hgvs/{hgvs}", headers=HEADERS, timeout=30)
    r.raise_for_status()
    return r.json()

# Transcript-level HGVS (coding) — note the build-aware default transcript set
ann = vep_hgvs("ENST00000269305.9:c.215C>G")   # TP53 example
v = ann[0]
print(v["most_severe_consequence"])            # e.g. "missense_variant"
for tc in v.get("transcript_consequences", []):
    print(tc["gene_symbol"], tc.get("hgvsp"), tc.get("sift_prediction"),
          tc.get("polyphen_prediction"))
```

Batch many variants with the POST endpoint (region "CHROM POS ID REF ALT . . ."
format, up to 200 per request):

```python
def vep_region_batch(variants: list[str]) -> list[dict]:
    body = {"variants": variants}   # ["17 7676154 . C G . . .", ...] 1-based
    r = requests.post(f"{REST}/vep/human/region", headers=HEADERS,
                      json=body, timeout=60)
    r.raise_for_status()
    return r.json()
```

Equivalent cURL:

```bash
curl 'https://rest.ensembl.org/vep/human/hgvs/ENST00000269305.9:c.215C>G' \
  -H 'Content-Type:application/json'
```

Response highlights per variant: `most_severe_consequence`,
`transcript_consequences[]` (`gene_symbol`, `hgvsc`, `hgvsp`, `sift_prediction`,
`polyphen_prediction`, `impact`), and `colocated_variants[]` (rsIDs and
population frequencies). Request gnomAD frequencies and ClinVar via VEP options /
plugins.

## Population frequencies via gnomAD (GraphQL)

For authoritative allele frequencies, query the **gnomAD GraphQL API** at
`https://gnomad.broadinstitute.org/api`. Use variant IDs in
`chrom-pos-ref-alt` form. Frequencies are derived from `ac`/`an` (allele count /
number) — request those, not a non-existent `af` on subpopulations.

```python
GNOMAD = "https://gnomad.broadinstitute.org/api"

QUERY = """
query Variant($id: String!, $ds: DatasetId!) {
  variant(variantId: $id, dataset: $ds) {
    variant_id rsids
    genome { ac an af homozygote_count }
    exome  { ac an af homozygote_count }
  }
}"""

def gnomad_freq(variant_id: str, dataset: str = "gnomad_r4") -> dict:
    r = requests.post(GNOMAD, json={"query": QUERY,
        "variables": {"id": variant_id, "ds": dataset}}, timeout=30)
    r.raise_for_status()
    return r.json()["data"]["variant"]

# gnomad_freq("17-7676154-C-G")  -> ac/an/af for exome and genome
```

## Offline annotation at scale

For whole-VCF jobs, run a local annotator instead of per-variant REST calls:

| Tool | Strengths | Notes |
| --- | --- | --- |
| **Ensembl VEP** (offline) | richest, plugin ecosystem (gnomAD, CADD, SpliceAI), HGVS | needs cache download per build |
| **SnpEff** | fast, self-contained genome databases | great for bulk consequence calling |
| **ANNOVAR** | many annotation databases | registration required; license terms apply |

All emit per-variant gene, consequence, and (with the right database) frequency
and clinical fields. Keep the **reference build (GRCh38)** consistent end to end.

## Workflow

1. **Normalize input** to a canonical form: left-align/trim VCF alleles; for HGVS,
   confirm the reference transcript and build.
2. **Annotate** — REST (`/vep/human/hgvs` or `/vep/human/region`) for a handful,
   offline VEP/SnpEff for a VCF.
3. **Attach frequencies** from gnomAD; flag common variants (e.g. AF > 1%).
4. **Filter/prioritize** by `most_severe_consequence`, impact, and rarity.
5. **Join to clinical context** from OpenMed (gene/variant mentions, oncology,
   phenotype) to assemble an interpretable record.

## Hand-off to / from OpenMed

- **OpenMed → variant context.** `openmed.analyze_text(report, model_name=<a
  Genomics or Oncology model>)` extracts gene symbols, variant mentions (e.g.
  "EGFR L858R"), and tumor/oncology findings from pathology or molecular reports.
  Use those to (a) select which VCF variants matter and (b) attach phenotype
  context to each annotation.
- **Variant → OpenMed.** Free-text variant descriptions in reports can be
  normalized to HGVS here, then the surrounding clinical narrative is structured by
  OpenMed — linking genotype to extracted phenotype/diagnosis.
- Keep genomic + clinical data **local**. The REST/GraphQL calls carry only the
  **variant coordinates** (public allele data), never patient identifiers — and
  any narrative is de-identified with `openmed.deidentify` first.

## Edge cases & gotchas

- **Build mismatch is the #1 error.** GRCh38 coordinates against a GRCh37 endpoint
  (or cache) give wrong genes. Use `grch37.rest.ensembl.org` only for GRCh37 data;
  default REST is GRCh38.
- **Transcript choice changes the HGVS.** `c.`/`p.` notation depends on the
  reference transcript (MANE Select vs others). Pin the transcript explicitly.
- **Normalize before annotating.** Un-left-aligned indels and multi-allelic VCF
  rows produce inconsistent annotations — decompose and normalize first
  (e.g. `bcftools norm`).
- **REST is rate-limited.** ~15 req/s and 200 variants/POST on the Ensembl REST
  server; switch to offline VEP for large VCFs. Honor `Retry-After` on 429.
- **gnomAD subpopulation fields.** Query `ac`/`an` (and compute AF) for
  subpopulations; some schema paths reject `af` directly — track the current
  schema version, which changes between gnomAD releases.
- **No clinical interpretation here.** Consequence ≠ pathogenicity. Pathogenicity
  classification (ACMG/AMP) uses curated evidence and licensed databases the user
  supplies; this skill produces annotations, not diagnoses.

## Standards & references

- VCF specification — https://samtools.github.io/hts-specs/VCFv4.4.pdf
- HGVS nomenclature — https://hgvs-nomenclature.org/
- Ensembl VEP REST (HGVS GET) — https://rest.ensembl.org/documentation/info/vep_hgvs_get
- Ensembl VEP REST (region POST) — https://rest.ensembl.org/documentation/info/vep_region_post
- Ensembl VEP (offline) — https://www.ensembl.org/info/docs/tools/vep/index.html
- SnpEff — https://pcingola.github.io/SnpEff/
- gnomAD API — https://gnomad.broadinstitute.org/api
- ClinVar — https://www.ncbi.nlm.nih.gov/clinvar/
