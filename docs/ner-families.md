# Biomedical NER Families

This page summarizes the trained biomedical and clinical NER families exposed by
the OpenMed model registry, then connects them to the zero-shot domain defaults
used by the GLiNER toolkit. The trained-family rows below are sourced from
`openmed.core.model_registry` helpers so the documented category names, entity
types, size tiers, and recommended confidence thresholds stay aligned with the
committed model manifest.

Use this page when you need to choose a trained family for a known extraction
task, or when you want to pair a trained token-classification family with a
zero-shot domain for exploratory labeling.

<!-- ner-family-categories:start -->

## Anatomy

| Field | Value |
| --- | --- |
| Registry category | `Anatomy` |
| Model count | 42 |
| Entity types | `ANATOMY`, `ORGAN`, `TISSUE` |
| Available tiers | Large, Medium, Small, Tiny, XLarge |
| Recommended confidence | 0.60 |
| Zero-shot correspondence | `clinical`, `biomedical` |

Use Anatomy models for organ, tissue, and body-structure extraction in clinical
notes or biomedical passages.

## Chemical

| Field | Value |
| --- | --- |
| Registry category | `Chemical` |
| Model count | 41 |
| Entity types | `CHEM`, `DRUG`, `MEDICATION`, `SIMPLE_CHEMICAL` |
| Available tiers | Large, Medium, Small, Tiny, XLarge |
| Recommended confidence | 0.60 |
| Zero-shot correspondence | `chemistry`, `biomedical` |

Use Chemical models when the target span is a molecule, compound, medication, or
chemistry-specific mention rather than a prescription workflow.

## Disease

| Field | Value |
| --- | --- |
| Registry category | `Disease` |
| Model count | 31 |
| Entity types | `CONDITION`, `DISEASE`, `PATHOLOGY` |
| Available tiers | Large, Medium, Tiny, XLarge |
| Recommended confidence | 0.60 |
| Zero-shot correspondence | `clinical`, `biomedical`, `public_health` |

Use Disease models for diagnoses, disorders, disease names, and condition spans.

## Genomics

| Field | Value |
| --- | --- |
| Registry category | `Genomics` |
| Model count | 92 |
| Entity types | `DNA`, `GENE`, `GENE_OR_GENE_PRODUCT`, `PROTEIN`, `RNA` |
| Available tiers | Large, Medium, Tiny, XLarge |
| Recommended confidence | 0.65 |
| Zero-shot correspondence | `genomic`, `biomedical` |

Use Genomics models for genes, variants, DNA/RNA mentions, transcripts, and
gene-product extraction.

## Hematology

| Field | Value |
| --- | --- |
| Registry category | `Hematology` |
| Model count | 30 |
| Entity types | `CANCER`, `DISEASE` |
| Available tiers | Large, Medium, Tiny, XLarge |
| Recommended confidence | 0.65 |
| Zero-shot correspondence | `clinical`, `biomedical` |

Use Hematology models for blood disorders and blood-cancer extraction workflows.

## Medical

| Field | Value |
| --- | --- |
| Registry category | `Medical` |
| Model count | 123 |
| Entity types | `CANCER`, `CELL`, `CHEM`, `CONDITION`, `DISEASE`, `DNA`, `DRUG`, `GENE`, `GENE_OR_GENE_PRODUCT`, `MEDICATION`, `ORGANISM`, `PATHOLOGY`, `PROTEIN`, `RNA`, `SIMPLE_CHEMICAL`, `SPECIES` |
| Available tiers | Large, Medium, Small, Tiny, Unknown, XLarge |
| Recommended confidence | 0.60 |
| Zero-shot correspondence | `biomedical`, `clinical` |

Use Medical models for broad biomedical extraction when a more specific category
does not match the input or when evaluating general-purpose catalog coverage.

## Oncology

| Field | Value |
| --- | --- |
| Registry category | `Oncology` |
| Model count | 42 |
| Entity types | `CANCER`, `CELL`, `CHEM`, `CONDITION`, `DISEASE`, `GENE_OR_GENE_PRODUCT`, `ORGANISM`, `PATHOLOGY`, `PROTEIN`, `SIMPLE_CHEMICAL`, `SPECIES` |
| Available tiers | Large, Medium, Small, Tiny, XLarge |
| Recommended confidence | 0.65 |
| Zero-shot correspondence | `clinical`, `biomedical`, `genomic` |

Use Oncology models for cancer, tumor, pathology, and therapy-related entity
extraction.

## Pathology

| Field | Value |
| --- | --- |
| Registry category | `Pathology` |
| Model count | 31 |
| Entity types | `CONDITION`, `DISEASE`, `PATHOLOGY` |
| Available tiers | Large, Medium, Tiny, XLarge |
| Recommended confidence | 0.60 |
| Zero-shot correspondence | `clinical`, `biomedical` |

Use Pathology models for histology, biopsy, disease-process, and pathology
report terminology.

## Pharmaceutical

| Field | Value |
| --- | --- |
| Registry category | `Pharmaceutical` |
| Model count | 32 |
| Entity types | `CHEM`, `DRUG`, `MEDICATION`, `SIMPLE_CHEMICAL` |
| Available tiers | Large, Medium, Tiny, XLarge |
| Recommended confidence | 0.65 |
| Zero-shot correspondence | `clinical`, `biomedical`, `chemistry` |

Use Pharmaceutical models for medications, drugs, and prescription-like mentions.

## Protein

| Field | Value |
| --- | --- |
| Registry category | `Protein` |
| Model count | 31 |
| Entity types | `GENE_OR_GENE_PRODUCT`, `PROTEIN` |
| Available tiers | Large, Medium, Tiny, XLarge |
| Recommended confidence | 0.60 |
| Zero-shot correspondence | `genomic`, `biomedical` |

Use Protein models for protein and gene-product spans where protein mentions are
the primary target.

## Species

| Field | Value |
| --- | --- |
| Registry category | `Species` |
| Model count | 59 |
| Entity types | `ORGANISM`, `SPECIES` |
| Available tiers | Large, Medium, Tiny, XLarge |
| Recommended confidence | 0.60 |
| Zero-shot correspondence | `organism`, `biomedical` |

Use Species models for organisms, strains, taxa, and species mentions.

<!-- ner-family-categories:end -->

## Zero-shot Domains

The zero-shot defaults are exposed by `openmed.ner.available_domains()` and
`openmed.ner.get_default_labels()`. They complement the trained families above
when you need exploratory labels or a domain that is not covered by a dedicated
token-classification model.

| Domain | Default labels |
| --- | --- |
| `biomedical` | Disease, Drug, Gene, Organism |
| `chemistry` | Compound, Reaction, Property, Unit |
| `clinical` | Problem, Treatment, Test, BodyPart |
| `cybersecurity` | Vulnerability, CVE, Software, Vendor |
| `ecommerce` | Product, Brand, Price, Attribute |
| `education` | Course, Topic, Institution, Degree |
| `finance` | Company, Ticker, Instrument, Event |
| `generic` | Person, Organization, Location, Date |
| `genomic` | Variant, Gene, Transcript, Phenotype |
| `legal` | Case, Court, Judge, Statute |
| `news` | Person, Organization, Location, Date |
| `organism` | Species, Strain, Taxon, Habitat |
| `public_health` | Condition, Intervention, Outcome, Population |
| `social` | Person, Handle, Hashtag, URL |

For GLiNER usage and label-resolution precedence, see the
[Zero-shot NER Toolkit](zero-shot-ner.md). For registry APIs such as
`list_model_categories()`, `get_models_by_category()`, and
`get_entity_types_by_category()`, see the [Model Registry](model-registry.md).
