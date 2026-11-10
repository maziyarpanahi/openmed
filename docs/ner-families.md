# Biomedical NER Families

OpenMed's manifest-backed [model registry](./model-registry.md) groups trained
named-entity recognition checkpoints into the families below. Entity types,
available size categories, and confidence defaults reflect the current registry;
the documentation coherence test prevents this catalog from drifting as the
manifest changes.

The linked zero-shot domains come from `available_domains()` and open the
generated [Clinical Domain Label Catalog](./clinical-domains.md). See the
[Zero-shot NER Toolkit](./zero-shot-ner.md#domain-defaults) for label-resolution
behavior and custom domain maps.

Recommended confidence values are starting thresholds, not universal operating
points. Validate them on representative local data before production use. This
catalog describes extraction coverage only and is not clinical guidance.

## Medical

- **Entity types:** `CANCER`, `CELL`, `CHEM`, `CONDITION`, `DISEASE`, `DNA`, `DRUG`, `GENE`, `GENE_OR_GENE_PRODUCT`, `MEDICATION`, `ORGANISM`, `PATHOLOGY`, `PROTEIN`, `RNA`, `SIMPLE_CHEMICAL`, `SPECIES`
- **Available size categories:** `Tiny`, `Small`, `Medium`, `Large`, `XLarge`, `Unknown`
- **Recommended confidence:** `0.60`
- **Zero-shot domain:** [`biomedical`](./clinical-domains.md#biomedical)

## Anatomy

- **Entity types:** `ANATOMY`, `ORGAN`, `TISSUE`
- **Available size categories:** `Tiny`, `Small`, `Medium`, `Large`, `XLarge`
- **Recommended confidence:** `0.60`
- **Zero-shot domain:** [`anatomy`](./clinical-domains.md#anatomy)

## Hematology

- **Entity types:** `CANCER`, `CELL`, `CL`, `DISEASE`
- **Available size categories:** `Tiny`, `Medium`, `Large`, `XLarge`
- **Recommended confidence:** `0.65`
- **Zero-shot domain:** [`hematology`](./clinical-domains.md#hematology)

## Chemical

- **Entity types:** `CHEM`, `CHEMICAL`, `DRUG`, `MEDICATION`, `SIMPLE_CHEMICAL`
- **Available size categories:** `Tiny`, `Small`, `Medium`, `Large`, `XLarge`
- **Recommended confidence:** `0.60`
- **Zero-shot domain:** [`chemical`](./clinical-domains.md#chemical)

## Disease

- **Entity types:** `CONDITION`, `DISEASE`, `PATHOLOGY`
- **Available size categories:** `Tiny`, `Medium`, `Large`, `XLarge`
- **Recommended confidence:** `0.60`
- **Zero-shot domain:** [`disease`](./clinical-domains.md#disease)

## Genomics

- **Entity types:** `CELL`, `CELL_LINE`, `CELL_TYPE`, `DNA`, `GENE`, `GENE_OR_GENE_PRODUCT`, `PROTEIN`, `RNA`
- **Available size categories:** `Tiny`, `Medium`, `Large`, `XLarge`
- **Recommended confidence:** `0.65`
- **Zero-shot domain:** [`genomics`](./clinical-domains.md#genomics)

## Oncology

- **Entity types:** `AMINO_ACID`, `ANATOMICAL_SYSTEM`, `ANATOMY`, `CANCER`, `CELL`, `CELLULAR_COMPONENT`, `CHEM`, `CHEMICAL`, `CONDITION`, `DEVELOPING_ANATOMICAL_STRUCTURE`, `DISEASE`, `GENE_OR_GENE_PRODUCT`, `IMMATERIAL_ANATOMICAL_ENTITY`, `MULTI_TISSUE_STRUCTURE`, `ORGAN`, `ORGANISM`, `ORGANISM_SUBDIVISION`, `ORGANISM_SUBSTANCE`, `PATHOLOGICAL_FORMATION`, `PATHOLOGY`, `PROTEIN`, `SIMPLE_CHEMICAL`, `SPECIES`, `TISSUE`
- **Available size categories:** `Tiny`, `Small`, `Medium`, `Large`, `XLarge`
- **Recommended confidence:** `0.65`
- **Zero-shot domain:** [`oncology`](./clinical-domains.md#oncology)

## Species

- **Entity types:** `ORGANISM`, `SPECIES`
- **Available size categories:** `Tiny`, `Medium`, `Large`, `XLarge`
- **Recommended confidence:** `0.60`
- **Zero-shot domain:** [`species`](./clinical-domains.md#species)

## Pathology

- **Entity types:** `CONDITION`, `DISEASE`, `PATHOLOGY`
- **Available size categories:** `Tiny`, `Medium`, `Large`, `XLarge`
- **Recommended confidence:** `0.60`
- **Zero-shot domain:** [`pathology`](./clinical-domains.md#pathology)

## Pharmaceutical

- **Entity types:** `CHEM`, `CHEMICAL`, `DRUG`, `MEDICATION`
- **Available size categories:** `Tiny`, `Medium`, `Large`, `XLarge`
- **Recommended confidence:** `0.65`
- **Zero-shot domain:** [`pharmaceutical`](./clinical-domains.md#pharmaceutical)

## Protein

- **Entity types:** `GENE_OR_GENE_PRODUCT`, `PROTEIN`, `PROTEIN_COMPLEX`, `PROTEIN_ENUM`, `PROTEIN_FAMILIY_OR_GROUP`, `PROTEIN_VARIANT`
- **Available size categories:** `Tiny`, `Medium`, `Large`, `XLarge`
- **Recommended confidence:** `0.60`
- **Zero-shot domain:** [`protein`](./clinical-domains.md#protein)
