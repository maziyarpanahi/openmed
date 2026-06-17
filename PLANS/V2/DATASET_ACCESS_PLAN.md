# OpenMed V2 Dataset Access Plan

OpenMed follows a public-by-default, DUA-eval-only, synthetic-augmentation data
strategy. The package must remain redistributable, local-first, and free of
gated clinical corpora.

## Data Tiers

| Tier | Use | Examples | Rule |
|---|---|---|---|
| Public-by-default | Ship, train, and evaluate when licenses permit | DrugProt, MedMentions, RxNorm, LOINC, ICD-10-CM, HPO, OMOP via Athena, FHIR, ARX, DataProfiler, ported ConText rules | Review license terms before bundling; record wrapped or ported assets in `NOTICE` |
| DUA-gated | Evaluation only | i2b2 2006/2014, n2c2 2014/2018, CEGS N-GRID, SHAC, MIMIC-derived notes, MedNLI, MADE 1.0, SHIELD if Redivis access requires a signed DUA | Credential once, keep held out, never redistribute, never commit, never bake into weights |
| Synthetic | Augment gaps | Locale PHI, social-history sections, structured IDs, OCR overlays, DICOM burned-in text | Augmentation only; never sole training source; validate against public or gated held-out sets |

## Credentialing

At least one maintainer role owns credentialing for DUA-gated evaluation pools.
Credentialing includes required training, data-use agreements, local storage
controls, and periodic access review. Access is limited to evaluation runs and
aggregate reporting.

Gated dataset paths are provided by local configuration or secret store entries.
They are never checked into source control and are never downloaded by default
CI jobs.

## Bundling Rules

- No DUA-gated dataset, note, annotation file, or derived row may be committed.
- No UMLS, SNOMED CT, CPT, MIMIC-lineage, i2b2, n2c2, or PhysioNet restricted
  content may be bundled into wheels, source distributions, fixtures, or model
  artifacts.
- Public datasets may be referenced by URL and loader instructions. Dataset
  files are bundled only after their exact redistribution terms are reviewed.
- Model cards and manifest rows track model-weight licenses separately from
  training-data licenses.
- `bigbio/*` loader stubs do not bypass the underlying dataset's access terms.
- SHIELD is review-required: if public redistribution terms become available,
  it can move to the public-by-default tier; if Redivis access requires a signed
  Data Use Agreement, it stays DUA-gated and eval-only.

## Evaluation Flow

Daily blocking gates run on committed synthetic golden fixtures and public
comparison data. DUA-gated leakage evaluation is a periodic promotion gate run
by a credentialed maintainer role. If DUA access is unavailable, daily public
and synthetic gates continue, but stable-promotion evidence must disclose the
missing gated evaluation.

## Audit Trail

Every dataset used for training or evaluation needs a manifest row or run record
with:

- name and source URL;
- license or DUA tier;
- allowed use: train, eval, benchmark, or augmentation;
- whether redistribution is allowed;
- retrieval date or local credentialing date;
- owner role responsible for review.
