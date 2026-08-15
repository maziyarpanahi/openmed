# Example Notebooks Gallery

Welcome to the curated **OpenMed Notebook Gallery**. These Jupyter notebooks provide
hands-on, runnable walkthroughs demonstrating end-to-end clinical NLP,
de-identification, batch processing, and interoperability workflows.

Every notebook runs **100% offline on synthetic fixtures** with zero network
calls or mandatory model weight downloads.

---

## Curated Gallery Walkthroughs

| Notebook | Focus | Primary APIs Exercised |
| :--- | :--- | :--- |
| [`01_quickstart_redaction.ipynb`](https://github.com/maziyarpanahi/openmed/blob/master/examples/notebooks/01_quickstart_redaction.ipynb) | Clinical de-identification basics: masking, reversible replacement, and cryptographic hashing. | `deidentify()`, `reidentify()` |
| [`02_batch_dataset.ipynb`](https://github.com/maziyarpanahi/openmed/blob/master/examples/notebooks/02_batch_dataset.ipynb) | Directory and dataset batch processing with error handling and output inspection. | `BatchProcessor.process_directory()` |
| [`03_fhir_export.ipynb`](https://github.com/maziyarpanahi/openmed/blob/master/examples/notebooks/03_fhir_export.ipynb) | Redacting clinical notes, extracting medical entities, and assembling a deterministic FHIR R4 Bundle. | `deidentify()`, `TextProcessor`, `to_bundle()` |
| [`04_eval_walkthrough.ipynb`](https://github.com/maziyarpanahi/openmed/blob/master/examples/notebooks/04_eval_walkthrough.ipynb) | Offline evaluation across bundled multilingual synthetic golden fixtures. | `load_golden_fixtures()`, `safety_sweep()` |

---

## 1. Quickstart Redaction (`01_quickstart_redaction.ipynb`)

Walks through the primary de-identification methods supported by OpenMed:

- **Masking (`method="mask"`)**: Replaces detected direct identifiers with standardized category tags (`[PERSON]`, `[DATE]`, `[PHONE]`, `[EMAIL]`).
- **Reversible Replacement (`method="replace"`)**: Replaces sensitive values with realistic synthetic stand-ins while capturing a persistent lookup mapping table for authorized downstream restoration via `reidentify()`.
- **Cryptographic Hashing (`method="hash"`)**: Produces keyed HMAC/SHA-256 digests to preserve linkage across disparate clinical tables without exposing cleartext PHI.

---

## 2. Batch Processing (`02_batch_dataset.ipynb`)

Demonstrates scalable batch processing across directories of clinical files:

- Initializes `BatchProcessor` with structured de-identification parameters and chunk sizes.
- Streams text files through the offline detector.
- Collects detailed item-level processing telemetry and writes redacted output files to a target directory.

---

## 3. Clinical Extraction & FHIR Export (`03_fhir_export.ipynb`)

Illustrates the end-to-end pipeline from unstructured clinical notes to standards-compliant HL7 FHIR R4 transactions:

1. **Intake Redaction**: Redacts direct patient identifiers from intake narratives.
2. **Clinical Entity Extraction**: Extracts condition mentions, vital signs, and medication dosages with `TextProcessor`.
3. **FHIR Resource Construction**: Maps extracted concepts to standard `Patient`, `Encounter`, `Observation`, and `MedicationStatement` resources.
4. **Transaction Bundle Assembly**: Uses `openmed.clinical.exporters.fhir.to_bundle()` with deterministic document seeds to build a byte-stable R4 transaction Bundle.

---

## 4. Offline Golden Evaluation (`04_eval_walkthrough.ipynb`)

Demonstrates how to evaluate entity extraction fidelity against bundled synthetic benchmarks:

- Loads versioned, synthetic benchmark fixtures via `load_golden_fixtures()`.
- Evaluates detection performance across multiple language locales.
- Computes standard evaluation metrics: Precision, Recall, F1 score, and Exact Span Match counts.

---

## Running Locally

To run the gallery notebooks on your local workstation:

```bash
# 1. Clone the repository and install development dependencies
git clone https://github.com/maziyarpanahi/openmed.git
cd openmed
uv pip install -e ".[dev]"

# 2. Start Jupyter
jupyter notebook examples/notebooks/
```

### Continuous Integration Guarantee

All gallery notebooks in `examples/notebooks/` are continuously executed in CI via `.github/workflows/notebooks-execute.yml` and tested locally via:

```bash
pytest tests/unit/test_notebook_gallery.py -v
```

CI fails automatically if any notebook fails to execute offline or if its committed outputs diverge from the executed state.
