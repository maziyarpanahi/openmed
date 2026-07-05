# Frequently Asked Questions

These answers cover the questions that come up most often when teams start using OpenMed in local clinical NLP,
PII detection, de-identification, and service deployments.

## Install and Run

### Can OpenMed run fully locally or in an air-gapped environment?

Yes. OpenMed can run without sending clinical text to an external service. For strict offline use, pre-download the
model files, point `model_name` or `model_id` at that local directory, and keep the runtime on `device="cpu"` or another
device available inside your environment.

When the identifier is an existing local path, OpenMed asks the underlying loader to use `local_files_only=True`, so
missing tokenizer, config, or weight files fail locally instead of downloading from the model hub. See
[Loading from a local path](analyze-text.md#loading-from-a-local-path).

Do not rely on an `OPENMED_OFFLINE` switch in the current package. That dedicated offline guard is tracked separately;
today, local model paths and pre-seeded caches are the supported offline controls.

### Which package extras should I install?

Use the smallest extra that matches your runtime:

- `openmed[hf]` for the standard Python model runtime.
- `openmed[hf,service]` when you need the REST service.
- `openmed[mlx]` for Python MLX acceleration on Apple Silicon.
- `openmed[multimodal]` for document/image intake and Tesseract OCR; install
  the system `tesseract` binary separately (`brew install tesseract` on macOS
  or `sudo apt-get install tesseract-ocr` on Debian/Ubuntu).
- `openmed[ocr-paddle]` for the heavier optional PaddleOCR backend.

Start with the [Quick Start](getting-started.md), then use
[Configuration & Validation](configuration.md) for cache paths, device selection, profiles, and environment overrides.

## Models and Languages

### Which model should I use?

For clinical entity extraction, pick a registry alias that matches the entity family you need, such as disease, drug,
anatomy, oncology, gene, or PII. The registry exposes metadata and helper functions for UI dropdowns, text-based
suggestions, model sizes, and recommended confidence thresholds. See the [Model Registry](model-registry.md).

For PII, `extract_pii(..., lang="<code>")` selects the default model for the requested language when you keep the default
model argument. Override `model_name` only when you need a specific checkpoint, local directory, or privacy-filter family.

### Which languages are supported?

PII extraction and de-identification support **16 supported PII language codes**:
`ar`, `de`, `en`, `es`, `fr`, `he`, `hi`, `id`, `it`, `ja`, `nl`, `pt`, `ro`, `te`, `th`, and `tr`.
The README keeps a short multilingual example set in
[Multilingual PII](https://github.com/maziyarpanahi/openmed#multilingual-pii-16-languages).

Clinical NER coverage depends on the selected registry model. Check each model's `languages`, `entity_types`, and
specialization in the [Model Registry](model-registry.md) before putting it behind an API or batch job.

## Privacy and De-identification

### Is de-identification reversible?

It depends on the method:

- `mask` and `remove` do not preserve the original value in the output.
- `replace` emits locale-aware synthetic surrogates; it is not reversible unless your own workflow stores an external
  mapping.
- `hash` is one-way, but deterministic for linking repeated values.
- `shift_dates` can be reversed only by someone who knows the shift amount.

Always review outputs before releasing data. PII detection is an assistive control, not a substitute for a privacy review
process. See [PII Anonymization](anonymization.md).

### Why are many national IDs grouped under `ID_NUM`?

OpenMed normalizes detector-specific labels into 50 canonical PII labels. `ID_NUM` is the canonical bucket for general
identifiers such as medical record numbers, national IDs, CPF/CNPJ, NIR, Steuer-ID, Codice Fiscale, DNI/NIE, BSN,
Aadhaar, and NPI. Labels with their own canonical category, such as `SSN`, `ACCOUNT_NUMBER`, or `CREDIT_CARD`, can still
stay separate when the detector emits them clearly.

This keeps multilingual model output consistent while policy profiles can still treat identifiers as high-risk direct
identifiers. The canonical taxonomy lives in
[openmed/core/labels.py](https://github.com/maziyarpanahi/openmed/blob/master/openmed/core/labels.py).

### Should I use reversible or irreversible de-identification?

Use irreversible output (`mask`, `remove`, or one-way `hash`) when the downstream workflow does not need the original
values. Use `replace` when clinicians, QA reviewers, or demos need realistic-looking synthetic text. Use `shift_dates`
only when preserving relative timelines matters and the offset can be governed like sensitive metadata.

## Licensing

### What license does OpenMed use?

The OpenMed package is released under Apache-2.0. See the
[repository license](https://github.com/maziyarpanahi/openmed/blob/master/LICENSE).

Model checkpoints may carry their own metadata, so verify the specific model card or registry row before redistributing
weights or shipping them in a product bundle.

## Performance

### Does OpenMed require a GPU?

No. CPU execution is supported and is the default safe baseline for local and CI environments. GPU acceleration can improve
latency and throughput for larger workloads:

- CUDA devices can be selected with `OpenMedConfig(device="cuda")`.
- Apple Silicon systems can use the MLX backend when the relevant extra and model artifacts are available.
- Batch processing can improve throughput for repeated extraction or de-identification jobs.

See [Configuration & Validation](configuration.md#cache-device-tips), [MLX Backend](mlx-backend.md),
[Batch Processing](batch-processing.md), and [Performance Profiling](profiling.md).

### How do I keep memory usage predictable?

Reuse a `ModelLoader` or batch processor instead of creating a new pipeline for every document. In the REST service, use
the model lifecycle endpoints to inspect and unload cached models. See [ModelLoader & Pipelines](model-loader.md) and
[REST Service](rest-service.md).
