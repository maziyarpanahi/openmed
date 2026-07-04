# Feature Map & Capabilities

This page inventories the main surfaced capabilities in OpenMed and maps them
back to source modules, docs, and runnable examples. For release-specific
coverage, see
[OpenMed v1.6-v1.7 Feature Coverage](./release/v1.6-v1.7-feature-coverage.md).

## Privacy And De-identification

| Area | What it covers | Where to look |
| --- | --- | --- |
| Policy-aware de-identification | `deidentify`, policy profiles, calibrated thresholds, arbitration, cascade routing, safety sweeps, custom recognizers, and clinical term protection. | `openmed/core/pii.py`, `openmed/core/pipeline.py`, `openmed/core/policy.py`, `openmed/core/clinical_protect.py`, [PII Anonymization](./anonymization.md) |
| Canonical span contracts | Versioned `OpenMedSpan`, schema fingerprints, redaction action schemas, provenance, and compatibility gates. | `openmed/core/schemas/`, [De-identification API](./api/deidentification.md), `examples/v16_policy_audit_release_gates.py` |
| Audit and review evidence | Signed audit reports, reproducibility hashes, review bundles, FHIR `Provenance`/`AuditEvent`, audit diffs, and PHI-safe previews. | `openmed/core/audit.py`, `openmed/core/redaction_preview.py`, `openmed/risk/audit_diff.py`, `openmed/clinical/exporters/fhir/provenance.py`, `examples/v16_policy_audit_release_gates.py` |
| Runtime de-identification features | `DeidentificationResult.to_dataframe`, surrogate vaults, patient-keyed date shifting, format-preserving redaction, minimum-necessary action selection, streaming redaction, explain traces, section stamping, and risk budgets. | `openmed/core/pii.py`, `openmed/core/surrogate_vault.py`, `openmed/core/date_shift.py`, `openmed/core/anonymizer/format_preserve.py`, `openmed/core/redaction_strength.py`, `openmed/core/streaming.py`, `openmed/core/explain.py`, `openmed/risk/budget.py` |
| Multilingual PII | 15 supported PII language codes: ar, de, en, es, fr, he, hi, id, it, ja, nl, pt, te, th, and tr; locale validators, script detection, date/number normalization, and deterministic locale PHI generation. | `openmed/core/pii_i18n.py`, `openmed/core/script_detect.py`, `openmed/core/locale_formats.py`, `openmed/training/synthetic/locale_phi.py`, `examples/pii_multilingual_new_languages.py` |

## Multimodal And Structured Inputs

| Area | What it covers | Where to look |
| --- | --- | --- |
| Multimodal document contract | `ExtractedDocument`, `SourceSpan`, lazy handler registration, source-offset mapping, and dispatcher-based redaction. | `openmed/multimodal/base.py`, `examples/v17_multimodal_browser_interop.py` |
| OCR and scanned documents | Shared OCR result contract, fake test engine, Tesseract, PaddleOCR, EasyOCR, docTR, OCR language selection, and available-engine discovery. | `openmed/multimodal/ocr.py`, `tests/unit/multimodal/test_ocr_engines.py`, `examples/v17_multimodal_browser_interop.py` |
| Markup and metadata | Markdown/AsciiDoc and EPUB extraction, source text redaction, image/PDF/DOCX metadata scrubbing, and residual metadata verification. | `openmed/multimodal/documents_markdown.py`, `openmed/multimodal/epub.py`, `openmed/multimodal/metadata_scrub.py`, `examples/v17_multimodal_browser_interop.py` |
| Chat logs and tabular data | JSONL chat-log redaction with speaker pseudonymization, CSV/TSV PHI column classification, column actions, and PHI-safe manifests. | `openmed/multimodal/chatlog_jsonl.py`, `openmed/multimodal/tabular_csv.py`, `examples/v17_multimodal_browser_interop.py` |

## Health-data Interop

| Area | What it covers | Where to look |
| --- | --- | --- |
| FHIR helpers | Deterministic R4 Bundles, stable `urn:uuid` references, `OperationOutcome`, `Provenance`, `AuditEvent`, CodeableConcept builders/checkers, and coding provenance. | `openmed/clinical/exporters/fhir/`, `openmed/clinical/exporters/codeable_concept_simple.py`, `openmed/clinical/exporters/codeable_concept_check.py`, [FHIR Interop Helpers](./fhir-interop.md) |
| FHIR operations and bulk export | `$de-identify` resource/Bundle wrappers and FHIR Bulk NDJSON de-identification summaries. | `openmed/interop/fhir_operations.py`, `openmed/interop/fhir_bulk.py`, `examples/v17_multimodal_browser_interop.py` |
| HL7 v2 and CDA/C-CDA | HL7 v2 segment/field redaction, CDA/C-CDA XML de-identification, and multimodal XML dispatch. | `openmed/interop/hl7v2.py`, `openmed/interop/cda.py`, [HL7 v2 De-identification](./hl7v2-deidentification.md) |
| Optional adapters | Presidio, PHILTER, pyDeid, GLiNER-BioMed, LangChain, and spaCy adapter surfaces. | `openmed/interop/`, [LangChain Redaction Wrapper](./integrations-langchain.md), [spaCy Pipeline Component](./spacy-component.md) |

## Clinical Extraction

| Area | What it covers | Where to look |
| --- | --- | --- |
| Clinical context | Negation, temporality, uncertainty, experiencer, section-aware priors, and context offsets. | `openmed/clinical/context.py`, [Clinical Context & Extraction Depth](./clinical/context-and-extraction.md) |
| Clinical normalization | Labs, vitals, medication sigs, problem-list status, summary cards, microbiology, dermatology/ophthalmology labels, status vocabulary, and flat-table export. | `openmed/clinical/`, `openmed/clinical/exporters/flat_table.py`, [Clinical Context & Extraction Depth](./clinical/context-and-extraction.md) |
| Zero-shot domains | GLiNER label maps, clinical NER families, cardiology, dermatology, ophthalmology, microbiology, nutrition, and routing metadata. | `openmed/zero_shot/`, `openmed/ner/families/`, [Zero-shot Toolkit](./zero-shot-ner.md), `examples/clinical_ner_families.py` |

## Models, Backends, And Browser Export

| Area | What it covers | Where to look |
| --- | --- | --- |
| Model registry and manifests | Curated metadata, manifest schema validation, manifest diffs, model recommendations, model card generation, and publishing metadata. | `models.jsonl`, `openmed/core/model_registry.py`, `openmed/core/manifest_schema.py`, `openmed/core/manifest_diff.py`, `openmed/core/model_card.py`, [Model Registry](./model-registry.md), [Model Manifest](./model-manifest.md) |
| One-call inference | `analyze_text`, typed `AnalyzeResult`, validation, grouping, and dict/JSON/HTML/CSV formatting. | `openmed/__init__.py`, `openmed/core/results.py`, [Analyze Text Helper](./analyze-text.md) |
| Loader and batch processing | Model loading, cache reuse, tokenizer caching, batch `analyze_text`, batch PII extraction, and ordered de-identification output. | `openmed/core/models.py`, `openmed/processing/tokenizer_cache.py`, `openmed/processing/batch.py`, [ModelLoader & Pipelines](./model-loader.md), [Batch Processing](./batch-processing.md) |
| Apple runtimes | Python MLX, Laneformer MLX-LM, Swift OpenMedKit, Core ML export, MLX quantized export, and PyTorch MPS selection/tuning. | `openmed/mlx/`, `openmed/coreml/`, `openmed/torch/device.py`, `swift/OpenMedKit`, [MLX Backend](./mlx-backend.md), [Swift Package](./swift-openmedkit.md), [CoreML Packaging](./coreml-export.md), [Torch MPS Performance](./performance-mps.md) |
| Quantized and browser formats | AWQ, GPTQ, bitsandbytes loading, ONNX/WebGPU, Transformers.js bundle export, and ONNX/quantized manifest metadata. | `openmed/torch/quantize_awq.py`, `openmed/torch/quantize_gptq.py`, `openmed/torch/loader_quant.py`, `openmed/onnx/`, [AWQ Export](./export-awq.md), [GPTQ Export](./export-gptq.md), [Transformers.js Export](./export-transformersjs.md) |
| Training infrastructure | DAPT corpus assembly, Mode-A distillation, DirectID contracts, hard-negative sampling, active learning, adjudication, and span-relation graph decoding. | `openmed/training/`, `openmed/core/decoding/graph.py` |

## Service, Clients, And Operations

| Area | What it covers | Where to look |
| --- | --- | --- |
| REST service | FastAPI endpoints, shared runtime, model warm pools, dynamic batching, request coalescing, rate/concurrency limits, trusted hosts, CORS, `/livez`, `/readyz`, and metrics. | `openmed/service/`, [REST Service](./rest-service.md) |
| Typed clients | Python service client and TypeScript REST client parity. | `openmed/service/client.py`, `clients/typescript/`, `clients/typescript/README.md` |
| Streaming and connectors | Core incremental de-identification and Kafka streaming de-identification. | `openmed/core/streaming.py`, `openmed/processing/kafka_connector.py` |
| CLI | `openmed deid`, FHIR bundle, model recommendation/diff/card, policy diff/lint, doctor, gates preview/bundle, audit/risk, active learning, benchmark, and calibration commands. | `openmed/cli/`, [Contributing & Releases](./contributing.md) |

## Evaluation, Risk, And Release Evidence

| Area | What it covers | Where to look |
| --- | --- | --- |
| Evaluation harness | Benchmark reports, public biomedical NER, DrugProt, i2b2 loader stubs, SHIELD, clinical PHI manifests, dataset cards, coverage, section recall, caching, fairness, robustness, and error analysis. | `openmed/eval/`, [Eval Harness & Metrics](./eval-harness.md), [Golden Benchmark](./benchmarks/golden.md) |
| Leakage and utility evidence | Leakage heatmaps, model scorecards, threshold sweeps, paired significance, flaky-run detection, calibration reliability, over-redaction, utility loss, policy compliance, and benchmark history diffs. | `openmed/eval/leakage_heatmap.py`, `openmed/eval/scorecard.py`, `openmed/eval/threshold_sweep.py`, `openmed/eval/utility.py`, `openmed/eval/history.py` |
| Risk metrics | Re-identification risk, membership/linkage probes, k-anonymity, l-diversity, t-closeness, risk budgets, dashboards, and audit diffs. | `openmed/risk/`, `openmed/eval/attacks/`, `examples/v16_policy_audit_release_gates.py` |
| Release gates and status | Fail-closed gates, evidence bundles, last-green baseline, device tiers, nano certification, status pages, and leaderboard pages. | `openmed/eval/release_gates.py`, `openmed/eval/evidence_bundle.py`, `openmed/eval/tiers.py`, `openmed/eval/nano_cert.py`, `docs/status/`, `docs/leaderboard/`, [Release Streams & Channels](./release/semver-and-channels.md) |

## Security And Supply Chain

| Area | What it covers | Where to look |
| --- | --- | --- |
| Responsible disclosure | Private vulnerability reporting, security issue routing, breach notification runbook, and breach report template. | `SECURITY.md`, `docs/security/disclosure-policy.md`, `docs/compliance/breach-notification-runbook.md`, `docs/compliance/templates/breach-report-template.md` |
| Dependency and release controls | License policy, pip-audit ignores, SBOM generation, reproducible locks, GitHub Actions ref validation, lockfile drift, repo policy, and doctest-backed examples. | `docs/security/`, `scripts/release/`, `docs/contributing/reproducible-dependencies.md`, `tests/unit/release/`, `tests/unit/test_doctests.py` |
| Privacy-safe diagnostics | PHI-safe progress callbacks, NDJSON errors, active-learning records, hashed examples, explain traces, dataset cards, metadata scrubbing, and no-raw-PHI guidance. | `openmed/multimodal/metadata_scrub.py`, `openmed/training/active_learning.py`, `openmed/eval/dataset_card.py`, `docs/compliance.md`, `docs/security/secret-handling.md` |

## Suggested Reading Order

1. [Quick Start](./getting-started.md) - install plus first inference.
2. [OpenMed v1.6-v1.7 Feature Coverage](./release/v1.6-v1.7-feature-coverage.md) - verify release coverage across docs and examples.
3. [Examples](./examples.md) - runnable notebooks and scripts.
4. [PII Anonymization](./anonymization.md) - de-identification methods and policy workflows.
5. [REST Service](./rest-service.md), [Swift Package](./swift-openmedkit.md), and [Transformers.js Export](./export-transformersjs.md) - deployment surfaces.
6. [Eval Harness & Metrics](./eval-harness.md), [Device Tiers & SLOs](./tiers.md), and [Release Streams & Channels](./release/semver-and-channels.md) - release evidence and operations.
