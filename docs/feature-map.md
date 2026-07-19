# Feature Map & Capabilities

This page inventories the main surfaced capabilities in OpenMed and maps them
back to source modules, docs, and runnable examples. For release-specific
coverage, see
[OpenMed v1.9.1 Release Notes](./release/v1.9.1.md), the
[OpenMed v1.8.0 Release Notes](./release/v1.8.0.md), and the historical
[OpenMed v1.6-v1.7 Feature Coverage](./release/v1.6-v1.7-feature-coverage.md).

## Privacy And De-identification

| Area | What it covers | Where to look |
| --- | --- | --- |
| Policy-aware de-identification | `deidentify`, policy profiles, calibrated thresholds, arbitration, cascade routing, safety sweeps, custom recognizers, and clinical term protection. | `openmed/core/pii.py`, `openmed/core/pipeline.py`, `openmed/core/policy.py`, `openmed/core/clinical_protect.py`, [PII Anonymization](./anonymization.md) |
| Canonical span contracts | Versioned `OpenMedSpan`, schema fingerprints, redaction action schemas, provenance, and compatibility gates. | `openmed/core/schemas/`, [De-identification API](./api/deidentification.md), `examples/v16_policy_audit_release_gates.py` |
| Audit and review evidence | Signed audit reports, reproducibility hashes, review bundles, FHIR `Provenance`/`AuditEvent`, audit diffs, and PHI-safe previews. | `openmed/core/audit.py`, `openmed/core/redaction_preview.py`, `openmed/risk/audit_diff.py`, `openmed/clinical/exporters/fhir/provenance.py`, `examples/v16_policy_audit_release_gates.py` |
| Runtime de-identification features | `DeidentificationResult.to_dataframe`, surrogate vaults, patient-keyed date shifting, format-preserving redaction, minimum-necessary action selection, streaming redaction, explain traces, section stamping, and risk budgets. | `openmed/core/pii.py`, `openmed/core/surrogate_vault.py`, `openmed/core/date_shift.py`, `openmed/core/anonymizer/format_preserve.py`, `openmed/core/redaction_strength.py`, `openmed/core/streaming.py`, `openmed/core/explain.py`, `openmed/risk/budget.py` |
| Multilingual PII | 17 supported PII language codes: ar, de, en, es, fr, he, hi, id, it, ja, ko, nl, pt, ro, te, th, and tr in the model-backed allow-list; locale validators, script detection, date/number normalization, deterministic locale PHI generation, and ID-only national-ID providers for additional locales. | `openmed/core/pii_i18n.py`, `openmed/core/script_detect.py`, `openmed/core/locale_formats.py`, `openmed/core/anonymizer/locales.py`, `openmed/training/synthetic/locale_phi.py`, `examples/pii_multilingual_new_languages.py` |

## Multimodal And Structured Inputs

| Area | What it covers | Where to look |
| --- | --- | --- |
| Multimodal document contract | `ExtractedDocument`, `SourceSpan`, lazy handler registration, source-offset mapping, and dispatcher-based redaction. | `openmed/multimodal/base.py`, `examples/v17_multimodal_browser_interop.py` |
| OCR and scanned documents | Shared OCR result contract, fake test engine, Tesseract, PaddleOCR, EasyOCR, docTR, OCR language selection, and available-engine discovery. | `openmed/multimodal/ocr.py`, `tests/unit/multimodal/test_ocr_engines.py`, `examples/v17_multimodal_browser_interop.py` |
| Documents and metadata | Markdown/AsciiDoc and EPUB extraction, DOCX offset extraction, PPTX slide/table/speaker-notes redaction, source text redaction, image/PDF/DOCX metadata scrubbing, and residual metadata verification. | `openmed/multimodal/documents_docx.py`, `openmed/multimodal/pptx.py`, [PPTX Redaction](./multimodal/pptx-redaction.md), `openmed/multimodal/epub.py`, `openmed/multimodal/metadata_scrub.py` |
| DICOM, contacts, and PDFs | DICOM header de-identification, burned-in pixel OCR redaction, vCard/iCalendar PHI redaction, and redacted-PDF text-layer leakage/fidelity checks. | `openmed/multimodal/dicom.py`, `openmed/multimodal/contacts_calendar.py`, `openmed/multimodal/verify_pdf.py`, [EPUB Extraction](./multimodal/epub-extraction.md) |
| Chat logs and tabular data | JSONL chat-log redaction with speaker pseudonymization, CSV/TSV PHI column classification, column actions, and PHI-safe manifests. | `openmed/multimodal/chatlog_jsonl.py`, `openmed/multimodal/tabular_csv.py`, `examples/v17_multimodal_browser_interop.py` |

## Health-data Interop

| Area | What it covers | Where to look |
| --- | --- | --- |
| FHIR helpers | Deterministic R4 Bundles, stable `urn:uuid` references, `OperationOutcome`, `Provenance`, `AuditEvent`, CodeableConcept builders/checkers, SMART-on-FHIR bulk ingestion, and coding provenance. | `openmed/clinical/exporters/fhir/`, `openmed/clinical/exporters/codeable_concept.py`, `openmed/clinical/exporters/codeable_concept_check.py`, `openmed/service/smart_backend.py`, [FHIR Interop Helpers](./fhir-interop.md) |
| OMOP and CDM loading | Deterministic note-to-CDM extraction and OMOP CDM loader foundations. | `openmed/interop/cdm_etl.py`, `openmed/interop/omop/cdm_loader.py` |
| FHIR operations and bulk export | `$de-identify` resource/Bundle wrappers and FHIR Bulk NDJSON de-identification summaries. | `openmed/interop/fhir_operations.py`, `openmed/interop/fhir_bulk.py`, `examples/v17_multimodal_browser_interop.py` |
| HL7 v2 and CDA/C-CDA | HL7 v2 segment/field redaction, CDA/C-CDA XML de-identification, and multimodal XML dispatch. | `openmed/interop/hl7v2.py`, `openmed/interop/cda.py`, [HL7 v2 De-identification](./hl7v2-deidentification.md) |
| Optional adapters | Presidio, PHILTER, pyDeid, GLiNER-BioMed, LangChain, and spaCy adapter surfaces. | `openmed/interop/`, [LangChain Redaction Wrapper](./integrations-langchain.md), [spaCy Pipeline Component](./spacy-component.md) |

## Clinical Extraction

| Area | What it covers | Where to look |
| --- | --- | --- |
| Clinical context | Negation, temporality, uncertainty, experiencer, section-aware priors, context offsets, multilingual cue lexicons, and negation-scope boundaries. | `openmed/clinical/context.py`, `openmed/clinical/negation_scope.py`, `openmed/clinical/lexicons/context_cues.py`, [Clinical Context & Extraction Depth](./clinical/context-and-extraction.md) |
| Clinical normalization | Labs, vitals, medication sigs, problem-list status, summary cards, assertion graphs, event frames, terminology normalization, UCUM units, and flat-table export. | `openmed/clinical/`, `openmed/clinical/assertion_graph.py`, `openmed/clinical/events/`, `openmed/clinical/normalization/`, `openmed/clinical/units/`, [Clinical Context & Extraction Depth](./clinical/context-and-extraction.md) |
| Clinical grounding | Free vocabulary grounding, lexical aliases, RxNorm, ICD-10-CM, HPO, and CodeableConcept export. | `openmed/clinical/grounding/`, `openmed/clinical/exporters/codeable_concept.py` |
| Zero-shot domains | GLiNER label maps, clinical NER families, cardiology, dermatology, ophthalmology, microbiology, nutrition, anesthesia, endocrinology, gastroenterology, clinical genomics, and routing metadata. | `openmed/zero_shot/`, `openmed/ner/families/`, [Zero-shot Toolkit](./zero-shot-ner.md), [Zero-shot NER How-to](./zero-shot-howto.md) |

## Models, Backends, And Browser Export

| Area | What it covers | Where to look |
| --- | --- | --- |
| Model registry and manifests | Curated metadata, manifest schema validation, manifest diffs, model recommendations, model card generation, and publishing metadata. | `models.jsonl`, `openmed/core/model_registry.py`, `openmed/core/manifest_schema.py`, `openmed/core/manifest_diff.py`, `openmed/core/model_card.py`, [Model Registry](./model-registry.md), [Model Manifest](./model-manifest.md) |
| One-call inference | `analyze_text`, typed `AnalyzeResult`, validation, grouping, and dict/JSON/HTML/CSV formatting. | `openmed/__init__.py`, `openmed/core/results.py`, [Analyze Text Helper](./analyze-text.md) |
| Loader and batch processing | Model loading, cache reuse, tokenizer caching, batch `analyze_text`, batch PII extraction, and ordered de-identification output. | `openmed/core/models.py`, `openmed/processing/tokenizer_cache.py`, `openmed/processing/batch.py`, [ModelLoader & Pipelines](./model-loader.md), [Batch Processing](./batch-processing.md) |
| Apple and mobile runtimes | Python MLX, Laneformer MLX-LM, Swift OpenMedKit, Kotlin/Android OpenMedKit, React Native bridge, Core ML export, MLX quantized export, mobile benchmarks, and PyTorch MPS selection/tuning. | `openmed/mlx/`, `openmed/coreml/`, `openmed/torch/`, `swift/OpenMedKit`, `android/openmedkit`, `js/openmedkit-react-native`, [MLX Backend](./mlx-backend.md), [Swift Package](./swift-openmedkit.md), [Android Span Parity](./android-parity.md), [CoreML Packaging](./coreml-export.md) |
| Quantized and browser formats | AWQ, GPTQ, bitsandbytes loading, ONNX/WebGPU, ONNX Runtime Web, ORT Mobile, OpenVINO, Transformers.js bundle export, and ONNX/quantized manifest metadata. | `openmed/onnx/`, `js/openmedkit-web/`, [Android ONNX Export](./export-onnx-android.md), [ONNX Runtime Web Loader](./runtimes/onnxruntime-web.md), [OpenVINO Runtime](./runtimes/openvino.md), [Transformers.js Export](./export-transformersjs.md) |
| Training infrastructure | DAPT corpus assembly, Mode-A distillation, DirectID contracts, hard-negative sampling, active learning, adjudication, and span-relation graph decoding. | `openmed/training/`, `openmed/core/decoding/graph.py` |

## Service, Clients, And Operations

| Area | What it covers | Where to look |
| --- | --- | --- |
| REST and gRPC service | FastAPI endpoints, shared runtime, API-key/JWT auth, no-PHI logging, tracing, gRPC, async jobs, webhooks, model warm pools, dynamic batching, request coalescing, rate/concurrency limits, trusted hosts, CORS, `/livez`, `/readyz`, and metrics. | `openmed/service/`, [REST Service](./rest-service.md), [REST Authentication](./serving/authentication.md), [gRPC Service](./serving/grpc.md), [Async REST Jobs & Webhooks](./serving/async-jobs.md), [REST Tracing](./serving/tracing.md) |
| Typed clients | Python service client and TypeScript REST client parity. | `openmed/service/client.py`, `clients/typescript/`, `clients/typescript/README.md` |
| Streaming and connectors | Core incremental de-identification, Kafka/Pulsar streaming, Spark structured streaming, object storage, Dask, DuckDB, lakehouse, columnar redaction, and warehouse transformation packages. | `openmed/processing/`, `openmed/integrations/`, `openmed/interop/duckdb_udf.py`, [Columnar Redactor](./integrations/columnar-redactor.md), [Lakehouse Table Redaction](./integrations/lakehouse-redaction.md), [Dask DataFrame De-identification](./integrations/dask.md) |
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
| Dependency and release controls | License policy, pip-audit ignores, SBOM generation, container SBOMs, image signing, SLSA provenance, secret scanning, vulnerability scanning, reproducible locks, GitHub Actions ref validation, lockfile drift, repo policy, and doctest-backed examples. | `docs/security/`, `docs/supply-chain/`, `scripts/release/`, `scripts/security/`, `docs/contributing/reproducible-dependencies.md`, `tests/unit/release/` |
| Privacy-safe diagnostics | PHI-safe progress callbacks, NDJSON errors, active-learning records, hashed examples, explain traces, dataset cards, metadata scrubbing, and no-raw-PHI guidance. | `openmed/multimodal/metadata_scrub.py`, `openmed/training/active_learning.py`, `openmed/eval/dataset_card.py`, `docs/compliance.md`, `docs/security/secret-handling.md` |

## Suggested Reading Order

1. [Quick Start](./getting-started.md) - install plus first inference.
2. [OpenMed 1.9.1 Release Notes](./release/v1.9.1.md) - review the latest patch fixes, release coverage, and migration notes.
3. [Examples](./examples.md) - runnable notebooks and scripts.
4. [PII Anonymization](./anonymization.md) - de-identification methods and policy workflows.
5. [REST Service](./rest-service.md), [Swift Package](./swift-openmedkit.md), and [Transformers.js Export](./export-transformersjs.md) - deployment surfaces.
6. [Eval Harness & Metrics](./eval-harness.md), [Device Tiers & SLOs](./tiers.md), and [Release Streams & Channels](./release/semver-and-channels.md) - release evidence and operations.
