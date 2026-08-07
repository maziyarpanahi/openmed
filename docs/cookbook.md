# Cookbook

Start with the task you need to complete, then open the linked script, notebook,
or recipe. The examples use synthetic data unless their own documentation says
otherwise. Run scripts from the repository root so their relative paths resolve
consistently.

## De-identify text and datasets

| Goal | Asset | What it covers |
| --- | --- | --- |
| De-identify a CSV column | [Deidentification Cookbook notebook](https://github.com/maziyarpanahi/openmed/blob/master/examples/notebooks/Deidentification_Cookbook.ipynb) | A list/CSV recipe that adds de-identified output without changing the source values. |
| Batch-redact a folder of text files | [Deidentification Cookbook notebook](https://github.com/maziyarpanahi/openmed/blob/master/examples/notebooks/Deidentification_Cookbook.ipynb) | `BatchProcessor.process_directory()` over synthetic `.txt` notes with mirrored redacted output. |
| Explore the complete PII API | [PII Detection Complete Guide notebook](https://github.com/maziyarpanahi/openmed/blob/master/examples/notebooks/PII_Detection_Complete_Guide.ipynb) | PII extraction, de-identification methods, entity review, and output handling. |
| Process a batch of notes | [pii_batch_processing.py](https://github.com/maziyarpanahi/openmed/blob/master/examples/pii_batch_processing.py) | `BatchProcessor` extraction and deterministic replacement over synthetic notes. |
| Run the first redaction-to-FHIR flow | [first_five_minutes_redact_extract_fhir.py](https://github.com/maziyarpanahi/openmed/blob/master/examples/first_five_minutes_redact_extract_fhir.py) | Redaction, deterministic clinical extraction, and FHIR Bundle assembly. |
| Exercise a bundled evaluation fixture | [datasets_walkthrough.py](https://github.com/maziyarpanahi/openmed/blob/master/examples/datasets_walkthrough.py) | Offline-first loading of synthetic fixtures through `extract_pii` and `deidentify`. |
| Protect retrieval context | [redaction_preserving_retrieval.py](https://github.com/maziyarpanahi/openmed/blob/master/examples/redaction_preserving_retrieval.py) | Redaction before retrieval and controlled use of protected context. |
| Redact before an external model call | [privacy_gateway_quickstart.py](https://github.com/maziyarpanahi/openmed/blob/master/examples/privacy_gateway_quickstart.py) | A privacy boundary with redaction before egress and safe re-identification afterward. |

## Process documents and messages

| Goal | Asset | What it covers |
| --- | --- | --- |
| Redact helpdesk SMS exports | [sms_deid_helpdesk_logs.py](https://github.com/maziyarpanahi/openmed/blob/master/examples/sms_deid_helpdesk_logs.py) | RapidPro JSON and CSV input, contact pseudonyms, coarse timestamps, and bounded batches. |
| De-identify community health forms | [chw_form_deid.py](https://github.com/maziyarpanahi/openmed/blob/master/examples/chw_form_deid.py) | Local ODK, CommCare, and KoBoToolbox JSON or CSV exports. |
| Inspect multimodal and document adapters | [v17_multimodal_browser_interop.py](https://github.com/maziyarpanahi/openmed/blob/master/examples/v17_multimodal_browser_interop.py) | OCR contracts, source offsets, chat JSONL, CSV manifests, FHIR, HL7 v2, and browser bundles. |
| Segment long text into batches | [Sentence Detection and Batching notebook](https://github.com/maziyarpanahi/openmed/blob/master/examples/notebooks/Sentence_Detection_Batching.ipynb) | Sentence segmentation, batched inference, and projection back to source paragraphs. |

## Select and compare models

| Goal | Asset | What it covers |
| --- | --- | --- |
| Compare PII models and methods | [pii_model_comparison.py](https://github.com/maziyarpanahi/openmed/blob/master/examples/pii_model_comparison.py) | Registry discovery, model-size tradeoffs, shared inputs, and de-identification methods. |
| Tour zero-shot NER | [Zero-Shot NER Tour notebook](https://github.com/maziyarpanahi/openmed/blob/master/examples/notebooks/ZeroShot_NER_Tour.ipynb) | GLiNER indexing, label defaults, inference, and BIO/BILOU span conversion. |
| Inspect medical tokenization | [Medical Tokenizer Demo notebook](https://github.com/maziyarpanahi/openmed/blob/master/examples/notebooks/Medical_Tokenizer_Demo.ipynb) | Medical-aware tokenization and token-to-source alignment. |
| Compare tokenizer behavior | [Medical Tokenizer Benchmark notebook](https://github.com/maziyarpanahi/openmed/blob/master/examples/notebooks/Medical_Tokenizer_Benchmark.ipynb) | Reproducible tokenizer comparisons over clinical text. |

## Extract clinical entities

| Goal | Asset | What it covers |
| --- | --- | --- |
| Run disease, pharmaceutical, and oncology NER | [clinical_ner_families.py](https://github.com/maziyarpanahi/openmed/blob/master/examples/clinical_ner_families.py) | Registry-based family selection, offline defaults, and labeled-span output. |
| Build a DataFrame extraction flow | [clinical_extraction_dataframe_api.py](https://github.com/maziyarpanahi/openmed/blob/master/examples/notebooks/clinical_extraction_dataframe_api.py) | Clinical extraction through the tabular API. |
| Start from a guided notebook | [Getting Started notebook](https://github.com/maziyarpanahi/openmed/blob/master/examples/notebooks/getting_started.ipynb) | Installation, registry exploration, and a first `analyze_text` call. |

## Run multilingual workflows

| Goal | Asset | What it covers |
| --- | --- | --- |
| De-identify Chinese, Hindi, and Hinglish notes | [Chinese and Hindi De-identification Tour notebook](https://github.com/maziyarpanahi/openmed/blob/master/examples/notebooks/Chinese_Hindi_Deid_Tour.ipynb) | Deterministic de-identification, structured review, UTF-8 output, and leak assertions. |
| Compare multilingual PII behavior | [Multilingual PII Detection Guide notebook](https://github.com/maziyarpanahi/openmed/blob/master/examples/notebooks/Multilingual_PII_Detection_Guide.ipynb) | Language-aware PII detection and de-identification. |
| Exercise additional language packs | [pii_multilingual_new_languages.py](https://github.com/maziyarpanahi/openmed/blob/master/examples/pii_multilingual_new_languages.py) | Registry entries, locale-specific recognizers, and optional model-backed extraction. |

## Serve and integrate OpenMed

| Goal | Asset | What it covers |
| --- | --- | --- |
| Start and call the REST service | [REST API Recipes](rest-recipes.md) | Service startup, health checks, analysis, PII extraction, de-identification, and production safeguards. |
| Build a local interactive demo | [gradio_deid_app.py](https://github.com/maziyarpanahi/openmed/blob/master/examples/gradio_deid_app.py) | A small Gradio interface for synthetic text and multiple de-identification methods. |
| Redact warehouse columns with dbt | [dbt de-identification example](https://github.com/maziyarpanahi/openmed/blob/master/examples/dbt-deidentify/README.md) | Redaction macros, synthetic seed data, and a redacted staging model. |
| Redact a Spark stream | [Spark streaming example](https://github.com/maziyarpanahi/openmed/blob/master/examples/spark-streaming/README.md) | Structured Streaming de-identification over synthetic records. |
| Add structured log redaction | [log redaction example](https://github.com/maziyarpanahi/openmed/blob/master/examples/log-redaction/README.md) | Logging integration that removes sensitive values before emission. |

For shorter copy-and-paste snippets and feature-specific documentation, see
[Examples and Copy/Paste Recipes](examples.md).
