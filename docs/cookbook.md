# OpenMed Cookbook

Start with the task you need to complete, then open the linked runnable script,
notebook, or copy-ready recipe. Repository examples use synthetic inputs by
default; review any optional model download or external-service boundary before
using the same workflow with real data.

## Pick a task

| Goal | Start here | What it demonstrates |
| --- | --- | --- |
| Redact one clinical note | [De-identification cookbook notebook](https://github.com/maziyarpanahi/openmed/blob/master/examples/notebooks/Deidentification_Cookbook.ipynb) | Masking, replacement, hashing, and rich notebook rendering. |
| De-identify a folder or CSV in batches | [Batch processing script](https://github.com/maziyarpanahi/openmed/blob/master/examples/pii_batch_processing.py) | Bounded extraction and de-identification with `BatchProcessor`. |
| Compare PII models | [Model comparison script](https://github.com/maziyarpanahi/openmed/blob/master/examples/pii_model_comparison.py) | Runs several model choices against shared synthetic text. |
| Explore clinical NER families | [Clinical NER families script](https://github.com/maziyarpanahi/openmed/blob/master/examples/clinical_ner_families.py) | Biomedical and clinical model-family selection. |
| Build a REST workflow | [REST recipes](./rest-recipes.md) | Copy-ready health, extraction, de-identification, batch, and streaming requests. |
| Redact, extract, and assemble FHIR | [First-five-minutes script](https://github.com/maziyarpanahi/openmed/blob/master/examples/first_five_minutes_redact_extract_fhir.py) | A synthetic local redaction-to-FHIR path. |
| Evaluate a model offline | [Evaluation walkthrough notebook](https://github.com/maziyarpanahi/openmed/blob/master/examples/notebooks/04_eval_walkthrough.ipynb) | Bundled synthetic fixtures and offline evaluation. |
| Process Chinese, Hindi, or Hinglish | [Multilingual tour notebook](https://github.com/maziyarpanahi/openmed/blob/master/examples/notebooks/Chinese_Hindi_Deid_Tour.ipynb) | Script-aware de-identification and zero-leak assertions. |
| Protect an agent or retrieval pipeline | [Privacy gateway script](https://github.com/maziyarpanahi/openmed/blob/master/examples/privacy_gateway_quickstart.py) | Local redaction before an explicit external boundary. |
| Review structured-data release risk | [Structured release-risk script](https://github.com/maziyarpanahi/openmed/blob/master/examples/structured_release_risk.py) | Advisory k/l/t policy review and aggregate evidence. |

## Runnable script index

### Privacy and de-identification

| Use when you want to... | Script |
| --- | --- |
| Batch extraction or de-identification | [`examples/pii_batch_processing.py`](https://github.com/maziyarpanahi/openmed/blob/master/examples/pii_batch_processing.py) |
| Compare model outputs | [`examples/pii_model_comparison.py`](https://github.com/maziyarpanahi/openmed/blob/master/examples/pii_model_comparison.py) |
| Exercise recently added language support | [`examples/pii_multilingual_new_languages.py`](https://github.com/maziyarpanahi/openmed/blob/master/examples/pii_multilingual_new_languages.py) |
| De-identify a fabricated Chinese note | [`examples/deid_chinese_clinical_note.py`](https://github.com/maziyarpanahi/openmed/blob/master/examples/deid_chinese_clinical_note.py) |
| De-identify fabricated Hindi and Hinglish notes | [`examples/deid_hindi_hinglish_note.py`](https://github.com/maziyarpanahi/openmed/blob/master/examples/deid_hindi_hinglish_note.py) |
| Redact RapidPro-style or CSV helpdesk logs | [`examples/sms_deid_helpdesk_logs.py`](https://github.com/maziyarpanahi/openmed/blob/master/examples/sms_deid_helpdesk_logs.py) |
| Redact ODK, CommCare, or KoBoToolbox exports | [`examples/chw_form_deid.py`](https://github.com/maziyarpanahi/openmed/blob/master/examples/chw_form_deid.py) |
| Try language-agnostic obfuscation | [`examples/obfuscation_demo.py`](https://github.com/maziyarpanahi/openmed/blob/master/examples/obfuscation_demo.py) |
| Compare the unified MLX and PyTorch privacy API | [`examples/privacy_filter_unified.py`](https://github.com/maziyarpanahi/openmed/blob/master/examples/privacy_filter_unified.py) |
| Launch the optional synthetic Gradio demo | [`examples/gradio_deid_app.py`](https://github.com/maziyarpanahi/openmed/blob/master/examples/gradio_deid_app.py) |
| Walk through a bundled synthetic dataset | [`examples/datasets_walkthrough.py`](https://github.com/maziyarpanahi/openmed/blob/master/examples/datasets_walkthrough.py) |

### Clinical extraction, grounding, and exchange

| Use when you want to... | Script |
| --- | --- |
| Compare clinical and biomedical NER families | [`examples/clinical_ner_families.py`](https://github.com/maziyarpanahi/openmed/blob/master/examples/clinical_ner_families.py) |
| Redact, extract, and build a FHIR Bundle | [`examples/first_five_minutes_redact_extract_fhir.py`](https://github.com/maziyarpanahi/openmed/blob/master/examples/first_five_minutes_redact_extract_fhir.py) |
| Export grounded spans for FHIR and OMOP workflows | [`examples/interop_fhir_export.py`](https://github.com/maziyarpanahi/openmed/blob/master/examples/interop_fhir_export.py) |
| Ground a synthetic mention offline | [`examples/offline_grounding.py`](https://github.com/maziyarpanahi/openmed/blob/master/examples/offline_grounding.py) |
| Prepare a local OpenMRS de-identified handoff | [`examples/openmrs_deid_handoff.py`](https://github.com/maziyarpanahi/openmed/blob/master/examples/openmrs_deid_handoff.py) |
| Build a de-identified DHIS2 district export | [`examples/dhis2_district_export.py`](https://github.com/maziyarpanahi/openmed/blob/master/examples/dhis2_district_export.py) |

### Models, agents, and local pipelines

| Use when you want to... | Script |
| --- | --- |
| Run MLX token classification | [`examples/mlx_token_classification_ner.py`](https://github.com/maziyarpanahi/openmed/blob/master/examples/mlx_token_classification_ner.py) |
| Run MLX GLiNER zero-shot NER | [`examples/mlx_gliner_zero_shot_ner.py`](https://github.com/maziyarpanahi/openmed/blob/master/examples/mlx_gliner_zero_shot_ner.py) |
| Render registry tools for agent frameworks | [`examples/agent_tools_quickstart.py`](https://github.com/maziyarpanahi/openmed/blob/master/examples/agent_tools_quickstart.py) |
| Add a privacy boundary to a graph flow | [`examples/graph_orchestration_privacy.py`](https://github.com/maziyarpanahi/openmed/blob/master/examples/graph_orchestration_privacy.py) |
| Redact before an external model call | [`examples/privacy_gateway_quickstart.py`](https://github.com/maziyarpanahi/openmed/blob/master/examples/privacy_gateway_quickstart.py) |
| Preserve retrieval utility after redaction | [`examples/redaction_preserving_retrieval.py`](https://github.com/maziyarpanahi/openmed/blob/master/examples/redaction_preserving_retrieval.py) |
| Protect a local search pipeline | [`examples/search_pipeline_privacy.py`](https://github.com/maziyarpanahi/openmed/blob/master/examples/search_pipeline_privacy.py) |

### Policy, risk, onboarding, and release evidence

| Use when you want to... | Script |
| --- | --- |
| Review a structured release | [`examples/structured_release_risk.py`](https://github.com/maziyarpanahi/openmed/blob/master/examples/structured_release_risk.py) |
| Compare a release with a reference population | [`examples/structured_population_risk.py`](https://github.com/maziyarpanahi/openmed/blob/master/examples/structured_population_risk.py) |
| Exercise policy, audit, and release gates | [`examples/v16_policy_audit_release_gates.py`](https://github.com/maziyarpanahi/openmed/blob/master/examples/v16_policy_audit_release_gates.py) |
| Exercise multimodal, interop, and browser exports | [`examples/v17_multimodal_browser_interop.py`](https://github.com/maziyarpanahi/openmed/blob/master/examples/v17_multimodal_browser_interop.py) |
| Warm a mirror-backed cache, then work offline | [`examples/onboarding_china_mirrors.py`](https://github.com/maziyarpanahi/openmed/blob/master/examples/onboarding_china_mirrors.py) |
| Review an India DPDP-aware workflow | [`examples/onboarding_india_dpdp.py`](https://github.com/maziyarpanahi/openmed/blob/master/examples/onboarding_india_dpdp.py) |

## Notebook index

| Goal | Notebook or notebook companion |
| --- | --- |
| First redaction workflow | [`examples/notebooks/01_quickstart_redaction.ipynb`](https://github.com/maziyarpanahi/openmed/blob/master/examples/notebooks/01_quickstart_redaction.ipynb) |
| Batch a synthetic dataset | [`examples/notebooks/02_batch_dataset.ipynb`](https://github.com/maziyarpanahi/openmed/blob/master/examples/notebooks/02_batch_dataset.ipynb) |
| Export a deterministic FHIR Bundle | [`examples/notebooks/03_fhir_export.ipynb`](https://github.com/maziyarpanahi/openmed/blob/master/examples/notebooks/03_fhir_export.ipynb) |
| Run the offline evaluation walkthrough | [`examples/notebooks/04_eval_walkthrough.ipynb`](https://github.com/maziyarpanahi/openmed/blob/master/examples/notebooks/04_eval_walkthrough.ipynb) |
| Tour Chinese, Hindi, and Hinglish de-identification | [`examples/notebooks/Chinese_Hindi_Deid_Tour.ipynb`](https://github.com/maziyarpanahi/openmed/blob/master/examples/notebooks/Chinese_Hindi_Deid_Tour.ipynb) |
| Explore masking, replacement, and hashing | [`examples/notebooks/Deidentification_Cookbook.ipynb`](https://github.com/maziyarpanahi/openmed/blob/master/examples/notebooks/Deidentification_Cookbook.ipynb) |
| Follow the original getting-started notebook | [`examples/notebooks/getting_started.ipynb`](https://github.com/maziyarpanahi/openmed/blob/master/examples/notebooks/getting_started.ipynb) |
| Benchmark the medical tokenizer | [`examples/notebooks/Medical_Tokenizer_Benchmark.ipynb`](https://github.com/maziyarpanahi/openmed/blob/master/examples/notebooks/Medical_Tokenizer_Benchmark.ipynb) |
| Explore medical tokenization interactively | [`examples/notebooks/Medical_Tokenizer_Demo.ipynb`](https://github.com/maziyarpanahi/openmed/blob/master/examples/notebooks/Medical_Tokenizer_Demo.ipynb) |
| Review multilingual PII detection | [`examples/notebooks/Multilingual_PII_Detection_Guide.ipynb`](https://github.com/maziyarpanahi/openmed/blob/master/examples/notebooks/Multilingual_PII_Detection_Guide.ipynb) |
| Follow the complete PII detection guide | [`examples/notebooks/PII_Detection_Complete_Guide.ipynb`](https://github.com/maziyarpanahi/openmed/blob/master/examples/notebooks/PII_Detection_Complete_Guide.ipynb) |
| Segment and batch sentences | [`examples/notebooks/Sentence_Detection_Batching.ipynb`](https://github.com/maziyarpanahi/openmed/blob/master/examples/notebooks/Sentence_Detection_Batching.ipynb) |
| Tour zero-shot NER | [`examples/notebooks/ZeroShot_NER_Tour.ipynb`](https://github.com/maziyarpanahi/openmed/blob/master/examples/notebooks/ZeroShot_NER_Tour.ipynb) |
| Use the clinical extraction DataFrame API | [`examples/notebooks/clinical_extraction_dataframe_api.py`](https://github.com/maziyarpanahi/openmed/blob/master/examples/notebooks/clinical_extraction_dataframe_api.py) |

For narrative snippets and integration-specific examples, continue to
[Examples & Copy/Paste Recipes](./examples.md) and the
[Notebook Gallery](./examples/notebook-gallery.md).
