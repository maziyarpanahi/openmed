# Cookbook

Start with the task you need to complete, then open the linked script,
notebook, or copy-ready documentation. The examples use synthetic data unless
their own instructions say otherwise. Use only authorized data, and keep raw
PHI out of logs, notebook outputs, and shared artifacts.

For a feature-oriented tour with inline snippets, see
[Examples & Copy/Paste Recipes](./examples.md).

## PII detection and de-identification

| Goal | Open this asset |
| --- | --- |
| Learn the complete PII workflow, from detection through de-identification | [`PII_Detection_Complete_Guide.ipynb`](https://github.com/maziyarpanahi/openmed/blob/master/examples/notebooks/PII_Detection_Complete_Guide.ipynb) |
| De-identify a list or CSV, batch a folder, or run a reversible replacement round trip | [`Deidentification_Cookbook.ipynb`](https://github.com/maziyarpanahi/openmed/blob/master/examples/notebooks/Deidentification_Cookbook.ipynb) |
| Batch PII extraction or de-identification in Python | [`pii_batch_processing.py`](https://github.com/maziyarpanahi/openmed/blob/master/examples/pii_batch_processing.py) |
| Compare PII models on the same synthetic text | [`pii_model_comparison.py`](https://github.com/maziyarpanahi/openmed/blob/master/examples/pii_model_comparison.py) |
| Evaluate multilingual PII detection and model selection | [`Multilingual_PII_Detection_Guide.ipynb`](https://github.com/maziyarpanahi/openmed/blob/master/examples/notebooks/Multilingual_PII_Detection_Guide.ipynb) and [`pii_multilingual_new_languages.py`](https://github.com/maziyarpanahi/openmed/blob/master/examples/pii_multilingual_new_languages.py) |
| Generate consistent, seeded, locale-aware replacement values | [`obfuscation_demo.py`](https://github.com/maziyarpanahi/openmed/blob/master/examples/obfuscation_demo.py) |
| Try de-identification in a small Gradio app | [`gradio_deid_app.py`](https://github.com/maziyarpanahi/openmed/blob/master/examples/gradio_deid_app.py) |
| Use one Privacy Filter API across MLX and PyTorch | [`privacy_filter_unified.py`](https://github.com/maziyarpanahi/openmed/blob/master/examples/privacy_filter_unified.py) |
| Compare Privacy Filter model families in an interactive app | [`privacy_filter_book/README.md`](https://github.com/maziyarpanahi/openmed/blob/master/examples/privacy_filter_book/README.md) and [`privacy_filter_book/app.py`](https://github.com/maziyarpanahi/openmed/blob/master/examples/privacy_filter_book/app.py) |
| Run a focused Privacy Filter studio | [`privacy_filter_studio/README.md`](https://github.com/maziyarpanahi/openmed/blob/master/examples/privacy_filter_studio/README.md) and [`privacy_filter_studio/app.py`](https://github.com/maziyarpanahi/openmed/blob/master/examples/privacy_filter_studio/app.py) |
| Compare multilingual Privacy Filter results side by side | [`privacy_filter_multilingual_studio/README.md`](https://github.com/maziyarpanahi/openmed/blob/master/examples/privacy_filter_multilingual_studio/README.md) and [`privacy_filter_multilingual_studio/app.py`](https://github.com/maziyarpanahi/openmed/blob/master/examples/privacy_filter_multilingual_studio/app.py) |
| Exercise the public PII API with a bundled synthetic fixture | [`datasets_walkthrough.py`](https://github.com/maziyarpanahi/openmed/blob/master/examples/datasets_walkthrough.py) |

## Clinical NER and model exploration

| Goal | Open this asset |
| --- | --- |
| Learn installation, registry exploration, and a first `analyze_text` call | [`getting_started.ipynb`](https://github.com/maziyarpanahi/openmed/blob/master/examples/notebooks/getting_started.ipynb) |
| Compare disease, pharmaceutical, and oncology NER families | [`clinical_ner_families.py`](https://github.com/maziyarpanahi/openmed/blob/master/examples/clinical_ner_families.py) |
| Explore zero-shot NER and custom entity labels | [`ZeroShot_NER_Tour.ipynb`](https://github.com/maziyarpanahi/openmed/blob/master/examples/notebooks/ZeroShot_NER_Tour.ipynb) |
| Run GLiNER zero-shot NER through MLX | [`mlx_gliner_zero_shot_ner.py`](https://github.com/maziyarpanahi/openmed/blob/master/examples/mlx_gliner_zero_shot_ner.py) |
| Run a converted token-classification model through MLX | [`mlx_token_classification_ner.py`](https://github.com/maziyarpanahi/openmed/blob/master/examples/mlx_token_classification_ner.py) |
| Export registry operations as agent-framework tool definitions | [`agent_tools_quickstart.py`](https://github.com/maziyarpanahi/openmed/blob/master/examples/agent_tools_quickstart.py) |
| Render clinical extraction results as a dataframe | [`clinical_extraction_dataframe_api.py`](https://github.com/maziyarpanahi/openmed/blob/master/examples/notebooks/clinical_extraction_dataframe_api.py) |

## Batching and tokenization

| Goal | Open this asset |
| --- | --- |
| Segment long text, batch sentences, and align predictions to paragraphs | [`Sentence_Detection_Batching.ipynb`](https://github.com/maziyarpanahi/openmed/blob/master/examples/notebooks/Sentence_Detection_Batching.ipynb) |
| Learn medical-aware tokenization interactively | [`Medical_Tokenizer_Demo.ipynb`](https://github.com/maziyarpanahi/openmed/blob/master/examples/notebooks/Medical_Tokenizer_Demo.ipynb) |
| Benchmark medical tokenization performance | [`Medical_Tokenizer_Benchmark.ipynb`](https://github.com/maziyarpanahi/openmed/blob/master/examples/notebooks/Medical_Tokenizer_Benchmark.ipynb) |
| Align custom tokens back to encoder outputs | [`custom_tokenize_alignment.py`](https://github.com/maziyarpanahi/openmed/blob/master/examples/custom_tokenizer/custom_tokenize_alignment.py) |
| Compare encoder output with medical remapping on and off | [`compare_medical_remap.py`](https://github.com/maziyarpanahi/openmed/blob/master/examples/custom_tokenizer/compare_medical_remap.py) |
| Evaluate tokenizer strategies on a clinical corpus | [`eval_tokenization_comparison.py`](https://github.com/maziyarpanahi/openmed/blob/master/examples/custom_tokenizer/eval_tokenization_comparison.py) |
| Review the custom-tokenizer setup and commands | [`custom_tokenizer/README.md`](https://github.com/maziyarpanahi/openmed/blob/master/examples/custom_tokenizer/README.md) |

## Services, pipelines, and safeguards

| Goal | Open this asset |
| --- | --- |
| Start the REST service and call health, analysis, PII, and de-identification endpoints | [REST API Recipes](./rest-recipes.md) |
| Redact before an external model call and restore identifiers locally | [`privacy_gateway_quickstart.py`](https://github.com/maziyarpanahi/openmed/blob/master/examples/privacy_gateway_quickstart.py) |
| Redact selected fields in NDJSON log events | [`log-redaction/README.md`](https://github.com/maziyarpanahi/openmed/blob/master/examples/log-redaction/README.md) |
| De-identify warehouse columns with dbt macros | [`dbt-deidentify/README.md`](https://github.com/maziyarpanahi/openmed/blob/master/examples/dbt-deidentify/README.md) |
| De-identify Spark Structured Streaming micro-batches | [`spark-streaming/README.md`](https://github.com/maziyarpanahi/openmed/blob/master/examples/spark-streaming/README.md) |
| Add a de-identification screen to a React Native app | [`react-native/RedactScreen.tsx`](https://github.com/maziyarpanahi/openmed/blob/master/examples/react-native/RedactScreen.tsx) |
| Review policy, audit, and release-evidence safeguards offline | [`v16_policy_audit_release_gates.py`](https://github.com/maziyarpanahi/openmed/blob/master/examples/v16_policy_audit_release_gates.py) |

Grounding, FHIR, and multimodal workflows are maintained in their dedicated
guides and are intentionally outside this cookbook's scope.
