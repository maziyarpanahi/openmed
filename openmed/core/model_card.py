"""Render OpenMed model cards from manifest rows."""

from __future__ import annotations

from pathlib import Path
from typing import Any, Sequence

from openmed.__about__ import __version__

DEFAULT_ARXIV = "2508.01630"

_MODEL_LANGUAGES = {
    "Arabic": "ar",
    "Bengali": "bn",
    "Chinese": "zh",
    "Dutch": "nl",
    "French": "fr",
    "German": "de",
    "Hindi": "hi",
    "Italian": "it",
    "Japanese": "ja",
    "Korean": "ko",
    "Portuguese": "pt",
    "Spanish": "es",
    "Telugu": "te",
    "Turkish": "tr",
    "Vietnamese": "vi",
}

_LANGUAGE_NAMES = {
    "ar": "Arabic",
    "bn": "Bengali",
    "de": "German",
    "en": "English",
    "es": "Spanish",
    "fr": "French",
    "hi": "Hindi",
    "it": "Italian",
    "ja": "Japanese",
    "ko": "Korean",
    "nl": "Dutch",
    "pt": "Portuguese",
    "te": "Telugu",
    "tr": "Turkish",
    "vi": "Vietnamese",
    "zh": "Chinese",
}

_NER_CAPABILITIES = (
    (
        "bloodcancerdetect",
        "Blood Cancer NER",
        "Blood cancer",
        "extracting blood cancer mentions",
        "The patient was diagnosed with acute myeloid leukemia.",
    ),
    (
        "anatomydetect",
        "Anatomy NER",
        "Anatomy",
        "extracting anatomy mentions",
        "The biopsy was taken from the left lung.",
    ),
    (
        "chemicaldetect",
        "Chemical NER",
        "Chemical",
        "extracting chemical mentions",
        "The sample was treated with sodium chloride.",
    ),
    (
        "diseasedetect",
        "Disease NER",
        "Disease",
        "extracting disease and condition mentions",
        "The patient has chronic myeloid leukemia.",
    ),
    (
        "dnadetect",
        "DNA NER",
        "DNA",
        "extracting DNA and gene mentions",
        "A BRCA1 mutation was detected in the tumor sample.",
    ),
    (
        "genomedetect",
        "Genome NER",
        "Genome",
        "extracting genome mentions",
        "Whole-genome sequencing identified a pathogenic variant.",
    ),
    (
        "genomicdetect",
        "Genomic NER",
        "Genomic",
        "extracting genomic concepts",
        "Genomic analysis identified a pathogenic variant.",
    ),
    (
        "oncologydetect",
        "Oncology NER",
        "Oncology",
        "extracting oncology concepts",
        "The tumor responded to radiation therapy.",
    ),
    (
        "organismdetect",
        "Organism NER",
        "Organism",
        "extracting organism mentions",
        "Escherichia coli was isolated from the culture.",
    ),
    (
        "pathologydetect",
        "Pathology NER",
        "Pathology",
        "extracting pathology findings",
        "Pathology showed invasive ductal carcinoma.",
    ),
    (
        "pharmadetect",
        "Pharmaceutical NER",
        "Pharmaceutical",
        "extracting drugs and pharmacological substances",
        "The patient started imatinib after the consultation.",
    ),
    (
        "proteindetect",
        "Protein NER",
        "Protein",
        "extracting protein mentions",
        "HER2 expression was elevated in the tumor.",
    ),
    (
        "speciesdetect",
        "Species NER",
        "Species",
        "extracting species mentions",
        "Escherichia coli was isolated from the culture.",
    ),
)


def render_model_card(row: dict[str, Any]) -> str:
    """Return a README.md model card for one manifest row."""

    repo_id = _string(row.get("repo_id"), "OpenMed/model")
    benchmark = row.get("benchmark") if isinstance(row.get("benchmark"), dict) else {}
    formats = _list(row.get("formats"))
    if _is_android_onnx_card(repo_id, formats):
        return _render_android_onnx_model_card(row, repo_id, formats)

    title = repo_id.rsplit("/", 1)[-1]
    languages = _list(row.get("languages"))
    labels = _list(row.get("canonical_labels"))
    arxiv = _string(row.get("arxiv"), DEFAULT_ARXIV)
    license_name = _string(row.get("license"), "Not specified")
    task = _string(row.get("task"), "unknown")

    lines = [
        "---",
        f"license: {license_name}",
        f"pipeline_tag: {task}",
        "library_name: openmed",
        "tags:",
        "- openmed",
        "- medical-nlp",
        "---",
        "",
        f"# {title}",
        "",
        "This model card is rendered from the OpenMed model manifest. Update `models.jsonl` and rerun the publish step instead of editing this file directly.",
    ]
    usage_lines = _onnx_usage_block(repo_id, formats)
    if usage_lines:
        lines.extend(["", *usage_lines])
    lines.extend(
        [
            "",
            "## Manifest Summary",
            "",
            "| Field | Value |",
            "|---|---|",
            f"| Repository | `{repo_id}` |",
            f"| Family | {_string(row.get('family'), 'Not specified')} |",
            f"| Task | {task} |",
            f"| Languages | {_comma_or_unspecified(languages)} |",
            f"| Tier | {_string(row.get('tier'), 'Not specified')} |",
            f"| Parameters | {_format_param_count(row.get('param_count'))} |",
            f"| Architecture | {_string(row.get('architecture'), 'Not specified')} |",
            f"| Base model | {_string(row.get('base_model'), 'Not specified')} |",
            f"| Formats | {_comma_or_unspecified(formats)} |",
            f"| License | {license_name} |",
            f"| arXiv | {_arxiv_link(arxiv)} |",
            f"| Reproducibility hash | `{_string(row.get('reproducibility_hash'), 'Not specified')}` |",
            f"| Released | {_string(row.get('released'), 'Not specified')} |",
            "",
            "## Benchmark",
            "",
            "| Dataset | Micro F1 | Recall |",
            "|---|---:|---:|",
            f"| {_string(benchmark.get('dataset'), 'Not reported')} | {_metric(benchmark.get('micro_f1'))} | {_metric(benchmark.get('recall'))} |",
            "",
            "## Canonical Labels",
            "",
            _labels_block(labels),
        ]
    )
    artifact_lines = _artifact_format_block(formats)
    if artifact_lines:
        lines.extend(["", "## Artifact Format", "", *artifact_lines])
    distillation_lines = _distillation_block(row)
    if distillation_lines:
        lines.extend(["", "## Distillation Evidence", "", *distillation_lines])
    training_provenance_lines = _training_provenance_block(row)
    if training_provenance_lines:
        lines.extend(["", "## Training Provenance", "", *training_provenance_lines])
    encoder_provenance_lines = _encoder_provenance_block(row)
    if encoder_provenance_lines:
        lines.extend(["", "## Encoder Provenance", "", *encoder_provenance_lines])
    return "\n".join(lines) + "\n"


def _render_android_onnx_model_card(
    row: dict[str, Any],
    repo_id: str,
    formats: list[str],
) -> str:
    family = _string(row.get("family"), "NER").upper()
    languages = _model_languages(repo_id, _list(row.get("languages")))
    language_names = [_language_name(code) for code in languages]
    labels = _display_labels(_list(row.get("canonical_labels")))
    capability = _model_capability(repo_id, family)
    param_count = row.get("param_count")
    compact_params = _compact_param_count(param_count)
    architecture = _string(row.get("architecture"), "Transformer")
    architecture_name = _architecture_name(architecture)
    source_model = _string(row.get("base_model"), "Not specified")
    license_name = _string(row.get("license"), "apache-2.0")
    arxiv = _string(row.get("arxiv"), DEFAULT_ARXIV)
    task_name = (
        "PII token classification"
        if family == "PII"
        else f"{capability['short']} named-entity recognition"
    )
    title_parts = ["OpenMed", capability["title"]]
    if compact_params != "Not specified":
        title_parts.append(compact_params)
    title = " ".join(title_parts)
    language_text = ", ".join(language_names)
    sample_text = capability["sample"]
    description = (
        f"A {compact_params} {architecture_name} model for {capability['description']} "
        f"in {language_text} clinical and biomedical text."
    )
    if compact_params == "Not specified":
        description = (
            f"A {architecture_name} model for {capability['description']} in "
            f"{language_text} clinical and biomedical text."
        )

    lines = [
        "---",
        "library_name: openmed",
        f"license: {license_name}",
        "language:",
        *[f"- {language}" for language in languages],
        "pipeline_tag: token-classification",
    ]
    if source_model != "Not specified":
        lines.append(f"base_model: {source_model}")
    lines.extend(
        [
            "tags:",
            "- openmed",
            "- onnx",
            "- medical-nlp",
            "- medical-ner" if family != "PII" else "- pii-detection",
            "- android",
            "- webassembly",
            "- webgpu",
            "---",
            "",
            '<div align="center">',
            '<img src="https://raw.githubusercontent.com/maziyarpanahi/openmed/master/docs/brand/openmed-mascot-lockup.png" alt="OpenMed: on-device clinical AI" width="360">',
            "",
            f"# {title}",
            "",
            f"`{repo_id}`",
            "",
            f"**{description}**  ",
            "Runs locally after download on Python CPU, in the browser, and on Android.",
            "",
            "[OpenMed](https://github.com/maziyarpanahi/openmed) | "
            "[Documentation](https://openmed.life/docs) | "
            "[Model collection](https://huggingface.co/OpenMed) | "
            f"[Paper](https://arxiv.org/abs/{arxiv})",
            "</div>",
            "",
            "## Model",
            "",
            "| Field | Value |",
            "|---|---|",
            f"| Task | {task_name} |",
            f"| Language | {language_text} |",
            f"| Architecture | {architecture_name} |",
            f"| Parameters | {_format_param_count(param_count)} |",
            "| Maximum sequence length | 512 tokens |",
            f"| Entity labels | {_labels_block(labels)} |",
            f"| Source model | {_model_link(source_model)} |",
            f"| License | {license_name} |",
            "",
            "## OpenMed in Python on CPU",
            "",
            "```bash",
            'pip install --upgrade "openmed[onnx-runtime]"',
            "```",
            "",
            "```python",
            "from openmed import OnnxModel",
            "",
            f'model = OnnxModel.from_pretrained("{repo_id}")',
            f'entities = model("{sample_text}")',
            "",
            "for entity in entities:",
            "    print(entity.to_dict())",
            "```",
            "",
            "OpenMed selects the CPU-oriented INT8 graph by default and returns "
            "labels, confidence scores, exact character offsets, and source text.",
            "",
            "## OpenMed in Web",
            "",
            "```bash",
            "npm install openmed @huggingface/transformers onnxruntime-web",
            "```",
            "",
            "```typescript",
            'import { loadOnnxModel } from "openmed";',
            "",
            f'const repo = "{repo_id}";',
            "const model = await loadOnnxModel(repo);",
            f'const entities = await model("{sample_text}");',
            "```",
            "",
            "The default INT8 path runs with WebAssembly. For WebGPU, select the "
            "FP16 graph:",
            "",
            "```typescript",
            "const model = await loadOnnxModel(repo, {",
            '  variant: "fp16",',
            '  device: "webgpu",',
            "});",
            "```",
            "",
            "## OpenMedKit for Android",
            "",
            "Add JitPack to the consumer application's `settings.gradle.kts`:",
            "",
            "```kotlin",
            "dependencyResolutionManagement {",
            "    repositories {",
            "        google()",
            "        mavenCentral()",
            "        maven {",
            '            url = uri("https://jitpack.io")',
            '            content { includeGroup("com.github.maziyarpanahi") }',
            "        }",
            "    }",
            "}",
            "```",
            "",
            f"Use the stable OpenMed `{__version__}` release:",
            "",
            "```kotlin",
            "dependencies {",
            f'    implementation("com.github.maziyarpanahi:openmed:v{__version__}")',
            "}",
            "```",
            "",
            "After downloading this model repository into an app-controlled directory:",
            "",
            "```kotlin",
            "import com.openmed.openmedkit.OpenMedKit",
            "",
            "suspend fun analyzeModel() {",
            "    OpenMedKit.fromDirectory(modelDirectory).use { model ->",
            f'        val entities = model.analyzeText("{sample_text}")',
            "    }",
            "}",
            "```",
            "",
            "Inference and tokenization remain on-device.",
            "",
            "## Included Artifacts",
            "",
            "| Artifact | Recommended use |",
            "|---|---|",
            *_onnx_artifact_rows(formats),
            "| `tokenizer.json` | Cross-platform tokenizer |",
            "| `openmed-onnx.json` | Runtime contract and operator metadata |",
            "",
            "All graphs use opset 18, dynamic batch and sequence axes, stable "
            "tensor names, and source-text offset metadata.",
            "",
            "## The OpenMed Ecosystem",
            "",
            "This model is part of **OpenMed**, an Apache-2.0, local-first "
            "clinical AI stack:",
            "",
            "- **2,000+ medical models** for clinical NER, biomedical extraction, and privacy.",
            "- **PII detection and de-identification** across 55+ identifier types and 20 languages.",
            "- **Python, MLX, Swift, Android, React Native, Web, REST, and gRPC** runtimes.",
            "- **Structured and multimodal intake** for OCR, documents, DICOM, FHIR, and HL7.",
            "- **Offline and air-gapped deployment** with no telemetry by default.",
            "",
            "## Intended Use and Limitations",
            "",
            f"This model is intended for {capability['description']}. It does not "
            "diagnose conditions or make clinical decisions. Evaluate recall, "
            "thresholds, and span behavior on appropriately governed data before "
            "deployment. Local execution supports privacy-preserving workflows but "
            "does not by itself guarantee regulatory compliance.",
            "",
            "All examples in this card are synthetic.",
            "",
            "## Citation",
            "",
            "```bibtex",
            "@misc{panahi2025openmedneropensourcedomainadapted,",
            "  title={OpenMed NER: Open-Source, Domain-Adapted State-of-the-Art",
            "         Transformers for Biomedical NER Across 12 Public Datasets},",
            "  author={Maziyar Panahi},",
            "  year={2025},",
            "  eprint={2508.01630},",
            "  archivePrefix={arXiv},",
            "  primaryClass={cs.CL}",
            "}",
            "```",
        ]
    )
    return "\n".join(lines) + "\n"


def _is_android_onnx_card(repo_id: str, formats: list[str]) -> bool:
    normalized = {_normalize_format(value) for value in formats}
    return repo_id.endswith("-onnx-android") or bool(
        normalized & {"onnx-android", "ort-android"}
    )


def _model_languages(repo_id: str, configured: list[str]) -> list[str]:
    for name, code in _MODEL_LANGUAGES.items():
        if f"-PII-{name}-" in repo_id:
            return [code]
    languages = [value.lower() for value in configured if value]
    return languages or ["en"]


def _language_name(code: str) -> str:
    return _LANGUAGE_NAMES.get(code.lower(), code)


def _model_capability(repo_id: str, family: str) -> dict[str, str]:
    if family == "PII":
        return {
            "title": "PII Detection",
            "short": "PII",
            "description": "detecting personal and clinical identifiers",
            "sample": "Patient Alice Nguyen can be reached at alice@example.org.",
        }

    normalized = repo_id.lower().replace("-", "")
    for marker, title, short, description, sample in _NER_CAPABILITIES:
        if marker in normalized:
            return {
                "title": title,
                "short": short,
                "description": description,
                "sample": sample,
            }
    return {
        "title": "Clinical NER",
        "short": "Clinical",
        "description": "extracting clinical and biomedical entities",
        "sample": "The patient started imatinib for chronic myeloid leukemia.",
    }


def _display_labels(labels: list[str]) -> list[str]:
    result: list[str] = []
    for raw_label in labels:
        label = str(raw_label).strip()
        if len(label) > 2 and label[1] in {"-", "_"}:
            label = label[2:]
        if not label or label.upper() == "O" or label in result:
            continue
        result.append(label)
    return result


def _compact_param_count(value: Any) -> str:
    formatted = _format_param_count(value)
    if formatted == "Not specified":
        return formatted
    return formatted.split(" ", 1)[0]


def _architecture_name(value: str) -> str:
    normalized = value.lower().replace("_", "-")
    names = {
        "bert": "BERT",
        "clinical-longformer": "Clinical Longformer",
        "deberta-v2": "DeBERTa-v2",
        "distilbert": "DistilBERT",
        "eurobert": "EuroBERT",
        "modernbert": "ModernBERT",
        "qwen": "Qwen",
        "roberta": "RoBERTa",
        "xlm-roberta": "XLM-RoBERTa",
    }
    return names.get(normalized, value)


def _model_link(model_id: str) -> str:
    if model_id == "Not specified":
        return model_id
    if "/" in model_id:
        return f"[`{model_id}`](https://huggingface.co/{model_id})"
    return f"`{model_id}`"


def _onnx_artifact_rows(formats: list[str]) -> list[str]:
    normalized = {_normalize_format(value) for value in formats}
    rows: list[str] = []
    if normalized & {"int8", "onnx-int8"}:
        rows.append("| `model_int8.onnx` | CPU, WebAssembly, and Android default |")
    if "onnx-android" in normalized:
        rows.extend(
            [
                "| `model_fp16.onnx` | WebGPU and compatible accelerated runtimes |",
                "| `model.onnx` | Full-precision reference |",
            ]
        )
    if "ort-android" in normalized:
        rows.append("| `model.ort` | Custom ONNX Runtime Mobile integration |")
    return rows


def write_model_card(path: str | Path, row: dict[str, Any]) -> Path:
    """Write a rendered model card to *path* and return the path."""

    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(render_model_card(row), encoding="utf-8")
    return path


def append_model_card_sections(card: str, sections: Sequence[str]) -> str:
    """Append generated Markdown sections to a rendered model card."""

    rendered = card if card.endswith("\n") else f"{card}\n"
    for section in sections:
        section_text = str(section).strip()
        if not section_text:
            continue
        rendered = f"{rendered.rstrip()}\n\n{section_text}\n"
    return rendered


def _string(value: Any, default: str) -> str:
    if value is None:
        return default
    text = str(value).strip()
    return text or default


def _list(value: Any) -> list[str]:
    if not isinstance(value, list):
        return []
    return [str(item) for item in value if str(item)]


def _comma_or_unspecified(values: list[str]) -> str:
    return ", ".join(values) if values else "Not specified"


def _format_param_count(value: Any) -> str:
    if not isinstance(value, int) or value <= 0:
        return "Not specified"
    if value >= 1_000_000_000:
        compact = f"{value / 1_000_000_000:g}B"
    elif value >= 1_000_000:
        compact = f"{value / 1_000_000:g}M"
    elif value >= 1_000:
        compact = f"{value / 1_000:g}K"
    else:
        compact = str(value)
    return f"{compact} ({value:,})"


def _metric(value: Any) -> str:
    if isinstance(value, (int, float)):
        return f"{float(value):.4f}"
    return "Not reported"


def _arxiv_link(value: str) -> str:
    if value == "Not specified":
        return value
    return f"[arXiv:{value}](https://arxiv.org/abs/{value})"


def _labels_block(labels: list[str]) -> str:
    if not labels:
        return "Not specified."
    return ", ".join(f"`{label}`" for label in labels)


def _artifact_format_block(formats: list[str]) -> list[str]:
    runtime_formats = []
    quantization_formats = []
    for value in formats:
        if _normalize_format(value) in {
            "onnx",
            "onnx-android",
            "ort-android",
            "transformersjs",
            "webgpu",
        }:
            runtime_formats.append(value)
        quantization = _quantization_label(value)
        if quantization is not None:
            quantization_formats.append(quantization)
    if not runtime_formats and not quantization_formats:
        return []

    return [
        "| Field | Value |",
        "|---|---|",
        f"| Runtime artifacts | {_comma_or_unspecified(runtime_formats)} |",
        f"| Quantization | {_comma_or_unspecified(quantization_formats)} |",
    ]


def _onnx_usage_block(repo_id: str, formats: list[str]) -> list[str]:
    if not any(
        _normalize_format(value).startswith(("onnx", "ort")) for value in formats
    ):
        return []
    text = "Patient Alice Nguyen was seen in cardiology."
    return [
        "## OpenMed in Python on CPU",
        "",
        "```python",
        "from openmed import OnnxModel",
        f'model = OnnxModel.from_pretrained("{repo_id}")',
        f'entities = model("{text}")',
        "```",
        "",
        "## OpenMed in Web",
        "",
        "```bash",
        "npm install openmed @huggingface/transformers onnxruntime-web",
        "```",
        "",
        "```typescript",
        'import { loadOnnxModel } from "openmed";',
        f'const model = await loadOnnxModel("{repo_id}");',
        f'const entities = await model("{text}");',
        "```",
        "",
        "## OpenMedKit for Android",
        "",
        "```kotlin",
        "val model = OpenMedKit.fromDirectory(modelDirectory)",
        f'val entities = model.analyzeText("{text}")',
        "```",
    ]


def _normalize_format(value: str) -> str:
    return value.lower().replace("_", "-")


def _quantization_label(value: str) -> str | None:
    normalized = _normalize_format(value)
    if normalized in {"int8", "8bit", "8-bit", "onnx-int8"}:
        return "int8"
    if normalized in {"int4", "4bit", "4-bit", "onnx-int4"}:
        return "int4"
    if normalized in {"awq", "openmed-awq"}:
        return "awq"
    if normalized in {"gptq", "openmed-gptq"}:
        return "gptq"
    return None


def _distillation_block(row: dict[str, Any]) -> list[str]:
    payload = _distillation_payload(row)
    if not payload:
        return []

    critical_drops = _list(payload.get("critical_label_drops"))
    lines = [
        "| Field | Value |",
        "|---|---|",
        f"| Teacher | `{_string(payload.get('teacher_id'), 'Not reported')}` |",
        f"| Student backbone | `{_string(payload.get('student_backbone'), 'Not reported')}` |",
        f"| Temperature | {_metric(payload.get('temperature'))} |",
        f"| Alpha | {_metric(payload.get('alpha'))} |",
        f"| Recall gate | {_gate_status(payload.get('recall_gate_passed'))} |",
        f"| Critical drops | {_critical_drops(critical_drops)} |",
    ]

    deltas = payload.get("per_label_recall_delta")
    if isinstance(deltas, list) and deltas:
        lines.extend(
            [
                "",
                "| Label | Teacher recall | Student recall | Delta | Critical drop |",
                "|---|---:|---:|---:|---|",
            ]
        )
        for item in deltas:
            if not isinstance(item, dict):
                continue
            lines.append(
                f"| {_string(item.get('label'), 'UNKNOWN')} | "
                f"{_metric(item.get('teacher_recall'))} | "
                f"{_metric(item.get('student_recall'))} | "
                f"{_metric(item.get('delta'))} | "
                f"{_yes_no(item.get('critical_drop'))} |"
            )
    return lines


def _distillation_payload(row: dict[str, Any]) -> dict[str, Any]:
    payload = row.get("distillation")
    if isinstance(payload, dict):
        return payload
    evidence = row.get("evidence")
    if isinstance(evidence, dict) and isinstance(evidence.get("distillation"), dict):
        return evidence["distillation"]
    return {}


def _training_provenance_block(row: dict[str, Any]) -> list[str]:
    payload = row.get("training_provenance")
    if not isinstance(payload, dict):
        return []

    seeds = payload.get("rng_seeds")
    seed_text = "Not reported"
    if isinstance(seeds, dict) and seeds:
        seed_text = ", ".join(
            f"`{str(key)}`={value}" for key, value in sorted(seeds.items())
        )

    lines = [
        "| Field | Value |",
        "|---|---|",
        f"| Provenance file | `{_string(payload.get('path'), 'training_provenance.json')}` |",
        f"| Base model revision | `{_string(payload.get('base_model_revision'), 'Not reported')}` |",
        f"| Git SHA | `{_string(payload.get('git_sha'), 'Not reported')}` |",
        f"| RNG seeds | {seed_text} |",
        f"| Data manifest hash | `{_string(payload.get('data_manifest_hash'), 'Not reported')}` |",
        f"| Recipe config hash | `{_string(payload.get('recipe_config_hash'), 'Not reported')}` |",
        f"| Environment lock digest | `{_string(payload.get('env_lock_digest'), 'Not reported')}` |",
        f"| Provenance reproducibility hash | `{_string(payload.get('reproducibility_hash'), 'Not reported')}` |",
    ]
    return lines


def _encoder_provenance_block(row: dict[str, Any]) -> list[str]:
    payload = row.get("encoder_provenance")
    if not isinstance(payload, dict):
        payload = row.get("indic_encoder")
    if not isinstance(payload, dict):
        return []

    family = _string(payload.get("family"), "Not specified")
    source = _string(
        payload.get("source") or payload.get("repo_id"),
        "Not specified",
    )
    license_name = _string(payload.get("license"), "Not specified")
    provenance = _string(payload.get("provenance"), "user-supplied")
    weights = _string(payload.get("weights"), "user-supplied; not bundled")
    transliterated = payload.get("supports_transliterated_text")
    transliterated_value = (
        "Yes"
        if transliterated is True
        else "No"
        if transliterated is False
        else "Not specified"
    )
    return [
        "| Field | Value |",
        "|---|---|",
        f"| Encoder family | {family} |",
        f"| Source | `{source}` |",
        f"| License | {license_name} |",
        f"| Provenance | {provenance} |",
        f"| Weights | {weights} |",
        f"| Transliterated text | {transliterated_value} |",
    ]


def _gate_status(value: Any) -> str:
    if value is True:
        return "passed"
    if value is False:
        return "failed"
    return "Not reported"


def _critical_drops(values: list[str]) -> str:
    return ", ".join(f"`{value}`" for value in values) if values else "None"


def _yes_no(value: Any) -> str:
    if value is True:
        return "yes"
    if value is False:
        return "no"
    return "Not reported"
