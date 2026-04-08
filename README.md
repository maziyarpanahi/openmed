# OpenMed

> **Production-ready medical NLP toolkit powered by state-of-the-art transformers**

Transform clinical text into structured insights with a single line of code. OpenMed delivers enterprise-grade entity extraction, assertion detection, and medical reasoning—no vendor lock-in, no compromise on accuracy.

[![License](https://img.shields.io/badge/License-Apache_2.0-blue.svg)](https://opensource.org/licenses/Apache-2.0)
[![Python 3.10+](https://img.shields.io/badge/python-3.10+-blue.svg)](https://www.python.org/downloads/)
[![arXiv](https://img.shields.io/badge/arXiv-2508.01630-b31b1b.svg)](https://arxiv.org/abs/2508.01630)
[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/drive/1x1xJjTZTWR3Z7uLJ0B5B_FyAomeeZGq5?usp=sharing)

```python
from openmed import analyze_text

result = analyze_text(
    "Patient started on imatinib for chronic myeloid leukemia.",
    model_name="disease_detection_superclinical"
)

for entity in result.entities:
    print(f"{entity.label:<12} {entity.text:<35} {entity.confidence:.2f}")
# DISEASE      chronic myeloid leukemia            0.98
# DRUG         imatinib                            0.95
```

---

## ✨ Why OpenMed?

- **Specialized Models**: 12+ curated medical NER models outperforming proprietary solutions
- **HIPAA-Compliant PII Detection**: Smart de-identification with all 18 Safe Harbor identifiers
- **One-Line Deployment**: From prototype to production in minutes
- **Dockerized REST API**: FastAPI endpoints for service deployments
- **Batch Processing**: Multi-file workflows with progress tracking
- **Production-Ready**: Configuration profiles, profiling tools, and medical-aware tokenization
- **Zero Lock-In**: Apache 2.0 licensed, runs on your infrastructure

---

## Quick Start

### Installation

```bash
# From a local checkout of this repo:
# Install with Hugging Face support
uv pip install -e ".[hf]"

# Or include REST service dependencies
uv pip install -e ".[hf,service]"
```

Apple Silicon acceleration in Python:

```bash
uv pip install -e ".[mlx]"
```

Swift apps on macOS and iOS use `OpenMedKit` with **CoreML-compatible model bundles**, not MLX:

```swift
dependencies: [
    .package(url: "https://github.com/maziyarpanahi/openmed.git", from: "1.0.0"),
]
```

OpenMedKit is public in `1.0.0`. The universal OpenMed model-packaging flow for Apple platforms is still being hardened across BERT, DistilBERT, RoBERTa, XLM-R, Longformer, ModernBERT, and related families, so treat conversion as active work rather than a stable public release surface.

After `1.0.0` is published to PyPI, the editable install examples above can be replaced with plain `uv pip install "openmed[...]"`.

### Three Ways to Use OpenMed

**1️⃣ Python API** — One-liner for scripts and notebooks

```python
from openmed import analyze_text

result = analyze_text(
    "Patient received 75mg clopidogrel for NSTEMI.",
    model_name="pharma_detection_superclinical"
)
```

**2️⃣ REST API Service** — FastAPI endpoints for app backends

```bash
uvicorn openmed.service.app:app --host 0.0.0.0 --port 8080
```

**3️⃣ Batch Processing** — Programmatic multi-document workflows

```python
from openmed import BatchProcessor

processor = BatchProcessor(
    model_name="disease_detection_superclinical",
    confidence_threshold=0.55,
    group_entities=True,
)

result = processor.process_texts([
    "Patient started metformin for type 2 diabetes.",
    "Imatinib started for chronic myeloid leukemia.",
])
```

---

## Key Features

### Core Capabilities

- **Curated Model Registry**: Metadata-rich catalog with 12+ specialized medical NER models
- **PII Detection & De-identification**: HIPAA-compliant de-identification with smart entity merging
- **Medical-Aware Tokenization**: Clean handling of clinical patterns (`COVID-19`, `CAR-T`, `IL-6`)
- **Advanced NER Processing**: Confidence filtering, entity grouping, and span alignment
- **Multiple Output Formats**: Dict, JSON, HTML, CSV for any downstream system

### Production Tools (v1.0.0)

- **Batch Processing**: Multi-text and multi-file workflows with progress tracking
- **Configuration Profiles**: `dev`/`prod`/`test`/`fast` presets with flexible overrides
- **Performance Profiling**: Built-in inference timing and bottleneck analysis
- **Dockerized REST API**: `GET /health`, `POST /analyze`, `POST /pii/extract`, `POST /pii/deidentify`
- **Service Reliability Hardening**: request validation, shared pipeline preload, and timeout/error envelopes

---

## Documentation

Comprehensive guides available at **[openmed.life/docs](https://openmed.life/docs/)**

Quick links:

- [Getting Started](https://openmed.life/docs/) — Installation and first analysis
- [Analyze Text Helper](https://openmed.life/docs/analyze-text) — Python API reference
- [MLX Backend](docs/mlx-backend.md) — Apple Silicon Python runtime
- [OpenMedKit (Swift)](docs/swift-openmedkit.md) — Use OpenMed models in macOS and iOS apps
- [PII Detection Guide](examples/notebooks/PII_Detection_Complete_Guide.ipynb) — Complete de-identification tutorial (v0.5.0)
- [Batch Processing](https://openmed.life/docs/batch-processing) — Multi-text and multi-file workflows
- [Configuration Profiles](https://openmed.life/docs/profiles) — Environment-specific presets
- [REST Service](docs/rest-service.md) — FastAPI and Docker usage
- [Model Registry](https://openmed.life/docs/model-registry) — Browse available models
- [Configuration](https://openmed.life/docs/configuration) — Settings and environment variables

---

## REST API (v1.0.0)

OpenMed includes a Docker-friendly FastAPI service with reliability hardening:

- `GET /health`
- `POST /analyze`
- `POST /pii/extract`
- `POST /pii/deidentify`

### Run locally

```bash
uv pip install -e ".[hf,service]"
uvicorn openmed.service.app:app --host 0.0.0.0 --port 8080
```

Optional shared model warm-up:

```bash
OPENMED_SERVICE_PRELOAD_MODELS=disease_detection_superclinical,OpenMed/OpenMed-PII-SuperClinical-Small-44M-v1 \
uvicorn openmed.service.app:app --host 0.0.0.0 --port 8080
```

### Run with Docker

```bash
docker build -t openmed:1.0.0 .
docker run --rm -p 8080:8080 -e OPENMED_PROFILE=prod openmed:1.0.0
```

### Example request

```bash
curl -X POST http://127.0.0.1:8080/pii/extract \
  -H "Content-Type: application/json" \
  -d '{"text":"Paciente: Maria Garcia, DNI: 12345678Z","lang":"es"}'
```

See the full service guide at [REST Service docs](docs/rest-service.md).

Non-2xx responses now use a unified envelope:

```json
{
  "error": {
    "code": "validation_error",
    "message": "Request validation failed",
    "details": [
      {
        "field": "body.text",
        "message": "Text must not be blank",
        "type": "value_error"
      }
    ]
  }
}
```

---

## Models

OpenMed includes a curated registry of 12+ specialized medical NER models:

| Model | Specialization | Entity Types | Size |
|-------|---------------|--------------|------|
| `disease_detection_superclinical` | Disease & Conditions | DISEASE, CONDITION, DIAGNOSIS | 434M |
| `pharma_detection_superclinical` | Drugs & Medications | DRUG, MEDICATION, TREATMENT | 434M |
| `pii_detection_superclinical` | PII & De-identification | NAME, DATE, SSN, PHONE, EMAIL, ADDRESS | 434M |
| `anatomy_detection_electramed` | Anatomy & Body Parts | ANATOMY, ORGAN, BODY_PART | 109M |
| `gene_detection_genecorpus` | Genes & Proteins | GENE, PROTEIN | 109M |

[📖 Full Model Catalog](https://openmed.life/docs/model-registry)

---

## Advanced Usage

### PII Detection & De-identification (v0.5.0)

```python
from openmed import extract_pii, deidentify

# Extract PII entities with smart merging (default)
result = extract_pii(
    "Patient: John Doe, DOB: 01/15/1970, SSN: 123-45-6789",
    model_name="pii_detection_superclinical",
    use_smart_merging=True  # Prevents entity fragmentation
)

# De-identify with multiple methods
masked = deidentify(text, method="mask")        # [NAME], [DATE]
removed = deidentify(text, method="remove")     # Complete removal
replaced = deidentify(text, method="replace")   # Synthetic data
hashed = deidentify(text, method="hash")        # Cryptographic hashing
shifted = deidentify(text, method="shift_dates", date_shift_days=180)
```

**Smart Entity Merging** (NEW in v0.5.0): Fixes tokenization fragmentation by merging split entities like dates (`01/15/1970` instead of `01` + `/15/1970`), ensuring production-ready de-identification.

**HIPAA Compliance**: Covers all 18 Safe Harbor identifiers with configurable confidence thresholds.

[📓 Complete PII Notebook](examples/notebooks/PII_Detection_Complete_Guide.ipynb) | [📖 Documentation](docs/pii-smart-merging.md)

### Multilingual PII (8 Languages)

OpenMed now supports multilingual PII extraction and de-identification across `en`, `fr`, `de`, `it`, `es`, `nl`, `hi`, and `te`.
French, German, Italian, and Spanish expose the full 35-model family; Dutch, Hindi, and Telugu currently ship one flagship public model each, bringing the total PII catalog to **179 models**.

```python
from openmed import extract_pii

dutch = extract_pii(
    "Patiënt: Eva de Vries, geboortedatum: 15 januari 1984, BSN: 123456782, telefoon: +31 6 12345678",
    lang="nl",
    model_name="OpenMed/OpenMed-PII-Dutch-SuperClinical-Large-434M-v1",
    use_smart_merging=True,
)

hindi = extract_pii(
    "रोगी: अनीता शर्मा, जन्मतिथि: 15 जनवरी 1984, फोन: +91 9876543210, पता: 12 गली संख्या 5, नई दिल्ली 110001",
    lang="hi",
    model_name="OpenMed/OpenMed-PII-Hindi-SuperClinical-Large-434M-v1",
    use_smart_merging=True,
)

telugu = extract_pii(
    "రోగి: సితా రెడ్డి, జన్మ తేదీ: 15 జనవరి 1984, ఫోన్: +91 9876543210, చిరునామా: 12 వీధి 5, హైదరాబాద్ 500001",
    lang="te",
    model_name="OpenMed/OpenMed-PII-Telugu-SuperClinical-Large-434M-v1",
    use_smart_merging=True,
)

print([(e.label, e.text) for e in dutch.entities])
print([(e.label, e.text) for e in hindi.entities])
print([(e.label, e.text) for e in telugu.entities])
```

### Batch Processing

```python
from openmed import BatchProcessor, OpenMedConfig

config = OpenMedConfig.from_profile("prod")
processor = BatchProcessor(
    model_name="disease_detection_superclinical",
    config=config,
    group_entities=True,
)

result = processor.process_texts([
    "Metastatic breast cancer treated with trastuzumab.",
    "Acute lymphoblastic leukemia diagnosed.",
])
```

### Configuration Profiles

```python
from openmed import analyze_text

# Apply a profile programmatically
result = analyze_text(
    text,
    model_name="disease_detection_superclinical",
    config_profile="prod"  # High confidence, grouped entities
)
```

### Performance Profiling

```python
from openmed import analyze_text, profile_inference

with profile_inference() as profiler:
    result = analyze_text(text, model_name="disease_detection_superclinical")

print(profiler.summary())  # Inference time, bottlenecks, recommendations
```

[📖 More Examples](https://openmed.life/docs/examples)

---

## Contributing

We welcome contributions! Whether it's bug reports, feature requests, or pull requests.

- 🐛 **Found a bug?** [Open an issue](https://github.com/maziyarpanahi/openmed/issues)

---

## License

OpenMed is released under the [Apache-2.0 License](LICENSE).

---

## Citation

If you use OpenMed in your research, please cite:

```bibtex
@misc{panahi2025openmedneropensourcedomainadapted,
      title={OpenMed NER: Open-Source, Domain-Adapted State-of-the-Art Transformers for Biomedical NER Across 12 Public Datasets},
      author={Maziyar Panahi},
      year={2025},
      eprint={2508.01630},
      archivePrefix={arXiv},
      primaryClass={cs.CL},
      url={https://arxiv.org/abs/2508.01630},
}
```

---

## Star History

If you find OpenMed useful, consider giving it a star ⭐ to help others discover it!

---

**Built with ❤️ by the OpenMed team**

[🌐 Website](https://openmed.life) • [📚 Documentation](https://openmed.life/docs) • [🐦 X/Twitter](https://x.com/openmed_ai) • [💬 LinkedIn](https://www.linkedin.com/company/openmed-ai/)
