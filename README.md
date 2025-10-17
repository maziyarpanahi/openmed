# OpenMed

OpenMed is a Python toolkit for biomedical and clinical NLP, built to deliver state-of-the-art models, including advanced large language models (LLMs) for healthcare, that rival and often outperform proprietary enterprise solutions. It unifies model discovery, assertion status detection, de-identification pipelines, advanced extraction and reasoning tools, and one-line orchestration for scripts, services, or notebooks, enabling teams to deploy production-grade healthcare AI without vendor lock-in.

It also bundles configuration management, model loading, support for cutting-edge medical LLMs, post-processing, and formatting utilities — making it seamless to integrate clinical AI into existing scripts, services, and research workflows.

> **Status:** The package is pre-release and the API may change. Feedback and contributions are
> welcome while the project stabilises.

## Features

- **Curated model registry** with metadata for the OpenMed Hugging Face collection, including
  category filters, entity coverage, and confidence guidance.
- **One-line model loading** via `ModelLoader`, with optional pipeline creation,
  caching, and authenticated access to private models.
- **Advanced NER post-processing** (`AdvancedNERProcessor`) that applies the filtering and
  grouping techniques proven in the OpenMed demos.
- **Text preprocessing & tokenisation helpers** tailored for medical text workflows.
- **Output formatting utilities** that convert raw predictions into dict/JSON/HTML/CSV for
  downstream systems.
- **Logging and validation helpers** to keep pipelines observable and inputs safe.

## Installation

### Requirements

- Python 3.8 or newer (the package metadata allows 3.6+, but Hugging Face tooling typically
  requires >=3.8).
- [`transformers`](https://huggingface.co/docs/transformers/index) and a compatible deep learning
  backend such as [PyTorch](https://pytorch.org/get-started/locally/).
- An optional `HF_TOKEN` environment variable if you need to access gated models.

### Install from PyPI

```bash
pip install openmed transformers
# Install a backend (PyTorch shown here; follow the instructions for your platform):
pip install torch --index-url https://download.pytorch.org/whl/cpu
```

If you plan to run on GPU, install the CUDA-enabled PyTorch wheels from the official instructions.

## Quick start

```python
from openmed.core import ModelLoader
from openmed.processing import format_predictions

loader = ModelLoader()  # uses the default configuration
ner = loader.create_pipeline(
    "disease_detection_superclinical",  # registry key or full model ID
    aggregation_strategy="simple",      # group sub-token predictions for quick wins
)

text = "Patient diagnosed with acute lymphoblastic leukemia and started on imatinib."
raw_predictions = ner(text)

result = format_predictions(raw_predictions, text, model_name="Disease Detection")
for entity in result.entities:
    print(f"{entity.label:<12} -> {entity.text} (confidence={entity.confidence:.2f})")
```

Use the convenience helper if you prefer a single call:

```python
from openmed import analyze_text

result = analyze_text(
    "Patient received 75mg clopidogrel for NSTEMI.",
    model_name="pharma_detection_superclinical"
)

for entity in result.entities:
    print(entity)
```

## Command-line usage

Install the package in the usual way and the `openmed` console command will be
available. It provides quick access to model discovery, text analysis, and
configuration management.

```bash
# List models from the bundled registry (add --include-remote for Hugging Face)
openmed models list
openmed models list --include-remote

# Analyse inline text or a file with a specific model
openmed analyze --model disease_detection_superclinical --text "Acute leukemia treated with imatinib."

# Inspect or edit the CLI configuration (defaults to ~/.config/openmed/config.toml)
openmed config show
openmed config set device cuda

# Inspect the model's inferred context window
openmed models info disease_detection_superclinical
```

Provide `--config-path /custom/path.toml` to work with a different configuration
file during automation or testing. Run `openmed --help` to see all options.

## Discovering models

```python
from openmed.core import ModelLoader
from openmed.core.model_registry import list_model_categories, get_models_by_category

loader = ModelLoader()
print(loader.list_available_models()[:5])  # Hugging Face + registry entries

suggestions = loader.get_model_suggestions(
    "Metastatic breast cancer treated with paclitaxel and trastuzumab"
)
for key, info, reason in suggestions:
    print(f"{info.display_name} -> {reason}")

print(list_model_categories())
for info in get_models_by_category("Oncology"):
    print(f"- {info.display_name} ({info.model_id})")

from openmed import get_model_max_length
print(get_model_max_length("disease_detection_superclinical"))
```

Or use the top-level helper:

```python
from openmed import list_models

print(list_models()[:10])
```

## Advanced NER processing

```python
from openmed.core import ModelLoader
from openmed.processing.advanced_ner import create_advanced_processor

loader = ModelLoader()
# aggregation_strategy=None yields raw token-level predictions for maximum control
ner = loader.create_pipeline("pharma_detection_superclinical", aggregation_strategy=None)

text = "Administered 75mg clopidogrel daily alongside aspirin for secondary stroke prevention."
raw = ner(text)

processor = create_advanced_processor(confidence_threshold=0.65)
entities = processor.process_pipeline_output(text, raw)
summary = processor.create_entity_summary(entities)

for entity in entities:
    print(f"{entity.label}: {entity.text} (score={entity.score:.3f})")

print(summary["by_type"])
```

## Text preprocessing & tokenisation

```python
from openmed.processing import TextProcessor, TokenizationHelper
from openmed.core import ModelLoader

text_processor = TextProcessor(normalize_whitespace=True, lowercase=False)
clean_text = text_processor.clean_text("BP 120/80, HR 88 bpm. Start Metformin 500mg bid.")
print(clean_text)

loader = ModelLoader()
model_data = loader.load_model("anatomy_detection_electramed")
token_helper = TokenizationHelper(model_data["tokenizer"])
encoding = token_helper.tokenize_with_alignment(clean_text)
print(encoding["tokens"][:10])
```

## Formatting outputs

```python
# Reuse `raw_predictions` and `text` from the quick start example
from openmed.processing import format_predictions

formatted = format_predictions(
    raw_predictions,
    text,
    model_name="Disease Detection",
    output_format="json",
    include_confidence=True,
    confidence_threshold=0.5,
)
print(formatted)  # JSON string ready for logging or storage
```

`format_predictions` can also return CSV rows or rich HTML snippets for dashboards.

## Configuration & logging

```python
from openmed.core import OpenMedConfig, ModelLoader
from openmed.utils import setup_logging

config = OpenMedConfig(
    default_org="OpenMed",
    cache_dir="/tmp/openmed-cache",
    device="cuda",  # "cpu", "cuda", or a specific device index
)
setup_logging(level="INFO")
loader = ModelLoader(config=config)
```

`OpenMedConfig` automatically picks up `HF_TOKEN` from the environment so you can access
private or gated models without storing credentials in code.

## Validation utilities

```python
from openmed.utils.validation import validate_input, validate_model_name

text = validate_input(user_supplied_text, max_length=2000)
model = validate_model_name("OpenMed/OpenMed-NER-DiseaseDetect-SuperClinical-434M")
```

Use these helpers to guard API endpoints or batch pipelines against malformed inputs.

## License

OpenMed is released under the Apache-2.0 License.

## Citing

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
