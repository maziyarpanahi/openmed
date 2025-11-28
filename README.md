# OpenMed

OpenMed is a Python toolkit for biomedical and clinical NLP, built to deliver state-of-the-art models, including advanced large language models (LLMs) for healthcare, that rival and often outperform proprietary enterprise solutions. It unifies model discovery, assertion status detection, de-identification pipelines, advanced extraction and reasoning tools, and one-line orchestration for scripts, services, or notebooks, enabling teams to deploy production-grade healthcare AI without vendor lock-in.

It also bundles configuration management, model loading, support for cutting-edge medical LLMs, post-processing, and formatting utilities — making it seamless to integrate clinical AI into existing scripts, services, and research workflows.

> **Status:** The package is pre-release and the API may change. Feedback and contributions are
> welcome while the project stabilises.

## TL;DR — run this first

```bash
# 1. Install uv (skip if you already have it)
curl -LsSf https://astral.sh/uv/install.sh | sh

# 2. Create and activate a fresh Python 3.11 environment in this repo
uv venv --python 3.11
source .venv/bin/activate
# 3. Install OpenMed (with Hugging Face extras) and run a one-liner demo
uv pip install "openmed[hf]"
```

```python
from openmed import analyze_text

result = analyze_text(
    "Patient started on imatinib for chronic myeloid leukemia.",
    model_name="disease_detection_superclinical",
)

for entity in result.entities:
    confidence = float(entity.confidence) if entity.confidence is not None else None
    print(f"{entity.label:<18} {entity.text:<35} confidence={confidence}")
```

### Try it in Google Colab (no setup required)

[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/drive/1x1xJjTZTWR3Z7uLJ0B5B_FyAomeeZGq5?usp=sharing)

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

- Python 3.10 or newer.
- Optional access to Hugging Face Hub (`HF_TOKEN`) if you consume gated models.
- A deep-learning runtime such as [PyTorch](https://pytorch.org/get-started/locally/) when you run on real hardware.

The core package keeps its mandatory dependency list intentionally small. Transformer-based pipelines live behind the
`hf` optional extra (see `pyproject.toml`) so minimal deployments—static data processing, registry exploration,
formatting—can install only what they need. When you want Hugging Face pipelines, install the extra.

### Install with `uv`

```bash
# inside your project virtualenv (e.g. `source .venv/bin/activate`)
# base install
uv pip install .

# include Hugging Face support (transformers + huggingface-hub + compatible versions)
uv pip install ".[hf]"

# add developer tooling (pytest, linters, coverage)
uv pip install ".[dev]"
```

Install the appropriate PyTorch wheel for your platform if you plan to execute models:

```bash
uv pip install "torch==2.9.0" --index-url https://download.pytorch.org/whl/cpu
```

Swap in a CUDA/CU121 index URL when targeting GPUs.

### Optional extras at a glance

Pick the extras that fit your workflow and stack them as needed:

- `.[hf]` – Hugging Face pipelines (`transformers`, `huggingface-hub`, `accelerate`)
- `.[gliner]` – Zero-shot GLiNER models, PyTorch, tokenizer deps
- `.[dev]` – Test and lint tooling (`pytest`, coverage, flake8)

Common install patterns with `uv`:

```bash
# Base toolkit only
uv pip install .

# Add Hugging Face integration
uv pip install ".[hf]"

# Add zero-shot GLiNER stack
uv pip install ".[gliner]"

# Developer tools for local hacking
uv pip install ".[dev]"

# Everything in one go (HF + GLiNER + dev)
uv pip install ".[dev,hf,gliner]"
```

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

### Zero-shot NER tooling

Install the optional extras first:

```bash
uv pip install ".[gliner]"
```

Then discover models, inspect domain defaults, and run zero-shot inference:

```bash
# Build or refresh the model index (scans your models directory)
python -m openmed.zero_shot.cli.index --models-dir /path/to/zero-shot-models

# Inspect default labels per domain
python -m openmed.zero_shot.cli.labels dump-defaults --json

# Run inference with custom labels or domain defaults
python -m openmed.zero_shot.cli.infer \
  --model-id gliner-biomed-tiny \
  --text "Imatinib inhibits BCR-ABL in CML." \
  --threshold 0.55 \
  --labels Drug,Gene

# Smoke-test multiple GLiNER models (requires models/index.json)
python scripts/smoke_gliner.py --limit 3 --adapter
```

Use `OPENMED_ZEROSHOT_MODELS_DIR` to avoid passing `--models-dir` every time. The
CLI utilities share the same default `models/index.json` location bundled in the
package when an external index is not supplied.

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

## Documentation

MkDocs + Material power the official docs served at <https://openmed.life/docs/>.
Every push to `master` rebuilds the site, but you can preview locally in seconds:

```bash
uv pip install ".[docs]"
make docs-serve          # http://127.0.0.1:8008 with live reload
make docs-build          # run the strict production build
make docs-stage          # copies docs into site/ alongside docs/website
python -m http.server --directory site 9000  # preview marketing+docs at http://127.0.0.1:9000
```

To stage a manual deployment (marketing site + docs bundle), run `make docs-deploy`; it builds the docs into `site/docs`,
copies `docs/website/` into `site/`, and pushes that bundle to the `gh-pages` branch.

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
