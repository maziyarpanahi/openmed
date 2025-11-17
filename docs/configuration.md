# Configuration & Validation

Pairing `OpenMedConfig` with the validation helpers lets you reproduce experiments, keep cache paths predictable, and
guard APIs against malformed inputs.

## OpenMedConfig sources

`OpenMedConfig` reads values in the following order:

1. Explicit keyword arguments when you instantiate it.
2. Environment variables prefixed with `OPENMED_`.
3. YAML file passed via `OPENMED_CONFIG_FILE` (or `openmed_config=` argument).
4. Sensible defaults (CPU device, `~/.cache/openmed`, unauthenticated Hugging Face access).

```python
from pathlib import Path
from openmed.core import ModelLoader, OpenMedConfig

config = OpenMedConfig.from_file(Path.home() / ".config/openmed/config.yaml")
loader = ModelLoader(config=config)
ner = loader.create_pipeline("disease_detection_superclinical", aggregation_strategy="simple")
entities = ner("Dapagliflozin added for HFpEF symptom relief.")
```

### Minimal YAML file

```yaml title="~/.config/openmed/config.yaml"
default_org: OpenMed
device: cuda
cache_dir: ~/.cache/openmed
hf_token: ${HF_TOKEN}  # optional
pipeline:
  aggregation_strategy: simple
  return_all_scores: false
```

Environment variables override YAML values, making it easy to swap devices or cache directories in CI/CD:

```bash
export OPENMED_DEVICE=cuda:1
export OPENMED_CACHE_DIR=/mnt/cache/openmed
```

## Validation helpers

```python
from openmed.utils.validation import (
    validate_input,
    validate_model_name,
)

text = validate_input(
    user_supplied_text,
    max_length=2000,
    allow_empty=False,
    strip=True,
)
model_id = validate_model_name("disease_detection_superclinical")
```

- `validate_input` trims whitespace, enforces max lengths, and raises informative errors for API clients.
- `validate_model_name` normalizes registry aliases and protects service endpoints from arbitrary HF IDs.

## Logging and tracing

```python
from openmed.utils import setup_logging
from openmed.core import ModelLoader

setup_logging(level="INFO", json=True)
loader = ModelLoader()
```

- Use JSON output with your log shipper or disable it during notebooks.
- Combine with `OPENMED_DISABLE_WARNINGS=1` when you want the quietest possible inference loop.

## Cache & device tips

- **CPU-only teams**: keep `device="cpu"` and rely on HF caching. PyTorch installs stay optional unless you add the
  `gliner` extra.
- **GPU nodes**: set `device="cuda"` and optionally `torch_dtype=float16` inside `OpenMedConfig.pipeline`.
- **Shared runners**: point `cache_dir` at an ephemeral volume per job to avoid artifacts leaking between builds.
