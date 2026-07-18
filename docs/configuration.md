# Configuration & Validation

Pairing `OpenMedConfig` with the validation helpers lets you reproduce experiments, keep cache paths predictable, and
guard APIs against malformed inputs.

## OpenMedConfig sources

Construct `OpenMedConfig` directly for explicit in-process settings, or load a
flat TOML file with `load_config_from_file()`. When no path is passed, the
loader checks `OPENMED_CONFIG` and then
`~/.config/openmed/config.toml` (or the matching XDG config directory).

Environment controls are field-specific rather than a generic mapping from
every `OPENMED_*` name. For example, `HF_TOKEN`, `OPENMED_OFFLINE`,
`OPENMED_PROFILE`, and `OPENMED_TORCH_ATTENTION_BACKEND` are supported. When
the device preference is unset or `"auto"`, `OPENMED_TORCH_DEVICE` (or the
legacy `OPENMED_DEVICE`) is checked before automatic **MPS → CUDA → CPU**
selection. Set the model cache with `cache_dir=` or the TOML `cache_dir` key;
`OPENMED_CACHE_DIR` is used by selected deployment and data tooling and is not
a generic `OpenMedConfig` override.

```python
from pathlib import Path
from openmed.core import ModelLoader
from openmed.core.config import load_config_from_file

config = load_config_from_file(Path.home() / ".config/openmed/config.toml")
loader = ModelLoader(config=config)
ner = loader.create_pipeline("disease_detection_superclinical", aggregation_strategy="simple")
entities = ner("Dapagliflozin added for HFpEF symptom relief.")
```

### Minimal TOML file

```toml title="~/.config/openmed/config.toml"
default_org = "OpenMed"
device = "cuda"
cache_dir = "/mnt/cache/openmed"
torch_attention_backend = "auto"
cjk_width_convention = "cjk"
```

Runtime environment controls can select the config path, provide Hub
credentials, or choose a device when the loaded config leaves it automatic:

```bash
export OPENMED_CONFIG=/etc/openmed/config.toml
export HF_TOKEN=hf_xxx
export OPENMED_TORCH_DEVICE=cuda:1
```

## CJK width normalization

`cjk_width_convention="cjk"` is the default. It converts full-width Latin
letters, digits, punctuation, and U+3000 ideographic spaces before PII
detection while preserving offsets into the original text. This lets existing
phone, date, and identifier patterns match full-width input without changing
the returned surface text.

Set `cjk_width_convention="nfkc"` to apply strict per-character NFKC
normalization instead. Both modes retain an explicit source map so expanded
compatibility characters still remap to their original code-point spans.

## PyTorch attention backends

`torch_attention_backend="auto"` is the default. In OpenMed 1.8.1 and later,
automatic mode leaves backend selection to Transformers so it can choose an
implementation supported by both the installed PyTorch runtime and the model
architecture.

Set an explicit backend only when you have verified that the model supports it:

```python
from openmed.core import OpenMedConfig

config = OpenMedConfig(torch_attention_backend="eager")
```

The equivalent environment override is:

```bash
export OPENMED_TORCH_ATTENTION_BACKEND=eager
```

Supported values are `auto`, `eager`, `sdpa`, and `flash_attention_2`. The
`eager` implementation is the compatibility fallback. `sdpa` and
`flash_attention_2` require support from the selected Transformers model in
addition to compatible PyTorch and hardware.

## Local-only offline mode

Set `OPENMED_OFFLINE=1` or instantiate `OpenMedConfig(local_only=True)` when
model files are already present in the configured cache or passed as a local
model path. Offline mode sets the standard cache-only loader flags
(`HF_HUB_OFFLINE=1`, `TRANSFORMERS_OFFLINE=1`, and `HF_DATASETS_OFFLINE=1`) and
passes `local_files_only=True` to Hub-backed model loaders.

```bash
export OPENMED_OFFLINE=1
```

```python
from openmed.core import OpenMedConfig

config = OpenMedConfig(local_only=True, cache_dir="~/.cache/openmed")
```

Download or warm the model cache before enabling this mode. Once active,
OpenMed blocks outbound socket connections during inference and
de-identification. A disallowed connection raises `OfflineModeError` with this
message prefix:

```text
OPENMED_OFFLINE/local_only=True blocks outbound network access after model loading.
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
)
model_id = validate_model_name("disease_detection_superclinical")
```

- `validate_input` trims whitespace, enforces max lengths, and raises informative errors for API clients.
- `validate_model_name` trims the value and accepts an existing local path, a
  bare model name, or one `org/model` identifier. It validates syntax; it does
  not enforce a model allowlist or resolve a registry alias.

## Logging and tracing

```python
from openmed.utils import setup_logging
from openmed.core import ModelLoader

setup_logging(level="INFO", include_timestamp=True)
loader = ModelLoader()
```

- `setup_logging` configures standard Python logging. Add structured logging in
  the host application when needed; OpenMed does not expose a `json=` option.
- Keep raw clinical text, entity values, model outputs, access tokens, and
  reversible mappings out of log records and tracing attributes.

## Cache & device tips

- **CPU-only teams**: keep `device="cpu"`, install a CPU-compatible PyTorch
  runtime for Hugging Face inference, and rely on the configured model cache.
- **GPU nodes**: set `device="cuda"` (or a concrete index such as `"cuda:1"`).
  Pass backend-specific model-loading options to `ModelLoader.create_pipeline`
  only after verifying the selected model and hardware support them;
  `OpenMedConfig` has no `pipeline` field.
- **Shared runners**: point `cache_dir` at an ephemeral volume per job to avoid artifacts leaking between builds.
