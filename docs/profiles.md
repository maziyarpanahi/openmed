# Configuration Profiles

OpenMed supports configuration profiles to quickly switch between different settings for development, production, testing, or custom workflows.

## Built-in Profiles

OpenMed ships with five built-in profiles:

| Profile | Backend | Batch size | Timeout | Use case |
|---------|---------|------------|---------|----------|
| `dev` | Auto | Backend default | 600s | Development and debugging |
| `prod` | Auto | Backend default | 300s | Production deployments |
| `test` | Auto | Backend default | 60s | Testing and CI |
| `fast` | Auto | Backend default | 120s | Quick experiments |
| `low_resource` | CPU ONNX INT8 | 1 | 900s | CPU-only machines with 4-8 GiB RAM |

The `low_resource` profile lazily loads the official small PII model, caps
inference at one note per batch and one worker, and never falls back to the
Torch backend. Install its runtime with `pip install "openmed[onnx-runtime]"`.
See the [low-resource benchmark](benchmarks/low-resource.md) for the memory
envelope and reproduction commands.

## Using Profiles

### Python API

```python
from openmed import OpenMedConfig, analyze_text

# Create config from profile
config = OpenMedConfig.from_profile("dev")
print(config.log_level)  # "DEBUG"

# Apply profile with overrides
config = OpenMedConfig.from_profile("prod", timeout=600)

# Apply profile to existing config
config = OpenMedConfig(default_org="MyOrg")
new_config = config.with_profile("dev")
```

### Environment Variable

Set the `OPENMED_PROFILE` environment variable:

```bash
export OPENMED_PROFILE=prod
python my_script.py
```

```python
from openmed import OpenMedConfig

# Profile is automatically applied
config = OpenMedConfig()
print(config.profile)  # "prod"
```

## Custom Profiles

### Creating Custom Profiles

Programmatic example:

```python
from openmed.core.config import save_profile

settings = {
    "log_level": "INFO",
    "timeout": 450,
    "device": "cuda",
    "use_medical_tokenizer": True,
}

save_profile("myproject", settings)
```

### Loading Custom Profiles

```python
from openmed import OpenMedConfig

config = OpenMedConfig.from_profile("myproject")
```

### Deleting Custom Profiles

```python
from openmed.core.config import delete_profile

delete_profile("myproject")
```

Note: Built-in profiles (dev, prod, test, fast, low_resource) cannot be deleted.

## Profile Storage

Custom profiles are stored in TOML format at:

```
~/.config/openmed/profiles/<name>.toml
```

Example profile file:

```toml
# OpenMed profile: myproject
# Custom profile configuration

log_level = "INFO"
timeout = 450
device = "cuda"
use_medical_tokenizer = true
```

## Profile Precedence

Profile settings are applied in this order (later overrides earlier):

1. OpenMedConfig defaults
2. Environment variables
3. Config file (`~/.config/openmed/config.toml`)
4. Profile settings
5. Explicit arguments

```python
# Profile overrides config file, explicit args override profile
config = OpenMedConfig.from_profile("prod", timeout=900)
# timeout=900 overrides prod's timeout=300
```

## Workflow Examples

### Development Workflow

```python
from openmed import OpenMedConfig, analyze_text

# Use dev profile for detailed logging
config = OpenMedConfig.from_profile("dev")

result = analyze_text(
    "Patient has chronic myeloid leukemia.",
    model_name="disease_detection_superclinical",
    config=config,
)
```

### Production Deployment

```python
from openmed import OpenMedConfig, BatchProcessor

# Production settings with minimal logging
config = OpenMedConfig.from_profile("prod")

processor = BatchProcessor(
    model_name="disease_detection_superclinical",
    config=config,
)

results = processor.process_files(clinical_notes)
```

### CI/CD Testing

```bash
export OPENMED_PROFILE=test
pytest tests/
```

### Switching Profiles Dynamically

```python
from openmed import OpenMedConfig

# Start with dev
config = OpenMedConfig.from_profile("dev")

# Switch to prod for final run
config = config.with_profile("prod")
```
