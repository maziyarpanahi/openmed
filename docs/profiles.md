# Configuration Profiles

OpenMed supports configuration profiles to quickly switch between different settings for development, production, testing, or custom workflows.

## Built-in Profiles

OpenMed ships with four built-in profiles:

| Profile | log_level | timeout | use_medical_tokenizer | Use Case |
|---------|-----------|---------|----------------------|----------|
| `dev` | DEBUG | 600s | True | Development & debugging |
| `prod` | WARNING | 300s | True | Production deployments |
| `test` | DEBUG | 60s | False | Testing & CI |
| `fast` | WARNING | 120s | False | Quick experiments |

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

Note: Built-in profiles (dev, prod, test, fast) cannot be deleted.

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
