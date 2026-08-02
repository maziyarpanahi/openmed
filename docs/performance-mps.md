# Torch MPS Performance

OpenMed's Hugging Face/PyTorch backend auto-selects the best local torch
device when `OpenMedConfig.device` is unset. The order is:

1. `mps` on Apple Silicon when PyTorch reports MPS as available.
2. `cuda` when CUDA is available.
3. `cpu` as the portable fallback.

This applies to `ModelLoader.create_pipeline()`, `ModelLoader.load_model()`,
and the torch privacy-filter route. The MLX backend remains separate; this
page covers the non-MLX torch path.

## Override device selection

Use config for application code:

```python
from openmed.core import ModelLoader, OpenMedConfig

loader = ModelLoader(OpenMedConfig(backend="hf", device="mps"))
ner = loader.create_pipeline("disease_detection_tiny", aggregation_strategy="simple")
```

Use environment variables for process-level overrides:

```bash
export OPENMED_TORCH_DEVICE=mps
# or use the broader compatibility variable:
export OPENMED_DEVICE=cpu
```

`OPENMED_TORCH_DEVICE` takes precedence over `OPENMED_DEVICE`. Explicit
config, such as `OpenMedConfig(device="cpu")`, takes precedence over both.

## MPS tuning defaults

When the resolved torch device is `mps`, OpenMed applies conservative PyTorch
MPS environment defaults before loading the model:

| Variable | Default | Why it is set |
|---|---:|---|
| `PYTORCH_ENABLE_MPS_FALLBACK` | `1` | Allows unsupported MPS ops to run on CPU instead of failing immediately. |
| `PYTORCH_MPS_HIGH_WATERMARK_RATIO` | `1.0` | Keeps allocations within PyTorch's recommended working-set size. |
| `PYTORCH_MPS_LOW_WATERMARK_RATIO` | `0.9` | Encourages earlier adaptive commits and garbage collection. |

Existing values are preserved, so deployment-specific choices from your shell,
service manager, or notebook are not overwritten. See the official
[PyTorch MPS environment variable reference](https://docs.pytorch.org/docs/stable/mps_environment_variables.html)
for the complete list of knobs.

## Caveats

MPS support is broad but not identical to CPU or CUDA. With
`PYTORCH_ENABLE_MPS_FALLBACK=1`, unsupported operations can move to CPU, which
may reduce throughput and introduce extra device transfers. For strict
reproducibility checks, set `device="cpu"` or `OPENMED_TORCH_DEVICE=cpu`.

The resolver recommends `float32` as the safe default dtype on MPS. Only use
`float16` or other reduced precision settings when you have validated recall
and span integrity for the specific model and dataset.
