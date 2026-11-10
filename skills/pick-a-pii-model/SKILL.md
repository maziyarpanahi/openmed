---
name: pick-a-pii-model
description: "Select an on-device OpenMed PII model from the committed registry by language, runtime format, and size budget, then require recall validation before deployment. Use when an agent must choose a local PII detector for CPU, Apple Silicon, or a mobile export without relying on live model discovery."
---

# Pick an on-device PII model

Use the committed registry to build an offline shortlist. Treat the language
default as the safety baseline, but never treat model size or format as proof
of recall.

## Procedure

1. Identify the input language and script before choosing a model.
2. Choose the runtime: `pytorch` for local CPU/GPU and mobile export sources,
   `mlx-fp` or `mlx-8bit` for Apple Silicon.
3. Read `get_default_pii_model(language)` as the baseline.
4. Filter `get_pii_models_by_language(language)` by runtime and device budget.
5. Prefer the baseline when it fits; otherwise select a compatible candidate.
6. Benchmark the candidate on direct identifiers, critical leakage, scripts,
   and the target quantization before shipping.

## Runnable offline shortlist

This snippet reads only the bundled manifest; it does not download weights.

```python
from openmed import get_default_pii_model, get_pii_models_by_language

LANGUAGE = "en"
TARGET_FORMAT = "mlx-fp"  # Use "pytorch" for CPU or as an export source.
MAX_PARAMETERS_M = 150

baseline_id = get_default_pii_model(LANGUAGE)
models = get_pii_models_by_language(LANGUAGE)

shortlist = [
    (key, info)
    for key, info in models.items()
    if TARGET_FORMAT in info.formats
    and info.size_mb is not None
    and info.size_mb <= MAX_PARAMETERS_M
]
shortlist.sort(
    key=lambda item: (
        item[1].model_id != baseline_id,
        item[1].size_mb,
        item[0],
    )
)

if not shortlist:
    raise RuntimeError("No compatible PII model fits the requested budget")

registry_key, selected = shortlist[0]
print(
    {
        "registry_key": registry_key,
        "model_id": selected.model_id,
        "format": TARGET_FORMAT,
        "parameters_m": selected.size_mb,
        "recommended_confidence": selected.recommended_confidence,
        "is_language_default": selected.model_id == baseline_id,
    }
)
print("Benchmark this candidate against the language default before release.")
```

For Android, Core ML, ONNX, or browser deployment, select a compatible
`pytorch` source and use the target export workflow. Re-run PII recall after
conversion or quantization.

## Selection rules

- Reject an unsupported language instead of silently falling back to English.
- Prefer audited script coverage over a model's name or marketing description.
- Treat parameter count as a rough capacity signal, not download size, latency,
  peak memory, or quality.
- Measure latency and peak memory on the real target device.
- Fail closed when conversion or quantization drops direct-identifier recall or
  introduces residual critical leakage.
- Cache approved weights locally and set offline mode for steady-state use.

## Repository example

Read
[the PII model comparison example](../../examples/pii_model_comparison.py) for
registry inspection and model-by-model inference.
