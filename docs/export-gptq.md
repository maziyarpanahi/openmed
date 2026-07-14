# GPTQ Export

Use GPTQ when your target Hugging Face serving runtime expects AutoGPTQ-style
4-bit checkpoints. GPTQ is useful for runtimes that do not accept AWQ artifacts
or that already standardize on GPTQ loaders. For deployments that explicitly
support AWQ and prefer its activation-aware recipe, use the AWQ export guide
instead.

For Apple apps or Apple Silicon Python inference, prefer the MLX and CoreML
paths described in the MLX backend and CoreML packaging guides. GPTQ is a
PyTorch-oriented export recipe for deployments that already serve Hugging Face
checkpoints with GPTQ-compatible runtimes.

## Install

```bash
uv pip install -e ".[gptq]"
```

The `gptq` extra installs `auto-gptq`. Importing `openmed` does not import
AutoGPTQ; the dependency is loaded only when `quantize_gptq()` is called.
Without the extra, the entry point raises an actionable install error instead
of failing at module import time.

Use a dedicated export environment for GPTQ conversion. AutoGPTQ manages its own
Transformers compatibility window, so do not reuse that environment as the
general OpenMed HF inference runtime unless you have verified the combined
dependency set.

## Quantize

```python
from openmed.torch.calibration import load_quantization_calibration_texts
from openmed.torch.quantize_gptq import quantize_gptq

calib_texts = load_quantization_calibration_texts()

result = quantize_gptq(
    "TinyLlama/TinyLlama-1.1B-Chat-v1.0",
    calib_texts,
    "artifacts/example-gptq",
    bits=4,
    group_size=128,
    desc_act=False,
    revision="main",
)

print(result.quant_config_path)
```

The committed calibration set is synthetic clinical-note style text and is the
same loader used by the AWQ recipe. Keeping AWQ and GPTQ calibration data
identical makes artifact metadata comparable across 4-bit formats. OpenMed
records a SHA-256 digest and sample count in the artifact metadata; it does not
write calibration text into the output directory.

## AWQ vs GPTQ

Choose the recipe by the runtime that will load the artifact:

- Use GPTQ for AutoGPTQ-compatible runtimes and serving stacks that require
  GPTQ quantization metadata.
- Use AWQ for AWQ-compatible runtimes that expect AutoAWQ artifacts.
- Use MLX or CoreML exports for Apple-first local inference paths instead of
  PyTorch weight-only checkpoints.

## Artifact Metadata

The recipe writes `quant_config.json` beside the GPTQ checkpoint. The file
records:

- `bits`
- `group_size`
- `desc_act`
- `calibration_sample_count`
- `calibration_sha256`
- `source_model_id`
- `source_revision`
- the AutoGPTQ quantization config passed to the backend

Pass an immutable model revision whenever possible. If `revision` is omitted,
OpenMed uses the Hugging Face config commit hash when it is available and marks
local directories as `local`.

## Release Expectation

GPTQ export creates the artifact, but it does not certify clinical recall by
itself. Before routing production traffic to a GPTQ artifact, run the existing
eval gates for the target family and compare recall deltas against the
quantized-model limits. Treat any direct-identifier recall regression as a
release blocker.
