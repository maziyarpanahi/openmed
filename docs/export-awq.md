# AWQ Export

Use AWQ when you need a compact 4-bit Hugging Face artifact for CUDA or CPU
serving and can run an offline calibration step before release. OpenMed keeps
AWQ separate from the base install because the quantization stack is large and
hardware-sensitive.

For Apple apps or Apple Silicon Python inference, prefer the MLX and CoreML
paths described in the MLX backend and CoreML packaging guides. AWQ is the
PyTorch-oriented export recipe for deployments that already serve Hugging Face
checkpoints with AWQ-compatible runtimes.

## Install

```bash
uv pip install -e ".[awq]"
```

The `awq` extra installs `autoawq`. Importing `openmed` does not import
AutoAWQ; the dependency is loaded only when `quantize_awq()` is called. Without
the extra, the entry point raises an actionable install error instead of failing
at module import time.

Use a dedicated export environment for AWQ conversion. AutoAWQ manages its own
Transformers compatibility window, so do not reuse that environment as the
general OpenMed HF inference runtime unless you have verified the combined
dependency set.

## Quantize

```python
from openmed.torch.calibration import load_quantization_calibration_texts
from openmed.torch.quantize_awq import quantize_awq

calib_texts = load_quantization_calibration_texts()

result = quantize_awq(
    "TinyLlama/TinyLlama-1.1B-Chat-v1.0",
    calib_texts,
    "artifacts/example-awq",
    w_bit=4,
    group_size=128,
    revision="main",
)

print(result.quant_config_path)
```

The committed calibration set is synthetic clinical-note style text. It is
deterministic, non-empty, and intended only to drive activation-aware scale
selection. OpenMed records a SHA-256 digest and sample count in the artifact
metadata; it does not write calibration text into the output directory.

## Artifact Metadata

The recipe writes `quant_config.json` beside the AWQ checkpoint. The file records:

- `w_bit`
- `group_size` and `q_group_size`
- `calibration_sample_count`
- `calibration_sha256`
- `source_model_id`
- `source_revision`
- the AutoAWQ quantization config passed to the backend

Pass an immutable model revision whenever possible. If `revision` is omitted,
OpenMed uses the Hugging Face config commit hash when it is available and marks
local directories as `local`.

## Release Expectation

AWQ export creates the artifact, but it does not certify clinical recall by
itself. Before publishing or routing production traffic to an AWQ artifact, run
the existing eval gates for the target family and compare recall deltas against
the quantized-model limits. Treat any direct-identifier recall regression as a
release blocker.
