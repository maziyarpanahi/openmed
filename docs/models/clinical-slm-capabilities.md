# Clinical SLM capability probes

`openmed.models.clinical_slm_capabilities` is a local preflight gate for a
clinical small-language-model package. It answers whether the package's
manifest supports bounded summarization, clinical NLI, or both before clinical
input reaches a model runtime.

The probe reads only a JSON-like manifest. It does not load weights, construct
a tokenizer, inspect patient data, open a socket, download a dependency, or
fall back to a cloud model. It is a deployment capability signal, not a
compliance certification, a medical-device claim, or an autonomous clinical
decision.

## Probe a package

Use an explicit local runtime profile in deployment tests and air-gapped
bootstrap code. This keeps the result deterministic; when the profile is
omitted, known Python-backed features are checked with `find_spec` without
importing them.

```python
from openmed.models.clinical_slm_capabilities import (
    probe_clinical_slm_capabilities,
)

manifest = {
    "supported_tasks": ["clinical-summarization", "nli"],
    "context_limits": {
        "max_context_tokens": 4608,
        "max_input_tokens": 4096,
        "max_output_tokens": 512,
    },
    "quantization": {"scheme": "int4", "bits": 4},
    "required_runtime_features": ["tokenizers"],
    "offline": True,
    "human_review_required": True,
}

report = probe_clinical_slm_capabilities(
    manifest,
    available_runtime_features={"tokenizers"},
)

if not report:
    # The report is safe to attach to a deployment gate or audit record.
    raise RuntimeError(report.to_json(indent=2))
```

The same function accepts a path to a local manifest, a JSON document, or a
mapping. For a directory it reads only these fixed filenames, in order:

1. `clinical-slm-capabilities.json`
2. `clinical-slm-manifest.json`
3. `manifest.json`

No directory scan or remote registry lookup is performed.

## Manifest contract

The task aliases `clinical-summarization`, `summarization`, and `summary` are
normalized to `bounded_summarization`. `clinical-nli` and
`natural-language-inference` are normalized to `nli`. A bounded summary must
declare positive input and output token limits. NLI must declare a positive
input or total context limit. An optional `min_context_tokens` requirement
checks the declared input limit for every requested capability.

Quantization is required metadata. The probe recognizes full precision,
`fp16`, `bf16`, `int2`/`int3`/`int4`/`int8`, and the bounded GGUF schemes
`q4_0`, `q4_k_m`, `q5_k_m`, and `q8_0`. Pass `allowed_quantizations` when a
deployment supports only a subset.

`required_runtime_features` is metadata, not an instruction to install or
download anything. Pass `available_runtime_features` to provide the local
runtime profile. Without it, the probe checks only known Python modules with
`importlib.util.find_spec`; it never imports those modules.

Clinical package manifests are required to set both `offline` (or
`local_only`) and `human_review_required` (or `review_required`) to `true` by
default. A manifest that advertises a cloud/remote fallback or requires a
network is unsupported. These checks keep the probe aligned with local-first
operation and the human-review boundary.

## Safe report and reason codes

`ClinicalSLMCapabilityReport` exposes `supported`, `ready`, `capabilities`,
`reason_codes`, and `manifest_fingerprint`. `to_dict()` and `to_json()` emit
only fixed capability names, reason codes, bounded numeric limits, booleans,
counts, and a SHA-256 fingerprint. Model identifiers, paths, arbitrary
manifest values, prompts, and clinical text are never copied into a report or
exception.

Each capability check has a `reason_codes` list. The most useful codes are:

| Code | Meaning |
| --- | --- |
| `task_not_declared` | The manifest does not advertise the requested task. |
| `context_limit_missing` | No positive input/total context limit is declared. |
| `context_limit_too_small` | The declared input limit is below the deployment requirement. |
| `output_limit_missing` | Bounded summarization has no positive output limit. |
| `quantization_missing` | Quantization metadata is absent. |
| `quantization_invalid` | Quantization metadata is malformed or unknown. |
| `quantization_unsupported` | The declared scheme is outside the deployment allow-list. |
| `runtime_feature_missing` | A required local runtime feature is unavailable. |
| `offline_required` | The package is not explicitly marked local/offline-only. |
| `human_review_required` | The package does not preserve the review gate. |
| `cloud_fallback_forbidden` | The manifest enables a remote fallback. |

Reason codes are deterministic and machine-readable. A blocked report should
be surfaced to an operator or human reviewer; it must not be converted into a
remote inference attempt.

This guide and its tests use synthetic metadata only. No restricted dataset,
credential, model weight, or clinical record is bundled.
