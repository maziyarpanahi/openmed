# Offline edge-VLM provider contract

`openmed.multimodal.vision.providers.base` defines the Python adapter boundary
for local image understanding. It loads no models, decodes no images, downloads
nothing and adds no dependencies. Backend implementations belong in `mlx`,
`coreml`, `onnx` or `torch`, rather than vision orchestration.

This is a non-diagnostic draft contract. Every result binds the mandatory notice
and `review_required: true`; `reviewed_text(reviewer_confirmed=True)` requires
explicit human confirmation. Outputs must never automatically trigger clinical
decisions. No provider, device, model or clinical workflow is qualified by these
types or by the synthetic tests.

## Typed inputs and outputs

| Type | Contract |
| --- | --- |
| `ImageInput` | Immutable encoded bytes, PNG/JPEG/WebP media type, declared source pixel dimensions and computed SHA-256. No paths, URLs or patient metadata. |
| `ImageRegion` | Image digest and half-open `(x0, y0, x1, y1)` source pixel coordinates, with a top-left origin. |
| `Prompt` | Protected text, explicit seed, output token ceiling and UTF-8 byte ceiling. Sampling must be disabled. |
| `VLMRequest` | One image, prompt and optional immutable region selection; its digest binds all inputs and generation parameters. |
| `VisualEvidence` | Source region plus half-open Unicode code-point offsets in generated text and confidence. No OCR text or copied values. |
| `Confidence` | Finite score in `[0, 1]`, or `None` for unknown. Scores are uncalibrated, not clinical probabilities. |
| `LocalProviderMetadata` | Closed runtime backend, explicit lowercase SHA-256 model and preprocessing digests, local-only and deterministic capability declarations. |
| `RuntimeMetadata` | Mandatory provider provenance, measured duration and actual generated token count. |
| `VLMResult` | Protected draft or shared inference-stage `AbstentionRecord`, with provenance, evidence and a mandatory non-diagnostic notice. |

Regions must belong to the input image and fit its source geometry. If the
request selects regions, evidence must reference those exact regions. Adapters
must reverse any preprocessing transforms before returning evidence. Generated
span offsets are code points, not UTF-8 bytes. Abstentions cannot contain text,
evidence, confidence scores or generated tokens; model and preprocessing digests
remain mandatory even when inference abstains.

## Adapter admission and bounds

Implement the narrow `EdgeVLMProvider` protocol: a side-effect-free `metadata`
property and `infer(request) -> VLMResult`. Call `run_local_vlm(provider, request)`
to check admission before inference and verify the returned result against the
request and pinned provider metadata. Network-dependent, non-local and
non-deterministic declarations are rejected. There is no hosted backend or cloud
fallback, including for Apple Foundation Models/PHI paths. Provision permissively
usable or caller-supplied assets separately before constructing a provider.

The contract caps encoded images at 32 MiB, dimensions at 65,536 pixels each,
area at 100 million pixels, prompt/output text at 64 KiB UTF-8, generation at
4,096 tokens and region/evidence tuples at 256 entries. Request-specific output
ceilings may be lower (defaults: 512 tokens and 4,096 bytes). Runtime counters
are bounded; durations are finite and at most one day. These are admission
ceilings, not a memory planner or a hostile-media decoder. Adapters must verify
encoded formats and dimensions, enforce limits during generation, and report
actual counters rather than invented measurements.

Declarations cannot sandbox arbitrary Python code. Adapter qualification must
independently verify offline initialization/inference, disabled telemetry,
deterministic generation, digests covering the exact model assets and complete
preprocessing configuration, and actual device behavior. Measured durations may
vary across identical runs. Contract serialization and validation are
deterministic; cross-device floating-point model output is not certified.

## Protected content and reporting

Image bytes, prompts and draft text remain protected in memory and are excluded
from object representations. `VLMResult.to_json()` is the supported audit format:
only digests, geometry, offsets, scores, counts, fixed codes and notices appear.
It does not serialize generated text or prompts. The boundary emits no logs and
sanitizes provider errors without retaining a sensitive exception chain.

Do not use generic dataclass serialization, inspect traceback locals, persist
the request/result objects, or log `text`/`data` fields. Hashes remain sensitive
linkage metadata and need access controls. Protected draft text may itself contain
identifiers; this contract does not claim to de-identify generated output.
Output screening belongs to #3813, and review confirmation is not a substitute
for that privacy gate.

## Synthetic example and platform boundary

```python
from openmed.multimodal.vision.providers.base import ImageInput, Prompt, VLMRequest

# Opaque synthetic bytes illustrate the contract, not a valid encoded image.
request = VLMRequest(
    image=ImageInput(data=b"synthetic", media_type="image/png", width=8, height=8),
    prompt=Prompt(text="Describe the synthetic shapes.", max_output_tokens=32),
)
assert len(request.digest) == 64
```

This adds a new Python subpackage without changing existing public interfaces.
The release plan separately assigns Swift evidence/provider mirroring to #3679;
that surface must preserve these coordinate, provenance, abstention and review
semantics before a reference iOS workflow is supported. No Swift VLM backend or
model qualification is claimed here. Pending resource planning (#3633) remains
independent of this contract.

Validation uses synthetic bytes and a deterministic fixture provider, with
network calls denied, provenance mismatch controls, multibyte span checks,
bounded outputs, shared abstentions and content-free diagnostic sweeps:

```sh
.venv/bin/python -m pytest tests/unit/multimodal/vision/providers/test_base.py -q
```
