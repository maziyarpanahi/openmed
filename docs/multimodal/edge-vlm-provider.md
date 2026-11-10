# Local edge VLM provider

`SmolVLMProvider` connects the [edge-VLM contract](vision/edge-vlm-provider-contract.md) to a real local SmolVLM2 checkpoint through MLX. It accepts one opaque RGB-compatible image and a bounded prompt, checks image quality before loading the model, generates greedily, and screens buffered text before releasing a draft. Every result has model and preprocessing digests and requires human review. This is a Python MLX implementation; it does not claim Core ML, Swift, diagnostic-image, or cross-device output parity.

## Installation and asset provenance

On Apple Silicon, install `openmed[edge-vlm]` in a separate environment. The optional backend uses `mlx-vlm`; it does not install weights or contact the Hub during inference. The `edge-vlm` extra conflicts with the legacy `scispacy` and `awq` extras because their NumPy and Transformers requirements differ. Use separate environments for those backends.

The reference is [SmolVLM2-2.2B-Instruct-mlx at revision `844516024a1c4400d34489b89ee067d794e432ed`](https://huggingface.co/mlx-community/SmolVLM2-2.2B-Instruct-mlx/tree/844516024a1c4400d34489b89ee067d794e432ed), converted from [HuggingFaceTB's model](https://huggingface.co/HuggingFaceTB/SmolVLM2-2.2B-Instruct/tree/482adb537c021c86670beed01cd58990d01e72e4). Its pinned card declares Apache-2.0. The reference checkpoint is unquantized; the 4,493,651,795-byte safetensors file has SHA-256 `ed6c59250704f09f921dce1a25e0d4eff611b6c9c53e382a7eb04ce9113f2773`. Download and materialize that snapshot separately into an owner-controlled local directory before using the provider. Do not point it at a mutable working download or symlink-based cache snapshot.

OpenMed's adapter remains Apache-2.0 independently of model-weight terms. `LocalVLMAssets` records a source revision and the digest of a caller-supplied model card or license snapshot. It does not interpret terms, maintain a model-license allowlist, accept terms, or recheck licenses over the network. Users must check their selected weights. Changed files, revisions, or license snapshots produce a new model identity; previous qualification evidence must not be carried over automatically.

The adapter supports one `model.safetensors` plus the model's declarative tokenizer, processor, and chat-template files. It rejects executable/extra configuration files, remote-code declarations, unsupported architectures, quantization configurations, and asset symlinks. It rehashes before loading. The directory must remain immutable while loading; these checks do not sandbox a hostile concurrent writer.

## In-memory use

Supply a task-specific `QualityPolicy` and a locally qualified `GeneratedTextGuard`. The provider has no permissive default detector. A missing, failing, or uncertain detector prevents text release. Same-image identifier commitments can supplement the detector through `provider.infer(request, identifiers=...)`; mismatched commitments abstain before inference. See the [generated-text guard](generated-text-guard.md) for detector and evidence contracts.

```python
from pathlib import Path

from openmed.multimodal.vision.providers.base import run_local_vlm
from openmed.multimodal.vision.providers.local_assets import LocalVLMAssets
from openmed.multimodal.vision.providers.smolvlm import SmolVLMProvider


def reviewed_local_draft(model_dir, revision, request, quality_policy, guard):
    """The application supplies validated input and a qualified local guard."""
    directory = Path(model_dir)
    assets = LocalVLMAssets(directory, revision, (directory / "README.md").read_bytes())
    provider = SmolVLMProvider(assets, quality_policy=quality_policy, guard=guard)
    result = run_local_vlm(provider, request)
    return result  # Show its notice; obtain explicit review before consequential use.
```

`result.to_json()` and `provider.provenance_json()` contain metadata only. The protected draft is excluded from representations and JSON; `result.reviewed_text(reviewer_confirmed=True)` requires explicit caller confirmation. Never log `result.text`, prompts, decoded pixels, or detector inputs. No token IDs, cache of previous conversations, or disk image artifacts are returned. Source-region evidence is empty because this model path cannot substantiate grounded bounding boxes.

Inputs are limited to 1536 pixels per side, 4096 prompt bytes, and 512 output tokens. The contract's encoded-byte and output-byte ceilings also apply. Animated, transparent, malformed, quality-rejected, nontrivially EXIF-oriented, or geometry-mismatched images are withheld; explicit region requests are unsupported. Runtime failures and concurrent model calls return content-free abstentions. Serial execution protects the MLX random seed and processor state. The Python adapter is cooperative code, not a hard memory or execution-time sandbox; run sensitive workloads in a dedicated local worker with operating-system network and resource controls.

The preprocessing identity includes the exact tokenizer/processor files and installed MLX, MLX-VLM, Transformers, tokenizers, Pillow, and NumPy versions. The pinned model processor performs its own resize, splitting, normalization, and tensor layout. The separate [synthetic preprocessing conformance contract](preprocessing-conformance.md) does not certify those operators. A different processor/runtime or quantized model requires fresh evidence.

## Reproducible qualification

The preregistered profile is `tests/fixtures/multimodal/edge_vlm_profile.json`. It fixes the model revision and weight digest, twelve synthetic English heading/color tasks, seven synthetic identifier cases, three greedy repeats, and gates for accuracy, output parity, privacy, abstention, latency, and peak memory. These are bounded engineering checks, not evidence of clinical accuracy or comprehensive PHI recall. The fixture-only detector is not a production privacy detector.

The profile targets an Apple M2 Max with 96 GiB unified memory and macOS 26.5.2. It requires at least 75% accuracy in each task family and 80% overall, no tested identifier leaks, complete invalid-input abstention and same-device repeat parity, warm p95 latency at most 30 seconds, first-request latency at most 120 seconds, peak MLX allocation at most 12 GiB, and peak process RSS at most 16 GiB. Timing gates run only when the optional profile is explicitly enabled; ordinary unit CI does not download a model or enforce machine timing.

```bash
export OPENMED_EDGE_VLM_PROFILE=smolvlm2-v1
export OPENMED_EDGE_VLM_MODEL_DIR=/absolute/path/to/materialized-model
export OPENMED_EDGE_VLM_REPORT=/absolute/path/to/private-report.json
export HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1
python -m pytest tests/integration/multimodal/test_edge_vlm_provider.py -q
```

The opt-in test fails if assets or dependencies are missing. It blocks Python socket connections and DNS resolution, retains only digests, counts, decisions, and measured timings in the report, and checks all gates without changing thresholds after a run. For stronger isolation on macOS, invoke the command through `sandbox-exec -p '(version 1) (allow default) (deny network*)'`. No real patient data or restricted evaluation datasets are included.

### Recorded reference result

On 2026-11-04, the frozen optional runtime passed all gates on the declared M2 Max under operating-system network denial. The profile and executable fixtures were committed before inference in `fe32646c0`; the profile SHA-256 is `4ad18f1dffc2ff227db526ad5c5db6430d4426d3711368458defb98157e3b625`. The run used MLX 0.32.4, MLX-VLM 0.7.11, Transformers 5.22.1, tokenizers 0.23.3, Pillow 13.0.0, and NumPy 2.4.6. Transformers 5.16.1 failed to load the required Pillow image processor, so the optional extra requires the verified newer version.

| Measurement | Observed | Gate |
| --- | --- | --- |
| Heading accuracy | 8/8, each correct across three repeats | At least 75% |
| Color accuracy | 4/4, each correct across three repeats | At least 75% |
| Overall task accuracy | 12/12 | At least 80% |
| Tested identifier leaks | 0 across seven synthetic cases | 0 |
| Invalid-input/missing-detector abstention | 5/5 | 100% |
| Same-device repeat parity | 12/12 | 100% |
| Warm p95 request latency | 1.204 s | At most 30 s |
| First request, including model load | 5.520 s | At most 120 s |
| Peak MLX allocation | 4.936 GiB | At most 12 GiB |
| Peak process RSS | 4.534 GiB | At most 16 GiB |
| Python network attempts | 0 | 0 |

The measured model identity is `bb7c392fd21edc791a9b0811f3ea38daeaef4e4c4f0bdb0187f23a55302271ea`; the runtime-bound preprocessing identity is `e9c985fadbcd97b4df8220132c205742fd5981f4a56309bdedb6b6903052c91d`. The private metadata report's SHA-256 is `2335e92b22f1774cb0bdd9242bf0fe560e95c53d64119505aed3223c8a3111ce`. This evidence covers the small declared engineering profile only. It does not establish clinical performance, comprehensive identifier recall, or latency on other devices.
