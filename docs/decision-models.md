# Local decision models

A decision model answers a question about an input by choosing among answers
the caller declares. It returns a probability for each answer instead of
generating text. Model cards call this request shape "System One" or the
Decision Index schema: a state (the input) plus named questions.

OpenMed runs decision models as backends of its
[fixed-option decision contract](api/fixed-option-decisions.md). The backend
only scores the options. OpenMed applies its own calibration and abstention to
those scores, records the backend's model, revision and license, and returns an
assistive result that always requires human review.

This page covers the question types, the safety rules every backend follows,
and which public models work through which path. The backend sections it links
to hold the installation steps and API details.

Model sizes, licenses and inputs were checked against the Hugging Face model
repositories on **2026-10-30**. Check the repository again before you rely on
a figure.

## Question types

| Question | System One | OpenAI-compatible Decisions | OpenMed mode | Successful result |
| --- | --- | --- | --- | --- |
| Pick one of several answers | `choice` | `choice` | `fixed_choice` | one `choice`, a score per option and a ranking |
| Yes or no | `noul` | `predicate` | `boolean_choice` over `("true", "false")` | one `choice` |
| Rate on an ordered scale | `score` | `score` | `scalar_score` with ordered `levels` | a `scalar_score`, `level_scores` and the `expected_level` |

The contract also has modes without a wire counterpart: `ordered_preference`
returns a full ranking and `multi_label` returns every option above a
threshold. `boolean_choice` accepts any two caller-defined labels.
[Modes and abstention](api/fixed-option-decisions.md#modes-and-abstention)
describes each mode.

Schema `1.1.0` of the contract carries the rest of a System One request: a
`question_set` of up to 32 named questions over one input, per-question
`instructions`, option descriptions, ordered score `levels`, and an input bound
that can be raised to 131,072 characters. `openmed.structured.decision_interop`
converts System One and OpenAI-compatible Decisions bodies to and from OpenMed
requests and results without network I/O. See
[Question sets and score levels](api/fixed-option-decisions.md)
and
[System One and OpenAI-compatible Decisions bodies](api/fixed-option-decisions.md).

Two differences matter when you bring an existing client:

- Text is the default. Schema `1.2.0` also accepts verified local images or
  video when the backend explicitly supports the media type (see
  [local images and video](#local-images-and-video)).
- OpenMed's REST route `POST /v1/decisions` accepts both the OpenMed contract
  and the text-only OpenAI-compatible Decisions body, and answers in the shape
  it received. An OpenAI-compatible body must name the model the service
  decides with in `model`; see
  [one route, two request shapes](api/fixed-option-decisions.md#one-route-two-request-shapes).

## How a decision runs

1. The access policy checks the request's namespace and purpose and the
   backend's family.
2. The backend returns scores only: one probability per option, one per level
   for a levelled score, or a single value for an unlevelled score. A choice or
   confidence reported by a model server does not override probability-based
   decisions. An expected-level-only score needs server confidence; otherwise
   OpenMed abstains.
3. The calibration profile named by `calibration_id` turns the scores into a
   result. A request whose confidence or margin is below the profile's
   thresholds returns `state: abstained`. Backend problems return typed
   `unsupported`, `conflict` or `failure` states, never a fabricated answer.
4. The result records the backend identity (model ID, revision, runtime and
   license), the calibration ID and version, `review.required: true` and
   `autonomous_action: false`.

The bundled profile `openmed.synthetic.fixed_option.v1` (minimum confidence
0.70, minimum margin 0.15) is fitted to synthetic fixtures. It is a tooling
default, not a qualified operating point for any model on this page.

## Safety rules

These rules apply to every backend and every path on this page.

- **Assistive only.** Every result carries the decision advisory, requires
  human review and has `autonomous_action: false`. The result type cannot be
  constructed otherwise.
- **Clinical uses go through `GuardedSuggestion`.** When a decision feeds a
  clinician-facing workflow, wrap each successful result in
  `openmed.clinical.decision_support.GuardedSuggestion`, with the source span
  the decision read and PHI-free provenance. The envelope adds the
  medical-device disclaimer, requires clinician review and keeps
  `autonomous_decision` at `False`. Abstained and failed results produce no
  suggestion; route the case to review instead. Declare the release with the
  [clinical output use policy](clinical/output-use-policy.md), whose default
  rule for `decision_support` needs an `approved` review state.
- **Local only.** A backend must declare `local=True`. OpenMed's adapter code is Apache-2.0. Model weights are obtained separately; their declared license describes the model revision, not the adapter. Users must check the publisher's terms for their intended use. In-process backends make
  no network calls after the one-time model download. The server bridge
  reaches loopback or a Unix socket; a private-network origin needs an explicit
  allowlist and is refused for requests marked as containing PHI; public hosts
  are always refused. There is no cloud fallback.
- **Qualify before use.** No model on this page ships a registry default or a
  released calibration. Run `openmed decision-qualify` on your own labelled
  development and held-out evaluation cases, then pass the profiles it emits to
  `decide()` explicitly. See
  [Qualify a backend offline](api/fixed-option-decisions.md).
  A qualified report means only that the backend met the floors you chose on
  the files you supplied. It is not clinical validation.
- **No autonomous clinical action.** Decisions can flag, rank and route work
  for review. They do not diagnose, order, enroll, contact a patient or approve
  PHI egress, and they add to the deterministic de-identification sweeps rather
  than replace them.

This example runs offline with a synthetic scorer in place of a model. It
decides a yes/no question and wraps a successful result in the guarded
envelope:

```python
# Runnable: offline, synthetic scorer, no model download.
from openmed.clinical.decision_support import build_guarded_suggestion
from openmed.structured import DecisionRequest, EncoderDecisionBackend, decide


def synthetic_scorer(input_text, options, mode):
    # Stands in for a local model: one probability per option, in order.
    return [0.92, 0.08]


backend = EncoderDecisionBackend(
    backend_id="example.local.encoder",
    model_id="example/synthetic-decision-encoder",
    revision="0" * 40,
    license_id="Apache-2.0",
    scorer=synthetic_scorer,
)
note = "Synthetic note: follow-up imaging is planned in three months."
result = decide(
    DecisionRequest(
        mode="boolean_choice",
        input_text=note,
        options=("yes", "no"),
    ),
    backend=backend,
)

if result.state.value == "success":
    suggestion = build_guarded_suggestion(
        {"follow_up_planned": result.choice},
        [{"start": 0, "end": len(note), "label": "decision_input"}],
        result.confidence,
        provenance={
            "model_id": result.backend["model_id"],
            "revision": result.backend["revision"],
            "calibration_id": result.calibration["calibration_id"],
        },
    )
    assert suggestion.requires_clinician_review
    assert suggestion.autonomous_decision is False
else:
    # Abstained, unsupported or failed: no suggestion; send the case to review.
    review_queue = [(result.state.value, result.code)]
```

## Paths into OpenMed

| Path | Backend | Install | Runs | Details |
| --- | --- | --- | --- | --- |
| Deterministic baseline | `DeterministicDecisionBackend` (the default) | base package | in process, no model | [Python](api/fixed-option-decisions.md#python) |
| Your own local scorer | `EncoderDecisionBackend`, `CrossEncoderDecisionBackend`, `SpecialistDecisionBackend` | base package | in process | [Python](api/fixed-option-decisions.md#python) |
| GLiNER2.5 | `GLiNER2DecisionBackend` | `openmed[gliner2]` | in process, CPU | [GLiNER2.5 decision backend](zero-shot-ner.md) |
| GLiClass v4 | `GLiClassDecisionBackend` | `openmed[gliclass]` (torch) or `openmed[onnx-runtime]` (ONNX int8) | in process, CPU | [GLiClass v4 backend](api/fixed-option-decisions.md) |
| Local decision server | `DecisionServerBackend` | base package | the model in its own process; OpenMed connects over loopback by default | [Local decision servers](api/fixed-option-decisions.md) |
| MedDecider-4B-NLI | `EncoderNLIBackend` (an NLI verifier, not a decision backend) | `openmed[hf]` | in process, CPU | [MedDecider-4B-NLI recipe](clinical/local-backends.md#recipe-meddecider-4b-nli) |

Request-aware scorers and the server bridge receive the full question
definition. Legacy three-argument scorers, including the deterministic baseline,
refuse per-request instructions and non-empty option descriptions rather than
dropping them. Native GLiNER2.5 and GLiClass backends also refuse those fields;
configure GLiClass prompts and examples on its backend settings instead.
Levelled scalar questions require a backend that implements their ordered
levels; the native GLiNER2.5 and GLiClass paths return `unsupported`.

**GLiNER2.5.** Each supported question becomes one GLiNER2 `classify_text`
task over the caller's options; all six public GLiNER2.5 checkpoints load through the
`gliner2` package with no remote code. The
[GLiNER2.5 checkpoints](zero-shot-ner.md) table lists
them. For language routing, the
[multilingual language-decision evaluation](zero-shot-ner.md)
measured `fastino/GLiNER2.5-multi-Decide` against the deterministic router. It
is evidence only; routing defaults do not change.

**GLiClass v4.** `knowledgator/gliclass-v4-edge` scores every option in one
forward pass, with an optional instruction prompt and few-shot examples, and
reads the model's learned calibrator. The `gliclass` extra needs
`transformers>=5` and cannot be installed with the `scispacy` extra. The int8
ONNX graph gives flatter probabilities than torch, so fit a calibration
profile for each runtime.

**Local decision servers.** Most 4B to 9B decision models need a CUDA serving
stack, a custom head or a pinned runtime, so OpenMed does not load them in
process. Use a documented compatible server on the same machine, or provide
a local HTTP adapter for a model whose package only exposes an in-process API. Then
connect `DecisionServerBackend` to that endpoint. The server must answer the
System One route `POST /v1/systemone` or the OpenAI-compatible
`POST /v1/decisions`; OpenMed does not ship a server for any model. The bridge
sends text or JSON text, and can forward validated images and video to a compatible local System One endpoint. It parses the server's probabilities, and lets
OpenMed's calibration decide every question. Declare the license of the
model revision you serve. Where available, supply its license file SHA-256 as `license_sha256`.

**MedDecider-4B-NLI.** This export of MedDecider-4B is a standard three-class
sequence classifier. It loads through the existing local NLI path and is
qualified with `openmed nli-qualify`, so it serves the NLI verifier in
`verify()` and the guarded brief rather than the decision contract.

### d3-edge on CPU

The optional `D3EdgeDecisionBackend` accepts text, local images and bounded video. Install the model runtime separately in an isolated environment (`transformers==5.17.0`, CPU PyTorch, torchvision, Pillow, NumPy and einops). Acquire the exact snapshot separately. Use an owner-controlled read-only snapshot and a private, trusted runtime cache. The adapter refuses a different runtime, additional executable files or changed code or model bytes. It does not download a model or accept its terms.

Set `HF_HUB_OFFLINE=1` and `OPENMED_TRUSTED_REMOTE_CODE_MODELS=vllm-sr/d3-edge` before starting Python. This authorizes the pinned repository's custom code to run in the process; use an isolated service account without unrelated credentials. Larger d3 variants still require a compatible local server.

```python
# Fragment: requires a local model snapshot and the optional pinned runtime.
from pathlib import Path
from openmed.structured.decision_d3 import D3EdgeDecisionBackend

backend = D3EdgeDecisionBackend(
    Path("/models/d3-edge/bee5e872f75b9c5d1405767bf244a6948b95c0af")
)
```

### Qualifying an in-process backend

`openmed decision-qualify --backend-factory module:function` imports a
trusted, installed function that returns a backend. Download the checkpoint
first and load it with `local_files_only=True`, so a cache miss fails instead
of reaching the network during qualification:

```python
# Fragment: needs openmed[gliner2] and a checkpoint already in the local cache.
# my_site/decisions.py, passed as --backend-factory my_site.decisions:build_backend
from openmed.ner.families.gliner2_decision import GLiNER2DecisionBackend


def build_backend():
    return GLiNER2DecisionBackend.from_pretrained(
        "fastino/GLiNER2.5-multi-Decide",
        revision="e173bca1f0c4217d7a55b3b0c705d5ece8a0218d",
        local_files_only=True,
        task="request",
    )
```

The command fits thresholds for `fixed_choice`, `boolean_choice`,
`ordered_preference` and `multi_label` cases. It does not cover
`scalar_score` questions; keep score answers as review signals until you have
measured them on your own held-out data.

## Support matrix

Checked against each model repository on 2026-10-30. Sizes are parameter
counts from the repository metadata. Inputs are what the model card documents;
The supported path column distinguishes documented model inputs from OpenMed adapter support.

| Model | Size | License | Inputs | Supported path |
| --- | --- | --- | --- | --- |
| `fastino/gliner2.5-small-v1` | 74M | Apache-2.0 | text (English) | In process: GLiNER2.5 |
| `fastino/gliner2.5-base-v1` | 194M | Apache-2.0 | text (English) | In process: GLiNER2.5 |
| `fastino/gliner2.5-multi-v1` | 287M | Apache-2.0 | text (multilingual) | In process: GLiNER2.5 |
| `fastino/GLiNER2.5-multi-Decide` | 287M | Apache-2.0 | text (multilingual) | In process: GLiNER2.5 |
| `fastino/GLiNER2.5-Decide` | 486M[^decide-size] | Apache-2.0 | text (English) | In process: GLiNER2.5 |
| `fastino/GLiNER2.5-Decide-1B` | 1.19B | Apache-2.0 | text (English) | In process: GLiNER2.5 |
| `knowledgator/gliclass-v4-edge` | 440M | Apache-2.0 | text, instruction prompt, few-shot examples | In process: GLiClass v4 |
| `lion-ai/MedDecider-4B-NLI` | 4.2B | Apache-2.0 | premise and claim pairs (English) | In process: NLI verifier through `runtime="torch"` |
| `vllm-sr/d3-edge` | 0.59B | Apache-2.0 | text or JSON, images, video | In process: pinned CPU adapter; also compatible local System One server |
| `vllm-sr/d3-mini` | 4.5B | Apache-2.0 | text, images, video | Local server through the bridge (text only) |
| `vllm-sr/d3-flash` | 8.4B | Apache-2.0 | text, images, video | Local server through the bridge (text only) |
| `perplexity-ai/pplx-decider-flash-v1.2` | 4.5B | Apache-2.0 | text, images | Compatible local HTTP adapter required (text only) |
| `sky7350/Mica-v0.5-4B` | 4.7B | Apache-2.0 | text or JSON, up to 32k tokens | Local server through the bridge |
| `h2oai/h2o-lightning-4b` | 4.7B | Apache-2.0 | text, images | Local server through the bridge (text only) |
| `Cloudflare/clef-flash` | 9.4B | Apache-2.0 | text or JSON, images, video | Local server through the bridge (text only) |
| `baselabs/elephant-1` | 9.4B | MIT | text, images | Compatible local HTTP endpoint required (text only) |
| `LiquidAI/d1-omni-600M` | 587M | LFM Open License 1.0 | text, images, English audio up to 30 s | In process: pinned d1-omni CPU adapter |
| `LiquidAI/d1-3B` | 3.1B | LFM Open License 1.0 | text, images | No native d1-3B adapter tested |

[^decide-size]: The model card states 340M; the repository metadata reports
    486M parameters.

What the bridge models need on the serving side, from their model cards:

| Model | Serving requirements |
| --- | --- |
| `vllm-sr/d3-edge`, `vllm-sr/d3-mini`, `vllm-sr/d3-flash` | Remote code and an exact `transformers` pin, with a server script (`d3_server.py`) shipped next to the weights. Only the pinned edge revision has an in-process adapter. |
| `perplexity-ai/pplx-decider-flash-v1.2` | Python 3.12+, a CUDA GPU with about 9 GiB for weights plus working memory, and the bundled package; its card demonstrates an in-process API, so provide a compatible HTTP adapter ([model card](https://huggingface.co/perplexity-ai/pplx-decider-flash-v1.2)) |
| `sky7350/Mica-v0.5-4B` | A CUDA INT8 server, or the GGUF files on llama.cpp (Apple silicon or CPU) |
| `h2oai/h2o-lightning-4b` | vLLM with the shim bundled with the weights; GGUF builds are also published |
| `Cloudflare/clef-flash` | CUDA, a joint head loaded from the repository, and SGLang |
| `baselabs/elephant-1` | One CUDA GPU and the patched OneJev runtime installed by `setup.sh`; supply a compatible local HTTP endpoint ([model card](https://huggingface.co/baselabs/elephant-1)) |

**LiquidAI d1.** The optional `D1OmniDecisionBackend` accepts text, images, or one English voice-command clip from the pinned d1-omni-600M snapshot. OpenMed records the model ID, immutable revision, license label and SHA-256 of the [license at the tested revision](https://huggingface.co/LiquidAI/d1-omni-600M/blob/02b55d7076f15129e59ab3f94783f32c4b088674/LICENSE). These are provenance, not permission for a particular use. Review the terms again when changing revisions. The adapter refuses changed model, code or license files. OpenMed does not bundle weights or upstream model code and the checks on shipped code and dependencies still apply.

Use an isolated offline environment with Transformers 5.17.0, CPU PyTorch, matching torchvision, Pillow and NumPy. Image inference requires torchvision. Download the model yourself, review its code and terms, set `HF_HUB_OFFLINE=1` and `OPENMED_TRUSTED_REMOTE_CODE_MODELS=LiquidAI/d1-omni-600M` before launching Python, and supply an absolute local snapshot path. A custom model's code runs with your process permissions. A different model revision needs its own reviewed pin. d1-3B is not qualified by this adapter.

```python
# Fragment: local snapshot and verified synthetic English voice-command WAV.
from pathlib import Path
from openmed.structured.decision import DecisionMode, DecisionRequest, decide
from openmed.structured.decision_d1 import D1OmniDecisionBackend
from openmed.structured.decision_media import DecisionMedia

backend = D1OmniDecisionBackend(Path("/models/d1-omni-600M/02b55d7076f15129e59ab3f94783f32c4b088674"))
clip = DecisionMedia(local_wav_bytes, media_type="audio/wav", byte_size=len(local_wav_bytes), sha256=expected_digest)
request = DecisionRequest(DecisionMode.BOOLEAN_CHOICE, "Synthetic intake desk", options=("true", "false"), instructions="Does this caller ask to reschedule?", media=(clip,))
result = decide(request, backend=backend)  # Route for human review only.
```

Audio is limited to 30 seconds of mono or stereo signed 16-bit PCM WAV. Approved standard sample rates are normalized to 16 kHz mono in memory (non-native rates require SciPy); overlong clips must be split explicitly. Audio result warnings include `audio_english_only`. The adapter does not transcribe. Mixed image/audio, multiple audio clips, video and audio forwarding over the System One bridge are unsupported.

MedDecider-4B-NLI is the tested NLI export. No native adapter is qualified here for the larger MedDecider models.

## On-device encoder runtimes

OpenMedKit for Swift, OpenMedKit Android and the `openmed` web package run
encoder decision models on the device, with no network call and no cloud
fallback. Each kit ships the same runtime as the Python GLiClass path
(`GLiClassOnnxRunner` with `decide`): it validates a decision request
(schema `1.0.0`, `1.1.0` or `1.2.0`, same field names), applies the access
policy and calibration profile, renders the GLiClass uni-encoder input, runs
the encoder and returns a result that validates against the `1.2.0` result
schema, including `review.required: true` and `autonomous_action: false`.

| Kit | Runtime | Encoder adapter | Tokenizer |
| --- | --- | --- | --- |
| Swift | `OpenMedEncoderDecisionRuntime` | `OpenMedCoreMLDecisionEncoder` over a Core ML `MLModel` | `OpenMedTransformersDecisionTokenizer` (swift-transformers) or any `OpenMedDecisionTokenizer` |
| Kotlin | `EncoderDecisionRuntime` | `OnnxDecisionEncoder` (ONNX Runtime Android, CPU) | any `DecisionTokenizer` |
| Web | `EncoderDecisionRuntime` | `createOrtDecisionEncoder` or `loadOrtWebDecisionEncoder` (ONNX Runtime Web, local paths only) | any `DecisionTokenizer`, such as a wrapped Transformers.js tokenizer |

The encoder contract is the one the Python runner enforces: `input_ids` and
`attention_mask` of shape `[1, tokens]` in, a `[1, labels]` `logits` tensor
out, one calibrated logit per `<<LABEL>>` slot. The tokenizer must add the
checkpoint's special tokens and must not truncate or pad.

The runtimes keep the Python behaviour:

- `fixed_choice`, `boolean_choice` and `ordered_preference` use a softmax over
  the options, `multi_label` a sigmoid per option, and `scalar_score` scores
  the runtime's score levels and returns the expected level scaled to `[0, 1]`.
- The instruction prompt, few-shot examples and score levels are runtime
  configuration. A request that carries `instructions`, option descriptions or
  `levels` returns `unsupported` with `backend_unsupported` instead of being
  scored without its wording.
- GLiClass control tokens in the input or options, and inputs over the token
  budget, return `unsupported`; inputs are never truncated. A label-slot
  mismatch or short or non-finite logits return `conflict`; an encoder error
  returns `failure` with `backend_failure`; a slow encoder returns `failure`
  with `backend_timeout`.
- Calibration profiles are plain objects with the eight result fields
  (`calibration_id` to `out_of_domain_digest`). Pass the profiles a device may
  use; a request selects one by `calibration_id`, and an unknown ID returns
  `unsupported` with `calibration_unavailable`. Without profiles the runtime
  uses `openmed.synthetic.fixed_option.v1`.
- The runtimes are text-only. A request with `media` returns `unsupported`
  with `backend_unsupported`; media bytes are checked for shape only and never
  decoded.
- Contract violations throw a typed error with a `kind`
  (`invalid_request`, `unsupported_contract` or `invalid_configuration`) and a
  fixed `reason` code. Errors and request descriptions never contain the input,
  options, prompt, examples or file paths.

```swift
// Swift: a compiled Core ML export and its tokenizer.json, both local.
let runtime = try OpenMedEncoderDecisionRuntime(
    encoder: OpenMedCoreMLDecisionEncoder(model: try MLModel(contentsOf: compiledModelURL)),
    tokenizer: OpenMedTransformersDecisionTokenizer(tokenizer),
    identity: try OpenMedDecisionBackendIdentity(
        backendID: "openmed.gliclass.v4.coreml", runtime: "coreml",
        modelID: "knowledgator/gliclass-v4-edge", revision: pinnedRevision, licenseID: "Apache-2.0"),
    calibrationProfiles: [try OpenMedDecisionCalibrationProfile(json: profileJSON)])
let result = try runtime.decide(json: requestJSON)  // Route for human review only.
```

```kotlin
// Kotlin: a local ONNX export and a tokenizer wrapper.
val runtime = EncoderDecisionRuntime(
    encoder = OnnxDecisionEncoder(File(modelDirectory, "model_q8.onnx")),
    tokenizer = tokenizer,
    identity = DecisionBackendIdentity(
        "openmed.gliclass.v4.onnxruntime-android", "onnxruntime-android",
        "knowledgator/gliclass-v4-edge", pinnedRevision, "Apache-2.0",
    ),
    calibrationProfiles = listOf(DecisionCalibrationProfile.fromJson(profileJson)),
)
val result = runtime.decide(requestJson) // Route for human review only.
```

```ts
// Web: a same-origin ONNX export and ONNX Runtime Web assets.
const runtime = new EncoderDecisionRuntime({
  encoder: await loadOrtWebDecisionEncoder({ modelPath: "/models/gliclass/model_q8.onnx", assetPath: "/ort/" }),
  tokenizer,
  identity: { backend_id: "openmed.gliclass.v4.onnxruntime-web", runtime: "onnxruntime-web",
    model_id: "knowledgator/gliclass-v4-edge", revision: pinnedRevision, license_id: "Apache-2.0" },
  calibrationProfiles: [profile],
});
const result = await runtime.decide(request); // Route for human review only.
```

`tests/fixtures/parity/decision_encoder_v1.json` holds the shared vectors:
requests, encoder inputs, logits and full expected results produced by the
Python reference path over a tiny randomly initialised encoder and word-level
tokenizer built in `tests/unit/structured/test_decision_encoder_parity.py`.
All three kits replay the same tokenizer and encoder from the fixture's
weights, compare encoder inputs exactly and scores within `1e-6` (logits within
`1e-5`, because encoders emit float32). Regenerate the fixture with
`python -m tests.unit.structured.test_decision_encoder_parity --write`.

Two small differences remain. Option uniqueness folds case by upper- then
lower-casing instead of Python's `casefold`, and the web runtime cannot tell an
integer `timeout_ms` from the same value written with a decimal point. The
vectors use synthetic weights only: no real decision checkpoint, Core ML
conversion or device latency and memory figure is part of this evidence.

## Local images and video

`DecisionMedia` accepts bytes or an absolute local file from a trusted caller, with the declared MIME type, byte count and expected SHA-256. File paths never cross the wire. JSON decision requests carry inline base64 only; System One uses `images` and `videos` lists of base64 data URLs. Remote URLs are refused. A backend must explicitly support each media type: text-only backends return `unsupported` without scoring. The local bridge sends media only over loopback or a Unix socket to the System One endpoint. Private-network endpoints and the text-only OpenAI Decisions protocol refuse it.

```python
# Fragment: needs a trusted local image and its previously verified capture digest.
from pathlib import Path
from openmed.structured.decision import DecisionMode, DecisionRequest
from openmed.structured.decision_media import DecisionMedia

asset = DecisionMedia(Path("/local/review.png"), media_type="image/png",
                      sha256=expected_capture_digest, byte_size=expected_byte_count)
request = DecisionRequest(mode=DecisionMode.BOOLEAN_CHOICE, input_text="Review the image.",
                          media=(asset,))
```

The Python adapter preflights original bytes, dimensions, frames and duration under explicit asset limits. It accepts single-frame PNG/JPEG with Pillow; it applies EXIF orientation, discards metadata including ICC/EXIF/text, and encodes the decoded pixels as PNG. MP4 support requires locally installed FFmpeg and ffprobe, including a lossless RGB H.264 encoder. Both run through bounded in-memory pipes. A short, constant-rate, single video track (H.264, MPEG-4 or MJPEG) is decoded to RGB, re-encoded without source metadata, and decoded again to verify every frame and its timing. Audio, subtitles, non-square pixels, rotated tracks, incomplete timestamps, unsupported codecs and inputs that require seeking beyond a pipe are refused. No recording is written to a temporary file.

The limits are four items per request, 8 MiB per item and 16 MiB in total. Each frame has a two-megapixel ceiling; clips are limited to 64 frames, 16 million total pixels and 30 seconds. The model may impose stricter limits. `request.safe_metadata()` records only sanitized digest, MIME and size for each item. `request.to_dict()`, `data_url()` and `protected_bytes()` contain protected media and must never be logged. Removing metadata does not remove identifiers visible in a frame; hosts own consent, pixel de-identification and review. Audio decisions use the d1-omni adapter described above.
