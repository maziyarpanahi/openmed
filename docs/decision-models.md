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
repositories on **2026-10-24**. Check the repository again before you rely on
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
[Question sets and score levels](api/fixed-option-decisions.md#question-sets-and-score-levels)
and
[System One and OpenAI-compatible Decisions bodies](api/fixed-option-decisions.md#system-one-and-openai-compatible-decisions-bodies).

Two differences matter when you bring an existing client:

- The contract is text only. It does not accept images, video or audio, so
  OpenMed uses multimodal decision models with text or JSON text input only.
- OpenMed's REST route `POST /v1/decisions` takes the OpenMed contract, not the
  OpenAI-compatible Decisions body. A client written for that API cannot call
  it unchanged; convert its body with the `decision_interop` helpers first.

## How a decision runs

1. The access policy checks the request's namespace and purpose and the
   backend's family.
2. The backend returns scores only: one probability per option, one per level
   for a levelled score, or a single value for an unlevelled score. A choice or
   confidence reported by a model server is not used.
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
- **Local only.** A backend identity must declare `local=True` and one of the
  permissive licenses Apache-2.0, BSD-2-Clause, BSD-3-Clause, MIT or MPL-2.0;
  anything else is rejected when the backend is built. In-process backends make
  no network calls after the one-time model download. The server bridge
  reaches loopback or a Unix socket; a private-network origin needs an explicit
  allowlist and is refused for requests marked as containing PHI; public hosts
  are always refused. There is no cloud fallback.
- **Qualify before use.** No model on this page ships a registry default or a
  released calibration. Run `openmed decision-qualify` on your own labelled
  development and held-out evaluation cases, then pass the profiles it emits to
  `decide()` explicitly. See
  [Qualify a backend offline](api/fixed-option-decisions.md#qualify-a-backend-offline).
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
| GLiNER2.5 | `GLiNER2DecisionBackend` | `openmed[gliner2]` | in process, CPU | [GLiNER2.5 decision backend](zero-shot-ner.md#gliner25-decision-backend) |
| GLiClass v4 | `GLiClassDecisionBackend` | `openmed[gliclass]` (torch) or `openmed[onnx-runtime]` (ONNX int8) | in process, CPU | [GLiClass v4 backend](api/fixed-option-decisions.md#gliclass-v4-backend) |
| Local decision server | `DecisionServerBackend` | base package | the model in its own process; OpenMed connects over loopback by default | [Local decision servers](api/fixed-option-decisions.md#local-decision-servers) |
| MedDecider-4B-NLI | `EncoderNLIBackend` (an NLI verifier, not a decision backend) | `openmed[hf]` | in process, CPU | [MedDecider-4B-NLI recipe](clinical/local-backends.md#recipe-meddecider-4b-nli) |

**GLiNER2.5.** Each request becomes one GLiNER2 `classify_text` task over the
caller's options; all six public GLiNER2.5 checkpoints load through the
`gliner2` package with no remote code. The
[GLiNER2.5 checkpoints](zero-shot-ner.md#gliner25-checkpoints) table lists
them. For language routing, the
[multilingual language-decision evaluation](zero-shot-ner.md#multilingual-language-decision-evaluation)
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
process. Serve the model with the server its model card documents, on the same
machine, and connect `DecisionServerBackend` to it. The server must answer the
System One route `POST /v1/systemone` or the OpenAI-compatible
`POST /v1/decisions`; OpenMed does not ship a server for any model. The bridge
sends text or JSON text only, parses the server's probabilities, and lets
OpenMed's calibration decide every question. Declare the license of the
weights you serve; a non-permissive declaration is rejected.

**MedDecider-4B-NLI.** This export of MedDecider-4B is a standard three-class
sequence classifier. It loads through the existing local NLI path and is
qualified with `openmed nli-qualify`, so it serves the NLI verifier in
`verify()` and the guarded brief rather than the decision contract.

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

Checked against each model repository on 2026-10-24. Sizes are parameter
counts from the repository metadata. Inputs are what the model card documents;
OpenMed passes text only.

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
| `vllm-sr/d3-edge` | 0.59B | Apache-2.0 | text or JSON, images, video | Local server through the bridge (text only) |
| `vllm-sr/d3-mini` | 4.5B | Apache-2.0 | text, images, video | Local server through the bridge (text only) |
| `vllm-sr/d3-flash` | 8.4B | Apache-2.0 | text, images, video | Local server through the bridge (text only) |
| `perplexity-ai/pplx-decider-flash-v1.2` | 4.5B | Apache-2.0 | text, images | Local server through the bridge (text only) |
| `sky7350/Mica-v0.5-4B` | 4.7B | Apache-2.0 | text or JSON, up to 32k tokens | Local server through the bridge |
| `h2oai/h2o-lightning-4b` | 4.7B | Apache-2.0 | text, images | Local server through the bridge (text only) |
| `Cloudflare/clef-flash` | 9.4B | Apache-2.0 | text or JSON, images, video | Local server through the bridge (text only) |
| `baselabs/elephant-1` | 9.4B | MIT | text, images | Local server through the bridge (text only) |
| `LiquidAI/d1-omni-600M` | 587M | LFM Open License 1.0 | text, images, English audio up to 30 s | Not supported by default (license) |
| `LiquidAI/d1-3B` | 3.1B | LFM Open License 1.0 | text, images | Not supported by default (license) |

[^decide-size]: The model card states 340M; the repository metadata reports
    486M parameters.

What the bridge models need on the serving side, from their model cards:

| Model | Serving requirements |
| --- | --- |
| `vllm-sr/d3-edge`, `vllm-sr/d3-mini`, `vllm-sr/d3-flash` | Remote code and an exact `transformers` pin, with a server script (`d3_server.py`) shipped next to the weights. The remote code and pin are also why `d3-edge` is not loaded in process. |
| `perplexity-ai/pplx-decider-flash-v1.2` | A CUDA GPU (about 9 GiB) and the package bundled with the weights |
| `sky7350/Mica-v0.5-4B` | A CUDA INT8 server, or the GGUF files on llama.cpp (Apple silicon or CPU) |
| `h2oai/h2o-lightning-4b` | vLLM with the shim bundled with the weights; GGUF builds are also published |
| `Cloudflare/clef-flash` | CUDA, a joint head loaded from the repository, and SGLang |
| `baselabs/elephant-1` | One CUDA GPU and the patched serving runtime set up by the repository's script |

**LiquidAI d1.** `LiquidAI/d1-omni-600M` and `LiquidAI/d1-3B` use the LFM
Open License 1.0, which does not license commercial use by an organization
above its USD 10 million annual-revenue threshold. It is not a permissive
license, so OpenMed rejects these models as decision backends, in process and
through the bridge. Do not declare a different license to get past the check.

MedDecider-4B-NLI is Apache-2.0. The larger MedDecider models (9B, 27B and
31B) are CC BY-NC 4.0 and are not supported.
