# Local backends and typed outcomes

Configure a local backend explicitly for [summarization](summarization.md),
[NLI verification](nli-verification.md) and a
[guarded brief](clinical-brief.md). Existing independently qualified local
checkpoints or caller-owned providers can satisfy these APIs; training or
publishing a new model is not a library-release prerequisite. Missing providers
fail closed. Remote provider names and URLs are rejected, with no cloud fallback.

The examples below use only synthetic text, metadata and runtime doubles. They
run without model downloads or optional inference packages. Synthetic scores,
calibration identifiers, review transitions and tiny artifact files demonstrate
contracts; they are **not qualified models, empirical calibration, clinical
validation or permission to approve real evidence**.

## Explicit extraction and trusted local summarizers

For standalone summarization, `model="extractive"` selects the deterministic CPU
baseline. Within a reviewed brief it selects whole sentences by fact coverage
and the independent omission and length rules; an infeasible selection is
refused without partial output. Omitting `model`
selects the registered, pinned MLX summarizer; extraction is never an automatic
fallback for missing MLX assets. For raw input, `summarize()` first de-identifies
locally and may therefore require a cached PII model even in extractive mode.
Use `summarize_deidentified()` only with a genuine completed
`DeidentificationResult`; a plain string is rejected by its ordering guard.

This synthetic fixture contains no identifiers and needs no PII model:

```python
from datetime import datetime

from openmed.clinical.summarize import summarize_deidentified
from openmed.core.pii import DeidentificationResult

text = "A cough is present. Symptoms improved. Follow-up was arranged."
deidentified = DeidentificationResult(
    original_text=text,
    deidentified_text=text,
    pii_entities=[],
    method="mask",
    timestamp=datetime(2026, 1, 1),
)
extractive = summarize_deidentified(deidentified, model="extractive")
assert extractive.summary == text
assert extractive.backend == "deterministic-extractive"


class SyntheticLocalSummarizer:
    def summarize(self, source, *, mode="bhc"):
        assert mode == "bhc" and source == text
        return "A cough is present."


custom = summarize_deidentified(deidentified, model=SyntheticLocalSummarizer())
assert custom.summary == "A cough is present."
assert custom.backend == "caller-supplied-local"
assert custom.leakage_check.passed
```

A callable `(deidentified_text) -> str` or an object exposing
`summarize(deidentified_text, *, mode=...) -> str` wraps an existing local
runtime. OpenMed passes only de-identified text, checks output size and source
identifier leakage, and blocks outbound Python socket calls during execution.
The callable is **trusted application code**, with the process's authority;
this guard does not isolate native code, subprocesses, files or other services.
Keep it local and quiet, bound its resource use, and qualify its actual model.

`summary` and `SummarizationResult.to_dict()` contain generated text and need
protected handling. For diagnostics, project controlled backend identifiers,
template digests and `leakage_check.to_dict()`; never log original text, prompts,
model responses or arbitrary upstream exception messages.

## Registered MLX and cache provisioning

`model="mlx"`, `model="maple"` and the registered summarizer alias resolve to
the same reviewed repository and immutable revision. `MLXSummarizerBackend`
accepts a memory budget; arbitrary paths and unregistered model strings are
rejected at this clinical entry point. Constructing it does not load weights.
The first inference requires the optional MLX runtime and a pre-provisioned
Hugging Face model snapshot; `snapshot_download(..., local_files_only=True)`
performs cache lookup without downloading.

Provision runtime packages and the exact registry-selected snapshot before
processing clinical input, using the commands in the
[summarization guide](summarization.md) and the
[offline bootstrap guide](../models/bundled-offline.md). Preserve the immutable
revision and the cache location selected by the deployment environment. The
summarizer's Hugging Face snapshot cache and the PII loader's OpenMed cache are
separate; placing a PII model in one does not provision summarizer weights in
the other. A missing cache must not trigger a remote attempt or implicit
extraction. No Hugging Face Space deployment or visibility change is involved.

Current MLX admission reads the cached config and weight sizes, checks declared
capabilities and a conservative memory budget, then loads the local runtime.
It reserves output context before loading and checks actual tokenizer length
before generation. It does not establish model quality or measured peak memory.
The generic artifact-manifest verifier is a separate caller-owned gate in the
current released implementation; a cache pin alone does not verify every file's
digest. Keep provisioned files immutable through admission and runtime loading.

## Local encoder NLI with an explicit class mapping

`EncoderNLIBackend` needs the checkpoint's class-index mapping and
`NLIThresholds` from its independently validated calibration. Do not infer that
class zero means entailment, or treat default thresholds or
`calibrated=True` as empirical proof. The three mapped states must be
`entailment`, `neutral` and `contradiction` at indices `0`, `1` and `2`.
The encoder produces a selective decision: low-confidence or overlong pairs
become `abstention`, preserving human review.

This injected loader mimics the local Torch adapter without importing Torch:

```python
from types import SimpleNamespace

from openmed.clinical.nli import verify
from openmed.clinical.nli_backends import EncoderNLIBackend
from openmed.clinical.nli_gate import NLIThresholds


class SyntheticEncoderLoader:
    def __init__(self):
        self.calls = []

    def load_local_sequence_classifier(self, model_ref, *, revision, runtime):
        self.calls.append((model_ref, revision, runtime))
        return {
            "tokenizer": lambda *args, **kwargs: {"input_ids": [[1, 2]]},
            "model": lambda **kwargs: SimpleNamespace(
                logits=[SimpleNamespace(tolist=lambda: [5.0, 0.0, 0.0])]
            ),
        }


loader = SyntheticEncoderLoader()
encoder = EncoderNLIBackend(
    "synthetic-local-checkpoint",
    revision="a" * 40,
    runtime="torch",
    label_mapping={"0": "entailment", "1": "neutral", "2": "contradiction"},
    thresholds=NLIThresholds(
        entailment=0.9,
        contradiction=0.9,
        margin=0.05,
        calibration_id="synthetic-docs-only",
        calibration_method="synthetic-fixture",
    ),
    loader=loader,
)
assert loader.calls == []  # Loading is lazy.
verdicts = verify(["A cough is present."], "A cough is present.", backend=encoder)
assert verdicts[0]["label"] == "entailment"
assert verdicts[0]["backend_id"] == "local-encoder"
assert loader.calls == [("synthetic-local-checkpoint", "a" * 40, "torch")]
```

Without an injected loader, the existing `ModelLoader` uses its cache-only local
sequence-classifier path with `runtime="torch"` or `runtime="onnx"`.
For a Hub model, supply a pinned revision and provision that snapshot separately;
local checkpoint directories also need protected permissions and verified
contents. Registry resolution requires a released, permissively licensed NLI
entry with a pinned revision, mapping and calibration metadata. If no qualified
default is registered, `verify(..., backend=None)` raises `LocalNLIError`.
It does not switch to the development heuristic automatically.

`verify()` accepts trusted local callables or `.predict()` objects returning
`label`, `score` and a safe `backend_id`. It returns controlled decisions and
review flags, omitting source and claim text. Explicit `backend="heuristic"`
is a development baseline, not a qualified encoder or a brief provider.
Empty claims return an empty list without resolving a backend; absence of a
verdict is not a successful verification.

## Artifact, capability and memory gates before construction

For a caller-owned runtime, verify the [artifact manifest](../models/clinical-slm-manifest.md),
probe the separate [capability metadata](../models/clinical-slm-capabilities.md)
and perform [memory preflight](../models/clinical-slm-memory-preflight.md)
**before** constructing a tokenizer, weight tensor or runtime. The strict
artifact manifest and capability manifest have different schemas; do not add
context/runtime fields to the artifact manifest. Pin the expected manifest
digest through a trusted distribution channel. A self-digest is not authenticity.

The following tiny files are synthetic contract fixtures, not loadable weights:

```python
import hashlib
from pathlib import Path
from tempfile import TemporaryDirectory

from openmed.models.clinical_slm_capabilities import probe_clinical_slm_capabilities
from openmed.models.clinical_slm_manifest import (
    ClinicalSLMArtifact,
    ClinicalSLMArtifactManifest,
    ClinicalSLMQuantization,
    load_clinical_slm_manifest,
    verify_clinical_slm_package,
)
from openmed.models.clinical_slm_memory import (
    ClinicalSLMRuntimeProfile,
    preflight_clinical_slm_memory,
)

payloads = {
    "weights": b"synthetic-weights",
    "tokenizer": b"synthetic-tokenizer",
    "templates": b"synthetic-template",
    "quantization": b"synthetic-int4",
}
manifest = ClinicalSLMArtifactManifest(
    model_id="OpenMed/Synthetic-Clinical-SLM",
    revision="a" * 40,
    components=tuple(
        ClinicalSLMArtifact(
            component=name,
            path=f"{name}.bin",
            sha256=hashlib.sha256(payload).hexdigest(),
            size_bytes=len(payload),
        )
        for name, payload in payloads.items()
    ),
    quantization=ClinicalSLMQuantization(scheme="int4", bits=4),
    licenses={name: "MIT" for name in payloads},
    supported_tasks=("clinical-summarization",),
)
capabilities = {
    "supported_tasks": ["clinical-summarization"],
    "context_limits": {
        "max_context_tokens": 512,
        "max_input_tokens": 384,
        "max_output_tokens": 128,
    },
    "quantization": {"scheme": "int4", "bits": 4},
    "required_runtime_features": ["tokenizers"],
    "offline": True,
    "human_review_required": True,
    "cloud_fallback": False,
}
with TemporaryDirectory(prefix="openmed-backend-docs-") as directory:
    package = Path(directory)
    for name, payload in payloads.items():
        (package / f"{name}.bin").write_bytes(payload)
    (package / "clinical-slm-manifest.json").write_text(manifest.to_json())
    checked = load_clinical_slm_manifest(package)
    assert checked.manifest_digest == manifest.manifest_digest
    integrity = verify_clinical_slm_package(package, checked)
    capability = probe_clinical_slm_capabilities(
        capabilities,
        required_tasks=["clinical-summarization"],
        available_runtime_features=["tokenizers"],
    )
    memory = preflight_clinical_slm_memory(
        {"weights_bytes": len(payloads["weights"])},
        ClinicalSLMRuntimeProfile(
            memory_budget_bytes=1_000_000,
            headroom_bytes=100_000,
            context_tokens=512,
            cache_bytes_per_token=16,
            context_bytes_per_token=8,
            batch_bytes=0,
            runtime_overhead_bytes=1_000,
        ),
    )
    if not (integrity.verified and capability.supported and memory.accepted):
        raise RuntimeError("synthetic_backend_admission_refused")
    constructed = {"backend_id": "synthetic-docs-only"}  # Trusted runtime double.
assert constructed == {"backend_id": "synthetic-docs-only"}
assert not package.exists()
```

Disk artifact verification requires POSIX no-follow/directory-descriptor support
and fails closed on unsupported platforms. Capability probing validates declared
metadata, not execution or model quality; memory preflight estimates resource
use, not measured peak allocation. Rejected capability/memory reports are normal
decisions with fixed reason codes, distinct from malformed-input exceptions.
Keep artifacts immutable until loading completes and measure the actual device.

## Wire reviewed evidence into `BriefContext`

Prefer a callback returning **three calibrated class probabilities** and the
matching `calibration_id`. The current composer also accepts the NLI gate's
legacy `label`/`score` mapping with that matching identifier. In that compatibility
path, the gate splits the residual probability equally between the other two
classes; those residuals are not measured class probabilities or evidence of
empirical calibration. A standalone verifier decision can contain abstention
labels and does not automatically satisfy this callback contract. Use an
independently qualified probability adapter instead of manufacturing calibration
from a selected label or score.
An existing local runtime can expose a separately qualified probability adapter.
The [offline NLI qualification protocol](../evaluation/nli-calibration.md)
binds caller-owned artifact bytes, mapping, runtime, separated development and
held-out data, and policy. Only a live qualifier-issued receipt can construct
`bind_qualified_nli()`; decoded audit JSON cannot authorize a callback.
Supply a local privacy detector too; an empty detector result is meaningful only
when the detector itself has been qualified.

This self-contained example creates review transitions only for synthetic
evidence. Production application review stores must provide actual reviewed
history and enforce their own record authorization. The current evidence packet
remains synthetic-only. The separate, opt-in `ReviewedLocalBriefContext`
requires current source custody, independently stored review authority and an
unexpired, unrevoked receipt; see [reviewed-local admission](clinical-brief.md).
This example does not manufacture that authority.

```python
import hashlib
import json
from datetime import datetime

from openmed.clinical.brief import (
    BriefContext,
    BriefFact,
    brief_policy_fingerprint,
    build_clinical_brief,
)
from openmed.clinical.evidence_packet import (
    build_evidence_packet,
    fingerprint_evidence_review,
)
from openmed.clinical.nli_gate import NLIThresholds
from openmed.clinical.review_state_machine import (
    ReviewState,
    ReviewStateMachine,
    make_opaque_event_id,
)
from openmed.core.pii import DeidentificationResult

sentences = (
    "The admission problem was dehydration.",
    "The discharge diagnosis was dehydration.",
    "Symptoms improved after fluids.",
)
fields = ("admission_reason", "discharge_diagnoses", "hospital_course")
text = " ".join(sentences)
facts = tuple(
    BriefFact(
        f"synthetic:ref-{index}", field, "affirmed", "certain", "recent", "patient"
    )
    for index, field in enumerate(fields)
)
policy = brief_policy_fingerprint(text, facts)
rows = []
for index, sentence in enumerate(sentences):
    reference_id = f"synthetic:ref-{index}"
    start = text.index(sentence)
    row = {
        "reference_id": reference_id,
        "source_id": "synthetic:document",
        "start": start,
        "end": start + len(sentence),
        "policy_fingerprint": policy,
        "review_state": "approved",
        "synthetic": True,
        "verified": True,
    }
    fingerprint = fingerprint_evidence_review(
        **{
            key: row[key]
            for key in (
                "reference_id",
                "source_id",
                "start",
                "end",
                "policy_fingerprint",
            )
        }
    )
    machine = ReviewStateMachine()
    for state in (ReviewState.IN_REVIEW, ReviewState.APPROVED):
        machine.transition(
            state, make_opaque_event_id((reference_id, state.value)), fingerprint
        )
    row["review_transitions"] = machine.transitions
    rows.append(row)
thresholds = NLIThresholds(
    calibration_id="synthetic-docs-only", calibration_method="synthetic-fixture"
)


def synthetic_probabilities(premise, hypothesis):
    assert premise == hypothesis
    return {
        "entailment": 1.0,
        "contradiction": 0.0,
        "neutral": 0.0,
        "calibration_id": thresholds.calibration_id,
    }


content_digest = (
    "sha256:"
    + hashlib.sha256(
        json.dumps(
            text, sort_keys=True, separators=(",", ":"), ensure_ascii=True
        ).encode()
    ).hexdigest()
)
context = BriefContext(
    packet=build_evidence_packet(rows, policy_fingerprint=policy),
    content_digest=content_digest,
    facts=facts,
    nli_predict=synthetic_probabilities,
    thresholds=thresholds,
    privacy_detector=lambda value: [],  # Synthetic test-only detector double.
)
deidentified = DeidentificationResult(text, text, [], "mask", datetime(2026, 1, 1))
brief = build_clinical_brief(deidentified, model="extractive", context=context)
assert brief.refusal_reason is None
audit = brief.to_dict()
assert audit["status"] == "needs_review"
assert audit["envelope"]["requires_human_review"]
assert text not in json.dumps(audit)
```

The content digest follows the composer's canonical JSON-string encoding.
The policy fingerprint binds text, reviewed axes and profile version; changed
content or annotations require new review. Generation must match unique reviewed
atomic source spans. Successful output still has `needs_review`; it is not
approval or a clinical decision. `brief.summary` and `to_response()` are
protected output, while `to_dict()` is the value-free audit view. Digests bind
records; they are not authorization or anonymization.

## Backend exceptions

The table covers public exception classes defined by `summarize`,
`summarize_backends`, `nli` and `nli_backends`, plus the separate admission
modules and optional-runtime error. Its drift test discovers exports rather
than keeping a second list of backend errors. Catch subclasses before base
classes. Never serialize arbitrary exception objects or their upstream context.

| Exception | Trigger at this boundary | Remediation |
|---|---|---|
| `ExtractiveSelectionError` | Reviewed fact coverage cannot satisfy the independent omission/length rules, or bounded selection is exhausted | Inspect its value-free selection diagnostics and re-review the budget/evidence; no partial summary is returned |
| `NLIQualificationError` | Invalid, unavailable, drifting or unqualified caller-owned artifact/receipt inputs | Reproduce the offline qualification protocol with trusted local inputs; do not promote an audit record into authority |
| `BriefInterrupted` | Direct summarization reaches a cancelled or expired cooperative checkpoint | Discard the late result; inspect only `cancelled` or `deadline_exceeded` and start a new request if appropriate |
| `LocalSummarizerError` | Unregistered/invalid backend, rejected admission or inference, invalid output, or trusted callback failure | Check fixed local configuration, artifact/runtime availability, limits and provider contract; keep the failure closed |
| `RemoteSummarizerError` | A remote provider name or colon-bearing backend string at summarizer resolution | Select a registered local alias or trusted local callable; never send the note to that provider |
| `SummarizationOrderError` | Guarded stage receives raw/invalid de-identification input or retained source identifiers | Complete and validate local de-identification before generation |
| `SummarizationLeakageError` | Generated output contains detected source identifiers | Discard output and investigate the local provider using synthetic reproduction |
| `LocalNLIError` | Missing/unqualified default or alias, invalid encoder configuration, unavailable checkpoint or inference failure | Provision and qualify the local checkpoint, mapping, calibration and runtime; keep claims unverified |
| `RemoteNLIBackendError` | A remote NLI provider name, URL or `api.` reference | Use a registered local checkpoint or trusted local predictor |
| `MissingOptionalDependencyError` | Built-in MLX runtime dependencies are absent | Install the documented optional extra before clinical execution; do not switch backend implicitly |
| `ClinicalSLMManifestError` | Base class for artifact-manifest failures | Refuse runtime construction and inspect fixed reason codes |
| `ClinicalSLMValidationError` | Invalid manifest metadata/schema, paths, licenses or revision | Correct trusted deployment metadata without weakening validation |
| `ClinicalSLMArtifactError` | Local filesystem verification cannot safely complete | Repair the protected immutable package/platform and reverify |
| `ClinicalSLMArtifactMissingError` | A required declared artifact is absent | Provision the exact declared package; do not fetch during clinical execution |
| `ClinicalSLMArtifactDigestMismatchError` | File size or digest differs from the manifest | Reject the package and restore its trusted immutable contents |
| `ClinicalSLMCapabilityError` | Malformed capability metadata or probe requirements | Correct the separate capability schema; an unsupported report instead carries reason codes |
| `ClinicalSLMMemoryError` | Base class for invalid memory artifact/profile metadata | Fix the conservative local profile; resource shortfall instead returns a rejected report |
| `ClinicalSLMMemoryValidationError` | Conflicting/invalid size, profile or report metadata | Correct the measurements and reject unknown assumptions |

Other public validation failures can be standard `TypeError` or `ValueError`
(for example invalid modes, claim/source alignment or malformed predictor
labels/scores). Do not relabel all such cases as `LocalNLIError`. The standalone
`verify()` result is a decision, not a `BriefRefusal` and not a clinical approval.

## Brief refusals

`build_clinical_brief()` returns a `ClinicalBrief` with a typed refusal and an
empty summary, rather than exposing backend exceptions. The table covers every
`BriefRefusal` member. Missing runtime packages, unknown local aliases and
uncached local artifacts become `MODEL_UNAVAILABLE` at de-identification or
generation. Remote providers remain prohibited; other unexpected exceptions,
including NLI callback failures, become `STAGE_FAILED`. A missing/incompatible
NLI callback or calibration ID reaches `NLI_UNAVAILABLE`. Record the enum and
controlled stage trace, not raw provider errors. The trace records entered
stages, not proof that each passed.

`LocalSummarizerError.reason` carries the closed reason vocabulary described in
[Summarization](summarization.md). Built-in errors preserve their fixed code;
foreign subclasses and callback errors become a fresh `execution_failed`
without reading foreign reason properties or retaining private exception chains.

| Member | Value | Current trigger | Remediation |
|---|---|---|---|
| `INVALID_INPUT` | `invalid_input` | Input type/size or guarded profile input is invalid | Supply bounded input and a supported profile |
| `REVIEW_REQUIRED` | `review_required` | No reviewed context was supplied | Resolve actually reviewed evidence from the application's authorized store |
| `EMPTY_EVIDENCE` | `empty_evidence` | The reviewed packet has no references | Stop generation and obtain appropriate reviewed evidence |
| `INVALID_EVIDENCE` | `invalid_evidence` | Content/policy binding, packet/fact limits or reference alignment is inconsistent | Rebuild and re-review the changed evidence; never copy old approval |
| `UNSUPPORTED_CLAIM` | `unsupported_claim` | A claim is not exactly and uniquely supported, or downstream support/coverage checks fail | Discard the summary and review the supported source spans |
| `NLI_UNAVAILABLE` | `nli_unavailable` | Missing/incompatible probability callback, thresholds or calibration binding | Supply a qualified local three-probability adapter and matching thresholds |
| `NLI_REJECTED` | `nli_rejected` | A claim does not pass selective entailment, including contradiction or abstention | Retain human review; do not force entailment or relax thresholds silently |
| `PRIVACY` | `privacy` | Leakage or final privacy validation rejects output | Discard output and investigate with protected handling and synthetic reproduction |
| `LENGTH_BUDGET_EXCEEDED` | `length_budget_exceeded` | Reviewed evidence exceeds a mapped class allocation before generation | Reduce admitted evidence with independent review; keep policy caps intact |
| `MODEL_UNAVAILABLE` | `model_unavailable` | A local alias, runtime package or cached artifact is unavailable at generation/de-identification | Provision the approved local runtime/artifact separately; do not substitute a remote provider |
| `STAGE_FAILED` | `stage_failed` | Other unexpected exception or incomplete stage contract | Inspect controlled configuration/stage information and restore the missing local contract |
| `INVALID_REVIEWED_EVIDENCE` | `invalid_reviewed_evidence` | The separately versioned local evidence contract is malformed | Obtain bounded exact reviewed evidence; do not reuse a malformed record |
| `REVIEW_RECEIPT_MISSING` | `review_receipt_missing` | No independently stored matching review receipt is available | Resolve the receipt through the trusted review store |
| `REVIEW_RECEIPT_EXPIRED` | `review_receipt_expired` | The receipt is outside its exclusive validity interval | Obtain fresh independent review |
| `REVIEW_RECEIPT_MISMATCHED` | `review_receipt_mismatched` | Receipt, current review state or evidence binding differs | Re-review the current evidence without copying old authority |
| `REVIEW_RECEIPT_REVOKED` | `review_receipt_revoked` | The trusted authority reports revocation | Stop generation and obtain current authorized review |
| `REVIEW_AUTHORITY_UNAVAILABLE` | `review_authority_unavailable` | The trusted current-review lookup fails or is unavailable | Restore the application-owned authority; never trust wire metadata alone |
| `REVIEW_SOURCE_UNAVAILABLE` | `review_source_unavailable` | Current source custody cannot be verified | Restore the independent source lookup |
| `REVIEW_SOURCE_CHANGED` | `review_source_changed` | Current source identity/content/offset binding has drifted | Rebuild evidence from the current source and obtain fresh review |
| `REVIEW_POLICY_CHANGED` | `review_policy_changed` | Current policy digest differs from the reviewed policy | Review the evidence against the current policy |
| `CANCELLED` | `cancelled` | Explicit cancellation or native task cancellation reaches a checkpoint | Discard output and any late callback result; cancellation does not forcibly kill a legacy callback |
| `DEADLINE_EXCEEDED` | `deadline_exceeded` | The started request budget expires at a checkpoint | Discard output; use an appropriate new request budget without treating the old result as complete |

These tables describe the current implementation. They do not alter exception
typing, refusal values, model qualification or the review/privacy requirements.
