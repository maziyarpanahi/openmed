# Sustained thermal and latency admission

`openmed.multimodal.runtime.thermal_gate` controls local VLM execution and speech PHI screening against a task's sustained edge budget. It measures elapsed time itself, rejects stale queued work, checks current memory and thermal observations before inference, and checks again before returning actual output. A provider's own duration estimate cannot authorize release.

The host supplies a monotonic clock, fresh local platform observations, already loaded adapters and an explicitly qualified mode ladder. There are no downloads, background probes, network calls or cloud fallbacks. Apple sensor acquisition and OpenMedKit admission are separate work tracked in #3836; this module implements the portable Python controller.

## Declare the task and its local modes

Each `LocalMode` binds a provider's model and preprocessing metadata to a host qualification digest, exact capability declarations, conservative additional allocation headroom, and maximum qualified thermal and memory-pressure states. Every mode must support the task's entire `ModalityProfile`. A capability marked `transform_required` is rejected: the gate does not resize images, drop source-coordinate requirements, shorten prompts or assume a smaller model preserves task quality.

The host must qualify each mode for the intended task and device. Digests record declarations; they do not authenticate evidence, certify model quality or isolate arbitrary adapter code. Provision weights beforehand, keep telemetry disabled and retain normal human review of non-diagnostic outputs.

`PlatformObservation` uses the same monotonic clock domain as enqueue times. `available_bytes` is conservative allocation headroom after accounting for already resident models and buffers. Unknown thermal state, unknown memory pressure, missing headroom, future measurements, stale measurements and regressing measurement timestamps all withhold work. Critical thermal or memory pressure cannot be enabled by a mode declaration. A stale sensor must never be replaced with an invented nominal reading.

## Deterministic local example

This example uses a synthetic provider and clock to demonstrate actual dispatch and withheld output. It does not load or qualify a medical model.

```python
from openmed.multimodal.capability_matrix import (
    ModalityCapability, ModalityProfile, ProviderCapabilityEnvelope,
)
from openmed.multimodal.runtime.thermal_gate import (
    EdgeBudget, GateReason, LocalMode, MemoryPressure,
    PlatformObservation, ThermalGate, ThermalState, vlm_adapter_digest,
)
from openmed.multimodal.vision.providers.base import (
    ImageInput, LocalProviderMetadata, Prompt, RuntimeBackend,
    RuntimeMetadata, VLMRequest, VLMResult,
)

now = [10.0]
thermal = [ThermalState.NOMINAL]


class SyntheticProvider:
    metadata = LocalProviderMetadata(
        backend=RuntimeBackend.SYNTHETIC,
        model_digest="a" * 64, preprocessing_digest="b" * 64,
    )

    def infer(self, request):
        now[0] += 0.02
        return VLMResult(
            request_digest=request.digest,
            runtime=RuntimeMetadata(
                provider=self.metadata, duration_ms=20, output_tokens=1,
            ),
            text="Synthetic non-diagnostic draft",
        )


provider = SyntheticProvider()
mode = LocalMode(
    adapter_digest=vlm_adapter_digest(provider.metadata),
    qualification_digest="c" * 64,
    capabilities=ProviderCapabilityEnvelope(
        provider_id="synthetic", capability_version="1.0.0",
        capabilities=(ModalityCapability("image"),),
    ),
    required_available_bytes=1024,
)
gate = ThermalGate(
    ModalityProfile("synthetic-task", "image"), (mode,),
    EdgeBudget(max_latency_ms=100, max_queue_ms=50, max_output_age_ms=140),
    lambda: PlatformObservation(
        observed_at=now[0], thermal=thermal[0],
        memory_pressure=MemoryPressure.NORMAL, available_bytes=4096,
    ),
    clock=lambda: now[0],
)
request = VLMRequest(
    image=ImageInput(data=b"synthetic", media_type="image/png", width=1, height=1),
    prompt=Prompt(text="Describe this synthetic image"),
)
first = gate.run_vlm((provider,), request, queued_at=now[0])
assert first.reason is GateReason.READY and first.value is not None
thermal[0] = ThermalState.CRITICAL
held = gate.run_vlm((provider,), request, queued_at=now[0])
assert held.reason is GateReason.RESOURCES and held.value is None
assert "Synthetic non-diagnostic draft" not in gate.to_json()
gate.close()
```

## Deadlines, degradation and recovery

Each gate is one task session. Serialize calls and preserve the same instance across sustained work. `max_queue_ms` rejects work before dispatch; `max_output_age_ms` includes queueing, execution and the post-execution sensor check. `max_latency_ms` caps individual calls and the nearest-rank rolling p95 in a bounded per-mode window. Measurements use the controller's clock, including failed calls. A sensor outage does not erase a measured slow call. Clock failure or regression closes admission permanently.

Modes are ordered from preferred to explicitly qualified fallback. Admission skips modes that cannot meet the observed resource constraints. Slow or failed execution withholds its result and advances the floor for later inputs; it never retries the same input automatically. Every switch preserves previous measurements. Once the last mode fails, the session abstains. Stale queue entries cannot be rescued by choosing another model.

There is deliberately no automatic upgrade. A host may resume a transient resource hold with fresh measurements, but the floor never moves backward. Recovery to a previously rejected mode requires host requalification and an explicit new session. Do not recreate gates for every input or erase their history to manufacture a pass. Hosts also manage model residency: merely selecting a smaller adapter does not unload the larger model or reclaim its memory.

The controller is cooperative. It can withhold a result after a native call returns; it cannot interrupt a hung call or guarantee hard real-time limits. Adapters must independently bound native execution, allocation and cancellation. The operation ceiling bounds session accounting, and each mode's history has a fixed maximum size.

## Preserve speech privacy across runtime holds

`release_speech(buffer, queued_at=oldest_pending_enqueue_time)` wraps the existing `WithholdingBuffer.release_ready()` call, including its local PHI screening. Configure one audio mode whose `adapter_digest` equals `buffer.screen_artifact_digest`. The existing screener still checks its live declaration against that pinned identity.

The wrapper keeps the original revisions, redaction mappings, trailing-context rules and insertion deadlines. It never finalizes provisional text or interprets runtime clearance as privacy clearance: a `READY` outcome may contain an empty tuple while PHI or context remains unresolved. A thermal, age, provider or budget hold closes the buffer and discards remaining text; an overheated or late screening result is not returned. Do not replay discarded text into a fresh buffer automatically.

This method gates screening and release; it does not control microphone capture or ASR inference. Call `buffer.tick()` while idle, close the buffer on cancellation, and let its original privacy deadline continue to run. Native streaming-ASR adapters must integrate admission at their own bounded execution boundary.

## Evidence and limits

`ThermalOutcome.value` contains protected output only on runtime clearance and is excluded from `repr`. `report_json` and `gate.to_json()` contain only controlled reason codes, counts, mode indices and aggregate p95 measurements. They contain no image bytes, speech text, prompts, patient identifiers, request digests, model names or filesystem paths. Do not serialize the protected value into logs or telemetry.

Tests use synthetic providers, injected clocks and local sensor observations, including mid-call overheating, delayed results, exhausted fallback ladders, sensor outages, immutable provider pins and PHI screening. They establish controller behavior, not hardware performance, sensor accuracy, clinical safety or OCR/ASR/model quality.

## Apple session admission

`MultimodalThermalSession` applies a workload's declared timing budget to a trusted `LocalMultimodalWork` operation. Its default `AppleSystemThermalSource` reads `ProcessInfo.thermalState` and observes `thermalStateDidChangeNotification`. The gate also reads the current state before dispatch and before releasing a result. The monotonic clock and sensor can be injected for deterministic tests. All budgets and timestamps use milliseconds in the same monotonic clock domain; do not pass a wall-clock timestamp.

| Apple state | Native admission policy |
| --- | --- |
| Nominal | Continue at the ordinary cadence and maximum window. |
| Fair | Admit only after the declared reduced-cadence interval has elapsed since the last start. |
| Serious | Enforce the reduced cadence and smaller declared window. |
| Critical or unavailable | Pause and abstain. No provider is called. |

This policy changes the admission of work on the same local provider. It never resizes an image, truncates a speech window, swaps models, or retries an input automatically. Pass `requestedWindowMs` for a time window so the controller can refuse an oversized submission. A provider declaring network use is refused. The caller must separately establish any model, capture-consent, privacy, or clinical-review requirements.

For each operating mode the controller keeps a bounded, expiring history of execution durations. A rolling nearest-rank p95 above `maxLatencyMs` requires a stricter profile; if there is no eligible profile it abstains. Queue admission uses `maxQueueMs` and total output age uses `maxOutputAgeMs`. A worsening state takes effect immediately. Recovery to a less restrictive profile requires the target profile to stay unchanged for `recoveryMs`; a new pressure notification restarts that interval. An intervening escalation invalidates ongoing work even if the device cools before it returns. This governs cadence and window admission; it does not undo the Python gate's model-qualification downgrade floor.

```swift
import OpenMedKit

func runLocal<Work: LocalMultimodalWork>(
    _ work: Work, session: MultimodalThermalSession, queuedAtMs: UInt64
) async -> Work.Output? {
    let outcome = await session.run(work, queuedAtMilliseconds: queuedAtMs,
                                    requestedWindowMs: 500)
    // Only a cleared result reaches the local consumer. Do not log its value.
    return outcome.value
}
```

Construct one session per workload using an explicit `MultimodalThermalBudget` and call `close()` when the workload ends. Only one operation runs at a time; concurrent or premature work is refused. Capture the queue timestamp when enqueueing, not when dispatching. Admission and release use the Python gate's controlled reason codes: `ready`, `resources`, `queue_expired`, `output_expired`, `latency`, `observation_unavailable`, `clock_invalid`, `provider_rejected`, `closed`, and `operation_limit`. `evidence()` exposes thermal state, mode, reason, counts, last queue age and rolling latency; it contains no inputs, results, identifiers, absolute timestamps or device names. Returning a local value is not clinical approval or PHI release authorization.

On supported MLX platforms, an already loaded `OpenMedVisionLanguageModel` exposes `generateIfAdmitted(_:localImageURL:maxTokens:temperature:releaseGuard:assetDigest:identifiers:queuedAtMilliseconds:in:)`. It feeds its existing local inference path through this gate and refuses remote image references before opening them. Use `load(modelDirectory:)` for an entirely offline model-loading path.

Pass the same qualified local `releaseGuard`, source `assetDigest` and source-bound identifier digests used by direct generation. The thermal adapter forwards them for both text and local-image requests; the [generated-text guard](generated-text-guard.md) still controls release. A missing guard or asset digest withholds text, and identifier evidence from another source is refused. A thermal `ready` outcome may therefore contain a generation with empty text and a privacy abstention envelope. Inspect `releaseEnvelope` before consuming output; neither thermal clearance nor privacy screening replaces explicit human review.
