# Local speaker diarization

`openmed.multimodal.streaming.speech.diarization` represents acoustic turns and feeds explicit cluster changes into the [speaker alias contract](speaker-aliases.md). It does not identify people, assign clinical roles, or approve clinical actions. Every turn remains unreviewed or unstable. Clinical role review is a separate [ambient workflow](ambient-speaker-roles.md).

## Provider-independent observations

The protocol accepts mono signed PCM16 little-endian audio at 16 kHz. Each request contains at most sixty seconds; offsets and half-open turn intervals use samples. There is no implicit resampling. Requests bind the actual audio bytes, revision, offset, and a caller-generated opaque request reference. Reports expose that binding, provenance digests, offsets, aliases and uncertainty; they contain no audio, embeddings, text or model paths. Treat even these reports as sensitive metadata.

This complete example uses synthetic silence and fixed observations to exercise the contract. It does not assert that a model detected speech in silence.

```python
from openmed.multimodal.streaming.speech.diarization import (
    AcousticTurn, ClusterEvidence, DiarizationBatch,
    DiarizationRequest, DiarizationSession,
)

request = DiarizationRequest(
    pcm=b"\x00\x00" * 16000,
    revision=0,
    offset_samples=0,
    request_ref="1" * 32,
)
batch = DiarizationBatch(
    request_binding=request.binding,
    provenance_digest="a" * 64,  # Synthetic fixture provenance only.
    turns=(
        AcousticTurn(start_sample=0, end_sample=10000, cluster_ref="2" * 32),
        AcousticTurn(
            start_sample=8000, end_sample=16000, cluster_ref="3" * 32,
            evidence=ClusterEvidence.UNSTABLE,
        ),
    ),
)
session = DiarizationSession("4" * 32)
report = session.apply(request, batch)
assert {turn["overlap"] for turn in report["turns"]} == {"observed"}
assert all(turn["clinical_role"] is None for turn in report["turns"])
session.close()
```

`DiarizationBatch.lineage` takes the existing explicit `LineageEvent` relabel, merge and split events. The session applies them atomically; stale or invalid revisions leave aliases unchanged. The session owns its alias registry and has no role-confirmation method. It cannot accidentally inherit approvals from an external registry. Unknown new clusters get new aliases. Providers must not infer continuity from a cluster's numeric position.

Intersecting intervals from distinct clusters produce `overlap="observed"`. Otherwise the value is `unknown`: this backend cannot prove that overlap is absent. Native segmentation frames can extend beyond the final input sample. The adapter intersects those final intervals with the input window, reports `boundary_clipped=true`, and marks the observation `unstable`. Reversed, nonfinite, negative-start or wholly out-of-window turns are rejected. Confidence remains `None` because the runtime does not supply a calibrated probability. Zero turns produce `insufficient_evidence`, not a successful identity decision. Fixed observations cover the contract independently of any particular model's accuracy.

## Optional native CPU adapter

Install `openmed[edge-diarization]` in a dedicated environment. It installs the Apache-2.0 Sherpa-ONNX runtime and its core package, NumPy, and psutil. Base OpenMed imports remain independent of these optional dependencies. The adapter currently supports Linux and macOS; unsupported platforms fail with `diarization_platform_unsupported`.

Supply two compatible, trusted local ONNX files using `LocalDiarizationArtifact`: a segmentation model and a speaker-embedding model. Each requires an absolute regular-file path, the expected file SHA-256, a digest of its immutable source/revision record, and a digest of its terms snapshot. Keep the actual source records and terms alongside your privately managed assets. These fields record provenance; they do not approve a license or restrict which weight license a user may choose. Model terms remain separate from OpenMed's Apache-2.0 adapter. Nothing downloads weights, accepts terms, polls remote licenses, or changes model visibility.

```python
from openmed.onnx.diarization import DiarizationLimits, SherpaDiarizationProvider

# segmentation and embedding are caller-created LocalDiarizationArtifact values.
provider = SherpaDiarizationProvider(
    segmentation=segmentation,
    embedding=embedding,
    limits=DiarizationLimits(timeout_seconds=30, max_rss_bytes=1024**3),
)
batch = provider.observe(request)
```

The pinned runtime is Sherpa-ONNX 2.0.1, CPU, one thread, automatic cluster count, clustering threshold 0.5, minimum speech duration 0.3 seconds and minimum silence duration 0.5 seconds. Both runtime package versions are checked before loading assets. [Sherpa's diarization documentation](https://k2-fsa.github.io/sherpa/onnx/speaker-diarization/index.html) describes compatible model families. Model files are checked against the supplied hashes before and after inference. They must stay unchanged during the call; this check is not protection against a malicious model or hostile native code.

Sherpa's numeric speaker labels are local to one batch. This adapter derives opaque cluster references from the full request binding and cluster number; it never guesses cross-window continuity. It does not return relabel/merge/split claims without evidence. Applications needing continuous acoustic tracking must supply an independently qualified provider that reports explicit lineage through the same protocol.

## Resource and privacy behavior

Each inference runs in a disposable process. A parent watchdog covers artifact verification, model loading and inference. It checks cancellation and the deadline during nonblocking pipe I/O and samples worker RSS every 20 ms. It terminates and reaps the process on failure. RSS sampling is a watchdog, not an OS memory allocation quota: a transient allocation can exceed the budget between samples. OS scheduling and process termination add cleanup latency. This is suitable for bounded local batches, not a real-time latency guarantee.

Audio buffers are bounded to 1,920,000 input bytes, responses to 256 KiB, results to 1,024 turns and 32 clusters. Audio flows through pipes and memory; no audio temp file, cache, log, native diagnostic, or biometric embedding is emitted. Parent and child may each hold bounded audio copies; Python and the native allocator do not promise cryptographic memory erasure. Cancellation accepts a `threading.Event`; failures return controlled codes and no partial observations. The caller still owns its input buffer and any retention policy. An arbitrary injected in-process provider must implement its own execution limits; the protocol alone cannot interrupt native code.

The built-in backend has no network operation or cloud fallback. Qualification should additionally run in an OS network-denied environment. This lifecycle wrapper is not a network sandbox for hostile dependencies; deploy with application-level network denial when enforcing that boundary.

## Qualification

Offline unit controls cover source binding, turn bounds, speaker changes, explicit lineage, overlapping intervals, insufficient evidence, role separation, backpressure, cancellation, timeouts, output limits and process cleanup. An opt-in integration suite runs actual local models, silence and invalid-artifact controls:

```bash
OPENMED_DIARIZATION_MANIFEST=/absolute/private/manifest.json \
  python -m pytest tests/integration/multimodal/test_local_diarization.py -q
```

The private manifest has `segmentation` and `embedding` objects containing the four `LocalDiarizationArtifact` fields. Missing assets skip only this opt-in suite; ordinary contract tests need no weights. No model, patient recording, private manifest or synthesized voice recording belongs in the repository. The native smoke establishes loading and execution behavior; it does not establish speaker-error rate, overlap recall, clinical validity, or cross-language accuracy. Those require caller-approved, representative audio and separately reviewed evaluation.
