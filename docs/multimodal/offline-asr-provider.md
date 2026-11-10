# Offline streaming ASR with MLX Whisper

`OfflineWhisperSession` runs one local MLX Whisper model on Apple Silicon. It accepts bounded mono PCM windows, produces private partial hypotheses, and releases finalized text only through the configured local PHI screen. The adapter is Apache-2.0; model weights remain a separate, caller-managed download with their own terms.

This implementation targets Python on macOS arm64. The existing OpenMedKit contracts for speech timing, endpoints, language routing and revisions remain available, but this PR does not add or qualify a Swift native Whisper engine. Speech output is non-diagnostic and requires human review before consequential use.

## Installation and assets

```bash
pip install 'openmed[edge-asr]'
```

The optional ASR profile uses the qualified NumPy 2 numerical stack. Keep `scispacy`, which requires NumPy 1.x, in a separate environment. Start sessions under an `if __name__ == "__main__":` guard in executable scripts because the runtime uses multiprocessing spawn.

Prepare an immutable local directory containing `config.json` and `weights.safetensors`. The adapter accepts an existing directory, never a Hub name or URL. It rejects executable configuration, quantized layouts and symlink assets. Keep the directory owner-controlled: manifest verification detects changed files before loading, but is not a sandbox against a concurrent writer with the same OS identity.

The reference model is [`mlx-community/whisper-large-v3-turbo`](https://huggingface.co/mlx-community/whisper-large-v3-turbo/tree/a4aaeec0636e6fef84abdcbe3544cb2bf7e9f6fb) at revision `a4aaeec0636e6fef84abdcbe3544cb2bf7e9f6fb`. Its weight SHA-256 is `951ed3fc1203e6a62467abb2144a96ce7eafca8fa77e3704fdb8635ff3e7f8a6`. The [upstream model card](https://huggingface.co/openai/whisper-large-v3-turbo/blob/41f01f3fe87f28c78e2fbf8b568835947dd65ed9/README.md) at the recorded revision declares MIT. Review the model's terms for your use; neither a runtime license allowlist nor an online license check is imposed.

```python
from pathlib import Path
from openmed.mlx.models.whisper import WhisperAssets

directory = Path("/owner-managed/models/whisper-turbo")
assets = WhisperAssets(
    directory,
    revision="a4aaeec0636e6fef84abdcbe3544cb2bf7e9f6fb",
    license_snapshot=(directory / "UPSTREAM_MODEL_CARD.md").read_bytes(),
)
```

The fingerprint binds the complete weights, configuration, source revision and supplied license snapshot. Changed files or terms produce a different identity. The metadata does not interpret legal rights, automatically accept terms, or reclassify previously downloaded revisions.

## Stream lifecycle

Construct `OfflineWhisperSession` with the assets, an explicitly local `LocalPhiScreener`, its qualified language coverage, and a callback returning fresh `PlatformObservation` measurements. An unavailable thermal or memory sensor is a refusal, not an assumed nominal reading. The default resource gate accepts nominal or fair thermal state, normal memory pressure and at least 3 GiB of measured available memory.

1. Call `push_chunk(AudioChunk(...))` with sequential, mono, signed-16-bit little-endian PCM at 16 kHz. Each chunk is at most one second, and its actual byte length must match its declared duration. File paths, compressed audio and microphone capture are outside this adapter.
2. Call `partial()` on a growing window. The returned `AsrUpdate` contains counts and language confidence, never raw provisional text or words. The session retains only a fingerprint of the previous hypothesis for stability comparison.
3. After an upstream local endpoint decision, append further audio and call `end_utterance(activity)`. The `VoiceActivityTimeline` must describe the exact current window. Missing evidence, non-speech, repetition, low confidence, truncated output, or a changed hypothesis cannot produce a final release. At least 200 ms of additional audio and an identical previous hypothesis are required by the default stability policy.
4. Finalized text enters the existing withholding buffer. Its PHI screen sees the complete pending context and must resolve every identifier before release. The default trailing-context hold spans utterances. Supply a `WithholdingPolicy` appropriate to the qualified local detector; a single-utterance policy must be justified by that detector's context needs.
5. Call `finish()` after the last completed utterance. Unfinished audio is discarded, not silently finalized. Read a released segment through `reviewed_text(reviewer_confirmed=True)` only after review. Use the context manager or `cancel()` on every abandonment path.

The returned word-level timings use the last accepted alignment and source stream offsets. They contain no original word strings, including when those words were redacted. The screened output includes source-to-output mappings and revision lineage. Speaker identity is always `unknown`: ASR confidence does not establish a person, clinician role or diarization result.

## Bounds, cancellation and privacy

The default window is at most 15 seconds, with at most five minutes of input per session. A non-renewable wall-clock audio deadline starts at the first chunk of each window; revisions do not extend it. Pending finalized text also has an idle timer tied to its original withholding deadline. Decode time, operation count, language confidence and available memory have explicit bounds. `OfflineAsrPolicy` exposes those limits without accepting arbitrary backend generation options or prompts. Host callbacks must themselves be bounded and local; terminating the ASR worker does not preempt an arbitrary caller-supplied detector or sensor callback.

Inference runs in a dedicated spawned process. Cancellation, timeout, resource refusal and finalization overwrite the parent's fixed audio buffer, discard pending text, terminate and join the native worker, and close the private pipe. In-flight results are discarded after cancellation. A cleanup failure remains visible in `report()` and prevents final release. The adapter neither retries protected audio automatically nor falls back to a cloud service.

The runtime uses bundled tokenizer assets and an existing model directory, blocks Python network and DNS calls in its worker, disables Hub telemetry, and suppresses backend output streams. Reports contain counters, controlled reasons, provenance and measured memory peaks. They contain no audio, recognized text or source filenames. The caller must independently qualify and control its PHI screener, VAD, resource observer and any downstream consumer.

AudioChunk input and previously returned reviewed text remain caller-owned. Native allocation release, overwriting owned mutable buffers and process termination are not a secure-erasure guarantee for Python strings, OS swap, crash dumps, GPU memory or copies made by callers. Use the existing [audio retention boundary](streaming/audio-retention.md) for the application's wider capture and artifact inventory.

## Reproduce the optional device profile

Ordinary CI uses synthetic fake runtimes and never downloads or loads an ASR model. The opt-in profile uses actual inference, owner-installed macOS voices and synthetic English, French and Spanish utterances. Its thresholds and reference inputs are committed before the first inference run in `tests/fixtures/multimodal/offline_asr_profile.json` and `synthetic_speech.py`.

```bash
OPENMED_ASR_PROFILE=1 \
OPENMED_ASR_MODEL_DIR=/owner-managed/models/whisper-turbo \
OPENMED_ASR_PROFILE_REPORT=/private/output/asr-profile.json \
HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 \
sandbox-exec -p '(version 1)(allow default)(deny network*)' \
.venv/bin/python -m pytest \
  tests/integration/multimodal/test_streaming_asr_provider.py -q
```

Enable the profile only after installing the optional runtime and preparing the pinned local assets, including the saved upstream card. The profile generates temporary synthetic WAV fixtures, removes them after reading, blocks network access during inference, and writes a metadata-only receipt when a report path is supplied. Missing assets or platform observations fail an enabled profile rather than skipping it.

The profile measures clean-transcript word error rate, language behavior, warm latency, real-time factor, repeat agreement, native memory peaks, actual thermal observations, unknown-speaker behavior, cancellation and injected resource-degradation cleanup. Its fixture PHI screen deliberately suppresses entire synthetic identifier utterances to test the release boundary. That measures enforcement of a screening decision, not clinical PHI-detector recall. Physical thermal states and injected serious-thermal controls are reported separately; nominal execution does not qualify behavior on a physically throttled device.

## Reference-device result

The frozen `dev + edge-asr` profile passed on 2026-11-04 on an Apple M2 Max with 96 GiB RAM, macOS 26.5.2 and Python 3.12.12. It processed 58 synthetic utterances covering 226.906 seconds of generated speech input over a sustained run exceeding two minutes. These measurements describe this device and fixture set only.

| Measurement | Observed result |
| --- | --- |
| Clean-transcript word error rate | 1.81% overall; English 5.26%, French 0%, Spanish 0% |
| Finalized utterances released through the configured screen | 58/58 |
| Whole-utterance identifier-screen enforcement | 0 leaks across 18 synthetic identifier utterances |
| Speaker status | Unknown for every utterance |
| Repeat agreement on the clean fixtures | 100% exact output agreement |
| First decode / warm p95 decode | 1.85 s / 1.02 s |
| Warm p95 real-time factor | 0.373 |
| Peak native-worker RSS / MLX allocation | 1.79 GiB / 2.29 GiB |
| In-flight cancellation and worker cleanup | 0.024 s; no owned audio retained |
| Physical thermal state | Nominal throughout |
| Injected serious-thermal control | Refused further input and cleared owned audio |
| Network attempts | 0; inference ran under OS network denial |
| Hub imported before worker startup | Cached offline setting enforced before model load |

The exact runtime used MLX 0.32.4, mlx-whisper 0.4.3, NumPy 2.4.6, Numba 0.68.1, SciPy 1.15.3 and tiktoken 0.12.0. The profile SHA-256 is `3b655c2b04fa0a4e55d9fa854ca92bd2fda5011908ccec1b1b2917c9bd143d3c`; the combined asset fingerprint is `2de238621af3d271e06bc84c6e1eade5c38cbd7ac44d1919b93dac32a06dce89`. The local metadata-only receipt SHA-256 is `92605ec0aed23541aa1246791549b0af94d232c830b5a065e301fd0698d721fc`. No clinical recordings, voice assets or weights are committed.
