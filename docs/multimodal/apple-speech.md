# Local Core ML speech in OpenMedKit

`OpenMedSpeech` is an optional Swift package product for a pinned English `whisper-tiny.en` Core ML model. It accepts caller-supplied mono Float32 audio at 16 kHz and emits protected partial hypotheses and finalized acoustic segments. It does not capture a microphone, infer consent, identify speakers, finalize clinical notes, or release text through a privacy gate.

## Install and supply the model

Add `.product(name: "OpenMedSpeech", package: "openmed")` to your application target and import `OpenMedKit` and `OpenMedSpeech`. Both the repository-root and nested Swift packages expose this product. The package requires Swift 5.10 or newer; the nested package's existing MLX dependencies retain their own newer compiler requirements.

The adapter uses MIT-licensed WhisperKit 1.1.1 at commit `1bbc57cb7fe6410248a685b1c5a3c6d56adc2ed4`. The existing tokenizer package moves from 0.1.24 to 1.0.0 because 0.1.24 constrains a shared command-line parser to an incompatible version. The runtime itself remains optional: selecting only the `OpenMedKit` product does not link WhisperKit. SwiftPM still resolves the package graph.

Download weights separately before use. Populate a local directory with these immutable sources, preserving each relative path:

| Local path | Source |
| --- | --- |
| `AudioEncoder.mlmodelc/`, `MelSpectrogram.mlmodelc/`, `TextDecoder.mlmodelc/` | [`argmaxinc/whisperkit-coreml`](https://huggingface.co/argmaxinc/whisperkit-coreml/tree/0f63a7800b00dd0226abd051b906c246e1907482/openai_whisper-tiny.en), revision `0f63a7800b00dd0226abd051b906c246e1907482`, under `openai_whisper-tiny.en/` |
| `tokenizer/config.json`, `tokenizer/tokenizer.json`, `tokenizer/tokenizer_config.json` | [`openai/whisper-tiny.en`](https://huggingface.co/openai/whisper-tiny.en/tree/87c7102498dcde7456f24cfd30239ca606ed9063), revision `87c7102498dcde7456f24cfd30239ca606ed9063` |

`VerifiedSpeechArtifact.files` publishes the expected byte count and SHA-256 for every consumed file. Verification uses these built-in pins, checks for unexpected runtime files and symbolic links, and repeats before and after loading. Keep the directory under exclusive application control and immutable for the engine's lifetime; checks cannot prevent a privileged concurrent filesystem writer. Core ML may create its own model cache. No audio is written there by this adapter.

The independently written adapter is Apache-2.0. Model terms remain separate and users must assess their intended use. The revision-bound model-card digest is provenance, not license acceptance or a permanent determination of rights. No model weights, license acceptance prompt, recurring license lookup, or inference download is bundled here.

## Stream caller-owned PCM

```swift
import Foundation
import OpenMedKit
import OpenMedSpeech

func transcribeSyntheticWindow(modelDirectory: URL) async throws {
    let artifact = try VerifiedSpeechArtifact.verifyTinyEnglish(at: modelDirectory)
    let engine = try await CoreMLSpeechEngine.load(artifact: artifact, compute: .cpuOnly)
    let session = try StreamingSpeechSession(engine: engine) { event in
        // Route protected text to your local review/privacy pipeline.
        // Never print or persist it as diagnostic evidence.
        switch event {
        case .partial: break
        case .finalized: break
        }
    }
    let syntheticSilence = [Float](repeating: 0, count: 16_000)
    let input = try syntheticSilence.withUnsafeBufferPointer {
        try SpeechPCMBuffer(samples: $0, startSample: 0)
    }
    do {
        try await session.append(input, endOfUtterance: true)
        try await session.finish()
    } catch {
        await session.cancel()
        throw error
    }
}
```

The buffer constructor copies the supplied samples into an owned allocation. The caller remains responsible for its original array. Passing the owned buffer to `append` transfers ownership: every admitted or refused submission clears it. Append calls must be awaited. A concurrent call receives `speech_decoder_busy`; its buffer is also cleared, so apply backpressure before submission. Samples must be finite, in `[-1, 1]`, and contiguous on the 16 kHz sample grid starting at zero.

The session accumulates one utterance, decoding cumulative windows at the configured stride. A window cannot exceed 30 seconds. The caller must mark utterance boundaries explicitly; overflow is refused without dropping old speech or inventing an endpoint. `finish` finalizes the remaining utterance and unloads the engine. Default limits also bound partial hypotheses and segments to 4,096 each and stream duration to 24 hours. This is bounded incremental window decoding, not a claim of realtime performance on every Apple device.

`requestCancellation()` immediately fences event delivery. `await cancel()` additionally waits for active decoding to drain and clears owned audio. Cancellation of the task awaiting append/finish propagates to the decoder. Core ML predictions are cooperatively cancellable between operations; an in-flight device operation cannot be forcibly interrupted. The adapter clears its own allocations but cannot promise secure erasure of internal Core ML or dependency copies.

## Timing, confidence and refusals

Partial hypotheses carry the input window interval and may omit word alignment. Final word timings come from the decoder's acoustic alignment. If estimated word boundaries exceed the supplied audio or tokenization cannot be verified, the adapter withholds all word timings and sets `alignmentStatus` to `outsideWindow` or `unverifiedTokenization`; it retains the source window interval and protected text for review, without clipping estimates into apparently valid timestamps. Times are stream-relative milliseconds, indices are zero-based, and probability estimates use integer parts per million. These fields match the Python speech-provider contract; shared synthetic fixtures check timing units, Unicode scalar lengths, provenance and invalid bounds. The configured language is English, not a guessed language confidence. Final decoder confidence is the exponent of the token-count-weighted mean segment log probability; partial confidence uses the decoder's current mean log probability. Decoder probability is uncalibrated and must not be presented as clinical certainty.

`.cpuOnly` is the default. `.cpuAndGPU` and `.cpuAndNeuralEngine` require the named accelerator to exist and are refused on simulators where the dependency otherwise substitutes CPU. Core ML decides individual operation placement within the permitted device set; selecting an accelerator does not certify that every operation ran there. There is no network or system speech fallback.

Errors expose controlled `StreamingSpeechError` codes. Descriptions redact transcript text and PCM. The dependency logger is globally disabled; the host must not re-enable it while this adapter is active. The adapter refuses a decode if logging has been enabled. Do not expose underlying dependency objects or run unrelated debug logging against the same runtime during protected processing.

## Validation

Ordinary `swift test` covers fake-engine sequencing, ownership, bounds, cancellation, stale callbacks, shared Python/Swift fields, digest verification and unavailable compute without weights or a microphone. An opt-in `SpeechArtifactTests/testPinnedNativeSyntheticSpeechWhenProvided` uses `OPENMED_TEST_SPEECH_MODEL` and `OPENMED_TEST_SYNTHETIC_PCM`; the latter must be a synthetic, headerless, little-endian Float32 16 kHz mono fixture. It never downloads assets. Run native inference inside an operating-system network-denial sandbox when qualifying a device/model combination. This implementation does not establish clinical ASR accuracy or device latency qualification.
