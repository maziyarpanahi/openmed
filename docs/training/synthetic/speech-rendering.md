# Synthetic speech rendering and timing truth

`openmed.training.synthetic.speech_render.render_dialogue` renders caller-authored
synthetic scripts through an injected local `SpeechSynthesizer`. It writes mono
16-bit little-endian PCM WAV and a version-1 manifest with turn, word and gold
PHI timing. This is Python training/evaluation fixture tooling, scoped to #3831;
there is no runtime speech synthesis or OpenMedKit adapter in this slice.

```python
import io

from openmed.multimodal.asr_audio_profile import (
    MONO_16K_PCM_PROFILE,
    check_asr_compatibility,
)
from openmed.multimodal.wav_metadata import read_wav_metadata
from openmed.training.synthetic.speech_render import (
    GoldPhiSpan,
    ModelFreeSynthesizer,
    ScriptedTurn,
    render_dialogue,
)

wav = io.BytesIO()
manifest = render_dialogue(
    (
        ScriptedTurn("patient", "en", "Synthetic One speaks", (GoldPhiSpan(0, 13),)),
        ScriptedTurn("reviewer", "fr", "Synthetic Deux répond", (GoldPhiSpan(0, 14),)),
    ),
    ModelFreeSynthesizer(),
    wav,
)
report = check_asr_compatibility(
    read_wav_metadata(wav.getvalue()), MONO_16K_PCM_PROFILE
)
assert report.is_compatible
assert manifest.to_dict()["reviewer_confirmation_required"]
```

The bundled model-free engine produces deterministic token waveforms, **not
intelligible speech**. Its alignments are known by construction. It exercises
plumbing and timing-sensitive privacy fixtures without weights, downloads,
network calls or speech-quality claims. A real engine and its measured word
alignment must be supplied for speech or ASR evaluation; no provider is qualified
by these tests.

## Script and adapter contract

Each `ScriptedTurn` carries a speaker role, language, text and sorted, disjoint
`GoldPhiSpan` intervals. All character offsets use Python Unicode code points;
all sample and character intervals are half-open `[start, end)`. Adjacent PHI
spans are allowed. Empty, whitespace-only, overlapping or out-of-range spans
fail closed before synthesis. Scripts must be synthetic, never real patient
records. Authoring, translation, PHI detection and withholding remain separate.

The renderer splits each turn at gold PHI boundaries and synthesizes each PHI
span independently from its surrounding text. Nonempty chunks are concatenated
with exactly `chunk_gap_samples` silent frames (default 160). Whitespace-only
surrounding slices are omitted. Turns are separated by `turn_gap_samples`
(default 1,600). A PHI interval encloses precisely its complete synthesized
chunk, including any leading/trailing silence supplied by the engine, without
surrounding chunks or declared gaps in the unaugmented output.

The protocol receives the chunk text, role, language and requested sample rate;
it returns `SynthesizedChunk(pcm16, words)`. PCM must be mono signed little-endian
16-bit samples at that rate. Each `WordAlignment` supplies character and sample
bounds for one `\S+` whitespace token, in order. Missing, overlapping, empty,
out-of-range or mismatched word alignments are rejected. Timing is never inferred
from text length for real adapters. Languages without whitespace have one token
per whitespace-delimited run; no linguistic segmentation is claimed.

A PHI boundary can split a token, such as `Ada` in `Ada,`. The manifest preserves
both engine-aligned fragments with their exact offsets and gives the original
script token their enclosing interval. That envelope can include the declared
inter-chunk gap. No intra-token alignment is fabricated.

Before synthesis, **both** engine and voice licenses must normalize to one of
MIT, Apache-2.0, BSD-2-Clause, BSD-3-Clause, ISC or 0BSD. Missing, unknown,
custom `LicenseRef`, compound and non-permissive declarations are rejected.
`SynthesizerDeclaration.offline` and `.synthetic_voice` must both be true.
The caller must review actual engine/voice terms, keep assets local and use only
synthetic voices that do not clone or imitate a real person. Declarations are
checked; they are not independent verification or sandboxing of arbitrary
adapter code. Adapters must avoid network IO, payload logging and retention.

## Augmentation truth

`Augmentations` records its seed, numeric parameters and controlled labels.
Operations run in this order: overlap mix, gain, seeded uniform integer noise,
optional centered FIR bandpass, then rounded PCM16 clipping.

| Label | Signal effect | Timing effect recorded |
| --- | --- | --- |
| `uniform_noise` | Adds noise bounded by `noise_amplitude` using a private seeded PRNG | Source intervals unchanged; output digests change |
| `gain` | Applies positive linear gain, at most 16 | Intervals unchanged; clipped-sample count measured after filtering |
| `speaker_overlap` | Mixes consecutive distinct speakers; removes turn gaps and advances each later turn by `overlap_samples` | Absolute intervals shifted; accumulated `timeline_shift_samples`, overlapping turn indices, sequential frame count and frame-count delta recorded |
| `fir_bandpass_5tap` | Centered `[-1, 0, 2, 0, -1] / 4` FIR with zero padding | No nominal delay; influence expands by two samples on both sides, clipped to output bounds |

Overlap must not exceed the preceding turn's duration. Overlap changes the final
frame count; amplitude transforms preserve it. Source PHI timings and source
PCM digests remain traceable after augmentation, while rendered PCM digests
identify the actual mixed/transformed interval. `affected_start_sample` and
`affected_end_sample` describe filter support expansion. Expanded support can
cross PHI/chunk/turn boundaries. Augmented PHI intervals contain a mixture or
transformed signal, so they must not be treated as isolated source chunks.

## Privacy, limits and review

The manifest contains digests, Unicode offsets, sample bounds, seconds, counts,
normalized licenses, augmentation parameters and controlled safety notices.
It contains no transcripts, role/language strings, paths or audio payloads.
`to_dict()` returns an independent copy; `to_json()` is deterministic canonical
JSON. Source text and PCM are excluded from input/chunk representations.
Provider and output failures use controlled `SpeechRenderError` codes without
retaining the provider exception as context. No temporary files or network
transports are created by the renderer. SHA-256 digests are provenance references,
not anonymization guarantees.

The caller owns the binary WAV destination and optional text manifest destination.
Input and synthesis validation finish before writing. IO failures may leave
partial caller-owned output; discard it on error. Do not publish or log the audio.
There are at most 100 turns and 100,000 characters per turn; `max_frames` bounds
both total synthesized turn frames and the final timeline to at most 9.6 million.
Supported declared rates are 8–48 kHz, default 16 kHz. Use the existing
`read_wav_metadata` and declared `AsrAudioProfile` checks before downstream use.
Adapters remain responsible for bounded internal resource use and truthful
sample rates/alignments.

Every manifest binds a non-diagnostic synthetic-fixture notice and
`reviewer_confirmation_required: true`. Any consequential downstream output
requires explicit reviewer confirmation. These fixtures authorize no clinical
decision, consent, deployment or release; there is no cloud fallback, bundled
voice, restricted dataset, model prerequisite or autonomous action. Date shifts,
surrogates, quantized-model recall and ASR provider safety are separate contracts,
not measurements supplied by this renderer.
