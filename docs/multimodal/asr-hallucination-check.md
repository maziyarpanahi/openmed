# ASR Hallucination Check

Local encoder-decoder speech models can produce fluent text over silence,
noise or music, and can fall into repeated-phrase loops. In an ambient visit
that text can read like a clinical statement or a name.
`check_asr_hallucinations` checks fixed ASR hypothesis segments against an
injected voice-activity timeline and provider-reported statistics, under a
versioned threshold profile, and marks unsupported or looping segments as
non-final review candidates.

The check is provider-neutral and stdlib-only. It runs no ASR, no
voice-activity detection and no window stitching; those stages supply its
inputs.

```python
from openmed.multimodal.asr_hallucination import (
    AsrHypothesisSegment,
    AsrSegmentStatistics,
    VoiceActivityInterval,
    VoiceActivityTimeline,
    check_asr_hallucinations,
    require_draft_eligible,
)

timeline = VoiceActivityTimeline(
    duration_ms=8_000,
    intervals=(
        VoiceActivityInterval(0, 4_000, "speech"),
        VoiceActivityInterval(4_000, 8_000, "silence"),
    ),
)
stats = AsrSegmentStatistics(
    no_speech_probability=0.05, avg_logprob=-0.3, compression_ratio=1.4
)
segments = (
    AsrHypothesisSegment(
        "seg-1", 0, 4_000, statistics=stats, text="synthetic words here"
    ),
    AsrHypothesisSegment(
        "seg-2", 4_000, 8_000, statistics=stats, text="synthetic words here"
    ),
)
report = check_asr_hallucinations(segments, timeline)
print(report.eligible_segment_ids)  # ('seg-1',)
print(report.decisions[1].reason_codes)
# ('speech_coverage_low', 'silence_dominant')
require_draft_eligible(report, ["seg-1"])  # passes
```

## Inputs

No ASR provider or voice-activity adapter ships yet, so the inputs are small
frozen declarations that any provider or detector can fill:

- `VoiceActivityTimeline(duration_ms, intervals)` holds sorted,
  non-overlapping, half-open `VoiceActivityInterval(start_ms, end_ms, kind)`
  values. `kind` is one of `speech`, `silence`, `noise` or `music`. Gaps are
  allowed and count as unlabeled time, never as speech.
- `AsrSegmentStatistics` carries the optional provider values
  `no_speech_probability` (0 through 1), `avg_logprob` (-100 through 0) and
  `compression_ratio` (greater than 0, at most 1000).
- `AsrHypothesisSegment(segment_id, start_ms, end_ms, statistics, text,
  token_ids)` is one fixed hypothesis. `text` and `token_ids` are read only to
  count repetition units. When both are given, `token_ids` wins; text is split
  into case-folded `\w+` words. Neither is ever stored, serialized or shown in
  `repr`.
- An optional successful `ProviderResultEnvelope` binds the report to the
  provider's input and output digests. A non-success envelope is rejected.

## Outcomes

| Outcome | Meaning | `final_eligible` | `review_required` |
| --- | --- | --- | --- |
| `supported` | All evidence present, no check fired. | `true` | `false` |
| `flagged` | At least one check fired. | `false` | `true` |
| `insufficient_evidence` | No check fired, but evidence was missing. | `false` | `true` |

Missing statistics, a missing timeline, partial timeline coverage or missing
repetition units give `insufficient_evidence`; they are never a pass. When a
check fires and evidence is also missing, the outcome is `flagged` and both
kinds of reason code are listed.

## Reason codes

Reason codes are returned in the fixed order of `HALLUCINATION_REASON_CODES`.

Flag codes (`FLAG_REASON_CODES`):

| Code | Fires when |
| --- | --- |
| `speech_coverage_low` | Speech fraction of the segment is below `min_speech_ratio`. |
| `silence_dominant`, `noise_dominant`, `music_dominant` | Added with `speech_coverage_low` for the largest labeled non-speech kind (ties: silence, noise, music). |
| `no_speech_probability_high` | `no_speech_probability` is above `max_no_speech_probability`. |
| `avg_logprob_low` | `avg_logprob` is below `min_avg_logprob`. |
| `compression_ratio_high` | `compression_ratio` is above `max_compression_ratio`. |
| `ngram_loop` | Some n-gram (n from 1 to `loop_max_ngram`) repeats back to back at least `min_loop_repeats` times, or `min_unigram_loop_repeats` times for a single unit. |
| `repeated_ngram_fraction_high` | The share of repeated `repetition_ngram`-grams is above `max_repeated_ngram_fraction`; scored only from `min_repetition_units` units. |
| `runaway_length` | Units per second of detected speech exceed `max_units_per_speech_second`, or the unit count exceeds `max_segment_units`. |

Insufficient-evidence codes (`INSUFFICIENT_EVIDENCE_REASON_CODES`):
`vad_coverage_missing` (no timeline, or the labeled fraction is below
`min_vad_coverage`; speech and rate checks are then skipped),
`no_speech_probability_missing`, `avg_logprob_missing`,
`compression_ratio_missing` and `token_units_missing`.

## Repetition detection

Loop detection is deterministic and linear in the unit count per n-gram size:
for each size `n`, the longest run of positions where `unit[i] == unit[i + n]`
gives the most consecutive copies, `run // n + 1`. The decision reports the
strongest loop as `loop_ngram_size` and `loop_repeats`: the qualifying loop
that covers the most units (ties go to the smaller `n`), or the strongest
non-qualifying repeat when none qualifies. `repeated_ngram_fraction` is
`1 - unique / total` over all `repetition_ngram`-grams. The unit rate is
`unit_count * 1000 / speech_ms` over the speech that overlaps the segment.

## Threshold profile

`HallucinationThresholds` is a frozen, validated profile with a
`MAJOR.MINOR.PATCH` `version` and a `digest`, the SHA-256 of its canonical
sorted-key JSON. Every decision and report records the version and digest, so
a changed value is always visible. A deployment that changes a value must
also change the version.

`DEFAULT_HALLUCINATION_THRESHOLDS` (`openmed-asr-hallucination-default`,
version `1.0.0`, digest
`5b1319020ca30d81ae9760de42630b6b51553f1930aa13ce12d092e44d036412`):

| Field | Default |
| --- | --- |
| `min_vad_coverage` | 0.95 |
| `min_speech_ratio` | 0.3 |
| `max_no_speech_probability` | 0.6 |
| `min_avg_logprob` | -1.0 |
| `max_compression_ratio` | 2.4 |
| `loop_max_ngram` | 6 |
| `min_loop_repeats` | 4 |
| `min_unigram_loop_repeats` | 6 |
| `repetition_ngram` | 3 |
| `min_repetition_units` | 12 |
| `max_repeated_ngram_fraction` | 0.5 |
| `max_units_per_speech_second` | 8.0 |
| `max_segment_units` | 600 |

A ratio, probability or rate fires only when it is strictly past its bound,
so a value exactly at the bound passes. These defaults are conservative
starting points, not clinically validated cut-offs. Rates count tokens when
`token_ids` are supplied and words otherwise, so tune
`max_units_per_speech_second` for the unit a provider reports.

## Keeping flagged segments out of drafts

A decision is a value-free record with `final_eligible` and
`review_required`. Construction rejects any decision whose outcome, reason
codes and eligibility disagree, so a record cannot claim eligibility its
reasons do not support.

Any later stage that turns transcript segments into cited draft evidence, or
that decides what PHI withholding may release, must call
`require_draft_eligible(report, segment_ids, thresholds=...)` first. It
raises `AsrHallucinationError` with:

- `asr_threshold_profile_mismatch` when the report was produced under a
  different threshold digest than the caller expects;
- `asr_segment_unknown` when a requested segment is not in the report;
- `asr_segment_review_required` when a requested segment is `flagged` or
  `insufficient_evidence`.

A blocked segment stays blocked until a reviewer or a later revision produces
a new report in which it is `supported`. `HallucinationDecision.abstention()`
returns a `post_process` / `low_quality` [abstention record](abstention-reasons.md)
for any non-final segment.

## Privacy and errors

Decisions and reports hold segment identifiers, half-open millisecond
offsets, closed reason codes, numeric scores rounded to six decimals, and
SHA-256 digests of the threshold profile, the timeline and the provider
envelope. There is no field for transcript text or token identifiers.
`to_dict()` preserves declared field order and `to_json()` sorts keys for
byte-identical payloads; `report.digest` hashes that JSON.

`AsrHallucinationError` is a `ValueError` whose message equals its stable
`.category`. Submitted values, text and token identifiers are never echoed.
Input categories include `asr_vad_interval_invalid`,
`asr_vad_kind_unsupported`, `asr_vad_intervals_unordered`,
`asr_vad_interval_out_of_range`, `asr_stat_invalid`,
`asr_segment_id_invalid`, `asr_segment_bounds_invalid`,
`asr_segment_text_invalid`, `asr_segment_tokens_invalid`,
`asr_segment_id_duplicate` and the `asr_thresholds_*_invalid` family.

The check is a quality gate over model output, not a clinical decision. It
triggers no clinical action and makes no network call.

## Verification

```bash
pytest tests/unit/multimodal/test_asr_hallucination.py -q
```

Fixtures are synthetic. Tests cover silent, noise-only and music-only
timelines, normal speech, n-gram loops from text and token identifiers,
runaway length, every missing-evidence path, sentinel text that must never
reach records, reprs or errors, and negative controls in which mutated
thresholds change the digest and are refused by the draft guard.
