# Streaming Audio Windows

`plan_audio_windows` turns a declared duration into reproducible half-open
windows before any audio is read. It is an explicit planning helper, not a
decoder, a diarizer, or an ASR context policy.

```python
from openmed.multimodal.audio_windows import TailPolicy, plan_audio_windows

plan = plan_audio_windows(
    1001, window_ms=500, min_tail_ms=100, tail_policy=TailPolicy.MERGE
)
for window in plan.windows:
    print(window.window_id, window.start_ms, window.end_ms, window.overlap_ms)
```

## Window arithmetic

Every offset is a whole millisecond and every window is half-open,
`[start_ms, end_ms)`. Windows start at zero and advance by the stride,
`window_ms - overlap_ms`. The final window is clipped to the declared
duration, and planning stops as soon as a window reaches the end, so a window
is never contained in its predecessor. A zero duration plans no windows.

`overlap_ms` on a window is its real overlap with the previous window, which
is `0` for the first window and can be smaller than the requested overlap on a
clipped tail. Because windows are contiguous, `covered_ms` is the end of the
last window.

## Tail policies

A final window shorter than `min_tail_ms` is resolved by `tail_policy`:

| Policy | Effect |
| --- | --- |
| `keep` | Emit the short tail as its own window. Coverage is complete. |
| `merge` | Extend the previous window to the end of the input. Coverage is complete. |
| `drop` | Discard the tail. `covered_ms` is then smaller than `duration_ms`. |

`min_tail_ms=0` disables the rule. The rule is never applied to a plan holding
a single window, because that window is the whole input rather than a tail, so
no policy can plan zero windows for a nonempty input. `drop` is the only
policy that leaves part of the declared duration uncovered, and the plan says
so through `covered_ms`.

## Limits and failures

`AudioWindowError` is a `ValueError` with a stable `.category`; its string is
the same category. Parameters must be integers, which excludes booleans
because `type(value) is int` is checked rather than `isinstance`. Categories
are `audio_duration_not_an_integer`, `audio_window_not_an_integer`,
`audio_overlap_not_an_integer`, `audio_min_tail_not_an_integer`, the matching
`_out_of_range` categories, `audio_window_size_invalid`,
`audio_overlap_not_less_than_window`, `audio_min_tail_exceeds_window`,
`audio_tail_policy_unsupported` and `audio_window_count_limit_exceeded`.

Defaults are `MAX_AUDIO_DURATION_MS=86_400_000`, `MAX_AUDIO_WINDOW_MS=3_600_000`
and `MAX_AUDIO_WINDOW_COUNT=100_000`. The window count is bounded by
arithmetic before any window is built, so an oversized request never allocates.

Reading audio, choosing an ASR model context, diarization, transcript merging,
and wall-clock or sample-clock synchronization are out of scope. A plan is
offsets and counts only; it carries no samples, transcripts, speaker labels,
paths, or clinical text.

## Verification

```bash
uv run --frozen --extra dev pytest tests/unit/multimodal/test_audio_windows.py -q
```

Fixtures are synthetic. Tests pin exact division, short input, a
one-millisecond tail, merged and dropped tails, overlap, zero duration,
gap-free coverage across many parameter pairs, and every failure category.
