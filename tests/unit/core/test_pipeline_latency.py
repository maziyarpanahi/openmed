"""Latency-scaling regression tests for the 10-stage pipeline (OM-364).

These tests bound the wall-clock behaviour of the de-identification pipeline on
long clinical notes. They use synthetic notes only (no real PHI) and assert on
*ratios* rather than absolute milliseconds so they stay tolerant of CI noise and
machine-to-machine variation.

The guarantees under test:

* De-identifying a note that is ``k`` times longer must not take
  disproportionately longer -- the pipeline must scale roughly linearly, not
  quadratically, with note length.
* No single stage is allowed to blow up super-linearly relative to the whole
  run as the note grows.
* Correctness is length-invariant: the per-block span pattern must be identical
  regardless of how many blocks the synthetic note contains.
"""

from __future__ import annotations

import time
from datetime import datetime

import pytest

from openmed.core.pipeline import STAGE_NAMES, Pipeline
from openmed.processing.outputs import EntityPrediction, PredictionResult

# One synthetic clinical block. Contains only fabricated identifiers so the
# corpus never carries real PHI. The phone number is the deterministic anchor
# used to assert length-invariant correctness.
_SYNTHETIC_BLOCK = (
    "Assessment: the patient remains clinically stable on the current regimen. "
    "Vitals are within normal limits and no acute distress was noted today. "
    "Contact number 555-0142 is on file for follow-up scheduling. "
    "Plan: continue the current medication and monitor labs on a weekly basis. "
)


def _synthetic_note(blocks: int) -> str:
    """Return a synthetic long clinical note built from ``blocks`` copies."""
    return (_SYNTHETIC_BLOCK * blocks).strip()


def _phone_only_detector(text, **kwargs):
    """Deterministic, model-free detector that flags the fake phone anchors."""
    entities = []
    token = "555-0142"
    start = 0
    while True:
        index = text.find(token, start)
        if index < 0:
            break
        entities.append(
            EntityPrediction(
                text=token,
                label="PHONE",
                start=index,
                end=index + len(token),
                confidence=0.95,
            )
        )
        start = index + len(token)
    return PredictionResult(
        text=text,
        entities=entities,
        model_name=kwargs["model_name"],
        timestamp=datetime.now().isoformat(),
    )


def _build_pipeline() -> Pipeline:
    # Model-free pipeline so the test measures pipeline overhead, not model
    # inference, and runs offline without transformers installed.
    return Pipeline(model_detector=_phone_only_detector)


def _best_run_seconds(text: str, *, repeats: int = 5) -> float:
    """Return the minimum wall-clock seconds across ``repeats`` runs.

    Taking the minimum discards scheduler noise and GC pauses, which is the
    standard way to make micro-benchmarks robust on shared CI runners.
    """
    pipeline = _build_pipeline()
    pipeline.run(text, method="mask")  # warm caches / imports before timing
    best = float("inf")
    for _ in range(repeats):
        start = time.perf_counter()
        pipeline.run(text, method="mask")
        best = min(best, time.perf_counter() - start)
    return best


def _best_stage_durations_ms(text: str, *, repeats: int = 5) -> dict[str, float]:
    """Return each stage's minimum duration across repeated warm runs."""
    pipeline = _build_pipeline()
    pipeline.run(text, method="mask")  # warm caches / imports before timing
    best = {stage_name: float("inf") for stage_name in STAGE_NAMES}
    for _ in range(repeats):
        result = pipeline.run(text, method="mask")
        for stage_name in STAGE_NAMES:
            best[stage_name] = min(
                best[stage_name], result.stage_duration_ms(stage_name)
            )
    return best


def test_pipeline_reports_per_stage_latency_for_all_stages():
    """Every stage exposes a non-negative, latency-only duration."""
    result = _build_pipeline().run(_synthetic_note(20), method="mask")

    for stage_name in STAGE_NAMES:
        duration = result.stage_duration_ms(stage_name)
        assert duration >= 0.0

    # Durations are plain floats keyed by stage name -- no document text, so no
    # raw PHI leaks into profiling output.
    durations = result.stage_durations_ms
    assert set(durations) == set(STAGE_NAMES)
    assert all(isinstance(value, float) for value in durations.values())

    # Wall-clock timings are non-deterministic, so they stay off the
    # reproducible audit record.
    assert "stage_durations_ms" not in result.audit_record


def test_long_note_spans_are_length_invariant():
    """Correctness must not depend on note length (acceptance criterion)."""
    small = _build_pipeline().run(_synthetic_note(4), method="mask")
    large = _build_pipeline().run(_synthetic_note(40), method="mask")

    # The synthetic note repeats one block, so both notes must contain the same
    # per-block span pattern scaled by the block count.
    assert len(small.spans) == 4
    assert len(large.spans) == 40
    assert len(large.spans) == 10 * len(small.spans)

    small_labels = {span.canonical_label for span in small.spans}
    large_labels = {span.canonical_label for span in large.spans}
    assert small_labels == large_labels

    # Redaction stays consistent: the fake phone anchor is redacted everywhere.
    assert "555-0142" not in large.redacted_text
    assert large.redacted_text.count("[") >= 40


@pytest.mark.slow
def test_pipeline_latency_scales_roughly_linearly_with_note_length():
    """A 4x longer note must not cost quadratically more (near-linear bound)."""
    base_blocks = 40
    factor = 4
    base_note = _synthetic_note(base_blocks)
    long_note = _synthetic_note(base_blocks * factor)

    # Sanity: the long note really is ~factor x the base length.
    length_ratio = len(long_note) / len(base_note)
    assert factor - 0.5 <= length_ratio <= factor + 0.5

    base_seconds = _best_run_seconds(base_note)
    long_seconds = _best_run_seconds(long_note)

    # Guard against a zero/near-zero denominator on very fast machines.
    base_seconds = max(base_seconds, 1e-4)
    time_ratio = long_seconds / base_seconds

    # Linear scaling would give time_ratio ~= factor (4x). We allow a generous
    # 3x headroom over linear (i.e. up to ~12x) to absorb CI noise and constant
    # overheads, while still failing loudly on quadratic (which would be ~16x+)
    # or worse regressions.
    assert time_ratio <= factor * 3, (
        f"pipeline latency scaled {time_ratio:.1f}x for a {length_ratio:.1f}x "
        f"longer note; expected roughly linear (<= {factor * 3}x)"
    )


@pytest.mark.slow
def test_no_single_stage_dominates_superlinearly_on_long_notes():
    """No stage's absolute latency should grow super-linearly.

    Runtime fractions are not a valid scaling signal: a linear stage can occupy
    a larger share when other stages have fixed costs. Compare repeated minimum
    absolute durations instead, with the same generous 3x-over-linear headroom
    as the whole-pipeline regression test.
    """
    short_blocks = 20
    factor = 8
    short_note = _synthetic_note(short_blocks)
    long_note = _synthetic_note(short_blocks * factor)

    length_ratio = len(long_note) / len(short_note)
    assert factor - 0.5 <= length_ratio <= factor + 0.5

    short_durations = _best_stage_durations_ms(short_note)
    long_durations = _best_stage_durations_ms(long_note)

    for stage_name in STAGE_NAMES:
        short_ms = max(short_durations[stage_name], 0.05)
        stage_ratio = long_durations[stage_name] / short_ms
        assert stage_ratio <= factor * 3, (
            f"stage {stage_name!r} latency scaled {stage_ratio:.1f}x for a "
            f"{length_ratio:.1f}x longer note; expected roughly linear "
            f"(<= {factor * 3}x)"
        )
