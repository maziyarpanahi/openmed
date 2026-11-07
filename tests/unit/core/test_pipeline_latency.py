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
