from __future__ import annotations

from datetime import datetime
from unittest.mock import patch

import pytest

from openmed import deidentify_stream
from openmed.core.pii import deidentify
from openmed.core.pipeline import Pipeline
from openmed.core.streaming import StreamingBufferError, StreamingDeidentifier
from openmed.processing.outputs import PredictionResult


def _empty_prediction(text: str, model_name: str = "stub") -> PredictionResult:
    return PredictionResult(
        text=text,
        entities=[],
        model_name=model_name,
        timestamp=datetime.now().isoformat(),
    )


def _empty_model_detector(text: str, **kwargs) -> PredictionResult:
    return _empty_prediction(text, model_name=kwargs.get("model_name", "stub"))


def _chunks(text: str, sizes: list[int]) -> list[str]:
    chunks = []
    index = 0
    for size in sizes:
        if index >= len(text):
            break
        chunks.append(text[index : index + size])
        index += size
    if index < len(text):
        chunks.append(text[index:])
    return chunks


@patch("openmed.core.pii.extract_pii")
def test_deidentify_stream_matches_single_pass_for_arbitrary_chunks(mock_extract):
    text = (
        "Patient emailed jane.patient@example.com, then called 555-123-4567 "
        "before discharge."
    )
    mock_extract.side_effect = lambda text, *args, **kwargs: _empty_prediction(text)

    single_pass = deidentify(text, method="mask")
    events = list(
        deidentify_stream(
            _chunks(text, [1, 7, 3, 11, 2, 5, 13, 8]),
            method="mask",
            max_buffer=48,
        )
    )

    assert "".join(event.redacted_text for event in events) == (
        single_pass.deidentified_text
    )
    assert events[-1].final is True
    assert events[-1].audit_record is not None


def test_boundary_split_identifier_is_not_partially_emitted():
    streamer = StreamingDeidentifier(
        pipeline=Pipeline(
            model_detector=_empty_model_detector,
            use_safety_sweep=True,
        ),
        max_buffer=32,
    )
    first_events = streamer.feed("Contact jane.patient@")
    second_events = streamer.feed("example.com before the appointment tomorrow.")
    assert streamer.carry_buffer_length <= streamer.max_buffer
    final_events = streamer.flush()

    emitted = [*first_events, *second_events, *final_events]
    redacted = "".join(event.redacted_text for event in emitted)

    assert redacted == "Contact [email] before the appointment tomorrow."
    for event in emitted:
        assert "jane.patient" not in event.redacted_text
        assert "example.com" not in event.redacted_text
        assert "patient@example" not in event.redacted_text


def test_too_small_buffer_raises_before_emitting_incomplete_identifier():
    streamer = StreamingDeidentifier(
        pipeline=Pipeline(
            model_detector=_empty_model_detector,
            use_safety_sweep=True,
        ),
        max_buffer=5,
    )

    with pytest.raises(StreamingBufferError):
        streamer.feed("Contact jane.patient@")


def test_carry_buffer_stays_within_configured_maximum_on_long_stream():
    streamer = StreamingDeidentifier(
        pipeline=Pipeline(
            model_detector=_empty_model_detector,
            use_safety_sweep=True,
        ),
        max_buffer=24,
    )

    for chunk in ["No PHI here. "] * 40:
        streamer.feed(chunk)
        assert streamer.carry_buffer_length <= streamer.max_buffer

    streamer.flush()
    assert streamer.max_observed_buffer <= streamer.max_buffer


def test_stream_audit_matches_single_pass_span_count_and_hashes():
    text = (
        "Email jane.patient@example.com about MRN 4111111111111111 and "
        "call 555-123-4567."
    )
    single_pipeline = Pipeline(
        model_detector=_empty_model_detector,
        use_safety_sweep=True,
    )
    stream_pipeline = Pipeline(
        model_detector=_empty_model_detector,
        use_safety_sweep=True,
    )
    single = single_pipeline.run(text, method="mask")

    streamer = StreamingDeidentifier(
        pipeline=stream_pipeline,
        max_buffer=48,
        method="mask",
    )
    events = []
    for chunk in _chunks(text, [6, 4, 9, 3, 12, 7, 5, 20]):
        events.extend(streamer.feed(chunk))
    events.extend(streamer.flush())

    audit_record = events[-1].audit_record
    assert audit_record is not None
    assert "".join(event.redacted_text for event in events) == single.redacted_text
    assert audit_record["span_count"] == single.audit_record["span_count"]
    assert audit_record["input_text_hash"] == single.audit_record["input_text_hash"]
    assert (
        audit_record["redacted_text_hash"]
        == (single.audit_record["redacted_text_hash"])
    )
    assert [span.text_hash for span in streamer.spans] == [
        span.text_hash for span in single.spans
    ]
