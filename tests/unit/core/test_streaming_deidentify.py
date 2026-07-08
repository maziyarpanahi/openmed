from __future__ import annotations

import json
import re
from datetime import datetime
from unittest.mock import patch

import pytest

from openmed import deidentify_stream
from openmed.core.decoding import (
    build_label_info,
    viterbi_decode,
    viterbi_decode_incremental,
)
from openmed.core.pii import deidentify
from openmed.core.pipeline import Pipeline
from openmed.core.streaming import StreamingBufferError, StreamingDeidentifier
from openmed.processing.advanced_ner import (
    StreamingTokenClassifier,
    replay_token_classifier,
)
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


EMAIL_RE = re.compile(r"\b[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Za-z]{2,}\b")
PHONE_RE = re.compile(r"\b\d{3}-\d{3}-\d{4}\b")
NAME_RE = re.compile(r"\b(?:Jane Doe|Jose Alvarez)\b")


def _regex_token_classifier(text: str):
    entities = []
    for label, pattern, score in (
        ("EMAIL", EMAIL_RE, 0.99),
        ("PHONE", PHONE_RE, 0.97),
        ("NAME", NAME_RE, 0.95),
    ):
        for match in pattern.finditer(text):
            entities.append(
                {
                    "entity_group": label,
                    "score": score,
                    "word": match.group(0),
                    "start": match.start(),
                    "end": match.end(),
                }
            )
    return sorted(entities, key=lambda item: (item["start"], item["end"]))


def _span_signature(spans):
    return [(span.label, span.start, span.end, span.text) for span in spans]


def test_token_classifier_streaming_replay_matches_batch_for_chars_and_chunks():
    text = (
        "Jose Alvarez emailed jane.patient@example.com, then called "
        "555-123-4567 before discharge."
    )
    kwargs = {
        "window_chars": 96,
        "tokenizer_context_chars": 16,
        "max_entity_chars": 48,
    }

    char_replay = replay_token_classifier(
        _regex_token_classifier,
        text,
        list(text),
        **kwargs,
    )
    chunk_replay = replay_token_classifier(
        _regex_token_classifier,
        text,
        _chunks(text, [1, 11, 7, 3, 19, 5, 8]),
        **kwargs,
    )

    assert char_replay.span_diff == []
    assert chunk_replay.span_diff == []
    assert _span_signature(char_replay.final_spans) == _span_signature(
        char_replay.batch_spans
    )
    assert _span_signature(chunk_replay.final_spans) == _span_signature(
        chunk_replay.batch_spans
    )


def test_token_classifier_emit_retract_reemits_extended_span_with_stable_id():
    def classifier(text: str):
        match = re.search(r"jane@example(?:\.com)?", text)
        if not match:
            return []
        return [
            {
                "entity_group": "EMAIL",
                "score": 0.99,
                "word": match.group(0),
                "start": match.start(),
                "end": match.end(),
            }
        ]

    streamer = StreamingTokenClassifier(
        classifier,
        window_chars=64,
        tokenizer_context_chars=16,
        max_entity_chars=32,
    )
    events = [
        *streamer.append("Contact jane@example"),
        *streamer.append(".com now"),
        *streamer.finish(),
    ]
    emitted = [event for event in events if event.type == "emit"]
    retracted = [event for event in events if event.type == "retract"]

    assert len(emitted) == 2
    assert len(retracted) == 1
    assert emitted[0].entity_id == retracted[0].entity_id == emitted[1].entity_id
    assert emitted[0].span is not None
    assert emitted[1].span is not None
    assert emitted[0].span.end < emitted[1].span.end
    assert len({span.id for span in streamer.final_spans}) == len(streamer.final_spans)


def test_token_classifier_streaming_window_size_stays_bounded_on_long_input():
    observed_window_lengths: list[int] = []

    def classifier(text: str):
        observed_window_lengths.append(len(text))
        return []

    text = "No PHI here. " * 4000
    replay = replay_token_classifier(
        classifier,
        text,
        _chunks(text, [113] * 1000),
        window_chars=256,
        tokenizer_context_chars=32,
        max_entity_chars=64,
    )

    streaming_window_lengths = observed_window_lengths[1:]
    assert max(streaming_window_lengths) <= 256
    assert replay.latency["p99_window_chars"] <= 256 * 1.5
    assert replay.latency["max_window_chars"] <= 256


def test_token_classifier_audit_log_omits_raw_phi_and_keeps_byte_offsets():
    text = "Jose Alvarez emailed jane.patient@example.com."
    streamer = StreamingTokenClassifier(
        _regex_token_classifier,
        window_chars=96,
        tokenizer_context_chars=16,
        max_entity_chars=48,
    )
    streamer.append(text[:18])
    streamer.append(text[18:])
    streamer.finish()

    audit_payload = json.dumps(streamer.audit_log)
    assert "Jose Alvarez" not in audit_payload
    assert "jane.patient" not in audit_payload

    email_span = next(span for span in streamer.final_spans if span.label == "EMAIL")
    assert email_span.byte_start == len(text[: email_span.start].encode("utf-8"))
    assert email_span.byte_end == len(text[: email_span.end].encode("utf-8"))


def test_incremental_viterbi_matches_full_decode_after_commit_boundary():
    id2label = {
        0: "O",
        1: "B-NAME",
        2: "I-NAME",
        3: "E-NAME",
        4: "S-EMAIL",
    }
    label_info = build_label_info(id2label)
    logprobs = [
        [-5.0, 0.0, -5.0, -5.0, -5.0],
        [-5.0, -5.0, -0.1, -5.0, -5.0],
        [-5.0, -5.0, -5.0, -0.2, -5.0],
        [-5.0, -5.0, -5.0, -5.0, -0.1],
        [0.0, -5.0, -5.0, -5.0, -5.0],
    ]

    full = viterbi_decode(logprobs, label_info=label_info, biases={})
    prefix, state = viterbi_decode_incremental(
        logprobs[:3],
        label_info=label_info,
        biases={},
    )
    suffix, resumed_state = viterbi_decode_incremental(
        logprobs[3:],
        label_info=label_info,
        biases={},
        state=state,
    )

    assert prefix + suffix == full
    assert state.token_count == 3
    assert state.last_backpointer
    assert resumed_state.token_count == len(logprobs)
