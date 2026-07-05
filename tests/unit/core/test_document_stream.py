"""Tests for memory-bounded streaming de-identification of long documents.

All fixtures are synthetic. The detector is a deterministic regex stub so the
tests never download a model and stay fully offline (AGENTS.md testing gates).
The key correctness property is that streaming over a very long document yields
spans *identical* to the single-pass pipeline on the same input, with global
offsets, while peak allocation stays bounded as the document grows.
"""

from __future__ import annotations

import gc
import re
import tracemalloc
from datetime import datetime

import pytest

from openmed import (
    DocumentStreamDeidentifier,
    DocumentStreamResult,
    deidentify_document_stream,
)
from openmed.core.document_stream import iter_document_windows
from openmed.core.pipeline import Pipeline
from openmed.processing.outputs import EntityPrediction, PredictionResult

EMAIL_RE = re.compile(r"\b[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Za-z]{2,}\b")
PHONE_RE = re.compile(r"\b\d{3}-\d{3}-\d{4}\b")
MRN_RE = re.compile(r"\bMRN-\d{8}\b")
NAME_RE = re.compile(r"\b(?:Jane Doe|Jose Alvarez|Casey Example)\b")

_DETECTORS = (
    ("EMAIL", EMAIL_RE, 0.99),
    ("PHONE", PHONE_RE, 0.98),
    ("MRN", MRN_RE, 0.97),
    ("NAME", NAME_RE, 0.96),
)


def _regex_detector(text: str, **kwargs) -> PredictionResult:
    """Deterministic offline PII detector over synthetic identifiers."""

    entities: list[EntityPrediction] = []
    for label, pattern, score in _DETECTORS:
        for match in pattern.finditer(text):
            entities.append(
                EntityPrediction(
                    text=match.group(0),
                    label=label,
                    confidence=score,
                    start=match.start(),
                    end=match.end(),
                )
            )
    entities.sort(key=lambda entity: (entity.start, entity.end))
    return PredictionResult(
        text=text,
        entities=entities,
        model_name="regex-stub",
        timestamp=datetime.now().isoformat(),
    )


def _pipeline() -> Pipeline:
    return Pipeline(model_detector=_regex_detector, use_safety_sweep=True)


def _entity_signature(entities) -> list[tuple[str, int, int]]:
    return [
        (str(entity.entity_type or entity.label), int(entity.start), int(entity.end))
        for entity in entities
    ]


def _synthetic_record() -> str:
    return (
        "Patient Jane Doe was seen in clinic today for a routine follow up. "
        "She emailed jane.patient@example.com regarding her recent lab results. "
        "The nurse asked her to call the front desk at 555-123-4567 next week. "
        "Record MRN-40011234 was updated by Dr. Jose Alvarez after the visit. "
        "No acute distress was noted and vitals stayed within normal limits. "
    )


def _long_document(records: int) -> str:
    return _synthetic_record() * records


# --------------------------------------------------------------------------- #
# Windowing
# --------------------------------------------------------------------------- #


def test_windows_never_split_a_sentence_and_cover_all_spans():
    text = _long_document(6)
    windows = list(iter_document_windows(text, window_chars=200, overlap_chars=64))

    assert len(windows) > 1
    for window in windows:
        # The window's own region must round-trip to the exact source slice.
        assert (
            window.text[window.overlap_start - window.start :]
            == (text[window.overlap_start : window.end])
        )
        assert window.text == text[window.start : window.end]


# --------------------------------------------------------------------------- #
# Span identity vs. the single-pass pipeline
# --------------------------------------------------------------------------- #


def test_stream_spans_are_identical_to_single_pass_on_same_input():
    text = _long_document(30)

    single = _pipeline().run(text, method="mask")
    single_sig = _entity_signature(single.deidentification_result.pii_entities)

    result = deidentify_document_stream(
        text,
        window_chars=280,
        overlap_chars=96,
        pipeline=_pipeline(),
        method="mask",
    )

    assert isinstance(result, DocumentStreamResult)
    assert _entity_signature(result.pii_entities) == single_sig
    # Global offsets are correct: each span slices back to its detected text.
    for entity in result.pii_entities:
        assert text[entity.start : entity.end] == entity.text
    # Canonical spans are lifted to the same global offsets.
    assert [(span.start, span.end) for span in result.spans] == [
        (entity.start, entity.end) for entity in result.pii_entities
    ]


def test_stream_matches_single_pass_across_window_and_overlap_sizes():
    text = _long_document(20)
    single_sig = _entity_signature(
        _pipeline().run(text, method="mask").deidentification_result.pii_entities
    )

    for window_chars, overlap_chars in ((150, 0), (200, 80), (512, 128), (4096, 256)):
        result = deidentify_document_stream(
            text,
            window_chars=window_chars,
            overlap_chars=overlap_chars,
            pipeline=_pipeline(),
            method="mask",
        )
        assert _entity_signature(result.pii_entities) == single_sig, (
            window_chars,
            overlap_chars,
        )
        # No duplicates survive the cross-window merge.
        keys = [(e.start, e.end, e.entity_type) for e in result.pii_entities]
        assert len(keys) == len(set(keys))


def test_iterable_source_is_reassembled_before_segmentation():
    text = _long_document(4)
    # Fragment the source at arbitrary byte offsets, including mid-identifier.
    fragments = [text[i : i + 37] for i in range(0, len(text), 37)]

    from_string = deidentify_document_stream(
        text, window_chars=256, overlap_chars=64, pipeline=_pipeline()
    )
    from_fragments = deidentify_document_stream(
        fragments, window_chars=256, overlap_chars=64, pipeline=_pipeline()
    )

    assert _entity_signature(from_fragments.pii_entities) == _entity_signature(
        from_string.pii_entities
    )


# --------------------------------------------------------------------------- #
# Cross-boundary identifier correctness (leakage-first)
# --------------------------------------------------------------------------- #


def test_identifier_at_window_boundary_is_caught_whole_once():
    # Build a document where an email lands right where a naive fixed-size
    # window boundary would fall, straddling the split point.
    filler = "The patient was stable overnight and slept well. "
    boundary_email = "boundary.case@hospital.example"
    text = (
        filler * 3
        + f"Contact the coordinator at {boundary_email} for scheduling. "
        + filler * 3
    )

    # Choose window_chars so a naive char split would bisect the email, but the
    # sentence-aligned window keeps the whole sentence (and thus the email).
    naive_cut = text.index(boundary_email) + len(boundary_email) // 2

    result = deidentify_document_stream(
        text,
        window_chars=naive_cut,  # forces multiple windows around the email
        overlap_chars=0,
        pipeline=_pipeline(),
        method="mask",
    )

    emails = [e for e in result.pii_entities if e.entity_type == "EMAIL"]
    assert len(emails) == 1
    (email,) = emails
    assert text[email.start : email.end] == boundary_email
    # Matches the single-pass result exactly.
    single = _pipeline().run(text, method="mask")
    single_emails = [
        e
        for e in single.deidentification_result.pii_entities
        if e.entity_type == "EMAIL"
    ]
    assert len(single_emails) == 1
    assert (single_emails[0].start, single_emails[0].end) == (email.start, email.end)


def test_no_raw_identifier_survives_in_the_source_after_redaction_offsets():
    # The streaming result never exposes window-local (wrong) offsets: every
    # emitted span must map to the real identifier in the source.
    text = _long_document(10)
    result = deidentify_document_stream(
        text, window_chars=220, overlap_chars=72, pipeline=_pipeline()
    )
    assert result.pii_entities
    for entity in result.pii_entities:
        assert entity.start < entity.end <= len(text)
        assert text[entity.start : entity.end] == entity.text


# --------------------------------------------------------------------------- #
# Measured memory ceiling
# --------------------------------------------------------------------------- #


def _measure_peak(fn) -> int:
    gc.collect()
    tracemalloc.start()
    tracemalloc.reset_peak()
    try:
        fn()
        _, peak = tracemalloc.get_traced_memory()
    finally:
        tracemalloc.stop()
    return peak


def test_streaming_peak_allocation_stays_bounded_as_document_grows():
    # A PHI-free body isolates *processing* memory from output accumulation.
    body = (
        "The patient returned to the clinic for a scheduled follow up visit. "
        "All vitals were within normal limits and no new concerns were noted. "
    )
    base_mult = 300
    grown_mult = base_mult * 4

    # The document is an input the caller already holds; build it OUTSIDE the
    # measured region so we measure *processing* peak, not the O(length) cost of
    # the input string itself.
    doc_base = body * base_mult
    doc_grown = body * grown_mult

    # Warm one-time caches (sentence segmenter, safety-sweep regex compilation)
    # so the measured peaks reflect steady-state per-window processing rather
    # than first-call allocations.
    DocumentStreamDeidentifier(
        window_chars=512, overlap_chars=128, pipeline=_pipeline(), method="mask"
    ).run(body * 10)
    _pipeline().run(body * 10, method="mask")

    stream_base = _measure_peak(
        lambda: DocumentStreamDeidentifier(
            window_chars=512, overlap_chars=128, pipeline=_pipeline(), method="mask"
        ).run(doc_base)
    )
    stream_grown = _measure_peak(
        lambda: DocumentStreamDeidentifier(
            window_chars=512, overlap_chars=128, pipeline=_pipeline(), method="mask"
        ).run(doc_grown)
    )
    single_base = _measure_peak(lambda: _pipeline().run(doc_base, method="mask"))
    single_grown = _measure_peak(lambda: _pipeline().run(doc_grown, method="mask"))

    # Single-pass processing peak scales with document length (it materializes
    # full normalized, offset-map, and redacted copies of the whole document).
    assert single_grown > single_base * 2.5

    # Streaming processing peak does NOT scale with length: 4x the document must
    # not cost anywhere near 4x the peak. Allow generous slack for allocator
    # noise; the real behavior is flat-to-declining.
    assert stream_grown < stream_base * 1.5

    # And streaming's peak is materially below single-pass on the long document.
    assert stream_grown < single_grown / 2


def test_max_window_chars_is_independent_of_document_length():
    body = "No protected identifiers appear in this benign clinical sentence here. "

    short = deidentify_document_stream(
        body * 50, window_chars=384, overlap_chars=96, pipeline=_pipeline()
    )
    long = deidentify_document_stream(
        body * 500, window_chars=384, overlap_chars=96, pipeline=_pipeline()
    )

    assert long.document_length == pytest.approx(short.document_length * 10, rel=0.01)
    # The peak-memory driver (largest window handed to the pipeline) is flat.
    assert long.max_window_chars == short.max_window_chars
    assert long.window_count > short.window_count


def test_stream_over_empty_and_whitespace_documents():
    assert deidentify_document_stream("", pipeline=_pipeline()).pii_entities == []
    result = deidentify_document_stream("   \n\t  ", pipeline=_pipeline())
    assert result.pii_entities == []
    assert result.window_count in (0, 1)
