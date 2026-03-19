"""Unit tests for span-boundary quality gates."""

from __future__ import annotations

import warnings

import pytest

from openmed.core.quality_gates import (
    SpanValidationWarning,
    detect_overlapping_entities,
    validate_entity_spans,
)
from openmed.processing.outputs import EntityPrediction


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _ent(text, label="NAME", start=0, end=None, confidence=0.9):
    if end is None:
        end = start + len(text)
    return EntityPrediction(
        text=text, label=label, start=start, end=end, confidence=confidence,
    )


# ---------------------------------------------------------------------------
# validate_entity_spans
# ---------------------------------------------------------------------------


class TestValidateEntitySpans:
    """Tests for validate_entity_spans."""

    def test_valid_entities_no_warnings(self):
        text = "Patient John Doe visited the clinic"
        entities = [_ent("John Doe", start=8, end=16)]
        with warnings.catch_warnings(record=True) as w:
            warnings.simplefilter("always")
            result = validate_entity_spans(entities, text)
            assert len(w) == 0
        assert result is entities  # returns same list
        assert entities[0].metadata["span_valid"] is True

    def test_inverted_span_warns(self):
        text = "Hello world"
        entities = [_ent("lo", start=5, end=3)]
        with warnings.catch_warnings(record=True) as w:
            warnings.simplefilter("always")
            validate_entity_spans(entities, text)
            assert any(issubclass(x.category, SpanValidationWarning) for x in w)
        assert entities[0].metadata["span_valid"] is False

    def test_zero_length_span_warns(self):
        text = "Hello world"
        entities = [_ent("", start=3, end=3)]
        with warnings.catch_warnings(record=True) as w:
            warnings.simplefilter("always")
            validate_entity_spans(entities, text)
            span_warns = [x for x in w if issubclass(x.category, SpanValidationWarning)]
            assert len(span_warns) >= 1
        assert entities[0].metadata["span_valid"] is False

    def test_negative_start_warns(self):
        text = "Hello"
        entities = [_ent("He", start=-1, end=2)]
        with warnings.catch_warnings(record=True) as w:
            warnings.simplefilter("always")
            validate_entity_spans(entities, text)
            assert any(issubclass(x.category, SpanValidationWarning) for x in w)
        assert entities[0].metadata["span_valid"] is False

    def test_end_exceeds_text_length_warns(self):
        text = "Hi"
        entities = [_ent("Hi!", start=0, end=5)]
        with warnings.catch_warnings(record=True) as w:
            warnings.simplefilter("always")
            validate_entity_spans(entities, text)
            assert any(issubclass(x.category, SpanValidationWarning) for x in w)
        assert entities[0].metadata["span_valid"] is False

    def test_text_mismatch_warns(self):
        text = "Patient John Doe visited"
        # Entity claims text is "Jane Doe" but span points to "John Doe"
        entities = [EntityPrediction(
            text="Jane Doe", label="NAME", start=8, end=16, confidence=0.9,
        )]
        with warnings.catch_warnings(record=True) as w:
            warnings.simplefilter("always")
            validate_entity_spans(entities, text)
            span_warns = [x for x in w if issubclass(x.category, SpanValidationWarning)]
            assert len(span_warns) == 1
            assert "text mismatch" in str(span_warns[0].message)
        assert entities[0].metadata["span_valid"] is False

    def test_entities_without_offsets_skipped(self):
        text = "Hello"
        entities = [EntityPrediction(
            text="Hello", label="GREETING", confidence=0.9,
        )]
        with warnings.catch_warnings(record=True) as w:
            warnings.simplefilter("always")
            validate_entity_spans(entities, text)
            assert len(w) == 0
        # No metadata added for offset-less entities
        assert entities[0].metadata is None

    def test_multiple_entities_mixed_validity(self):
        text = "John Doe 555-1234"
        entities = [
            _ent("John Doe", start=0, end=8),
            _ent("wrong", start=9, end=17),  # text mismatch
        ]
        with warnings.catch_warnings(record=True) as w:
            warnings.simplefilter("always")
            validate_entity_spans(entities, text)
            assert len([x for x in w if issubclass(x.category, SpanValidationWarning)]) == 1
        assert entities[0].metadata["span_valid"] is True
        assert entities[1].metadata["span_valid"] is False

    def test_preserves_existing_metadata(self):
        text = "John"
        entities = [EntityPrediction(
            text="John", label="NAME", start=0, end=4, confidence=0.9,
            metadata={"source": "model"},
        )]
        validate_entity_spans(entities, text)
        assert entities[0].metadata["source"] == "model"
        assert entities[0].metadata["span_valid"] is True


# ---------------------------------------------------------------------------
# detect_overlapping_entities
# ---------------------------------------------------------------------------


class TestDetectOverlappingEntities:
    """Tests for detect_overlapping_entities."""

    def test_no_overlaps(self):
        entities = [
            _ent("John", start=0, end=4),
            _ent("Doe", start=5, end=8),
        ]
        assert detect_overlapping_entities(entities) == []

    def test_adjacent_entities_no_overlap(self):
        entities = [
            _ent("John", start=0, end=4),
            _ent(" Doe", start=4, end=8),
        ]
        assert detect_overlapping_entities(entities) == []

    def test_simple_overlap(self):
        entities = [
            _ent("John D", start=0, end=6),
            _ent("Doe", start=5, end=8),
        ]
        overlaps = detect_overlapping_entities(entities)
        assert len(overlaps) == 1
        assert overlaps[0][0].text == "John D"
        assert overlaps[0][1].text == "Doe"

    def test_nested_entity(self):
        entities = [
            _ent("John Doe", start=0, end=8),
            _ent("Doe", start=5, end=8),
        ]
        overlaps = detect_overlapping_entities(entities)
        assert len(overlaps) == 1

    def test_multiple_overlaps(self):
        entities = [
            _ent("AAAA", start=0, end=4),
            _ent("AABB", start=2, end=6),
            _ent("BBCC", start=4, end=8),
        ]
        overlaps = detect_overlapping_entities(entities)
        # A overlaps B, B overlaps C
        assert len(overlaps) == 2

    def test_entities_without_offsets_skipped(self):
        entities = [
            EntityPrediction(text="John", label="NAME", confidence=0.9),
            _ent("Doe", start=5, end=8),
        ]
        assert detect_overlapping_entities(entities) == []

    def test_unordered_input_still_works(self):
        entities = [
            _ent("Doe", start=5, end=8),
            _ent("John D", start=0, end=6),
        ]
        overlaps = detect_overlapping_entities(entities)
        assert len(overlaps) == 1

    def test_empty_list(self):
        assert detect_overlapping_entities([]) == []

    def test_single_entity(self):
        assert detect_overlapping_entities([_ent("John", start=0, end=4)]) == []


# ---------------------------------------------------------------------------
# Integration: _fix_entity_spans output
# ---------------------------------------------------------------------------


class TestIntegrationWithFixEntitySpans:
    """Verify guards work on output of _fix_entity_spans."""

    def test_fix_entity_spans_output_passes_validation(self):
        from openmed.processing.outputs import OutputFormatter
        text = "Patient John visited on 2024-01-15"
        entities = [
            EntityPrediction(
                text="Joh", label="NAME", start=8, end=11, confidence=0.9,
            ),
        ]
        fixed = OutputFormatter._fix_entity_spans(entities, text)
        with warnings.catch_warnings(record=True) as w:
            warnings.simplefilter("always")
            validate_entity_spans(fixed, text)
            span_warns = [x for x in w if issubclass(x.category, SpanValidationWarning)]
            assert len(span_warns) == 0
        assert all(e.metadata.get("span_valid") for e in fixed)
