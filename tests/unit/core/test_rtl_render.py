"""Tests for the RTL-aware redacted-output renderer.

All fixtures are synthetic. Arabic and Hebrew lines are constructed inline with
placeholder masks and Latin surrogates so no real PHI is committed. The tests
assert bidi-safe output (First Strong Isolate / Pop Directional Isolate around
each replacement), byte-for-byte LTR passthrough, correct source offsets, and a
lossless strip round-trip.
"""

from __future__ import annotations

import pytest

from openmed.core.rtl_render import (
    RenderedRedaction,
    base_direction,
    contains_rtl,
    render_redacted,
    strip_bidi_controls,
    wrap_mask,
)

# Bidi control code points asserted independently of the module constants so
# the tests pin the actual Unicode characters, not the module's own aliases.
_FSI = "⁨"  # First Strong Isolate
_PDI = "⁩"  # Pop Directional Isolate
_LRM = "‎"  # Left-to-Right Mark
_RLM = "‏"  # Right-to-Left Mark
_RLE = "‫"  # Right-to-Left Embedding
_PDF = "‬"  # Pop Directional Formatting

# Synthetic redacted lines: masks/surrogates already substituted for PII.
_ARABIC_WITH_SURROGATE = "المريض John Smith زار"
_ARABIC_WITH_MASK = "المريض [NAME] زار العيادة"
_HEBREW_WITH_MASK = "המטופל [PERSON] הגיע"
_LTR_WITH_MASK = "Patient [NAME] visited the clinic"


def _span_of(text: str, token: str) -> tuple[int, int]:
    start = text.index(token)
    return start, start + len(token)


# --- Base direction ----------------------------------------------------------


def test_base_direction_arabic_is_rtl():
    assert base_direction(_ARABIC_WITH_MASK) == "rtl"


def test_base_direction_hebrew_is_rtl():
    assert base_direction(_HEBREW_WITH_MASK) == "rtl"


def test_base_direction_latin_is_ltr():
    assert base_direction(_LTR_WITH_MASK) == "ltr"


def test_base_direction_empty_is_ltr():
    assert base_direction("") == "ltr"


def test_base_direction_digits_only_is_ltr():
    assert base_direction("12345 - 67890") == "ltr"


def test_contains_rtl_detects_arabic_run():
    assert contains_rtl(_ARABIC_WITH_MASK) is True
    assert contains_rtl(_HEBREW_WITH_MASK) is True
    assert contains_rtl(_LTR_WITH_MASK) is False


# --- Arabic line with an injected Latin surrogate ----------------------------


def test_arabic_latin_surrogate_is_isolated():
    span = _span_of(_ARABIC_WITH_SURROGATE, "John Smith")
    result = render_redacted(_ARABIC_WITH_SURROGATE, [span])

    assert result.base_direction == "rtl"
    assert result.isolated is True
    # The surrogate is wrapped in FSI ... PDI so the Latin run cannot reorder.
    assert f"{_FSI}John Smith{_PDI}" in result.text
    assert result.source_offsets == (span,)


def test_arabic_latin_surrogate_round_trips_through_strip():
    span = _span_of(_ARABIC_WITH_SURROGATE, "John Smith")
    result = render_redacted(_ARABIC_WITH_SURROGATE, [span])

    # Round-trip render -> strip recovers the original redacted line exactly,
    # surrogate text intact.
    assert strip_bidi_controls(result.text) == _ARABIC_WITH_SURROGATE
    assert "John Smith" in strip_bidi_controls(result.text)


# --- Hebrew line with a [PERSON] mask ----------------------------------------


def test_hebrew_person_mask_is_isolated():
    span = _span_of(_HEBREW_WITH_MASK, "[PERSON]")
    result = render_redacted(_HEBREW_WITH_MASK, [span])

    assert result.base_direction == "rtl"
    assert result.isolated is True
    assert f"{_FSI}[PERSON]{_PDI}" in result.text
    assert strip_bidi_controls(result.text) == _HEBREW_WITH_MASK


# --- Pure-LTR no-op ----------------------------------------------------------


def test_pure_ltr_is_unchanged_byte_for_byte():
    span = _span_of(_LTR_WITH_MASK, "[NAME]")
    result = render_redacted(_LTR_WITH_MASK, [span])

    assert result.base_direction == "ltr"
    assert result.isolated is False
    # No isolate characters are added for a pure-LTR document.
    assert result.text == _LTR_WITH_MASK
    assert _FSI not in result.text
    assert _PDI not in result.text
    assert result.source_offsets == ()


def test_pure_ltr_render_output_encodes_identically():
    result = render_redacted(_LTR_WITH_MASK, [_span_of(_LTR_WITH_MASK, "[NAME]")])
    assert result.text.encode("utf-8") == _LTR_WITH_MASK.encode("utf-8")


# --- Offsets and multi-span ordering -----------------------------------------


def test_multiple_masks_each_isolated_in_order():
    text = "שם [PERSON] מספר [ID] סוף"
    person = _span_of(text, "[PERSON]")
    ident = _span_of(text, "[ID]")
    result = render_redacted(text, [ident, person])  # deliberately unsorted

    assert result.text.count(_FSI) == 2
    assert result.text.count(_PDI) == 2
    # Source offsets are returned sorted by position regardless of input order.
    assert result.source_offsets == (person, ident)
    assert strip_bidi_controls(result.text) == text


def test_isolate_boundaries_match_span_offsets():
    span = _span_of(_ARABIC_WITH_MASK, "[NAME]")
    result = render_redacted(_ARABIC_WITH_MASK, [span])

    fsi_index = result.text.index(_FSI)
    pdi_index = result.text.index(_PDI)
    # Exactly the span text sits between the isolates.
    assert result.text[fsi_index + 1 : pdi_index] == "[NAME]"


def test_prefix_and_suffix_preserved_around_isolates():
    span = _span_of(_ARABIC_WITH_MASK, "[NAME]")
    start, end = span
    result = render_redacted(_ARABIC_WITH_MASK, [span])

    prefix = _ARABIC_WITH_MASK[:start]
    suffix = _ARABIC_WITH_MASK[end:]
    assert result.text == f"{prefix}{_FSI}[NAME]{_PDI}{suffix}"


def test_span_accepts_mapping_and_object_forms():
    span_tuple = _span_of(_HEBREW_WITH_MASK, "[PERSON]")
    mapping_span = {"start": span_tuple[0], "end": span_tuple[1]}

    class _Entity:
        start, end = span_tuple

    for spans in ([span_tuple], [mapping_span], [_Entity()]):
        result = render_redacted(_HEBREW_WITH_MASK, spans)
        assert f"{_FSI}[PERSON]{_PDI}" in result.text
        assert result.source_offsets == (span_tuple,)


# --- Direction override ------------------------------------------------------


def test_direction_forced_rtl_isolates_even_for_latin():
    span = _span_of(_LTR_WITH_MASK, "[NAME]")
    result = render_redacted(_LTR_WITH_MASK, [span], direction="rtl")

    assert result.base_direction == "rtl"
    assert result.isolated is True
    assert f"{_FSI}[NAME]{_PDI}" in result.text


def test_direction_forced_ltr_skips_isolation_for_rtl_text():
    span = _span_of(_ARABIC_WITH_MASK, "[NAME]")
    result = render_redacted(_ARABIC_WITH_MASK, [span], direction="ltr")

    assert result.base_direction == "ltr"
    assert result.isolated is False
    assert result.text == _ARABIC_WITH_MASK


def test_invalid_direction_rejected():
    with pytest.raises(ValueError):
        render_redacted(_LTR_WITH_MASK, [], direction="sideways")


# --- Empty / degenerate spans ------------------------------------------------


def test_no_spans_reports_direction_without_isolating():
    result = render_redacted(_ARABIC_WITH_MASK)
    assert result.base_direction == "rtl"
    assert result.isolated is False
    assert result.text == _ARABIC_WITH_MASK


def test_zero_width_span_is_ignored():
    result = render_redacted(_ARABIC_WITH_MASK, [(3, 3)])
    assert result.isolated is False
    assert result.text == _ARABIC_WITH_MASK


def test_overlapping_spans_rejected():
    with pytest.raises(ValueError):
        render_redacted(_ARABIC_WITH_MASK, [(0, 5), (3, 8)])


def test_span_out_of_range_rejected():
    with pytest.raises(ValueError):
        render_redacted(_ARABIC_WITH_MASK, [(0, len(_ARABIC_WITH_MASK) + 1)])


def test_non_string_text_rejected():
    with pytest.raises(TypeError):
        render_redacted(None)  # type: ignore[arg-type]


# --- strip_bidi_controls -----------------------------------------------------


def test_strip_removes_isolates_and_recovers_replacement():
    wrapped = f"{_FSI}[NAME]{_PDI}"
    assert strip_bidi_controls(wrapped) == "[NAME]"


def test_strip_removes_marks_and_embeddings():
    noisy = f"{_RLE}{_LRM}[ID]{_RLM}{_PDF}"
    assert strip_bidi_controls(noisy) == "[ID]"


def test_strip_leaves_clean_text_unchanged():
    clean = _ARABIC_WITH_MASK
    assert strip_bidi_controls(clean) == clean


def test_strip_rejects_non_string():
    with pytest.raises(TypeError):
        strip_bidi_controls(None)  # type: ignore[arg-type]


def test_render_then_strip_is_identity_for_rtl():
    for text, token in (
        (_ARABIC_WITH_SURROGATE, "John Smith"),
        (_ARABIC_WITH_MASK, "[NAME]"),
        (_HEBREW_WITH_MASK, "[PERSON]"),
    ):
        span = _span_of(text, token)
        rendered = render_redacted(text, [span])
        assert rendered.isolated is True
        assert strip_bidi_controls(rendered.text) == text


# --- wrap_mask convenience ---------------------------------------------------


def test_wrap_mask_isolates_by_default():
    assert wrap_mask("[NAME]") == f"{_FSI}[NAME]{_PDI}"


def test_wrap_mask_ltr_is_noop():
    assert wrap_mask("[NAME]", direction="ltr") == "[NAME]"


def test_wrap_mask_round_trips_through_strip():
    assert strip_bidi_controls(wrap_mask("[PERSON]")) == "[PERSON]"


def test_wrap_mask_rejects_non_string():
    with pytest.raises(TypeError):
        wrap_mask(None)  # type: ignore[arg-type]


# --- Result container --------------------------------------------------------


def test_result_is_frozen_and_str_returns_text():
    result = render_redacted(_LTR_WITH_MASK, [_span_of(_LTR_WITH_MASK, "[NAME]")])
    assert isinstance(result, RenderedRedaction)
    assert str(result) == result.text
    assert result.is_rtl is False
    with pytest.raises(Exception):
        result.text = "mutated"  # type: ignore[misc]
