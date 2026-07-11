"""Tests for the RTL-aware redacted-output renderer.

All fixtures are synthetic. Arabic and Hebrew lines are constructed inline with
placeholder masks and Latin surrogates so no real PHI is committed. The tests
assert bidi-safe output (First Strong Isolate / Pop Directional Isolate around
each replacement), byte-for-byte LTR passthrough, correct source offsets, and a
lossless strip round-trip.
"""

from __future__ import annotations

from decimal import Decimal
from fractions import Fraction
from pathlib import Path

import pytest

import openmed.core.rtl_render as rtl_render
from openmed.core.rtl_render import (
    BIDI_CONTROL_CHARS,
    RenderedRedaction,
    base_direction,
    contains_rtl,
    render_redacted,
    strip_bidi_controls,
    wrap_mask,
)

# Bidi control code points asserted independently of the module constants so
# the tests pin the actual Unicode characters, not the module's own aliases.
_FSI = "\u2068"  # First Strong Isolate
_PDI = "\u2069"  # Pop Directional Isolate
_LRM = "\u200e"  # Left-to-Right Mark
_RLM = "\u200f"  # Right-to-Left Mark
_RLE = "\u202b"  # Right-to-Left Embedding
_PDF = "\u202c"  # Pop Directional Formatting
_ZWJ = "\u200d"  # Zero Width Joiner (not a bidi control)

_EXPECTED_BIDI_CONTROLS = frozenset(
    {
        "\u061c",
        "\u200e",
        "\u200f",
        *(chr(codepoint) for codepoint in range(0x202A, 0x202F)),
        *(chr(codepoint) for codepoint in range(0x2066, 0x206A)),
    }
)
_PARAGRAPH_SEPARATORS = ("\n", "\r", "\x1c", "\x1d", "\x1e", "\x85", "\u2029")

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


def test_source_contains_no_literal_bidi_controls():
    source = Path(rtl_render.__file__).read_text(encoding="utf-8")
    assert BIDI_CONTROL_CHARS.isdisjoint(source)


def test_bidi_control_set_matches_unicode_bidi_control_property():
    assert BIDI_CONTROL_CHARS == _EXPECTED_BIDI_CONTROLS


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


def test_length_changing_replacement_uses_rendered_offsets_only():
    original = "المريض John Smith زار"
    redacted = "المريض [NAME] زار"
    original_span = _span_of(original, "John Smith")
    rendered_span = _span_of(redacted, "[NAME]")
    assert original_span != rendered_span

    result = render_redacted(redacted, [rendered_span])

    isolated = result.text.split(_FSI, 1)[1].split(_PDI, 1)[0]
    assert isolated == "[NAME]"
    assert result.text.endswith(" زار")
    assert result.text == (
        f"{redacted[: rendered_span[0]]}{_FSI}[NAME]{_PDI}"
        f"{redacted[rendered_span[1] :]}"
    )


def test_span_accepts_mapping_and_object_forms():
    span_tuple = _span_of(_HEBREW_WITH_MASK, "[PERSON]")
    mapping_span = {"start": span_tuple[0], "end": span_tuple[1]}

    class _Entity:
        start, end = span_tuple

    for spans in ([span_tuple], [mapping_span], [_Entity()]):
        result = render_redacted(_HEBREW_WITH_MASK, spans)
        assert f"{_FSI}[PERSON]{_PDI}" in result.text
        assert result.source_offsets == (span_tuple,)


@pytest.mark.parametrize(
    "span",
    [
        (7.9, 13),
        (7.0, 13),
        (7, "13"),
        (True, 13),
        (7, False),
        (Decimal(7), 13),
        (7, Decimal(13)),
        (Fraction(7, 1), 13),
        (7, Fraction(13, 1)),
    ],
)
def test_span_offsets_reject_lossy_non_integer_values(span):
    with pytest.raises(ValueError, match="offset must be an integer"):
        render_redacted(_ARABIC_WITH_MASK, [span])


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


def test_non_string_direction_rejected():
    with pytest.raises(TypeError, match="direction must be a string"):
        render_redacted(_LTR_WITH_MASK, [], direction=None)  # type: ignore[arg-type]


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


def test_zero_width_span_nested_inside_replacement_is_ignored():
    name_span = _span_of(_ARABIC_WITH_MASK, "[NAME]")
    nested_zero_width = (name_span[0] + 1, name_span[0] + 1)

    result = render_redacted(_ARABIC_WITH_MASK, [name_span, nested_zero_width])

    assert result.text.count(_FSI) == 1
    assert result.text.count(_PDI) == 1
    assert result.source_offsets == (name_span,)
    assert strip_bidi_controls(result.text) == _ARABIC_WITH_MASK


def test_overlapping_spans_rejected():
    with pytest.raises(ValueError):
        render_redacted(_ARABIC_WITH_MASK, [(0, 5), (3, 8)])


def test_span_out_of_range_rejected():
    with pytest.raises(ValueError):
        render_redacted(_ARABIC_WITH_MASK, [(0, len(_ARABIC_WITH_MASK) + 1)])


@pytest.mark.parametrize("control", sorted(_EXPECTED_BIDI_CONTROLS))
def test_replacement_bidi_controls_are_rejected(control):
    text = f"المريض [NA{control}ME] زار"
    span = _span_of(text, f"[NA{control}ME]")

    with pytest.raises(ValueError, match="bidi control"):
        render_redacted(text, [span])


def test_ltr_replacement_bidi_control_is_rejected_before_noop():
    token = f"[NA{_PDI}ME]"
    text = f"Patient {token} visited"

    with pytest.raises(ValueError, match="bidi control"):
        render_redacted(text, [_span_of(text, token)])


@pytest.mark.parametrize("separator", _PARAGRAPH_SEPARATORS)
def test_replacement_paragraph_separators_are_rejected(separator):
    token = f"[NA{separator}ME]"
    text = f"المريض {token} زار"

    with pytest.raises(ValueError, match="paragraph separator"):
        render_redacted(text, [_span_of(text, token)])


def test_ltr_replacement_paragraph_separator_is_rejected_before_noop():
    token = "[NA\nME]"
    text = f"Patient {token} visited"

    with pytest.raises(ValueError, match="paragraph separator"):
        render_redacted(text, [_span_of(text, token)])


@pytest.mark.parametrize("separator", _PARAGRAPH_SEPARATORS)
def test_wrap_mask_rejects_paragraph_separators(separator):
    with pytest.raises(ValueError, match="paragraph separator"):
        wrap_mask(f"[NA{separator}ME]")

    with pytest.raises(ValueError, match="paragraph separator"):
        wrap_mask(f"[NA{separator}ME]", direction="ltr")


def test_non_bidi_zero_width_character_is_preserved_inside_replacement():
    token = f"علي{_ZWJ}رضا"
    text = f"المريض {token} زار"
    result = render_redacted(text, [_span_of(text, token)])

    assert f"{_FSI}{token}{_PDI}" in result.text
    assert strip_bidi_controls(result.text) == text


def test_html_like_replacement_remains_plain_text_for_caller_to_escape():
    token = "<b>[NAME]</b>"
    text = f"المريض {token} زار"
    result = render_redacted(text, [_span_of(text, token)])

    assert f"{_FSI}{token}{_PDI}" in result.text
    assert "&lt;" not in result.text
    assert strip_bidi_controls(result.text) == text


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


def test_strip_sanitizes_controls_that_predated_rendering():
    text = f"المريض{_RLM} [NAME] زار"
    result = render_redacted(text, [_span_of(text, "[NAME]")])

    assert _RLM in result.text
    assert strip_bidi_controls(result.text) == text.replace(_RLM, "")


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


def test_wrap_mask_invalid_direction_rejected():
    with pytest.raises(ValueError):
        wrap_mask("[NAME]", direction="sideways")


def test_wrap_mask_rejects_embedded_bidi_control():
    with pytest.raises(ValueError, match="bidi control"):
        wrap_mask(f"[NA{_PDI}ME]")

    with pytest.raises(ValueError, match="bidi control"):
        wrap_mask(f"[NA{_PDI}ME]", direction="ltr")


def test_wrap_mask_non_string_direction_rejected():
    with pytest.raises(TypeError, match="direction must be a string"):
        wrap_mask("[NAME]", direction=None)  # type: ignore[arg-type]


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
