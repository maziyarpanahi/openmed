"""Tests for full-width/half-width normalization with offset preservation."""

from __future__ import annotations

import re

import pytest

import openmed
from openmed.core.config import OpenMedConfig
from openmed.core.decoding.spans import trim_span_whitespace
from openmed.core.pii_entity_merger import find_semantic_units
from openmed.core.pii_i18n import get_patterns_for_language
from openmed.core.script_detect import normalize_for_pii_detection
from openmed.processing.outputs import PredictionResult
from openmed.processing.zh_normalize import (
    CJK_CONVENTION,
    STRICT_NFKC,
    WidthNormalization,
    detect_width_normalized,
    normalize_width,
)

# --------------------------------------------------------------------------
# Core width normalization
# --------------------------------------------------------------------------


def test_fullwidth_ascii_maps_to_halfwidth():
    result = normalize_width("ＡＢＣ１２３")

    assert isinstance(result, WidthNormalization)
    assert result.text == "ABC123"
    # 1:1 code-point mapping keeps offsets identical.
    assert result.char_origins == tuple((i, i + 1) for i in range(6))


def test_ideographic_space_maps_to_configurable_target():
    assert normalize_width("Ａ　Ｂ").text == "A B"
    assert normalize_width("Ａ　Ｂ", space_target="\t").text == "A\tB"


def test_genuine_cjk_is_left_unchanged():
    text = "中文「引用」测试"
    result = normalize_width(text)

    assert result.text == text
    assert result.char_origins == tuple((i, i + 1) for i in range(len(text)))


# --------------------------------------------------------------------------
# Offset round-trip back to original code points
# --------------------------------------------------------------------------


def test_fullwidth_phone_matches_halfwidth_pattern_and_maps_back():
    original = "电话（１３８１２３４５６７８）"
    result = normalize_width(original)

    match = re.search(r"\d{11}", result.text)
    assert match is not None

    start, end = result.to_original_span(match.start(), match.end())
    # The mapped span covers exactly the original full-width digit run.
    assert original[start:end] == "１３８１２３４５６７８"


def test_offset_round_trip_gate_over_synthetic_strings():
    # 500 deterministic synthetic strings mixing full-width, U+3000 and Han.
    fw_digits = [chr(0xFF10 + d) for d in range(10)]
    for n in range(500):
        parts = []
        for k in range(6):
            token = (n + k) % 4
            if token == 0:
                parts.append(fw_digits[(n + k) % 10])
            elif token == 1:
                parts.append("　")
            elif token == 2:
                parts.append(chr(0xFF21 + ((n + k) % 26)))  # full-width A-Z
            else:
                parts.append("中")
        original = "".join(parts)
        result = normalize_width(original)
        # The whole normalized span round-trips to the whole original.
        assert result.to_original_span(0, len(result.text)) == (0, len(original))
        # Every normalized character maps back to a source char that, when
        # normalized, reproduces it -- a real round-trip, not just valid bounds.
        for i in range(len(result.text)):
            o_start, o_end = result.to_original_span(i, i + 1)
            assert 0 <= o_start < o_end <= len(original)
            assert result.text[i] in normalize_width(original[o_start:o_end]).text


def test_strict_nfkc_handles_many_to_one_expansion():
    result = normalize_width("㎏", convention=STRICT_NFKC)

    assert result.text == "kg"
    # Both expanded chars trace back to the single original code point.
    assert result.char_origins == ((0, 1), (0, 1))
    assert result.to_original_span(0, 2) == (0, 1)


def test_cjk_convention_keeps_han_but_nfkc_would_change_it_is_noop():
    # In CJK convention Han stays full-width; the two conventions differ only
    # on compatibility characters, not on plain Han.
    assert normalize_width("中", convention=CJK_CONVENTION).text == "中"


def test_invalid_convention_raises():
    with pytest.raises(ValueError):
        normalize_width("Ａ", convention="bogus")


# --------------------------------------------------------------------------
# U+3000 whitespace trimming in spans
# --------------------------------------------------------------------------


def test_trim_span_whitespace_trims_ideographic_space():
    text = "　中文　"
    start, end = trim_span_whitespace(0, len(text), text)
    assert text[start:end] == "中文"


def test_trim_does_not_strip_interior_han():
    text = "中　文"
    start, end = trim_span_whitespace(0, len(text), text)
    assert text[start:end] == text  # interior space kept, no Han stripped


# --------------------------------------------------------------------------
# Pre-pass integration: existing PHI engine matches full-width after normalize
# --------------------------------------------------------------------------


def test_prepass_detects_fullwidth_date_via_existing_engine():
    original = "就诊日期：２０２４－０１－１５"
    patterns = get_patterns_for_language("en")

    def matcher(normalized: str):
        return [(u[0], u[1], u[2]) for u in find_semantic_units(normalized, patterns)]

    results = detect_width_normalized(original, matcher)

    assert results, "full-width date should be detected after normalization"
    start, end, label = results[0]
    assert label == "date"
    # Span maps back to the exact original full-width code points.
    assert original[start:end] == "２０２４－０１－１５"


def test_detection_normalization_composes_width_and_source_maps():
    original = "ＩＤ　㎏"
    result = normalize_for_pii_detection(original, width_convention="nfkc")

    assert result.text == "ID kg"
    assert result.remap_span(0, 2) == (0, 2)
    assert result.remap_span(3, 5) == (3, 4)


def test_extract_pii_matches_normalized_width_and_returns_original_span(monkeypatch):
    original = "就诊日期：２０２４－０１－１５"
    observed_inputs = []

    def fake_analyze_text(text, **_kwargs):
        observed_inputs.append(text)
        return PredictionResult(
            text=text,
            entities=[],
            model_name="fixture-pii-model",
            timestamp="2026-01-01T00:00:00",
        )

    monkeypatch.setattr(openmed, "analyze_text", fake_analyze_text)

    result = openmed.extract_pii(original, model_name="fixture-pii-model")

    assert observed_inputs == ["就诊日期:2024-01-15"]
    assert [(entity.text, entity.start, entity.end) for entity in result.entities] == [
        ("２０２４－０１－１５", 5, 15)
    ]


# --------------------------------------------------------------------------
# Config policy switch
# --------------------------------------------------------------------------


def test_config_defaults_to_cjk_convention():
    assert OpenMedConfig().cjk_width_convention == "cjk"


def test_config_accepts_nfkc_and_rejects_unknown():
    assert OpenMedConfig(cjk_width_convention="nfkc").cjk_width_convention == "nfkc"
    with pytest.raises(ValueError):
        OpenMedConfig(cjk_width_convention="bogus")


def test_config_from_dict_preserves_width_convention():
    config = OpenMedConfig.from_dict({"cjk_width_convention": "nfkc"})
    assert config.cjk_width_convention == "nfkc"
    assert config.to_dict()["cjk_width_convention"] == "nfkc"
