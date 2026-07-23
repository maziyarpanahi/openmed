"""Tests for Chinese normalization with offset preservation."""

from __future__ import annotations

import re
import warnings

import pytest

import openmed
from openmed.core.config import OpenMedConfig
from openmed.core.decoding.spans import trim_span_whitespace
from openmed.core.pii_entity_merger import find_semantic_units
from openmed.core.pii_i18n import get_patterns_for_language
from openmed.core.script_detect import (
    TRADITIONAL_VARIANT_CHARS,
    ChineseScriptVariant,
    detect_chinese_script,
    normalize_for_pii_detection,
)
from openmed.processing.outputs import EntityPrediction, PredictionResult
from openmed.processing.zh_normalize import (
    CJK_CONVENTION,
    STRICT_NFKC,
    ChineseTargetScript,
    OpenCCConfig,
    OpenCCUnavailableWarning,
    ScriptConversion,
    WidthNormalization,
    convert_script,
    detect_variant_normalized,
    detect_width_normalized,
    normalize_chinese_variants,
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


# --------------------------------------------------------------------------
# Simplified/Traditional conversion and offset alignment
# --------------------------------------------------------------------------


def test_opencc_round_trip_is_exact_with_identity_alignment():
    original = "患者头痛，服用药物。"

    traditional = convert_script(original, OpenCCConfig.S2T)
    simplified = convert_script(traditional.text, OpenCCConfig.T2S)

    assert isinstance(traditional, ScriptConversion)
    assert traditional.text == "患者頭痛，服用藥物。"
    assert simplified.text == original
    identity = tuple((index, index + 1) for index in range(len(original)))
    assert traditional.char_origins == identity
    assert simplified.char_origins == identity


def test_phrase_conversion_maps_unchanged_phi_name_to_exact_source_span():
    original = "患者王明使用互联网挂号。"
    converted = convert_script(original, OpenCCConfig.S2TWP)

    assert "網際網路" in converted.text
    assert len(converted.text) != len(original)
    start = converted.text.index("王明")
    original_start, original_end = converted.to_original_span(start, start + 2)

    assert original[original_start:original_end] == "王明"


def test_length_changing_phrase_alignment_covers_the_complete_source_phrase():
    original = "互联网鼠标"
    converted = convert_script(original, OpenCCConfig.S2TWP)

    assert converted.text == "網際網路滑鼠"
    assert converted.to_original_span(0, len(converted.text)) == (
        0,
        len(original),
    )
    assert converted.offset_map == converted.alignment == converted.char_origins


def test_single_character_phi_anchor_next_to_phrase_maps_exactly():
    original = "鼠标王互联网"
    converted = convert_script(original, OpenCCConfig.S2TWP)

    assert converted.text == "滑鼠王網際網路"
    start = converted.text.index("王")
    original_start, original_end = converted.to_original_span(start, start + 1)

    assert (original_start, original_end) == (2, 3)
    assert original[original_start:original_end] == "王"


def test_context_dependent_phrase_alignment_fails_closed():
    converted = convert_script("鼠标", OpenCCConfig.S2TWP)

    assert converted.text == "滑鼠"
    assert converted.char_origins == ((0, 2), (0, 2))
    assert converted.to_original_span(0, 1) == (0, 2)
    assert converted.to_original_span(1, 2) == (0, 2)


def test_repeated_phrase_alignment_preserves_each_phi_anchor():
    original = "鼠标王互联网，" * 50
    converted = convert_script(original, OpenCCConfig.S2TWP)

    for index, character in enumerate(converted.text):
        if character != "王":
            continue
        original_start, original_end = converted.to_original_span(index, index + 1)
        assert original[original_start:original_end] == "王"


@pytest.mark.parametrize("config", list(OpenCCConfig))
def test_all_supported_opencc_configs_load(config):
    result = convert_script("患者头痛", config)

    assert result.opencc_available is True
    assert len(result.char_origins) == len(result.text)


def test_conversion_result_supports_text_alignment_unpacking():
    converted_text, alignment = convert_script("头痛", "s2t.json")

    assert converted_text == "頭痛"
    assert alignment == ((0, 1), (1, 2))


def test_mixed_variant_detection_and_target_normalization():
    mixed = "患者头痛并服用藥物，病歷记录检查與休息。"
    estimate = detect_chinese_script(mixed)

    assert estimate.variant is ChineseScriptVariant.MIXED
    assert estimate.simplified_ratio == pytest.approx(0.5)
    assert estimate.traditional_ratio == pytest.approx(0.5)

    normalized = normalize_chinese_variants(
        mixed,
        ChineseTargetScript.SIMPLIFIED,
    )
    assert not (set(normalized.text) & TRADITIONAL_VARIANT_CHARS)
    assert "药物" in normalized.text
    assert "病历" in normalized.text
    assert "与休息" in normalized.text


def test_detection_helper_reports_predominant_variant():
    estimate = detect_chinese_script("头痛用药记录完整，病歷已核对。")

    assert estimate.variant is ChineseScriptVariant.SIMPLIFIED
    assert estimate.mixed is True
    assert estimate.simplified_ratio > estimate.traditional_ratio


def test_missing_opencc_warns_once_and_returns_identity(monkeypatch):
    import openmed.processing.zh_normalize as zh_normalize

    def missing_converter(_config):
        raise ModuleNotFoundError("No module named 'opencc'")

    monkeypatch.setattr(zh_normalize, "_opencc_converter", missing_converter)
    monkeypatch.setattr(zh_normalize, "_OPENCC_NOTICE_EMITTED", False)

    with pytest.warns(OpenCCUnavailableWarning, match="openmed\\[zh\\]"):
        first = convert_script("患者头痛", OpenCCConfig.S2T)
    with warnings.catch_warnings(record=True) as repeated:
        warnings.simplefilter("always")
        second = convert_script("患者头痛", OpenCCConfig.S2T)

    assert not repeated
    assert first.text == second.text == "患者头痛"
    assert (
        first.char_origins
        == second.char_origins
        == tuple((index, index + 1) for index in range(4))
    )
    assert first.opencc_available is second.opencc_available is False


def test_region_configs_produce_taiwan_and_hong_kong_variants():
    assert convert_script("里面", OpenCCConfig.S2TW).text == "裡面"
    assert convert_script("里面", OpenCCConfig.S2HK).text == "裏面"


def test_variant_normalized_matcher_restores_original_offsets():
    original = "患者王明使用互联网挂号。"

    def matcher(normalized: str):
        start = normalized.index("王明")
        return [(start, start + 2, "PERSON")]

    results = detect_variant_normalized(
        original,
        matcher,
        target=ChineseTargetScript.TRADITIONAL,
    )

    start, end, label = results[0]
    assert label == "PERSON"
    assert original[start:end] == "王明"


def test_detection_normalization_composes_phrase_alignment():
    original = "患者王明使用互联网挂号。"
    normalized = normalize_for_pii_detection(
        original,
        chinese_target_script="traditional",
    )

    start = normalized.text.index("王明")
    original_start, original_end = normalized.remap_span(start, start + 2)
    assert original[original_start:original_end] == "王明"
    assert normalized.chinese_variant_normalized is True
    assert normalized.opencc_available is True


def test_extract_pii_maps_chinese_normalized_entity_to_original(monkeypatch):
    original = "患者王明服用藥物。"
    observed_inputs = []

    def fake_analyze_text(text, **_kwargs):
        observed_inputs.append(text)
        start = text.index("王明")
        return PredictionResult(
            text=text,
            entities=[
                EntityPrediction(
                    text="王明",
                    label="NAME",
                    start=start,
                    end=start + 2,
                    confidence=0.99,
                )
            ],
            model_name="fixture-pii-model",
            timestamp="2026-01-01T00:00:00",
        )

    monkeypatch.setattr(openmed, "analyze_text", fake_analyze_text)
    result = openmed.extract_pii(
        original,
        model_name="fixture-pii-model",
        config=OpenMedConfig(chinese_target_script="simplified"),
        use_smart_merging=False,
    )

    assert observed_inputs == ["患者王明服用药物。"]
    assert [(entity.text, entity.start, entity.end) for entity in result.entities] == [
        ("王明", 2, 4)
    ]


def test_pipeline_redacts_original_name_after_variant_conversion():
    from openmed.core.pipeline import Pipeline

    original = "患者王明使用互联网挂号。"

    def model_detector(text, **kwargs):
        start = text.index("王明")
        return PredictionResult(
            text=text,
            entities=[
                EntityPrediction(
                    text="王明",
                    label="NAME",
                    start=start,
                    end=start + 2,
                    confidence=0.99,
                )
            ],
            model_name=kwargs["model_name"],
            timestamp="2026-01-01T00:00:00",
        )

    result = Pipeline(
        config=OpenMedConfig(chinese_target_script="traditional"),
        model_detector=model_detector,
        use_safety_sweep=False,
    ).run(original, method="mask")

    entity = result.deidentification_result.pii_entities[0]
    assert result.normalized_text == "患者王明使用互聯網掛號。"
    assert original[entity.start : entity.end] == entity.text == "王明"
    assert "王明" not in result.redacted_text


def test_config_exposes_optional_chinese_target_script():
    assert OpenMedConfig().chinese_target_script is None
    config = OpenMedConfig.from_dict({"chinese_target_script": "traditional"})
    assert config.chinese_target_script == "traditional"
    assert config.to_dict()["chinese_target_script"] == "traditional"
    with pytest.raises(ValueError, match="chinese_target_script"):
        OpenMedConfig(chinese_target_script="mixed")
