"""Acceptance gates for ISCII and caller-supplied legacy-font conversion."""

from __future__ import annotations

import json
import unicodedata

import pytest

from openmed.core.pipeline import Pipeline
from openmed.core.script_detect import normalize_for_pii_detection
from openmed.processing.legacy_encoding import (
    ISCII_MAPPING_PROVENANCE,
    LegacyFontMap,
    convert_legacy_encoding,
    detect_legacy_encoding,
    iscii_to_unicode,
    unicode_to_iscii,
)
from openmed.processing.text import normalize_indic_text

# Synthetic non-Vedic Hindi fixture: "रमेश शर्मा" in ISCII-1991.
_ISCII_NAME = bytes.fromhex("cf cc e1 d5 20 d5 cf e8 cc da")


def _synthetic_font_map() -> LegacyFontMap:
    # Visual-order "f" is the pre-base I glyph; S is a virama glyph.
    return LegacyFontMap(
        name="synthetic-devanagari",
        mapping={
            ord("f"): "ि",
            ord("k"): "क",
            ord("S"): "्",
            ord("x"): "ष",
        },
        provenance="synthetic-test-fixture",
    )


def test_non_vedic_iscii_round_trips_byte_identically():
    converted = iscii_to_unicode(_ISCII_NAME)

    assert converted.text == "रमेश शर्मा"
    assert unicodedata.normalize("NFC", converted.text) == converted.text
    assert unicode_to_iscii(converted.text) == _ISCII_NAME


def test_nukta_and_virama_sequences_are_logically_ordered_and_lossless():
    # QA + virama: KA + NUKTA + HALANT in ISCII.
    source = bytes.fromhex("b3 e9 e8")
    converted = iscii_to_unicode(source)

    assert converted.text == "क़्"
    assert [unicodedata.combining(char) for char in converted.text[1:]] == [7, 9]
    assert unicode_to_iscii(converted.text) == source


@pytest.mark.parametrize(
    ("source", "join_control"),
    [
        (bytes.fromhex("b3 e8 e8"), "\u200c"),
        (bytes.fromhex("b3 e8 e9"), "\u200d"),
    ],
)
def test_contextual_virama_controls_round_trip(source, join_control):
    converted = iscii_to_unicode(source)

    assert converted.text == f"क्{join_control}"
    assert unicode_to_iscii(converted.text) == source


def test_visual_order_legacy_font_run_becomes_idempotent_unicode():
    source = b"fkSx"
    font_map = _synthetic_font_map()

    assert detect_legacy_encoding(source, legacy_font_map=font_map) == "legacy-font"
    converted = convert_legacy_encoding(source, legacy_font_map=font_map)

    assert converted.text == "क्षि"
    assert normalize_indic_text(converted.text).text == converted.text
    assert unicodedata.normalize("NFC", converted.text) == converted.text


def test_conversion_offset_map_returns_original_phi_byte_span():
    font_map = LegacyFontMap(
        name="synthetic-name-glyphs",
        mapping={
            ord("A"): "राम",
            ord("B"): " ",
            ord("C"): "शर्मा",
        },
    )
    source = b"ID=ABC;"
    converted = convert_legacy_encoding(
        source,
        encoding="legacy-font",
        legacy_font_map=font_map,
    )
    start = converted.text.index("राम शर्मा")
    end = start + len("राम शर्मा")

    original_start, original_end = converted.to_original_span(start, end)

    assert source[original_start:original_end] == b"ABC"
    assert converted.offset_map.to_converted_span(3, 6) == (3, end)


def test_pure_latin_clinical_note_is_not_auto_converted():
    note = "Patient John Smith visited clinic for BP follow-up."
    font_map = _synthetic_font_map()

    assert detect_legacy_encoding(note, legacy_font_map=font_map) == "unicode"
    converted = convert_legacy_encoding(note, legacy_font_map=font_map)

    assert converted.text == note
    assert converted.encoding == "unicode"
    assert not converted.changed


def test_pure_latin_note_is_unchanged_with_dense_user_map():
    note = "Patient John Smith visited clinic for BP follow-up."
    dense_map = LegacyFontMap(
        name="dense-synthetic-map",
        mapping={
            ord(char): "क"
            for char in "ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz"
        },
    )

    assert detect_legacy_encoding(note, legacy_font_map=dense_map) == "unicode"
    assert convert_legacy_encoding(note, legacy_font_map=dense_map).text == note


def test_valid_utf8_devanagari_bytes_are_not_misdetected_as_iscii():
    source = "रमेश शर्मा".encode()

    assert detect_legacy_encoding(source) == "unicode"
    converted = convert_legacy_encoding(source)

    assert converted.text == "रमेश शर्मा"
    assert converted.encoding == "unicode"


def test_c1_control_bytes_prevent_false_iscii_detection():
    text = "\x80\xa0\xb3"

    assert detect_legacy_encoding(text) == "unicode"
    assert convert_legacy_encoding(text).text == text


def test_detection_normalizer_routes_iscii_and_maps_span_to_source():
    # Latin-1 is the compatibility representation used when an application has
    # already placed the raw legacy bytes in a Python string.
    original = _ISCII_NAME.decode("latin-1")
    normalized = normalize_for_pii_detection(original)

    assert normalized.text == "रमेश शर्मा"
    assert normalized.legacy_encoding == "iscii"
    assert normalized.converted_legacy_bytes == len(_ISCII_NAME) - 1
    assert normalized.remap_span(0, len(normalized.text)) == (0, len(original))
    assert normalized.to_metadata()["legacy_encoding"] == "iscii"


def test_pipeline_converts_iscii_before_script_routing():
    original = _ISCII_NAME.decode("latin-1")
    pipeline = Pipeline(lang="auto")

    document = pipeline.stage1_normalize(original)
    route = pipeline.stage2_language_script(document.normalized_text)

    assert document.normalized_text == "रमेश शर्मा"
    assert document.offset_map.normalized_span_to_original_offsets(
        0, len(document.normalized_text)
    ) == (0, len(original))
    assert document.metadata["legacy_encoding"] == {
        "encoding": "iscii",
        "changed": True,
        "converted_bytes": len(_ISCII_NAME) - 1,
    }
    assert route.script == "Devanagari"
    assert route.lang == "hi"


def test_detection_normalizer_preserves_attached_indic_marks():
    normalized = normalize_for_pii_detection("क़्")

    assert normalized.text == "क़्"
    assert normalized.stripped_combining_marks == 0
    assert not normalized.changed


def test_detection_normalizer_strips_standalone_indic_mark():
    normalized = normalize_for_pii_detection("़Patient")

    assert normalized.text == "Patient"
    assert normalized.stripped_combining_marks == 1


def test_user_mapping_file_loads_without_bundled_font_data(tmp_path):
    path = tmp_path / "custom-font.json"
    path.write_text(
        json.dumps(
            {
                "name": "hospital-archive-font",
                "provenance": "licensed by data owner",
                "mapping": {"0x41": "र", "B": "ा", "67": "म"},
            }
        ),
        encoding="utf-8",
    )

    font_map = LegacyFontMap.from_file(path)

    assert font_map.name == "hospital-archive-font"
    assert dict(font_map.mapping) == {0x41: "र", ord("B"): "ा", 67: "म"}
    assert font_map.provenance == "licensed by data owner"


def test_invalid_iscii_and_invalid_mapping_are_rejected(tmp_path):
    with pytest.raises(UnicodeDecodeError):
        iscii_to_unicode(b"\xff")

    path = tmp_path / "bad.txt"
    path.write_text("{}", encoding="utf-8")
    with pytest.raises(ValueError, match="JSON, YAML, or YML"):
        LegacyFontMap.from_file(path)


def test_mapping_provenance_documents_public_standard_and_license():
    assert "IS 13194:1991" in ISCII_MAPPING_PROVENANCE
    assert "Apache-2.0" in ISCII_MAPPING_PROVENANCE
